"""DQN 五子棋自对弈训练主脚本。"""
import argparse
import csv
import os
import random
import time
from typing import Optional

import numpy as np
import torch

from alpha_zero.envs.gomoku import GomokuEnv

from .agent import DQNAgent, ReplayBuffer
from .evaluate import evaluate
from .opponents import OPPONENTS
from .selfplay import play_episode

CSV_FIELDS = [
    'step',
    'episodes',
    'mix_episodes',
    'epsilon',
    'avg_loss',
    'avg_ep_len',
    'selfplay_black_win_rate',
    'vs_random',
    'vs_heuristic',
    'vs_prev',
    'buffer_size',
    'elapsed_sec',
]


def resolve_device(name: str) -> str:
    """auto: mps 可用则用 mps，否则 cpu。"""
    if name != 'auto':
        return name
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def linear_epsilon(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    """按环境步数从 eps_start 线性衰减到 eps_end。"""
    if decay_steps <= 0:
        return eps_end
    frac = min(1.0, step / decay_steps)
    return eps_start + frac * (eps_end - eps_start)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='DQN 五子棋自对弈训练')
    parser.add_argument('--board_size', type=int, default=9)
    parser.add_argument('--num_stack', type=int, default=1)
    parser.add_argument('--total_steps', type=int, default=300000)
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--eps_start', type=float, default=1.0)
    parser.add_argument('--eps_end', type=float, default=0.1)
    parser.add_argument('--eps_decay_steps', type=int, default=100000)
    parser.add_argument('--learn_start', type=int, default=5000)
    parser.add_argument('--train_every', type=int, default=1)
    parser.add_argument('--target_sync', type=int, default=2000)
    parser.add_argument('--eval_every', type=int, default=10000)
    parser.add_argument('--eval_games', type=int, default=40)
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/dqn_gomoku/9x9')
    parser.add_argument('--log_dir', type=str, default='logs/dqn_gomoku/9x9')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--head', type=str, choices=['conv', 'linear'], default='conv')
    parser.add_argument('--opponent_mix', type=float, default=0.0, help='每局以该概率对启发式对手而非纯自对弈')
    parser.add_argument('--init_ckpt', type=str, default='', help='非空时从该 checkpoint 热启动')
    parser.add_argument('--eval_random_opening', type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert args.learn_start >= args.batch_size, 'learn_start 必须 >= batch_size，否则采样时 buffer 里样本不够'
    set_seed(args.seed)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    device = resolve_device(args.device)
    in_planes = 2 * args.num_stack + 1
    augment = not args.no_augment

    env = GomokuEnv(board_size=args.board_size, num_stack=args.num_stack)
    if args.init_ckpt:
        agent = DQNAgent.from_checkpoint(args.init_ckpt, device=device, lr=args.lr, gamma=args.gamma)
        print(f'从 checkpoint 热启动: {args.init_ckpt} (head={agent.head})')
    else:
        agent = DQNAgent(
            board_size=args.board_size,
            in_planes=in_planes,
            device=device,
            lr=args.lr,
            gamma=args.gamma,
            channels=args.channels,
            num_layers=args.num_layers,
            head=args.head,
        )
    buffer = ReplayBuffer(args.buffer_size)

    csv_path = os.path.join(args.log_dir, 'train.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    csv_writer.writeheader()

    step = 0
    episodes = 0
    start_time = time.time()
    next_eval_at = args.eval_every
    prev_snapshot: Optional[DQNAgent] = None

    # 自上次评估以来的统计量
    since_eval_losses = []
    since_eval_ep_lens = []
    since_eval_black_wins = 0
    since_eval_selfplay_episodes = 0  # 纯自对弈局数（不含对启发式对手的混合局）
    since_eval_mix_episodes = 0
    epsilon = args.eps_start

    try:
        while step < args.total_steps:
            epsilon = linear_epsilon(step, args.eps_start, args.eps_end, args.eps_decay_steps)

            if random.random() < args.opponent_mix:
                opponent = OPPONENTS['heuristic']
                opponent_color = random.choice((1, 2))
            else:
                opponent = None
                opponent_color = None

            transitions, winner, ep_steps = play_episode(env, agent, epsilon, opponent=opponent, opponent_color=opponent_color)
            episodes += 1
            since_eval_ep_lens.append(ep_steps)
            if opponent is not None:
                since_eval_mix_episodes += 1
            else:
                since_eval_selfplay_episodes += 1
                if winner == env.black_player:
                    since_eval_black_wins += 1

            for t in transitions:
                buffer.push(t)
                step += 1

                if len(buffer) >= args.learn_start and step % args.train_every == 0:
                    batch = buffer.sample(args.batch_size)
                    loss = agent.update(batch, augment=augment)
                    since_eval_losses.append(loss)

                if step % args.target_sync == 0:
                    agent.sync_target()

                if step >= args.total_steps:
                    break

            # 评估检查点放在一局结束之后，避免打断正在推进的一局
            if step >= next_eval_at:
                elapsed = time.time() - start_time
                avg_loss = float(np.mean(since_eval_losses)) if since_eval_losses else float('nan')
                avg_ep_len = float(np.mean(since_eval_ep_lens)) if since_eval_ep_lens else float('nan')
                selfplay_black_win_rate = (
                    since_eval_black_wins / since_eval_selfplay_episodes if since_eval_selfplay_episodes else float('nan')
                )

                greedy = agent.greedy_policy()
                vs_random = evaluate(
                    greedy, OPPONENTS['random'], env, args.eval_games, random_opening=args.eval_random_opening
                )['win_rate']
                vs_heuristic = evaluate(
                    greedy, OPPONENTS['heuristic'], env, args.eval_games, random_opening=args.eval_random_opening
                )['win_rate']
                if prev_snapshot is not None:
                    vs_prev = evaluate(
                        greedy, prev_snapshot.greedy_policy(), env, args.eval_games, random_opening=args.eval_random_opening
                    )['win_rate']
                else:
                    vs_prev = float('nan')

                ckpt_path = os.path.join(args.ckpt_dir, f'step_{step}.pt')
                agent.save(ckpt_path, extra={'step': step})
                prev_snapshot = DQNAgent.from_checkpoint(ckpt_path, device='cpu')

                row = {
                    'step': step,
                    'episodes': episodes,
                    'mix_episodes': since_eval_mix_episodes,
                    'epsilon': epsilon,
                    'avg_loss': avg_loss,
                    'avg_ep_len': avg_ep_len,
                    'selfplay_black_win_rate': selfplay_black_win_rate,
                    'vs_random': vs_random,
                    'vs_heuristic': vs_heuristic,
                    'vs_prev': vs_prev,
                    'buffer_size': len(buffer),
                    'elapsed_sec': elapsed,
                }
                csv_writer.writerow(row)
                csv_file.flush()
                print(
                    f'step={step} episodes={episodes} mix_episodes={since_eval_mix_episodes} eps={epsilon:.3f} loss={avg_loss:.4f} '
                    f'ep_len={avg_ep_len:.1f} black_wr={selfplay_black_win_rate:.3f} '
                    f'vs_random={vs_random:.3f} vs_heuristic={vs_heuristic:.3f} vs_prev={vs_prev:.3f} '
                    f'buffer={len(buffer)} elapsed={elapsed:.1f}s'
                )

                since_eval_losses = []
                since_eval_ep_lens = []
                since_eval_black_wins = 0
                since_eval_selfplay_episodes = 0
                since_eval_mix_episodes = 0
                # 若一局跨过多个评估边界，直接跳到下一个未越过的边界，避免用同一个 step 重复评估
                next_eval_at = (step // args.eval_every + 1) * args.eval_every

        agent.save(os.path.join(args.ckpt_dir, 'final.pt'), extra={'step': step})
        print(f'训练完成，共 {step} 步，{episodes} 局，用时 {time.time() - start_time:.1f}s')
    except KeyboardInterrupt:
        agent.save(os.path.join(args.ckpt_dir, 'interrupted.pt'), extra={'step': step})
        print(f'训练被中断，已保存 interrupted.pt (step={step})')
    except Exception:
        agent.save(os.path.join(args.ckpt_dir, 'interrupted.pt'), extra={'step': step})
        print(f'训练异常终止，已保存 interrupted.pt (step={step})')
        raise
    finally:
        csv_file.close()


if __name__ == '__main__':
    main()
