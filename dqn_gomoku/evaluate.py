"""评估工具：让两个策略对弈，统计胜率。"""
import argparse
import random
from typing import Callable, Dict

import numpy as np

from alpha_zero.envs.gomoku import GomokuEnv

from .agent import DQNAgent
from .opponents import OPPONENTS

Policy = Callable[..., int]


def play_match(env, policy_black: Policy, policy_white: Policy, random_opening: int = 0) -> int:
    """下一局，返回赢家（1 黑 / 2 白 / 0 平局）。

    前 random_opening 手（双方合计）在合法动作中均匀随机落子，之后再交给策略。
    """
    env.reset()
    moves = 0
    while not env.is_game_over():
        if moves < random_opening:
            legal = np.flatnonzero(env.legal_actions == 1)
            # random.choice 在 Python 3.11.1 下不接受 numpy 数组，改用下标随机
            action = int(legal[random.randrange(len(legal))])
        else:
            policy = policy_black if env.to_play == env.black_player else policy_white
            action = policy(env)
        env.step(action)
        moves += 1
    return env.winner if env.winner is not None else 0


def evaluate(agent_policy: Policy, opponent_policy: Policy, env, num_games: int, random_opening: int = 0) -> Dict[str, float]:
    """agent 一半执黑一半执白，与 opponent 对弈 num_games 局，返回统计结果。"""
    black_games = num_games // 2
    white_games = num_games - black_games

    wins = 0
    black_wins = 0
    white_wins = 0
    draws = 0
    total_len = 0

    for _ in range(black_games):
        winner = play_match(env, agent_policy, opponent_policy, random_opening=random_opening)
        total_len += env.steps
        if winner == env.black_player:
            wins += 1
            black_wins += 1
        elif winner == 0:
            draws += 1

    for _ in range(white_games):
        winner = play_match(env, opponent_policy, agent_policy, random_opening=random_opening)
        total_len += env.steps
        if winner == env.white_player:
            wins += 1
            white_wins += 1
        elif winner == 0:
            draws += 1

    return {
        'win_rate': wins / num_games if num_games else 0.0,
        'black_win_rate': black_wins / black_games if black_games else 0.0,
        'white_win_rate': white_wins / white_games if white_games else 0.0,
        'draw_rate': draws / num_games if num_games else 0.0,
        'avg_len': total_len / num_games if num_games else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='评估 DQN 五子棋 checkpoint 对固定对手的胜率')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--opponent', type=str, default='heuristic', choices=list(OPPONENTS.keys()))
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--board_size', type=int, default=9)
    parser.add_argument('--random_opening', type=int, default=2)
    args = parser.parse_args()

    agent = DQNAgent.from_checkpoint(args.ckpt, device='cpu')
    num_stack = (agent.in_planes - 1) // 2
    env = GomokuEnv(board_size=agent.board_size, num_stack=num_stack)

    result = evaluate(agent.greedy_policy(), OPPONENTS[args.opponent], env, args.games, random_opening=args.random_opening)
    print(f'ckpt={args.ckpt} opponent={args.opponent} games={args.games}')
    for k, v in result.items():
        print(f'  {k}: {v:.4f}')


if __name__ == '__main__':
    main()
