"""终端人机对弈：人输入 GTP 坐标，AI 用 greedy_policy。"""
import argparse

from alpha_zero.envs.gomoku import GomokuEnv

from .agent import DQNAgent


def main() -> None:
    parser = argparse.ArgumentParser(description='终端人机对弈 DQN 五子棋')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--human_black', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    agent = DQNAgent.from_checkpoint(args.ckpt, device='cpu')
    num_stack = (agent.in_planes - 1) // 2
    env = GomokuEnv(board_size=agent.board_size, num_stack=num_stack)
    env.reset()

    ai_policy = agent.greedy_policy()
    human_player = env.black_player if args.human_black else env.white_player

    env.render('terminal')
    while not env.is_game_over():
        if env.to_play == human_player:
            action = None
            while action is None:
                gtp = input('你的落子（如 E5）: ').strip()
                action = env.gtp_to_action(gtp)
                if action is None:
                    print('非法坐标，请重新输入。')
        else:
            action = ai_policy(env)
        env.step(action)
        env.render('terminal')

    if env.winner is None:
        print('结果：平局')
    elif env.winner == human_player:
        print('结果：你赢了！')
    else:
        print('结果：AI 赢了！')


if __name__ == '__main__':
    main()
