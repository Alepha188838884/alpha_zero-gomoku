import pygame
import numpy as np
import torch
from typing import Tuple

# 从你的训练代码中导入 DQN 网络定义及 device 变量
from training_gomoku_dqnV4 import DQN, device
# 从你的 MCTS 算法文件中导入 uct_search 函数
from alpha_zero.core.mcts_v2 import uct_search
# 从 gomoku.py 中导入环境定义
from alpha_zero.envs.gomoku import GomokuEnv

# ------------------ 参数设置 ------------------
BOARD_SIZE = 15  # 棋盘大小：15×15
WINDOW_SIZE = 600  # 窗口大小：600×600 像素
CELL_SIZE = WINDOW_SIZE // BOARD_SIZE  # 每个格子的尺寸

# 定义颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# ------------------ 初始化 Pygame ------------------
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Gomoku: DQN vs Human")

# ------------------ 实例化环境 ------------------
# 假设 GomokuEnv 在 reset() 时自动初始化棋盘，并提供 board, observation, step, legal_actions, is_game_over 等接口
game_env = GomokuEnv()
game_env.reset()  # 初始化环境

# ------------------ 模型加载 ------------------
# 模型训练时的输入尺寸为 15×15×17 = 3825，动作数为 225
state_dim = BOARD_SIZE * BOARD_SIZE * 17
action_dim = BOARD_SIZE * BOARD_SIZE
dqn_model = DQN(state_dim, action_dim).to(device)
model_path = "dqn_gomoku.pth"  # 确保路径正确
dqn_model.load_state_dict(torch.load(model_path, map_location=device))
dqn_model.eval()  # 设置为评估模式


# ------------------ 定义 DQN 评估函数 ------------------
def dqn_eval_func(observation: np.ndarray, is_batched: bool) -> Tuple[np.ndarray, float]:
    """
    输入：
        observation: 模型所需的观测（应展平为长度 3825 的向量），或批量输入
        is_batched: 是否批量数据
    输出：
        (动作先验概率, 状态价值)
    """
    if not is_batched:
        obs_tensor = torch.from_numpy(observation).unsqueeze(0).float().to(device)
    else:
        obs_tensor = torch.from_numpy(observation).float().to(device)

    with torch.no_grad():
        q_values = dqn_model(obs_tensor)

    # 用 softmax 将 Q 值转换为概率
    action_probs = torch.softmax(q_values, dim=-1).cpu().numpy()
    value = q_values.mean().item()

    if is_batched:
        return action_probs.tolist(), [value] * len(observation)
    else:
        return action_probs[0].tolist(), [value]


# ------------------ 使用 MCTS 选择动作 ------------------
def mcts_search(num_simulations: int = 100) -> Tuple[int, int]:
    """
    利用 MCTS（通过 uct_search）选择最佳落子位置。
    这里：
      - 从 game_env.observation() 获取当前观测（形状 (15,15,17)），展平后长度为3825，
        作为 root_node 信息传入 MCTS；
      - 第一个参数传入 game_env（它应实现 is_game_over()、legal_actions() 等接口）。
    返回：
        (row, col) —— 最佳落子位置
    """
    # 获取当前观测，并展平为一维向量
    obs = game_env.observation()  # 假设返回形状为 (15,15,17)
    state_flat = obs.flatten().astype(np.float32)
    root_node = state_flat.copy()
    c_puct_base = 1.0
    c_puct_init = 1.0
    root_noise = False
    eval_func = dqn_eval_func

    # 调用 uct_search，此处第一个参数传入 game_env，而非单纯的 numpy 数组
    action, _, _, _, _ = uct_search(game_env, eval_func, root_node, c_puct_base, c_puct_init, num_simulations,
                                    root_noise)

    # 假设 uct_search 返回的 action 为整数索引（0～224），转换为 (row, col)
    if isinstance(action, int):
        row = action // BOARD_SIZE
        col = action % BOARD_SIZE
        return row, col
    else:
        return action


# ------------------ 绘制棋盘和棋子 ------------------
def draw_board():
    # 假设 game_env.board 是一个 (15,15) 的 numpy 数组，存储棋盘状态
    board = game_env.board
    screen.fill(WHITE)
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            rect = (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)
            if board[row, col] == 1:
                pygame.draw.circle(screen, BLACK,
                                   (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2),
                                   CELL_SIZE // 3)
            elif board[row, col] == -1:
                pygame.draw.circle(screen, WHITE,
                                   (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2),
                                   CELL_SIZE // 3)
    pygame.display.update()


# ------------------ 主游戏循环 ------------------
def main():
    turn = 1  # 1 表示玩家（黑子），-1 表示 AI（白子）
    running = True
    clock = pygame.time.Clock()

    while running:
        draw_board()
        # 如果环境自身提供了游戏结束判断，也可直接调用 game_env.is_game_over()
        if game_env.is_game_over():
            # 根据你的环境实现，可能需要通过 game_env.get_winner() 获取赢家
            print("Game Over!")
            running = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # 玩家回合：鼠标点击选择落子
            if event.type == pygame.MOUSEBUTTONDOWN and turn == 1:
                pos = pygame.mouse.get_pos()
                row = pos[1] // CELL_SIZE
                col = pos[0] // CELL_SIZE
                action = row * BOARD_SIZE + col
                if action in game_env.legal_actions:
                    # 执行玩家落子，game_env.step(action) 会更新内部棋盘状态
                    game_env.step(action)
                    turn = -1

        # AI 回合：调用 MCTS 搜索决策
        if turn == -1:
            print("AI is thinking...")
            row, col = mcts_search(num_simulations=100)
            action = row * BOARD_SIZE + col
            if action in game_env.legal_actions:
                game_env.step(action)
                turn = 1

        clock.tick(10)  # 每秒 10 帧

    pygame.quit()


if __name__ == "__main__":
    main()
