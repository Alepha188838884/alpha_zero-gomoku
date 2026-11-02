"""两个固定策略的对手，用于自对弈训练中的评估基线。"""
import random
from typing import List

import numpy as np


def random_policy(env) -> int:
    """在合法动作中均匀随机选择。"""
    legal = np.flatnonzero(env.legal_actions == 1)
    # random.choice 在 Python 3.11.1 下不接受 numpy 数组，改用下标随机
    return int(legal[random.randrange(len(legal))])


def _run_length(board: np.ndarray, r: int, c: int, color: int, dr: int, dc: int, board_size: int) -> int:
    """从 (r, c) 沿 (dr, dc) 方向（不含起点）统计连续同色子数。"""
    count = 0
    rr, cc = r + dr, c + dc
    while 0 <= rr < board_size and 0 <= cc < board_size and board[rr, cc] == color:
        count += 1
        rr += dr
        cc += dc
    return count


def _max_line_through_point(board: np.ndarray, r: int, c: int, color: int, board_size: int) -> int:
    """假设在 (r, c) 处落下 color 方的子，返回 4 个方向中穿过该点的最长连续同色子数（含该点）。"""
    dirs = ((0, 1), (1, 0), (1, 1), (1, -1))
    best = 0
    for dr, dc in dirs:
        length = 1 + _run_length(board, r, c, color, dr, dc, board_size) + _run_length(board, r, c, color, -dr, -dc, board_size)
        best = max(best, length)
    return best


def _has_neighbor(board: np.ndarray, r: int, c: int, board_size: int) -> bool:
    """判断 (r, c) 的 8 邻域内是否已有棋子（切比雪夫距离 <= 1）。"""
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < board_size and 0 <= cc < board_size and board[rr, cc] != 0:
                return True
    return False


def _line_score(length: int) -> float:
    """把连续子数映射为分数。"""
    table = {4: 1e4, 3: 1e2, 2: 10, 1: 1}
    if length >= 5:
        return 1e6
    return table.get(length, 0.0)


def heuristic_policy(env) -> int:
    """规则型对手：对每个合法动作，综合己方成线潜力与对方成线威胁打分，取最高分（并列随机）。"""
    board_size = env.board_size
    board = env.board
    me = env.to_play
    opp = env.white_player if me == env.black_player else env.black_player

    best_score = None
    best_actions: List[int] = []
    for action in np.flatnonzero(env.legal_actions == 1):
        r, c = int(action) // board_size, int(action) % board_size
        l_me = _max_line_through_point(board, r, c, me, board_size)
        l_op = _max_line_through_point(board, r, c, opp, board_size)
        score = _line_score(l_me) + 0.9 * _line_score(l_op)
        if _has_neighbor(board, r, c, board_size):
            score += 0.5

        if best_score is None or score > best_score:
            best_score = score
            best_actions = [int(action)]
        elif score == best_score:
            best_actions.append(int(action))

    return random.choice(best_actions)


OPPONENTS = {'random': random_policy, 'heuristic': heuristic_policy}
