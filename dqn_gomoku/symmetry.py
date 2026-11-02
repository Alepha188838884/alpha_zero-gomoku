"""棋盘的 8 种对称变换（4 个旋转 x 是否水平翻转），用于数据增强。"""
import numpy as np


def transform_planes(planes: np.ndarray, sym: int) -> np.ndarray:
    """对特征平面做对称变换。

    Args:
        planes: 形状 (C, B, B) 的数组。
        sym: 0..7 的对称编号，sym % 4 决定旋转次数，sym >= 4 再做一次水平翻转（沿最后一维）。

    Returns:
        变换后的连续数组，形状不变。
    """
    out = np.rot90(planes, k=sym % 4, axes=(-2, -1))
    if sym >= 4:
        out = np.flip(out, axis=-1)
    return np.ascontiguousarray(out)


def transform_action(action: int, board_size: int, sym: int) -> int:
    """把动作视为 board_size*board_size 的 one-hot 平面，用同一套变换后取 argmax。"""
    one_hot = np.zeros((1, board_size, board_size), dtype=np.int8)
    row, col = action // board_size, action % board_size
    one_hot[0, row, col] = 1
    transformed = transform_planes(one_hot, sym)
    return int(np.argmax(transformed.reshape(-1)))


def transform_mask(mask: np.ndarray, board_size: int, sym: int) -> np.ndarray:
    """对长度 B*B 的合法动作掩码做和 transform_planes 一致的变换。"""
    reshaped = mask.reshape(1, board_size, board_size)
    transformed = transform_planes(reshaped, sym)
    return transformed.reshape(-1)
