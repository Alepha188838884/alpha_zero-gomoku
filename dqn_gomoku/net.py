"""DQN 五子棋使用的 Q 网络结构。"""
from typing import Tuple

import torch
import torch.nn as nn


class QNet(nn.Module):
    """简单的全卷积 Q 网络：若干层 3x3 卷积后接一个输出 B*B 个 Q 值的线性头。"""

    def __init__(self, in_planes: int, board_size: int, channels: int = 64, num_layers: int = 4, head: str = 'conv') -> None:
        super().__init__()
        self.in_planes = in_planes
        self.board_size = board_size
        self.head_type = head

        conv_layers = []
        for i in range(num_layers):
            c_in = in_planes if i == 0 else channels
            conv_layers.append(nn.Conv2d(c_in, channels, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv_layers)

        if head == 'linear':
            self.head = nn.Sequential(
                nn.Conv2d(channels, 2, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(2 * board_size * board_size, board_size * board_size),
            )
        elif head == 'conv':
            # 平移等变的全卷积头：不接全连接层，能学会"在任意位置堵四"这类局部模式。
            self.head = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, 1, kernel_size=1),
                nn.Flatten(),
            )
        else:
            raise ValueError(f'未知的 head 类型: {head}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 (N, in_planes, B, B) 的浮点张量，输出 (N, B*B) 的 Q 值。"""
        x = self.conv(x)
        return self.head(x)


def masked_argmax(q: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """在合法动作范围内取 Q 值最大的动作下标。

    Args:
        q: (N, A) 的 Q 值张量。
        legal: (N, A) 的 0/1 合法动作掩码，与 q 同 shape。

    Returns:
        (N,) 的 LongTensor，每行合法动作中 Q 值最大的下标。
    """
    masked_q = q.masked_fill(legal == 0, float('-inf'))
    return torch.argmax(masked_q, dim=-1)
