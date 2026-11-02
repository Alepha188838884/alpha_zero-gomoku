"""DQN 智能体：经验回放、Double DQN + negamax 目标、对称数据增强。"""
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .net import QNet, masked_argmax
from .symmetry import transform_action, transform_mask, transform_planes


@dataclass
class Transition:
    """一步转移记录，state/next_state 均为落子方视角的观测。"""

    state: np.ndarray  # (C, B, B) int8
    action: int
    reward: float
    next_state: np.ndarray  # (C, B, B) int8，对手视角
    next_legal: np.ndarray  # (B*B,) int8
    done: bool


class ReplayBuffer:
    """定长环形经验回放池。"""

    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """持有 online/target 两个 QNet 的 DQN 智能体。"""

    def __init__(
        self,
        board_size: int,
        in_planes: int,
        device,
        lr: float = 5e-4,
        gamma: float = 0.99,
        channels: int = 64,
        num_layers: int = 4,
        head: str = 'conv',
    ) -> None:
        self.board_size = board_size
        self.in_planes = in_planes
        self.device = torch.device(device)
        self.gamma = gamma
        self.channels = channels
        self.num_layers = num_layers
        self.head = head

        self.online = QNet(in_planes, board_size, channels, num_layers, head=head).to(self.device)
        self.target = QNet(in_planes, board_size, channels, num_layers, head=head).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)

    def select_action(self, obs: np.ndarray, legal: np.ndarray, epsilon: float) -> int:
        """ε-greedy 选动作：ε 概率合法动作中随机，否则取 online 网络的 masked_argmax。"""
        if random.random() < epsilon:
            legal_actions = np.flatnonzero(legal == 1)
            # random.choice 在 Python 3.11.1 下不接受 numpy 数组，改用下标随机
            return int(legal_actions[random.randrange(len(legal_actions))])

        with torch.no_grad():
            obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device).unsqueeze(0)
            legal_t = torch.as_tensor(legal, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(obs_t)
            action = masked_argmax(q, legal_t)
        return int(action.item())

    def compute_targets(
        self,
        rewards: np.ndarray,
        next_states: np.ndarray,
        next_legal: np.ndarray,
        dones: np.ndarray,
    ) -> torch.Tensor:
        """计算 TD 目标（Double DQN + negamax）。

        next_state 是对手落子后的视角：对手会在 next_state 上最大化自己的 Q 值，
        而对手的最优收益恰好是我方的最差收益，所以要在其 Q 值前加负号，
        并乘以折扣因子 gamma，得到我方在当前状态下的期望目标。
        动作选择用 online 网络（Double DQN 里的 "select"），
        动作评估用 target 网络（Double DQN 里的 "evaluate"），以降低过估计。
        对于终局样本（done=True），目标直接是环境返回的即时奖励。
        """
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        next_legal_t = torch.as_tensor(next_legal, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            q_online_next = self.online(next_states_t)
            best_next_action = masked_argmax(q_online_next, next_legal_t)
            q_target_next = self.target(next_states_t)
            q_next = q_target_next.gather(1, best_next_action.unsqueeze(1)).squeeze(1)

        targets = -self.gamma * q_next
        targets = torch.where(dones_t, rewards_t, targets)
        return targets

    def update(self, batch: List[Transition], augment: bool = True) -> float:
        """用一个 batch 的转移做一次梯度更新，返回 Huber loss 数值。"""
        states = np.stack([t.state for t in batch]).astype(np.int8)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch]).astype(np.int8)
        next_legal = np.stack([t.next_legal for t in batch]).astype(np.int8)
        dones = np.array([t.done for t in batch], dtype=np.bool_)

        if augment:
            # 每个样本独立随机选一种对称变换，同时作用于 state/action/next_state/next_legal。
            for i in range(len(batch)):
                sym = random.randint(0, 7)
                states[i] = transform_planes(states[i], sym)
                actions[i] = transform_action(int(actions[i]), self.board_size, sym)
                next_states[i] = transform_planes(next_states[i], sym)
                next_legal[i] = transform_mask(next_legal[i], self.board_size, sym)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)

        q_values = self.online(states_t)
        q_sa = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        targets = self.compute_targets(rewards, next_states, next_legal, dones)

        loss = F.smooth_l1_loss(q_sa, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path: str, extra: Optional[dict] = None) -> None:
        extra = extra or {}
        payload = {
            'online': self.online.state_dict(),
            'board_size': self.board_size,
            'in_planes': self.in_planes,
            'channels': self.channels,
            'num_layers': self.num_layers,
            'head': self.head,
        }
        payload.update(extra)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def from_checkpoint(cls, path: str, device, lr: float = 5e-4, gamma: float = 0.99) -> 'DQNAgent':
        payload = torch.load(path, map_location=device)
        agent = cls(
            board_size=payload['board_size'],
            in_planes=payload['in_planes'],
            device=device,
            lr=lr,
            gamma=gamma,
            channels=payload.get('channels', 64),
            num_layers=payload.get('num_layers', 4),
            head=payload.get('head', 'linear'),
        )
        agent.online.load_state_dict(payload['online'])
        agent.sync_target()
        return agent

    def greedy_policy(self):
        """返回一个 act(env) -> int 的闭包（ε=0），供评估和对弈使用。"""

        def act(env) -> int:
            obs = env.observation()
            return self.select_action(obs, env.legal_actions, epsilon=0.0)

        return act
