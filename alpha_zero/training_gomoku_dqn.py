# File: alpha_zero/dqn_train.py
import os
import random
import numpy as np
import sys
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


# 正确导入同级目录下的模块
from alpha_zero.envs.gomoku import GomokuEnv
#from envs.gomoku import GomokuEnv
from alpha_zero.utils.csv_writer import CsvWriter

# 配置参数
BOARD_SIZE = 15
BUFFER_SIZE = 10000
BATCH_SIZE = 512
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.999
TARGET_UPDATE = 10
NUM_EPISODES = 1000

# 使用MPS加速（Mac M2 GPU）
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class DQN(nn.Module):
    """深度Q网络"""

    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * input_shape[0] * input_shape[1], 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN智能体"""

    def __init__(self, env):
        self.env = env
        self.input_shape = (BOARD_SIZE, BOARD_SIZE)
        self.num_actions = BOARD_SIZE * BOARD_SIZE

        self.policy_net = DQN(self.input_shape, self.num_actions).to(device)
        self.target_net = DQN(self.input_shape, self.num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.eps = EPS_START

    def select_action(self, state, legal_actions):
        """使用epsilon-greedy策略选择动作"""
        if random.random() > self.eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
                q_values = self.policy_net(state).cpu().numpy().flatten()
                legal_q = q_values[legal_actions]
                return legal_actions[np.argmax(legal_q)]
        else:
            return random.choice(legal_actions)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        batch = list(zip(*transitions))

        state_batch = torch.FloatTensor(np.array(batch[0])).unsqueeze(1).to(device)
        action_batch = torch.LongTensor(batch[1]).to(device)
        reward_batch = torch.FloatTensor(batch[2]).to(device)
        next_state_batch = torch.FloatTensor(np.array(batch[3])).unsqueeze(1).to(device)
        done_batch = torch.FloatTensor(batch[4]).to(device)

        q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + (1 - done_batch) * GAMMA * next_q_values

        loss = nn.MSELoss()(q_values.squeeze(), expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新epsilon
        self.eps = max(EPS_END, self.eps * EPS_DECAY)


def train():
    env = GomokuEnv(board_size=BOARD_SIZE)
    agent = DQNAgent(env)
    csv_writer = CsvWriter('training_data.csv')  # CSV文件保存在当前目录

    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            legal_actions = np.where(env.legal_actions == 1)[0].tolist()
            action = agent.select_action(state, legal_actions)

            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            agent.optimize_model()

        # 更新目标网络
        if episode % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        # 记录训练数据
        csv_writer.write({
            'Episode': episode,
            'Total Reward': total_reward,
            'Epsilon': agent.eps
        })

        print(f"Episode {episode}: Reward={total_reward:.2f}, Epsilon={agent.eps:.3f}")

    csv_writer.close()


if __name__ == "__main__":
    # 确保当前工作目录正确（如果从其他位置运行需要调整）
    torch.set_num_threads(1)  # 避免MPS多线程问题
    train()