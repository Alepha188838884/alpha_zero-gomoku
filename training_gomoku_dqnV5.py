#!/usr/bin/env python
import os
import random
import threading
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 这里假设你的环境和其他依赖能够从下面路径导入
from alpha_zero.envs.gomoku import GomokuEnv


# ============================
# 工具函数
# ============================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================
# 超参数设置
# ============================
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995  # 每个 episode 的指数衰减因子
TARGET_UPDATE = 100  # 每 TARGET_UPDATE 个 episode 同步一次目标网络
MEMORY_SIZE = 10000
LEARNING_RATE = 1e-3
NUM_THREADS = 4  # 使用的训练线程数
NUM_EPISODES = 5000  # 总训练 episode 数（将在各线程间均摊）

# 选择设备（Mac M2 上优先使用 MPS，否则使用 CPU）
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)


# ============================
# DQN 网络定义
# ============================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ============================
# 经验回放（线程安全）
# ============================
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.lock = threading.Lock()

    def push(self, state, action, reward, next_state, done):
        with self.lock:
            self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        with self.lock:
            return random.sample(self.memory, batch_size)

    def __len__(self):
        with self.lock:
            return len(self.memory)


# ============================
# 全局对象初始化
# ============================
# 为获得状态维度和动作数，创建一个临时环境实例
env_temp = GomokuEnv()
state_dim = int(np.prod(env_temp.observation().shape))
action_dim = env_temp.board_size * env_temp.board_size

# 全局网络和目标网络（多线程共享同一份网络参数）
global_policy_net = DQN(state_dim, action_dim).to(device)
global_target_net = DQN(state_dim, action_dim).to(device)
global_target_net.load_state_dict(global_policy_net.state_dict())
global_target_net.eval()

optimizer = optim.Adam(global_policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(MEMORY_SIZE)

# 用于记录各线程训练时每个 episode 的奖励（便于后续绘图）
reward_histories = [[] for _ in range(NUM_THREADS)]
# 锁，用于在更新目标网络时保证线程安全
target_update_lock = threading.Lock()


# ============================
# 训练线程函数（数据采集 + 参数更新）
# ============================
def train_worker(thread_id):
    local_epsilon = EPS_START
    num_episodes_thread = NUM_EPISODES // NUM_THREADS  # 每个线程负责的 episode 数
    env = GomokuEnv()  # 每个线程创建独立环境实例
    for ep in range(num_episodes_thread):
        state = env.reset().flatten()
        done = False
        total_reward = 0
        while not done:
            # ε-greedy 策略
            if random.random() < local_epsilon:
                legal_indices = [i for i in range(len(env.legal_actions)) if env.legal_actions[i] == 1]
                action = random.choice(legal_indices)
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    q_values = global_policy_net(state_tensor)
                    legal_mask = torch.tensor(env.legal_actions, dtype=torch.float32).to(device)
                    # 对非法动作加上大负值，确保不被选中
                    masked_q_values = q_values + (legal_mask - 1) * 1e6
                    action = masked_q_values.argmax().item()
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # 如果经验足够，则更新全局网络
            if len(memory) >= BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.tensor(states, dtype=torch.float32).to(device)
                actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
                q_values = global_policy_net(states).gather(1, actions)
                max_next_q = global_target_net(next_states).max(1, keepdim=True)[0].detach()
                target_q = rewards + GAMMA * max_next_q * (1 - dones)
                loss = nn.functional.mse_loss(q_values, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 每个 episode 结束后衰减 ε
        local_epsilon = max(EPS_END, local_epsilon * EPS_DECAY)
        reward_histories[thread_id].append(total_reward)

        # 每 TARGET_UPDATE 个 episode，同步一次目标网络
        if (ep + 1) % TARGET_UPDATE == 0:
            with target_update_lock:
                global_target_net.load_state_dict(global_policy_net.state_dict())

        print(f"Thread {thread_id} - Episode {ep}: Total Reward = {total_reward}")


# ============================
# 评估函数（固定策略，不再探索）
# ============================
def evaluate_policy(num_episodes=100):
    env = GomokuEnv()
    total_rewards = []
    global_policy_net.eval()
    for ep in range(num_episodes):
        state = env.reset().flatten()
        done = False
        ep_reward = 0
        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = global_policy_net(state_tensor)
                legal_mask = torch.tensor(env.legal_actions, dtype=torch.float32).to(device)
                masked_q_values = q_values + (legal_mask - 1) * 1e6
                action = masked_q_values.argmax().item()
            next_state, reward, done, _ = env.step(action)
            state = next_state.flatten()
            ep_reward += reward
        total_rewards.append(ep_reward)
        print(f"Evaluation Episode {ep}: Reward = {ep_reward}")
    avg_reward = np.mean(total_rewards)
    print(f"Average Evaluation Reward over {num_episodes} episodes: {avg_reward}")
    return total_rewards


# ============================
# 主函数：启动训练、绘制奖励曲线、评估模型
# ============================
def main():
    set_seed(42)
    threads = []
    for tid in range(NUM_THREADS):
        t = threading.Thread(target=train_worker, args=(tid,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    # 合并各线程奖励记录用于绘图
    all_rewards = []
    for hist in reward_histories:
        all_rewards.extend(hist)
    plt.figure(figsize=(10, 6))
    plt.plot(all_rewards, label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward Over Episodes (Multithreaded)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 保存训练好的模型
    torch.save(global_policy_net.state_dict(), "dqn_gomoku.pth")
    print("Model saved.")

    # 评估模型
    evaluate_policy()


if __name__ == "__main__":
    main()
