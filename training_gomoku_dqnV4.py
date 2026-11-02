import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from alpha_zero.envs.gomoku import GomokuEnv

import json  # 用于存储数据

rewards_list = []  # 用于存储每个 episode 的 total_reward

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 100000
TARGET_UPDATE = 1000
MEMORY_SIZE = 10000
LEARNING_RATE = 1e-3

# 选择 MPS 设备（Mac GPU）或 CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# Define the Q-Network
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

# Experience Replay Buffer
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Training loop
def train_dqn():
    env = GomokuEnv()
    state_dim = int(np.prod(env.observation().shape))
    action_dim = env.board_size * env.board_size

    # 将网络迁移到 device 上（MPS 或 CPU）
    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)

    steps_done = 0
    epsilon = EPS_START

    for episode in range(5000):  # Number of episodes
        state = env.reset()
        state = state.flatten()
        done = False
        total_reward = 0

        while not done:
            # Select action (epsilon-greedy policy)
            if random.random() < epsilon:
                legal_indices = [i for i in range(len(env.legal_actions)) if env.legal_actions[i] == 1]
                action = random.choice(legal_indices)
            else:
                with torch.no_grad():
                    # 注意将 state 转换到 device 上
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    q_values = policy_net(state_tensor)
                    legal_mask = torch.tensor(env.legal_actions, dtype=torch.float32).to(device)
                    masked_q_values = q_values + (legal_mask - 1) * 1e6  # Mask illegal moves
                    action = masked_q_values.argmax().item()

            # Execute action
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Train the network
            if len(memory) > BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                # 将 batch 数据转移到 device 上
                states = torch.tensor(states, dtype=torch.float32).to(device)
                actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

                q_values = policy_net(states).gather(1, actions)
                max_next_q_values = target_net(next_states).max(1, keepdim=True)[0].detach()
                target_q_values = rewards + GAMMA * max_next_q_values * (1 - dones)

                loss = nn.functional.mse_loss(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            steps_done += 1
            epsilon = max(EPS_END, EPS_START - steps_done / EPS_DECAY)

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

    torch.save(policy_net.state_dict(), "dqn_gomoku.pth")
    print("Training finished and model saved.")
def evaluate_dqn(num_episodes=100):
    env = GomokuEnv()
    state_dim = int(np.prod(env.observation().shape))
    action_dim = env.board_size * env.board_size

    # 创建网络并加载训练好的权重
    policy_net = DQN(state_dim, action_dim).to(device)
    policy_net.load_state_dict(torch.load("dqn_gomoku.pth", map_location=device))
    policy_net.eval()  # 设置为评估模式

    total_rewards = []
    for episode in range(num_episodes):
        state = env.reset().flatten()
        done = False
        episode_reward = 0

        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = policy_net(state_tensor)
                legal_mask = torch.tensor(env.legal_actions, dtype=torch.float32).to(device)
                # 这里取消随机探索，始终选择最高 Q 值的合法动作
                masked_q_values = q_values + (legal_mask - 1) * 1e6
                action = masked_q_values.argmax().item()

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state.flatten()

        total_rewards.append(episode_reward)
        print(f"Episode {episode+1}: reward = {episode_reward}")

    print("Average reward over {} episodes: {:.2f}".format(num_episodes, np.mean(total_rewards)))

def continue_training_dqn(start_episode=5000, additional_episodes=2000):
    env = GomokuEnv()
    state_dim = int(np.prod(env.observation().shape))
    action_dim = env.board_size * env.board_size

    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)

    # 加载之前保存的模型权重
    policy_net.load_state_dict(torch.load("dqn_gomoku.pth", map_location=device))
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)

    # 可以选择将 epsilon 固定在一个较小的值，比如 EPS_END，减少随机性
    epsilon = EPS_END

    steps_done = 0

    for episode in range(start_episode, start_episode + additional_episodes):
        state = env.reset().flatten()
        done = False
        total_reward = 0

        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = policy_net(state_tensor)
                legal_mask = torch.tensor(env.legal_actions, dtype=torch.float32).to(device)
                masked_q_values = q_values + (legal_mask - 1) * 1e6
                action = masked_q_values.argmax().item()

            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(memory) > BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.tensor(states, dtype=torch.float32).to(device)
                actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

                q_values = policy_net(states).gather(1, actions)
                max_next_q_values = target_net(next_states).max(1, keepdim=True)[0].detach()
                target_q_values = rewards + GAMMA * max_next_q_values * (1 - dones)

                loss = nn.functional.mse_loss(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            steps_done += 1

        # 存储 episode 的 reward
        rewards_list.append(total_reward)

        # 每 100 个 episode 保存一次数据
        if episode:
            with open("rewards.json", "w") as f:
                json.dump(rewards_list, f)

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode}, Total Reward: {total_reward}")

    # 保存继续训练后的模型
    # 训练结束后，保存最终的 reward 数据
    with open("rewards.json", "w") as f:
        json.dump(rewards_list, f)
    torch.save(policy_net.state_dict(), "dqn_gomoku_finetuned.pth")
    print("Continued training finished and model saved.")

if __name__ == "__main__":
    print("Start Training")
    #train_dqn()
    continue_training_dqn(start_episode=5000, additional_episodes=100)
    with open("rewards.json", "r") as f:
        rewards_list = json.load(f)

    # 画图
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(rewards_list)), rewards_list, label="Total Reward per Episode", color="b")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Progress: Episode vs Reward")
    plt.legend()
    plt.grid()
    plt.show()
