import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

from alpha_zero.envs.gomoku import GomokuEnv

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 10000
TARGET_UPDATE = 1000
MEMORY_SIZE = 10000
LEARNING_RATE = 1e-3

device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")


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
    action_dim = env.board_size*env.board_size

    policy_net = DQN(state_dim, action_dim).to(torch.device("cpu"))
    target_net = DQN(state_dim, action_dim).to(torch.device("cpu"))
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)

    steps_done = 0
    epsilon = EPS_START

    for episode in range(5000):  # Number of episodes
        state = env.reset()
        print("State",state)
        print("State shape",state.shape)
        state=state.flatten()
        done = False
        total_reward = 0

        while not done:
            # Select action (epsilon-greedy policy)
            if random.random() < epsilon:
                print("legal action",env.legal_actions)
                legal_indices = [i for i in range(len(env.legal_actions)) if env.legal_actions[i] == 1]
                print("Legal indices:", legal_indices, "Type:", type(legal_indices))
                action = random.choice(legal_indices)
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                    legal_mask = torch.tensor(env.legal_actions, dtype=torch.float32)
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

                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

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


if __name__ == "__main__":
    print("Start Training")
    train_dqn()
