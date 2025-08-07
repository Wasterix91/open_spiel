# dqn_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple

DQNConfig = namedtuple("DQNConfig", [
    "learning_rate", "batch_size", "gamma", "epsilon_start",
    "epsilon_end", "epsilon_decay", "buffer_size", "target_update_freq"
])

DEFAULT_CONFIG = DQNConfig(
    learning_rate=0.001,
    batch_size=64,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay=0.995,
    buffer_size=100_000,
    target_update_freq=1000
)

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])

    def add(self, *args):
        self.buffer.append(self.Experience(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, num_actions, config=None, device="cpu"):
        self.device = torch.device(device)
        self.state_size = state_size
        self.num_actions = num_actions
        self.config = config or DEFAULT_CONFIG

        self.q_network = QNetwork(state_size, num_actions).to(self.device)
        self.target_network = QNetwork(state_size, num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)
        self.buffer = ReplayBuffer(self.config.buffer_size)

        self.epsilon = self.config.epsilon_start
        self.steps_done = 0

    def select_action(self, state, legal_actions):
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor).cpu().numpy().squeeze()
                masked_q = np.full_like(q_values, -np.inf)
                masked_q[legal_actions] = q_values[legal_actions]
                return int(np.argmax(masked_q))

    def train_step(self):
        if len(self.buffer) < self.config.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.config.batch_size)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            targets = rewards + self.config.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

    def save(self, path):
        torch.save(self.q_network.state_dict(), path)

    def restore(self, path):
        self.q_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(self.q_network.state_dict())
