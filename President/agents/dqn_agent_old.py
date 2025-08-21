# agents/dqn_agent.py
import random
from collections import deque, namedtuple
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ===================== DQN Config ===================== #
DQNConfig = namedtuple(
    "DQNConfig",
    [
        "learning_rate",
        "batch_size",
        "gamma",
        "epsilon_start",
        "epsilon_end",
        "epsilon_decay",
        "buffer_size",
        "target_update_freq",  # Hard-Update alle N train_steps (wenn tau == 0)
        "soft_target_tau",     # Polyak/Soft-Update (0 => aus)
        "max_grad_norm",       # Grad-Clipping (None/0 => aus)
        "use_double_dqn",      # Double-DQN Targets
        "loss_huber_delta",    # None => MSE, sonst SmoothL1Loss mit delta
    ],
)

DEFAULT_CONFIG = DQNConfig(
    learning_rate=1e-3,
    batch_size=64,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay=0.995,
    buffer_size=100_000,
    target_update_freq=1000,
    soft_target_tau=0.0,
    max_grad_norm=0.0,
    use_double_dqn=True,
    loss_huber_delta=1.0,
)


# ===================== Netzarchitektur ===================== #
class QNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ===================== Replay Buffer ===================== #
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        fields = ["state", "action", "reward", "next_state", "done"]
        self.Experience = namedtuple("Experience", fields)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append(self.Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ===================== DQN Agent ===================== #
class DQNAgent:
    def __init__(self, state_size: int, num_actions: int, config: DQNConfig = None, device="cpu"):
        self.device = torch.device(device)
        self.state_size = state_size
        self.num_actions = num_actions
        self.config = config or DEFAULT_CONFIG

        self.q_network = QNetwork(state_size, num_actions).to(self.device)
        self.target_network = QNetwork(state_size, num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)
        if self.config.loss_huber_delta is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.SmoothL1Loss(beta=self.config.loss_huber_delta)

        self.buffer = ReplayBuffer(self.config.buffer_size)

        self.epsilon = float(self.config.epsilon_start)
        self.steps_done = 0

    def select_action(self, state: np.ndarray, legal_actions: Sequence[int]) -> int:
        if random.random() < self.epsilon:
            return int(random.choice(list(legal_actions)))
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_network(state_t).squeeze(0).cpu().numpy()
        masked_q = np.full_like(q_values, -np.inf, dtype=np.float32)
        masked_q[list(legal_actions)] = q_values[list(legal_actions)]
        return int(np.argmax(masked_q))

    def train_step(self):
        if len(self.buffer) < self.config.batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.config.batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            q_next_online = self.q_network(next_states)
            q_next_target = self.target_network(next_states)
            if self.config.use_double_dqn:
                next_actions = torch.argmax(q_next_online, dim=1, keepdim=True)
                q_next = q_next_target.gather(1, next_actions)
            else:
                q_next, _ = torch.max(q_next_target, dim=1, keepdim=True)
            targets = rewards + self.config.gamma * q_next * (1.0 - dones)

        loss = self.criterion(q_sa, targets)
        self.optimizer.zero_grad()
        loss.backward()
        if self.config.max_grad_norm and self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        self.steps_done += 1
        self._update_target_net()
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

    def _update_target_net(self):
        tau = float(self.config.soft_target_tau or 0.0)
        if tau > 0.0:
            with torch.no_grad():
                for tgt, src in zip(self.target_network.parameters(), self.q_network.parameters()):
                    tgt.data.mul_(1.0 - tau).add_(tau * src.data)
        else:
            if self.steps_done % int(self.config.target_update_freq) == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
                self.target_network.eval()

    def save(self, path_base: str):
        torch.save(self.q_network.state_dict(), f"{path_base}_qnet.pt")
        torch.save(self.target_network.state_dict(), f"{path_base}_tgt.pt")

    def restore(self, path_base: str):
        self.q_network.load_state_dict(torch.load(f"{path_base}_qnet.pt", map_location=self.device))
        self.target_network.load_state_dict(torch.load(f"{path_base}_tgt.pt", map_location=self.device))
        self.target_network.eval()
