import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

from nfsp_buffers import ExperienceReplayBuffer, ReservoirBuffer
from nfsp_network import RLQNetwork, SLPolicyNetwork

class NFSPAgent:
    def __init__(
        self,
        state_size,
        num_actions,
        anticipatory_param=0.1,
        replay_capacity=100000,
        reservoir_capacity=100000,
        batch_size=128,
        rl_lr=1e-3,
        sl_lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=10000,
        device="cpu"
    ):
        self.state_size = state_size
        self.num_actions = num_actions
        self.anticipatory_param = anticipatory_param
        self.batch_size = batch_size
        self.device = device

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps = 0

        # Networks
        self.rl_net = RLQNetwork(state_size, num_actions).to(device)
        self.target_net = RLQNetwork(state_size, num_actions).to(device)
        self.target_net.load_state_dict(self.rl_net.state_dict())
        self.target_net.eval()

        self.sl_net = SLPolicyNetwork(state_size, num_actions).to(device)

        # Optimizers
        self.rl_optimizer = optim.Adam(self.rl_net.parameters(), lr=rl_lr)
        self.sl_optimizer = optim.Adam(self.sl_net.parameters(), lr=sl_lr)

        # Buffers
        self.replay_buffer = ExperienceReplayBuffer(replay_capacity)
        self.reservoir_buffer = ReservoirBuffer(reservoir_capacity)

        # Mode (RL or SL)
        self.use_rl = True
        
    def act_sl(self, obs, legal_actions):
        state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits = self.sl_net(state).detach().cpu().numpy()[0]
        masked_logits = np.full_like(logits, -np.inf)
        masked_logits[legal_actions] = logits[legal_actions]
        action = int(np.argmax(masked_logits))
        return action

    def select_action(self, state, legal_actions):
        self.steps += 1
        self.epsilon = max(self.epsilon_end, self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay)

        # Convert state
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Decide whether to use RL or SL
        self.use_rl = np.random.rand() < self.anticipatory_param

        if self.use_rl:
            # ε-greedy RL policy
            if np.random.rand() < self.epsilon:
                return random.choice(legal_actions)
            with torch.no_grad():
                q_values = self.rl_net(state_tensor).cpu().numpy()[0]
            legal_qs = {a: q_values[a] for a in legal_actions}
            return max(legal_qs, key=legal_qs.get)
        else:
            # SL policy
            with torch.no_grad():
                probs = self.sl_net(state_tensor).cpu().numpy()[0]
            legal_probs = np.array([probs[a] if a in legal_actions else 0.0 for a in range(self.num_actions)])
            if legal_probs.sum() == 0:
                return random.choice(legal_actions)
            legal_probs /= legal_probs.sum()
            return np.random.choice(np.arange(self.num_actions), p=legal_probs)

    def observe_transition(self, state, action, reward, next_state, done):
        # Store for RL
        self.replay_buffer.add((state, action, reward, next_state, done))

        # Store for SL if in RL mode (only best response updates SL)
        if self.use_rl:
            self.reservoir_buffer.add((state, action))

    def train_step(self):
        if len(self.replay_buffer) >= self.batch_size:
            transitions = self.replay_buffer.sample(self.batch_size)
            states, actions, rewards, next_states, dones = zip(*transitions)

            states = torch.tensor(states, dtype=torch.float32, device=self.device)
            actions = torch.tensor(actions, dtype=torch.long, device=self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
            dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

            # Compute target Q values
            with torch.no_grad():
                target_q = self.target_net(next_states).max(1)[0]
                targets = rewards + (1 - dones) * 0.99 * target_q

            q_values = self.rl_net(states).gather(1, actions.unsqueeze(1)).squeeze()
            rl_loss = nn.functional.mse_loss(q_values, targets)

            self.rl_optimizer.zero_grad()
            rl_loss.backward()
            self.rl_optimizer.step()

        if len(self.reservoir_buffer) >= self.batch_size:
            samples = self.reservoir_buffer.sample(self.batch_size)
            states, actions = zip(*samples)
            states = torch.tensor(states, dtype=torch.float32, device=self.device)
            actions = torch.tensor(actions, dtype=torch.long, device=self.device)

            logits = self.sl_net(states)
            sl_loss = nn.functional.cross_entropy(logits, actions)

            self.sl_optimizer.zero_grad()
            sl_loss.backward()
            self.sl_optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.rl_net.state_dict())

    def save(self, path_prefix):
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        torch.save(self.rl_net.state_dict(), f"{path_prefix}_q.pt")
        torch.save(self.sl_net.state_dict(), f"{path_prefix}_sl.pt")

    def load(self, path_prefix):
        self.rl_net.load_state_dict(torch.load(f"{path_prefix}_q.pt", map_location=self.device))
        self.sl_net.load_state_dict(torch.load(f"{path_prefix}_sl.pt", map_location=self.device))
        self.update_target_network()
