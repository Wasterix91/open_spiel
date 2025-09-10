# agents/dqn_agent.py
import random
from collections import deque, namedtuple
from typing import Optional, Sequence, Iterable, List

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
        "epsilon_decay",       # multiplicative decay per train_step (0<decay<=1) or frames-based if <1? -> we keep multiplicative
        "buffer_size",
        "target_update_freq",  # hard update every N train steps (ignored if soft_target_tau>0)
        "soft_target_tau",     # Polyak/soft update (0 => off)
        "max_grad_norm",       # grad clipping (None/0 => off)
        "use_double_dqn",      # Double-DQN targets
        "loss_huber_delta",    # None => MSE, else SmoothL1Loss with delta
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


# ===================== Replay Buffer (mit Legal-Mask) ===================== #
class ReplayBuffer:
    """
    Kompatibel zu deinem k1a2-Skript:
      - hat self.buffer (deque) aus Experience(...)
      - Experience Felder: state, action, reward, next_state, done, next_legal_mask
      - add(...) akzeptiert next_legal_actions=
    """
    def __init__(self, capacity: int, num_actions: int):
        self.buffer = deque(maxlen=int(capacity))
        self.num_actions = int(num_actions)
        fields = ["state", "action", "reward", "next_state", "done", "next_legal_mask"]
        self.Experience = namedtuple("Experience", fields)

    def _to_mask(self, next_legal_actions: Optional[Iterable[int]]):
        if next_legal_actions is None:
            return None
        mask = np.zeros((self.num_actions,), dtype=np.float32)
        if isinstance(next_legal_actions, np.ndarray):
            idxs = next_legal_actions.tolist()
        else:
            idxs = list(next_legal_actions)
        if len(idxs) > 0:
            mask[idxs] = 1.0
        return mask

    def add(self, state, action, reward, next_state, done, next_legal_actions: Optional[Iterable[int]] = None):
        mask = self._to_mask(next_legal_actions)
        self.buffer.append(self.Experience(state, action, reward, next_state, done, mask))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)

        # Falls einzelne Transitions keine Maske enthalten, behandle als "alle legal"
        masks_filled = []
        for m in next_masks:
            if m is None:
                mf = np.ones((self.num_actions,), dtype=np.float32)
            else:
                mf = m.astype(np.float32, copy=False)
            masks_filled.append(mf)
        next_masks_arr = np.stack(masks_filled, axis=0)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            next_masks_arr,  # shape: [B, A]
        )

    def __len__(self):
        return len(self.buffer)


# ===================== DQN Agent ===================== #
class DQNAgent:
    """
    - select_action(state, legal_actions): maskiert ILLEGAL mit -inf
    - Buffer speichert next_legal_mask und train_step maskiert Targets
    - epsilon ist kompatibel zu deinem Skript (wird dort ausgedruckt / auf 0.0 gesetzt)
    - save/restore nutzt *_qnet.pt / *_tgt.pt wie zuvor
    """
    def __init__(self, state_size: int, num_actions: int, config: DQNConfig = None, device="cpu"):
        self.device = torch.device(device)
        self.state_size = int(state_size)
        self.num_actions = int(num_actions)
        self.config = config or DEFAULT_CONFIG

        self.q_network = QNetwork(self.state_size, self.num_actions).to(self.device)
        self.target_network = QNetwork(self.state_size, self.num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)
        if self.config.loss_huber_delta is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.SmoothL1Loss(beta=self.config.loss_huber_delta)

        self.buffer = ReplayBuffer(self.config.buffer_size, num_actions=self.num_actions)

        # Epsilon-Handling kompatibel halten
        self.epsilon = float(self.config.epsilon_start)
        self.steps_done = 0  # train_step-Zähler (für target_update_freq)

    # ---------- Utils ---------- #
    @staticmethod
    def _mask_q_numpy(q_values: np.ndarray, legal_actions: Sequence[int]) -> np.ndarray:
        masked = np.full_like(q_values, -np.inf, dtype=np.float32)
        masked[list(legal_actions)] = q_values[list(legal_actions)]
        return masked

    @staticmethod
    def _masked_argmax_torch(q_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        q_values: [B, A], mask: [B, A] in {0,1}
        Gibt indices [B] der argmax über maskierte q zurück.
        """
        neg_inf = torch.finfo(q_values.dtype).min
        q_masked = torch.where(mask > 0, q_values, neg_inf)
        return torch.argmax(q_masked, dim=1)

    @staticmethod
    def _masked_max_torch(q_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        neg_inf = torch.finfo(q_values.dtype).min
        q_masked = torch.where(mask > 0, q_values, neg_inf)
        return torch.max(q_masked, dim=1).values

    # ---------- API ---------- #
    def select_action(self, state: np.ndarray, legal_actions: Sequence[int]) -> int:
        # Epsilon-Greedy nur über LEGAL actions
        if random.random() < self.epsilon:
            return int(random.choice(list(legal_actions)))
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_network(state_t).squeeze(0).cpu().numpy()
        q_masked = self._mask_q_numpy(q_values, legal_actions)
        return int(np.argmax(q_masked))

    def train_step(self):
        if len(self.buffer) < self.config.batch_size:
            return

        states, actions, rewards, next_states, dones, next_masks = self.buffer.sample(self.config.batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)  # [B,1]
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)  # [B,1]
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)  # [B,1]
        next_masks = torch.tensor(next_masks, dtype=torch.float32, device=self.device)  # [B, A]

        # Q(s,a)
        q_sa = self.q_network(states).gather(1, actions).squeeze(1)  # [B]

        with torch.no_grad():
            q_next_online = self.q_network(next_states)   # [B, A]
            q_next_target = self.target_network(next_states)  # [B, A]

            if self.config.use_double_dqn:
                # a* = argmax_a q_online(ns,a) über LEGAL
                a_star = self._masked_argmax_torch(q_next_online, next_masks)  # [B]
                q_next_sa = q_next_target.gather(1, a_star.view(-1, 1)).squeeze(1)  # [B]
            else:
                q_next_sa = self._masked_max_torch(q_next_target, next_masks)  # [B]

            targets = rewards.squeeze(1) + self.config.gamma * q_next_sa * (1.0 - dones.squeeze(1))  # [B]

        loss = self.criterion(q_sa, targets)

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.max_grad_norm and self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        # Schrittzähler & Epsilon-Decay (multiplikativ wie zuvor)
        self.steps_done += 1
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

        # Target-Update
        tau = float(self.config.soft_target_tau or 0.0)
        if tau > 0.0:
            with torch.no_grad():
                for tgt, src in zip(self.target_network.parameters(), self.q_network.parameters()):
                    tgt.data.mul_(1.0 - tau).add_(tau * src.data)
        else:
            if self.steps_done % int(self.config.target_update_freq) == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
                self.target_network.eval()

    # ---------- Persistence ---------- #
    def save(self, path_base: str):
        torch.save(self.q_network.state_dict(), f"{path_base}_qnet.pt")
        torch.save(self.target_network.state_dict(), f"{path_base}_tgt.pt")

    def restore(self, path_base: str):
        self.q_network.load_state_dict(torch.load(f"{path_base}_qnet.pt", map_location=self.device))
        self.target_network.load_state_dict(torch.load(f"{path_base}_tgt.pt", map_location=self.device))
        self.target_network.eval()
