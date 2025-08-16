import random
from collections import deque, namedtuple
from typing import Optional, Sequence, Tuple

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
    soft_target_tau=0.0,       # 0.005 wäre z.B. ein guter Startwert bei Soft-Updates
    max_grad_norm=0.0,         # 0.5–1.0 optional
    use_double_dqn=True,
    loss_huber_delta=1.0,      # auf None setzen, um MSE zu nutzen
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
    """Speichert optional die Next-Legal-Mask für masked Bellman-Targets."""
    def __init__(self, capacity: int, num_actions: Optional[int] = None):
        self.buffer = deque(maxlen=capacity)
        fields = ["state", "action", "reward", "next_state", "done", "next_legal_mask"]
        self.Experience = namedtuple("Experience", fields)
        self.num_actions = num_actions

    @staticmethod
    def _mask_from_legal(legal_actions: Sequence[int], num_actions: int) -> np.ndarray:
        mask = np.zeros(num_actions, dtype=np.float32)
        mask[list(legal_actions)] = 1.0
        return mask

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_legal_actions: Optional[Sequence[int]] = None,
    ):
        if next_legal_actions is not None:
            assert self.num_actions is not None, "num_actions muss gesetzt sein, um Masken zu bauen."
            next_mask = self._mask_from_legal(next_legal_actions, self.num_actions)
        else:
            next_mask = None
        self.buffer.append(self.Experience(state, action, reward, next_state, done, next_mask))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        # next_masks kann None enthalten → in train_step behandeln
        return states, actions, rewards, next_states, dones, next_masks

    def __len__(self) -> int:
        return len(self.buffer)


# ===================== DQN Agent ===================== #
class DQNAgent:
    def __init__(self, state_size: int, num_actions: int, config: DQNConfig = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.state_size = int(state_size)
        self.num_actions = int(num_actions)
        self.config = config or DEFAULT_CONFIG

        # Netze
        self.q_network = QNetwork(self.state_size, self.num_actions).to(self.device)
        self.target_network = QNetwork(self.state_size, self.num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimierer & Loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)
        if self.config.loss_huber_delta is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.SmoothL1Loss(beta=self.config.loss_huber_delta)

        # Replay
        self.buffer = ReplayBuffer(self.config.buffer_size, num_actions=self.num_actions)

        # Exploration
        self.epsilon = float(self.config.epsilon_start)
        self.steps_done = 0

    # --------- Action Selection (mit Legal-Masking) --------- #
    def select_action(self, state: np.ndarray, legal_actions: Sequence[int]) -> int:
        if random.random() < self.epsilon:
            return int(random.choice(list(legal_actions)))

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_network(state_t).squeeze(0).detach().cpu().numpy()

        masked_q = np.full_like(q_values, -np.inf, dtype=np.float32)
        masked_q[list(legal_actions)] = q_values[list(legal_actions)]
        action = int(np.argmax(masked_q))
        return action

    # --------- Training Step --------- #
    def train_step(self):
        if len(self.buffer) < self.config.batch_size:
            return

        (
            states_np,
            actions_np,
            rewards_np,
            next_states_np,
            dones_np,
            next_masks_list,
        ) = self.buffer.sample(self.config.batch_size)

        states = torch.tensor(states_np, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions_np, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards_np, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states_np, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones_np, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a)
        q_sa = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            q_next_online = self.q_network(next_states)     # für Double-DQN Argmax
            q_next_target = self.target_network(next_states)

            # Robust: Masken nur anwenden, wenn es überhaupt welche gibt.
            # Für Einträge ohne Maske (z.B. terminal) verwenden wir Ones -> kein Masking-Effekt.
            has_any_mask = any(m is not None for m in next_masks_list)
            if has_any_mask:
                masks_np = []
                for m, d in zip(next_masks_list, dones_np):
                    if m is None:
                        # Terminal-Transitions brauchen eigentlich kein Masking (gamma*(1-done)=0).
                        # Ones vermeidet -inf Effekte und behält "kein Masking".
                        masks_np.append(np.ones(self.num_actions, dtype=np.float32))
                    else:
                        masks_np.append(m)
                next_masks = torch.tensor(np.stack(masks_np, axis=0), dtype=torch.float32, device=self.device)

                neg_inf = torch.tensor(-1e9, dtype=q_next_online.dtype, device=self.device)
                q_next_online = torch.where(next_masks > 0, q_next_online, neg_inf)
                q_next_target = torch.where(next_masks > 0, q_next_target, neg_inf)

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

        # Epsilon-Decay (pro train_step)
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

    # --------- Speicher/Restore --------- #
    def save(self, path_base: str):
        """
        Speichert zwei Dateien:
          {path_base}_qnet.pt
          {path_base}_tgt.pt
        Beispiel:
          agent.save("models/dqn_model_01/train/run1_ep0001000")
          -> .../run1_ep0001000_qnet.pt
             .../run1_ep0001000_tgt.pt
        """
        torch.save(self.q_network.state_dict(), f"{path_base}_qnet.pt")
        torch.save(self.target_network.state_dict(), f"{path_base}_tgt.pt")

    def restore(self, path_base: str):
        """
        Lädt zwei Dateien:
          {path_base}_qnet.pt
          {path_base}_tgt.pt
        """
        self.q_network.load_state_dict(torch.load(f"{path_base}_qnet.pt", map_location=self.device))
        self.target_network.load_state_dict(torch.load(f"{path_base}_tgt.pt", map_location=self.device))
        self.target_network.eval()

    # Alternative Pfad-API mit expliziten Dateinamen (falls du fest benennen willst)
    def save_named(self, qnet_path: str, tgt_path: str):
        torch.save(self.q_network.state_dict(), qnet_path)
        torch.save(self.target_network.state_dict(), tgt_path)

    def restore_named(self, qnet_path: str, tgt_path: str):
        self.q_network.load_state_dict(torch.load(qnet_path, map_location=self.device))
        self.target_network.load_state_dict(torch.load(tgt_path, map_location=self.device))
        self.target_network.eval()

    # --------- Sonstiges --------- #
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_legal_actions: Optional[Sequence[int]] = None,
    ):
        """Bequemer Wrapper für ReplayBuffer.add mit optionaler Next-Legal-Maske."""
        self.buffer.add(state, action, reward, next_state, done, next_legal_actions)

    def _update_target_net(self):
        tau = float(self.config.soft_target_tau or 0.0)
        if tau > 0.0:
            with torch.no_grad():
                for tgt_param, src_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                    tgt_param.data.mul_(1.0 - tau).add_(tau * src_param.data)
        else:
            if self.steps_done % int(self.config.target_update_freq) == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
                self.target_network.eval()
