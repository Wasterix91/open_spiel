# agents/dqn_agent.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, random
from dataclasses import dataclass
from typing import Optional, Sequence, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ===================== DQN Config ===================== #
@dataclass
class DQNConfig:
    # Optimizer
    learning_rate: float = 1e-3
    optimizer: str = "adam"           # "adam" | "rmsprop"
    rmsprop_alpha: float = 0.95
    rmsprop_eps: float = 1e-2

    # Training
    batch_size: int = 64
    gamma: float = 0.99
    max_grad_norm: float = 0.0        # 0/None => aus

    # Exploration (ε-greedy)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.995       # benutzt bei epsilon_decay_type="multiplicative"
    epsilon_decay_type: str = "multiplicative"  # "multiplicative" | "linear"
    epsilon_decay_frames: int = 1_000_000       # benutzt bei "linear" (paper-nah)

    # Replay
    buffer_size: int = 100_000

    # Target-Net
    target_update_freq: int = 1000     # harte Kopie alle N train steps (wenn tau==0)
    soft_target_tau: float = 0.0       # Polyak (0 => aus)

    # Algorithmus-Variante
    use_double_dqn: bool = True

    # Loss
    loss_huber_delta: Optional[float] = 1.0     # None => MSE, sonst SmoothL1 mit beta=delta

    # Netz
    hidden_sizes: Tuple[int, int] = (128, 128)

    # Device
    device: str = "cpu"


DEFAULT_CONFIG = DQNConfig()


# ===================== Netzarchitektur ===================== #
class QNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden: Sequence[int] = (128, 128)):
        super().__init__()
        layers: List[nn.Module] = []
        last = input_size
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, output_size)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ===================== Replay Buffer (mit Legal-Mask) ===================== #
class ReplayBuffer:
    """
    Speichert (state, action, reward, next_state, done, next_legal_mask).
    next_legal_mask ist eine 0/1-Maske der Größe [num_actions].
    """
    class Experience(tuple):  # nur Typmarker
        pass

    def __init__(self, capacity: int, num_actions: int):
        from collections import deque, namedtuple
        self.buffer = deque(maxlen=int(capacity))
        self.num_actions = int(num_actions)
        self.Experience = namedtuple(
            "Experience", ["state", "action", "reward", "next_state", "done", "next_legal_mask"]
        )

    def _to_mask(self, next_legal_actions: Optional[Iterable[int]]):
        if next_legal_actions is None:
            return None
        mask = np.zeros((self.num_actions,), dtype=np.float32)
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

        # fehlende Masken => "alle legal"
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
            next_masks_arr,  # [B, A]
        )

    def __len__(self):
        return len(self.buffer)


# ===================== DQN Agent ===================== #
class DQNAgent:
    """
    - ε-greedy über LEGAL actions (illegale werden maskiert)
    - Replay speichert next_legal_mask; Targets werden entsprechend maskiert
    - Hard- oder Polyak-Update des Target-Netzes
    - save/restore kompatibel zu *_qnet.pt / *_tgt.pt
    """
    def __init__(self, state_size: int, num_actions: int, config: Optional[DQNConfig] = None, device: Optional[str] = None):
        self.config = config or DEFAULT_CONFIG
        self.device = torch.device(device or self.config.device)
        self.state_size = int(state_size)
        self.num_actions = int(num_actions)

        # Netze
        self.q_network = QNetwork(self.state_size, self.num_actions, hidden=self.config.hidden_sizes).to(self.device)
        self.target_network = QNetwork(self.state_size, self.num_actions, hidden=self.config.hidden_sizes).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        if (self.config.optimizer or "adam").lower() == "rmsprop":
            self.optimizer = optim.RMSprop(
                self.q_network.parameters(),
                lr=self.config.learning_rate,
                alpha=self.config.rmsprop_alpha,
                eps=self.config.rmsprop_eps,
            )
        else:
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)

        # Loss
        if self.config.loss_huber_delta is None:
            self.criterion = nn.MSELoss()
        else:
            # PyTorch SmoothL1Loss(beta=delta)
            self.criterion = nn.SmoothL1Loss(beta=float(self.config.loss_huber_delta))

        # Replay
        self.buffer = ReplayBuffer(self.config.buffer_size, num_actions=self.num_actions)

        # Epsilon / Schrittzähler
        self.epsilon = float(self.config.epsilon_start)
        self._eps_steps_done = 0            # zählt Trainingsschritte (train_step Aufrufe)
        self._eps_mode = (self.config.epsilon_decay_type or "multiplicative").lower()
        self._eps_decay = float(self.config.epsilon_decay)
        self._eps_frames = max(1, int(self.config.epsilon_decay_frames))

        self.steps_done = 0                 # Zähler für Target-Update

        # ---------- DEBUG: Trainings-Tensor-Inspektion ----------
        self._debug = False
        self._debug_max_batches = 0
        self._debug_batches_seen = 0

    # ---------- Debug-API ---------- #
    def enable_debug(self, enabled: bool = True, max_batches: int = 3):
        """
        Aktiviert einen ausführlichen Dump des Trainings-Batches in train_step():
        - Formen/Dtypen/Min/Max der states/next_states/next_masks
        - NaN/Inf-Check
        - Beispiel-Slices der ersten Transition ([:32])
        - Persistierte .pt-Dateien je Batch: debug_states_batch_XXX.pt etc.
        """
        self._debug = bool(enabled)
        self._debug_max_batches = int(max_batches)
        self._debug_batches_seen = 0

    # ---------- Utils ---------- #
    @staticmethod
    def _mask_q_numpy(q_values: np.ndarray, legal_actions: Sequence[int]) -> np.ndarray:
        masked = np.full_like(q_values, -np.inf, dtype=np.float32)
        masked[list(legal_actions)] = q_values[list(legal_actions)]
        return masked

    @staticmethod
    def _masked_argmax_torch(q_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """q_values: [B,A], mask: [B,A] in {0,1}"""
        neg_inf = torch.finfo(q_values.dtype).min
        q_masked = torch.where(mask > 0, q_values, neg_inf)
        return torch.argmax(q_masked, dim=1)

    @staticmethod
    def _masked_max_torch(q_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        neg_inf = torch.finfo(q_values.dtype).min
        q_masked = torch.where(mask > 0, q_values, neg_inf)
        return torch.max(q_masked, dim=1).values

    def _update_epsilon(self):
        self._eps_steps_done += 1
        if self._eps_mode == "linear":
            frac = min(1.0, self._eps_steps_done / float(self._eps_frames))
            self.epsilon = self.config.epsilon_start + frac * (self.config.epsilon_end - self.config.epsilon_start)
        else:
            # multiplikativ (bisheriges Verhalten)
            self.epsilon = max(self.config.epsilon_end, self.epsilon * self._eps_decay)

    # ---------- API ---------- #
    def select_action(self, state: np.ndarray, legal_actions: Sequence[int]) -> int:
        # ε-greedy NUR über LEGAL actions
        if random.random() < self.epsilon:
            if len(legal_actions) == 0:
                return int(random.randrange(self.num_actions))
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

        states = torch.tensor(states, dtype=torch.float32, device=self.device)                     # [B,D]
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)        # [B,1]
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)     # [B,1]
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)          # [B,D]
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)         # [B,1]
        next_masks = torch.tensor(next_masks, dtype=torch.float32, device=self.device)            # [B,A]

        # ---------- DEBUG-DUMP DES TRAININGS-TENSORS ----------
        if getattr(self, "_debug", False) and self._debug_batches_seen < self._debug_max_batches:
            with torch.no_grad():
                try:
                    s_min = float(states.min().detach().cpu())
                    s_max = float(states.max().detach().cpu())
                    ns_min = float(next_states.min().detach().cpu())
                    ns_max = float(next_states.max().detach().cpu())
                except Exception:
                    s_min = s_max = ns_min = ns_max = float("nan")

                print("[DQN.train_step] states:", tuple(states.shape), states.dtype,
                      "min/max:", f"{s_min:.6f}/{s_max:.6f}")
                print("[DQN.train_step] next_states:", tuple(next_states.shape), next_states.dtype,
                      "min/max:", f"{ns_min:.6f}/{ns_max:.6f}")
                print("[DQN.train_step] actions:", tuple(actions.shape), actions.dtype)
                print("[DQN.train_step] next_masks:", tuple(next_masks.shape), next_masks.dtype)

                bad_states = torch.isnan(states).any() or torch.isinf(states).any()
                bad_next   = torch.isnan(next_states).any() or torch.isinf(next_states).any()
                print("[DQN.train_step] has_nan_or_inf(states):", bool(bad_states),
                      "has_nan_or_inf(next_states):", bool(bad_next))

                # Beispiel-Transition
                s0 = states[0]
                ns0 = next_states[0]
                m0 = next_masks[0]
                print("[DQN.train_step] states[0][:32]:", s0[:32].tolist())
                print("[DQN.train_step] next_states[0][:32]:", ns0[:32].tolist())
                nz = torch.nonzero(m0, as_tuple=False).squeeze(-1).detach().cpu()
                print("[DQN.train_step] next_masks[0].nonzero:", nz.tolist() if nz.numel() > 0 else [])

                # Persistieren zur Offline-Analyse
                torch.save(states.detach().cpu(),      f"debug_states_batch_{self._debug_batches_seen:03d}.pt")
                torch.save(next_states.detach().cpu(), f"debug_next_states_batch_{self._debug_batches_seen:03d}.pt")
                torch.save(next_masks.detach().cpu(),  f"debug_next_masks_batch_{self._debug_batches_seen:03d}.pt")

            self._debug_batches_seen += 1
        # ---------- Ende DEBUG ----------

        # Q(s,a)
        q_sa = self.q_network(states).gather(1, actions).squeeze(1)                               # [B]

        with torch.no_grad():
            q_next_online = self.q_network(next_states)                                           # [B,A]
            q_next_target = self.target_network(next_states)                                      # [B,A]

            if self.config.use_double_dqn:
                a_star = self._masked_argmax_torch(q_next_online, next_masks)                     # [B]
                q_next_sa = q_next_target.gather(1, a_star.view(-1, 1)).squeeze(1)               # [B]
            else:
                q_next_sa = self._masked_max_torch(q_next_target, next_masks)                     # [B]

            targets = rewards.squeeze(1) + self.config.gamma * q_next_sa * (1.0 - dones.squeeze(1))

        loss = self.criterion(q_sa, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.max_grad_norm and self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        # Schritt- & Epsilon-Update
        self.steps_done += 1
        self._update_epsilon()

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
        os.makedirs(os.path.dirname(path_base), exist_ok=True)
        torch.save(self.q_network.state_dict(), f"{path_base}_qnet.pt")
        torch.save(self.target_network.state_dict(), f"{path_base}_tgt.pt")

    def restore(self, path_base: str):
        self.q_network.load_state_dict(torch.load(f"{path_base}_qnet.pt", map_location=self.device))
        self.target_network.load_state_dict(torch.load(f"{path_base}_tgt.pt", map_location=self.device))
        self.target_network.eval()
