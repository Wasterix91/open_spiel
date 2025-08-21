# dqn_agent.py
import random, math
from collections import deque, namedtuple
from typing import Optional, Sequence, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ===================== Config ===================== #
class DQNConfig:
    def __init__(
        self,
        learning_rate=3e-4,
        batch_size=128,
        gamma=0.995,
        buffer_size=200_000,
        target_update_freq=5000,     # wenn tau == 0
        soft_target_tau=0.0,         # 0.005 für Soft-Update
        max_grad_norm=1.0,
        n_step=3,
        per_alpha=0.6,               # 0 => uniform
        per_beta_start=0.4,          # IS-Weights Start
        per_beta_frames=1_000_000,   # bis 1.0 annealen
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_frames=500_000,    # linearer Decay über Frames
        loss_huber_delta=1.0,        # None => MSE
        dueling=True,
        device="cpu",
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.target_update_freq = target_update_freq
        self.soft_target_tau = soft_target_tau
        self.max_grad_norm = max_grad_norm
        self.n_step = n_step
        self.per_alpha = per_alpha
        self.per_beta_start = per_beta_start
        self.per_beta_frames = per_beta_frames
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_frames = eps_decay_frames
        self.loss_huber_delta = loss_huber_delta
        self.dueling = dueling
        self.device = device


# ===================== Netzarchitektur ===================== #
class DuelingQNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.val = nn.Linear(128, 1)
        self.adv = nn.Linear(128, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        v = self.val(h)                   # [B,1]
        a = self.adv(h)                   # [B,A]
        return v + a - a.mean(dim=1, keepdim=True)


class QNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_size),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ===================== Prioritized Replay ===================== #
class PERBuffer:
    """Einfacher proportionaler PER-Buffer mit N-Step-Rückgaben."""
    def __init__(self, capacity: int, num_actions: int, n_step: int, gamma: float, alpha: float):
        self.capacity = capacity
        self.num_actions = num_actions
        self.n_step = max(1, int(n_step))
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.Exp = namedtuple("E", ["s","a","r","ns","done","next_mask"])
        # N-step FIFO (episode-lokal)
        self._nstep_fifo = deque(maxlen=self.n_step)

    @staticmethod
    def _mask_from_legal(legal_actions: Sequence[int], num_actions: int) -> np.ndarray:
        mask = np.zeros(num_actions, dtype=np.float32)
        mask[list(legal_actions)] = 1.0
        return mask

    def __len__(self):
        return len(self.buffer)

    def push_raw(self, s, a, r, ns, done, next_legal_actions: Optional[Sequence[int]]):
        next_mask = self._mask_from_legal(next_legal_actions, self.num_actions) if next_legal_actions is not None else None
        e = self.Exp(s, int(a), float(r), ns, bool(done), next_mask)
        if len(self.buffer) < self.capacity:
            self.buffer.append(e)
        else:
            self.buffer[self.pos] = e
        # neue Samples bekommen max-Priorität → sicher gesampelt
        max_p = self.priorities.max() if self.priorities.any() else 1.0
        self.priorities[self.pos] = max_p
        self.pos = (self.pos + 1) % self.capacity

    def add(self, state, action, reward, next_state, done, next_legal_actions=None):
        """N-Step-Akkumulation → schreibt eine (s,a, R^n, s', done', mask') Transition."""
        self._nstep_fifo.append((state, action, reward, next_state, done, next_legal_actions))
        # flush, sobald wir n haben oder die Episode endet
        if len(self._nstep_fifo) == self.n_step or done:
            R, s0, a0 = 0.0, None, None
            last_ns, last_done, last_mask = next_state, done, self._nstep_fifo[-1][5]
            for i, (s, a, r, ns, d, nleg) in enumerate(self._nstep_fifo):
                if i == 0:
                    s0, a0 = s, a
                R = r + (self.gamma ** i) * R
                last_ns, last_done, last_mask = ns, d, nleg
                if d: break
            self.push_raw(s0, a0, R, last_ns, float(done or last_done), last_mask)
        if done:
            self._nstep_fifo.clear()

    def sample(self, batch_size: int, beta: float):
        assert len(self.buffer) >= batch_size
        # proportional sampling
        scaled = self.priorities[:len(self.buffer)] ** self.alpha
        probs = scaled / scaled.sum()
        idxs = np.random.choice(len(self.buffer), size=batch_size, p=probs, replace=False)
        samples = [self.buffer[i] for i in idxs]

        # IS-Weights
        N = len(self.buffer)
        weights = (N * probs[idxs]) ** (-beta)
        weights /= weights.max()  # normalize to 1

        states = np.stack([e.s for e in samples]).astype(np.float32)
        actions = np.array([e.a for e in samples], dtype=np.int64)
        rewards = np.array([e.r for e in samples], dtype=np.float32)
        next_states = np.stack([e.ns for e in samples]).astype(np.float32)
        dones = np.array([e.done for e in samples], dtype=np.float32)
        next_masks = [e.next_mask for e in samples]  # kann None enthalten

        return idxs, states, actions, rewards, next_states, dones, next_masks, weights.astype(np.float32)

    def update_priorities(self, idxs: np.ndarray, new_priorities: np.ndarray, eps: float = 1e-6):
        self.priorities[idxs] = np.abs(new_priorities).astype(np.float32) + eps


# ===================== DQN Agent ===================== #
class DQNAgent:
    def __init__(self, state_size: int, num_actions: int, cfg: DQNConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.state_size = int(state_size)
        self.num_actions = int(num_actions)

        Net = DuelingQNetwork if cfg.dueling else QNetwork
        self.q = Net(self.state_size, self.num_actions).to(self.device)
        self.tgt = Net(self.state_size, self.num_actions).to(self.device)
        self.tgt.load_state_dict(self.q.state_dict())
        self.tgt.eval()

        self.optim = optim.Adam(self.q.parameters(), lr=cfg.learning_rate)
        if cfg.loss_huber_delta is None:
            self.criterion = nn.MSELoss(reduction="none")
        else:
            self.criterion = nn.SmoothL1Loss(beta=cfg.loss_huber_delta, reduction="none")

        self.buffer = PERBuffer(cfg.buffer_size, self.num_actions, cfg.n_step, cfg.gamma, cfg.per_alpha)

        self.frame = 0  # für epsilon/beta schedules

    # --------- Schedules --------- #
    def _epsilon(self):
        frac = max(0.0, 1.0 - self.frame / float(self.cfg.eps_decay_frames))
        return self.cfg.eps_end + (self.cfg.eps_start - self.cfg.eps_end) * frac

    def _beta(self):
        return min(1.0, self.cfg.per_beta_start + (1.0 - self.cfg.per_beta_start) * (self.frame / float(self.cfg.per_beta_frames)))

    # --------- Action Selection (Legal-Masking, ε-greedy) --------- #
    def select_action(self, state: np.ndarray, legal_actions: Sequence[int]) -> int:
        eps = self._epsilon()
        if random.random() < eps:
            # random legal
            return int(random.choice(list(legal_actions)))
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            qvals = self.q(s).squeeze(0).detach().cpu().numpy()
        masked = np.full_like(qvals, -1e9, dtype=np.float32)
        masked[list(legal_actions)] = qvals[list(legal_actions)]
        # tie-break random
        maxv = masked.max()
        candidates = np.flatnonzero(masked == maxv)
        return int(np.random.choice(candidates))

    # --------- Store Transition --------- #
    def store(self, state, action, reward, next_state, done, next_legal_actions=None):
        self.buffer.add(state, action, reward, next_state, done, next_legal_actions)
        self.frame += 1

    # --------- Train Step --------- #
    def train_step(self):
        if len(self.buffer) < self.cfg.batch_size:
            return {}

        idxs, s_np, a_np, r_np, ns_np, d_np, next_masks_list, w_np = self.buffer.sample(self.cfg.batch_size, beta=self._beta())

        s = torch.tensor(s_np, dtype=torch.float32, device=self.device)
        a = torch.tensor(a_np, dtype=torch.long, device=self.device).unsqueeze(1)
        r = torch.tensor(r_np, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.tensor(ns_np, dtype=torch.float32, device=self.device)
        d = torch.tensor(d_np, dtype=torch.float32, device=self.device).unsqueeze(1)
        w = torch.tensor(w_np, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a)
        q_sa = self.q(s).gather(1, a)

        with torch.no_grad():
            q_next_online = self.q(ns)
            q_next_target = self.tgt(ns)

            # Masking
            has_mask = any(m is not None for m in next_masks_list)
            if has_mask:
                masks_np = []
                for m in next_masks_list:
                    if m is None:
                        masks_np.append(np.ones(self.num_actions, dtype=np.float32))
                    else:
                        masks_np.append(m)
                next_masks = torch.tensor(np.stack(masks_np, axis=0), dtype=torch.float32, device=self.device)
                neg_large = torch.finfo(q_next_online.dtype).min / 2
                q_next_online = torch.where(next_masks > 0, q_next_online, neg_large)
                q_next_target = torch.where(next_masks > 0, q_next_target, neg_large)

            # Double-DQN target
            next_actions = q_next_online.argmax(dim=1, keepdim=True)
            q_next = q_next_target.gather(1, next_actions)

            # Keine legalen Actions? -> setze q_next=0 (falls je auftritt)
            if has_mask:
                no_legal = (next_masks.sum(dim=1, keepdim=True) == 0)
                q_next = torch.where(no_legal, torch.zeros_like(q_next), q_next)

            targets = r + (self.cfg.gamma ** self.buffer.n_step) * q_next * (1.0 - d)

        td = q_sa - targets
        loss_per = self.criterion(q_sa, targets)
        loss = (w * loss_per).mean()

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.max_grad_norm and self.cfg.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.max_grad_norm)
        self.optim.step()

        # Prioritäten updaten
        td_abs = td.detach().abs().squeeze(1).cpu().numpy()
        self.buffer.update_priorities(idxs, td_abs)

        # Target-Update
        tau = float(self.cfg.soft_target_tau or 0.0)
        if tau > 0.0:
            with torch.no_grad():
                for tp, sp in zip(self.tgt.parameters(), self.q.parameters()):
                    tp.data.mul_(1.0 - tau).add_(tau * sp.data)
        else:
            if (self.frame % int(self.cfg.target_update_freq)) == 0:
                self.tgt.load_state_dict(self.q.state_dict())
                self.tgt.eval()

        return {
            "loss": float(loss.detach().cpu().item()),
            "epsilon": float(self._epsilon()),
            "beta": float(self._beta()),
        }

    # --------- Save/Load --------- #
    def save(self, path_base: str):
        torch.save({
            "q": self.q.state_dict(),
            "tgt": self.tgt.state_dict(),
            "frame": self.frame,
        }, f"{path_base}_dqn.pt")

    def restore(self, path_base: str):
        ckpt = torch.load(f"{path_base}_dqn.pt", map_location=self.device)
        self.q.load_state_dict(ckpt["q"])
        self.tgt.load_state_dict(ckpt["tgt"])
        self.frame = ckpt.get("frame", 0)
        self.tgt.eval()
