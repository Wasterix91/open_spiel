# President/agents/ppo_agent.py
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ================= PPO Config ================= #
PPOConfig = collections.namedtuple(
    "PPOConfig",
    [
        "learning_rate",
        "num_epochs",
        "batch_size",
        "entropy_cost",
        "gamma",
        "gae_lambda",
        "clip_eps",
        "value_coef",
        "max_grad_norm",
    ],
)

DEFAULT_CONFIG = PPOConfig(
    learning_rate=3e-4,
    num_epochs=4,
    batch_size=256,
    entropy_cost=0.01,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    value_coef=0.5,
    max_grad_norm=0.5,
)

# ================= Networks ================= #
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),  # Logits (kein Softmax hier)
        )

    def forward(self, x):
        return self.net(x)

class ValueNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# =============== Utils =============== #
def masked_softmax(logits, mask):
    # mask: 1.0 für legal, 0.0 für illegal
    # Stelle sicher, dass logits und mask auf demselben Device liegen
    mask = mask.to(logits.device)
    masked_logits = logits.clone()
    masked_logits = torch.where(
        mask > 0,
        masked_logits,
        torch.tensor(-1e9, device=logits.device, dtype=logits.dtype),
    )
    return torch.softmax(masked_logits, dim=-1)

# =============== Replay Buffer =============== #
class ReplayBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.legal_masks = []
        self.player_ids = []

    def add(self, state, action, reward, done, log_prob, value, legal_mask, player_id=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.legal_masks.append(legal_mask)
        self.player_ids.append(player_id)

    def finalize_last_reward(self, delta):
        if self.rewards:
            self.rewards[-1] += delta

    def clear(self):
        self.__init__()

# =============== PPO Agent =============== #
class PPOAgent:
    def __init__(self, info_state_size, num_actions, seat_id_dim=0, config=None, device="cpu"):
        self.device = torch.device(device)
        self._base_state_dim = info_state_size
        self._seat_id_dim = int(seat_id_dim)
        self._input_dim = self._base_state_dim + self._seat_id_dim
        self._num_actions = num_actions
        self._config = config or DEFAULT_CONFIG

        self._policy = PolicyNetwork(self._input_dim, num_actions).to(self.device)
        self._value = ValueNetwork(self._input_dim).to(self.device)
        self._optimizer = optim.Adam(
            list(self._policy.parameters()) + list(self._value.parameters()),
            lr=self._config.learning_rate,
        )

        self._buffer = ReplayBuffer()

    # ---- Hilfsfunktion, um optional Seat-One-Hot anzuhängen ----
    def _make_input(self, info_state, seat_one_hot=None):
        x = np.array(info_state, dtype=np.float32)
        if self._seat_id_dim > 0:
            assert seat_one_hot is not None and len(seat_one_hot) == self._seat_id_dim
            x = np.concatenate([x, np.array(seat_one_hot, dtype=np.float32)], axis=0)
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def step(self, info_state, legal_actions, seat_one_hot=None, player_id=None):
        # legal mask (1=legal, 0=illegal)
        legal_mask = np.zeros(self._num_actions, dtype=np.float32)
        legal_mask[legal_actions] = 1.0
        legal_mask_t = torch.tensor(legal_mask, dtype=torch.float32, device=self.device)

        x = self._make_input(info_state, seat_one_hot)

        with torch.no_grad():
            logits = self._policy(x)
            probs = masked_softmax(logits, legal_mask_t)
            dist = torch.distributions.Categorical(probs=probs)  # Fix: richtiger Konstruktor
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self._value(x)

        # Store transition with placeholder reward (0); will be updated via post_step
        self._buffer.add(
            state=x.detach().cpu().numpy(),
            action=int(action.item()),
            reward=0.0,
            done=False,
            log_prob=float(log_prob.detach().cpu().item()),
            value=float(value.detach().cpu().item()),
            legal_mask=legal_mask,
            player_id=player_id,
        )

        return int(action.item())

    def post_step(self, reward, done=False):
        # Update last transition's reward (+ possibly done)
        self._buffer.rewards[-1] = float(reward)
        self._buffer.dones[-1] = bool(done)

    def _compute_gae(self, rewards, values, dones, gamma, lam):
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        lastgaelam = 0.0
        # Terminal bootstrapping: value_{T} = 0
        for t in reversed(range(T)):
            next_nonterminal = 0.0 if (t == T - 1 or dones[t]) else 1.0
            next_value = 0.0 if (t == T - 1 or dones[t]) else values[t + 1]
            delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
            lastgaelam = delta + gamma * lam * next_nonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values
        return advantages, returns

    def _compute_gae_segmented(self, rewards, values, dones, player_ids, gamma, lam):
        """
        GAE, aber Bootstrapping nur innerhalb desselben player_id.
        Das verhindert Credit-Assignment über Spielerwechsel hinweg (Shared Policy).
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(T)):
            same_actor_next = (t < T - 1) and (player_ids[t] == player_ids[t + 1]) and (not dones[t])
            next_nonterminal = 1.0 if same_actor_next else 0.0
            next_value = values[t + 1] if same_actor_next else 0.0
            delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
            lastgaelam = delta + gamma * lam * next_nonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values
        return advantages, returns

    def train(self):
        if len(self._buffer.states) == 0:
            return {}

        cfg = self._config
        states = torch.tensor(np.array(self._buffer.states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(self._buffer.actions, dtype=torch.long, device=self.device)
        rewards = np.array(self._buffer.rewards, dtype=np.float32)
        dones = np.array(self._buffer.dones, dtype=np.bool_)
        old_log_probs = torch.tensor(self._buffer.log_probs, dtype=torch.float32, device=self.device)
        values_np = np.array(self._buffer.values, dtype=np.float32)
        legal_masks = torch.tensor(np.array(self._buffer.legal_masks), dtype=torch.float32, device=self.device)

        # --- GAE (segmentiert, falls player_ids vorhanden) ---
        pids = self._buffer.player_ids
        use_segmented = (len(pids) == len(rewards)) and all(pid is not None for pid in pids)
        if use_segmented:
            player_ids = np.array(pids, dtype=np.int64)
            adv_np, ret_np = self._compute_gae_segmented(
                rewards, values_np, dones, player_ids, cfg.gamma, cfg.gae_lambda
            )
        else:
            adv_np, ret_np = self._compute_gae(rewards, values_np, dones, cfg.gamma, cfg.gae_lambda)

        advantages = torch.tensor(adv_np, dtype=torch.float32, device=self.device)
        returns = torch.tensor(ret_np, dtype=torch.float32, device=self.device)

        # --- Advantage-Norm ---
        adv_mean = float(advantages.mean().detach().cpu().item())
        adv_std  = float(advantages.std(unbiased=False).detach().cpu().item())
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        N = states.shape[0]
        idx = np.arange(N)

        # Laufende Sammelgrößen für Metriken
        policy_losses, value_losses, entropies = [], [], []
        clip_fracs, approx_kls = [], []

        for _ in range(cfg.num_epochs):
            np.random.shuffle(idx)
            for start in range(0, N, cfg.batch_size):
                end = min(start + cfg.batch_size, N)
                mb = idx[start:end]

                s_mb = states[mb]
                a_mb = actions[mb]
                adv_mb = advantages[mb]
                ret_mb = returns[mb]
                old_logp_mb = old_log_probs[mb]
                mask_mb = legal_masks[mb]

                logits = self._policy(s_mb)
                probs = masked_softmax(logits, mask_mb)
                dist = torch.distributions.Categorical(probs=probs)
                new_logp = dist.log_prob(a_mb)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - old_logp_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_mb
                policy_loss = -torch.mean(torch.min(surr1, surr2))

                values = self._value(s_mb)
                value_loss = torch.mean((ret_mb - values) ** 2)

                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_cost * entropy

                self._optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self._policy.parameters()) + list(self._value.parameters()),
                    cfg.max_grad_norm
                )
                self._optimizer.step()

                # --- Metriken sammeln ---
                with torch.no_grad():
                    # approx-KL: E[logp_old - logp_new]
                    approx_kl = (old_logp_mb - new_logp).mean().abs().item()
                    clipped = (torch.abs(ratio - 1.0) > cfg.clip_eps).float().mean().item()

                policy_losses.append(float(policy_loss.detach().cpu().item()))
                value_losses.append(float(value_loss.detach().cpu().item()))
                entropies.append(float(entropy.detach().cpu().item()))
                approx_kls.append(float(approx_kl))
                clip_fracs.append(float(clipped))

        # Buffer leeren
        self._buffer.clear()

        # --- Zusammenstellen & zurückgeben ---
        metrics = {
            # --- Rewards & Returns ---
            "reward_mean": float(np.mean(rewards)) if len(rewards) else 0.0,
            "reward_std":  float(np.std(rewards)) if len(rewards) else 0.0,
            "return_mean": float(np.mean(ret_np)) if len(ret_np) else 0.0,

            # --- Advantages (vor Normalisierung) ---
            "adv_mean_raw": adv_mean,
            "adv_std_raw":  adv_std,

            # --- PPO Verluste ---
            "policy_loss":  float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss":   float(np.mean(value_losses)) if value_losses else 0.0,

            # --- Exploration / Regularisierung ---
            "entropy":      float(np.mean(entropies)) if entropies else 0.0,

            # --- PPO Stabilitätsmetriken ---
            "approx_kl":    float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clip_frac":    float(np.mean(clip_fracs)) if clip_fracs else 0.0,
        }

        return metrics

    def save(self, path):
        torch.save(self._policy.state_dict(), path + "_policy.pt")
        torch.save(self._value.state_dict(), path + "_value.pt")

    def restore(self, path):
        self._policy.load_state_dict(torch.load(path + "_policy.pt", map_location=self.device))
        self._value.load_state_dict(torch.load(path + "_value.pt", map_location=self.device))
