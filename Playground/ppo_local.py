import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pyspiel

# === PPO Config ===
PPOConfig = collections.namedtuple(
    "PPOConfig",
    ["learning_rate", "num_epochs", "batch_size", "entropy_cost"]
)

DEFAULT_CONFIG = PPOConfig(
    learning_rate=0.001,
    num_epochs=5,
    batch_size=32,
    entropy_cost=0.01,
)

# === PPO Networks ===
class PolicyNetwork(nn.Module):
    """Simple feed-forward policy network."""
    def __init__(self, input_size, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

class ValueNetwork(nn.Module):
    """Value function approximator."""
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# === PPO Agent ===
class PPOAgent:
    """PPO Agent: handles policy/value networks and training."""
    def __init__(self, info_state_size, num_actions, config=None, device="cpu"):
        self.device = torch.device(device)
        self._info_state_size = info_state_size
        self._num_actions = num_actions
        self._config = config or DEFAULT_CONFIG

        self._policy = PolicyNetwork(info_state_size, num_actions).to(self.device)
        self._value = ValueNetwork(info_state_size).to(self.device)
        self._optimizer = optim.Adam(
            list(self._policy.parameters()) + list(self._value.parameters()),
            lr=self._config.learning_rate
        )
        self._buffer = []

    def step(self, time_step, legal_actions):
        """Takes a step: chooses action or learns on terminal."""
        if time_step.last():
            # On terminal: overwrite rewards, train, clear buffer.
            final_reward = time_step.rewards[0]
            self._buffer = [(s, a, final_reward) for (s, a, _) in self._buffer]
            self._train()
            self._buffer = []
            return None

        if not legal_actions:
            # No legal actions → skip.
            return None

        # Get info state & compute logits.
        info_state = np.array(time_step.observations["info_state"][0], dtype=np.float32)
        info_state_tensor = torch.tensor(info_state, dtype=torch.float32).to(self.device)

        logits = self._policy(info_state_tensor).detach().cpu().numpy()

        # Mask logits to respect legal actions only.
        masked_logits = np.zeros_like(logits)
        masked_logits[legal_actions] = logits[legal_actions]

        if masked_logits.sum() == 0:
            # Fallback: uniform random legal action.
            masked_probs = np.zeros_like(logits)
            masked_probs[legal_actions] = 1.0 / len(legal_actions)
        else:
            masked_probs = masked_logits / masked_logits.sum()

        action = np.random.choice(len(masked_probs), p=masked_probs)
        self._buffer.append((info_state, action, 0.0))

        return collections.namedtuple("AgentOutput", ["action"])(action=action)

    def _train(self):
        """One training pass through buffer."""
        if not self._buffer:
            return
        states, actions, rewards = zip(*self._buffer)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        returns = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        for _ in range(self._config.num_epochs):
            logits = self._policy(states)
            values = self._value(states)
            action_probs = logits.gather(1, actions.unsqueeze(1)).squeeze(1)

            advantage = returns - values
            policy_loss = -torch.mean(torch.log(action_probs + 1e-10) * advantage.detach())
            value_loss = torch.mean(advantage ** 2)
            entropy = -torch.mean(torch.sum(logits * torch.log(logits + 1e-10), dim=1))
            loss = policy_loss + 0.5 * value_loss - self._config.entropy_cost * entropy

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

    def save(self, path):
        """Save policy & value network weights."""
        torch.save(self._policy.state_dict(), path + "_policy.pt")
        torch.save(self._value.state_dict(), path + "_value.pt")

    def restore(self, path):
        """Load policy & value network weights."""
        self._policy.load_state_dict(torch.load(path + "_policy.pt"))
        self._value.load_state_dict(torch.load(path + "_value.pt"))
