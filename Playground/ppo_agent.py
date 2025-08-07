import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pyspiel


# === PPO Config ===
PPOConfig = collections.namedtuple(
    "PPOConfig",
    ["learning_rate", "num_epochs", "batch_size", "entropy_cost", "gamma"]
)

DEFAULT_CONFIG = PPOConfig(
    learning_rate=0.001,
    num_epochs=1,
    batch_size=32,
    entropy_cost=0.01,
    gamma=0.99,
)

# === Policy Network ===
class PolicyNetwork(nn.Module):
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

# === Value Network ===
class ValueNetwork(nn.Module):
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

# === Replay Buffer ===
class ReplayBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def add(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def compute_returns(self, gamma=0.99):
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + gamma * G
            returns.insert(0, G)
        return np.array(returns, dtype=np.float32)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []

# === PPO Agent ===
class PPOAgent:
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

        self._buffer = ReplayBuffer()

    def step(self, time_step, legal_actions):
        if not legal_actions:
            return None

        info_state = np.array(time_step.observations["info_state"][0], dtype=np.float32)
        info_state_tensor = torch.tensor(info_state, dtype=torch.float32).to(self.device)
        logits = self._policy(info_state_tensor).detach().cpu().numpy()

        masked_logits = np.zeros_like(logits)
        masked_logits[legal_actions] = logits[legal_actions]

        if masked_logits.sum() == 0:
            probs = np.zeros_like(logits)
            probs[legal_actions] = 1.0 / len(legal_actions)
        else:
            probs = masked_logits / masked_logits.sum()

        action = np.random.choice(len(probs), p=probs)
        self._buffer.add(info_state, action, 0.0)  # reward is updated at the end
        return collections.namedtuple("AgentOutput", ["action"])(action=action)
    
    def post_step(self, reward):
        self._buffer.rewards[-1] = reward

    def train(self):
        if not self._buffer.states:
            return

        returns = self._buffer.compute_returns(self._config.gamma)
        states = torch.tensor(np.array(self._buffer.states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(self._buffer.actions, dtype=torch.long).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        for epoch in range(self._config.num_epochs):
            logits = self._policy(states)
            dist = torch.distributions.Categorical(probs=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            values = self._value(states)
            advantages = returns - values.detach()

            policy_loss = -torch.mean(log_probs * advantages)
            value_loss = torch.mean((returns - values) ** 2)

            loss = policy_loss + 0.5 * value_loss - self._config.entropy_cost * entropy

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            #print(f"[Epoch {epoch}] Policy Loss: {policy_loss.item():.4f}, "
            #     f"Value Loss: {value_loss.item():.4f}, Entropy: {entropy.item():.4f}")

        self._buffer.clear()

    def save(self, path):
        torch.save(self._policy.state_dict(), path + "_policy.pt")
        torch.save(self._value.state_dict(), path + "_value.pt")

    def restore(self, path):
        self._policy.load_state_dict(torch.load(path + "_policy.pt"))
        self._value.load_state_dict(torch.load(path + "_value.pt"))
