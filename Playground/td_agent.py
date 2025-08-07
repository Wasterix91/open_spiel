import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, obs_size, num_actions):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        return self.net(x)

class TDAgent:
    def __init__(self, obs_size, num_actions, lr=1e-3, gamma=0.99, epsilon=0.1):
        self.obs_size = obs_size
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_net = QNetwork(obs_size, num_actions)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    def step(self, time_step, legal_actions):
        if time_step.last():
            return None
        player = time_step.current_player()
        state = time_step.observations["info_state"][player]

        action = self.select_action(state, legal_actions)
        return AgentOutput(action)

    def select_action(self, state_tensor, legal_actions):
        if np.random.rand() < self.epsilon:
            return np.random.choice(legal_actions)
        else:
            with torch.no_grad():
                q_values = self.q_net(torch.tensor(state_tensor).float().unsqueeze(0)).squeeze()
            legal_qs = q_values[legal_actions]
            return legal_actions[torch.argmax(legal_qs).item()]

    def train(self, transition):
        s, a, r, s_next, done, legal_next = transition

        s = torch.tensor(s).float()
        s_next = torch.tensor(s_next).float()
        a = torch.tensor(a)
        r = torch.tensor(r).float()

        q_values = self.q_net(s)
        target = r
        if not done and legal_next:
            with torch.no_grad():
                q_next = self.q_net(s_next)
                q_next_legal = q_next[legal_next]
                target += self.gamma * torch.max(q_next_legal)

        loss = (q_values[a] - target).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def observe_transition(self, time_step, action, next_time_step):
        s = time_step.observations["info_state"][time_step.current_player()]
        s_next = next_time_step.observations["info_state"][time_step.current_player()]
        r = next_time_step.rewards[time_step.current_player()]
        done = next_time_step.last()
        legal_next = next_time_step.observations["legal_actions"][time_step.current_player()]
        self.train((s, action, r, s_next, done, legal_next))

    def end_episode(self):
        pass

    def predict(self, obs_tensor):
        with torch.no_grad():
            return self.q_net(obs_tensor.unsqueeze(0)).squeeze(0)

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.q_net.eval()

class AgentOutput:
    def __init__(self, action):
        self.action = action
