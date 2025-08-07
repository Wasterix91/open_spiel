import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = []

    def add(self, x):
        if len(self.data) >= self.capacity:
            self.data.pop(0)
        self.data.append(x)

    def sample(self, batch_size):
        return random.sample(self.data, min(batch_size, len(self.data)))

    def __len__(self):
        return len(self.data)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, is_policy=False):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_layers:
            layers += [nn.Linear(last_dim, h), nn.ReLU()]
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        if is_policy:
            layers.append(nn.Softmax(dim=-1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DeepCFRSolver:
    def __init__(self, game, policy_network_layers, advantage_network_layers,
                 num_iterations, num_traversals, learning_rate,
                 batch_size, memory_capacity, policy_network_train_steps,
                 advantage_network_train_steps, device="cpu"):

        self.game = game
        self.num_players = game.num_players()
        self.info_state_size = game.information_state_tensor_size()
        self.num_actions = game.num_distinct_actions()
        self.device = device

        self.policy_net = MLP(self.info_state_size, policy_network_layers,
                              self.num_actions, is_policy=True).to(device)
        self.advantage_net = MLP(self.info_state_size, advantage_network_layers,
                                 self.num_actions).to(device)

        self.policy_opt = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.advantage_opt = optim.Adam(self.advantage_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.num_iterations = num_iterations
        self.num_traversals = num_traversals
        self.batch_size = batch_size
        self.memory_capacity = memory_capacity
        self.policy_train_steps = policy_network_train_steps
        self.advantage_train_steps = advantage_network_train_steps

        self.policy_memory = ReplayBuffer(memory_capacity)
        self.adv_memory = ReplayBuffer(memory_capacity)
        self.exploitability_progress = []

    def traverse(self, state, player, pi_reach, opponent_reach, depth=0, max_depth=80):
        if depth % 25 == 0:
            print(f"[Depth {depth}] Player {state.current_player()} | Terminal: {state.is_terminal()}")

        if state.is_terminal() or depth >= max_depth:
            if state.is_terminal():
                return state.returns()[player]
            else:
                return 0.0  # Tiefe überschritten, Behandlung als Null-Reward

        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            action, _ = random.choice(outcomes)
            next_state = state.clone()
            next_state.apply_action(action)
            return self.traverse(next_state, player, pi_reach, opponent_reach, depth + 1, max_depth)

        current_player = state.current_player()
        legal = state.legal_actions()

        # Strategie bestimmen (Softmax über Advantage Network oder Policy-Net)
        info_state = np.array(state.information_state_tensor(current_player), dtype=np.float32)
        info_tensor = torch.tensor(info_state).to(self.device)

        with torch.no_grad():
            strategy_full = self.policy_net(info_tensor).cpu().numpy()
        
        # Maskierung auf legale Aktionen
        strategy = np.zeros_like(strategy_full)
        strategy[legal] = strategy_full[legal]
        total = strategy.sum()
        if total <= 0 or np.any(np.isnan(strategy)):
            # Gleichverteilung fallback
            strategy[legal] = 1.0 / len(legal)
        else:
            strategy /= total

        # Spieler ist aktuell zu trainierender Player
        if current_player == player:
            action = np.random.choice(legal, p=strategy[legal])

            # Counterfactual values für alle legalen Aktionen
            values = []
            for a in legal:
                next_state = state.clone()
                next_state.apply_action(a)
                value = self.traverse(next_state, player, pi_reach * strategy[a], opponent_reach, depth + 1, max_depth)
                values.append(value)

            values = np.array(values)
            baseline = values.mean()
            advantages = values - baseline

            # Speichern in Replay Buffer
            one_hot = np.zeros(self.num_actions)
            for i, a in enumerate(legal):
                one_hot[a] = advantages[i]
            self.adv_memory.add((info_state, one_hot))

            # Weiter mit gesampelter Aktion
            next_state = state.clone()
            next_state.apply_action(action)
            return self.traverse(next_state, player, pi_reach * strategy[action], opponent_reach, depth + 1, max_depth)

        else:
            # Gegner: Sampling aus deren Policy (egal ob trainiert oder nicht)
            action = np.random.choice(legal, p=strategy[legal])
            next_state = state.clone()
            next_state.apply_action(action)
            return self.traverse(next_state, player, pi_reach, opponent_reach * strategy[action], depth + 1, max_depth)




    def solve(self):
        for it in range(1, self.num_iterations + 1):
            for p in range(self.num_players):
                for _ in range(self.num_traversals):
                    state = self.game.new_initial_state()
                    self.traverse(state, p, 1.0, 1.0)

            self.train_advantage()
            self.train_policy()
            self.exploitability_progress.append(self._estimate_exploitability())

        return self.exploitability_progress

    def train_advantage(self):
        for _ in range(self.advantage_train_steps):
            batch = self.adv_memory.sample(self.batch_size)
            if not batch:
                continue
            obs, targets = zip(*batch)
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
            targets = torch.tensor(targets, dtype=torch.float32).to(self.device)

            pred = self.advantage_net(obs)
            loss = self.loss_fn(pred, targets)
            self.advantage_opt.zero_grad()
            loss.backward()
            self.advantage_opt.step()

    def train_policy(self):
        for _ in range(self.policy_train_steps):
            batch = self.adv_memory.sample(self.batch_size)
            if not batch:
                continue
            obs, targets = zip(*batch)
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                adv = self.advantage_net(obs)
            probs = torch.nn.functional.softmax(adv, dim=-1)
            pred = self.policy_net(obs)
            loss = self.loss_fn(pred, probs)
            self.policy_opt.zero_grad()
            loss.backward()
            self.policy_opt.step()

    def exploitabilities(self):
        return self.exploitability_progress

    def _estimate_exploitability(self):
        # Dummy estimator
        return random.uniform(0.1, 1.5)
