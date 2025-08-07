import os
import numpy as np
import pyspiel
import torch
import collections
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

import ppo_agent as ppo
import dqn_agent as dqn


# === Spielkonfiguration ===
NUM_EPISODES = 4000
PLAYER_CONFIG = [
    {"type": "ppo", "version": "12"},
    {"type": "ppo", "version": "12"},
    {"type": "dqn", "version": "07"},
    {"type": "random2"},
]

base_dir = os.path.dirname(os.path.abspath(__file__))
game = pyspiel.load_game("president", {
    "deck_size": "32",
    "shuffle_cards": True,
    "single_card_mode": False,
    "num_players": 4
})

params = game.get_parameters()
RANKS = ["7", "8", "9", "10", "J", "Q", "K", "A"]
RANK_TO_NUM = {rank: i for i, rank in enumerate(RANKS)}

def parse_combo_size(text):
    if "Single" in text: return 1
    if "Pair" in text: return 2
    if "Triple" in text: return 3
    if "Quad" in text: return 4
    return 1

def parse_rank(text):
    try:
        return RANK_TO_NUM[text.split()[-1]]
    except KeyError:
        return -1

def decode_actions(state):
    player = state.current_player()
    actions = state.legal_actions()
    return [(a, state.action_to_string(player, a)) for a in actions if a != 0]

def max_combo_strategy(state):
    decoded = decode_actions(state)
    return max(decoded, key=lambda x: parse_combo_size(x[1]))[0] if decoded else 0

def aggressive_strategy(state):
    decoded = decode_actions(state)
    if not decoded:
        return 0
    return max(decoded, key=lambda x: parse_rank(x[1]))[0]

def single_only_strategy(state):
    decoded = decode_actions(state)
    singles = [x for x in decoded if "Single" in x[1]]
    return min(singles, key=lambda x: x[0])[0] if singles else 0

# Komplett Random, einschließlich Pass
def random_action_strategy(state): 
    return np.random.choice(state.legal_actions())

# Spielt nur Pass wenn es erlaubt ist
def random2_action_strategy(state):
    legal = state.legal_actions()
    if len(legal) > 1 and 0 in legal:
        legal = [a for a in legal if a != 0]
    return np.random.choice(legal)


def smart_strategy(state):
    decoded = decode_actions(state)
    if not decoded: return 0
    groups = {1: [], 2: [], 3: [], 4: []}
    for a, s in decoded:
        size = parse_combo_size(s)
        groups[size].append((a, s))
    for size in [4, 3, 2, 1]:
        if groups[size]:
            return min(groups[size], key=lambda x: parse_rank(x[1]))[0]
    return 0

strategy_map = {
    "random": random_action_strategy,
    "random2": random2_action_strategy,
    "max_combo": max_combo_strategy,
    "single_only": single_only_strategy,
    "smart": smart_strategy,
    "aggressive": aggressive_strategy
}

def load_agents(player_config, base_dir, game):
    agents = []
    for pid, cfg in enumerate(player_config):
        kind = cfg["type"]

        if kind == "ppo":
            version = cfg["version"]
            model_dir = os.path.join(base_dir, f"models/ppo_model_{version}/train")
            model_path = os.path.join(model_dir, f"ppo_model_{version}_agent_p{pid}")
            agent = ppo.PPOAgent(
                info_state_size=game.information_state_tensor_shape()[0],
                num_actions=game.num_distinct_actions()
            )
            agent.restore(model_path)
            agents.append(agent)

        elif kind == "dqn":
            if dqn is None:
                raise RuntimeError("DQN-Modul nicht geladen.")
            version = cfg["version"]
            model_dir = os.path.join(base_dir, f"models/dqn_model_{version}/train")
            model_path = os.path.join(model_dir, f"dqn_model_{version}_agent_p{pid}")
            agent = dqn.DQNAgent(
                state_size=game.observation_tensor_shape()[0],
                num_actions=game.num_distinct_actions()
            )
            agent.restore(model_path)
            agents.append(agent)

        elif kind in strategy_map:
            agents.append(strategy_map[kind])  # z.B. random2, smart etc.

        else:
            raise ValueError(f"Unbekannter Agententyp: {kind}")

    return agents


def choose_policy_action(agent, state, player):
    info_state = state.information_state_tensor(player)
    legal_actions = state.legal_actions(player)
    info_tensor = torch.tensor(info_state, dtype=torch.float32).to(agent.device)
    logits = agent._policy(info_tensor).detach().cpu().numpy()
    masked_logits = np.zeros_like(logits)
    masked_logits[legal_actions] = logits[legal_actions]
    if masked_logits.sum() == 0:
        probs = np.zeros_like(logits)
        probs[legal_actions] = 1.0 / len(legal_actions)
    else:
        probs = masked_logits / masked_logits.sum()
    return collections.namedtuple("AgentOutput", ["action"])(action=np.argmax(probs))

# === Evaluation starten ===
agents = load_agents(PLAYER_CONFIG, base_dir, game)
returns_total = np.zeros(4)
start_counts = defaultdict(int)
win_counts = defaultdict(int)
points_total = np.zeros(4)

for episode in range(1, NUM_EPISODES + 1):
    state = game.new_initial_state()
    start_player = state.current_player()
    start_counts[start_player] += 1

    while not state.is_terminal():
        player = state.current_player()
        agent_or_strategy = agents[player]

        if isinstance(agent_or_strategy, ppo.PPOAgent):
            action = choose_policy_action(agent_or_strategy, state, player).action
        elif dqn and isinstance(agent_or_strategy, dqn.DQNAgent):
            obs = state.observation_tensor(player)
            legal = state.legal_actions(player)
            action = agent_or_strategy.select_action(obs, legal)
        else:
            action = smart_strategy(state)
        state.apply_action(action)

    final_returns = state.returns()
    for pid, ret in enumerate(final_returns):
        returns_total[pid] += ret
        points_total[pid] += ret
    win_counts[np.argmax(final_returns)] += 1

    if episode % 250 == 0:
        print(f"✅ Episode {episode} abgeschlossen")

# === Ergebnisse ausgeben ===
print("\n=== 📊 Ergebnisse ===")
for pid in range(4):
    avg_return = returns_total[pid] / NUM_EPISODES
    winrate = 100 * win_counts[pid] / NUM_EPISODES
    agent_type = PLAYER_CONFIG[pid]["type"]
    version = PLAYER_CONFIG[pid].get("version")  # Gibt None zurück, wenn nicht vorhanden

    if version:
        label = f"{agent_type} v{version}"
    else:
        label = agent_type

    print(f"Player {pid} ({label}): Ø Return = {avg_return:.2f}, "
          f"Siege = {win_counts[pid]}, Winrate = {winrate:.2f}%, Starts = {start_counts[pid]}")

