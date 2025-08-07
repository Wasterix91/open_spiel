import os
import re
import numpy as np
import pyspiel
import torch
import collections
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

# === Agenten laden ===
import ppo_agent as ppo
try:
    import dqn_agent as dqn
except ImportError:
    dqn = None

# === Spielkonfiguration ===
MODEL_TYPE = "ppo"  # Hauptmodelltyp zur Auswertung: "ppo" oder "dqn"
VERSION_NUM = "11"
NUM_EPISODES = 10_000
PLAYER_TYPES = ["ppo", "random2", "ppo", "random2"]

game = pyspiel.load_game(
    "president",
    {
        "deck_size": "64",
        "shuffle_cards": True,
        "single_card_mode": False,
        "num_players": 4
    }
)

base_dir = os.path.dirname(os.path.abspath(__file__))
GENERATE_PLOTS = True  # Kein Plot in dieser Version

# === Aktionsparser / Heuristikstrategien ===
RANKS = ["7", "8", "9", "10", "J", "Q", "K", "A"]
RANK_TO_NUM = {rank: i for i, rank in enumerate(RANKS)}

def parse_combo_size(text):
    if "Single" in text: return 1
    if "Pair" in text: return 2
    if "Triple" in text: return 3
    if "Quad" in text: return 4
    return 1

def parse_rank(text):
    try: return RANK_TO_NUM[text.split()[-1]]
    except: return -1

def decode_actions(state):
    player = state.current_player()
    actions = state.legal_actions()
    return [(a, state.action_to_string(player, a)) for a in actions if a != 0]

def max_combo_strategy(state):
    decoded = decode_actions(state)
    return max(decoded, key=lambda x: parse_combo_size(x[1]))[0] if decoded else 0

def aggressive_strategy(state):
    decoded = decode_actions(state)
    return max(decoded, key=lambda x: parse_rank(x[1]))[0] if decoded else 0

def single_only_strategy(state):
    decoded = decode_actions(state)
    singles = [x for x in decoded if "Single" in x[1]]
    return min(singles, key=lambda x: x[0])[0] if singles else 0

def random_action_strategy(state):
    return np.random.choice(state.legal_actions())

def random2_action_strategy(state):
    legal = state.legal_actions()
    if len(legal) > 1 and 0 in legal:
        legal = [a for a in legal if a != 0]
    return np.random.choice(legal)

def smart_strategy(state):
    decoded = decode_actions(state)
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

# === Agenten initialisieren ===
agents = []
for pid, ptype in enumerate(PLAYER_TYPES):
    if ptype == "ppo":
        model_path = os.path.join(base_dir, f"models/ppo_model_{VERSION_NUM}/train/ppo_model_{VERSION_NUM}_agent_p{pid}")
        agent = ppo.PPOAgent(
            info_state_size=game.information_state_tensor_shape()[0],
            num_actions=game.num_distinct_actions()
        )
        agent.restore(model_path)
        agents.append(agent)
    elif ptype == "dqn":
        if dqn is None:
            raise ImportError("DQN-Agent konnte nicht geladen werden.")
        model_path = os.path.join(base_dir, f"models/dqn_model_{VERSION_NUM}/train/dqn_model_{VERSION_NUM}_agent_p{pid}")
        agent = dqn.DQNAgent(
            state_size=game.observation_tensor_shape()[0],
            num_actions=game.num_distinct_actions()
        )
        agent.restore(model_path)
        agents.append(agent)
    elif ptype in strategy_map:
        agents.append(strategy_map[ptype])
    else:
        raise ValueError(f"Unbekannter Spielertyp: {ptype}")

# === PPO Aktionswahlfunktion ===
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
            action = agent_or_strategy(state)

        state.apply_action(action)

    final_returns = state.returns()
    for pid, ret in enumerate(final_returns):
        returns_total[pid] += ret
        points_total[pid] += ret
    win_counts[np.argmax(final_returns)] += 1

    if episode % 250 == 0:
        print(f"✅ Episode {episode} abgeschlossen")

""" # === Ergebnisse ausgeben ===
print("\n=== 📊 Ergebnisse ===")
for pid in range(4):
    print(f"Player {pid} ({PLAYER_TYPES[pid]}): "
          f"Ø Return = {returns_total[pid] / NUM_EPISODES:.2f}, "
          f"Siege = {win_counts[pid]}, "
          f"Winrate = {100 * win_counts[pid] / NUM_EPISODES:.2f}%, "
          f"Starts = {start_counts[pid]}")

# === CSV speichern ===
eval_dir = os.path.join(base_dir, f"models/{MODEL_TYPE}_model_{VERSION_NUM}/eval")
os.makedirs(eval_dir, exist_ok=True)
eval_num = len([f for f in os.listdir(eval_dir) if f.endswith(".csv")]) + 1
csv_path = os.path.join(eval_dir, f"evaluation_summary_v{VERSION_NUM}_{eval_num:02d}.csv")

summary_rows = []
for pid in range(4):
    avg_return = returns_total[pid] / NUM_EPISODES
    total_points = points_total[pid]
    wins = win_counts[pid]
    win_rate = 100 * wins / NUM_EPISODES
    starts = start_counts[pid]
    starts_percent = 100 * starts / NUM_EPISODES
    strategy = PLAYER_TYPES[pid]

    row = {
        "version": VERSION_NUM,
        "eval_id": eval_num,
        "player": pid,
        "strategy": strategy,
        "win_rate_percent": round(win_rate, 2),
        "avg_return": round(avg_return, 2),
        "total_points": round(total_points, 1),
        "num_wins": wins,
        "num_starts": starts,
        "start_rate_percent": round(starts_percent, 2)
    }
    summary_rows.append(row)

pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
print(f"\n📄 Evaluation gespeichert unter: {csv_path}") """


# === Ergebnisse sammeln ===
summary_rows = []
for pid in range(4):
    avg_return = returns_total[pid] / NUM_EPISODES
    total_points = points_total[pid]
    wins = win_counts[pid]
    win_rate = 100 * wins / NUM_EPISODES
    starts = start_counts[pid]
    starts_percent = 100 * starts / NUM_EPISODES
    strategy = PLAYER_TYPES[pid]

    row = {
        "Player": f"Player {pid} ({strategy})",
        "Ø Return": round(avg_return, 2),
        "Siege": wins,
        "Winrate": f"{round(win_rate, 2)}%",
        "Starts": starts
    }
    summary_rows.append(row)

# === Konsolenausgabe als formatierte Tabelle ===
print("\n=== 📊 Ergebnisse (tabellarisch) ===")
df_summary = pd.DataFrame(summary_rows)
print(df_summary.to_string(index=False))

# === CSV speichern ===
eval_dir = os.path.join(base_dir, f"models/{MODEL_TYPE}_model_{VERSION_NUM}/eval")
os.makedirs(eval_dir, exist_ok=True)
eval_num = len([f for f in os.listdir(eval_dir) if f.endswith(".csv")]) + 1
csv_path = os.path.join(eval_dir, f"evaluation_summary_v{VERSION_NUM}_{eval_num:02d}.csv")

# Zusätzliche Metriken für CSV (detaillierter)
csv_rows = []
for pid in range(4):
    avg_return = returns_total[pid] / NUM_EPISODES
    total_points = points_total[pid]
    wins = win_counts[pid]
    win_rate = 100 * wins / NUM_EPISODES
    starts = start_counts[pid]
    starts_percent = 100 * starts / NUM_EPISODES
    strategy = PLAYER_TYPES[pid]

    row = {
        "version": VERSION_NUM,
        "eval_id": eval_num,
        "player": pid,
        "strategy": strategy,
        "win_rate_percent": round(win_rate, 2),
        "avg_return": round(avg_return, 2),
        "total_points": round(total_points, 1),
        "num_wins": wins,
        "num_starts": starts,
        "start_rate_percent": round(starts_percent, 2)
    }
    csv_rows.append(row)

pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
print(f"\n📄 Evaluation gespeichert unter: {csv_path}")

