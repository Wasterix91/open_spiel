import os
import re
import numpy as np
import pyspiel
import torch
import collections
from collections import defaultdict
import ppo_local_2 as ppo  # ggf. anpassen



# === 1️⃣ Spiel erstellen ===
game = pyspiel.load_game(
    "president",
    {
        "deck_size": "32",
        "shuffle_cards": True,
        "single_card_mode": False,
        "num_players": 4
    }
)

print("=== President Game ===")
print(f"Num players: {game.num_players()}")
print(f"Num distinct actions: {game.num_distinct_actions()}")
print(f"Observation tensor shape: {game.observation_tensor_shape()}")

# === 🔢 Versionsnummer definieren ===
VERSION_NUM = "08"  # z. B. Eingabe über CLI oder oben ändern

# === 2️⃣ Agenten vorbereiten ===
PLAYER_TYPES = ["ppo", "random", "random", "random"]
MODEL_DIR = f"/home/wasterix/OpenSpiel/open_spiel/Playground/models/selfplay_president_{VERSION_NUM}/train"

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
    return max(decoded, key=lambda x: parse_rank(x[1]))[0] if decoded else 0

def single_only_strategy(state):
    decoded = decode_actions(state)
    singles = [x for x in decoded if "Single" in x[1]]
    return min(singles, key=lambda x: x[0])[0] if singles else 0

def random_action_strategy(state):
    return np.random.choice(state.legal_actions())

def aggressive_strategy(state):
    decoded = decode_actions(state)
    if not decoded:
        return 0
    # Alle sind "Single" – wähle die höchste
    return max(decoded, key=lambda x: parse_rank(x[1]))[0]

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
    "max_combo": max_combo_strategy,
    "single_only": single_only_strategy,
    "smart": smart_strategy,
    "aggressive": aggressive_strategy
}


agents = []
for pid, ptype in enumerate(PLAYER_TYPES):
    if ptype == "ppo":
        agent = ppo.PPOAgent(
            info_state_size=game.information_state_tensor_shape()[0],
            num_actions=game.num_distinct_actions()
        )
        model_path = f"{MODEL_DIR}/selfplay_president_{VERSION_NUM}_agent_p{pid}"
        agent.restore(model_path)
        agents.append(agent)
    elif ptype in strategy_map:
        agents.append(strategy_map[ptype])
    else:
        raise ValueError(f"Unbekannter Spielertyp: {ptype}")


# === 3️⃣ PPO-Auswahlfunktion ===
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

    legal_probs = np.array([probs[a] for a in legal_actions])
    return collections.namedtuple("AgentOutput", ["action", "probs", "legal_actions"])(
        action=np.argmax(probs), probs=legal_probs, legal_actions=legal_actions)

# === 4️⃣ Mehrere Spiele ausführen ===
NUM_EPISODES = 1_000
returns_total = np.zeros(4)
start_counts = defaultdict(int)
win_counts = defaultdict(int)  # 🆕 Neu: Zähle Siege
points_total = np.zeros(4)     # 🆕 Neu: Punkte über alle Spiele

for episode in range(1, NUM_EPISODES + 1):
    state = game.new_initial_state()
    start_player = state.current_player()
    start_counts[start_player] += 1

    while not state.is_terminal():
        player = state.current_player()
        agent_or_strategy = agents[player]
        if isinstance(agent_or_strategy, ppo.PPOAgent):
            agent_out = choose_policy_action(agent_or_strategy, state, player)
            action = agent_out.action
        else:
            action = agent_or_strategy(state)
        state.apply_action(action)

    final_returns = state.returns()

    # 🆕 Punkte & Siege erfassen
    for pid, ret in enumerate(final_returns):
        returns_total[pid] += ret
        points_total[pid] += ret

    winner_pid = np.argmax(final_returns)
    win_counts[winner_pid] += 1

    if episode % 100 == 0:
        print(f"✅ Episode {episode} abgeschlossen")


# === 5️⃣ Ergebnisse anzeigen ===
print("\n=== ✅ Auswertung nach 1000 Spielen ===")
print("Durchschnittliche Returns pro Spieler:")
for pid, total_return in enumerate(returns_total):
    avg_return = total_return / NUM_EPISODES
    print(f"Player {pid} ({PLAYER_TYPES[pid]}): Ø Return = {avg_return:.2f}")

print("\nGesamte Punkte pro Spieler:")
for pid, points in enumerate(points_total):
    print(f"Player {pid} ({PLAYER_TYPES[pid]}): {points:.1f} Punkte")

print("\nAnzahl gewonnener Spiele:")
for pid in range(4):
    print(f"Player {pid} ({PLAYER_TYPES[pid]}): {win_counts[pid]} Siege")

print("\nSiegrate pro Spieler (%):")
for pid in range(4):
    wins = win_counts[pid]
    win_rate = 100 * wins / NUM_EPISODES
    print(f"Player {pid} ({PLAYER_TYPES[pid]}): {win_rate:.2f}%")

print("\nStartspieler-Häufigkeit:")
total_starts = sum(start_counts.values())
for pid in range(4):
    count = start_counts[pid]
    share = 100 * count / total_starts if total_starts else 0
    print(f"Player {pid} started {count} times ({share:.2f}%)")

