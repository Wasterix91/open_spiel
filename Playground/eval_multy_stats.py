import os
import re
import numpy as np
import pyspiel
import torch
import collections
from collections import defaultdict
import pandas as pd
import ppo_local_2 as ppo  # ggf. anpassen
import matplotlib.pyplot as plt

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
VERSION_NUM = "03"

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
    if not decoded:
        return 0
    return max(decoded, key=lambda x: parse_rank(x[1]))[0]

def single_only_strategy(state):
    decoded = decode_actions(state)
    singles = [x for x in decoded if "Single" in x[1]]
    return min(singles, key=lambda x: x[0])[0] if singles else 0

def random_action_strategy(state):
    return np.random.choice(state.legal_actions())

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

# === 🧮 Aktionszählung initialisieren ===
action_counts = defaultdict(lambda: defaultdict(int))

# === 4️⃣ Mehrere Spiele ausführen ===
NUM_EPISODES = 1_000
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
            agent_out = choose_policy_action(agent_or_strategy, state, player)
            action = agent_out.action
        else:
            action = agent_or_strategy(state)
        state.apply_action(action)

        # Aktion zählen
        action_counts[player][action] += 1

    final_returns = state.returns()

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

# === 📊 Aktionsverteilung als Tabelle ===
num_actions = game.num_distinct_actions()
action_labels = [f"Action {i}" for i in range(num_actions)]
action_table = pd.DataFrame(index=[f"Player {i}" for i in range(4)],
                            columns=action_labels)

for pid in range(4):
    total = sum(action_counts[pid].values())
    for aid in range(num_actions):
        count = action_counts[pid].get(aid, 0)
        percent = 100 * count / total if total > 0 else 0
        action_table.loc[f"Player {pid}", f"Action {aid}"] = f"{count} ({percent:.1f}%)"

# Optional: Speichern als CSV
csv_path = f"aktionsverteilung_v{VERSION_NUM}.csv"
action_table.to_csv(csv_path)
print(f"\nTabelle gespeichert!")

labels = [f"Action {i}" for i in range(num_actions)]
counts_per_action = {aid: [action_counts[pid].get(aid, 0) for pid in range(4)] for aid in range(num_actions)}

x = np.arange(len(labels))
width = 0.2
fig, ax = plt.subplots(figsize=(20, 8))
ax.set_ylim(0, 100)

for pid in range(4):
    total_actions = sum(action_counts[pid].values()) or 1  # Vermeide Division durch 0
    counts = [100 * counts_per_action[aid][pid] / total_actions for aid in range(num_actions)]
    ax.bar(x + width * pid, counts, width, label=f"Player {pid}")

ax.set_xlabel("Action")
ax.set_ylabel("Relative Häufigkeit (%)")
ax.set_title("Action Counts per Player")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=90)
ax.legend()
fig.tight_layout()

plot_path = f"aktionsverteilung_v{VERSION_NUM}.jpg"
plt.savefig(plot_path)

# === 📊 Zusätzliche Plots: Verteilung einzelner Aktionen pro Kombo-Typ ===

from matplotlib import cm

# Schritt 1: Action-Typ pro Aktion identifizieren
num_actions = game.num_distinct_actions()
action_types = {}  # {action_id: "Single" / "Pair" / "Triple" / "Quad"}

# Ein Beispielzustand für die String-Dekodierung
dummy_state = game.new_initial_state()
for aid in range(num_actions):
    try:
        action_str = dummy_state.action_to_string(0, aid)
        for combo in ["Single", "Pair", "Triple", "Quad"]:
            if combo in action_str:
                action_types[aid] = combo
                break
    except:
        continue  # manche Aktionen werfen evtl. Fehler bei .to_string()

# Schritt 2: Zähle Aktionen pro Spieler & Kombotyp
combo_actions = {combo: [] for combo in ["Single", "Pair", "Triple", "Quad"]}
for aid, combo in action_types.items():
    combo_actions[combo].append(aid)

# Schritt 3: Plotte für jeden Kombotyp ein Balkendiagramm
for combo, aids in combo_actions.items():
    if not aids:
        continue  # falls keine Aktionen für diesen Typ

    labels = [f"Action {aid}" for aid in aids]
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(10, len(labels)), 6))

    # Daten vorbereiten & maximale Höhe für Skalierung berechnen
    max_height = 0
    for pid in range(4):
        total_actions = sum(action_counts[pid].values()) or 1
        counts = [100 * action_counts[pid].get(aid, 0) / total_actions for aid in aids]
        max_height = max(max_height, max(counts, default=0))
        ax.bar(x + width * pid, counts, width, label=f"Player {pid}")

    ax.set_xlabel(f"{combo}-Actions")
    ax.set_ylabel("Relative Häufigkeit (%)")
    ax.set_title(f"{combo}-Action-Verteilung pro Spieler")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels, rotation=90)

    # Dynamische Y-Achse mit 10% Puffer
    ax.set_ylim(0, max(1, max_height * 1.1))
    ax.legend()
    fig.tight_layout()

    plot_path = f"aktionsverteilung_v{VERSION_NUM}_{combo.lower()}_detailliert.jpg"
    plt.savefig(plot_path)



# === 📊 Plot: Anteil gespielter Kombotypen pro Spieler (relativ in %) ===

# Schritt 1: Initialisiere Zähler
combo_totals = {pid: {"Single": 0, "Pair": 0, "Triple": 0, "Quad": 0} for pid in range(4)}

# Mapping: Action-ID → Kombotyp
num_actions = game.num_distinct_actions()
action_types = {}
dummy_state = game.new_initial_state()
for aid in range(num_actions):
    try:
        action_str = dummy_state.action_to_string(0, aid)
        for combo in ["Single", "Pair", "Triple", "Quad"]:
            if combo in action_str:
                action_types[aid] = combo
                break
    except:
        continue

# Schritt 2: Aktionen pro Spieler & Typ aufsummieren
for pid in range(4):
    for aid, count in action_counts[pid].items():
        combo = action_types.get(aid, None)
        if combo:
            combo_totals[pid][combo] += count

# Schritt 3: Plot vorbereiten
combo_types = ["Single", "Pair", "Triple", "Quad"]
player_labels = [f"Player {i}" for i in range(4)]
x = np.arange(len(player_labels))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))

for i, combo in enumerate(combo_types):
    counts = []
    for pid in range(4):
        total = sum(combo_totals[pid].values()) or 1
        percent = 100 * combo_totals[pid][combo] / total
        counts.append(percent)
    ax.bar(x + i * width, counts, width, label=combo, alpha=0.9)

ax.set_xlabel("Spieler")
ax.set_ylabel("Anteil an allen Aktionen (%)")
ax.set_title("Anteil gespielter Singles, Pairs, Triples und Quads pro Spieler")
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(player_labels)
ax.set_ylim(0, 100)
ax.legend()
fig.tight_layout()

plot_path = f"aktionsverteilung_v{VERSION_NUM}_kombotypen_anteile.jpg"
plt.savefig(plot_path)

# === 📊 Plot: Anteil von „Pass“ vs. Spiel-Aktionen pro Spieler (nebeneinander) ===

pass_stats = {pid: {"Pass": 0, "Play": 0} for pid in range(4)}

# Mapping: action_id → "Pass" oder "Play"
dummy_state = game.new_initial_state()
action_labels_map = {}
for aid in range(game.num_distinct_actions()):
    try:
        action_str = dummy_state.action_to_string(0, aid)
        if "Pass" in action_str:
            action_labels_map[aid] = "Pass"
        else:
            action_labels_map[aid] = "Play"
    except:
        continue

# Zähle Aktionen pro Spieler
for pid in range(4):
    for aid, count in action_counts[pid].items():
        label = action_labels_map.get(aid, "Play")
        pass_stats[pid][label] += count

# Prozentwerte vorbereiten
play_counts = []
pass_counts = []

for pid in range(4):
    total = sum(pass_stats[pid].values()) or 1
    play_pct = 100 * pass_stats[pid]["Play"] / total
    pass_pct = 100 * pass_stats[pid]["Pass"] / total
    play_counts.append(play_pct)
    pass_counts.append(pass_pct)

# Plot erstellen: nebeneinander
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(4)  # Spieler
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width / 2, play_counts, width, label="Play", color="#1f77b4")
ax.bar(x + width / 2, pass_counts, width, label="Pass", color="#d62728")

ax.set_ylabel("Anteil an allen Aktionen (%)")
ax.set_xlabel("Spieler")
ax.set_title("Anteil von Pass vs. Spiel-Aktionen pro Spieler")
ax.set_xticks(x)
ax.set_xticklabels([f"Player {i}" for i in range(4)])
ax.set_ylim(0, 100)
ax.legend()
fig.tight_layout()

plot_path = f"aktionsverteilung_v{VERSION_NUM}_pass_vs_play.jpg"
plt.savefig(plot_path)

print("Plots gespeichert!")
