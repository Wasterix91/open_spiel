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

# === 🧠 Spielkonfiguration ============================
game = pyspiel.load_game(
    "president",
    {
        "deck_size": "32",
        "shuffle_cards": True,
        "single_card_mode": False,
        "num_players": 4
    }
)

# === ⚙️ Evaluationsparameter ===========================
VERSION_NUM = "02"
NUM_EPISODES = 1_000
PLAYER_TYPES = ["ppo", "random", "random", "random"]  # Alternativen: random, max_combo, single_only, smart, aggressive
MODEL_DIR = f"/home/wasterix/OpenSpiel/open_spiel/Playground/models/selfplay_president_{VERSION_NUM}/train"
GENERATE_PLOTS = True  # False → keine Plots erzeugen

# === 🖨️ Übersichtsausgabe =============================
print("=== 🧪 President Game Evaluation ===")
print(f"🎮 Spielerzahl:           {game.num_players()}  → {PLAYER_TYPES}")
print(f"🤖 PPO-Agent Version:     v{VERSION_NUM}")
print(f"🔢 Distinct Actions:      {game.num_distinct_actions()}")
print(f"🧠 Observation Tensor:    {', '.join(map(str, game.observation_tensor_shape()))}")
print(f"🎲 Anzahl Episoden:       {NUM_EPISODES}")
print(f"📊 Plots aktiv:           {GENERATE_PLOTS}")


print("\n=== Start Evaluation ===")

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

    if episode % 250 == 0:
        print(f"✅ Episode {episode} abgeschlossen")

# === 5️⃣ Ergebnisse anzeigen ===
print("\n=== Auswertung nach 1000 Spielen ===")
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


if GENERATE_PLOTS:
    # === 📁 Evaluations-Ausgabepfad vorbereiten ===
    BASE_EVAL_DIR = os.path.join(os.path.dirname(MODEL_DIR), "eval")

    os.makedirs(BASE_EVAL_DIR, exist_ok=True)

    # Finde nächste freie Nummer (z. B. 01, 02, ...)
    existing_dirs = sorted([d for d in os.listdir(BASE_EVAL_DIR) if d.isdigit()])
    if existing_dirs:
        next_eval_num = int(existing_dirs[-1]) + 1
    else:
        next_eval_num = 1

    eval_subdir = os.path.join(BASE_EVAL_DIR, f"{next_eval_num:02d}")
    os.makedirs(eval_subdir)
    print(f"📁 Ergebnisse und Plots werden gespeichert in: {eval_subdir}")


    # === 📊 Aktionsverteilung als Tabelle speichern ===
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

    # CSV speichern
    csv_path = os.path.join(eval_subdir, f"aktionsverteilung_v{VERSION_NUM}.csv")
    action_table.to_csv(csv_path)
    print(f"📁 Tabelle gespeichert unter: {csv_path}")

    # === 📊 Gesamte Aktionsverteilung (alle Aktionen) ===
    counts_per_action = {aid: [action_counts[pid].get(aid, 0) for pid in range(4)] for aid in range(num_actions)}
    x = np.arange(len(action_labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(20, 8))
    for pid in range(4):
        total_actions = sum(action_counts[pid].values()) or 1
        counts = [100 * counts_per_action[aid][pid] / total_actions for aid in range(num_actions)]
        ax.bar(x + width * pid, counts, width, label=f"Player {pid}")
    ax.set_xlabel("Action")
    ax.set_ylabel("Relative Häufigkeit (%)")
    ax.set_title("Action Counts per Player")
    ax.set_xticks(x)
    ax.set_xticklabels(action_labels, rotation=90)
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(eval_subdir, f"aktionsverteilung_v{VERSION_NUM}.jpg"))

    # === 📊 Einzelplots pro Kombotyp ===
    combo_labels = ["Single", "Pair", "Triple", "Quad"]
    action_types = {}
    dummy_state = game.new_initial_state()
    for aid in range(num_actions):
        try:
            label = dummy_state.action_to_string(0, aid)
            for ctype in combo_labels:
                if ctype in label:
                    action_types[aid] = ctype
                    break
        except:
            continue

    combo_actions = {ctype: [] for ctype in combo_labels}
    for aid, ctype in action_types.items():
        combo_actions[ctype].append(aid)

    for combo, aids in combo_actions.items():
        if not aids: continue
        labels = [f"Action {aid}" for aid in aids]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(max(10, len(labels)), 6))
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
        ax.set_ylim(0, max(1, max_height * 1.1))
        ax.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(eval_subdir, f"aktionsverteilung_v{VERSION_NUM}_{combo.lower()}_detailliert.jpg"))

    # === 📊 Kombotypen-Anteil pro Spieler ===
    combo_totals = {pid: {ctype: 0 for ctype in combo_labels} for pid in range(4)}
    for pid in range(4):
        for aid, count in action_counts[pid].items():
            combo = action_types.get(aid)
            if combo:
                combo_totals[pid][combo] += count
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, combo in enumerate(combo_labels):
        counts = []
        for pid in range(4):
            total = sum(combo_totals[pid].values()) or 1
            percent = 100 * combo_totals[pid][combo] / total
            counts.append(percent)
        ax.bar(x + i * width, counts, width, label=combo)
    ax.set_xlabel("Spieler")
    ax.set_ylabel("Anteil an allen Aktionen (%)")
    ax.set_title("Anteil gespielter Singles, Pairs, Triples und Quads pro Spieler")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"Player {i}" for i in range(4)])
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(eval_subdir, f"aktionsverteilung_v{VERSION_NUM}_kombotypen_anteile.jpg"))

    # === 📊 Pass vs. Play Plot (nebeneinander) ===
    pass_stats = {pid: {"Pass": 0, "Play": 0} for pid in range(4)}
    dummy_state = game.new_initial_state()
    action_labels_map = {}
    for aid in range(num_actions):
        try:
            label = dummy_state.action_to_string(0, aid)
            action_labels_map[aid] = "Pass" if "Pass" in label else "Play"
        except:
            continue
    for pid in range(4):
        for aid, count in action_counts[pid].items():
            label = action_labels_map.get(aid, "Play")
            pass_stats[pid][label] += count
    play_counts, pass_counts = [], []
    for pid in range(4):
        total = sum(pass_stats[pid].values()) or 1
        play_counts.append(100 * pass_stats[pid]["Play"] / total)
        pass_counts.append(100 * pass_stats[pid]["Pass"] / total)
    x = np.arange(4)
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
    plt.savefig(os.path.join(eval_subdir, f"aktionsverteilung_v{VERSION_NUM}_pass_vs_play.jpg"))

    print("✅ Alle Plots erfolgreich gespeichert!")

