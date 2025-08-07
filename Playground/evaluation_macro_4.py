import os
import numpy as np
import pyspiel
import torch
import collections
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import math

import ppo_agent as ppo
import dqn_agent as dqn

# === Feste Konfiguration ===
NUM_EPISODES = 4000
PLAYER_CONFIG = [
    {"name": "Player0", "type": "ppo", "version": "14"},
    {"name": "Player1", "type": "dqn", "version": "08"},
    {"name": "Player2", "type": "ppo", "version": "14"},
    {"name": "Player3", "type": "dqn", "version": "08"}
]

GENERATE_PLOTS = True
EVAL_OUTPUT = True

# === Speicherlogik ===
base_dir = os.path.dirname(os.path.abspath(__file__))
eval_macro_root = os.path.join(base_dir, "eval_macro")
os.makedirs(eval_macro_root, exist_ok=True)

existing_macro_dirs = sorted([d for d in os.listdir(eval_macro_root) if d.startswith("eval_macro_")])
next_macro_num = int(existing_macro_dirs[-1].split("_")[-1]) + 1 if existing_macro_dirs else 1
macro_dir = os.path.join(eval_macro_root, f"eval_macro_{next_macro_num:02d}")
os.makedirs(macro_dir, exist_ok=True)

csv_dir = os.path.join(macro_dir, "csv")
plot_dir = os.path.join(macro_dir, "plots")
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# === Konfiguration speichern ===
pd.DataFrame(PLAYER_CONFIG).to_csv(os.path.join(macro_dir, "player_config.csv"), index=False)

# === Spielinitialisierung ===
game = pyspiel.load_game("president", {
    "deck_size": "32",
    "shuffle_cards": True,
    "single_card_mode": False,
    "num_players": 4
})

RANKS = ["7", "8", "9", "10", "J", "Q", "K", "A"]
RANK_TO_NUM = {rank: i for i, rank in enumerate(RANKS)}

# === Strategien ===
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
            model_path = os.path.join(base_dir, f"models/ppo_model_{version}/train/ppo_model_{version}_agent_p{pid}")
            agent = ppo.PPOAgent(
                info_state_size=game.information_state_tensor_shape()[0],
                num_actions=game.num_distinct_actions()
            )
            agent.restore(model_path)
            agents.append(agent)

        elif kind == "dqn":
            version = cfg["version"]
            model_path = os.path.join(base_dir, f"models/dqn_model_{version}/train/dqn_model_{version}_agent_p{pid}")
            agent = dqn.DQNAgent(
                state_size=game.observation_tensor_shape()[0],
                num_actions=game.num_distinct_actions()
            )
            agent.restore(model_path)
            agents.append(agent)

        elif kind in strategy_map:
            agents.append(strategy_map[kind])
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
action_counts = defaultdict(lambda: defaultdict(int))

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
        action_counts[player][action] += 1

    final_returns = state.returns()
    for pid, ret in enumerate(final_returns):
        returns_total[pid] += ret
        points_total[pid] += ret
    win_counts[np.argmax(final_returns)] += 1

    if episode % 250 == 0:
        print(f"✅ Episode {episode} abgeschlossen")

# === Ergebnisse ausgeben ===
print("\n=== 📊 Ergebnisse ===")
summary_rows = []
for pid in range(4):
    cfg = PLAYER_CONFIG[pid]
    avg_return = returns_total[pid] / NUM_EPISODES
    winrate = 100 * win_counts[pid] / NUM_EPISODES
    version = cfg.get("version")
    label = f"{cfg['type']} v{version}" if version else cfg["type"]
    print(f"{cfg['name']} ({label}): Ø Return = {avg_return:.2f}, "
          f"Siege = {win_counts[pid]}, Winrate = {winrate:.2f}%, Starts = {start_counts[pid]}")

    row = {
        "macro_id": next_macro_num,
        "player": cfg["name"],
        "strategy": label,
        "win_rate_percent": round(winrate, 2),
        "avg_return": round(avg_return, 2),
        "total_points": round(points_total[pid], 1),
        "num_wins": win_counts[pid],
        "num_starts": start_counts[pid],
        "start_rate_percent": round(100 * start_counts[pid] / NUM_EPISODES, 2)
    }
    summary_rows.append(row)

if EVAL_OUTPUT:
    df_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(csv_dir, "evaluation_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"📄 Evaluationsergebnisse gespeichert unter: {summary_path}")

# === Aktionsverteilung plotten ===
""" if GENERATE_PLOTS:
    print(f"📈 Plots werden gespeichert unter: {plot_dir}")
    num_actions = game.num_distinct_actions()
    action_labels = [f"A{i}" for i in range(num_actions)]
    counts_per_action = {aid: [action_counts[pid].get(aid, 0) for pid in range(4)] for aid in range(num_actions)}
    x = np.arange(len(action_labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(20, 8))
    for pid in range(4):
        total_actions = sum(action_counts[pid].values()) or 1
        counts = [100 * counts_per_action[aid][pid] / total_actions for aid in range(num_actions)]
        label = f"{PLAYER_CONFIG[pid]['name']} - {PLAYER_CONFIG[pid]['type']} ({win_counts[pid] / NUM_EPISODES:.1%})"
        ax.bar(x + width * pid, counts, width, label=label)

    ax.set_xlabel("Action-ID")
    ax.set_ylabel("Relative Häufigkeit (%)")
    ax.set_title("Aktionen pro Spieler")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(action_labels, rotation=90)
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(plot_dir, "01_aktionsverteilung.jpg"))
    plt.close()
    print("✅ Aktionsplot gespeichert.") """

if GENERATE_PLOTS:
    print(f"📁 Ergebnisse und Plots werden gespeichert in: {plot_dir}")

    # === 00 - Aktionsverteilung als Tabelle ===
    num_actions = game.num_distinct_actions()
    action_labels = [f"Action {i}" for i in range(num_actions)]
    action_table = pd.DataFrame(index=[f"Player {i}" for i in range(4)], columns=action_labels)

    for pid in range(4):
        total = sum(action_counts[pid].values())
        for aid in range(num_actions):
            count = action_counts[pid].get(aid, 0)
            percent = 100 * count / total if total > 0 else 0
            action_table.loc[f"Player {pid}", f"Action {aid}"] = f"{count} ({percent:.1f}%)"

    csv_path = os.path.join(plot_dir, f"00_aktionsverteilung.csv")
    action_table.to_csv(csv_path)
    print(f"📁 Tabelle gespeichert unter: {csv_path}")

    # === 01 - Gesamte Aktionsverteilung ===
    counts_per_action = {aid: [action_counts[pid].get(aid, 0) for pid in range(4)] for aid in range(num_actions)}
    x = np.arange(len(action_labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(20, 8))
    for pid in range(4):
        total_actions = sum(action_counts[pid].values()) or 1
        counts = [100 * counts_per_action[aid][pid] / total_actions for aid in range(num_actions)]
        strategy = PLAYER_CONFIG[pid]["type"]
        winrate = 100 * win_counts[pid] / NUM_EPISODES
        ax.bar(x + width * pid, counts, width, label=f"Player {pid} ({strategy}, {winrate:.1f}%)")

    ax.set_xlabel("Action")
    ax.set_ylabel("Relative Häufigkeit (%)")
    ax.set_title("01 - Action Counts per Player")
    ax.set_xticks(x)
    ax.set_xticklabels(action_labels, rotation=90)
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"01_aktionsverteilung_gesamt.jpg"))

    # === 02 - Pass vs. Play ===
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

    play_counts = []
    pass_counts = []
    for pid in range(4):
        total = sum(pass_stats[pid].values()) or 1
        play_counts.append(100 * pass_stats[pid]["Play"] / total)
        pass_counts.append(100 * pass_stats[pid]["Pass"] / total)

    x = np.arange(4)
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    player_labels = [
    f"{PLAYER_CONFIG[i]['name']} ({PLAYER_CONFIG[i]['type']}, {win_counts[i] / NUM_EPISODES:.1%})"
    for i in range(4)
    ]

    ax.bar(x - width / 2, play_counts, width, label="Play", color="#1f77b4")
    ax.bar(x + width / 2, pass_counts, width, label="Pass", color="#d62728")
    ax.set_ylabel("Anteil an allen Aktionen (%)")
    ax.set_xlabel("Spieler")
    ax.set_title("02 - Anteil von Pass vs. Spiel-Aktionen pro Spieler")
    ax.set_xticks(x)
    ax.set_xticklabels(player_labels, rotation=0)
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"02_aktionsverteilung_pass_vs_play.jpg"))

    # === 03 - Kombotyp-Anteile je Spieler ===
    combo_labels = ["Single", "Pair", "Triple", "Quad"]
    action_types = {}
    for aid in range(num_actions):
        try:
            label = dummy_state.action_to_string(0, aid)
            for ctype in combo_labels:
                if ctype in label:
                    action_types[aid] = ctype
                    break
        except:
            continue

    combo_totals = {pid: {ctype: 0 for ctype in combo_labels} for pid in range(4)}
    for pid in range(4):
        for aid, count in action_counts[pid].items():
            combo = action_types.get(aid)
            if combo:
                combo_totals[pid][combo] += count

    x = np.arange(4)
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, combo in enumerate(combo_labels):
        counts = []
        for pid in range(4):
            total = sum(combo_totals[pid].values()) or 1
            percent = 100 * combo_totals[pid][combo] / total
            counts.append(percent)
        ax.bar(x + i * width, counts, width, label=combo)

    ax.set_xlabel("Spieler")
    ax.set_ylabel("Anteil an allen Aktionen (%)")
    ax.set_title("03 - Anteil gespielter Kombotypen pro Spieler")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(player_labels, rotation=0)
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"03_aktionsverteilung_kombotypen_anteile.jpg"))

    # === 04-07 - Detaillierte Kombotyp-Plots ===
    combo_plot_index = {"Single": "04", "Pair": "05", "Triple": "06", "Quad": "07"}
    combo_actions = {ctype: [] for ctype in combo_labels}
    for aid, ctype in action_types.items():
        combo_actions[ctype].append(aid)

    for combo, aids in combo_actions.items():
        if not aids:
            continue
        labels = [f"Action {aid}" for aid in aids]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(max(10, len(labels)), 6))
        max_height = 0
        for pid in range(4):
            total_actions = sum(action_counts[pid].values()) or 1
            counts = [100 * action_counts[pid].get(aid, 0) / total_actions for aid in aids]
            max_height = max(max_height, max(counts, default=0))
            strategy = PLAYER_CONFIG[pid]["type"]
            winrate = 100 * win_counts[pid] / NUM_EPISODES
            ax.bar(x + width * pid, counts, width, label=f"Player {pid} ({strategy}, {winrate:.1f}%)")

        ax.set_xlabel(f"{combo}-Actions")
        ax.set_ylabel("Relative Häufigkeit (%)")
        ax.set_title(f"{combo_plot_index[combo]} - {combo}-Action-Verteilung pro Spieler")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(labels, rotation=90)
        rounded_top = math.ceil(max(max_height * 1.2, 0.05) * 20) / 20
        ax.set_ylim(0, rounded_top)
        ax.legend()
        fig.tight_layout()
        filename = f"{combo_plot_index[combo]}_aktionsverteilung_{combo.lower()}_detailliert.jpg"
        plt.savefig(os.path.join(plot_dir, filename))

    print("✅ Alle Plots erfolgreich gespeichert!")
