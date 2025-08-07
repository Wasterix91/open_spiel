import os
import numpy as np
import pyspiel
import torch
import collections
import pandas as pd
import matplotlib.pyplot as plt
import math

import ppo_agent as ppo
import dqn_agent as dqn
import td_agent as td  # ✅ Neu hinzugefügt
from collections import defaultdict

# === Konfiguration ===
NUM_EPISODES = 10_000
PLAYER_CONFIG = [
    {"name": "Player0", "type": "max_combo2"},
    {"name": "Player1", "type": "random2"},
    {"name": "Player2", "type": "random2"},
    {"name": "Player3", "type": "random2"}
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

pd.DataFrame(PLAYER_CONFIG).to_csv(os.path.join(macro_dir, "player_config.csv"), index=False)

# === Spielinitialisierung ===
game = pyspiel.load_game("president", {
    "deck_size": "64",
    "shuffle_cards": True,
    "single_card_mode": False,
    "num_players": 4
})

params = game.get_parameters()
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

def max_combo_strategy2(state):
    decoded = [(a, state.action_to_string(state.current_player(), a)) for a in state.legal_actions()]
    if not decoded:
        return 0
    # Wähle Kombination mit größter Combo-Größe: Quad > Triple > Pair > Single
    def combo_size_priority(s):
        if "Quad" in s: return 4
        if "Triple" in s: return 3
        if "Pair" in s: return 2
        if "Single" in s: return 1
        return 0
    best = max(decoded, key=lambda x: (combo_size_priority(x[1]), -x[0]))
    return best[0]

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
    "max_combo2": max_combo_strategy2,
    "single_only": single_only_strategy,
    "smart": smart_strategy,
    "aggressive": aggressive_strategy
}

# === Agent-Ladefunktion ===
def load_agents(player_config, base_dir, game):
    agents = []
    for pid, cfg in enumerate(player_config):
        kind = cfg["type"]
        version = cfg.get("version", "01")

        if kind == "ppo":
            model_path = os.path.join(base_dir, f"models/ppo_model_{version}/train/ppo_model_{version}_agent_p{pid}")
            agent = ppo.PPOAgent(
                info_state_size=game.information_state_tensor_shape()[0],
                num_actions=game.num_distinct_actions()
            )
            agent.restore(model_path)
            agents.append(agent)

        elif kind == "dqn":
            model_path = os.path.join(base_dir, f"models/dqn_model_{version}/train/dqn_model_{version}_agent_p{pid}")
            agent = dqn.DQNAgent(
                state_size=game.observation_tensor_shape()[0],
                num_actions=game.num_distinct_actions()
            )
            agent.restore(model_path)
            agents.append(agent)

        elif kind == "td":
            model_path = os.path.join(base_dir, f"models/td_model_{version}/train/td_model_{version}_agent_p{pid}.pt")
            agent = td.TDAgent(
                obs_size=game.information_state_tensor_shape()[0],
                num_actions=game.num_distinct_actions()
            )
            agent.load(model_path)
            agents.append(agent)

        elif kind in strategy_map:
            agents.append(strategy_map[kind])

        else:
            raise ValueError(f"Unbekannter Agententyp: {kind}")
    return agents
""" 
# === Auswahlfunktion für Aktion ===
def choose_policy_action(agent, state, player):
    legal = state.legal_actions(player)

    if isinstance(agent, ppo.PPOAgent):
        obs = state.information_state_tensor(player)
        info_tensor = torch.tensor(obs, dtype=torch.float32).to(agent.device)
        logits = agent._policy(info_tensor).detach().cpu().numpy()
        masked = np.zeros_like(logits)
        masked[legal] = logits[legal]
        probs = masked / masked.sum() if masked.sum() > 0 else np.ones_like(masked) / len(legal)
        return collections.namedtuple("AgentOutput", ["action"])(action=np.argmax(probs))

    elif isinstance(agent, dqn.DQNAgent):
        obs = state.observation_tensor(player)
        return collections.namedtuple("AgentOutput", ["action"])(action=agent.select_action(obs, legal))

    elif isinstance(agent, td.TDAgent):
        obs = state.information_state_tensor(player)
        q_values = agent.predict(torch.tensor(obs, dtype=torch.float32))
        legal_qs = q_values[legal]
        return collections.namedtuple("AgentOutput", ["action"])(action=legal[torch.argmax(legal_qs).item()])

    else:
        raise ValueError("Unbekannter Agententyp bei choose_policy_action.") """
    

def choose_policy_action(agent, state, player):
    legal = state.legal_actions(player)

    if isinstance(agent, ppo.PPOAgent):
        obs = state.information_state_tensor(player)
        info_tensor = torch.tensor(obs, dtype=torch.float32).to(agent.device).unsqueeze(0)  # Batch-Dim hinzufügen
        with torch.no_grad():
            probs = agent._policy(info_tensor)[0].cpu().numpy()

        masked_probs = np.zeros_like(probs)
        masked_probs[legal] = probs[legal]

        total_prob = masked_probs.sum()
        if total_prob == 0:
            # Fallback: Gleichverteilung auf alle legalen Aktionen
            masked_probs[legal] = 1.0 / len(legal)
        else:
            masked_probs /= total_prob

        action = np.random.choice(len(masked_probs), p=masked_probs)
        return collections.namedtuple("AgentOutput", ["action"])(action=action)

    elif isinstance(agent, dqn.DQNAgent):
        obs = state.observation_tensor(player)
        return collections.namedtuple("AgentOutput", ["action"])(action=agent.select_action(obs, legal))

    elif isinstance(agent, td.TDAgent):
        obs = state.information_state_tensor(player)
        q_values = agent.predict(torch.tensor(obs, dtype=torch.float32).to(agent.device))
        legal_qs = q_values[legal]
        action = legal[torch.argmax(legal_qs).item()]
        return collections.namedtuple("AgentOutput", ["action"])(action=action)

    elif callable(agent):  # z.B. Smart-Strategie, Random etc.
        action = agent(state)
        # Optional: Sicherstellen, dass Aktion legal ist
        if action not in legal:
            action = np.random.choice(legal)
        return collections.namedtuple("AgentOutput", ["action"])(action=action)

    else:
        raise ValueError("Unbekannter Agententyp bei choose_policy_action.")


# === Evaluation starten ===
action_counts = defaultdict(lambda: defaultdict(int))
agents = load_agents(PLAYER_CONFIG, base_dir, game)
returns_total = np.zeros(4)
start_counts = collections.defaultdict(int)
win_counts = collections.defaultdict(int)

for episode in range(1, NUM_EPISODES + 1):
    state = game.new_initial_state()
    start_counts[state.current_player()] += 1

    while not state.is_terminal():
        pid = state.current_player()
        agent = agents[pid]

        if callable(agent):  # Random etc.
            action = agent(state)
        else:
            action = choose_policy_action(agent, state, pid).action

        action_counts[pid][action] += 1
        state.apply_action(action)
        


    final_returns = state.returns()
    for i, ret in enumerate(final_returns):
        returns_total[i] += ret
    win_counts[np.argmax(final_returns)] += 1

    if episode % 250 == 0:
        print(f"✅ Episode {episode} abgeschlossen")

# === Ergebnisse speichern ===
summary_rows = []
for i in range(4):
    config = PLAYER_CONFIG[i]
    avg_ret = returns_total[i] / NUM_EPISODES
    winrate = 100 * win_counts[i] / NUM_EPISODES
    label = f"{config['type']} v{config.get('version', '')}".strip()
    row = {
        "macro_id": next_macro_num,
        "player": config["name"],
        "strategy": label,
        "avg_return": round(avg_ret, 2),
        "win_rate_percent": round(winrate, 2),
        "num_wins": win_counts[i],
        "num_starts": start_counts[i],
    }
    summary_rows.append(row)

df = pd.DataFrame(summary_rows)
summary_path = os.path.join(csv_dir, "evaluation_summary.csv")
df.to_csv(summary_path, index=False)
print(f"📄 Evaluationsergebnisse gespeichert unter: {summary_path}")

if GENERATE_PLOTS:
    print(f"📁 Ergebnisse und Plots werden gespeichert in: {plot_dir}")

    num_actions = game.num_distinct_actions()
    action_labels = [f"Action {i}" for i in range(num_actions)]

    # Deckgröße ermitteln (int)
    deck_size = int(game.get_parameters().get("deck_size", "64"))

    # Combo-Labels entsprechend deck_size erweitern
    if deck_size == 64:
        combo_labels = ["Single", "Pair", "Triple", "Quad"] + [f"{i}-of-a-kind" for i in range(5, 9)]
    else:
        combo_labels = ["Single", "Pair", "Triple", "Quad"]

    # === 00 - Aktionsverteilung als Tabelle ===
    action_table = pd.DataFrame(index=[f"Player {i}" for i in range(4)], columns=action_labels)
    for pid in range(4):
        total = sum(action_counts[pid].values())
        for aid in range(num_actions):
            count = action_counts[pid].get(aid, 0)
            percent = 100 * count / total if total > 0 else 0
            action_table.loc[f"Player {pid}", f"Action {aid}"] = f"{count} ({percent:.1f}%)"

    table_path = os.path.join(csv_dir, "00_action_distribution.csv")
    action_table.to_csv(table_path)
    print(f"📄 Tabelle gespeichert unter: {table_path}")

    # === 01 - Gesamte Aktionsverteilung ===
    counts_per_action = {aid: [action_counts[pid].get(aid, 0) for pid in range(4)] for aid in range(num_actions)}
    x = np.arange(len(action_labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(20, 8))
    for pid in range(4):
        total_actions = sum(action_counts[pid].values()) or 1
        counts = [100 * counts_per_action[aid][pid] / total_actions for aid in range(num_actions)]
        strat = PLAYER_CONFIG[pid]["type"]
        winrate = 100 * win_counts[pid] / NUM_EPISODES
        ax.bar(x + width * pid, counts, width, label=f"Player {pid} ({strat}, {winrate:.1f}%)")

    ax.set_xlabel("Action")
    ax.set_ylabel("Relative Häufigkeit (%)")
    ax.set_title("01 - Action Counts per Player")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(action_labels, rotation=90)
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(plot_dir, "01_action_distribution_total.jpg"))

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
    labels = [
        f"Player {i} ({PLAYER_CONFIG[i]['type']}, {win_counts[i] / NUM_EPISODES:.1%})"
        for i in range(4)
    ]
    ax.bar(x - width / 2, play_counts, width, label="Play", color="#1f77b4")
    ax.bar(x + width / 2, pass_counts, width, label="Pass", color="#d62728")
    ax.set_ylabel("Anteil an allen Aktionen (%)")
    ax.set_xlabel("Spieler")
    ax.set_title("02 - Anteil von Pass vs. Spiel-Aktionen")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(plot_dir, "02_pass_vs_play.jpg"))

    # === 03 - Kombotyp-Anteile je Spieler ===
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
    ax.set_title("03 - Anteil gespielter Kombitypen pro Spieler")
    ax.set_xticks(x + width * (len(combo_labels)-1) / 2)
    ax.set_xticklabels([f"Player {i}" for i in range(4)])
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(plot_dir, "03_combo_types_per_player.jpg"))

    # === 04-11 - Detaillierte Kombitypenplots ===
    combo_plot_index = {ctype: f"{4+i:02d}" for i, ctype in enumerate(combo_labels)}
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
            strat = PLAYER_CONFIG[pid]["type"]
            winrate = 100 * win_counts[pid] / NUM_EPISODES
            ax.bar(x + width * pid, counts, width, label=f"Player {pid} ({strat}, {winrate:.1f}%)")

        ax.set_xlabel(f"{combo}-Actions")
        ax.set_ylabel("Relative Häufigkeit (%)")
        ax.set_title(f"{combo_plot_index[combo]} - {combo}-Actions pro Spieler")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(labels, rotation=90)
        ax.set_ylim(0, math.ceil(max_height * 1.2 / 5) * 5)
        ax.legend()
        fig.tight_layout()
        filename = f"{combo_plot_index[combo]}_combo_{combo.lower().replace('-','_')}_detailed.jpg"
        plt.savefig(os.path.join(plot_dir, filename))

    print("✅ Alle Plots erfolgreich gespeichert!")
