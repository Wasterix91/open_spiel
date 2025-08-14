import os
import numpy as np
import pyspiel
import torch
import collections
import pandas as pd
import matplotlib.pyplot as plt
import math

#import ppo_agent as ppo
import ppo_agent_self_new as ppo
import dqn_agent as dqn
import td_agent as td  
from collections import defaultdict

# === Konfiguration ===
NUM_EPISODES = 10_000
PLAYER_CONFIG = [
    {"name": "Player0", "type": "ppo", "version": "69", "episode": 430_000},
    {"name": "Player1", "type": "max_combo"},
    {"name": "Player2", "type": "max_combo"},
    {"name": "Player3", "type": "max_combo"}
]

"""
Beispielaufruf für Spielerkonfiguration

PLAYER_CONFIG = [
    {"name": "Player0", "type": "ppo", "version": "57", "episode": 90_000},
    {"name": "Player1", "type": "max_combo"},
    {"name": "Player2", "type": "max_combo"},
    {"name": "Player3", "type": "max_combo"}
]

strategy_map = {
    "random": random_action_strategy,
    "random2": random2_action_strategy,
    "max_combo": max_combo_strategy,
    "max_combo2": max_combo_strategy2,
    "single_only": single_only_strategy,
    "smart": smart_strategy,
    "aggressive": aggressive_strategy
}
"""

GENERATE_PLOTS = True
EVAL_OUTPUT = True

# === Plot-Flags ===
USE_READABLE_LABELS = True       # True = "Pass", "S 7", "P J", ... ; False = "Action X"
READABLE_USE_NUMBERS = False     # True = 1/2/3/4 statt S/P/T/Q
MAX_ACTION_ID_TO_SHOW = 32       # None = alle; sonst max. Action-ID auf der X-Achse

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

def _infer_seat_id_dim_from_checkpoint(policy_path: str, info_state_size: int) -> int:
    sd = torch.load(policy_path, map_location="cpu")
    # erster Linear-Layer in Policy: 'net.0.weight' hat Shape [128, input_dim]
    w = sd["net.0.weight"]
    input_dim = w.shape[1]
    seat_id_dim = max(0, int(input_dim) - int(info_state_size))
    return seat_id_dim


# === Episode-spezifisches PPO-Laden ===
def _pad_episode(ep: int, width: int = 7) -> str:
    return f"{ep:0{width}d}"

def _load_ppo_episode(agent, base_dir, version, pid, episode: int):
    ep_tag = _pad_episode(episode)
    prefix = os.path.join(
        base_dir,
        f"models/ppo_model_{version}/train/ppo_model_{version}_agent_p{pid}_ep{ep_tag}"
    )
    policy_path = f"{prefix}_policy.pt"
    value_path  = f"{prefix}_value.pt"

    if not (os.path.exists(policy_path) and os.path.exists(value_path)):
        raise FileNotFoundError(f"Episode-Files nicht gefunden: {policy_path} / {value_path}")

    map_loc = agent.device if hasattr(agent, "device") else "cpu"
    policy_sd = torch.load(policy_path, map_location=map_loc)
    value_sd  = torch.load(value_path,  map_location=map_loc)

    agent._policy.load_state_dict(policy_sd)
    agent._value.load_state_dict(value_sd)

    agent._policy.to(map_loc).eval()
    agent._value.to(map_loc).eval()
    agent._loaded_episode = episode

def load_agents(player_config, base_dir, game):
    agents = []
    for pid, cfg in enumerate(player_config):
        kind = cfg["type"]
        version = cfg.get("version", "01")
        episode = cfg.get("episode")

        if kind == "ppo":
            info_dim = game.information_state_tensor_shape()[0]
            n_actions = game.num_distinct_actions()

            # Pfad-Basis
            train_dir = os.path.join(base_dir, f"models/ppo_model_{version}/train")

            # Episodenpfad vorbereiten (falls gegeben)
            seat_id_dim = 0
            policy_path_for_infer = None
            if episode is not None:
                ep_tag = _pad_episode(int(episode))
                prefix = os.path.join(train_dir, f"ppo_model_{version}_agent_p{pid}_ep{ep_tag}")
                policy_path_for_infer = f"{prefix}_policy.pt"
                if not os.path.exists(policy_path_for_infer):
                    raise FileNotFoundError(f"Policy-File nicht gefunden: {policy_path_for_infer}")
            else:
                # keinen Ep-Wert gegeben → nimm neueste Episode für diesen Player
                candidates = [fn for fn in os.listdir(train_dir) if fn.startswith(f"ppo_model_{version}_agent_p{pid}_ep") and fn.endswith("_policy.pt")]
                if not candidates:
                    raise FileNotFoundError(f"Kein Snapshot für PPO v{version} p{pid} in {train_dir} gefunden.")
                # höchste Episode wählen
                def _epnum(name):
                    # ..._ep0001234_policy.pt
                    s = name.split("_ep")[1].split("_policy.pt")[0]
                    return int(s)
                best = max(candidates, key=_epnum)
                policy_path_for_infer = os.path.join(train_dir, best)
                episode = _epnum(best)

            # Seat-ID aus Checkpoint ableiten
            seat_id_dim = _infer_seat_id_dim_from_checkpoint(policy_path_for_infer, info_dim)

            # Agent mit passendem Input bauen
            agent = ppo.PPOAgent(info_state_size=info_dim, num_actions=n_actions, seat_id_dim=seat_id_dim)

            # Dann das gewünschte Episode-Snapshot laden
            _load_ppo_episode(agent, base_dir, version, pid, int(episode))
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

# === Beschriftungs-Helfer ===
def get_action_labels_readable(game, use_numbers=False):
    dummy_state = game.new_initial_state()
    labels = []
    for aid in range(game.num_distinct_actions()):
        try:
            text = dummy_state.action_to_string(0, aid)
        except Exception:
            labels.append(f"Action {aid}")
            continue

        if "Pass" in text:
            labels.append("Pass")
            continue

        if "Single" in text:
            prefix = "1" if use_numbers else "S"
        elif "Pair" in text:
            prefix = "2" if use_numbers else "P"
        elif "Triple" in text:
            prefix = "3" if use_numbers else "T"
        elif "Quad" in text:
            prefix = "4" if use_numbers else "Q"
        else:
            prefix = "?"

        rank = text.split()[-1]
        labels.append(f"{prefix} {rank}")
    return labels

def make_action_id_range(num_actions, start_id=0, include_pass=True):
    """
    Liefert eine Liste von Action-IDs unter Beachtung MAX_ACTION_ID_TO_SHOW.
    include_pass=False -> beginnt mindestens bei 1.
    """
    max_id = num_actions - 1 if MAX_ACTION_ID_TO_SHOW is None else min(MAX_ACTION_ID_TO_SHOW, num_actions - 1)
    first = start_id
    if not include_pass:
        first = max(1, first)
    return list(range(first, max_id + 1))

# === Aktionswahl ===
def choose_policy_action(agent, state, player):
    legal = state.legal_actions(player)

    if isinstance(agent, ppo.PPOAgent):
        legal = state.legal_actions(player)
        obs = state.information_state_tensor(player)

        x = np.array(obs, dtype=np.float32)
        if getattr(agent, "_seat_id_dim", 0) > 0:
            seat_oh = np.zeros(agent._seat_id_dim, dtype=np.float32)
            # player-Index muss < seat_id_dim sein; bei 4-Spieler-Spielen passt das
            if 0 <= player < agent._seat_id_dim:
                seat_oh[player] = 1.0
            x = np.concatenate([x, seat_oh], axis=0)

        x_t = torch.tensor(x, dtype=torch.float32, device=agent.device).unsqueeze(0)
        with torch.no_grad():
            logits = agent._policy(x_t).squeeze(0)
            legal_mask = torch.zeros(agent._num_actions, dtype=torch.float32, device=agent.device)
            legal_mask[legal] = 1.0
            probs_t = ppo.masked_softmax(logits, legal_mask)
            probs = probs_t.detach().cpu().numpy()

        action = int(np.random.choice(len(probs), p=probs))
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

    elif callable(agent):
        action = agent(state)
        if action not in legal:
            action = np.random.choice(legal)
        return collections.namedtuple("AgentOutput", ["action"])(action=action)

    else:
        raise ValueError("Unbekannter Agententyp bei choose_policy_action.")

# === Evaluation starten ===
action_counts = defaultdict(lambda: defaultdict(int))              # alle Aktionen
first_action_counts = defaultdict(lambda: defaultdict(int))        # nur erste Aktion des Startspielers
agents = load_agents(PLAYER_CONFIG, base_dir, game)
returns_total = np.zeros(4)
start_counts = collections.defaultdict(int)
win_counts = collections.defaultdict(int)

for episode in range(1, NUM_EPISODES + 1):
    state = game.new_initial_state()
    start_pid = state.current_player()
    start_counts[start_pid] += 1
    first_action_done = False

    while not state.is_terminal():
        pid = state.current_player()
        agent = agents[pid]

        if callable(agent):
            action = agent(state)
        else:
            action = choose_policy_action(agent, state, pid).action

        # Erste Aktion des Startspielers im Spiel erfassen
        if not first_action_done and pid == start_pid:
            first_action_counts[pid][action] += 1
            first_action_done = True

        action_counts[pid][action] += 1
        state.apply_action(action)

    final_returns = state.returns()
    for i, ret in enumerate(final_returns):
        returns_total[i] += ret
    win_counts[np.argmax(final_returns)] += 1

    if episode % 1000 == 0:
        current_winrates = [100 * win_counts[i] / episode for i in range(4)]
        wr_str = " | ".join(
            f"P{i} ({PLAYER_CONFIG[i]['type']} v{PLAYER_CONFIG[i].get('version','')}" +
            (f", ep{int(PLAYER_CONFIG[i]['episode']):,}".replace(",", ".") if PLAYER_CONFIG[i].get('episode') is not None else "") +
            f"): {wr:.1f}%"
            for i, wr in enumerate(current_winrates)
        )
        print(f"✅ Episode {episode} abgeschlossen – Winrates: {wr_str}")

# === Ergebnisse speichern ===
summary_rows = []
for i in range(4):
    config = PLAYER_CONFIG[i]
    avg_ret = returns_total[i] / NUM_EPISODES
    winrate = 100 * win_counts[i] / NUM_EPISODES
    label = f"{config['type']} v{config.get('version','')}".strip()
    if 'episode' in config and config['episode'] is not None:
        label += f" ep{int(config['episode']):,}".replace(",", ".")  # 430.000

    row = {
        "macro_id": next_macro_num,
        "player": config["name"],
        "strategy": label,
        "type": config["type"],
        "version": config.get("version", ""),
        "episode": config.get("episode", ""),
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


def legend_inside(ax, loc="upper right", font_size=10):
    return ax.legend(
        loc=loc,
        frameon=True,
        framealpha=0.9,
        borderaxespad=0.5,
        fontsize=font_size
    )



# === Plots === #
if GENERATE_PLOTS:
    print(f"📁 Ergebnisse und Plots werden gespeichert in: {plot_dir}")

    num_actions = game.num_distinct_actions()

    # Labels vorbereiten (global umschaltbar)
    if USE_READABLE_LABELS:
        all_labels = get_action_labels_readable(game, use_numbers=READABLE_USE_NUMBERS)
    else:
        all_labels = [f"Action {i}" for i in range(num_actions)]

    # Deckgröße ermitteln (int)
    deck_size = int(game.get_parameters().get("deck_size", "64"))

    # Combo-Labels entsprechend deck_size erweitern
    if deck_size == 64:
        combo_labels = ["Single", "Pair", "Triple", "Quad"] + [f"{i}-of-a-kind" for i in range(5, 9)]
    else:
        combo_labels = ["Single", "Pair", "Triple", "Quad"]

    # === 00 - Aktionsverteilung als Tabelle (alle Aktionen; keine Limitierung) ===
    action_labels_table = [f"Action {i}" for i in range(num_actions)]  # für CSV neutral lassen
    action_table = pd.DataFrame(index=[f"Player {i}" for i in range(4)], columns=action_labels_table)
    for pid in range(4):
        total = sum(action_counts[pid].values())
        for aid in range(num_actions):
            count = action_counts[pid].get(aid, 0)
            percent = 100 * count / total if total > 0 else 0
            action_table.loc[f"Player {pid}", f"Action {aid}"] = f"{count} ({percent:.1f}%)"

    table_path = os.path.join(csv_dir, "00_action_distribution.csv")
    action_table.to_csv(table_path)
    print(f"📄 Tabelle gespeichert unter: {table_path}")

    # === 01 - First Action Distribution (nur erste Aktion des Startspielers) ===
    action_ids = make_action_id_range(num_actions, start_id=0, include_pass=True)
    action_labels = [all_labels[aid] for aid in action_ids]
    counts_first_per_action = {aid: [first_action_counts[pid].get(aid, 0) for pid in range(4)]
                               for aid in action_ids}
    x = np.arange(len(action_labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(min(24, max(12, len(action_labels) * 0.6)), 8))
    for pid in range(4):
        total_first = sum(first_action_counts[pid].values()) or 1
        counts = [100 * counts_first_per_action[aid][pid] / total_first for aid in action_ids]
        strat = PLAYER_CONFIG[pid]["type"]
        ver   = PLAYER_CONFIG[pid].get("version", "")
        epi   = PLAYER_CONFIG[pid].get("episode")
        epi_s = f", ep{int(epi):,}".replace(",", ".") if epi is not None else ""
        winrate = 100 * win_counts[pid] / NUM_EPISODES
        ax.bar(x + width * pid, counts, width, label=f"Player {pid} ({strat} v{ver}{epi_s}, {winrate:.1f}%)")
    ax.set_xlabel("Action (erste Aktion des Startspielers)")
    ax.set_ylabel("Relative Häufigkeit (%)")
    ax.set_title("01 - First Action Distribution (Startspieler)")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(action_labels, rotation=90)
    legend_inside(ax, font_size=12)  # optional: legend_inside(ax, loc="upper left")
    fig.tight_layout()
    plt.savefig(os.path.join(plot_dir, "01_first_action_distribution_startplayer.jpg"))

    # === 02 - Gesamte Aktionsverteilung (ohne Pass) ===
    action_ids_no_pass = make_action_id_range(num_actions, start_id=1, include_pass=False)
    non_pass_labels = [all_labels[aid] for aid in action_ids_no_pass]
    counts_per_action_no_pass = {aid: [action_counts[pid].get(aid, 0) for pid in range(4)]
                                 for aid in action_ids_no_pass}
    x = np.arange(len(non_pass_labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(min(24, max(12, len(non_pass_labels) * 0.6)), 8))
    for pid in range(4):
        total_actions = sum(action_counts[pid].values())
        pass_count = action_counts[pid].get(0, 0)
        total_non_pass = (total_actions - pass_count) or 1
        counts = [100 * counts_per_action_no_pass[aid][pid] / total_non_pass for aid in action_ids_no_pass]
        strat = PLAYER_CONFIG[pid]["type"]
        ver   = PLAYER_CONFIG[pid].get("version", "")
        epi   = PLAYER_CONFIG[pid].get("episode")
        epi_s = f", ep{int(epi):,}".replace(",", ".") if epi is not None else ""
        winrate = 100 * win_counts[pid] / NUM_EPISODES
        ax.bar(x + width * pid, counts, width, label=f"Player {pid} ({strat} v{ver}{epi_s}, {winrate:.1f}%)")
    ax.set_xlabel("Action (ohne Pass)")
    ax.set_ylabel("Relative Häufigkeit (%)")
    ax.set_title("02 - Action Counts per Player (No Pass)")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(non_pass_labels, rotation=90)
    legend_inside(ax, font_size=12)  # optional: legend_inside(ax, loc="upper left")
    fig.tight_layout()

    plt.savefig(os.path.join(plot_dir, "02_action_distribution_total_no_pass.jpg"))

    # === 03 - Gesamte Aktionsverteilung (inkl. Pass) ===
    action_ids_all = make_action_id_range(num_actions, start_id=0, include_pass=True)
    action_labels_all = [all_labels[aid] for aid in action_ids_all]
    counts_per_action = {aid: [action_counts[pid].get(aid, 0) for pid in range(4)] for aid in action_ids_all}
    x = np.arange(len(action_labels_all))
    width = 0.2
    fig, ax = plt.subplots(figsize=(min(24, max(12, len(action_labels_all) * 0.6)), 8))
    for pid in range(4):
        total_actions = sum(action_counts[pid].values()) or 1
        counts = [100 * counts_per_action[aid][pid] / total_actions for aid in action_ids_all]
        strat = PLAYER_CONFIG[pid]["type"]
        ver   = PLAYER_CONFIG[pid].get("version", "")
        epi   = PLAYER_CONFIG[pid].get("episode")
        epi_s = f", ep{int(epi):,}".replace(",", ".") if epi is not None else ""
        winrate = 100 * win_counts[pid] / NUM_EPISODES
        ax.bar(x + width * pid, counts, width, label=f"Player {pid} ({strat} v{ver}{epi_s}, {winrate:.1f}%)")
    ax.set_xlabel("Action")
    ax.set_ylabel("Relative Häufigkeit (%)")
    ax.set_title("03 - Action Counts per Player")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(action_labels_all, rotation=90)
    legend_inside(ax, font_size=12)  # optional: legend_inside(ax, loc="upper left")
    fig.tight_layout()

    plt.savefig(os.path.join(plot_dir, "03_action_distribution_total.jpg"))

    # === 04 - Pass vs. Play ===
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
    labels = []
    for i in range(4):
        strat = PLAYER_CONFIG[i]["type"]
        ver   = PLAYER_CONFIG[i].get("version", "")
        epi   = PLAYER_CONFIG[i].get("episode")
        epi_s = f", ep{int(epi):,}".replace(",", ".") if epi is not None else ""
        winrate = win_counts[i] / NUM_EPISODES
        labels.append(f"Player {i} ({strat} v{ver}{epi_s}, {winrate:.1%})")

    ax.bar(x - width / 2, play_counts, width, label="Play")
    ax.bar(x + width / 2, pass_counts, width, label="Pass")
    ax.set_ylabel("Anteil an allen Aktionen (%)")
    ax.set_xlabel("Spieler")
    ax.set_title("04 - Anteil von Pass vs. Spiel-Aktionen")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 100)
    legend_inside(ax)  # optional: legend_inside(ax, loc="upper left")
    fig.tight_layout()

    plt.savefig(os.path.join(plot_dir, "04_pass_vs_play.jpg"))

    # === 05 - Kombotyp-Anteile je Spieler ===
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
    ax.set_title("05 - Anteil gespielter Kombitypen pro Spieler")
    ax.set_xticks(x + width * (len(combo_labels)-1) / 2)
    ax.set_xticklabels([f"Player {i}" for i in range(4)])
    ax.set_ylim(0, 100)
    legend_inside(ax)  # optional: legend_inside(ax, loc="upper left")
    fig.tight_layout()
    plt.savefig(os.path.join(plot_dir, "05_combo_types_per_player.jpg"))

    # === 06+ - Detaillierte Kombitypenplots ===
    combo_plot_start_idx = 6  # verschobene Startnummer
    combo_plot_index = {ctype: f"{combo_plot_start_idx+i:02d}" for i, ctype in enumerate(combo_labels)}
    combo_actions = {ctype: [] for ctype in combo_labels}
    for aid, ctype in action_types.items():
        combo_actions[ctype].append(aid)

    for combo, aids in combo_actions.items():
        if not aids:
            continue
        # wende MAX_ACTION_ID_TO_SHOW auch hier an (falls Labels/IDs begrenzt werden sollen)
        aids = [aid for aid in aids if (MAX_ACTION_ID_TO_SHOW is None or aid <= MAX_ACTION_ID_TO_SHOW)]
        if not aids:
            continue

        labels = [all_labels[aid] for aid in aids]
        x = np.arange(len(labels))
        width = 0.2
        fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.6), 6))
        max_height = 0
        for pid in range(4):
            total_actions = sum(action_counts[pid].values()) or 1
            counts = [100 * action_counts[pid].get(aid, 0) / total_actions for aid in aids]
            max_height = max(max_height, max(counts, default=0))
            strat = PLAYER_CONFIG[pid]["type"]
            ver   = PLAYER_CONFIG[pid].get("version", "")
            epi   = PLAYER_CONFIG[pid].get("episode")
            epi_s = f", ep{int(epi):,}".replace(",", ".") if epi is not None else ""
            winrate = 100 * win_counts[pid] / NUM_EPISODES
            ax.bar(x + width * pid, counts, width, label=f"Player {pid} ({strat} v{ver}{epi_s}, {winrate:.1f}%)")

        ax.set_xlabel(f"{combo}-Actions")
        ax.set_ylabel("Relative Häufigkeit (%)")
        ax.set_title(f"{combo_plot_index[combo]} - {combo}-Actions pro Spieler")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(labels, rotation=90)

        ylim_top = max(1, math.ceil(max_height * 1.2 / 5) * 5)
        ax.set_ylim(0, ylim_top)

        legend_inside(ax)  # optional: legend_inside(ax, loc="upper left")
        fig.tight_layout()

        filename = f"{combo_plot_index[combo]}_combo_{combo.lower().replace('-','_')}_detailed.jpg"
        plt.savefig(os.path.join(plot_dir, filename))

    print("✅ Alle Plots erfolgreich gespeichert!")
