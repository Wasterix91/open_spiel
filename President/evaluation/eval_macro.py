# evaluation/eval_macro.py
# -*- coding: utf-8 -*-
import os, re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections

import pyspiel
import torch

from agents import ppo_agent as ppo
from agents import dqn_agent as dqn
from collections import defaultdict
from utils.strategies import STRATS  

from utils.load_save_a1_ppo import load_checkpoint_ppo
from utils.load_save_a2_dqn import load_checkpoint_dqn

# ===================== Konfiguration ===================== #
NUM_EPISODES = 10_000
DECK = "64",  # "12" | "16" | "20" | "24" | "32" | "52" | "64"
# Beispiel: PPO vs 3x Heuristik
PLAYER_CONFIG = [
    {"name": "P0", "type": "max_combo"},
    {"name": "P1", "type": "max_combo"},
    {"name": "P2", "type": "dqn", "family": "k1a2", "version": "38", "episode": 75_000, "from_pid": 0},
    {"name": "P3", "type": "max_combo"},
]

""" PLAYER_CONFIG = [
    {"name": "P0", "type": "ppo", "family": "k3a1", "version": "05", "episode": 20_000, "from_pid": 0},
    {"name": "P1", "type": "ppo", "family": "k1a1", "version": "57", "episode": 200, "from_pid": 0},
    {"name": "P2", "type": "ppo", "family": "k3a1", "version": "05", "episode": 20_000, "from_pid": 0},
    {"name": "P3", "type": "ppo", "family": "k1a1", "version": "57", "episode": 200, "from_pid": 0},
] """

"""
Weitere Beispiele:

# Vier PPO (k2a1) gegeneinander (je eigener Seat-Checkpoint)
PLAYER_CONFIG = [
    {"name": "P0", "type": "ppo", "family": "k2a1", "version": "02", "episode": 20000, "from_pid": 0},
    {"name": "P1", "type": "ppo", "family": "k2a1", "version": "02", "episode": 20000, "from_pid": 1},
    {"name": "P2", "type": "ppo", "family": "k2a1", "version": "02", "episode": 20000, "from_pid": 2},
    {"name": "P3", "type": "ppo", "family": "k2a1", "version": "02", "episode": 20000, "from_pid": 3},
]

# K4 (shared policy) – gleiche Policy auf allen Seats (wir laden p0 und setzen from_pid)
PLAYER_CONFIG = [
    {"name": "P0", "type": "ppo", "family": "k4a1", "version": "01", "episode": 12000, "from_pid": 0},
    {"name": "P1", "type": "ppo", "family": "k4a1", "version": "01", "episode": 12000, "from_pid": 0},
    {"name": "P2", "type": "ppo", "family": "k4a1", "version": "01", "episode": 12000, "from_pid": 0},
    {"name": "P3", "type": "ppo", "family": "k4a1", "version": "01", "episode": 12000, "from_pid": 0},
]

# DQN vs Heuristiken
PLAYER_CONFIG = [
    {"name":"P0","type":"dqn","family":"k1a2","version":"03","episode":"150_000"},
    {"name": "P1", "type": "max_combo"},
    {"name": "P2", "type": "random2"},
    {"name": "P3", "type": "single_only"},
]
"""

GENERATE_PLOTS = True
EVAL_OUTPUT = True

# Plot-Flags
USE_READABLE_LABELS = True       # True = "Pass", "S 7", "P J", ... ; False = "Action X"
READABLE_USE_NUMBERS = False     # True = 1/2/3/4 statt S/P/T/Q
MAX_ACTION_ID_TO_SHOW = 32       # None = alle; sonst max. Action-ID auf der X-Achse


# ===================== Spielinitialisierung ===================== #
GAME_PARAMS = {
    "deck_size": DECK[0],
    "shuffle_cards": True,
    "single_card_mode": False,
    "num_players": 4
}
game = pyspiel.load_game("president", GAME_PARAMS)

NUM_PLAYERS = game.num_players()
INFO_DIM = game.information_state_tensor_shape()[0]
OBS_DIM  = game.observation_tensor_shape()[0]
NUM_ACTIONS = game.num_distinct_actions()

# Speicher-Root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== Lesbare Labels (nur kosmetisch) =====
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
            m = re.search(r"(\d+)-of-a-kind", text)
            prefix = m.group(1) if m else "?"
        rank = text.split()[-1]
        labels.append(f"{prefix} {rank}")
    return labels

def make_action_id_range(num_actions, start_id=0, include_pass=True):
    max_id = num_actions - 1 if MAX_ACTION_ID_TO_SHOW is None else min(MAX_ACTION_ID_TO_SHOW, num_actions - 1)
    first = start_id
    if not include_pass:
        first = max(1, first)
    return list(range(first, max_id + 1))

# ===================== Utils: Files & Episodes ===================== #
MODELS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

def _fatal(msg, tried=None):
    lines = [f"[FATAL] {msg}"]
    if tried:
        lines.append("  Versucht:")
        for t in tried[:30]:
            lines.append(f"    {t}")
        if len(tried) > 30:
            lines.append("    ...")
    raise SystemExit("\n".join(lines))

def _norm_episode(ep):
    if isinstance(ep, int):
        return ep
    if isinstance(ep, str):
        s = ep.replace("_", "").strip()
        if s.isdigit():
            return int(s)
    _fatal(f"Episode muss explizit gesetzt sein (int oder numerischer String), erhalten: {ep!r}")

def _ppo_expected_stem(family: str, version: str, seat_on_disk: int, episode: int):
    base_dir = os.path.join(MODELS_ROOT, family, f"model_{version}", "models")
    stem = os.path.join(base_dir, f"{family}_model_{version}_agent_p{seat_on_disk}_ep{episode:07d}")
    pol, val = stem + "_policy.pt", stem + "_value.pt"
    if not (os.path.exists(pol) and os.path.exists(val)):
        _fatal(
            f"PPO-Checkpoint fehlt: family={family}, version={version}, seat={seat_on_disk}, episode={episode}",
            tried=[pol, val],
        )
    return stem

def _dqn_expected_stem(family: str, version: str, seat_on_disk: int, episode: int):
    base_dir = os.path.join(MODELS_ROOT, family, f"model_{version}", "models")
    stem = os.path.join(base_dir, f"{family}_model_{version}_agent_p{seat_on_disk}_ep{episode:07d}")
    # neue Trainings-Namen
    qnet = stem + "_qnet.pt"
    tgt  = stem + "_tgt.pt"
    # legacy-Variante (falls vorhanden)
    legacy_q = stem + "_q.pt"

    # Mit neuem Loader reicht _qnet.pt (Target optional) ODER Legacy _q.pt
    if not (os.path.exists(qnet) or os.path.exists(legacy_q)):
        _fatal(
            f"DQN-Checkpoint fehlt: family={family}, version={version}, seat={seat_on_disk}, episode={episode}",
            tried=[qnet, tgt, legacy_q],
        )
    return stem



def _alias_dqn_attrs(agent):
    # Adapter: Richte Aliase ein, damit die Loader (q_net/target_net) mit deinem Agent (q_network/target_network) sprechen.
    if not hasattr(agent, "q_net") and hasattr(agent, "q_network"):
        agent.q_net = agent.q_network
    if not hasattr(agent, "target_net") and hasattr(agent, "target_network"):
        agent.target_net = agent.target_network
    return agent

def _load_ppo_agent(info_dim, num_actions, seat_id_dim, *, family, version, episode, from_pid, device="cpu"):
    ep = _norm_episode(episode)
    stem = _ppo_expected_stem(family, version, from_pid, ep)

    ag = ppo.PPOAgent(info_state_size=info_dim, num_actions=num_actions, seat_id_dim=seat_id_dim, device=device)
    weights_dir = os.path.dirname(stem)
    tag = os.path.basename(stem)
    try:
        load_checkpoint_ppo(ag, weights_dir, tag)  # nutzt deine Trainings-Loader
        ag._policy.eval(); ag._value.eval()
        return ag
    except Exception as e:
        _fatal("Fehler beim Laden via load_checkpoint_ppo(...).", tried=[f"{stem}: {e}"])

def _load_dqn_agent(obs_dim, num_actions, *, family, version, episode, from_pid, num_players, device="cpu"):
    ep = _norm_episode(episode)
    stem = _dqn_expected_stem(family, version, from_pid, ep)

    tried = []
    weights_dir = os.path.dirname(stem)
    tag = os.path.basename(stem)

    for state_size in (obs_dim, obs_dim + num_players):
        ag = dqn.DQNAgent(state_size=state_size, num_actions=num_actions, device=device)
        _alias_dqn_attrs(ag)

        try:
            # Neuer Loader kümmert sich um _qnet/_tgt sowie Legacy _q/_target
            load_checkpoint_dqn(ag, weights_dir, tag)
            if hasattr(ag, "epsilon"):
                ag.epsilon = 0.0
            return ag
        except Exception as e:
            tried.append(f"{stem} (state_size={state_size}): {e}")

    _fatal("Fehler beim Laden von DQN-Gewichten.", tried=tried)



def load_agents(player_config, game):
    agents = []
    info_dim    = game.information_state_tensor_shape()[0]
    obs_dim     = game.observation_tensor_shape()[0]
    num_actions = game.num_distinct_actions()
    num_players = game.num_players()

    for pid, cfg in enumerate(player_config):
        kind = cfg["type"]

        if kind == "ppo":
            # ZWINGEND: family, version, episode
            if not all(k in cfg for k in ("family", "version", "episode")):
                _fatal(f"PPO-Spieler P{pid}: 'family', 'version' und 'episode' sind Pflicht. Erhalten: {cfg}")
            ag = None
            # 1) mit Seat-One-Hot laden
            try:
                ag = _load_ppo_agent(info_dim, num_actions, num_players,
                                     family=cfg["family"], version=cfg["version"], episode=cfg["episode"],
                                     from_pid=cfg.get("from_pid", pid))
            except SystemExit:
                # 2) ohne Seat-One-Hot
                ag = _load_ppo_agent(info_dim, num_actions, 0,
                                     family=cfg["family"], version=cfg["version"], episode=cfg["episode"],
                                     from_pid=cfg.get("from_pid", pid))
            agents.append(ag)
            continue

        if kind == "dqn":
            if not all(k in cfg for k in ("family", "version", "episode")):
                _fatal(f"DQN-Spieler P{pid}: 'family', 'version' und 'episode' sind Pflicht. Erhalten: {cfg}")
            ag = _load_dqn_agent(obs_dim, num_actions,
                                 family=cfg["family"], version=cfg["version"], episode=cfg["episode"],
                                 from_pid=cfg.get("from_pid", pid), num_players=num_players)
            agents.append(ag)
            continue

        if kind in STRATS:
            agents.append(STRATS[kind])
            continue

        _fatal(f"Unbekannter Agententyp bei P{pid}: {kind!r}")
    return agents

# ===================== Eval-Vorwärtswege ===================== #
def _masked_softmax_numpy(logits, legal):
    logits = np.asarray(logits, dtype=np.float32)
    if len(legal) == 0:
        # rare: keine legalen Aktionen -> uniform über alle
        return np.ones_like(logits) / len(logits)
    mask = np.full_like(logits, -np.inf, dtype=np.float32)
    mask[list(legal)] = logits[list(legal)]
    m = np.max(mask[list(legal)])
    ex = np.exp(mask - m)
    ex[~np.isfinite(ex)] = 0.0
    s = ex.sum()
    if s <= 0 or not np.isfinite(s):
        p = np.zeros_like(logits, dtype=np.float32)
        p[list(legal)] = 1.0 / len(legal)
        return p
    return ex / s

def _forward_policy_with_make_input(agent: ppo.PPOAgent, info_state_1d: np.ndarray, seat_id: int, legal):
    seat_oh = np.zeros(NUM_PLAYERS, dtype=np.float32); seat_oh[seat_id] = 1.0
    x = agent._make_input(info_state_1d, seat_one_hot=seat_oh)  
    with torch.no_grad():
        logits = agent._policy(x).detach().cpu().numpy()
    probs = _masked_softmax_numpy(logits, legal)    
    return int(np.argmax(probs))

def _forward_policy_autopad(policy_net: torch.nn.Module, obs_1d: np.ndarray, device):
    # robust gegen kleinere/größere Eingaben
    x = np.asarray(obs_1d, dtype=np.float32)
    try:
        in_features = policy_net.net[0].in_features
    except Exception:
        in_features = x.shape[0]
    if x.shape[0] < in_features:
        x = np.concatenate([x, np.zeros(in_features - x.shape[0], dtype=np.float32)], axis=0)
    elif x.shape[0] > in_features:
        x = x[:in_features]
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        out = policy_net(xt).squeeze(0).detach().cpu().numpy()
    return out

def choose_policy_action(agent, state, player):
    legal = state.legal_actions(player)

    # PPO-Agent
    if isinstance(agent, ppo.PPOAgent):
        info_vec = np.array(state.information_state_tensor(player), dtype=np.float32)

        # bevorzuge _make_input (Seat-OneHot korrekt)
        if hasattr(agent, "_make_input"):
            return collections.namedtuple("AgentOutput", ["action"])(action=_forward_policy_with_make_input(
                agent, info_vec, seat_id=player, legal=legal
            ))

        # Fallback: Autopad + masked softmax
        device = getattr(agent, "device", "cpu")
        logits = _forward_policy_autopad(agent._policy, info_vec, device)
        probs = _masked_softmax_numpy(logits, legal)
        action = int(np.argmax(probs))
        return collections.namedtuple("AgentOutput", ["action"])(action=action)

    # DQN-Agent (greedy)
    if isinstance(agent, dqn.DQNAgent):
        obs_vec = np.array(state.observation_tensor(player), dtype=np.float32)
        # optional: autopad auf q_network.in_features (inkl. Seat-OneHot, falls benötigt)
        try:
            in_features = agent.q_network.net[0].in_features
        except Exception:
            in_features = obs_vec.shape[0]
        if obs_vec.shape[0] < in_features:
            extra = in_features - obs_vec.shape[0]
            if extra == NUM_PLAYERS:
                seat_oh = np.zeros(NUM_PLAYERS, dtype=np.float32); seat_oh[player] = 1.0
                obs_vec = np.concatenate([obs_vec, seat_oh], axis=0)
            else:
                obs_vec = np.concatenate([obs_vec, np.zeros(extra, dtype=np.float32)], axis=0)
        elif obs_vec.shape[0] > in_features:
            obs_vec = obs_vec[:in_features]

        old_eps = getattr(agent, "epsilon", 0.0)
        agent.epsilon = 0.0
        try:
            action = int(agent.select_action(obs_vec, legal))
        finally:
            agent.epsilon = old_eps
        return collections.namedtuple("AgentOutput", ["action"])(action=action)

    # Heuristik (callable)
    if callable(agent):
        action = agent(state)
        if action not in legal:
            action = np.random.choice(legal)
        return collections.namedtuple("AgentOutput", ["action"])(action=action)

    raise ValueError("Unbekannter Agententyp bei choose_policy_action.")

# ===================== Evaluation ===================== #
def main():
    action_counts = defaultdict(lambda: defaultdict(int))       # alle Aktionen
    first_action_counts = defaultdict(lambda: defaultdict(int)) # nur erste Aktion des Startspielers
    agents = load_agents(PLAYER_CONFIG, game)

    # --- Verzeichnisse jetzt (erst nach erfolgreichem Laden) anlegen ---
    EVAL_MACRO_ROOT = os.path.join(BASE_DIR, "eval_macro")
    existing_macro_dirs = sorted([d for d in os.listdir(EVAL_MACRO_ROOT)] if os.path.isdir(EVAL_MACRO_ROOT) else [])
    existing_macro_dirs = [d for d in existing_macro_dirs if d.startswith("eval_macro_")]
    next_macro_num = int(existing_macro_dirs[-1].split("_")[-1]) + 1 if existing_macro_dirs else 1

    MACRO_DIR = os.path.join(EVAL_MACRO_ROOT, f"eval_macro_{next_macro_num:02d}")
    os.makedirs(MACRO_DIR, exist_ok=True)
    CSV_DIR = os.path.join(MACRO_DIR, "csv");  os.makedirs(CSV_DIR, exist_ok=True)
    PLOT_DIR = os.path.join(MACRO_DIR, "plots"); os.makedirs(PLOT_DIR, exist_ok=True)

    # Konfiguration abspeichern (erst jetzt)
    pd.DataFrame(PLAYER_CONFIG).to_csv(os.path.join(MACRO_DIR, "player_config.csv"), index=False)

    returns_total = np.zeros(NUM_PLAYERS, dtype=np.float64)
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

            if not first_action_done and pid == start_pid:
                first_action_counts[pid][action] += 1
                first_action_done = True

            action_counts[pid][action] += 1
            state.apply_action(action)

        final_returns = state.returns()
        for i, ret in enumerate(final_returns):
            returns_total[i] += ret
        win_counts[int(np.argmax(final_returns))] += 1

        if episode % 1000 == 0 and EVAL_OUTPUT:
            current_winrates = [100 * win_counts[i] / episode for i in range(NUM_PLAYERS)]
            def _fmt(i):
                cfg = PLAYER_CONFIG[i]
                ver = cfg.get('version','')
                epi = cfg.get('episode')
                epi_s = f", ep{int(str(epi).replace('_','')):,}".replace(",", ".") if epi is not None else ""
                return f"P{i} ({cfg['type']} v{ver}{epi_s})"
            wr_str = " | ".join(f"{_fmt(i)}: {wr:.1f}%" for i, wr in enumerate(current_winrates))
            print(f"✅ Episode {episode} abgeschlossen – Winrates: {wr_str}")

    # === Ergebnisse speichern ===
    summary_rows = []
    for i in range(NUM_PLAYERS):
        cfg = PLAYER_CONFIG[i]
        avg_ret = returns_total[i] / NUM_EPISODES
        winrate = 100 * win_counts[i] / NUM_EPISODES
        label = f"{cfg['type']} v{cfg.get('version','')}".strip()
        epi = cfg.get("episode", "")
        if epi not in ("", None):
            epi_int = _norm_episode(epi)
            label += f" ep{epi_int:,}".replace(",", ".")
        row = {
            "macro_id": next_macro_num,
            "player": cfg.get("name", f"P{i}"),
            "strategy": label,
            "type": cfg["type"],
            "family": cfg.get("family",""),
            "version": cfg.get("version", ""),
            "episode": epi if epi is not None else "",
            "avg_return": round(float(avg_ret), 2),
            "win_rate_percent": round(float(winrate), 2),
            "num_wins": int(win_counts[i]),
            "num_starts": int(start_counts[i]),
        }
        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(CSV_DIR, "evaluation_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"📄 Evaluationsergebnisse gespeichert unter: {summary_path}")

    if not GENERATE_PLOTS:
        return

    print(f"📁 Ergebnisse und Plots werden gespeichert in: {PLOT_DIR}")
    all_labels = get_action_labels_readable(game, use_numbers=READABLE_USE_NUMBERS) if USE_READABLE_LABELS \
                 else [f"Action {i}" for i in range(NUM_ACTIONS)]

    # 00 - Aktionsverteilung als Tabelle
    action_labels_table = [f"Action {i}" for i in range(NUM_ACTIONS)]
    action_table = pd.DataFrame(index=[f"Player {i}" for i in range(NUM_PLAYERS)], columns=action_labels_table)
    for pid in range(NUM_PLAYERS):
        total = sum(action_counts[pid].values())
        for aid in range(NUM_ACTIONS):
            count = action_counts[pid].get(aid, 0)
            percent = 100 * count / total if total > 0 else 0
            action_table.loc[f"Player {pid}", f"Action {aid}"] = f"{count} ({percent:.1f}%)"

    table_path = os.path.join(CSV_DIR, "00_action_distribution.csv")
    action_table.to_csv(table_path)
    print(f"📄 Tabelle gespeichert unter: {table_path}")

    def legend_inside(ax, loc="upper right", font_size=10):
        return ax.legend(loc=loc, frameon=True, framealpha=0.9, borderaxespad=0.5, fontsize=font_size)

    # 01 - First Action Distribution (Startspieler)
    action_ids = make_action_id_range(NUM_ACTIONS, start_id=0, include_pass=True)
    action_labels = [all_labels[aid] for aid in action_ids]
    counts_first_per_action = {aid: [first_action_counts[pid].get(aid, 0) for pid in range(NUM_PLAYERS)] for aid in action_ids}
    x = np.arange(len(action_labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(min(24, max(12, len(action_labels) * 0.6)), 8))
    for pid in range(NUM_PLAYERS):
        total_first = sum(first_action_counts[pid].values()) or 1
        counts = [100 * counts_first_per_action[aid][pid] / total_first for aid in action_ids]
        cfg = PLAYER_CONFIG[pid]
        ver = cfg.get("version","")
        epi = cfg.get("episode")
        epi_s = f", ep{_norm_episode(epi):,}".replace(",", ".") if epi is not None else ""
        winrate = 100 * win_counts[pid] / NUM_EPISODES
        ax.bar(x + width * pid, counts, width, label=f"P{pid} ({cfg['type']} v{ver}{epi_s}, {winrate:.1f}%)")
    ax.set_xlabel("Action (erste Aktion des Startspielers)")
    ax.set_ylabel("Relative Häufigkeit (%)")
    ax.set_title("01 - First Action Distribution (Startspieler)")
    ax.set_xticks(x + width * (NUM_PLAYERS-1)/2)
    ax.set_xticklabels(action_labels, rotation=90)
    legend_inside(ax, font_size=12)
    fig.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "01_first_action_distribution_startplayer.jpg"))

    # 02 - Gesamte Aktionsverteilung (ohne Pass)
    action_ids_no_pass = make_action_id_range(NUM_ACTIONS, start_id=1, include_pass=False)
    non_pass_labels = [all_labels[aid] for aid in action_ids_no_pass]
    counts_per_action_no_pass = {aid: [action_counts[pid].get(aid, 0) for pid in range(NUM_PLAYERS)] for aid in action_ids_no_pass}
    x = np.arange(len(non_pass_labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(min(24, max(12, len(non_pass_labels) * 0.6)), 8))
    for pid in range(NUM_PLAYERS):
        total_actions = sum(action_counts[pid].values())
        pass_count = action_counts[pid].get(0, 0)
        total_non_pass = (total_actions - pass_count) or 1
        counts = [100 * counts_per_action_no_pass[aid][pid] / total_non_pass for aid in action_ids_no_pass]
        cfg = PLAYER_CONFIG[pid]
        ver = cfg.get("version",""); epi = cfg.get("episode")
        epi_s = f", ep{_norm_episode(epi):,}".replace(",", ".") if epi is not None else ""
        winrate = 100 * win_counts[pid] / NUM_EPISODES
        ax.bar(x + width * pid, counts, width, label=f"P{pid} ({cfg['type']} v{ver}{epi_s}, {winrate:.1f}%)")
    ax.set_xlabel("Action (ohne Pass)")
    ax.set_ylabel("Relative Häufigkeit (%)")
    ax.set_title("02 - Action Counts per Player (No Pass)")
    ax.set_xticks(x + width * (NUM_PLAYERS-1)/2)
    ax.set_xticklabels(non_pass_labels, rotation=90)
    legend_inside(ax, font_size=12)
    fig.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "02_action_distribution_total_no_pass.jpg"))

    # 03 - Gesamte Aktionsverteilung (inkl. Pass)
    action_ids_all = make_action_id_range(NUM_ACTIONS, start_id=0, include_pass=True)
    action_labels_all = [all_labels[aid] for aid in action_ids_all]
    counts_per_action = {aid: [action_counts[pid].get(aid, 0) for pid in range(NUM_PLAYERS)] for aid in action_ids_all}
    x = np.arange(len(action_labels_all))
    width = 0.2
    fig, ax = plt.subplots(figsize=(min(24, max(12, len(action_labels_all) * 0.6)), 8))
    for pid in range(NUM_PLAYERS):
        total_actions = sum(action_counts[pid].values()) or 1
        counts = [100 * counts_per_action[aid][pid] / total_actions for aid in action_ids_all]
        cfg = PLAYER_CONFIG[pid]
        ver = cfg.get("version",""); epi = cfg.get("episode")
        epi_s = f", ep{_norm_episode(epi):,}".replace(",", ".") if epi is not None else ""
        winrate = 100 * win_counts[pid] / NUM_EPISODES
        ax.bar(x + width * pid, counts, width, label=f"P{pid} ({cfg['type']} v{ver}{epi_s}, {winrate:.1f}%)")
    ax.set_xlabel("Action")
    ax.set_ylabel("Relative Häufigkeit (%)")
    ax.set_title("03 - Action Counts per Player")
    ax.set_xticks(x + width * (NUM_PLAYERS-1)/2)
    ax.set_xticklabels(action_labels_all, rotation=90)
    legend_inside(ax, font_size=12)
    fig.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "03_action_distribution_total.jpg"))

    # 04 - Pass vs. Play
    pass_stats = {pid: {"Pass": 0, "Play": 0} for pid in range(NUM_PLAYERS)}
    dummy_state = game.new_initial_state()
    action_labels_map = {}
    for aid in range(NUM_ACTIONS):
        try:
            label = dummy_state.action_to_string(0, aid)
            action_labels_map[aid] = "Pass" if "Pass" in label else "Play"
        except:
            continue

    for pid in range(NUM_PLAYERS):
        for aid, count in action_counts[pid].items():
            label = action_labels_map.get(aid, "Play")
            pass_stats[pid][label] += count

    play_counts = []
    pass_counts = []
    for pid in range(NUM_PLAYERS):
        total = sum(pass_stats[pid].values()) or 1
        play_counts.append(100 * pass_stats[pid]["Play"] / total)
        pass_counts.append(100 * pass_stats[pid]["Pass"] / total)

    x = np.arange(NUM_PLAYERS)
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = []
    for i in range(NUM_PLAYERS):
        cfg = PLAYER_CONFIG[i]
        ver = cfg.get("version",""); epi = cfg.get("episode")
        epi_s = f", ep{_norm_episode(epi):,}".replace(",", ".") if epi is not None else ""
        winrate = win_counts[i] / NUM_EPISODES
        labels.append(f"P{i} ({cfg['type']} v{ver}{epi_s}, {winrate:.1%})")

    ax.bar(x - width / 2, play_counts, width, label="Play")
    ax.bar(x + width / 2, pass_counts, width, label="Pass")
    ax.set_ylabel("Anteil an allen Aktionen (%)")
    ax.set_xlabel("Spieler")
    ax.set_title("04 - Anteil von Pass vs. Spiel-Aktionen")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 100)
    legend_inside(ax)
    fig.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "04_pass_vs_play.jpg"))

    # 05 - Kombotyp-Anteile je Spieler
    # (wir leiten die Kombos wie gehabt aus den Action-Strings ab)
    combo_labels = ["Single", "Pair", "Triple", "Quad", "5-of-a-kind", "6-of-a-kind", "7-of-a-kind", "8-of-a-kind"]
    action_types = {}
    for aid in range(NUM_ACTIONS):
        try:
            label = dummy_state.action_to_string(0, aid)
            for ctype in combo_labels:
                if ctype in label:
                    action_types[aid] = ctype
                    break
        except:
            continue

    combo_totals = {pid: {ctype: 0 for ctype in combo_labels} for pid in range(NUM_PLAYERS)}
    for pid in range(NUM_PLAYERS):
        for aid, count in action_counts[pid].items():
            combo = action_types.get(aid)
            if combo:
                combo_totals[pid][combo] += count

    x = np.arange(NUM_PLAYERS)
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, combo in enumerate(combo_labels):
        counts = []
        for pid in range(NUM_PLAYERS):
            total = sum(combo_totals[pid].values()) or 1
            percent = 100 * combo_totals[pid][combo] / total
            counts.append(percent)
        ax.bar(x + i * width, counts, width, label=combo)

    ax.set_xlabel("Spieler")
    ax.set_ylabel("Anteil an allen Aktionen (%)")
    ax.set_title("05 - Anteil gespielter Kombitypen pro Spieler")
    ax.set_xticks(x + width * (len(combo_labels)-1) / 2)
    ax.set_xticklabels([f"P{i}" for i in range(NUM_PLAYERS)])
    ax.set_ylim(0, 100)
    legend_inside(ax)
    fig.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "05_combo_types_per_player.jpg"))

    # 06+ - Detaillierte Kombitypen-Plots
    combo_plot_start_idx = 6
    combo_plot_index = {ctype: f"{combo_plot_start_idx+i:02d}" for i, ctype in enumerate(combo_labels)}
    combo_actions = {ctype: [] for ctype in combo_labels}
    for aid, ctype in action_types.items():
        combo_actions[ctype].append(aid)

    for combo, aids in combo_actions.items():
        if not aids:
            continue
        aids = [aid for aid in aids if (MAX_ACTION_ID_TO_SHOW is None or aid <= MAX_ACTION_ID_TO_SHOW)]
        if not aids:
            continue

        labels = [all_labels[aid] for aid in aids]
        x = np.arange(len(labels))
        width = 0.2
        fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.6), 6))
        max_height = 0
        for pid in range(NUM_PLAYERS):
            total_actions = sum(action_counts[pid].values()) or 1
            counts = [100 * action_counts[pid].get(aid, 0) / total_actions for aid in aids]
            max_height = max(max_height, max(counts, default=0))
            cfg = PLAYER_CONFIG[pid]; ver = cfg.get("version",""); epi = cfg.get("episode")
            epi_s = f", ep{_norm_episode(epi):,}".replace(",", ".") if epi is not None else ""
            winrate = 100 * win_counts[pid] / NUM_EPISODES
            ax.bar(x + width * pid, counts, width, label=f"P{pid} ({cfg['type']} v{ver}{epi_s}, {winrate:.1f}%)")

        ax.set_xlabel(f"{combo}-Actions")
        ax.set_ylabel("Relative Häufigkeit (%)")
        ax.set_title(f"{combo_plot_index[combo]} - {combo}-Actions pro Spieler")
        ax.set_xticks(x + width * (NUM_PLAYERS-1)/2)
        ax.set_xticklabels(labels, rotation=90)

        ylim_top = max(1, math.ceil(max_height * 1.2 / 5) * 5)
        ax.set_ylim(0, ylim_top)

        legend_inside(ax)
        fig.tight_layout()
        filename = f"{combo_plot_index[combo]}_combo_{combo.lower().replace('-','_')}_detailed.jpg"
        plt.savefig(os.path.join(PLOT_DIR, filename))

    print("✅ Alle Plots erfolgreich gespeichert!")

if __name__ == "__main__":
    main()
