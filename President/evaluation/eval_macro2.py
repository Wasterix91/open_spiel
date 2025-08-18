# evaluation/eval_macro.py
import os
import re
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections

import pyspiel
import torch

# Agent-Imports (an deine Struktur angepasst)
from agents import ppo_agent as ppo
from agents import dqn_agent as dqn
from collections import defaultdict
from utils import STRATS


# ===================== Konfiguration ===================== #
NUM_EPISODES = 10_000

# Beispiel: PPO vs 3x Heuristik
PLAYER_CONFIG = [
    {"name":"P0","type":"dqn","family":"k1a2","version":"03","episode":"150_000"}, 
    {"name": "Player1", "type": "max_combo"},
    {"name": "Player2", "type": "max_combo"},
    {"name": "Player3", "type": "max_combo"},
]

# Heuristiken: random_action, random2, single_only, max_combo, max_combo2, aggressive, smart

"""
PLAYER_CONFIG = [
    {"name":"P0", "type":"ppo", "version":"03", "episode":10000},                 # echte p0-Gewichte
    {"name":"P1", "type":"ppo", "version":"03", "episode":10000, "from_pid":0},  # nutzt p0-Dateien
    {"name":"P2", "type":"dqn", "version":"01", "episode":None},
    {"name":"P3", "type":"max_combo"},
]

"""

GENERATE_PLOTS = True
EVAL_OUTPUT = True

# Plot-Flags
USE_READABLE_LABELS = True       # True = "Pass", "S 7", "P J", ... ; False = "Action X"
READABLE_USE_NUMBERS = False     # True = 1/2/3/4 statt S/P/T/Q
MAX_ACTION_ID_TO_SHOW = 32       # None = alle; sonst max. Action-ID auf der X-Achse

# Speicher-Root (unter evaluation/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_MACRO_ROOT = os.path.join(BASE_DIR, "eval_macro")
os.makedirs(EVAL_MACRO_ROOT, exist_ok=True)

existing_macro_dirs = sorted([d for d in os.listdir(EVAL_MACRO_ROOT) if d.startswith("eval_macro_")])
next_macro_num = int(existing_macro_dirs[-1].split("_")[-1]) + 1 if existing_macro_dirs else 1
MACRO_DIR = os.path.join(EVAL_MACRO_ROOT, f"eval_macro_{next_macro_num:02d}")
os.makedirs(MACRO_DIR, exist_ok=True)

CSV_DIR = os.path.join(MACRO_DIR, "csv")
PLOT_DIR = os.path.join(MACRO_DIR, "plots")
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Konfiguration abspeichern (wie früher player_config.csv)
pd.DataFrame(PLAYER_CONFIG).to_csv(os.path.join(MACRO_DIR, "player_config.csv"), index=False)

# ===================== Spielinitialisierung ===================== #
game = pyspiel.load_game("president", {
    "deck_size": "64",
    "shuffle_cards": True,
    "single_card_mode": False,
    "num_players": 4
})

NUM_PLAYERS = game.num_players()
INFO_DIM = game.information_state_tensor_shape()[0]
OBS_DIM  = game.observation_tensor_shape()[0]


RANKS = ["7", "8", "9", "10", "J", "Q", "K", "A"]
RANK_TO_NUM = {rank: i for i, rank in enumerate(RANKS)}

# ===================== Heuristiken ===================== #
def parse_combo_size(text):
    if "Single" in text: return 1
    if "Pair" in text: return 2
    if "Triple" in text: return 3
    if "Quad" in text: return 4
    for k in range(5, 9):
        if f"{k}-of-a-kind" in text:
            return k
    return 1

def parse_rank(text):
    try:
        return RANK_TO_NUM[text.split()[-1]]
    except KeyError:
        return -1

# ===================== PPO/DQN Laden ===================== #
MODELS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

def _norm_episode(ep):
    """Erzwingt explizite Episode; erlaubt int oder '10_000' als String."""
    if ep is None:
        raise RuntimeError("Episode ist None!")
    if isinstance(ep, int):
        return ep
    if isinstance(ep, str):
        s = ep.replace("_", "").strip()
        if s.isdigit():
            return int(s)
    raise ValueError(f"Ungültige Episode: {ep!r} (erwartet int oder numerischen String)")

def _stem_new_layout(family: str, version: str, pid_disk: int | None, episode: int, tag: str | None = None):
    """
    Baut den Stem (ohne Suffix) für das neue Layout:
      models/<family>/model_<version>/models/<family>_model_<version>_..._epXXXXXXX
    """
    base_dir = os.path.join(MODELS_ROOT, family, f"model_{version}", "models")
    if family in ("k1a1", "k3a1"):  # PPO (policy/value)
        assert pid_disk is not None
        name = f"{family}_model_{version}_agent_p{pid_disk}_ep{episode:07d}"
    elif family == "k3a2":          # DQN League: MAIN oder agent_i
        if tag and tag.upper() == "MAIN":
            name = f"k3a2_model_{version}_MAIN_ep{episode:07d}"
        else:
            assert pid_disk is not None
            name = f"k3a2_model_{version}_agent_{pid_disk}_ep{episode:07d}"
    elif family in ("k1a2", "k2a2", "k4a2"):  # DQN
        assert pid_disk is not None
        name = f"{family}_model_{version}_agent_p{pid_disk}_ep{episode:07d}"
    else:
        raise ValueError(f"Unbekannte family: {family}")
    return os.path.join(base_dir, name)

def _candidate_stems(kind: str, cfg: dict, pid_runtime: int) -> list[str]:
    """
    Liefert eine Liste möglicher Checkpoint-Stems (ohne .pt-Suffix), vom
    spezifischsten zum allgemeinsten. Berücksichtigt neues Layout sowie
    alte Fallbacks (ppo_model_XX/dqn_model_XX/dqn_league_XX).
    """
    stems = []

    # Direkter Stem override
    direct = cfg.get("path_stem") or cfg.get("stem")
    if direct:
        stems.append(os.path.abspath(direct))
        return stems

    version = cfg.get("version", "01")
    episode = _norm_episode(cfg.get("episode"))
    pid_disk = cfg.get("from_pid", pid_runtime)  # benutze p0-Gewichte etc.
    tag = cfg.get("tag")  # z.B. "MAIN" für k3a2

    # gewünschte family, sonst sinnvolle Default-Suchreihenfolge
    if "family" in cfg and cfg["family"]:
        families = [cfg["family"]]
    else:
        families = (["k3a1", "k1a1"] if kind == "ppo"
                    else ["k4a2", "k2a2", "k3a2", "k1a2"])

    # neues Layout
    for fam in families:
        try:
            stems.append(_stem_new_layout(fam, version, pid_disk if fam != "k3a2" or not (tag and tag.upper()=="MAIN") else None,
                                          episode, tag=tag))
        except Exception:
            pass

    # Fallbacks: alte Layouts
    root_old = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    if kind == "ppo":
        base = os.path.join(root_old, f"ppo_model_{version}", "train",
                            f"ppo_model_{version}_agent_p{pid_disk}_ep{episode:07d}")
        stems.append(base)
    elif kind == "dqn":
        for fam in ("dqn_model", "dqn_league"):
            base = os.path.join(root_old, f"{fam}_{version}", "train",
                                f"{fam}_{version}_agent_p{pid_disk}_ep{episode:07d}")
            stems.append(base)

    # expandiere evtl. vorhandene Episodenmuster (falls der explizite Stem fehlt)
    expanded = []
    for s in stems:
        if os.path.exists(s + ".pt") or os.path.exists(s + "_policy.pt") or os.path.exists(s + "_q.pt"):
            expanded.append(s)
            continue
        # sonst probiere: alle passenden Episoden im gleichen Ordner einsammeln
        pat = re.sub(r"_ep\d{7}$", "_ep*", s)
        expanded += [p[:-3] if p.endswith(".pt") else p for p in glob.glob(pat + "*.pt")]
    # deduplizieren, Reihenfolge bewahren
    seen = set(); stems = [x for x in (expanded or stems) if not (x in seen or seen.add(x))]
    return stems

def load_agents(player_config, game):
    agents = []
    info_dim = game.information_state_tensor_shape()[0]
    obs_dim  = game.observation_tensor_shape()[0]
    num_actions = game.num_distinct_actions()

    for pid, cfg in enumerate(player_config):
        kind = cfg["type"]

        if kind == "ppo":
            stems = _candidate_stems("ppo", cfg, pid_runtime=pid)
            tried, agent = [], None
            for seat_id_dim in (0, NUM_PLAYERS):  # ohne & mit Seat-One-Hot
                ag = ppo.PPOAgent(info_state_size=info_dim, num_actions=num_actions, seat_id_dim=seat_id_dim)
                for stem in stems:
                    try:
                        ag._policy.load_state_dict(torch.load(stem + "_policy.pt", map_location=ag.device))
                        ag._value.load_state_dict(torch.load(stem + "_value.pt",  map_location=ag.device))
                        ag._policy.eval(); ag._value.eval()
                        agent = ag
                        break
                    except FileNotFoundError as e:
                        tried.append(f"{stem}: {e}")
                        continue
                    except RuntimeError as e:
                        tried.append(f"{stem}: {e}")
                        break
                    except Exception as e:
                        tried.append(f"{stem}: {e}")
                        continue
                if agent is not None:
                    break
            if agent is None:
                print(f"[WARN] PPO-Weights nicht ladbar für P{pid} – random Init.\n  Versucht:\n    " + "\n    ".join(tried[:6]) + ("..." if len(tried)>6 else ""))
                agent = ppo.PPOAgent(info_state_size=info_dim, num_actions=num_actions, seat_id_dim=0)
            agents.append(agent)

        elif kind == "dqn":
            family  = cfg.get("family", "k1a2")   # z.B. k1a2, k2a2, k3a2, k4a2
            version = cfg.get("version", "01")
            episode = _norm_episode(cfg.get("episode"))  # "150_000" -> 150000

            ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            model_dir = os.path.join(ROOT, "models", family, f"model_{version}", "models")
            base_name = f"{family}_model_{version}_agent_p{pid}"

            def _candidate_stems():
                stems = []
                if episode is not None:
                    stem = os.path.join(model_dir, f"{base_name}_ep{episode:07d}")
                    # Prüfe preferiert den exakt gewünschten Checkpoint
                    if os.path.exists(stem + "_qnet.pt"):
                        stems.append(stem)
                # Fallback: alle qnet-Dateien einsammeln und zu Basis-Stems kürzen
                for f in glob.glob(os.path.join(model_dir, f"{base_name}_ep*_qnet.pt")):
                    stems.append(f[:-len("_qnet.pt")])
                # deduplizieren, Reihenfolge stabil
                seen = set()
                return [s for s in stems if not (s in seen or seen.add(s))]

            stems = _candidate_stems()

            # 1) Ohne, 2) mit Seat-One-Hot versuchen (robust ggü. Modellen mit/ohne OneHot)
            tried, agent = [], None
            for state_size in (OBS_DIM, OBS_DIM + NUM_PLAYERS):
                ag = dqn.DQNAgent(state_size=state_size, num_actions=num_actions)
                loaded = False
                for stem in stems:
                    try:
                        ag.restore(stem)   # erwartet Basis-Stem; hängt selbst _qnet/_tgt an
                        loaded = True
                        break
                    except FileNotFoundError as e:
                        tried.append(f"{stem}: {e}")
                        continue
                    except RuntimeError as e:
                        # Shape-Mismatch -> nächstes state_size probieren
                        tried.append(f"{stem}: {e}")
                        break
                    except Exception as e:
                        tried.append(f"{stem}: {e}")
                        continue
                if loaded:
                    ag.epsilon = 0.0
                    agent = ag
                    break

            if agent is None:
                print("[WARN] Kein DQN-Checkpoint für P{} – random Init.\n  Versucht:\n    {}".format(
                    pid, "\n    ".join(tried[:12] + (["..."] if len(tried) > 12 else []))
                ))
                agent = dqn.DQNAgent(state_size=OBS_DIM, num_actions=num_actions)

            agents.append(agent)


        elif kind in STRATS:
            agents.append(STRATS[kind])

        else:
            raise ValueError(f"Unbekannter Agententyp: {kind}")

    return agents



# ===================== Labels & Hilfsfunktionen ===================== #
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

def _masked_softmax_numpy(logits, legal):
    logits = np.asarray(logits, dtype=np.float32)
    mask = np.full_like(logits, -np.inf, dtype=np.float32)
    mask[legal] = logits[legal]
    m = np.max(mask[legal]) if len(legal) else 0.0
    ex = np.exp(mask - m)
    ex[~np.isfinite(ex)] = 0.0
    s = ex.sum()
    if s <= 0 or not np.isfinite(s):
        p = np.zeros_like(logits, dtype=np.float32)
        if len(legal):
            p[legal] = 1.0 / len(legal)
        return p
    return ex / s

def _forward_policy_with_autopad(policy_net, obs_1d, device):
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

# ===================== Aktionswahl ===================== #
def choose_policy_action(agent, state, player):
    legal = state.legal_actions(player)

    if isinstance(agent, ppo.PPOAgent):
        obs = np.array(state.information_state_tensor(player), dtype=np.float32)
        # prüfe, ob das Netz mehr Features erwartet (z.B. Seat-One-Hot)
        try:
            in_features = agent._policy.net[0].in_features
        except Exception:
            in_features = obs.shape[0]
        if in_features > obs.shape[0]:
            extra = in_features - obs.shape[0]
            if extra == NUM_PLAYERS:
                seat_oh = np.zeros(NUM_PLAYERS, dtype=np.float32); seat_oh[player] = 1.0
                obs = np.concatenate([obs, seat_oh], axis=0)

        device = getattr(agent, "device", "cpu")
        logits = _forward_policy_with_autopad(agent._policy, obs, device)
        probs = _masked_softmax_numpy(logits, legal)
        action = int(np.random.choice(len(probs), p=probs))
        return collections.namedtuple("AgentOutput", ["action"])(action=action)

    elif isinstance(agent, dqn.DQNAgent):
        obs = np.array(state.observation_tensor(player), dtype=np.float32)
        try:
            in_features = agent.q_network.net[0].in_features
        except Exception:
            in_features = obs.shape[0]
        if in_features > obs.shape[0]:
            extra = in_features - obs.shape[0]
            if extra == NUM_PLAYERS:
                seat_oh = np.zeros(NUM_PLAYERS, dtype=np.float32); seat_oh[player] = 1.0
                obs = np.concatenate([obs, seat_oh], axis=0)

        old_eps = getattr(agent, "epsilon", 0.0)
        agent.epsilon = 0.0  # greedy Eval
        try:
            action = int(agent.select_action(obs, legal))
        finally:
            agent.epsilon = old_eps
        return collections.namedtuple("AgentOutput", ["action"])(action=action)

    elif callable(agent):
        action = agent(state)
        if action not in legal:
            action = np.random.choice(legal)
        return collections.namedtuple("AgentOutput", ["action"])(action=action)

    else:
        raise ValueError("Unbekannter Agententyp bei choose_policy_action.")


# ===================== Evaluation ===================== #
def main():
    action_counts = defaultdict(lambda: defaultdict(int))       # alle Aktionen
    first_action_counts = defaultdict(lambda: defaultdict(int)) # nur erste Aktion des Startspielers
    agents = load_agents(PLAYER_CONFIG, game)
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

            if not first_action_done and pid == start_pid:
                first_action_counts[pid][action] += 1
                first_action_done = True

            action_counts[pid][action] += 1
            state.apply_action(action)

        final_returns = state.returns()
        for i, ret in enumerate(final_returns):
            returns_total[i] += ret
        win_counts[np.argmax(final_returns)] += 1

        if episode % 1000 == 0 and EVAL_OUTPUT:
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
            label += f" ep{int(config['episode']):,}".replace(",", ".")
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
    summary_path = os.path.join(CSV_DIR, "evaluation_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"📄 Evaluationsergebnisse gespeichert unter: {summary_path}")

    if not GENERATE_PLOTS:
        return

    print(f"📁 Ergebnisse und Plots werden gespeichert in: {PLOT_DIR}")
    num_actions = game.num_distinct_actions()

    # Labels vorbereiten
    if USE_READABLE_LABELS:
        all_labels = get_action_labels_readable(game, use_numbers=READABLE_USE_NUMBERS)
    else:
        all_labels = [f"Action {i}" for i in range(num_actions)]

    # Deckgröße
    deck_size = int(game.get_parameters().get("deck_size", "64"))

    # Combo-Labels je Deck
    if deck_size == 64:
        combo_labels = ["Single", "Pair", "Triple", "Quad"] + [f"{i}-of-a-kind" for i in range(5, 9)]
    else:
        combo_labels = ["Single", "Pair", "Triple", "Quad"]

    # 00 - Aktionsverteilung als Tabelle
    action_labels_table = [f"Action {i}" for i in range(num_actions)]
    action_table = pd.DataFrame(index=[f"Player {i}" for i in range(4)], columns=action_labels_table)
    for pid in range(4):
        total = sum(action_counts[pid].values())
        for aid in range(num_actions):
            count = action_counts[pid].get(aid, 0)
            percent = 100 * count / total if total > 0 else 0
            action_table.loc[f"Player {pid}", f"Action {aid}"] = f"{count} ({percent:.1f}%)"

    table_path = os.path.join(CSV_DIR, "00_action_distribution.csv")
    action_table.to_csv(table_path)
    print(f"📄 Tabelle gespeichert unter: {table_path}")

    def legend_inside(ax, loc="upper right", font_size=10):
        return ax.legend(loc=loc, frameon=True, framealpha=0.9, borderaxespad=0.5, fontsize=font_size)

    # 01 - First Action Distribution (Startspieler)
    action_ids = make_action_id_range(num_actions, start_id=0, include_pass=True)
    action_labels = [all_labels[aid] for aid in action_ids]
    counts_first_per_action = {aid: [first_action_counts[pid].get(aid, 0) for pid in range(4)] for aid in action_ids}
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
    legend_inside(ax, font_size=12)
    fig.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "01_first_action_distribution_startplayer.jpg"))

    # 02 - Gesamte Aktionsverteilung (ohne Pass)
    action_ids_no_pass = make_action_id_range(num_actions, start_id=1, include_pass=False)
    non_pass_labels = [all_labels[aid] for aid in action_ids_no_pass]
    counts_per_action_no_pass = {aid: [action_counts[pid].get(aid, 0) for pid in range(4)] for aid in action_ids_no_pass}
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
    legend_inside(ax, font_size=12)
    fig.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "02_action_distribution_total_no_pass.jpg"))

    # 03 - Gesamte Aktionsverteilung (inkl. Pass)
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
    legend_inside(ax, font_size=12)
    fig.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "03_action_distribution_total.jpg"))

    # 04 - Pass vs. Play
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
    legend_inside(ax)
    fig.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "04_pass_vs_play.jpg"))

    # 05 - Kombotyp-Anteile je Spieler
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

        legend_inside(ax)
        fig.tight_layout()
        filename = f"{combo_plot_index[combo]}_combo_{combo.lower().replace('-','_')}_detailed.jpg"
        plt.savefig(os.path.join(PLOT_DIR, filename))

    print("✅ Alle Plots erfolgreich gespeichert!")

if __name__ == "__main__":
    main()
