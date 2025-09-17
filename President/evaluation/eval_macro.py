# evaluation/eval_macro.py
# -*- coding: utf-8 -*-

import os, re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
import logging

import pyspiel
import torch

from agents import ppo_agent as ppo
from agents import dqn_agent as dqn
from agents import v_table_agent
from collections import defaultdict
from utils.strategies import STRATS
from utils.deck import ranks_for_deck

from utils.load_save_a1_ppo import load_checkpoint_ppo
from utils.load_save_a2_dqn import load_checkpoint_dqn

# Dedizierter Logger für die Evaluation
logger = logging.getLogger("eval.macro")

# Fallback-Console-Logging NUR wenn noch nichts konfiguriert ist
if not logging.getLogger().handlers:
    _console_fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.INFO)
    _ch.setFormatter(_console_fmt)
    logging.getLogger().addHandler(_ch)
    logging.getLogger().setLevel(logging.INFO)

# ===================== Konfiguration ===================== #
NUM_EPISODES = 500_000
EVAL_EPISODE = 100_000
DECK = "64",  # "12" | "16" | "20" | "24" | "32" | "52" | "64"

# Beispiel-Setup (anpassen):

# 1v3 Gegen Max Combo
""" PLAYER_CONFIG = [
    {"name": "P0", "type": "dqn", "family": "k4a2", "version": "34", "episode": 1_000_000, "from_pid": 0},
    {"name": "P1", "type": "max_combo"},
    {"name": "P2", "type": "max_combo"},
    {"name": "P3", "type": "max_combo"},
]  """

# 1v3 Gegen Single Only

""" PLAYER_CONFIG = [
    {"name": "P0", "type": "dqn", "family": "k4a2", "version": "34", "episode": 1_000_000, "from_pid": 0},
    {"name": "P1", "type": "single_only"},
    {"name": "P2", "type": "single_only"},
    {"name": "P3", "type": "single_only"},
]   """

# 1v3 Gegen Random2
""" PLAYER_CONFIG = [
    {"name": "P0", "type": "dqn", "family": "k4a2", "version": "34", "episode": 1_000_000, "from_pid": 0},
    {"name": "P1", "type": "random2"},
    {"name": "P2", "type": "random2"},
    {"name": "P3", "type": "random2"},
]   """

# 2v2 über Kreuz vs Max Combo
""" PLAYER_CONFIG = [ 
    {"name": "P0", "type": "dqn", "family": "k4a2", "version": "34", "episode": 1_000_000, "from_pid": 0},
    {"name": "P1", "type": "max_combo"},
    {"name": "P2", "type": "dqn", "family": "k4a2", "version": "34", "episode": 1_000_000, "from_pid": 0},
    {"name": "P3", "type": "max_combo"}
]    """

# 2v2 über Kreuz vs Single Only
""" PLAYER_CONFIG = [ 
    {"name": "P0", "type": "dqn", "family": "k4a2", "version": "34", "episode": 1_000_000, "from_pid": 0},
    {"name": "P1", "type": "single_only"},
    {"name": "P2", "type": "dqn", "family": "k4a2", "version": "34", "episode": 1_000_000, "from_pid": 0},
    {"name": "P3", "type": "single_only"}
]    """

# 2v2 über Kreuz vs Random2

""" PLAYER_CONFIG = [ 
    {"name": "P0", "type": "dqn", "family": "k4a2", "version": "34", "episode": 1_000_000, "from_pid": 0},
    {"name": "P1", "type": "random2"},
    {"name": "P2", "type": "dqn", "family": "k4a2", "version": "34", "episode": 1_000_000, "from_pid": 0},
    {"name": "P3", "type": "random2"}
]   """

###############
###############
###############


# 1v3 Gegen V-Table (nur für 16 Karten)
""" PLAYER_CONFIG = [
    {"name": "P0", "type": "dqn", "family": "k4a2", "version": "32", "episode": 100_000, "from_pid": 0},
    {"name": "P1", "type": "v_table"},
    {"name": "P2", "type": "v_table"},
    {"name": "P3", "type": "v_table"},
]  """

# 2v2 über Kreuz vs V-Table (nur für 16 Karten)
""" PLAYER_CONFIG = [ 
    {"name": "P0", "type": "dqn", "family": "k4a2", "version": "32", "episode": 100_000, "from_pid": 0},
    {"name": "P1", "type": "v_table"},
    {"name": "P2", "type": "dqn", "family": "k4a2", "version": "32", "episode": 100_000, "from_pid": 0},
    {"name": "P3", "type": "v_table"}
]    """

""" # Vier Heuristiken
PLAYER_CONFIG = [ 
    {"name": "P0", "type": "max_combo"},
    {"name": "P1", "type": "single_only"}, 
    {"name": "P2", "type": "random2"},
    {"name": "P3", "type": "v_table"}
] """


# Vier verschiedene Spieler (Agents)
""" PLAYER_CONFIG = [ 
    {"name": "P0: K1 (vs Max Combo)", "type": "dqn", "family": "k1a2", "version": "36", "episode": 20_000, "from_pid": 0},
    {"name": "P1: K1 (vs Single Only)", "type": "dqn", "family": "k1a2", "version": "39", "episode": 20_000, "from_pid": 0},
    {"name": "P2: K3 (vs Tabelle)", "type": "dqn", "family": "k1a2", "version": "37", "episode": 20_000, "from_pid": 0},
    {"name": "P3: K4 (vs POP2)", "type": "dqn", "family": "k1a2", "version": "38", "episode": 20_000, "from_pid": 0}
] """

# Vier verschiedene Agents: 
PLAYER_CONFIG = [ 
    {"name": "P0: K3 (PPO Snapshot SP)", "type": "ppo", "family": "k3a1", "version": "32", "episode": 1_000_000, "from_pid": 0},
    {"name": "P1: K4 (PPO Shared Policy SP)", "type": "ppo", "family": "k4a1", "version": "05", "episode": 500_000, "from_pid": 0},
    {"name": "P2: K3 (DQN Snapshot SP)", "type": "dqn", "family": "k3a2", "version": "32", "episode": 210_000, "from_pid": 0},
    {"name": "P3: K4 (DQN Shared Policy SP)", "type": "dqn", "family": "k4a2", "version": "34", "episode": 1_000_000, "from_pid": 0},
]   

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
DECK_INT = int(GAME_PARAMS["deck_size"])
NUM_RANKS = ranks_for_deck(DECK_INT)
BASE_INFO_DIM = NUM_RANKS + (NUM_PLAYERS - 1) + 3
FULL_INFO_DIM = BASE_INFO_DIM + NUM_RANKS
BASE_OBS_DIM  = OBS_DIM - NUM_RANKS  # identisch zu BASE_INFO_DIM

# Speicher-Root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _log_player_config(cfg_list):
    headers = ["Idx", "Name", "Type", "Family", "Version", "Episode", "from_pid"]
    rows = []
    for i, c in enumerate(cfg_list):
        rows.append([
            i,
            c.get("name", f"P{i}"),
            c.get("type", ""),
            c.get("family", ""),
            c.get("version", ""),
            str(c.get("episode", "")).replace("_", ""),
            (c.get("from_pid", i) if c.get("type") in ("ppo", "dqn") else "")
        ])
    cols = list(zip(*([headers] + rows))) if rows else [headers]
    widths = [max(len(str(x)) for x in col) for col in cols]
    def fmt_line(values):
        return " | ".join(str(v).ljust(w) for v, w in zip(values, widths))
    logger.info("Player Configuration:")
    logger.info(fmt_line(headers))
    logger.info("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for r in rows:
        logger.info(fmt_line(r))

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
    qnet = stem + "_qnet.pt"
    tgt  = stem + "_tgt.pt"
    legacy_q = stem + "_q.pt"
    if not (os.path.exists(qnet) or os.path.exists(legacy_q)):
        _fatal(
            f"DQN-Checkpoint fehlt: family={family}, version={version}, seat={seat_on_disk}, episode={episode}",
            tried=[qnet, tgt, legacy_q],
        )
    return stem

def _alias_dqn_attrs(agent):
    if not hasattr(agent, "q_net") and hasattr(agent, "q_network"):
        agent.q_net = agent.q_network
    if not hasattr(agent, "target_net") and hasattr(agent, "target_network"):
        agent.target_net = agent.target_network
    return agent

def _load_ppo_agent_trydims(num_actions, *, family, version, episode, from_pid, device="cpu"):
    ep = _norm_episode(episode)
    stem = _ppo_expected_stem(family, version, from_pid, ep)
    weights_dir, tag = os.path.dirname(stem), os.path.basename(stem)
    tried = []
    for info_dim, seat_id_dim in [
        (FULL_INFO_DIM, NUM_PLAYERS),
        (FULL_INFO_DIM, 0),
        (BASE_INFO_DIM, NUM_PLAYERS),
        (BASE_INFO_DIM, 0),
    ]:
        try:
            ag = ppo.PPOAgent(info_state_size=info_dim, num_actions=num_actions,
                              seat_id_dim=seat_id_dim, device=device)
            load_checkpoint_ppo(ag, weights_dir, tag)
            ag._policy.eval(); ag._value.eval()
            return ag
        except Exception as e:
            tried.append(f"info_dim={info_dim}, seat_id_dim={seat_id_dim}: {e}")
    _fatal("Fehler beim Laden via load_checkpoint_ppo(...).", tried=[f"{stem}"] + tried)

def _load_dqn_agent(num_actions, *, family, version, episode, from_pid, num_players, device="cpu"):
    ep = _norm_episode(episode)
    stem = _dqn_expected_stem(family, version, from_pid, ep)
    tried, weights_dir, tag = [], os.path.dirname(stem), os.path.basename(stem)
    for state_size in (OBS_DIM, OBS_DIM + num_players, BASE_OBS_DIM, BASE_OBS_DIM + num_players):
        ag = dqn.DQNAgent(state_size=state_size, num_actions=num_actions, device=device)
        _alias_dqn_attrs(ag)
        try:
            load_checkpoint_dqn(ag, weights_dir, tag)
            if hasattr(ag, "epsilon"):
                ag.epsilon = 0.0
            return ag
        except Exception as e:
            tried.append(f"{stem} (state_size={state_size}): {e}")
    _fatal("Fehler beim Laden von DQN-Gewichten.", tried=tried)

def load_agents(player_config, game):
    agents = []
    num_actions = game.num_distinct_actions()
    num_players = game.num_players()
    for pid, cfg in enumerate(player_config):
        kind = cfg["type"]
        if kind == "ppo":
            if not all(k in cfg for k in ("family", "version", "episode")):
                _fatal(f"PPO-Spieler P{pid}: 'family', 'version' und 'episode' sind Pflicht. Erhalten: {cfg}")
            ag = _load_ppo_agent_trydims(num_actions,
                                         family=cfg["family"], version=cfg["version"],
                                         episode=cfg["episode"], from_pid=cfg.get("from_pid", pid))
            agents.append(ag); continue
        if kind == "dqn":
            if not all(k in cfg for k in ("family", "version", "episode")):
                _fatal(f"DQN-Spieler P{pid}: 'family', 'version' und 'episode' sind Pflicht. Erhalten: {cfg}")
            ag = _load_dqn_agent(num_actions,
                                 family=cfg["family"], version=cfg["version"], episode=cfg["episode"],
                                 from_pid=cfg.get("from_pid", pid), num_players=num_players)
            agents.append(ag); continue
        if kind == "v_table":
            ag = v_table_agent.ValueTableAgent("agents/tables/v_table_4_4_4")
            agents.append(ag); continue
        if kind in STRATS:
            agents.append(STRATS[kind]); continue
        _fatal(f"Unbekannter Agententyp bei P{pid}: {kind!r}")
    return agents

# ===================== Eval-Vorwärtswege ===================== #
def _masked_softmax_numpy(logits, legal):
    logits = np.asarray(logits, dtype=np.float32)
    if len(legal) == 0:
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
    base = info_state_1d[:getattr(agent, "_base_state_dim", len(info_state_1d))]
    x = agent._make_input(base, seat_one_hot=seat_oh)
    with torch.no_grad():
        logits = agent._policy(x).detach().cpu().numpy()
    probs = _masked_softmax_numpy(logits, legal)
    return int(np.argmax(probs))

def _forward_policy_autopad(policy_net: torch.nn.Module, obs_1d: np.ndarray, device):
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
    if isinstance(agent, ppo.PPOAgent):
        info_vec = np.array(state.information_state_tensor(player), dtype=np.float32)
        if hasattr(agent, "_make_input"):
            return collections.namedtuple("AgentOutput", ["action"])(action=_forward_policy_with_make_input(
                agent, info_vec, seat_id=player, legal=legal
            ))
        device = getattr(agent, "device", "cpu")
        logits = _forward_policy_autopad(agent._policy, info_vec, device)
        probs = _masked_softmax_numpy(logits, legal)
        action = int(np.argmax(probs))
        return collections.namedtuple("AgentOutput", ["action"])(action=action)
    if isinstance(agent, dqn.DQNAgent):
        obs_vec = np.array(state.observation_tensor(player), dtype=np.float32)
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
    if callable(agent):
        action = agent(state)
        if action not in legal:
            raise RuntimeError("illegal action:", action)
        return collections.namedtuple("AgentOutput", ["action"])(action=action)
    raise ValueError("Unbekannter Agententyp bei choose_policy_action.")

# ===================== Logging-Helfer ===================== #
def _setup_eval_file_logging(run_dir: str):
    log_path = os.path.join(run_dir, "run.log")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    have_same = any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == os.path.abspath(log_path)
                    for h in logger.handlers)
    if not have_same:
        logger.addHandler(fh)
    logger.propagate = True
    return log_path

# ===================== Helpers: CSV/Labeling ===================== #
RANK_LABELS = ["7", "8", "9", "10", "J", "Q", "K", "A"]
COMBO_ORDER = ["Single", "Pair", "Triple", "Quad", "Quint", "Sext", "Sept", "Oct"]

# Deck-abhängige aktive Labels (Tabellen 00/01/02/03) und Plot-Labels (06/07)
if DECK_INT == 16:
    ACTIVE_RANK_LABELS = ["J", "Q", "K", "A"]
    ACTIVE_COMBO_ORDER = ["Single", "Pair", "Triple", "Quad"]
    # Für Plots 06/07: hier heißen 5+ in den Action-Strings "5-of-a-kind", ...
    combo_labels = ["Single", "Pair", "Triple", "Quad"]
else:
    ACTIVE_RANK_LABELS = RANK_LABELS
    ACTIVE_COMBO_ORDER = COMBO_ORDER
    combo_labels = ["Single", "Pair", "Triple", "Quad",
                    "5-of-a-kind", "6-of-a-kind", "7-of-a-kind", "8-of-a-kind"]




def _write_split_variants(df_counts: pd.DataFrame, df_percent: pd.DataFrame,
                          path_counts: str, path_percent: str, path_combined: str = None,
                          percent_decimals: int = 1):
    """Schreibt Counts/Percent und optional Combined (count (x.y%)). Index/Spalten müssen identisch sein."""
    df_counts.to_csv(path_counts, encoding="utf-8")
    df_percent.to_csv(path_percent, encoding="utf-8")
    if path_combined is not None:
        comb = df_counts.copy()
        for col in df_counts.columns:
            if col in df_percent.columns:
                comb[col] = [f"{c} ({p:.{percent_decimals}f}%)" for c, p in zip(df_counts[col], df_percent[col])]
        comb.to_csv(path_combined, encoding="utf-8")

def _aid_to_combo_rank(dummy_state, aid: int):
    """Parst eine Action-ID zu (combo_label, rank_label) bzw. ('Pass', None) oder (None, None) bei unbekannt."""
    try:
        lab = dummy_state.action_to_string(0, aid)
    except Exception:
        return (None, None)
    if "Pass" in lab:
        return ("Pass", None)
    rank = lab.split()[-1]
    combo = None
    if "Single" in lab: combo = "Single"
    elif "Pair" in lab: combo = "Pair"
    elif "Triple" in lab: combo = "Triple"
    elif "Quad" in lab: combo = "Quad"
    else:
        m = re.search(r"(\d+)-of-a-kind", lab)
        if m:
            k = int(m.group(1))
            if   k == 5: combo = "Quint"
            elif k == 6: combo = "Sext"
            elif k == 7: combo = "Sept"
            elif k == 8: combo = "Oct"
    return (combo, rank) if combo else (None, None)

def _make_00_columns():
    cols = ["Pass"]
    for combo in ACTIVE_COMBO_ORDER:
        for r in ACTIVE_RANK_LABELS:
            cols.append(f"{combo} {r}")
    return cols


def _aggregate_counts_by_combo_rank(action_counts_for_pid: dict, dummy_state):
    col_counts = {c: 0 for c in _make_00_columns()}
    for aid, c in action_counts_for_pid.items():
        combo, rank = _aid_to_combo_rank(dummy_state, aid)
        if combo == "Pass":
            col_counts["Pass"] += c
        elif combo in ACTIVE_COMBO_ORDER and rank in ACTIVE_RANK_LABELS:
            key = f"{combo} {rank}"
            if key in col_counts:
                col_counts[key] += c
    return col_counts

def _aggregate_counts_by_combo_only(counts_source: dict, dummy_state, only_first_nonpass=False):
    res = {k: 0 for k in ACTIVE_COMBO_ORDER}
    for aid, c in counts_source.items():
        combo, rank = _aid_to_combo_rank(dummy_state, aid)
        if combo == "Pass":
            if only_first_nonpass:
                continue
        if combo in ACTIVE_COMBO_ORDER:
            res[combo] += c
    return res


# ===================== Evaluation ===================== #
def main():
    action_counts = defaultdict(lambda: defaultdict(int))       # alle Aktionen
    first_action_counts = defaultdict(lambda: defaultdict(int)) # nur erste Aktion des Startspielers
    agents = load_agents(PLAYER_CONFIG, game)

    # --- Verzeichnisse jetzt (erst nach erfolgreichem Laden) anlegen ---
    EVAL_MACRO_ROOT = os.path.join(BASE_DIR, "eval_macro")

    existing_macro_dirs = []
    if os.path.isdir(EVAL_MACRO_ROOT):
        for d in os.listdir(EVAL_MACRO_ROOT):
            p = os.path.join(EVAL_MACRO_ROOT, d)
            if os.path.isdir(p) and d.startswith("eval_macro_"):
                existing_macro_dirs.append(d)

    nums = []
    for d in existing_macro_dirs:
        tail = d.split("_")[-1]
        if tail.isdigit():
            nums.append(int(tail))
    next_macro_num = (max(nums) + 1) if nums else 1

    MACRO_DIR = os.path.join(EVAL_MACRO_ROOT, f"eval_macro_{next_macro_num:03d}")
    os.makedirs(MACRO_DIR, exist_ok=True)
    CSV_DIR = os.path.join(MACRO_DIR, "csv");  os.makedirs(CSV_DIR, exist_ok=True)
    PLOT_DIR = os.path.join(MACRO_DIR, "plots"); os.makedirs(PLOT_DIR, exist_ok=True)

    run_log_path = _setup_eval_file_logging(MACRO_DIR)
    logger.info("New Evaluation (macro_%02d)", next_macro_num)
    logger.info("Deck_Size: %s", GAME_PARAMS["deck_size"])
    logger.info("Episodes: %s", f"{NUM_EPISODES:,}".replace(",", "."))
    logger.info("Path: %s", MACRO_DIR)
    logger.info("")
    _log_player_config(PLAYER_CONFIG)

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

        if episode % EVAL_EPISODE == 0 and EVAL_OUTPUT:
            current_winrates = [100 * win_counts[i] / episode for i in range(NUM_PLAYERS)]
            def _fmt_label(i):
                cfg = PLAYER_CONFIG[i]
                ver = cfg.get('version', '')
                epi = cfg.get('episode')
                epi_s = f", ep{int(str(epi).replace('_','')):,}".replace(",", ".") if epi is not None else ""
                return f"P{i} ({cfg['type']} v{ver}{epi_s})"
            wr_str = " | ".join(f"{_fmt_label(i)}: {wr:.1f}%" for i, wr in enumerate(current_winrates))
            logger.info("")
            logger.info("Benchmark @eval_ep %7d", episode)
            logger.info("Winrates: %s", wr_str)
            for i in range(NUM_PLAYERS):
                cfg = PLAYER_CONFIG[i]
                ver = cfg.get("version", "")
                epi = cfg.get("episode")
                epi_s = f" ep{int(str(epi).replace('_','')):,}".replace(",", ".") if epi is not None else ""
                wr  = 100.0 * win_counts[i] / episode
                avg = returns_total[i] / episode
                wins_i   = int(win_counts[i])
                starts_i = int(start_counts[i])
                logger.info(
                    "P%d (%s v%s%s) | winrate: %5.1f%% | avg_return: %7.2f | wins: %6d | starts: %6d",
                    i, cfg["type"], ver, epi_s, wr, avg, wins_i, starts_i
                )

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
            "family": cfg.get("family", ""),
            "version": cfg.get("version", ""),
            "episode": epi if epi is not None else "",
            "win_rate_percent": round(float(winrate), 1),
            "avg_return": round(float(avg_ret), 2),
            "num_wins": int(win_counts[i]),
            "num_starts": int(start_counts[i]),
        }
        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(CSV_DIR, "evaluation_summary.csv")
    df.to_csv(summary_path, index=False)
    logger.info("Evaluation summary saved: %s", summary_path)
    logger.info("CSV_DIR: %s", CSV_DIR)

    if not GENERATE_PLOTS:
        logger.info("Plots disabled.")
        logger.info("run.log: %s", run_log_path)
        return

    logger.info("Plots directory: %s", PLOT_DIR)
    all_labels = get_action_labels_readable(game, use_numbers=READABLE_USE_NUMBERS) if USE_READABLE_LABELS \
                 else [f"Action {i}" for i in range(NUM_ACTIONS)]

    dummy_state = game.new_initial_state()

    # ---------------- 00 - Aktionsverteilung (Tabellen, neues Schema) ----------------
    cols_00 = _make_00_columns()
    idx_players = [f"Player {i}" for i in range(NUM_PLAYERS)]
    action_table_counts   = pd.DataFrame(0,   index=idx_players, columns=cols_00, dtype=int)
    action_table_percent  = pd.DataFrame(0.0, index=idx_players, columns=cols_00, dtype=float)
    action_table_combined = pd.DataFrame(index=idx_players, columns=cols_00, dtype=object)

    for pid in range(NUM_PLAYERS):
        col_counts = _aggregate_counts_by_combo_rank(action_counts[pid], dummy_state)
        total = sum(col_counts.values()) or 1
        for col in cols_00:
            c = int(col_counts[col])
            p = round(100.0 * c / total, 1)
            action_table_counts.loc[f"Player {pid}", col]  = c
            action_table_percent.loc[f"Player {pid}", col] = p
            action_table_combined.loc[f"Player {pid}", col] = f"{c} ({p:.1f}%)"

    _write_split_variants(
        action_table_counts, action_table_percent,
        os.path.join(CSV_DIR, "00_action_distribution_counts.csv"),
        os.path.join(CSV_DIR, "00_action_distribution_percent.csv"),
        os.path.join(CSV_DIR, "00_action_distribution.csv")
    )

    # --- 00 (zusätzlich): Action-Distribution OHNE Pass ---
    # Prozente MUESSEN hier mit neuem Nenner (ohne "Pass") berechnet werden.
    no_pass_cols = [c for c in action_table_counts.columns if str(c) != "Pass"]

    # Counts ohne Pass
    action_table_counts_no_pass = action_table_counts[no_pass_cols].copy()

    # Prozente ohne Pass (Zeilensummen ohne Pass als Nenner)
    row_sums_np = action_table_counts_no_pass.sum(axis=1).replace(0, 1)  # 0->1, um Division durch 0 zu vermeiden
    action_table_percent_no_pass = (action_table_counts_no_pass
                                    .div(row_sums_np, axis=0)
                                    .mul(100.0)
                                    .round(1))

    # Combined ohne Pass (count (x.y%))
    action_table_combined_no_pass = action_table_counts_no_pass.copy().astype(object)
    for col in no_pass_cols:
        action_table_combined_no_pass[col] = [
            f"{int(c)} ({p:.1f}%)" for c, p in zip(action_table_counts_no_pass[col], action_table_percent_no_pass[col])
        ]

    # Speichern: combined + split (ohne Pass)
    table_np_path = os.path.join(CSV_DIR, "01_action_distribution_no_pass.csv")
    action_table_combined_no_pass.to_csv(table_np_path, encoding="utf-8")
    _write_split_variants(
        action_table_counts_no_pass,
        action_table_percent_no_pass,
        os.path.join(CSV_DIR, "01_action_distribution_counts_no_pass.csv"),
        os.path.join(CSV_DIR, "01_action_distribution_percent_no_pass.csv")
    )
    logger.info("01-tables saved (with/without pass, recomputed no-pass percentages).")


    def legend_inside(ax, loc="upper right", font_size=10):
        return ax.legend(loc=loc, frameon=True, framealpha=0.9, borderaxespad=0.5, fontsize=font_size)

    # ---------------- 01 - First Action Distribution (Plot, OHNE Pass) ----------------
    action_ids_no_pass = make_action_id_range(NUM_ACTIONS, start_id=1, include_pass=False)
    action_labels_no_pass = [all_labels[aid] for aid in action_ids_no_pass]
    counts_first_per_action = {aid: [first_action_counts[pid].get(aid, 0) for pid in range(NUM_PLAYERS)]
                               for aid in action_ids_no_pass}

    x = np.arange(len(action_labels_no_pass))
    width = 0.2
    fig, ax = plt.subplots(figsize=(min(24, max(12, len(action_labels_no_pass) * 0.6)), 8))
    for pid in range(NUM_PLAYERS):
        total_first_nonpass = sum(c for aid, c in first_action_counts[pid].items() if aid != 0) or 1
        counts = [100 * counts_first_per_action[aid][pid] / total_first_nonpass for aid in action_ids_no_pass]
        cfg = PLAYER_CONFIG[pid]
        ver = cfg.get("version","")
        epi = cfg.get("episode")
        epi_s = f", ep{_norm_episode(epi):,}".replace(",", ".") if epi is not None else ""
        winrate = 100 * win_counts[pid] / max(1, NUM_EPISODES)
        ax.bar(x + width * pid, counts, width, label=f"P{pid} ({cfg['type']} v{ver}{epi_s}, {winrate:.1f}%)")
    ax.set_xlabel("Action (erste Aktion des Startspielers, Pass ausgeschlossen)")
    ax.set_ylabel("Relative Häufigkeit (%)")
    ax.set_title("01 - First Action Distribution (Startspieler, ohne Pass)")
    ax.set_xticks(x + width * (NUM_PLAYERS-1)/2)
    ax.set_xticklabels(action_labels_no_pass, rotation=90)
    legend_inside(ax, font_size=12)
    fig.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "01_first_action_distribution_startplayer.jpg"))

    # ---------------- 02 - First Action Distribution (CSVs NACH KOMBO, Single..Oct) ----------------
    fa_counts_combo   = pd.DataFrame(0,   index=idx_players, columns=ACTIVE_COMBO_ORDER, dtype=int)
    fa_percent_combo  = pd.DataFrame(0.0, index=idx_players, columns=ACTIVE_COMBO_ORDER, dtype=float)
    fa_combined_combo = pd.DataFrame(index=idx_players, columns=ACTIVE_COMBO_ORDER, dtype=object)

    for pid in range(NUM_PLAYERS):
        combo_counts = _aggregate_counts_by_combo_only(first_action_counts[pid], dummy_state, only_first_nonpass=True)
        total = sum(combo_counts.values()) or 1
        for combo in ACTIVE_COMBO_ORDER:
            c = int(combo_counts.get(combo, 0))
            p = round(100.0 * c / total, 1)
            fa_counts_combo.loc[f"Player {pid}", combo]  = c
            fa_percent_combo.loc[f"Player {pid}", combo] = p
            fa_combined_combo.loc[f"Player {pid}", combo] = f"{c} ({p:.1f}%)"


    _write_split_variants(
        fa_counts_combo, fa_percent_combo,
        os.path.join(CSV_DIR, "02_first_action_distribution_counts_bucket.csv"),
        os.path.join(CSV_DIR, "02_first_action_distribution_percent_bucket.csv"),
        os.path.join(CSV_DIR, "02_first_action_distribution_bucket.csv")
    )

    # ---------------- 03 - First Action Distribution (CSVs DETAILLIERT: Combo+Rank, OHNE Pass) ----------------
    cols_03 = [f"{combo} {r}" for combo in ACTIVE_COMBO_ORDER for r in ACTIVE_RANK_LABELS]
    fa_det_counts   = pd.DataFrame(0,   index=idx_players, columns=cols_03, dtype=int)
    fa_det_percent  = pd.DataFrame(0.0, index=idx_players, columns=cols_03, dtype=float)
    fa_det_combined = pd.DataFrame(index=idx_players, columns=cols_03, dtype=object)

    for pid in range(NUM_PLAYERS):
        det_counts = {c: 0 for c in cols_03}
        for aid, cnt in first_action_counts[pid].items():
            combo, rank = _aid_to_combo_rank(dummy_state, aid)
            if combo in ACTIVE_COMBO_ORDER and rank in ACTIVE_RANK_LABELS:
                det_counts[f"{combo} {rank}"] += cnt


        total_first_nonpass = sum(det_counts.values()) or 1
        for col in cols_03:
            c = int(det_counts[col])
            p = round(100.0 * c / total_first_nonpass, 1)
            fa_det_counts.loc[f"Player {pid}", col]   = c
            fa_det_percent.loc[f"Player {pid}", col]  = p
            fa_det_combined.loc[f"Player {pid}", col] = f"{c} ({p:.1f}%)"

    _write_split_variants(
        fa_det_counts,
        fa_det_percent,
        os.path.join(CSV_DIR, "03_first_action_distribution_counts.csv"),
        os.path.join(CSV_DIR, "03_first_action_distribution_percent.csv"),
        os.path.join(CSV_DIR, "03_first_action_distribution.csv")
    )

    # ---------------- 04 - First-Action by Rank (CSVs + Plot) ----------------
    aid_to_rank = {}
    rank_order = []
    for aid in range(NUM_ACTIONS):
        try:
            lab = dummy_state.action_to_string(0, aid)
        except Exception:
            continue
        if "Pass" in lab:
            continue
        rank = lab.split()[-1]
        aid_to_rank[aid] = rank
        if rank not in rank_order:
            rank_order.append(rank)

    if aid_to_rank:
        fa_by_rank_counts = {pid: {r: 0 for r in rank_order} for pid in range(NUM_PLAYERS)}
        fa_totals_by_rank = {pid: 0 for pid in range(NUM_PLAYERS)}
        for pid in range(NUM_PLAYERS):
            for aid, c in first_action_counts[pid].items():
                r = aid_to_rank.get(aid)
                if r is None:
                    continue
                fa_by_rank_counts[pid][r] += c
                fa_totals_by_rank[pid]    += c

        df_fa_rank_counts = pd.DataFrame.from_dict(fa_by_rank_counts, orient="index")[rank_order]
        df_fa_rank_counts.index = idx_players; df_fa_rank_counts.index.name = "Player"
        denom = [max(fa_totals_by_rank[pid], 1) for pid in range(NUM_PLAYERS)]
        df_fa_rank_percent = (df_fa_rank_counts[rank_order].div(denom, axis=0) * 100.0).round(1)
        df_fa_rank_percent.index = idx_players; df_fa_rank_percent.index.name = "Player"

        _write_split_variants(
            df_fa_rank_counts, df_fa_rank_percent,
            os.path.join(CSV_DIR, "04_first_action_by_rank_counts.csv"),
            os.path.join(CSV_DIR, "04_first_action_by_rank_percent.csv"),
            os.path.join(CSV_DIR, "04_first_action_by_rank.csv")
        )

        # Plot
        ranks = rank_order[:]
        x = np.arange(len(ranks))
        width = min(0.8 / max(NUM_PLAYERS, 1), 0.25)
        fig_width = max(12, len(ranks) * 0.55)
        fig, ax = plt.subplots(figsize=(fig_width, 6))
        for pid in range(NUM_PLAYERS):
            percents = [float(df_fa_rank_percent.loc[f"Player {pid}", r]) if r in df_fa_rank_percent.columns else 0.0
                        for r in ranks]
            cfg = PLAYER_CONFIG[pid]
            ver = cfg.get("version", "")
            epi = cfg.get("episode")
            epi_s = f", ep{_norm_episode(epi):,}".replace(",", ".") if epi is not None else ""
            winrate = 100.0 * (win_counts[pid] / max(1, NUM_EPISODES))
            ax.bar(x + width * pid, percents, width,
                   label=f"P{pid} ({cfg['type']} v{ver}{epi_s}, {winrate:.1f}%)")
        ax.set_xlabel("Rang (erste gespielte Aktion, Pass ausgeschlossen)")
        ax.set_ylabel("Anteil an allen ersten (nicht-Pass) Aktionen (%)")
        ax.set_title("04 - First-Action by Rank (Percent)")
        ax.set_xticks(x + width * (NUM_PLAYERS - 1) / 2)
        ax.set_xticklabels(ranks, rotation=90)
        ax.set_ylim(0, 100)
        legend_inside(ax, font_size=12)
        fig.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "04_first_action_by_rank_percent.jpg"))
        plt.close(fig)
    else:
        logger.info("First-Action by Rank übersprungen (keine nicht-Pass-Aktionen erkannt).")

    # ---------------- 05 - Pass vs. Play – Plot ----------------
    pass_stats = {pid: {"Pass": 0, "Play": 0} for pid in range(NUM_PLAYERS)}
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

    play_counts = []; pass_counts = []
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
    ax.set_title("05 - Anteil von Pass vs. Spiel-Aktionen")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 100)
    legend_inside(ax)
    fig.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "05_pass_vs_play.jpg"))

    # ---------------- 06 - Kombotyp-Anteile je Spieler (Plot, Referenz) ----------------
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
    ax.set_title("06 - Anteil gespielter Kombitypen pro Spieler")
    ax.set_xticks(x + width * (len(combo_labels)-1) / 2)
    ax.set_xticklabels([f"P{i}" for i in range(NUM_PLAYERS)])
    ax.set_ylim(0, 100)
    legend_inside(ax)
    fig.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "06_combo_types_per_player.jpg"))

    # ---------------- 07+ - Detaillierte Kombitypen-Plots ----------------
    combo_plot_start_idx = 7
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

    logger.info("All plots & CSVs saved.")
    logger.info("run.log: %s", run_log_path)

if __name__ == "__main__":
    main()
