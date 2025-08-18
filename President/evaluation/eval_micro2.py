# -*- coding: utf-8 -*-
# evaluation/eval_micro.py — Single-Game Debug/Eval
# - Lädt PPO/DQN Checkpoints robust (mit/ohne Seat-One-Hot; DQN auch League/Family-Checkpoints)
# - Hängt bei Bedarf automatisch eine Seat-One-Hot an (nur wenn das Netz sie erwartet)
# - Loggt ein einzelnes Spiel detailliert in CSV + Mini-Plot

import os, re, json, glob, numpy as np, pandas as pd, torch, pyspiel, matplotlib.pyplot as plt
from collections import defaultdict, namedtuple

from agents import ppo_agent as ppo
from agents import dqn_agent as dqn
from utils import STRATS

# ====== Setup ======
GAME_SETTINGS = {
    "num_players": 4,
    "deck_size": "64",
    "shuffle_cards": True,
    "single_card_mode": False,
}

# Beispiel-Config (beliebig anpassen)
PLAYER_CONFIG = [
    {"name":"P0","type":"ppo","family":"k1a1","version":"06","episode":"20_000"}, 
    {"name": "Player1", "type": "max_combo"},
    {"name": "Player2", "type": "max_combo"},
    {"name": "Player3", "type": "max_combo"},
]

# --- Versioniertes Output wie eval_macro ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_MICRO_ROOT = os.path.join(BASE_DIR, "eval_micro")
os.makedirs(EVAL_MICRO_ROOT, exist_ok=True)

existing_runs = sorted([d for d in os.listdir(EVAL_MICRO_ROOT) if d.startswith("eval_micro_")])
next_run_num = int(existing_runs[-1].split("_")[-1]) + 1 if existing_runs else 1
RUN_DIR  = os.path.join(EVAL_MICRO_ROOT, f"eval_micro_{next_run_num:02d}")
CSV_DIR  = os.path.join(RUN_DIR, "csv")
PLOTS_DIR = os.path.join(RUN_DIR, "plots")
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Dateien in den Run-Ordner
CONFIG_CSV = os.path.join(RUN_DIR, "config.csv")
LOG_CSV    = os.path.join(CSV_DIR, "game_log.csv")
PLOT_PNG   = os.path.join(PLOTS_DIR, "plots.png")

with open(os.path.join(RUN_DIR, "game_settings.json"), "w") as f:
    json.dump(GAME_SETTINGS, f, indent=2)
pd.DataFrame(PLAYER_CONFIG).to_csv(os.path.join(RUN_DIR, "player_config.csv"), index=False)
print(f"📁 Eval-Micro Run-Ordner: {RUN_DIR}")

# ====== Helpers (gemeinsam mit eval_macro-Logik) ======
ROOT_MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

def _norm_episode(ep):
    if ep is None:
        return None
    s = str(ep).replace("_", "").strip()
    if s.isdigit():
        return int(s)
    raise ValueError(f"Ungültige Episode: {ep!r}")

def _strip_suffix(stem_or_file: str) -> str:
    """
    Entfernt bekannte Dateiendungen/Suffixe, so dass nur der 'Stem' bleibt,
    den dqn_agent.restore(...) bzw. PPO-Lader erwarten.
    """
    s = stem_or_file
    for suf in (
        ".pt", "_q.pt", "_qnet.pt", "_tgt.pt", "_tgt_q.pt", "_tgt_qnet.pt",
        "_policy.pt", "_value.pt"
    ):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s

def _list_pt_files(dirpath: str, prefix: str = ""):
    if not os.path.isdir(dirpath):
        return []
    if prefix:
        return [os.path.join(dirpath, f) for f in os.listdir(dirpath)
                if f.startswith(prefix) and f.endswith(".pt")]
    return [os.path.join(dirpath, f) for f in os.listdir(dirpath) if f.endswith(".pt")]

def _candidate_stems_new_family(kind: str, family: str, version: str,
                                pid_runtime: int, episode: int|None, pid_files: int|None):
    """
    Neue Speicherlogik:
      models/<family>/model_<version>/models/
        - DQN:  <family>_model_<version>_agent_p{X}_ep{NNNNNNN}_qnet.pt / _tgt_qnet.pt (o.ä.)
                ggf. MAIN: <family>_model_<version>_MAIN_ep{NNNNNNN}_*.pt
        - PPO:  <family>_model_<version>_agent_p0_ep{NNNNNNN}_policy.pt / _value.pt
    """
    dir_models = os.path.join(ROOT_MODELS, family, f"model_{version}", "models")
    pid_disk = pid_files if pid_files is not None else pid_runtime

    # Priorisierte Muster
    preferred = []
    if episode is not None:
        if kind == "ppo":
            preferred.append(f"{family}_model_{version}_agent_p{pid_disk}_ep{episode:07d}")
        else:  # dqn
            preferred.append(f"{family}_model_{version}_agent_p{pid_disk}_ep{episode:07d}")
            # League-MAIN Variante (z. B. k3a2)
            preferred.append(f"{family}_model_{version}_MAIN_ep{episode:07d}")

    # Alles, was passt, aus dem Ordner einsammeln (auch ohne Episode → letzter/alle)
    stems = []
    files = _list_pt_files(dir_models)
    for f in files:
        name = os.path.basename(f)
        if not name.startswith(f"{family}_model_{version}_"):
            continue
        # nur Policy/Value/QNet/Target-Varianten
        if not any(x in name for x in ("_policy.pt", "_value.pt", "_q.pt", "_qnet.pt", "_tgt", "_target")):
            continue
        if episode is not None and f"_ep{episode:07d}" not in name:
            continue
        # Wenn eine pid-Fixierung existiert, diese priorisieren
        if pid_files is not None and f"_agent_p{pid_files}_" not in name and "_MAIN_" not in name:
            continue
        stems.append(_strip_suffix(os.path.join(dir_models, name)))

    # Preferred Stems voran (falls existieren), danach alle gefundenen (dedupliziert, stabil)
    ordered = []
    seen = set()
    for p in preferred + stems:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered

def _candidate_stems_old_world(kind: str, version: str, pid_runtime: int,
                               episode: int|None, pid_files: int|None):
    """
    Abwärtskompatible Pfade:
      - ppo_model_<ver>/train/ppo_model_<ver>_agent_p{pid}_epXXXXXXX_(policy|value).pt
      - dqn_model_<ver>/train/dqn_model_<ver>_agent_p{pid}_epXXXXXXX_*.pt
      - dqn_league_<ver>/train/dqn_league_<ver>_agent_p{pid}_epXXXXXXX_*.pt
    """
    pid_disk = pid_files if pid_files is not None else pid_runtime
    families = ["ppo_model"] if kind == "ppo" else ["dqn_model", "dqn_league"]
    stems = []
    for fam in families:
        train_dir = os.path.join(ROOT_MODELS, f"{fam}_{version}", "train")
        if not os.path.isdir(train_dir):
            continue
        prefix = f"{fam}_{version}_agent_p{pid_disk}"
        if episode is not None:
            base = os.path.join(train_dir, f"{prefix}_ep{episode:07d}")
            stems.append(base)
        # plus alle ep-Stämme
        for f in _list_pt_files(train_dir, prefix=prefix):
            stems.append(_strip_suffix(os.path.join(train_dir, f)))
    # Dedupe stabil
    seen=set(); stems=[s for s in stems if not (s in seen or seen.add(s))]
    return stems

def _collect_candidate_stems(kind: str, cfg: dict, pid_runtime: int):
    version = cfg.get("version", "01")
    episode = _norm_episode(cfg.get("episode"))
    pid_files = cfg.get("from_pid")
    stems = []

    # 1) neue Family-Struktur, falls angegeben
    family = cfg.get("family")
    if family:
        stems.extend(_candidate_stems_new_family(kind, family, version, pid_runtime, episode, pid_files))

    # 2) alte Welt als Fallback
    stems.extend(_candidate_stems_old_world(kind, version, pid_runtime, episode, pid_files))

    # Dedupe stabil
    seen=set(); stems=[s for s in stems if not (s in seen or seen.add(s))]
    return stems

# ====== Loader ======
AgentOut = namedtuple("AgentOut", ["action"])  # kleines Wrapper-Objekt

def load_agents(cfgs, game):
    num_actions = game.num_distinct_actions()
    INFO_DIM = game.information_state_tensor_shape()[0]
    OBS_DIM  = game.observation_tensor_shape()[0]
    NUM_PLAYERS = game.num_players()

    agents = []
    for pid, cfg in enumerate(cfgs):
        kind = cfg["type"]

        if kind == "ppo":
            stems = _collect_candidate_stems("ppo", cfg, pid)
            loaded = None
            # zuerst ohne, dann mit Seat-One-Hot versuchen
            for seat_id_dim in (0, NUM_PLAYERS):
                ag = ppo.PPOAgent(info_state_size=INFO_DIM, num_actions=num_actions, seat_id_dim=seat_id_dim)
                ok=False
                for stem in stems:
                    try:
                        ag._policy.load_state_dict(torch.load(stem + "_policy.pt", map_location=ag.device))
                        ag._value.load_state_dict(torch.load(stem + "_value.pt",  map_location=ag.device))
                        ag._policy.eval(); ag._value.eval()
                        ok=True; break
                    except FileNotFoundError:
                        continue
                    except RuntimeError:
                        # Shape mismatch → anderen seat_id_dim testen
                        ok=False; break
                    except Exception:
                        continue
                if ok:
                    loaded = ag; break
            if loaded is None:
                print(f"[WARN] PPO-Checkpoint für P{pid} nicht gefunden – Random-Init.")
                loaded = ppo.PPOAgent(info_state_size=INFO_DIM, num_actions=num_actions, seat_id_dim=0)
            agents.append(loaded)

        elif kind == "dqn":
            stems = _collect_candidate_stems("dqn", cfg, pid)
            loaded = None
            # Zwei mögliche Input-Dims (ohne/mit Seat-One-Hot)
            for state_size in (OBS_DIM, OBS_DIM + NUM_PLAYERS):
                ag = dqn.DQNAgent(state_size=state_size, num_actions=num_actions)
                ok=False
                for stem in stems:
                    try:
                        ag.restore(stem)   # erwartet, dass zu 'stem' passende *_qnet.pt / *_tgt_*.pt existieren
                        ok=True; break
                    except FileNotFoundError:
                        continue
                    except RuntimeError:
                        # Shape mismatch → anderen state_size probieren
                        ok=False; break
                    except Exception:
                        continue
                if ok:
                    ag.epsilon = 0.0  # greedy Eval
                    loaded = ag; break
            if loaded is None:
                # hilfreiche Debug-Ausgabe
                print("[WARN] Kein DQN-Checkpoint für P{} – random Init.\n  Versucht:".format(pid))
                for s in stems[:12]:
                    print(f"    {s}_qnet(.pt) / _tgt_*(.pt)")
                if len(stems) > 12:
                    print(f"    ... (+{len(stems)-12} weitere)")
                loaded = dqn.DQNAgent(state_size=OBS_DIM, num_actions=num_actions)
            agents.append(loaded)

        elif kind in STRATS:
            agents.append(STRATS[kind])

        else:
            raise ValueError(f"Unbekannter Agententyp: {kind}")
    return agents

# ====== Softmax mit Legal-Masking (für PPO-Debug) ======
def _masked_softmax_numpy(logits, legal):
    logits = np.asarray(logits, dtype=np.float32)
    mask = np.full_like(logits, -np.inf, dtype=np.float32)
    if len(legal):
        mask[legal] = logits[legal]
        m = np.max(mask[legal])
    else:
        m = 0.0
    ex = np.exp(mask - m)
    ex[~np.isfinite(ex)] = 0.0
    s = ex.sum()
    if s <= 0 or not np.isfinite(s):
        p = np.zeros_like(logits, dtype=np.float32)
        if len(legal): p[legal] = 1.0 / len(legal)
        return p
    return ex / s

# ====== Forward mit Auto-Pad (falls Netz mehr Input erwartet) ======
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

# ====== Aktionswahl (hängt Seat-1Hot nur an, wenn nötig) ======
def choose_policy_action(agent, state, player, NUM_PLAYERS):
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
        return AgentOut(action)

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
        agent.epsilon = 0.0
        try:
            action = int(agent.select_action(obs, legal))
        finally:
            agent.epsilon = old_eps
        return AgentOut(action)

    elif callable(agent):
        action = int(agent(state))
        if action not in legal:
            action = int(np.random.choice(legal))
        return AgentOut(action)

    else:
        raise ValueError("Unbekannter Agententyp bei choose_policy_action.")

# ====== 1-Spiel-Debug ======
def main():
    game = pyspiel.load_game("president", GAME_SETTINGS)
    NUM_PLAYERS = game.num_players()

    agents = load_agents(PLAYER_CONFIG, game)

    # Config schreiben
    pd.DataFrame([{
        "game_settings": json.dumps(GAME_SETTINGS),
        "players": json.dumps(PLAYER_CONFIG),
    }]).to_csv(CONFIG_CSV, index=False)

    # Log vorbereiten
    rows = []
    state = game.new_initial_state()
    turn = 0

    while not state.is_terminal():
        pid = state.current_player()
        legal = state.legal_actions(pid)
        legal_txt = [state.action_to_string(pid, a) for a in legal]

        agent = agents[pid]
        ao = choose_policy_action(agent, state, pid, NUM_PLAYERS)
        action = int(ao.action)

        # Für PPO: Debug-Probabilities über legal actions (optional)
        probs_info = ""
        if isinstance(agent, ppo.PPOAgent):
            obs = np.array(state.information_state_tensor(pid), dtype=np.float32)
            try:
                in_features = agent._policy.net[0].in_features
            except Exception:
                in_features = obs.shape[0]
            if in_features > obs.shape[0] and (in_features - obs.shape[0]) == NUM_PLAYERS:
                seat_oh = np.zeros(NUM_PLAYERS, dtype=np.float32); seat_oh[pid] = 1.0
                obs = np.concatenate([obs, seat_oh], axis=0)
            device = getattr(agent, "device", "cpu")
            logits = _forward_policy_with_autopad(agent._policy, obs, device)
            probs = _masked_softmax_numpy(logits, legal)
            probs_info = "; ".join(f"{i}:{p:.2f}" for i, p in enumerate(probs) if p > 0)

        action_txt = state.action_to_string(pid, action)
        rows.append({
            "turn": turn, "player": pid,
            "legal_actions": str(list(zip(legal, legal_txt))),
            "chosen_action": f"{action} ({action_txt})",
            "policy_probs_over_legal": probs_info
        })

        state.apply_action(action)
        turn += 1

    # Finale Returns
    rets = state.returns()
    rows.append({"turn": turn, "player": "terminal", "legal_actions": "", "chosen_action": "", "policy_probs_over_legal": f"returns={rets}"})
    pd.DataFrame(rows).to_csv(LOG_CSV, index=False)
    print(f"📄 Game-Log gespeichert: {LOG_CSV}")

    # Minimaler Platzhalter-Plot (Zugnummern)
    plt.figure(figsize=(6,2))
    plt.plot(range(turn))
    plt.title("Eval Micro – Zugverlauf")
    plt.tight_layout()
    plt.savefig(PLOT_PNG); plt.close()
    print(f"🖼️ Plot gespeichert: {PLOT_PNG}")

    summary = {
        "micro_id": next_run_num,
        "num_turns": turn,
        "returns": rets,
    }
    with open(os.path.join(RUN_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
