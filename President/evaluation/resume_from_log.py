# -*- coding: utf-8 -*-
# evaluation/eval_resume_from_log.py
# Resume a real game from an existing eval_micro_X run, at a chosen "turn".
# Requirements:
#  - The source run directory must contain a full OpenSpiel action history, incl. chance outcomes:
#      run_dir/history.json   (a JSON array of integers = state.history())
#  - We will replay history until the chosen turn (i.e., that many *player* actions applied),
#    then continue to the end using configured agents (greedy).
#
# Outputs a fresh resume run folder with game_log.csv + action_probs.csv + summary.json

import os, json, argparse, numpy as np, pandas as pd, torch, pyspiel
from collections import namedtuple

from agents import ppo_agent as ppo
from agents import dqn_agent as dqn
from utils.strategies import STRATS
from utils.load_save_a1_ppo import load_checkpoint_ppo
from utils.load_save_a2_dqn import load_checkpoint_dqn

# ===========================
# resume_from_log.py — CONFIG
# ===========================
# Kurze Erklärung:
# - source_run_dir: Ordner eines früheren eval_micro-Runs (enthält game_settings.json + history.json)
# - from_turn: Zugindex, bis zu dem die History nachgespielt wird; ab diesem Zug wird weiter evaluiert
# - players: Player-Setup für die Fortsetzung (du kannst hier andere Agents verwenden als damals)
# - output_root: Basisordner für neue Resume-Runs (es wird eval_resume_<nn>/ angelegt)
# - greedy_eval: Policies werden greedy ausgewählt (PPO: argmax masked softmax; DQN: argmax Q)
# - save_files: steuert, welche Artefakte geschrieben werden
# - checks: zusätzliche Sicherheitsprüfungen beim Replay

CONFIG = {
    "source_run_dir": "/ABS/PFAD/zu/eval_micro_01",  # z.B. "/home/user/OpenSpiel/President/evaluation/eval_micro/eval_micro_07"
    "from_turn": 0,  # 0 = direkt nach Initialzustand; n = nach dem n-ten apply_action(...)
    "output_root": None,  # None => Standard: "<dieses_verzeichnis>/eval_resume"
    "greedy_eval": True,

    # Player-Setup (wie in eval_micro): ppo/dqn brauchen family/version/episode; Heuristiken via STRATS-Key
    "players": [
        {"name": "P0", "type": "ppo", "family": "k3a1", "version": "05", "episode": 20000, "from_pid": 0},
        {"name": "P1", "type": "max_combo"},
        {"name": "P2", "type": "max_combo"},
        {"name": "P3", "type": "max_combo"},
    ],

    # Dateien, die geschrieben werden
    "save_files": {
        "game_log": True,            # csv/game_log.csv (ab from_turn)
        "action_probs": True,        # csv/action_probs.csv
        "player_config": True,       # player_config.csv
        "game_settings": True,       # game_settings.json (aus Source übernommen)
        "summary": True,             # summary.json
        "resume_meta": True          # resume_meta.json (Quelle + from_turn)
    },

    # Sicherheits-Checks beim Replay
    "checks": {
        "assert_game_settings_equal": True,  # erzwingt identische game_settings.json wie im Source-Run
        "verify_replay_prefix": True,        # prüft, dass history() nach Replay genau 'from_turn' lang ist
    },
}


# ---------- Defaults for continuation agents (all 4 players) ----------
# Replace with your desired continuation policies:
RESUME_PLAYERS = [
    {"name":"P0", "type":"ppo", "family":"k3a1", "version":"05", "episode":20000, "from_pid":0},
    {"name":"P1", "type":"max_combo"},
    {"name":"P2", "type":"max_combo"},
    {"name":"P3", "type":"ppo", "family":"k3a1", "version":"05", "episode":20000, "from_pid":0},
]

AgentOut = namedtuple("AgentOut", ["action"])

def _fatal(msg):
    raise SystemExit("[FATAL] " + msg)

def _ppo_expected_stem(models_root, family, version, seat_on_disk, episode):
    base_dir = os.path.join(models_root, family, f"model_{version}", "models")
    stem = os.path.join(base_dir, f"{family}_model_{version}_agent_p{seat_on_disk}_ep{int(episode):07d}")
    pol, val = stem + "_policy.pt", stem + "_value.pt"
    if not (os.path.exists(pol) and os.path.exists(val)):
        _fatal(f"Missing PPO checkpoint: {pol} / {val}")
    return stem

def _dqn_expected_stem(models_root, family, version, seat_on_disk, episode):
    base_dir = os.path.join(models_root, family, f"model_{version}", "models")
    stem = os.path.join(base_dir, f"{family}_model_{version}_agent_p{seat_on_disk}_ep{int(episode):07d}")
    if not os.path.exists(stem + "_q.pt"):
        _fatal(f"Missing DQN checkpoint: {stem}_q.pt")
    return stem

def load_ppo_agent(game, cfg, models_root):
    info_dim = game.information_state_tensor_shape()[0]
    num_actions = game.num_distinct_actions()
    num_players = game.num_players()
    seat_id_dim = num_players
    stem = _ppo_expected_stem(models_root, cfg["family"], cfg["version"], cfg.get("from_pid", 0), cfg["episode"])
    ag = ppo.PPOAgent(info_state_size=info_dim, num_actions=num_actions, seat_id_dim=seat_id_dim, device="cpu")
    load_checkpoint_ppo(ag, os.path.dirname(stem), os.path.basename(stem))
    ag._policy.eval(); ag._value.eval()
    return ag

def load_dqn_agent(game, cfg, models_root):
    obs_dim = game.observation_tensor_shape()[0]
    num_actions = game.num_distinct_actions()
    num_players = game.num_players()
    stem = _dqn_expected_stem(models_root, cfg["family"], cfg["version"], cfg.get("from_pid", 0), cfg["episode"])
    tried = []
    for state_size in (obs_dim, obs_dim + num_players):
        ag = dqn.DQNAgent(state_size=state_size, num_actions=num_actions, device="cpu")
        try:
            load_checkpoint_dqn(ag, os.path.dirname(stem), os.path.basename(stem))
            if hasattr(ag, "epsilon"): ag.epsilon = 0.0
            return ag
        except Exception as e:
            tried.append(str(e))
    _fatal("Failed to load DQN: " + " | ".join(tried))

def masked_softmax(scores, legal):
    scores = np.asarray(scores, dtype=np.float32)
    if len(legal) == 0:
        return np.zeros_like(scores, dtype=np.float32)
    mask = np.full_like(scores, -np.inf, dtype=np.float32)
    mask[list(legal)] = scores[list(legal)]
    m = np.max(mask[list(legal)])
    ex = np.exp(mask - m); ex[~np.isfinite(ex)] = 0.0
    s = ex.sum()
    if s <= 0 or not np.isfinite(s):
        p = np.zeros_like(scores, dtype=np.float32)
        p[list(legal)] = 1.0 / len(legal)
        return p
    return ex / s

def ppo_logits(ag, state, pid, num_players):
    ist = np.asarray(state.information_state_tensor(pid), dtype=np.float32)
    seat_oh = np.zeros(num_players, dtype=np.float32); seat_oh[pid] = 1.0
    x = ag._make_input(ist, seat_one_hot=seat_oh)
    with torch.no_grad():
        return ag._policy(x).detach().cpu().numpy()

def dqn_qvalues(ag, state, pid, num_players):
    obs = np.asarray(state.observation_tensor(pid), dtype=np.float32)
    try:
        in_features = ag.q_network.net[0].in_features
    except Exception:
        in_features = obs.shape[0]
    if obs.shape[0] < in_features:
        extra = in_features - obs.shape[0]
        if extra == num_players:
            seat_oh = np.zeros(num_players, dtype=np.float32); seat_oh[pid] = 1.0
            obs = np.concatenate([obs, seat_oh], axis=0)
        else:
            obs = np.concatenate([obs, np.zeros(extra, dtype=np.float32)], axis=0)
    elif obs.shape[0] > in_features:
        obs = obs[:in_features]
    with torch.no_grad():
        xt = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        q = ag.q_network(xt).squeeze(0).detach().cpu().numpy()
    return q

def choose_with_probs(agent, state, pid, num_players):
    legal = state.legal_actions(pid)
    if callable(agent):
        a = int(agent(state))
        if a not in legal:
            a = int(np.random.choice(legal)) if legal else 0
        probs = np.zeros(state.num_distinct_actions(), dtype=np.float32)
        if legal: probs[legal] = 1.0 / len(legal)
        return AgentOut(a), probs, probs
    if isinstance(agent, ppo.PPOAgent):
        logits = ppo_logits(agent, state, pid, num_players)
        probs  = masked_softmax(logits, legal)
        return AgentOut(int(np.argmax(probs))), probs, logits
    if isinstance(agent, dqn.DQNAgent):
        qvals = dqn_qvalues(agent, state, pid, num_players)
        probs = masked_softmax(qvals, legal)
        return AgentOut(int(np.argmax(probs))), probs, qvals
    _fatal("Unknown agent type in choose_with_probs.")

def load_agents_for_resume(game, configs, models_root):
    agents = []
    for i, cfg in enumerate(configs):
        t = cfg.get("type")
        if t in STRATS:
            agents.append(STRATS[t]); continue
        if t == "ppo":
            agents.append(load_ppo_agent(game, cfg, models_root)); continue
        if t == "dqn":
            agents.append(load_dqn_agent(game, cfg, models_root)); continue
        _fatal(f"Unknown type for player {i}: {t}")
    return agents

def replay_history_until_turn(game, history: list[int], target_turn: int):
    """
    Apply history (chance + player actions) until 'target_turn' player-actions
    have been applied. Returns the resulting state and the index in history
    where we stopped (to aid debugging).
    """
    state = game.new_initial_state()
    applied_player_actions = 0
    idx = 0
    H = list(history)
    while not state.is_terminal() and idx < len(H) and applied_player_actions < target_turn:
        a = H[idx]
        # Sanity: if chance node, 'a' must be a legal chance outcome
        if state.is_chance_node():
            # We trust history here; OpenSpiel accepts any valid chance id
            state.apply_action(a)
            idx += 1
            continue
        # Decision node
        legal = state.legal_actions(state.current_player())
        if a not in legal:
            raise RuntimeError(f"History mismatch at idx={idx}, player={state.current_player()}, action={a} not legal {legal}")
        state.apply_action(a)
        applied_player_actions += 1
        idx += 1
    return state, idx, applied_player_actions

def main():
    ap = argparse.ArgumentParser(description="Resume an eval_micro run from a chosen turn.")
    ap.add_argument("--run_dir", required=True, help="Path to eval_micro_XX directory (with game_settings.json and history.json).")
    ap.add_argument("--resume_turn", type=int, required=True, help="Number of *player* actions to apply from the beginning before resuming.")
    ap.add_argument("--models_root", type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))
    ap.add_argument("--out_root", type=str, default=None, help="Where to write the resume results (default: evaluation/eval_resume).")
    args = ap.parse_args()

    settings_path = os.path.join(args.run_dir, "game_settings.json")
    if not os.path.exists(settings_path):
        _fatal(f"{settings_path} not found. Need original game settings.")
    with open(settings_path, "r") as f:
        GAME_SETTINGS = json.load(f)

    hist_path = os.path.join(args.run_dir, "history.json")
    if not os.path.exists(hist_path):
        _fatal(f"{hist_path} not found. Please add history saving to eval_micro (see patch below).")

    with open(hist_path, "r") as f:
        history = json.load(f)
    if not isinstance(history, list) or not all(isinstance(x, int) for x in history):
        _fatal("history.json must be a JSON array of integers (OpenSpiel action IDs).")

    game = pyspiel.load_game("president", GAME_SETTINGS)
    NUM_PLAYERS = game.num_players()

    # Replay to chosen turn
    try:
        state, used_idx, applied = replay_history_until_turn(game, history, args.resume_turn)
    except Exception as e:
        _fatal(f"Failed to replay history: {e}")

    # Prepare output dir
    out_root = args.out_root or os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_resume")
    os.makedirs(out_root, exist_ok=True)
    existing = sorted([d for d in os.listdir(out_root) if d.startswith("resume_")])
    next_id = int(existing[-1].split("_")[-1]) + 1 if existing else 1
    run_dir = os.path.join(out_root, f"resume_{next_id:02d}")
    csv_dir = os.path.join(run_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    # Save minimal config of this resume run
    with open(os.path.join(run_dir, "resume_from.json"), "w") as f:
        json.dump({"source_run_dir": args.run_dir, "resume_turn": args.resume_turn,
                   "used_history_prefix_len": used_idx, "applied_player_actions": applied}, f, indent=2)

    # Load continuation agents
    agents = load_agents_for_resume(game, RESUME_PLAYERS, args.models_root)

    # Continue to terminal (log like eval_micro)
    rows_log = []
    rows_probs = []
    turn = args.resume_turn

    while not state.is_terminal():
        pid = state.current_player()
        legal = state.legal_actions(pid)
        legal_txt = [state.action_to_string(pid, a) for a in legal]
        ao, probs, scores = choose_with_probs(agents[pid], state, pid, NUM_PLAYERS)
        action = int(ao.action)

        rows_log.append({
            "turn": turn, "player": pid,
            "legal_actions": str(list(zip(legal, legal_txt))),
            "chosen_action": f"{action} ({state.action_to_string(pid, action)})",
        })
        for a, txt in zip(legal, legal_txt):
            rows_probs.append({
                "turn": turn,
                "player": pid,
                "action_id": a,
                "action_text": txt,
                "prob": float(probs[a]),
                "score": float(scores[a]),
                "chosen": int(a == action),
            })

        state.apply_action(action)
        turn += 1

    rets = state.returns()
    rows_log.append({"turn": turn, "player": "terminal", "legal_actions": "", "chosen_action": ""})

    pd.DataFrame(rows_log).to_csv(os.path.join(csv_dir, "game_log.csv"), index=False)
    pd.DataFrame(rows_probs, columns=["turn","player","action_id","action_text","prob","score","chosen"])\
      .to_csv(os.path.join(csv_dir, "action_probs.csv"), index=False)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump({"resume_id": next_id, "start_turn": args.resume_turn,
                   "num_turns_added": turn - args.resume_turn, "returns": rets}, f, indent=2)

    print(f"\n✅ Resume complete. Output in: {run_dir}")
    print("Returns:", rets)

if __name__ == "__main__":
    main()
