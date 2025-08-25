# evaluation/eval_micro.py
# -*- coding: utf-8 -*-
# evaluation/eval_micro.py — Single-Game Debug/Eval (greedy, explizite Checkpoints)
# - bricht ab, wenn Checkpoints fehlen (kein Random-Init)
# - nutzt Trainings-Loader (load_checkpoint_ppo/dqn)
# - erstellt Run-Ordner erst nach erfolgreichem Laden
# - schreibt zusätzlich action_probs.csv mit Wahrscheinlichkeiten über legalen Aktionen pro Zug

import os, re, json, numpy as np, pandas as pd, torch, pyspiel
from collections import namedtuple

from agents import ppo_agent as ppo
from agents import dqn_agent as dqn
from agents import v_table_agent
from utils.strategies import STRATS
from utils.load_save_a1_ppo import load_checkpoint_ppo
from utils.load_save_a2_dqn import load_checkpoint_dqn
from utils.deck import ranks_for_deck

# ====== Setup ======
GAME_SETTINGS = {
    "num_players": 4,
    "deck_size": "16",
    "shuffle_cards": True,
    "single_card_mode": False,
}

# Beispiel-Config
PLAYER_CONFIG = [
    {"name": "P0", "type": "dqn", "family": "k1a2", "version": "46", "episode": 40_000, "from_pid": 0},
    {"name": "P1", "type": "v_table"},
    {"name": "P2", "type": "max_combo"},
    {"name": "P3", "type": "max_combo"},
]

# ====== Pfade & kleine Utils ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
    s = str(ep).replace("_","").strip()
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

# ====== Softmax mit Legal-Masking ======
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

# ====== Laden & Vorwärtswege ======
AgentOut = namedtuple("AgentOut", ["action"])

def _load_ppo_agent(info_dim, num_actions, seat_id_dim, *, family, version, episode, from_pid, device="cpu"):
    ep = _norm_episode(episode)
    stem = _ppo_expected_stem(family, version, from_pid, ep)
    ag = ppo.PPOAgent(info_state_size=info_dim, num_actions=num_actions, seat_id_dim=seat_id_dim, device=device)
    weights_dir, tag = os.path.dirname(stem), os.path.basename(stem)
    try:
        load_checkpoint_ppo(ag, weights_dir, tag)
        ag._policy.eval(); ag._value.eval()
        return ag
    except Exception as e:
        _fatal("Fehler beim Laden via load_checkpoint_ppo(...).", tried=[f"{stem}: {e}"])

def _load_dqn_agent(obs_dim, num_actions, *, family, version, episode, from_pid, num_players, device="cpu"):
    ep = _norm_episode(episode)
    stem = _dqn_expected_stem(family, version, from_pid, ep)
    tried = []

    # deck-aware: auch base obs (ohne Historie) versuchen
    num_ranks = ranks_for_deck(int(GAME_SETTINGS["deck_size"]))
    base_obs = obs_dim - num_ranks

    for state_size in (obs_dim, obs_dim + num_players, base_obs, base_obs + num_players):
        ag = dqn.DQNAgent(state_size=state_size, num_actions=num_actions, device=device)
        _alias_dqn_attrs(ag)
        weights_dir, tag = os.path.dirname(stem), os.path.basename(stem)
        try:
            load_checkpoint_dqn(ag, weights_dir, tag)
            if hasattr(ag, "epsilon"): ag.epsilon = 0.0
            return ag
        except Exception as e:
            tried.append(f"{stem} (state_size={state_size}): {e}")
    _fatal("Fehler beim Laden von DQN-Gewichten.", tried=tried)

def load_agents(cfgs, game):
    info_dim    = game.information_state_tensor_shape()[0]
    obs_dim     = game.observation_tensor_shape()[0]
    num_actions = game.num_distinct_actions()
    num_players = game.num_players()

    num_ranks   = ranks_for_deck(int(GAME_SETTINGS["deck_size"]))
    base_info   = num_ranks + (num_players - 1) + 3
    full_info   = base_info + num_ranks

    agents = []
    for pid, cfg in enumerate(cfgs):
        kind = cfg["type"]

        if kind == "ppo":
            if not all(k in cfg for k in ("family","version","episode")):
                _fatal(f"PPO-Spieler P{pid}: 'family', 'version' und 'episode' sind Pflicht. Erhalten: {cfg}")
            # Versuche (full/base) x (mit/ohne Seat-One-Hot)
            tried = []
            for idim, seatdim in [(full_info, num_players), (full_info, 0), (base_info, num_players), (base_info, 0)]:
                try:
                    ag = _load_ppo_agent(idim, num_actions, seatdim,
                                         family=cfg["family"], version=cfg["version"], episode=cfg["episode"],
                                         from_pid=cfg.get("from_pid", pid))
                    break
                except SystemExit as e:
                    tried.append(str(e))
                    ag = None
            if ag is None:
                _fatal("PPO-Checkpoint konnte mit keiner Dim geladen werden.", tried=tried)
            agents.append(ag); continue

        if kind == "dqn":
            if not all(k in cfg for k in ("family","version","episode")):
                _fatal(f"DQN-Spieler P{pid}: 'family', 'version' und 'episode' sind Pflicht. Erhalten: {cfg}")
            ag = _load_dqn_agent(obs_dim, game.num_distinct_actions(),
                                 family=cfg["family"], version=cfg["version"], episode=cfg["episode"],
                                 from_pid=cfg.get("from_pid", pid), num_players=num_players)
            agents.append(ag); continue
        
        if kind == "v_table":
            ag = v_table_agent.ValueTableAgent("agents/tables/v_table_4_4_4")
            agents.append(ag)
            continue

        if kind in STRATS:
            agents.append(STRATS[kind]); continue

        _fatal(f"Unbekannter Agententyp bei P{pid}: {kind!r}")

    return agents

def _ppo_logits(agent: ppo.PPOAgent, info_state_1d: np.ndarray, seat_id: int, num_players: int):
    # immer Seat-OneHot bauen; _make_input ignoriert sie, wenn _seat_id_dim==0
    seat_oh = np.zeros(num_players, dtype=np.float32); seat_oh[seat_id] = 1.0
    # Wichtig: auf die erwartete Basislänge kürzen (kompatibel V1/V2)
    base = info_state_1d[:getattr(agent, "_base_state_dim", len(info_state_1d))]
    x = agent._make_input(base, seat_one_hot=seat_oh)
    with torch.no_grad():
        return agent._policy(x).detach().cpu().numpy()

def _dqn_qvalues(agent: dqn.DQNAgent, obs_1d: np.ndarray, seat_id: int, num_players: int):
    # robust: ggf. um Seat-OneHot erweitern, falls Netz mehr Eingänge erwartet
    x = np.asarray(obs_1d, dtype=np.float32)
    try:
        in_features = agent.q_network.net[0].in_features
    except Exception:
        in_features = x.shape[0]
    if x.shape[0] < in_features:
        extra = in_features - x.shape[0]
        if extra == num_players:
            seat_oh = np.zeros(num_players, dtype=np.float32); seat_oh[seat_id] = 1.0
            x = np.concatenate([x, seat_oh], axis=0)
        else:
            x = np.concatenate([x, np.zeros(extra, dtype=np.float32)], axis=0)
    elif x.shape[0] > in_features:
        x = x[:in_features]
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32, device=getattr(agent, "device", "cpu")).unsqueeze(0)
        q = agent.q_network(xt).squeeze(0).detach().cpu().numpy()
    return q

def choose_policy_action(agent, state, player, num_players, num_actions):
    legal = state.legal_actions(player)

    # PPO: greedy (argmax über masked softmax der Logits)
    if isinstance(agent, ppo.PPOAgent):
        info_vec = np.array(state.information_state_tensor(player), dtype=np.float32)
        logits = _ppo_logits(agent, info_vec, player, num_players)
        probs  = _masked_softmax_numpy(logits, legal)
        return AgentOut(int(np.argmax(probs))), probs, logits

    # DQN: greedy (argmax der Qs). Für Logging: Softmax über legalen Qs als Pseudo-Prob.
    if isinstance(agent, dqn.DQNAgent):
        obs_vec = np.array(state.observation_tensor(player), dtype=np.float32)
        qvals = _dqn_qvalues(agent, obs_vec, player, num_players)
        probs = _masked_softmax_numpy(qvals, legal)
        return AgentOut(int(np.argmax(probs))), probs, qvals

    # Heuristik
    if callable(agent):
        a = int(agent(state))
        if a not in legal: a = int(np.random.choice(legal))
        probs = np.zeros(num_actions, dtype=np.float32)
        probs[legal] = 1.0 / max(1, len(legal))
        scores = probs.copy()
        return AgentOut(a), probs, scores

    _fatal("Unbekannter Agententyp in choose_policy_action.")

# ====== Initiale Hände (CSV) ======
def write_initial_ist_csv(game, state, csv_dir, filename="initial_info_state_tensor.csv"):
    info_dim = game.information_state_tensor_shape()[0]
    rows = []
    for pid in range(game.num_players()):
        ist = state.information_state_tensor(pid)
        vec = list(map(float, ist))[:info_dim]
        row = {"player": pid}
        row.update({f"f{i}": vec[i] for i in range(info_dim)})
        rows.append(row)
    cols = ["player"] + [f"f{i}" for i in range(info_dim)]
    out = os.path.join(csv_dir, filename)
    pd.DataFrame(rows, columns=cols).to_csv(out, index=False)
    print(f"🧮 Initialer InformationStateTensor gespeichert: {out}")

# ====== 1-Spiel-Debug ======
def main():
    game = pyspiel.load_game("president", GAME_SETTINGS)
    NUM_PLAYERS = game.num_players()
    NUM_ACTIONS = game.num_distinct_actions()

    # --- Agents laden (bricht ggf. mit FATAL ab) ---
    agents = load_agents(PLAYER_CONFIG, game)

    # --- Run-Verzeichnisse jetzt (erst nach erfolgreichem Laden) anlegen ---
    eval_root = os.path.join(BASE_DIR, "eval_micro")
    existing = sorted([d for d in os.listdir(eval_root)] if os.path.isdir(eval_root) else [])
    existing = [d for d in existing if d.startswith("eval_micro_")]
    next_run_num = int(existing[-1].split("_")[-1]) + 1 if existing else 1

    run_dir   = os.path.join(eval_root, f"eval_micro_{next_run_num:02d}")
    csv_dir   = os.path.join(run_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    CONFIG_CSV = os.path.join(run_dir, "config.csv")
    LOG_CSV    = os.path.join(csv_dir, "game_log.csv")
    PROBS_CSV  = os.path.join(csv_dir, "action_probs.csv")

    with open(os.path.join(run_dir, "game_settings.json"), "w") as f:
        json.dump(GAME_SETTINGS, f, indent=2)
    pd.DataFrame(PLAYER_CONFIG).to_csv(os.path.join(run_dir, "player_config.csv"), index=False)
    print(f"📁 Eval-Micro Run-Ordner: {run_dir}")

    pd.DataFrame([{
        "game_settings": json.dumps(GAME_SETTINGS),
        "players": json.dumps(PLAYER_CONFIG),
    }]).to_csv(CONFIG_CSV, index=False)

    # --- Spiel starten ---
    rows_log = []
    rows_probs = []  # NEU: pro Zug Wahrscheinlichkeiten über LEGAL actions
    state = game.new_initial_state()

    # Initiale Hände (vor dem ersten Zug) dumpen
    write_initial_ist_csv(game, state, csv_dir)
    turn = 0

    while not state.is_terminal():
        pid = state.current_player()
        legal = state.legal_actions(pid)
        legal_txt = [state.action_to_string(pid, a) for a in legal]

        agent = agents[pid]
        ao, probs, scores = choose_policy_action(agent, state, pid, NUM_PLAYERS, NUM_ACTIONS)
        action = int(ao.action)
        action_txt = state.action_to_string(pid, action)

        # --- Haupt-Log ---
        rows_log.append({
            "turn": turn, "player": pid,
            "legal_actions": str(list(zip(legal, legal_txt))),
            "chosen_action": f"{action} ({action_txt})",
        })

        # --- Probs-Log (nur mögliche/legale Aktionen) ---
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

    # Finale Returns
    rets = state.returns()
    rows_log.append({"turn": turn, "player": "terminal", "legal_actions": "", "chosen_action": "",})

    # --- Dateien schreiben ---
    pd.DataFrame(rows_log).to_csv(LOG_CSV, index=False)
    pd.DataFrame(rows_probs,
                 columns=["turn","player","action_id","action_text","prob","score","chosen"]
                 ).to_csv(PROBS_CSV, index=False)
    print(f"📄 Game-Log gespeichert: {LOG_CSV}")
    print(f"📄 Aktions-Wahrscheinlichkeiten gespeichert: {PROBS_CSV}")

    summary = {
        "micro_id": next_run_num,
        "num_turns": turn,
        "returns": rets,
    }
    # Save full OpenSpiel action history (incl. chance outcomes)
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(state.history(), f)

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
if __name__ == "__main__":
    main()
