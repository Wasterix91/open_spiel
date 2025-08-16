# -*- coding: utf-8 -*-
# Interactive CLI: Du spielst Sitz 0 (Human). Gegner = Heuristiken oder geladene Policies.
# Update: robustes Laden (PPO+DQN, inkl. *_league*), Auto‑Seat‑One‑Hot wenn das Netz es erwartet,
#         greedy Eval für DQN (epsilon=0), optionaler InfoState‑Dump.

import os, glob, numpy as np, torch, pyspiel

from agents import ppo_agent as ppo
from agents import dqn_agent as dqn
from utils import STRATS

# ===== Ausgabe für InfoState steuern =====
SHOW_INFOSTATE = True     # auf False setzen, wenn du die Ausgabe mal nicht willst
INFOSTATE_DECIMALS = 3    # Rundung der Werte
INFOSTATE_PER_LINE = 16   # wie viele Werte pro Zeile drucken

# ===== Spiel-Setup =====
GAME_SETTINGS = {
    "num_players": 4,
    "deck_size": "64",
    "shuffle_cards": True,
    "single_card_mode": False,
}

# Gegner: Heuristiken oder Policies; optional "from_pid" für geteilte Gewichte
OPPONENTS = [
    {"type": "max_combo"},                     # Player1
    {"type": "random2"},                      # Player2
    {"name": "Player0", "type": "ppo", "version": "15", "episode": "20_000", "from_pid":0},  # Player3
]

"""
PLAYER_CONFIG = [
    {"name":"P0", "type":"ppo", "version":"03", "episode":10000},                 # echte p0-Gewichte
    {"name":"P1", "type":"ppo", "version":"03", "episode":10000, "from_pid":0},  # nutzt p0-Dateien
    {"name":"P2", "type":"dqn", "version":"01", "episode":None},
    {"name":"P3", "type":"max_combo"},
]

"""

# ===== Pfade/Helper =====

def _root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _norm_episode(ep):
    if ep is None:
        return None
    s = str(ep).replace("_", "").strip()
    if s.isdigit():
        return int(s)
    raise ValueError(f"Ungültige Episode: {ep!r}")

# Liefert eine Liste möglicher Modell-Stämme (ohne Suffixe), inkl. *_league* Familien
# kind_family ∈ {"ppo_model","ppo_league","dqn_model","dqn_league"}
def _model_bases(kind_family: str, version: str, pid_runtime: int, episode, pid_files: int|None, *, file=__file__):
    pid_disk = pid_files if pid_files is not None else pid_runtime
    base = os.path.join(_root(), "models", f"{kind_family}_{version}", "train",
                        f"{kind_family}_{version}_agent_p{pid_disk}")
    ep = _norm_episode(episode)
    if ep is not None:
        return [f"{base}_ep{ep:07d}"]
    # latest + alle ep‑Stämme
    stems = [base]
    stems += [p[:-3] if p.endswith(".pt") else p for p in glob.glob(base + "_ep*")]
    return stems

# ===== Loader =====

def load_opponent(cfg, pid, game):
    info_dim = game.information_state_tensor_shape()[0]
    obs_dim  = game.observation_tensor_shape()[0]
    num_actions = game.num_distinct_actions()
    NUM_PLAYERS = game.num_players()

    t = cfg.get("type")

    if t == "ppo":
        version = cfg.get("version", "01")
        episode = cfg.get("episode")
        from_pid = cfg.get("from_pid")
        # Versuche zuerst ppo_model, dann ppo_league
        stems = []
        for fam in ("ppo_model", "ppo_league"):
            stems.extend(_model_bases(fam, version, pid, episode, from_pid))
        # Dedupliziere
        seen=set(); stems=[s for s in stems if not (s in seen or seen.add(s))]

        # Zwei Versuche: ohne/mit Seat‑One‑Hot
        for seat_id_dim in (0, NUM_PLAYERS):
            ag = ppo.PPOAgent(info_state_size=info_dim, num_actions=num_actions, seat_id_dim=seat_id_dim)
            for stem in stems:
                try:
                    ag._policy.load_state_dict(torch.load(stem + "_policy.pt", map_location=ag.device))
                    ag._policy.eval()
                    # Value ist für Actionwahl nicht zwingend, aber versuchen zu laden
                    try:
                        ag._value.load_state_dict(torch.load(stem + "_value.pt", map_location=ag.device))
                        ag._value.eval()
                    except FileNotFoundError:
                        pass
                    return ag
                except FileNotFoundError:
                    continue
                except RuntimeError:
                    # Shape mismatch → anderen seat_id_dim probieren
                    break
                except Exception:
                    continue
        print(f"[WARN] Kein PPO-Checkpoint für P{pid} gefunden – Random-Init.")
        return ppo.PPOAgent(info_state_size=info_dim, num_actions=num_actions)

    if t == "dqn":
        version = cfg.get("version", "01")
        episode = cfg.get("episode")
        from_pid = cfg.get("from_pid")
        # DQN: beide Familien durchsuchen
        stems = []
        for fam in ("dqn_model", "dqn_league"):
            stems.extend(_model_bases(fam, version, pid, episode, from_pid))
        seen=set(); stems=[s for s in stems if not (s in seen or seen.add(s))]

        # Zwei Inputgrößen probieren (ohne/mit Seat‑One‑Hot)
        for state_size in (obs_dim, obs_dim + NUM_PLAYERS):
            ag = dqn.DQNAgent(state_size=state_size, num_actions=num_actions)
            for stem in stems:
                try:
                    ag.restore(stem)  # hängt selbst _qnet/_tgt an
                    ag.epsilon = 0.0
                    return ag
                except FileNotFoundError:
                    continue
                except RuntimeError:
                    # Shape mismatch → nächste state_size testen
                    break
                except Exception:
                    continue
        print(f"[WARN] Kein DQN-Checkpoint für P{pid} gefunden – Random-Init.")
        return dqn.DQNAgent(state_size=obs_dim, num_actions=num_actions)

    # Heuristiken
    if t in STRATS:
        return STRATS[t]

    raise ValueError(f"Unbekannter Gegner-Typ: {t}")

# ===== Aktionswahl =====

def choose_by_agent(agent, state, pid, NUM_PLAYERS):
    legal = state.legal_actions(pid)

    if callable(agent):
        a = int(agent(state))
        return a if a in legal else int(np.random.choice(legal))

    if isinstance(agent, ppo.PPOAgent):
        obs = np.array(state.information_state_tensor(pid), dtype=np.float32)
        # Erwartet das Netz mehr Input? Dann ggf. Seat‑One‑Hot anhängen
        try:
            in_features = agent._policy.net[0].in_features
        except Exception:
            in_features = obs.shape[0]
        if in_features > obs.shape[0]:
            extra = in_features - obs.shape[0]
            if extra == NUM_PLAYERS:
                seat_oh = np.zeros(NUM_PLAYERS, dtype=np.float32); seat_oh[pid] = 1.0
                obs = np.concatenate([obs, seat_oh], axis=0)
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = agent._policy(x)[0]
            mask = torch.zeros_like(logits); mask[legal] = 1.0
            probs = ppo.masked_softmax(logits, mask).cpu().numpy()
        return int(np.random.choice(len(probs), p=probs))

    if isinstance(agent, dqn.DQNAgent):
        obs = np.array(state.observation_tensor(pid), dtype=np.float32)
        try:
            in_features = agent.q_network.net[0].in_features
        except Exception:
            in_features = obs.shape[0]
        if in_features > obs.shape[0]:
            extra = in_features - obs.shape[0]
            if extra == NUM_PLAYERS:
                seat_oh = np.zeros(NUM_PLAYERS, dtype=np.float32); seat_oh[pid] = 1.0
                obs = np.concatenate([obs, seat_oh], axis=0)
        old_eps = getattr(agent, "epsilon", 0.0)
        agent.epsilon = 0.0
        try:
            return int(agent.select_action(obs, legal))
        finally:
            agent.epsilon = old_eps

    raise ValueError("Unbekannter Agententyp")

# ---- hübsche Ausgabe des InfoState ----

def print_info_state(state, pid, decimals=INFOSTATE_DECIMALS, per_line=INFOSTATE_PER_LINE):
    vec = np.asarray(state.information_state_tensor(pid), dtype=np.float32)
    if decimals is not None:
        vec = np.round(vec, decimals)
    print(f"[InfoState P{pid}] len={len(vec)}")
    for i in range(0, len(vec), per_line):
        chunk = vec[i:i+per_line]
        if decimals is not None:
            line = " ".join(f"{v:.{decimals}f}" for v in chunk)
        else:
            line = " ".join(str(v) for v in chunk)
        print("  " + line)

# ===== Main =====

def main():
    game = pyspiel.load_game("president", GAME_SETTINGS)
    NUM_PLAYERS = game.num_players()

    opponents = [load_opponent(OPPONENTS[i-1], i, game) for i in [1,2,3]]

    state = game.new_initial_state()
    print("\n== Human Play: Du bist Player 0 ==")
    print("Mit Enter bestätigst du die Zahl neben der gewünschten Action-ID.\n")

    while not state.is_terminal():
        pid = state.current_player()
        if pid == 0:
            legal = state.legal_actions(0)

            if SHOW_INFOSTATE:
                print_info_state(state, 0)

            print("\n--- Dein Zug ---")
            for a in legal:
                print(f"{a:3d}: {state.action_to_string(0, a)}")
            choice = None
            while choice not in legal:
                try:
                    text = input("Action-ID wählen: ").strip()
                    choice = int(text)
                except Exception:
                    choice = None
            action = choice
        else:
            action = choose_by_agent(opponents[pid-1], state, pid, NUM_PLAYERS)

        print(f"P{pid} spielt: {state.action_to_string(pid, action)}")
        state.apply_action(action)

    print("\n== Spielende ==")
    print("Returns:", state.returns())
    best = int(np.argmax(state.returns()))
    print(f"Sieger: Player {best}")

if __name__ == "__main__":
    main()
