# -*- coding: utf-8 -*-
# evaluation/eval_micro.py — Single-Game Debug/Eval
# - Lädt PPO/DQN Checkpoints robust (mit/ohne Seat-One-Hot; DQN auch League-Checkpoints)
# - Hängt bei Bedarf automatisch eine Seat-One-Hot an (nur wenn das Netz sie erwartet)
# - Loggt ein einzelnes Spiel detailliert in CSV + Mini-Plot

import os, json, glob, numpy as np, pandas as pd, torch, pyspiel, matplotlib.pyplot as plt
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
    {"name": "Player0", "type": "ppo", "version": "15", "episode": "20_000"}, 
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

# ====== Helpers ======
def _norm_episode(ep):
    if ep is None:
        return None
    s = str(ep).replace("_", "").strip()
    if s.isdigit():
        return int(s)
    raise ValueError(f"Ungültige Episode: {ep!r}")

def _root_here(file=__file__):
    return os.path.dirname(os.path.dirname(os.path.abspath(file)))

# Basispfad für *Modell*-Familien aufbauen (ohne Suffixe)
def _model_bases(kind_family: str, version: str, pid_runtime: int, episode: int|None, pid_files: int|None, *, file=__file__):
    # kind_family ∈ {"ppo_model", "dqn_model", "dqn_league"}
    pid_disk = pid_files if pid_files is not None else pid_runtime
    base = os.path.join(_root_here(file), "models", f"{kind_family}_{version}", "train",
                        f"{kind_family}_{version}_agent_p{pid_disk}")
    if episode is not None:
        return [f"{base}_ep{episode:07d}"]
    else:
        # sowohl "latest"-Stamm als auch alle ep-Stämme (für Fallback)
        stems = [base]
        stems += [p[:-3] if p.endswith(".pt") else p for p in glob.glob(base + "_ep*")]
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
        version = cfg.get("version", "01")
        episode = _norm_episode(cfg.get("episode"))
        from_pid = cfg.get("from_pid")

        if kind == "ppo":
            # Versuche erst ohne, dann mit Seat-One-Hot zu laden
            bases = _model_bases("ppo_model", version, pid, episode, from_pid, file=__file__)
            loaded = None
            for seat_id_dim in (0, NUM_PLAYERS):
                ag = ppo.PPOAgent(info_state_size=INFO_DIM, num_actions=num_actions, seat_id_dim=seat_id_dim)
                ok=False
                for stem in bases:
                    try:
                        ag._policy.load_state_dict(torch.load(stem + "_policy.pt", map_location=ag.device))
                        ag._value.load_state_dict(torch.load(stem + "_value.pt",  map_location=ag.device))
                        ag._policy.eval(); ag._value.eval()
                        ok=True; break
                    except FileNotFoundError:
                        continue
                    except RuntimeError:
                        # Shape mismatch → nächsten seat_id_dim probieren
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
            # Probiere beide DQN-Familien + zwei mögliche State-Sizes (ohne/mit Seat-One-Hot)
            families = ("dqn_model", "dqn_league")
            stems = []
            for fam in families:
                stems.extend(_model_bases(fam, version, pid, episode, from_pid, file=__file__))
            # Dedupliziere Reihenfolge beibehaltend
            seen=set(); stems=[s for s in stems if not (s in seen or seen.add(s))]

            loaded = None
            for state_size in (OBS_DIM, OBS_DIM + NUM_PLAYERS):
                ag = dqn.DQNAgent(state_size=state_size, num_actions=num_actions)
                ok=False
                for stem in stems:
                    try:
                        ag.restore(stem)
                        ok=True; break
                    except FileNotFoundError:
                        continue
                    except RuntimeError:
                        # Shape mismatch → andere state_size testen
                        ok=False; break
                    except Exception:
                        continue
                if ok:
                    ag.epsilon = 0.0  # greedy in Eval
                    loaded = ag; break
            if loaded is None:
                print(f"[WARN] DQN-Checkpoint für P{pid} nicht gefunden – Random-Init.")
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
