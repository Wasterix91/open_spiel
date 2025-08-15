import os, json
import numpy as np
import pandas as pd
import torch
import pyspiel
import matplotlib.pyplot as plt

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

"""
# Für ein schnelles Debug: P0=ppo (optional laden), andere Heuristiken
PLAYER_CONFIG = [
    {"name": "Player0", "type": "ppo", "version": "01", "episode": None},
    {"name": "Player1", "type": "max_combo"},
    {"name": "Player2", "type": "random2"},
    {"name": "Player3", "type": "single_only"},
]
"""

PLAYER_CONFIG = [
    {"name": "Player0", "type": "dqn", "version": "05", "episode": "10000"}, 
    {"name": "Player1", "type": "ppo", "version": "03", "episode": "10000", "from_pid": 0},
    {"name": "Player2", "type": "ppo", "version": "06", "episode": "20_000", "from_pid": 0},
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

# Optional: gleiche Metadateien wie im Macro
with open(os.path.join(RUN_DIR, "game_settings.json"), "w") as f:
    json.dump(GAME_SETTINGS, f, indent=2)

pd.DataFrame(PLAYER_CONFIG).to_csv(os.path.join(RUN_DIR, "player_config.csv"), index=False)
print(f"📁 Eval-Micro Run-Ordner: {RUN_DIR}")


# ====== Heuristics ======

def _norm_episode(ep):
    if ep is None:
        raise RuntimeError("Episode muss explizit gesetzt sein (kein Fallback).")
    s = str(ep).replace("_", "").strip()
    if s.isdigit():
        return int(s)
    raise ValueError(f"Ungültige Episode: {ep!r}")

def _root_here(file=__file__):
    return os.path.dirname(os.path.dirname(os.path.abspath(file)))

def model_base(kind, version, pid_runtime, episode, pid_files=None, *, file=__file__):
    pid_disk = pid_files if pid_files is not None else pid_runtime
    base = os.path.join(_root_here(file), "models", f"{kind}_model_{version}", "train",
                        f"{kind}_model_{version}_agent_p{pid_disk}")
    ep = _norm_episode(episode)
    return f"{base}_ep{ep:07d}"



# ====== Loader ======
def load_agents(cfgs, game):
    info_dim = game.information_state_tensor_shape()[0]
    num_actions = game.num_distinct_actions()
    agents = []
    for pid, cfg in enumerate(cfgs):
        t = cfg["type"]
        if t == "ppo":
            a = ppo.PPOAgent(info_state_size=info_dim, num_actions=num_actions)
            base = model_base("ppo", cfg.get("version","01"), pid,
                            cfg.get("episode"), cfg.get("from_pid"), file=__file__)

            pol = base + "_policy.pt"
            val = base + "_value.pt"
            if not os.path.exists(pol) or not os.path.exists(val):
                raise FileNotFoundError(
                    f"PPO-Checkpoint fehlt für P{pid}: erwartet\n  {pol}\n  {val}"
                )

            a._policy.load_state_dict(torch.load(pol, map_location=a.device))
            a._value.load_state_dict(torch.load(val, map_location=a.device))
            a._policy.eval(); a._value.eval()
            agents.append(a)

        elif t == "dqn":
            a = dqn.DQNAgent(state_size=info_dim, num_actions=num_actions)  # <-- info_dim
            base = model_base("dqn", cfg.get("version","01"), pid,
                              cfg.get("episode"), cfg.get("from_pid"), file=__file__)
            a.restore(base)  # lässt bewusst Exceptions durch
            agents.append(a)

        elif t in STRATS:
            agents.append(STRATS[t])
        else:
            raise ValueError(f"Unbekannter type: {t}")
    return agents

def masked_probs(agent, logits, legal):
    mask = torch.zeros_like(logits)
    mask[legal] = 1.0
    p = ppo.masked_softmax(logits, mask)
    return p.detach().cpu().numpy()

# ====== 1-Spiel-Debug ======
def main():
    game = pyspiel.load_game("president", GAME_SETTINGS)
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

        # Aktion wählen + Debug
        if callable(agents[pid]):
            action = int(agents[pid](state))
            probs_info = ""
        elif isinstance(agents[pid], ppo.PPOAgent):
            obs = state.information_state_tensor(pid)
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = agents[pid]._policy(x)[0]
            probs = masked_probs(agents[pid], logits, legal)
            action = int(np.random.choice(len(probs), p=probs))
            probs_info = "; ".join(f"{i}:{p:.2f}" for i, p in enumerate(probs) if p > 0)
        elif isinstance(agents[pid], dqn.DQNAgent):
            obs = state.information_state_tensor(pid)                     # <-- info_state
            action = int(agents[pid].select_action(np.array(obs, dtype=np.float32),
                                                legal))
            probs_info = ""


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
