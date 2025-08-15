import os
import numpy as np
import torch
import pyspiel

from agents import ppo_agent as ppo
from agents import dqn_agent as dqn

from utils import STRATS


# ===== Ausgabe für InfoState steuern =====
SHOW_INFOSTATE = True     # auf False setzen, wenn du die Ausgabe mal nicht willst
INFOSTATE_DECIMALS = 3    # Rundung der Werte
INFOSTATE_PER_LINE = 16   # wie viele Werte pro Zeile drucken


# Einfaches CLI, in dem du an Sitzplatz 0 spielst.
# Gegner: Heuristiken oder geladene Policies (einfach unten einstellen).
GAME_SETTINGS = {
    "num_players": 4,
    "deck_size": "64",
    "shuffle_cards": True,
    "single_card_mode": False,
}

OPPONENTS = [
    {"type": "max_combo"},  # Player1
    {"type": "random2"},    # Player2
    {"name": "Player3", "type": "ppo", "version": "03", "episode": "10000"} # Player3
]

def _root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _norm_episode(ep):
    """Erzwingt explizite Episode; erlaubt int oder '10_000' als String."""
    if ep is None:
        raise RuntimeError("In dieser Version muss 'episode' explizit gesetzt sein (kein Auto-Fallback).")
    if isinstance(ep, int):
        return ep
    if isinstance(ep, str):
        s = ep.replace("_", "").strip()
        if s.isdigit():
            return int(s)
    raise ValueError(f"Ungültige Episode: {ep!r} (erwartet int oder numerischen String)")

def model_base(kind, version, pid, episode=None):
    base = os.path.join(_root(), "models", f"{kind}_model_{version}", "train",
                        f"{kind}_model_{version}_agent_p{pid}")
    ep = _norm_episode(episode)
    return f"{base}_ep{ep:07d}" if ep is not None else base




def load_opponent(cfg, pid, game):
    info_dim = game.information_state_tensor_shape()[0]
    obs_dim  = game.observation_tensor_shape()[0]
    A = game.num_distinct_actions()
    if cfg["type"] == "ppo":
        agent = ppo.PPOAgent(info_state_size=info_dim, num_actions=A)
        base = model_base("ppo", cfg.get("version","01"), pid, cfg.get("episode"))
        try:
            agent._policy.load_state_dict(torch.load(base + "_policy.pt", map_location=agent.device))
            agent._policy.eval()
        except FileNotFoundError:
            print(f"[WARN] Kein PPO-Checkpoint für P{pid}, nutze random Init.")
        return agent
    if cfg["type"] == "dqn":
        agent = dqn.DQNAgent(state_size=obs_dim, num_actions=A)
        base = model_base("dqn", cfg.get("version","01"), pid)
        try:
            agent.restore(base + ".pt")
        except FileNotFoundError:
            print(f"[WARN] Kein DQN-Checkpoint für P{pid}.")
        return agent
    # heuristics
    def random2(state):
        legal = state.legal_actions()
        if len(legal) > 1 and 0 in legal:
            legal = [a for a in legal if a != 0]
        return int(np.random.choice(legal))
    def max_combo(state):
        pid = state.current_player()
        dec = [(a, state.action_to_string(pid, a)) for a in state.legal_actions()]
        if not dec: return 0
        def cs(s):
            return 4 if "Quad" in s else 3 if "Triple" in s else 2 if "Pair" in s else 1
        return max(dec, key=lambda x: (cs(x[1]), -x[0]))[0]
    def single_only(state):
        pid = state.current_player()
        dec = [(a, state.action_to_string(pid, a)) for a in state.legal_actions()]
        singles = [x for x in dec if "Single" in x[1]]
        return singles[0][0] if singles else 0
    STRATS = {"random2": random2, "max_combo": max_combo, "single_only": single_only}
    return STRATS[cfg["type"]]

def choose_by_agent(agent, state, pid):
    legal = state.legal_actions(pid)
    if callable(agent):
        a = int(agent(state))
        return a if a in legal else int(np.random.choice(legal))
    if isinstance(agent, ppo.PPOAgent):
        obs = state.information_state_tensor(pid)
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = agent._policy(x)[0]
        mask = torch.zeros_like(logits); mask[legal] = 1.0
        probs = ppo.masked_softmax(logits, mask).cpu().numpy()
        return int(np.random.choice(len(probs), p=probs))
    if isinstance(agent, dqn.DQNAgent):
        obs = state.observation_tensor(pid)
        return int(agent.select_action(obs, legal))
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

def main():
    game = pyspiel.load_game("president", GAME_SETTINGS)
    opponents = [load_opponent(OPPONENTS[i-1], i, game) for i in [1,2,3]]

    state = game.new_initial_state()
    print("\n== Human Play: Du bist Player 0 ==")
    print("Mit Enter bestätigst du die Zahl neben der gewünschten Action-ID.\n")

    while not state.is_terminal():
        pid = state.current_player()
        if pid == 0:
            legal = state.legal_actions(0)

            # <<< NEU: InfoState immer anzeigen >>>
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
            action = choose_by_agent(opponents[pid-1], state, pid)

        print(f"P{pid} spielt: {state.action_to_string(pid, action)}")
        state.apply_action(action)

    print("\n== Spielende ==")
    print("Returns:", state.returns())
    best = np.argmax(state.returns())
    print(f"Sieger: Player {best}")

if __name__ == "__main__":
    main()
