import pyspiel
import numpy as np
import torch
import collections
import os

import ppo_agent_self as ppo
import dqn_agent as dqn
import td_agent as td

# =============================
# 0) Einstellungen
# =============================

# === Spiel erstellen ===
game = pyspiel.load_game(
    "president",
    {
        "deck_size": "64",
        "shuffle_cards": False,
        "single_card_mode": False,
        "num_players": 4
    }
)

state = game.new_initial_state()

print("=== President Game ===")
print(f"Num players: {game.num_players()}")
print(f"Num distinct actions: {game.num_distinct_actions()}")
print(f"Observation tensor shape: {game.observation_tensor_shape()}")

params = game.get_parameters()
print(f"shuffle_cards: {params['shuffle_cards']}")
print(f"single_card_mode: {params['single_card_mode']}")
print(f"deck_size: {params['deck_size']}")

# =============================
# 1) Agenten-Konfiguration
# =============================
PLAYER_CONFIG = [
    {"name": "Player0", "type": "human"},                   # <- du spielst auf Sitz 0
    {"name": "Player1", "type": "ppo", "version": "57", "episode": 90_000},
    {"name": "Player2", "type": "ppo", "version": "57", "episode": 90_000},
    {"name": "Player3", "type": "random2"},
]

# Passe den Pfad bei dir an:
MODEL_ROOT = "/home/wasterix/OpenSpiel/open_spiel/Playground/models"

# =============================
# 2) Kartenränge (dynamisch)
# =============================
def get_ranks(deck_size):
    if deck_size == "32":
        return ["7", "8", "9", "10", "J", "Q", "K", "A"]
    elif deck_size == "52":
        return ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
    elif deck_size == "64":
        return ["7", "8", "9", "10", "J", "Q", "K", "A"]
    else:
        raise ValueError(f"Unknown deck_size: {deck_size}")

RANKS = get_ranks(params['deck_size'])
RANK_TO_NUM = {rank: i for i, rank in enumerate(RANKS)}

def parse_combo_size(text):
    if "Single" in text: return 1
    if "Pair" in text: return 2
    if "Triple" in text: return 3
    if "Quad" in text: return 4
    return 1

def parse_rank(text):
    try:
        return RANK_TO_NUM[text.split()[-1]]
    except KeyError:
        return -1

def decode_actions(state):
    player = state.current_player()
    actions = state.legal_actions()
    return [(a, state.action_to_string(player, a)) for a in actions if a != 0]

def max_combo(state):
    decoded = decode_actions(state)
    return max(decoded, key=lambda x: parse_combo_size(x[1]))[0] if decoded else 0

def aggressive_strategy(state):
    decoded = decode_actions(state)
    if not decoded:
        return 0
    return max(decoded, key=lambda x: parse_rank(x[1]))[0]

def single_only(state):
    decoded = decode_actions(state)
    singles = [x for x in decoded if "Single" in x[1]]
    return min(singles, key=lambda x: x[0])[0] if singles else 0

def random_action(state):
    return np.random.choice(state.legal_actions())

def random2_action_strategy(state):
    legal = state.legal_actions()
    if len(legal) > 1 and 0 in legal:
        legal = [a for a in legal if a != 0]
    return np.random.choice(legal)

def smart_strategy(state):
    decoded = decode_actions(state)
    if not decoded: return 0
    groups = {1: [], 2: [], 3: [], 4: []}
    for a, s in decoded:
        size = parse_combo_size(s)
        groups[size].append((a, s))
    for size in [4, 3, 2, 1]:
        if groups[size]:
            return min(groups[size], key=lambda x: parse_rank(x[1]))[0]
    return 0

strategy_map = {
    "random": random_action,
    "random2": random2_action_strategy,
    "max_combo": max_combo,
    "single_only": single_only,
    "smart": smart_strategy,
    "aggressive": aggressive_strategy,
}

# =============================
# 3) PPO: Episode-spezifisches Laden
# =============================
def _pad_episode(ep: int, width: int = 7) -> str:
    return f"{ep:0{width}d}"

def _load_ppo_episode(agent, model_root, version, pid, episode: int):
    ep_tag = _pad_episode(episode)
    prefix = os.path.join(
        model_root,
        f"ppo_model_{version}",
        "train",
        f"ppo_model_{version}_agent_p{pid}_ep{ep_tag}"
    )
    policy_path = f"{prefix}_policy.pt"
    value_path  = f"{prefix}_value.pt"

    if not (os.path.exists(policy_path) and os.path.exists(value_path)):
        raise FileNotFoundError(f"Episode-Files nicht gefunden: {policy_path} / {value_path}")

    map_loc = agent.device if hasattr(agent, "device") else "cpu"
    policy_sd = torch.load(policy_path, map_location=map_loc)
    value_sd  = torch.load(value_path,  map_location=map_loc)

    agent._policy.load_state_dict(policy_sd)
    agent._value.load_state_dict(value_sd)

    agent._policy.to(map_loc).eval()
    agent._value.to(map_loc).eval()
    agent._loaded_episode = episode

# =============================
# 4) Agenten laden (inkl. "human")
# =============================
agents = []
for pid, cfg in enumerate(PLAYER_CONFIG):
    kind = cfg["type"]
    version = cfg.get("version", "01")
    episode = cfg.get("episode", None)

    if kind == "human":
        agents.append("human")
        continue

    if kind == "ppo":
        # Basis-Agent
        model_path = os.path.join(MODEL_ROOT, f"ppo_model_{version}", "train", f"ppo_model_{version}_agent_p{pid}")
        agent = ppo.PPOAgent(
            info_state_size=game.information_state_tensor_shape()[0],
            num_actions=game.num_distinct_actions()
        )
        # Wenn Episode angegeben, zuerst versuchen, die episodenspezifischen Gewichte zu laden
        if episode is not None:
            try:
                _load_ppo_episode(agent, MODEL_ROOT, version, pid, int(episode))
            except Exception as e:
                print(f"[WARN] PPO Episode {episode} konnte nicht geladen werden (Player {pid}): {e}")
                print("       -> Fallback auf Standard-Checkpoint (restore)")
                agent.restore(model_path)
        else:
            agent.restore(model_path)
        agents.append(agent)

    elif kind == "dqn":
        model_path = os.path.join(MODEL_ROOT, f"dqn_model_{version}", "train", f"dqn_model_{version}_agent_p{pid}")
        agent = dqn.DQNAgent(
            state_size=game.observation_tensor_shape()[0],
            num_actions=game.num_distinct_actions()
        )
        agent.restore(model_path)
        agents.append(agent)

    elif kind == "td":
        model_path = os.path.join(MODEL_ROOT, f"td_model_{version}", "train", f"td_model_{version}_agent_p{pid}.pt")
        agent = td.TDAgent(
            obs_size=game.information_state_tensor_shape()[0],
            num_actions=game.num_distinct_actions()
        )
        agent.load(model_path)
        agents.append(agent)

    elif kind in strategy_map:
        agents.append(strategy_map[kind])

    else:
        raise ValueError(f"Unbekannter Spielertyp: {kind}")

# =============================
# 5) Aktionswahl
# =============================
def choose_policy_action(agent, state, player):
    legal = state.legal_actions(player)

    if isinstance(agent, ppo.PPOAgent):
        obs = state.information_state_tensor(player)
        info_tensor = torch.tensor(obs, dtype=torch.float32).to(agent.device).unsqueeze(0)
        with torch.no_grad():
            logits = agent._policy(info_tensor)[0].cpu().numpy()

        # In choose_policy_action() im PPO-Zweig
        masked = np.zeros_like(logits, dtype=np.float64)
        masked[legal] = logits[legal]
        s = masked.sum()
        probs = masked / s if s > 0 else np.zeros_like(masked)
        legal_probs = probs[legal]
        if legal_probs.sum() == 0:
            legal_probs = np.ones_like(legal_probs) / len(legal)
        else:
            legal_probs = legal_probs / legal_probs.sum()
        action = np.random.choice(legal, p=legal_probs)
        return collections.namedtuple("AgentOutput", ["action", "probs"])(action=action, probs=probs)


    elif isinstance(agent, dqn.DQNAgent):
        obs = state.observation_tensor(player)
        action = agent.select_action(obs, legal)
        return collections.namedtuple("AgentOutput", ["action", "probs"])(action=action, probs=None)

    elif isinstance(agent, td.TDAgent):
        obs = state.information_state_tensor(player)
        with torch.no_grad():
            q_values = agent.predict(torch.tensor(obs, dtype=torch.float32).to(agent.device))
        legal_qs = q_values[legal]
        action = legal[int(torch.argmax(legal_qs).item())]
        return collections.namedtuple("AgentOutput", ["action", "probs"])(action=action, probs=None)

    elif callable(agent):
        action = agent(state)
        if action not in state.legal_actions():
            action = np.random.choice(state.legal_actions())
        return collections.namedtuple("AgentOutput", ["action", "probs"])(action=action, probs=None)

    else:
        raise ValueError("Unbekannter Agententyp bei choose_policy_action")

# =============================
# 6) Human-Input
# =============================
def prompt_human_action(state, player):
    legal = state.legal_actions(player)
    labels = [state.action_to_string(player, a) for a in legal]
    print("\nDein Zug! Wähle eine Aktion:")
    for idx, (aid, lab) in enumerate(zip(legal, labels), start=1):
        print(f"  {idx:2d}) {lab}  [id={aid}]")
    print("  Tipp: Zahl eingeben (Index), oder direkt Action-ID.")

    while True:
        raw = input("Deine Wahl: ").strip().lower()
        if raw == "":
            continue
        # Versuche Index
        if raw.isdigit():
            val = int(raw)
            # erst als Index, dann als Action-ID interpretieren
            if 1 <= val <= len(legal):
                return legal[val - 1]
            if val in legal:
                return val
        print("Ungültige Eingabe – bitte erneut versuchen.")

# =============================
# 7) Spiel ausführen
# =============================
for move in range(300):
    if state.is_terminal():
        break

    player = state.current_player()
    actions = state.legal_actions()
    action_strs = [state.action_to_string(player, a) for a in actions]

    print(f"\n=== Runde {move + 1} ===")
    print(state)
    print(f"Player {player} legal actions: {action_strs}")

    agent_or_strategy = agents[player]

    if agent_or_strategy == "human":
        chosen = prompt_human_action(state, player)
    elif callable(agent_or_strategy):
        chosen = agent_or_strategy(state)
    else:
        agent_out = choose_policy_action(agent_or_strategy, state, player)
        chosen = agent_out.action
        if agent_out.probs is not None:
            # nur die für den Spieler relevanten (legalen) Wahrscheinlichkeiten anzeigen
            legals = state.legal_actions(player)
            probs_show = np.array([agent_out.probs[a] for a in legals], dtype=float)
            probs_show /= probs_show.sum() if probs_show.sum() > 0 else 1
            print("Policy (legal) ~", np.round(probs_show, 3))

    print(f"{PLAYER_CONFIG[player]['name']} wählt: {state.action_to_string(player, chosen)}")
    state.apply_action(chosen)

# === Ergebnis anzeigen ===
if state.is_terminal():
    print("\nSpiel beendet.\n")
    print(state)
    print("Returns:", state.returns())
