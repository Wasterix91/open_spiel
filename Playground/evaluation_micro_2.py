import pyspiel
import numpy as np
import torch
import collections

import ppo_agent as ppo
import dqn_agent as dqn
import td_agent as td

# === 1️⃣ Spiel erstellen ===
game = pyspiel.load_game(
    "president",
    {
        "deck_size": "32",
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

# === 🔢 Agentenkonfiguration ===
PLAYER_CONFIG = [
    {"name": "Player0", "type": "ppo", "version": "14"},
    {"name": "Player1", "type": "random2"},
    {"name": "Player2", "type": "random2"},
    {"name": "Player3", "type": "random2"}
]
MODEL_ROOT = "/home/wasterix/OpenSpiel/open_spiel/Playground/models"

# === Kartenränge (dynamisch) ===
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

def random2_action_strategy(state):
    legal = state.legal_actions()
    if len(legal) > 1 and 0 in legal:
        legal = [a for a in legal if a != 0]
    return np.random.choice(legal)

strategy_map = {
    "random": random_action,
    "random2": random2_action_strategy,
    "max_combo": max_combo,
    "single_only": single_only,
    "smart": smart_strategy,
}

# === 2️⃣ Agenten laden ===
agents = []
for pid, cfg in enumerate(PLAYER_CONFIG):
    kind = cfg["type"]
    version = cfg.get("version", "01")

    if kind == "ppo":
        model_path = f"{MODEL_ROOT}/ppo_model_{version}/train/ppo_model_{version}_agent_p{pid}"
        agent = ppo.PPOAgent(
            info_state_size=game.information_state_tensor_shape()[0],
            num_actions=game.num_distinct_actions()
        )
        agent.restore(model_path)
        agents.append(agent)

    elif kind == "dqn":
        model_path = f"{MODEL_ROOT}/dqn_model_{version}/train/dqn_model_{version}_agent_p{pid}"
        agent = dqn.DQNAgent(
            state_size=game.observation_tensor_shape()[0],
            num_actions=game.num_distinct_actions()
        )
        agent.restore(model_path)
        agents.append(agent)

    elif kind == "td":
        model_path = f"{MODEL_ROOT}/td_model_{version}/train/td_model_{version}_agent_p{pid}.pt"
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

# === 3️⃣ Auswahlfunktion für Aktionen ===
def choose_policy_action(agent, state, player):
    legal = state.legal_actions(player)

    if isinstance(agent, ppo.PPOAgent):
        obs = state.information_state_tensor(player)
        info_tensor = torch.tensor(obs, dtype=torch.float32).to(agent.device)
        logits = agent._policy(info_tensor).detach().cpu().numpy()
        masked = np.zeros_like(logits)
        masked[legal] = logits[legal]
        probs = masked / masked.sum() if masked.sum() > 0 else np.ones_like(masked) / len(legal)
        return collections.namedtuple("AgentOutput", ["action", "probs"])(action=np.argmax(probs), probs=probs)

    elif isinstance(agent, dqn.DQNAgent):
        obs = state.observation_tensor(player)
        action = agent.select_action(obs, legal)
        return collections.namedtuple("AgentOutput", ["action", "probs"])(action=action, probs=None)

    elif isinstance(agent, td.TDAgent):
        obs = state.information_state_tensor(player)
        q_values = agent.predict(torch.tensor(obs, dtype=torch.float32))
        legal_qs = q_values[legal]
        action = legal[torch.argmax(legal_qs).item()]
        return collections.namedtuple("AgentOutput", ["action", "probs"])(action=action, probs=None)

    else:
        raise ValueError("Unbekannter Agententyp bei choose_policy_action")

# === 4️⃣ Spiel ausführen ===
for move in range(300):
    if state.is_terminal():
        break

    player = state.current_player()
    actions = state.legal_actions()
    action_strs = [state.action_to_string(player, a) for a in actions]

    print(f"\n=== Runde {move + 1} ===")
    print(state)
    print(f"Observation Tensor Player {player}: {state.observation_tensor()}")
    print(f"Player {player} legal actions: {action_strs}")

    agent_or_strategy = agents[player]
    if callable(agent_or_strategy):
        chosen = agent_or_strategy(state)
    else:
        agent_out = choose_policy_action(agent_or_strategy, state, player)
        chosen = agent_out.action
        if agent_out.probs is not None:
            print(f"Policy-Probs: {np.round(agent_out.probs, 2)}")

    print(f"{PLAYER_CONFIG[player]['name']} wählt: {state.action_to_string(player, chosen)}")
    state.apply_action(chosen)

# === 5️⃣ Ergebnis anzeigen ===
if state.is_terminal():
    print("\nSpiel beendet.\n")
    print(state)
    print("Returns:", state.returns())
