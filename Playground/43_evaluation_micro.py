import pyspiel
import numpy as np
import torch
import collections
import ppo_agent as ppo  # <- ggf. anpassen

# === 1️⃣ Spiel erstellen ===
game = pyspiel.load_game(
    "president",
    {
        "deck_size": "32",
        "shuffle_cards": True,
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

# === 🔢 Versionsnummer definieren ===
VERSION_NUM = "18"  # z. B. Eingabe über CLI oder oben ändern

# === 2️⃣ Agenten vorbereiten ===
PLAYER_TYPES = ["ppo", "random", "random", "random"]
MODEL_DIR = f"/home/wasterix/OpenSpiel/open_spiel/Playground/models/ppo_model_{VERSION_NUM}/train"

# Dynamisch basierend auf deck_size
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
    # Alle sind "Single" – wähle die höchste
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

strategy_map = {
    "random": random_action,
    "max_combo": max_combo,
    "single_only": single_only,
    "smart": smart_strategy,
}

agents = []
for pid, ptype in enumerate(PLAYER_TYPES):
    if ptype == "ppo":
        agent = ppo.PPOAgent(
            info_state_size=game.information_state_tensor_shape()[0],
            num_actions=game.num_distinct_actions()
        )
        model_path = f"{MODEL_DIR}/ppo_model_{VERSION_NUM}_agent_p{pid}"
        agent.restore(model_path)
        agents.append(agent)
    elif ptype in strategy_map:
        agents.append(strategy_map[ptype])
    else:
        raise ValueError(f"Unbekannter Spielertyp: {ptype}")

# === 3️⃣ PPO-Auswahlfunktion ===
def choose_policy_action(agent, state, player):
    info_state = state.information_state_tensor(player)
    legal_actions = state.legal_actions(player)
    info_tensor = torch.tensor(info_state, dtype=torch.float32).to(agent.device)
    logits = agent._policy(info_tensor).detach().cpu().numpy()

    masked_logits = np.zeros_like(logits)
    masked_logits[legal_actions] = logits[legal_actions]

    if masked_logits.sum() == 0:
        probs = np.zeros_like(logits)
        probs[legal_actions] = 1.0 / len(legal_actions)
    else:
        probs = masked_logits / masked_logits.sum()

    legal_probs = np.array([probs[a] for a in legal_actions])
    return collections.namedtuple("AgentOutput", ["action", "probs", "legal_actions"])(
        action=np.argmax(probs), probs=legal_probs, legal_actions=legal_actions)

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
    if isinstance(agent_or_strategy, ppo.PPOAgent):
        agent_out = choose_policy_action(agent_or_strategy, state, player)
        chosen = agent_out.action
        action_labels = [state.action_to_string(player, a) for a in agent_out.legal_actions]
        print(f"Policy-Probs: {np.round(agent_out.probs, 2)}")
    else:
        chosen = agent_or_strategy(state)

    print(f"Player {player} wählt: {state.action_to_string(player, chosen)}")
    state.apply_action(chosen)

# === 5️⃣ Ergebnis anzeigen ===
if state.is_terminal():
    print()
    print("Spiel beendet.")
    print()
    print(state)
    print("Returns:", state.returns())
