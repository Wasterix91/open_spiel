import os
import numpy as np
import pyspiel
from open_spiel.python import rl_environment
import ppo_local as ppo

# === 1️⃣ Einstellungen ===
NUM_GAMES = 1000
MODEL_DIR = "Playground/models/selfplay_president_03/train"  # Passe ggf. an

# Spieler-Setup:
# "ppo" → lade Modell
# "max_combo", "single_only", "smart", "random" → feste Strategie
PLAYER_TYPES = ["ppo", "random", "random", "random"]

# === 2️⃣ Spiel & Umgebung ===
game = pyspiel.load_game("president", {
    "num_players": 4,
    "deck_size": "32",
    "shuffle_cards": True,
    "single_card_mode": False,
})
env = rl_environment.Environment(game)
obs_shape = env.observation_spec()["info_state"]
input_size = int(np.prod(obs_shape))
num_actions = env.action_spec()["num_actions"]

def parse_combo_size(text):
    if "Single" in text:
        return 1
    if "Pair" in text:
        return 2
    if "Triple" in text:
        return 3
    if "Quad" in text:
        return 4
    return 1


# === 3️⃣ Strategien ===
def decode_actions(state):
    player = state.current_player()
    actions = state.legal_actions()
    decoded = [(a, state.action_to_string(player, a)) for a in actions if a != 0]
    return decoded

def max_combo(state):
    decoded = decode_actions(state)
    return max(decoded, key=lambda x: parse_combo_size(x[1]))[0] if decoded else 0


def single_only(state):
    decoded = decode_actions(state)
    singles = [x for x in decoded if "Single" in x[1]]
    return min(singles, key=lambda x: x[0])[0] if singles else 0

def random_action(state):
    return np.random.choice(state.legal_actions())

def smart_strategy(state):
    decoded = decode_actions(state)
    if not decoded:
        return 0
    # Gruppen nach Kombogröße
    def parse_rank(s): return int(s.split()[-1])
    def parse_size(s):
        if "Single" in s: return 1
        if "Pair" in s: return 2
        if "Triple" in s: return 3
        if "Quad" in s: return 4
        return 1

    groups = {1: [], 2: [], 3: [], 4: []}
    for a, s in decoded:
        groups[parse_size(s)].append((a, s))
    for size in [4, 3, 2, 1]:
        if groups[size]:
            return min(groups[size], key=lambda x: parse_rank(x[1]))[0]
    return 0

strategy_map = {
    "max_combo": max_combo,
    "single_only": single_only,
    "random": random_action,
    "smart": smart_strategy,
}

# === 4️⃣ Agenten laden / Strategien zuweisen ===
agents = []
for pid, ptype in enumerate(PLAYER_TYPES):
    if ptype == "ppo":
        agent = ppo.PPOAgent(input_size, num_actions)
        model_path = os.path.join(MODEL_DIR, f"selfplay_president_03_agent_p{pid}")
        agent.restore(model_path)
        agents.append(agent)
    elif ptype in strategy_map:
        agents.append(strategy_map[ptype])
    else:
        raise ValueError(f"Unbekannter Spielertyp: {ptype}")

# === 5️⃣ Tests ===
win_counts = [0] * 4
total_returns = [0] * 4

for episode in range(1, NUM_GAMES + 1):
    time_step = env.reset()
    state = env.get_state

    while not time_step.last():
        p = time_step.observations["current_player"]
        obs = np.array(time_step.observations["info_state"][p], dtype=np.float32)
        legal = time_step.observations["legal_actions"][p]

        if isinstance(agents[p], ppo.PPOAgent):
            obs_tensor = np.array(obs, dtype=np.float32)
            agent_out = agents[p].step(time_step, legal)
            action = agent_out.action if agent_out else np.random.choice(legal)
        else:
            action = agents[p](env.get_state)

        time_step = env.step([action])

    # Statistik
    rewards = time_step.rewards
    for pid in range(4):
        total_returns[pid] += rewards[pid]
    winner = np.argmax(rewards)
    win_counts[winner] += 1

    if episode % 10 == 0:
        print(f"✅ Spiel {episode} abgeschlossen")

# === 6️⃣ Ergebnisse ===
print("\n=== Test-Ergebnisse ===")
for pid in range(4):
    avg_return = total_returns[pid] / NUM_GAMES
    win_rate = win_counts[pid] / NUM_GAMES * 100
    print(f"Player {pid} ({PLAYER_TYPES[pid]}): Return Ø {avg_return:.2f}, Winrate {win_rate:.1f}%")
