import pyspiel
import numpy as np
from open_spiel.python import rl_environment
import ppo_local as ppo  # deine lokale PPO-Datei

# === Setup ===
game = pyspiel.load_game("president", {
    "deck_size": "32",
    "shuffle_cards": True,
    "single_card_mode": False,
    "start_player_mode": "loser"
})
env = rl_environment.Environment(game)

agent = ppo.PPOAgent(
    env.observation_spec()["info_state"][0],
    env.action_spec()["num_actions"],
    ppo.DEFAULT_CONFIG
)

# === Deine Heuristiken ===
RANKS = ["7", "8", "9", "10", "J", "Q", "K", "A"]
RANK_TO_NUM = {rank: i for i, rank in enumerate(RANKS)}
def parse_rank(s): return RANK_TO_NUM[s.split()[-1]]
def parse_combo_size(s):
    if "Single" in s: return 1
    if "Pair" in s: return 2
    if "Triple" in s: return 3
    if "Quad" in s: return 4
    return 1

def choose_heuristic(player, state):
    actions = state.legal_actions(player)
    decoded = [(a, state.action_to_string(player, a)) for a in actions if a != 0]
    if not decoded: return 0
    if player == 1:
        return max(decoded, key=lambda x: parse_combo_size(x[1]))[0]
    elif player == 2:
        return min(decoded, key=lambda x: (parse_combo_size(x[1]), parse_rank(x[1])))[0]
    elif player == 3:
        singles = [x for x in decoded if "Single" in x[1]]
        if singles:
            return min(singles, key=lambda x: parse_rank(x[1]))[0]
        return 0
    else:
        raise ValueError("Invalid player")

# === Training ===
num_episodes = 5000
returns = []

for ep in range(1, num_episodes + 1):
    time_step = env.reset()
    while not time_step.last():
        p = time_step.observations["current_player"]
        if p == 0:
            agent_out = agent.step(time_step)
            action = agent_out.action
        else:
            action = choose_heuristic(p, env.get_state)
        time_step = env.step([action])
    agent.step(time_step)

    returns.append(sum(time_step.rewards))
    if ep % 100 == 0:
        avg = np.mean(returns[-100:])
        print(f"[Episode {ep}] Ø Return (last 100): {avg:.2f}")

# === Speichern ===
agent.save("ppo_president")
print("\n✅ Modell gespeichert unter: ppo_president_policy & ppo_president_value")
