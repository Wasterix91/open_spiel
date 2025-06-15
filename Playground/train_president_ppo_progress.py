import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Ruhe!

import pyspiel
import numpy as np
from open_spiel.python import rl_environment
import ppo_local as ppo
from tqdm import trange

# === 1️⃣ Spiel & Env ===
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

RANKS = ["7", "8", "9", "10", "J", "Q", "K", "A"]
RANK_TO_NUM = {rank: i for i, rank in enumerate(RANKS)}
def parse_rank(s): return RANK_TO_NUM[s.split()[-1]]
def parse_combo_size(s):
    if "Single" in s: return 1
    if "Pair" in s: return 2
    if "Triple" in s: return 3
    if "Quad" in s: return 4
    return 1

def safe_choose(player, state):
    assert state.current_player() == player
    actions = state.legal_actions()
    if len(actions) == 1:
        return actions[0]  # nur Pass

    decoded = [(a, state.action_to_string(player, a)) for a in actions if a != 0]

    if not decoded:
        return 0

    if player == 1:
        choice = max(decoded, key=lambda x: parse_combo_size(x[1]))[0]
    elif player == 2:
        choice = min(decoded, key=lambda x: (parse_combo_size(x[1]), parse_rank(x[1])))[0]
    elif player == 3:
        singles = [x for x in decoded if "Single" in x[1]]
        if singles:
            choice = min(singles, key=lambda x: parse_rank(x[1]))[0]
        else:
            choice = min(decoded, key=lambda x: parse_rank(x[1]))[0]
    else:
        raise ValueError(f"Invalid player {player}")

    # Forced Play: falls Heuristik Pass sagt, spiele kleinste Karte!
    if choice == 0:
        choice = min(decoded, key=lambda x: parse_rank(x[1]))[0]

    return choice

# === 2️⃣ Training ===
num_episodes = 5000
returns = []
progress = trange(1, num_episodes + 1, desc="Training", unit="episode")

for ep in progress:
    time_step = env.reset()
    state = env.get_state
    steps = 0

    while not time_step.last():
        p = time_step.observations["current_player"]

        if p == 0:
            agent_out = agent.step(time_step)
            action = agent_out.action
        else:
            action = safe_choose(p, state)

        time_step = env.step([action])
        steps += 1

        if steps > 200:
            progress.write(f"⚠️ Episode {ep} aborted after 200 steps (safety)")
            break

    agent.step(time_step)
    returns.append(sum(time_step.rewards))

    if ep % 100 == 0:
        avg = np.mean(returns[-100:])
        progress.write(f"[Episode {ep}] Ø Return (last 100): {avg:.2f}")

agent.save("ppo_president")
progress.write("\n✅ Modell gespeichert: ppo_president_policy & ppo_president_value")
np.save("training_returns.npy", returns)
