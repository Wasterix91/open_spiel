# train_dqn_president.py
import pyspiel
import numpy as np
import os
from dqn_agent import DQNAgent
import datetime

# === Spiel laden ===
game = pyspiel.load_game("president", {
    "deck_size": "32",
    "shuffle_cards": True,
    "single_card_mode": False,
    "num_players": 4
})

state_size = game.observation_tensor_shape()[0]
num_actions = game.num_distinct_actions()

# === Agent initialisieren ===
agent = DQNAgent(state_size, num_actions)
save_dir = f"models/dqn_model_v01"
os.makedirs(save_dir, exist_ok=True)

# === Trainingsparameter ===
NUM_EPISODES = 100_000

for episode in range(1, NUM_EPISODES + 1):
    state = game.new_initial_state()
    while not state.is_terminal():
        player = state.current_player()
        obs = state.observation_tensor(player)
        legal_actions = state.legal_actions(player)

        action = agent.select_action(obs, legal_actions)
        prev_obs = obs.copy()

        state.apply_action(action)

        reward = state.rewards()[player] if state.is_terminal() else 0
        done = state.is_terminal()
        next_obs = state.observation_tensor(player) if not done else np.zeros_like(obs)

        agent.buffer.add(prev_obs, action, reward, next_obs, done)
        agent.train_step()

    if episode % 500 == 0:
        print(f"[Episode {episode}] Epsilon: {agent.epsilon:.3f}")

    if episode % 5000 == 0:
        model_path = os.path.join(save_dir, f"dqn_model_v01_agent_p0_ep{episode}.pt")
        agent.save(model_path)
        print(f"💾 Modell gespeichert unter: {model_path}")
