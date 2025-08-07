import os
import re
import numpy as np
import pandas as pd
import torch
import datetime
import matplotlib.pyplot as plt
import pyspiel
from dqn_agent import DQNAgent

# === 📁 Verzeichnisstruktur & Versionierung ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(SCRIPT_DIR, "models")

def find_next_version(base_dir, prefix="dqn_model_"):
    pattern = re.compile(rf"{prefix}(\d{{2}})$")
    existing = [
        int(m.group(1))
        for m in (pattern.match(name) for name in os.listdir(base_dir))
        if m
    ]
    return f"{max(existing)+1:02d}" if existing else "01"

VERSION = find_next_version(MODELS_ROOT)
MODEL_BASE = os.path.join(MODELS_ROOT, f"dqn_model_{VERSION}", "train")
MODEL_PATH = os.path.join(MODEL_BASE, f"dqn_model_{VERSION}_agent_p0.pt")
os.makedirs(MODEL_BASE, exist_ok=True)
print(f"📁 Neue DQN-Trainingsversion: {VERSION}")

# === ⚙️ Konfiguration ===
player_types = ["dqn", "random2", "random2", "random2"]
game_settings = {
    "num_players": 4,
    "deck_size": "32",
    "shuffle_cards": True,
    "single_card_mode": False
}
NUM_EPISODES = 20_000
EVAL_INTERVAL = 200
EVAL_EPISODES = 5000

game = pyspiel.load_game("president", game_settings)
state_size = game.observation_tensor_shape()[0]
num_actions = game.num_distinct_actions()

agent = DQNAgent(state_size, num_actions)

# === Gegnerstrategie (Random2) ===
def random2_strategy(state):
    legal = state.legal_actions()
    if 0 in legal and len(legal) > 1:
        legal = [a for a in legal if a != 0]
    return np.random.choice(legal)

# === Logging der Konfiguration ===
metadata = {
    "version": VERSION,
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "agent_type": "DQN",
    "num_episodes": NUM_EPISODES,
    "eval_interval": EVAL_INTERVAL,
    "eval_episodes": EVAL_EPISODES,
    "num_players": game_settings["num_players"],
    "deck_size": game_settings["deck_size"],
    "shuffle_cards": game_settings["shuffle_cards"],
    "single_card_mode": game_settings["single_card_mode"],
    "observation_dim": state_size,
    "num_actions": num_actions,
    "player_types": ",".join(player_types),
    "model_path": MODEL_PATH,
    "model_version_dir": MODEL_BASE
}

csv_file = os.path.join(os.path.dirname(MODEL_BASE), "training_runs.csv")
columns_order = list(metadata.keys())
df = pd.DataFrame([metadata])
df.to_csv(csv_file, index=False, columns=columns_order)
print(f"📄 Konfiguration gespeichert unter: {csv_file}")

# === Trainingsloop ===
winrates = []

for episode in range(1, NUM_EPISODES + 1):
    state = game.new_initial_state()

    while not state.is_terminal():
        player = state.current_player()
        obs = state.observation_tensor(player)
        legal = state.legal_actions(player)

        if player == 0:
            action = agent.select_action(obs, legal)
            prev_obs = obs.copy()
        else:
            action = random2_strategy(state)

        state.apply_action(action)

        if player == 0:
            reward = state.returns()[player] if state.is_terminal() else 0
            next_obs = np.zeros_like(obs) if state.is_terminal() else state.observation_tensor(player)
            agent.buffer.add(prev_obs, action, reward, next_obs, state.is_terminal())
            agent.train_step()

    if episode % 500 == 0:
        print(f"[Episode {episode}] Epsilon: {agent.epsilon:.3f}")

    # === Evaluation ===
    if episode % EVAL_INTERVAL == 0:
        wins = 0
        for _ in range(EVAL_EPISODES):
            s = game.new_initial_state()
            while not s.is_terminal():
                pid = s.current_player()
                legal = s.legal_actions(pid)

                if pid == 0:
                    obs = s.observation_tensor(pid)
                    action = agent.select_action(obs, legal)
                else:
                    action = random2_strategy(s)

                s.apply_action(action)

            if s.returns()[0] == max(s.returns()):
                wins += 1

        winrate = 100 * wins / EVAL_EPISODES
        winrates.append(winrate)
        print(f"✅ Eval @ Ep {episode}: Winrate vs random2 = {winrate:.1f}%")

# === Modell speichern ===
agent.save(MODEL_PATH)
print(f"✅ Modell gespeichert unter: {MODEL_PATH}")

# === Lernkurve speichern ===
plt.figure(figsize=(10, 6))
eval_x = list(range(EVAL_INTERVAL, NUM_EPISODES + 1, EVAL_INTERVAL))
plt.plot(eval_x, winrates, marker="o")
plt.title("DQN – Lernkurve gegen Random2")
plt.xlabel("Episode")
plt.ylabel("Winrate (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_BASE, "lernkurve.png"))
