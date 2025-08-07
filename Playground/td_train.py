import os
import re
import numpy as np
import pandas as pd
import torch
import datetime
import matplotlib.pyplot as plt
import pyspiel
from open_spiel.python import rl_environment
import td_agent as td

# === Speicherort vorbereiten ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(SCRIPT_DIR, "models")

def find_next_version(base_dir):
    pattern = re.compile(r"td_model_(\d{2})$")
    existing = [
        int(m.group(1))
        for m in (
            pattern.match(name)
            for name in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, name))
        )
        if m
    ]
    return f"{max(existing) + 1:02d}" if existing else "01"

VERSION = find_next_version(MODELS_ROOT)
MODEL_BASE = os.path.join(MODELS_ROOT, f"td_model_{VERSION}", "train")
MODEL_PATH = os.path.join(MODEL_BASE, f"td_model_{VERSION}_agent_p0")
os.makedirs(MODEL_BASE, exist_ok=True)
print(f"📁 Neue Trainingsversion: {VERSION}")

# === Spielkonfiguration ===
player_types = ["td", "td", "td", "td"]
game_settings = {
    "num_players": 4,
    "deck_size": "32",
    "shuffle_cards": True,
    "single_card_mode": False
}
NUM_EPISODES = 10_000
EVAL_INTERVAL = 200
EVAL_EPISODES = 5_000

# === Environment laden ===
game = pyspiel.load_game("president", game_settings)
env = rl_environment.Environment(game)
info_state_size = env.observation_spec()["info_state"][0]
num_actions = env.action_spec()["num_actions"]

# === TD-Agenten initialisieren ===
agents = [td.TDAgent(info_state_size, num_actions) for _ in range(4)]

# === Evaluation Gegnerstrategie ===
def random2_action_strategy(state):
    legal = state.legal_actions()
    if len(legal) > 1 and 0 in legal:
        legal = [a for a in legal if a != 0]
    return np.random.choice(legal)

opponent_strategy = random2_action_strategy
winrates = []

# === Trainingsloop ===
for episode in range(1, NUM_EPISODES + 1):
    time_step = env.reset()
    episode_data = []

    while not time_step.last():
        player = time_step.observations["current_player"]
        legal = time_step.observations["legal_actions"][player]
        agent_out = agents[player].step(time_step, legal)
        action = agent_out.action if agent_out else np.random.choice(legal)
        next_time_step = env.step([action])
        agents[player].observe_transition(time_step, action, next_time_step)
        time_step = next_time_step

    # Terminal update
    for agent in agents:
        agent.end_episode()

    # === Evaluation ===
    if episode % EVAL_INTERVAL == 0:
        wins = 0
        for _ in range(EVAL_EPISODES):
            state = game.new_initial_state()
            while not state.is_terminal():
                pid = state.current_player()
                legal = state.legal_actions(pid)
                if pid == 0:
                    obs = state.information_state_tensor(pid)
                    q_vals = agents[0].predict(torch.tensor(obs, dtype=torch.float32))
                    q_vals = q_vals.detach().numpy()
                    mask = np.full_like(q_vals, -np.inf)
                    mask[legal] = q_vals[legal]
                    action = np.argmax(mask)
                else:
                    action = opponent_strategy(state)
                state.apply_action(action)

            if state.returns()[0] == max(state.returns()):
                wins += 1

        winrate = 100 * wins / EVAL_EPISODES
        winrates.append(winrate)
        print(f"✅ Eval {episode}: Winrate vs Random2 = {winrate:.2f}%")

# === Modelle speichern ===
for i, ag in enumerate(agents):
    model_path_i = os.path.join(MODEL_BASE, f"td_model_{VERSION}_agent_p{i}.pt")
    ag.save(model_path_i)
    print(f"💾 Modell für Agent {i} gespeichert: {model_path_i}")

# === Lernkurve speichern ===
eval_steps = list(range(EVAL_INTERVAL, NUM_EPISODES + 1, EVAL_INTERVAL))
plt.figure(figsize=(10, 6))
plt.plot(eval_steps, winrates, marker='o')
plt.title(f"Lernkurve – Winrate von Player 0 gegen Random2")
plt.xlabel("Trainings-Episode")
plt.ylabel("Winrate (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_BASE, "lernkurve.png"))

# === Metadaten-Logging ===
metadata = {
    "version": VERSION,
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "num_episodes": NUM_EPISODES,
    "eval_interval": EVAL_INTERVAL,
    "eval_episodes": EVAL_EPISODES,
    "num_players": game_settings["num_players"],
    "deck_size": game_settings["deck_size"],
    "shuffle_cards": game_settings["shuffle_cards"],
    "single_card_mode": game_settings["single_card_mode"],
    "player_types": ",".join(player_types),
    "observation_dim": info_state_size,
    "num_actions": num_actions,
    "agent_type": "TD",
    "model_path": MODEL_PATH,
    "model_version_dir": MODEL_BASE
}
csv_file = os.path.join(os.path.dirname(MODEL_BASE), "training_runs.csv")
columns_order = [
    "version", "timestamp", "agent_type", "num_episodes", "eval_interval", "eval_episodes",
    "num_players", "deck_size", "shuffle_cards", "single_card_mode",
    "observation_dim", "num_actions", "player_types", "model_path", "model_version_dir"
]
df = pd.DataFrame([metadata])
df.to_csv(csv_file, mode="a", header=not os.path.exists(csv_file), index=False, columns=columns_order)
print(f"\n📄 Konfiguration gespeichert: {csv_file}")
