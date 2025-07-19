import os
import re
import numpy as np
import pandas as pd
import torch
import datetime
import matplotlib.pyplot as plt
import pyspiel
from open_spiel.python import rl_environment
import ppo_local_2 as ppo

# === Relativer Basispfad zum Speicherort des Skripts ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(SCRIPT_DIR, "models")

# === Automatische Versionserkennung ===
def find_next_version(base_dir):
    pattern = re.compile(r"selfplay_president_(\d{2})$")
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
MODEL_BASE = os.path.join(MODELS_ROOT, f"selfplay_president_{VERSION}", "train")
MODEL_PATH = os.path.join(MODEL_BASE, f"selfplay_president_{VERSION}_agent_p0")
os.makedirs(MODEL_BASE, exist_ok=True)
print(f"📁 Neue Trainingsversion: {VERSION}")

# === Spielparameter & Gegnerkonfiguration ===
player_types = ["ppo", "random", "random", "random"]
game_settings = {
    "num_players": 4,
    "deck_size": "32",
    "shuffle_cards": True,
    "single_card_mode": False
}
NUM_EPISODES = 2_500
EVAL_INTERVAL = 50
EVAL_EPISODES = 10_000

# === Spiel und Environment ===
game = pyspiel.load_game("president", game_settings)
env = rl_environment.Environment(game)
info_state_size = env.observation_spec()["info_state"][0]
num_actions = env.action_spec()["num_actions"]

# === PPO-Agent für Player 0 ===
agent = ppo.PPOAgent(info_state_size, num_actions)

# === Konfigurations-Logging ===
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
    "agent_type": "PPO",
    "model_path": MODEL_PATH,
    "model_version_dir": MODEL_BASE
}

# === Speichere nur aktuelle Konfiguration in selfplay_president_XX/training_runs.csv ===
csv_file = os.path.join(os.path.dirname(MODEL_BASE), "training_runs.csv")
columns_order = [
    "version", "timestamp", "agent_type", "num_episodes", "eval_interval", "eval_episodes",
    "num_players", "deck_size", "shuffle_cards", "single_card_mode",
    "observation_dim", "num_actions", "player_types", "model_path", "model_version_dir"
]

df = pd.DataFrame([metadata])
df.to_csv(csv_file, index=False, columns=columns_order)
print(f"📄 Konfiguration gespeichert unter: {csv_file}")

# === Trainingsloop ===
winrates = []  # Zum Plotten der Lernkurve

for episode in range(1, NUM_EPISODES + 1):
    time_step = env.reset()
    total_reward = 0

    while not time_step.last():
        player = time_step.observations["current_player"]
        legal = time_step.observations["legal_actions"][player]

        if player == 0:
            agent_out = agent.step(time_step, legal)
            action = agent_out.action if agent_out else np.random.choice(legal)
        else:
            action = np.random.choice(legal)

        time_step = env.step([action])

        if player == 0:
            hand = time_step.observations["info_state"][0]
            hand_size = sum(hand[:8])  # 8 Ränge bei 32er Deck
            reward = -hand_size
            agent._buffer.rewards[-1] = reward
            total_reward += reward

    final_scores = time_step.rewards
    player_score = final_scores[0]
    if player_score == max(final_scores):
        total_reward += 10
    elif player_score == min(final_scores):
        total_reward -= 5

    agent._buffer.rewards[-1] = total_reward
    agent.step(time_step, [0])
    agent.train()

    """     if episode % 100 == 0:
        print(f"[{episode}] Training abgeschlossen.") """

    if episode % EVAL_INTERVAL == 0:
        wins = 0
        for _ in range(EVAL_EPISODES):
            state = game.new_initial_state()
            while not state.is_terminal():
                pid = state.current_player()
                legal = state.legal_actions(pid)
                if pid == 0:
                    obs = state.information_state_tensor(0)
                    logits = agent._policy(torch.tensor(obs, dtype=torch.float32)).detach().numpy()
                    masked = np.zeros_like(logits)
                    masked[legal] = logits[legal]
                    masked = np.nan_to_num(masked, nan=0.0)
                    if masked.sum() <= 0 or np.any(np.isnan(masked)):
                        probs = np.zeros_like(logits)
                        probs[legal] = 1.0 / len(legal)
                    else:
                        probs = masked / masked.sum()
                    action = np.random.choice(len(probs), p=probs)
                else:
                    action = np.random.choice(legal)
                state.apply_action(action)

            if state.returns()[0] == max(state.returns()):
                wins += 1

        winrate = 100 * wins / EVAL_EPISODES
        winrates.append(winrate)  # Für Plot
        print(f"✅ Evaluation nach {episode} Episoden: Winrate gegen Random = {winrate:.1f}%")
        agent.save(MODEL_PATH)

# === Finales Modell speichern ===
agent.save(MODEL_PATH)
print(f"✅ Finales Modell gespeichert unter: {MODEL_PATH}")

# === Lernkurve plotten ===
eval_intervals = list(range(EVAL_INTERVAL, NUM_EPISODES + 1, EVAL_INTERVAL))
plt.figure(figsize=(10, 6))
plt.plot(eval_intervals, winrates, marker='o')
plt.title("Lernkurve – Winrate gegen Random-Gegner")
plt.xlabel("Trainings-Episode")
plt.ylabel("Winrate (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_BASE, "lernkurve.png"))
plt.show()
