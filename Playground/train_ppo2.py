import os
import re
import numpy as np
import pandas as pd
import torch
import datetime
import matplotlib.pyplot as plt
import pyspiel
from open_spiel.python import rl_environment
import ppo_agent as ppo

# === Relativer Basispfad zum Speicherort des Skripts ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(SCRIPT_DIR, "models")

# === Automatische Versionserkennung ===
def find_next_version(base_dir):
    pattern = re.compile(r"ppo_model_(\d{2})$")
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
MODEL_BASE = os.path.join(MODELS_ROOT, f"ppo_model_{VERSION}", "train")
os.makedirs(MODEL_BASE, exist_ok=True)
print(f"📁 Neue Trainingsversion: {VERSION}")

# === Spielparameter & Gegnerkonfiguration ===
player_types = ["ppo", "ppo", "ppo", "ppo"]
game_settings = {
    "num_players": 4,
    "deck_size": "64",
    "shuffle_cards": True,
    "single_card_mode": False
}
NUM_EPISODES = 300_000
EVAL_INTERVAL = 200
EVAL_EPISODES = 10_000

# === Spiel und Environment ===
game = pyspiel.load_game("president", game_settings)
env = rl_environment.Environment(game)
info_state_size = env.observation_spec()["info_state"][0]
num_actions = env.action_spec()["num_actions"]

# === PPO-Agenten für alle Spieler ===
agents = [ppo.PPOAgent(info_state_size, num_actions) for _ in range(4)]

# === Reward Shaping ===
def calculate_step_reward(player_id, time_step, deck_size):
    hand = time_step.observations["info_state"][player_id]
    num_ranks = 8 if deck_size == 32 else 16
    hand_size = sum(hand[:num_ranks])
    return -hand_size

def calculate_final_bonus_reward(returns, player_id):
    if returns[player_id] == max(returns):
        return 10
    elif returns[player_id] == min(returns):
        return -5
    return 0

# === Konfigurations-Logging ===
MODEL_PATH = os.path.join(MODEL_BASE, f"ppo_model_{VERSION}_agent_p0")
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
csv_file = os.path.join(os.path.dirname(MODEL_BASE), "training_runs.csv")
columns_order = [
    "version", "timestamp", "agent_type", "num_episodes", "eval_interval", "eval_episodes",
    "num_players", "deck_size", "shuffle_cards", "single_card_mode",
    "observation_dim", "num_actions", "player_types", "model_path", "model_version_dir"
]
df = pd.DataFrame([metadata])
df.to_csv(csv_file, index=False, columns=columns_order)
print(f"📄 Konfiguration gespeichert unter: {csv_file}")

# === Konfigurierbare Gegnerstrategie für Evaluation ===
EVAL_OPPONENT_STRATEGY = "random"  # "random" oder "random2"

def random_action_strategy(state):
    return np.random.choice(state.legal_actions())

def random2_action_strategy(state):
    legal = state.legal_actions()
    if len(legal) > 1 and 0 in legal:
        legal = [a for a in legal if a != 0]
    return np.random.choice(legal)

strategy_map = {
    "random": random_action_strategy,
    "random2": random2_action_strategy
}
opponent_strategy = strategy_map[EVAL_OPPONENT_STRATEGY]

# === Trainingsloop ===
winrates = []
deck_size = int(game_settings["deck_size"])

for episode in range(1, NUM_EPISODES + 1):
    time_step = env.reset()

    while not time_step.last():
        player = time_step.observations["current_player"]
        legal = time_step.observations["legal_actions"][player]

        agent_out = agents[player].step(time_step, legal)
        action = agent_out.action if agent_out else np.random.choice(legal)
        time_step = env.step([action])

    # Reward Shaping & Training
    for i in range(4):
        shaped_reward = calculate_step_reward(i, time_step, deck_size)
        shaped_reward += calculate_final_bonus_reward(time_step.rewards, i)

        agents[i]._buffer.rewards[-1] = shaped_reward
        agents[i].step(time_step, [0])
        agents[i].train()

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
                    logits = agents[0]._policy(torch.tensor(obs, dtype=torch.float32)).detach().numpy()
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
                    action = opponent_strategy(state)

                state.apply_action(action)

            if state.returns()[0] == max(state.returns()):
                wins += 1

        winrate = 100 * wins / EVAL_EPISODES
        winrates.append(winrate)
        print(f"✅ Evaluation nach {episode} Episoden: Winrate gegen {EVAL_OPPONENT_STRATEGY} = {winrate:.1f}%")

# === Finales Modell speichern ===
for i, ag in enumerate(agents):
    model_path_i = os.path.join(MODEL_BASE, f"ppo_model_{VERSION}_agent_p{i}")
    ag.save(model_path_i)
    print(f"💾 Modell von Agent {i} gespeichert unter: {model_path_i}")

# === Lernkurve plotten ===
eval_intervals = list(range(EVAL_INTERVAL, NUM_EPISODES + 1, EVAL_INTERVAL))
plt.figure(figsize=(10, 6))
plt.plot(eval_intervals, winrates, marker='o')
plt.title(f"Lernkurve – Winrate von Player 0 gegen {EVAL_OPPONENT_STRATEGY}")
plt.xlabel("Trainings-Episode")
plt.ylabel("Winrate (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_BASE, "lernkurve.png"))
