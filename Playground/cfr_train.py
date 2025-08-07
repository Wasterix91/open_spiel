import os
import re
import numpy as np
import pandas as pd
import torch
import datetime
import matplotlib.pyplot as plt
import pyspiel
from deep_cfr import DeepCFRSolver

# === Relativer Basispfad zum Speicherort des Skripts ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(SCRIPT_DIR, "models")

# === Automatische Versionserkennung ===
def find_next_version(base_dir):
    pattern = re.compile(r"deepcfr_model_(\d{2})$")
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
MODEL_BASE = os.path.join(MODELS_ROOT, f"deepcfr_model_{VERSION}", "train")
os.makedirs(MODEL_BASE, exist_ok=True)
print(f"📁 Neue Trainingsversion: {VERSION}")

# === Spielparameter & Gegnerkonfiguration ===
player_types = ["deepcfr", "deepcfr", "deepcfr", "deepcfr"]
game_settings = {
    "num_players": 4,
    "deck_size": "32",
    "shuffle_cards": True,
    "single_card_mode": False
}
NUM_ITERATIONS = 5
NUM_TRAVERSALS = 2
EVAL_INTERVAL = 5
EVAL_EPISODES = 7

# === Spiel laden ===
game = pyspiel.load_game("president", game_settings)
num_players = game.num_players()
info_state_size = game.information_state_tensor_size()
num_actions = game.num_distinct_actions()

# === DeepCFR Initialisierung ===
solver = DeepCFRSolver(
    game=game,
    policy_network_layers=[64, 64],
    advantage_network_layers=[64, 64],
    num_iterations=NUM_ITERATIONS,
    num_traversals=NUM_TRAVERSALS,
    learning_rate=1e-2,
    batch_size=32,
    memory_capacity=1000,
    policy_network_train_steps=5,
    advantage_network_train_steps=5,
    device="cpu"
)

# === Gegnerstrategie für Evaluation ===
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

# === Evaluation gegen Random-Gegner ===
def evaluate_vs_random(solver, num_episodes=1000):
    wins = 0
    for _ in range(num_episodes):
        state = solver.game.new_initial_state()
        while not state.is_terminal():
            player = state.current_player()
            legal = state.legal_actions()

            if player == 0:
                obs = np.array(state.information_state_tensor(player), dtype=np.float32)
                logits = solver.policy_net(torch.tensor(obs).to(solver.device)).detach().cpu().numpy()
                masked = logits[legal]
                probs = masked / masked.sum() if masked.sum() > 0 else np.ones(len(legal)) / len(legal)
                action = np.random.choice(legal, p=probs)
            else:
                action = opponent_strategy(state)

            state.apply_action(action)

        if state.returns()[0] == max(state.returns()):
            wins += 1
    return 100 * wins / num_episodes

# === Trainingsloop ===
winrates = []

for it in range(1, NUM_ITERATIONS + 1):
    print(f"\n🔁 Iteration {it}/{NUM_ITERATIONS}")
    for p in range(num_players):
        for _ in range(NUM_TRAVERSALS):
            state = game.new_initial_state()
            solver.traverse(state, p, 1.0, 1.0)

    solver.train_advantage()
    solver.train_policy()

    # Evaluation
    if it % EVAL_INTERVAL == 0 or it == NUM_ITERATIONS:
        winrate = evaluate_vs_random(solver, EVAL_EPISODES)
        winrates.append((it, winrate))
        print(f"✅ Evaluation: Iteration {it}: Winrate vs {EVAL_OPPONENT_STRATEGY} = {winrate:.2f}%")

# === Konfigurations-Logging ===
metadata = {
    "version": VERSION,
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "num_iterations": NUM_ITERATIONS,
    "eval_interval": EVAL_INTERVAL,
    "eval_episodes": EVAL_EPISODES,
    "num_players": game_settings["num_players"],
    "deck_size": game_settings["deck_size"],
    "shuffle_cards": game_settings["shuffle_cards"],
    "single_card_mode": game_settings["single_card_mode"],
    "player_types": ",".join(player_types),
    "observation_dim": info_state_size,
    "num_actions": num_actions,
    "agent_type": "DeepCFR",
    "model_path": os.path.join(MODEL_BASE, f"deepcfr_model_{VERSION}_policy"),
    "model_version_dir": MODEL_BASE
}

csv_file = os.path.join(os.path.dirname(MODEL_BASE), "training_runs.csv")
columns_order = [
    "version", "timestamp", "agent_type", "num_iterations", "eval_interval", "eval_episodes",
    "num_players", "deck_size", "shuffle_cards", "single_card_mode",
    "observation_dim", "num_actions", "player_types", "model_path", "model_version_dir"
]

df = pd.DataFrame([metadata])
df.to_csv(csv_file, mode="a", header=not os.path.exists(csv_file), index=False, columns=columns_order)
print(f"\n📄 Konfiguration gespeichert unter: {csv_file}")

# === Lernkurve plotten ===
eval_iters, winrate_vals = zip(*winrates)
plt.figure(figsize=(10, 6))
plt.plot(eval_iters, winrate_vals, marker='o')
plt.title(f"Lernkurve – Winrate von Player 0 gegen {EVAL_OPPONENT_STRATEGY}")
plt.xlabel("Iteration")
plt.ylabel("Winrate (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_BASE, "lernkurve.png"))
print(f"📈 Lernkurve gespeichert unter: {os.path.join(MODEL_BASE, 'lernkurve.png')}")



