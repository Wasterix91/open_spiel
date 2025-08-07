import datetime
import os
import numpy as np
import re
import torch
import matplotlib.pyplot as plt
import pyspiel
import pandas as pd
from dqn_agent2 import DQNAgent

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(SCRIPT_DIR, "models")

def find_next_version(base_dir):
    pattern = re.compile(r"dqn_model_(\d{2})$")
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
MODEL_BASE = os.path.join(MODELS_ROOT, f"dqn_model_{VERSION}", "train")
os.makedirs(MODEL_BASE, exist_ok=True)
print(f"📁 Neue Trainingsversion: {VERSION}")

# === Parameter ===
NUM_EPISODES = 10_000
EVAL_INTERVAL = 200
EVAL_EPISODES = 10_000
strategy = "random"

# === Random Baseline ===
def random_policy(state):
    legal = state.legal_actions()
    if 0 in legal and len(legal) > 1:
        legal.remove(0)
    return np.random.choice(legal)

# === Trainingsloop für alle 4 Spielerpositionen ===
for player_id in range(4):
    print(f"\n=== 🎮 Training DQN-Agent auf Position {player_id} ===")
    
    game = pyspiel.load_game("president", {
        "num_players": 4,
        "deck_size": "32",
        "shuffle_cards": True,
        "single_card_mode": False,
    })

    state_size = game.observation_tensor_shape()[0]
    num_actions = game.num_distinct_actions()
    agent = DQNAgent(state_size, num_actions)
    MODEL_PATH = os.path.join(MODEL_BASE, f"dqn_model_{VERSION}_agent_p{player_id}")

    player_types = ["random"] * 4
    player_types[player_id] = "dqn"
    player_types_str = ",".join(player_types)

    # === Konfigurationslogging ===
    metadata = {
        "version": VERSION,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent_type": "DQN",
        "num_episodes": NUM_EPISODES,
        "eval_interval": EVAL_INTERVAL,
        "eval_episodes": EVAL_EPISODES,
        "num_players": 4,
        "deck_size": "32",
        "shuffle_cards": True,
        "single_card_mode": False,
        "observation_dim": state_size,
        "num_actions": num_actions,
        "player_types": player_types_str,
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
    if not os.path.exists(csv_file):
        df.to_csv(csv_file, index=False, columns=columns_order)
    else:
        df.to_csv(csv_file, mode='a', header=False, index=False, columns=columns_order)
    print(f"📄 Konfiguration gespeichert für p{player_id}")

    # === Trainingsloop ===
    winrates = []

    for episode in range(1, NUM_EPISODES + 1):
        state = game.new_initial_state()
        while not state.is_terminal():
            player = state.current_player()
            legal = state.legal_actions(player)

            if player == player_id:
                obs = state.observation_tensor(player)
                action = agent.select_action(obs, legal)
                prev_obs = obs.copy()
            else:
                action = random_policy(state)

            state.apply_action(action)

        if player == player_id:
            next_obs = np.zeros_like(obs) if state.is_terminal() else state.observation_tensor(player)
            old_hand_size = sum(prev_obs[:8])
            new_hand_size = sum(next_obs[:8]) if not state.is_terminal() else 0
            reward = old_hand_size - new_hand_size
            if state.is_terminal():
                reward += state.returns()[player_id]
            agent.buffer.add(prev_obs, action, reward, next_obs, state.is_terminal())
            agent.train()

        if episode % 500 == 0:
            print(f"[p{player_id} Ep {episode}] Epsilon: {agent.epsilon:.3f}")

        if episode % EVAL_INTERVAL == 0:
            wins = 0
            for _ in range(EVAL_EPISODES):
                s = game.new_initial_state()
                while not s.is_terminal():
                    pid = s.current_player()
                    legal = s.legal_actions(pid)
                    if pid == player_id:
                        obs = s.observation_tensor(pid)
                        action = agent.select_action(obs, legal)
                    else:
                        action = random_policy(s)
                    s.apply_action(action)

                if s.returns()[player_id] == max(s.returns()):
                    wins += 1

            winrate = 100 * wins / EVAL_EPISODES
            winrates.append(winrate)
            print(f"✅ Eval @ Ep {episode}: p{player_id} Winrate vs {strategy} = {winrate:.1f}%")

    # === Lernkurve speichern ===
    x_vals = list(range(EVAL_INTERVAL, NUM_EPISODES + 1, EVAL_INTERVAL))
    plt.figure()
    plt.plot(x_vals, winrates, marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Winrate (%)")
    plt.title(f"DQN (p{player_id}) vs {strategy} – President")
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(MODEL_BASE, f"lernkurve_p{player_id}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"📈 Lernkurve gespeichert unter: {plot_path}")

    # === Modell speichern ===
    agent.save(MODEL_PATH)
    print(f"💾 Modell gespeichert unter: {MODEL_PATH}")

    # === Q-Werte für Beispielzustände anzeigen ===
example_states = [
    game.new_initial_state(),  # initialer Zustand
    # Optional: Weitere vordefinierte Spielzustände
]

for idx, state in enumerate(example_states):
    obs = state.observation_tensor(0)  # Spieler 0 z. B.
    state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(agent.device)
    q_values = agent.q_net(state_tensor).detach().cpu().numpy().squeeze()
    
    legal_actions = state.legal_actions(0)
    q_legal = {a: q_values[a] for a in legal_actions}

    print(f"\n📊 Q-Werte (legal) für Beispielzustand {idx}:")
    for action, q in sorted(q_legal.items()):
        print(f"  Aktion {action}: Q={q:.3f}")

    # Optional: Visualisierung
    import matplotlib.pyplot as plt
    plt.figure()
    plt.bar(q_legal.keys(), q_legal.values())
    plt.title(f"Q-Werte für Beispielzustand {idx} (legal actions)")
    plt.xlabel("Aktion")
    plt.ylabel("Q-Wert")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_BASE, f"q_values_example_{idx}.png"))
    plt.close()

