import os
import re
import numpy as np
import pandas as pd
import torch
import datetime
from functools import partial
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
NUM_EPISODES = 800_000
EVAL_INTERVAL = 10_000
EVAL_EPISODES = 10_000

# === Spiel und Environment ===
game = pyspiel.load_game("president", game_settings)
env = rl_environment.Environment(game)
info_state_size = env.observation_spec()["info_state"][0]
num_actions = env.action_spec()["num_actions"]

# === PPO-Agenten für alle Spieler ===
agents = [ppo.PPOAgent(info_state_size, num_actions) for _ in range(4)]

# === Reward Shaping ===
def calculate_step_reward(time_step, player_id, deck_size):
    hand = time_step.observations["info_state"][player_id]
    if deck_size == 32 or deck_size == 64:
        num_ranks = 8
    elif deck_size == 52:
        num_ranks = 13
    else:
        raise NotImplementedError
    hand_size = sum(hand[:num_ranks])
    return -0.01 * hand_size

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

# === Strategien für Evaluation ===
def random_action_strategy(state):
    return np.random.choice(state.legal_actions())

def random2_action_strategy(state):
    legal = state.legal_actions()
    if len(legal) > 1 and 0 in legal:
        legal = [a for a in legal if a != 0]
    return np.random.choice(legal)

def max_combo_strategy(state):
    decoded = [(a, state.action_to_string(state.current_player(), a)) for a in state.legal_actions()]
    if not decoded:
        return 0
    # Wähle Kombination mit größter Combo-Größe: Quad > Triple > Pair > Single
    def combo_size_priority(s):
        if "Quad" in s: return 4
        if "Triple" in s: return 3
        if "Pair" in s: return 2
        if "Single" in s: return 1
        return 0
    best = max(decoded, key=lambda x: (combo_size_priority(x[1]), -x[0]))
    return best[0]

def aggressive_strategy(state):
    decoded = [(a, state.action_to_string(state.current_player(), a)) for a in state.legal_actions()]
    if not decoded:
        return 0
    rank_order = {"7":0,"8":1,"9":2,"10":3,"J":4,"Q":5,"K":6,"A":7}
    def parse_rank(s):
        for r in rank_order:
            if r in s:
                return rank_order[r]
        return -1
    best = max(decoded, key=lambda x: (parse_rank(x[1]), -x[0]))
    return best[0]

def single_only_strategy(state):
    decoded = [(a, state.action_to_string(state.current_player(), a)) for a in state.legal_actions()]
    singles = [x for x in decoded if "Single" in x[1]]
    if singles:
        return singles[0][0]
    else:
        return 0

strategy_map = {
    #"random": random_action_strategy,
    "random2": random2_action_strategy,
    "max_combo": max_combo_strategy,
    #"aggressive": aggressive_strategy,
    "single_only": single_only_strategy
}

opponent_strategies = list(strategy_map.keys())

winrates = {name: [] for name in opponent_strategies}
deck_size = int(game_settings["deck_size"])

# === Trainingsloop mit player_types ===
for episode in range(1, NUM_EPISODES + 1):
    time_step = env.reset()

    while not time_step.last():
        player = time_step.observations["current_player"]
        legal = time_step.observations["legal_actions"][player]

        if player_types[player] == "ppo":
            agent_out = agents[player].step(time_step, legal)
            action = agent_out.action if agent_out else np.random.choice(legal)
        else:
            # Zugriff auf das interne OpenSpiel-State-Objekt für Strategien
            action = strategy_map[player_types[player]](env._state)

        time_step = env.step([action])
        
        # Reward shaping pro gespielter Karte
        if player_types[player] == "ppo":
            agents[player].post_step(calculate_step_reward(time_step, player_id=player, deck_size=deck_size))

    # === Reward Shaping & Training ===
    for i in range(4):
        if player_types[i] == "ppo":
            agents[i]._buffer.rewards[-1] += time_step.rewards[i]
            agents[i]._buffer.rewards[-1] += calculate_final_bonus_reward(time_step.rewards, i)
            agents[i].train()


    # === Evaluation ===
    if episode % EVAL_INTERVAL == 0:
        for opponent_name in opponent_strategies:
            opponent_strategy = strategy_map[opponent_name]
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
            winrates[opponent_name].append(winrate)
            print(f"✅ Evaluation nach {episode} Episoden: Winrate gegen {opponent_name} = {winrate:.1f}%")

        # === Modelle speichern ===
        for i, ag in enumerate(agents):
            model_path_i = os.path.join(MODEL_BASE, f"ppo_model_{VERSION}_agent_p{i}_ep{episode:07d}")
            ag.save(model_path_i)
            print(f"💾 Modell von Agent {i} gespeichert unter: {model_path_i}")

        # === Lernkurven einzeln plotten ===
        eval_intervals = list(range(EVAL_INTERVAL, episode + 1, EVAL_INTERVAL))
        for opponent_name in opponent_strategies:
            plt.figure(figsize=(10, 6))
            plt.plot(eval_intervals, winrates[opponent_name], marker='o')
            plt.title(f"Lernkurve – Winrate von Player 0 gegen {opponent_name}")
            plt.xlabel("Trainings-Episode")
            plt.ylabel("Winrate (%)")
            plt.grid(True)
            plt.tight_layout()
            filename = os.path.join(MODEL_BASE, f"lernkurve_{opponent_name}.png")
            plt.savefig(filename)
            plt.close()
            print(f"📄 Lernkurve für {opponent_name} gespeichert unter: {filename}")

        # === Gemeinsamer Plot aller Lernkurven ===
        plt.figure(figsize=(12, 8))
        for opponent_name in opponent_strategies:
            plt.plot(eval_intervals, winrates[opponent_name], marker='o', label=opponent_name)
        plt.title("Lernkurven – Winrate von Player 0 gegen verschiedene Gegnerstrategien")
        plt.xlabel("Trainings-Episode")
        plt.ylabel("Winrate (%)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        joint_plot_file = os.path.join(MODEL_BASE, "lernkurven_alle_strategien.png")
        plt.savefig(joint_plot_file)
        plt.show()
        print(f"📄 Gemeinsamer Lernkurven-Plot gespeichert unter: {joint_plot_file}")
