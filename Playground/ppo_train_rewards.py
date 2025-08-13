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

"""
ppo_train.py — Train-Skript mit wählbarem Reward-System und Platzierungs-Boni

Stelle das gewünschte Reward-System im Abschnitt "REWARD-KONFIG" unten ein.
Verfügbare Modi:
- "env_only"           : Nur Environment-Reward (kein Shaping)
- "dense_hand"         : Dichte Schritt-Strafe proportional zur Handgröße
- "final_reward_only"  : Finale Platzierungs-Boni/Mali am Episodenende
- "dense+terminal"     : Kombination aus dichter Strafe und terminalem Bonus
- "custom"             : Wie "dense+terminal", aber mit frei einstellbaren Gewichten

Parameter für Platzierungsboni:
- BONUS_WIN / BONUS_2ND / BONUS_3RD / BONUS_LAST
"""

# ===================== REWARD-KONFIG ===================== #
REWARD_SYSTEM = "dense+terminal"   # "env_only" | "dense_hand" | "final_reward_only" | "dense+terminal" | "custom"
HAND_PENALTY_COEFF = -0.1         # z.B. -0.01 pro Karte in der Hand
BONUS_WIN = 10.0                   # 1. Platz
BONUS_2ND = 3.0                    # 2. Platz
BONUS_3RD = 2.0                    # 3. Platz
BONUS_LAST = 0.0                  # 4. Platz (Letzter)
INCLUDE_ENV_REWARD = False          # True = Environment-Reward einbeziehen
# ========================================================= #

# === Pfade ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(SCRIPT_DIR, "models")

# === Automatische Versionserkennung ===
def find_next_version(base_dir):
    pattern = re.compile(r"ppo_model_(\d{2})$")
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    existing = [int(m.group(1)) for m in (pattern.match(name) for name in os.listdir(base_dir)) if m]
    return f"{max(existing) + 1:02d}" if existing else "01"

VERSION = find_next_version(MODELS_ROOT)
MODEL_BASE = os.path.join(MODELS_ROOT, f"ppo_model_{VERSION}", "train")
os.makedirs(MODEL_BASE, exist_ok=True)
print(f"📁 Neue Trainingsversion: {VERSION}")

# === Spielparameter & Gegnerkonfiguration ===
player_types = ["ppo", "max_combo", "max_combo", "max_combo"]

game_settings = {
    "num_players": 4,
    "deck_size": "64",          # "32" | "52" | "64"
    "shuffle_cards": True,
    "single_card_mode": False,
}

NUM_EPISODES = 1_000_000
EVAL_INTERVAL = 10_000
EVAL_EPISODES = 10_000

# === Spiel und Environment ===
game = pyspiel.load_game("president", game_settings)
env = rl_environment.Environment(game)
info_state_size = env.observation_spec()["info_state"][0]
num_actions = env.action_spec()["num_actions"]

# === PPO-Agenten für alle Spieler ===
agents = [ppo.PPOAgent(info_state_size, num_actions) for _ in range(4)]

# === Reward-Shaper ===
class RewardShaper:
    def __init__(self, mode, hand_coeff, bonus_win, bonus_2nd, bonus_3rd, bonus_last, include_env):
        self.mode = mode
        self.hand_coeff = hand_coeff
        self.bonus_win = bonus_win
        self.bonus_2nd = bonus_2nd
        self.bonus_3rd = bonus_3rd
        self.bonus_last = bonus_last
        self.include_env_flag = include_env

    @staticmethod
    def _num_ranks_for_deck(deck_size):
        if deck_size in (32, 64):
            return 8
        if deck_size == 52:
            return 13
        raise NotImplementedError(f"Unsupported deck size: {deck_size}")

    def step_reward(self, time_step, player_id, deck_size):
        """Dichte Strafe basierend auf Handgröße (wenn aktiviert)."""
        if self.mode not in ("dense_hand", "dense+terminal", "custom"):
            return 0.0
        hand = time_step.observations["info_state"][player_id]
        num_ranks = self._num_ranks_for_deck(deck_size)
        hand_size = sum(hand[:num_ranks])
        return self.hand_coeff * hand_size

    def final_bonus(self, final_rewards, player_id):
        """Terminale Platzierungsboni/Mali. Annahme: eindeutige Rangfolge (keine Ties)."""
        if self.mode not in ("final_reward_only", "dense+terminal", "custom"):
            return 0.0
        # Eindeutiges Ranking (höchster Reward = Platz 1)
        order = sorted(range(len(final_rewards)), key=lambda p: final_rewards[p], reverse=True)
        place = order.index(player_id) + 1
        if place == 1:
            return float(self.bonus_win)
        if place == 2:
            return float(self.bonus_2nd)
        if place == 3:
            return float(self.bonus_3rd)
        return float(self.bonus_last)

    def include_env_reward(self):
        return bool(self.include_env_flag)

# Shaper instanziieren
shaper = RewardShaper(
    REWARD_SYSTEM,
    HAND_PENALTY_COEFF,
    BONUS_WIN,
    BONUS_2ND,
    BONUS_3RD,
    BONUS_LAST,
    INCLUDE_ENV_REWARD,
)

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
    "model_version_dir": MODEL_BASE,
    "reward_system": REWARD_SYSTEM,
    "hand_penalty_coeff": HAND_PENALTY_COEFF,
    "bonus_win": BONUS_WIN,
    "bonus_second": BONUS_2ND,
    "bonus_third": BONUS_3RD,
    "bonus_last": BONUS_LAST,
    "include_env_reward": INCLUDE_ENV_REWARD,
}

csv_file = os.path.join(os.path.dirname(MODEL_BASE), "training_runs.csv")
columns_order = [
    "version", "timestamp", "agent_type", "num_episodes", "eval_interval", "eval_episodes",
    "num_players", "deck_size", "shuffle_cards", "single_card_mode",
    "observation_dim", "num_actions", "player_types", "model_path", "model_version_dir",
    "reward_system", "hand_penalty_coeff", "bonus_win", "bonus_second", "bonus_third", "bonus_last", "include_env_reward",
]

pd.DataFrame([metadata]).to_csv(csv_file, index=False, columns=columns_order)
print(f"📄 Konfiguration gespeichert unter: {csv_file}")

# === Gegner-Strategien (für Evaluation) ===
def random2_action_strategy(state):
    legal = state.legal_actions()
    if len(legal) > 1 and 0 in legal:
        legal = [a for a in legal if a != 0]
    return np.random.choice(legal)

def max_combo_strategy(state):
    decoded = [(a, state.action_to_string(state.current_player(), a)) for a in state.legal_actions()]
    if not decoded:
        return 0
    def combo_size_priority(s):
        if "Quad" in s: return 4
        if "Triple" in s: return 3
        if "Pair" in s: return 2
        if "Single" in s: return 1
        return 0
    best = max(decoded, key=lambda x: (combo_size_priority(x[1]), -x[0]))
    return best[0]

def single_only_strategy(state):
    decoded = [(a, state.action_to_string(state.current_player(), a)) for a in state.legal_actions()]
    singles = [x for x in decoded if "Single" in x[1]]
    if singles:
        return singles[0][0]
    else:
        return 0

strategy_map = {
    "random2": random2_action_strategy,
    "max_combo": max_combo_strategy,
    "single_only": single_only_strategy,
}

opponent_strategies = list(strategy_map.keys())
winrates = {name: [] for name in opponent_strategies}
deck_size = int(game_settings["deck_size"])

# === Trainingsloop ===
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

        # Schritt-Shaping
        if player_types[player] == "ppo":
            r_step = shaper.step_reward(time_step, player_id=player, deck_size=deck_size)
            agents[player].post_step(r_step)

    # === Episodenende: terminales Shaping & Training ===
    for i in range(4):
        if player_types[i] != "ppo":
            continue
        # Env-Reward (i. d. R. nur am Ende ≠ 0)
        if shaper.include_env_reward():
            agents[i]._buffer.rewards[-1] += time_step.rewards[i]
        # Terminaler Bonus gemäß eindeutiger Platzierung
        agents[i]._buffer.rewards[-1] += shaper.final_bonus(time_step.rewards, i)
        # Trainieren
        agents[i].train()

    # === Evaluation & Speichern ===
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
