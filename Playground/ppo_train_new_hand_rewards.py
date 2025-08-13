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
ppo_train.py — Train-Skript mit modularen Rewards

Konfiguriere separat:
- STEP_REWARD: "none" | "delta_hand" | "hand_penalty"
- FINAL_REWARD: "none" | "placement_bonus" (Alias: "final_reward")
- ENV_REWARD: True/False (Environment-Reward am Episodenende addieren)

Spezial:
- delta_hand: + (#Karten_vor - #Karten_nach), negatives Delta wird verworfen (geclippt auf 0).
- DELTA_WEIGHT skaliert den delta_hand-Reward (z. B. 1.0 == +1 pro abgelegter Karte).

"""

# ===================== REWARD-KONFIG ===================== #
# Schritt-Reward pro Zug:
#   - "none"         : kein Schritt-Reward
#   - "delta_hand"   : +(#Karten_vor - #Karten_nach), geclippt auf >= 0
#   - "hand_penalty" : dichte Strafe proportional zur aktuellen Handgröße
STEP_REWARD = "delta_hand"
DELTA_WEIGHT = 1.0          # Skaliert delta_hand (z. B. 0.5, 1.0, 2.0, ...)
HAND_PENALTY_COEFF = 0.0  # Für "hand_penalty"

# Terminaler Zusatz-Reward am Episodenende:
#   - "none"             : kein terminaler Bonus
#   - "placement_bonus"  : Boni/Mali gemäß Platzierung (Alias "final_reward" möglich)
FINAL_REWARD = "none"  # oder "final_reward"

# Platzierungs-Boni für FINAL_REWARD == "placement_bonus"/"final_reward"
BONUS_WIN  = 0.0
BONUS_2ND  = 0.0
BONUS_3RD  = 0.0
BONUS_LAST = 0.0

# Environment-Reward berücksichtigen?
#   True  -> addiere env.time_step.rewards[i] am Ende zur letzten Transition
#   False -> ignoriere Environment-Reward komplett
ENV_REWARD = True


# ==================================================================== #

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
player_types = ["ppo", "ppo", "ppo", "ppo"]

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

# === RewardShaper (modular) ===
class RewardShaper:
    def __init__(self, step_mode, final_mode, env_flag,
                 delta_weight, hand_coeff,
                 bonus_win, bonus_2nd, bonus_3rd, bonus_last):
        # Normalisiere Aliase
        final_mode = "placement_bonus" if final_mode in ("final_reward", "placement_bonus") else final_mode
        self.step_mode = step_mode or "none"
        self.final_mode = final_mode or "none"
        self.env_flag = bool(env_flag)

        self.delta_weight = float(delta_weight)
        self.hand_coeff = float(hand_coeff)

        self.bonus_win = float(bonus_win)
        self.bonus_2nd = float(bonus_2nd)
        self.bonus_3rd = float(bonus_3rd)
        self.bonus_last = float(bonus_last)

    @staticmethod
    def _num_ranks_for_deck(deck_size):
        if deck_size in (32, 64):
            return 8
        if deck_size == 52:
            return 13
        raise NotImplementedError(f"Unsupported deck size: {deck_size}")

    def hand_size(self, time_step, player_id, deck_size: int) -> int:
        num_ranks = self._num_ranks_for_deck(deck_size)
        hand = time_step.observations["info_state"][player_id]
        return int(sum(hand[:num_ranks]))

    # ---- Schritt-Reward (in der Schleife aufgerufen) ----
    def step_reward(self, *, mode, hand_before=None, hand_after=None, time_step=None,
                    player_id=None, deck_size=None) -> float:
        mode = mode or self.step_mode
        if mode == "none":
            return 0.0
        if mode == "delta_hand":
            if hand_before is None or hand_after is None:
                raise ValueError("delta_hand benötigt hand_before und hand_after")
            diff = float(hand_before - hand_after)
            # Negatives Delta unzulässig: clip auf 0
            diff = max(0.0, diff)
            return self.delta_weight * diff
        if mode == "hand_penalty":
            if time_step is None or player_id is None or deck_size is None:
                raise ValueError("hand_penalty benötigt time_step, player_id, deck_size")
            size = self.hand_size(time_step, player_id, deck_size)
            return self.hand_coeff * float(size)
        raise ValueError(f"Unbekannter STEP_REWARD: {mode}")

    # ---- Terminaler Bonus auf Basis finaler Platzierung ----
    def final_bonus(self, *, mode, final_rewards, player_id) -> float:
        mode = ("placement_bonus" if mode in ("final_reward", "placement_bonus") else mode) or self.final_mode
        if mode == "none":
            return 0.0
        if mode == "placement_bonus":
            order = sorted(range(len(final_rewards)), key=lambda p: final_rewards[p], reverse=True)
            place = order.index(player_id) + 1
            if place == 1: return self.bonus_win
            if place == 2: return self.bonus_2nd
            if place == 3: return self.bonus_3rd
            return self.bonus_last
        raise ValueError(f"Unbekannter FINAL_REWARD: {mode}")

    def include_env_reward(self) -> bool:
        return self.env_flag

# Shaper instanziieren
shaper = RewardShaper(
    step_mode=STEP_REWARD,
    final_mode=FINAL_REWARD,
    env_flag=ENV_REWARD,
    delta_weight=DELTA_WEIGHT,
    hand_coeff=HAND_PENALTY_COEFF,
    bonus_win=BONUS_WIN,
    bonus_2nd=BONUS_2ND,
    bonus_3rd=BONUS_3RD,
    bonus_last=BONUS_LAST,
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

    "step_reward": STEP_REWARD,
    "delta_weight": DELTA_WEIGHT,
    "final_reward": FINAL_REWARD,
    "env_reward": ENV_REWARD,
    "hand_penalty_coeff": HAND_PENALTY_COEFF,
    "bonus_win": BONUS_WIN,
    "bonus_second": BONUS_2ND,
    "bonus_third": BONUS_3RD,
    "bonus_last": BONUS_LAST,
}

csv_file = os.path.join(os.path.dirname(MODEL_BASE), "training_runs.csv")
columns_order = [
    "version", "timestamp", "agent_type", "num_episodes", "eval_interval", "eval_episodes",
    "num_players", "deck_size", "shuffle_cards", "single_card_mode",
    "observation_dim", "num_actions", "player_types", "model_path", "model_version_dir",
    "step_reward", "delta_weight", "final_reward", "env_reward",
    "hand_penalty_coeff", "bonus_win", "bonus_second", "bonus_third", "bonus_last",
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

macro_winrates = []  # speichert den Macro Average je Eval-Zeitpunkt

# === Trainingsloop ===
for episode in range(1, NUM_EPISODES + 1):
    time_step = env.reset()

    while not time_step.last():
        player = time_step.observations["current_player"]
        legal = time_step.observations["legal_actions"][player]

        # Handgröße VOR dem Zug merken (für delta_hand)
        hand_before = shaper.hand_size(time_step, player_id=player, deck_size=deck_size)

        if player_types[player] == "ppo":
            agent_out = agents[player].step(time_step, legal)
            action = agent_out.action if agent_out else np.random.choice(legal)
        else:
            # Zugriff auf das interne OpenSpiel-State-Objekt für Strategien
            action = strategy_map[player_types[player]](env._state)

        # Schritt ausführen -> neuer time_step (Zustand NACH dem Zug)
        time_step = env.step([action])

        # Schritt-Shaping
        if player_types[player] == "ppo":
            hand_after = shaper.hand_size(time_step, player_id=player, deck_size=deck_size)
            r_step = shaper.step_reward(
                mode=STEP_REWARD,
                hand_before=hand_before,
                hand_after=hand_after,
                time_step=time_step,
                player_id=player,
                deck_size=deck_size
            )
            agents[player].post_step(r_step)

    # === Episodenende: terminales Shaping & Training ===
    for i in range(4):
        if player_types[i] != "ppo":
            continue
        # Optional: Env-Reward addieren (typisch nur am Ende != 0)
        if shaper.include_env_reward():
            agents[i]._buffer.rewards[-1] += time_step.rewards[i]
        # Terminaler Bonus gemäß eindeutiger Platzierung
        agents[i]._buffer.rewards[-1] += shaper.final_bonus(
            mode=FINAL_REWARD,
            final_rewards=time_step.rewards,
            player_id=i
        )
        # Trainieren
        agents[i].train()

    if episode % EVAL_INTERVAL == 0:
        per_opponent_winrates = []  # sammelt die drei Winrates für den Macro-Average

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

            winrate = 100.0 * wins / EVAL_EPISODES
            winrates[opponent_name].append(winrate)
            per_opponent_winrates.append(winrate)
            print(f"✅ Evaluation nach {episode} Episoden: Winrate gegen {opponent_name} = {winrate:.1f}%")

        # ---- Macro Average berechnen & speichern ----
        macro_avg = float(np.mean(per_opponent_winrates)) if per_opponent_winrates else 0.0
        macro_winrates.append(macro_avg)
        print(f"📊 Macro Average (gleich gewichtet): {macro_avg:.2f}%")

        # ---- Modelle speichern ----
        for i, ag in enumerate(agents):
            model_path_i = os.path.join(MODEL_BASE, f"ppo_model_{VERSION}_agent_p{i}_ep{episode:07d}")
            ag.save(model_path_i)
            print(f"💾 Modell von Agent {i} gespeichert unter: {model_path_i}")

        # ---- Lernkurven einzeln (pro Gegner) ----
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

        # ---- Gemeinsamer Plot (nur die drei Strategien) ----
        plt.figure(figsize=(12, 8))
        for opponent_name in opponent_strategies:
            plt.plot(eval_intervals, winrates[opponent_name], marker='o', label=opponent_name)
        plt.title("Lernkurve – Winrate von Player 0 gegen verschiedene Gegnerstrategien")
        plt.xlabel("Trainings-Episode")
        plt.ylabel("Winrate (%)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        joint_plot_file = os.path.join(MODEL_BASE, "lernkurve_alle_strategien.png")
        plt.savefig(joint_plot_file)
        plt.close()
        print(f"📄 Gemeinsamer Lernkurven-Plot gespeichert unter: {joint_plot_file}")

        # ---- Gemeinsamer Plot + Macro Average (Overlay) ----
        plt.figure(figsize=(12, 8))
        for opponent_name in opponent_strategies:
            plt.plot(eval_intervals, winrates[opponent_name], marker='o', label=opponent_name)
        # Macro Average hinzufügen
        plt.plot(eval_intervals, macro_winrates, marker='o', linestyle='--', label='macro_avg')
        plt.title("Lernkurve – Winrate (mit Macro Average)")
        plt.xlabel("Trainings-Episode")
        plt.ylabel("Winrate (%)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        joint_avg_plot_file = os.path.join(MODEL_BASE, "lernkurve_alle_strategien_avg.png")
        plt.savefig(joint_avg_plot_file)
        plt.close()
        print(f"📄 Lernkurven-Plot mit Macro Average gespeichert unter: {joint_avg_plot_file}")

        # ---- Nur Macro Average (alleine) ----
        plt.figure(figsize=(10, 6))
        plt.plot(eval_intervals, macro_winrates, marker='o')
        plt.title("Lernkurve – Macro Average Winrate")
        plt.xlabel("Trainings-Episode")
        plt.ylabel("Winrate (%)")
        plt.grid(True)
        plt.tight_layout()
        macro_only_file = os.path.join(MODEL_BASE, "lernkurve_avg_macro.png")
        plt.savefig(macro_only_file)
        plt.close()
        print(f"📄 Macro-Average-Plot gespeichert unter: {macro_only_file}")
