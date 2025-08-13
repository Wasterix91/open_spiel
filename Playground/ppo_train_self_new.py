import os
import re
import copy
import numpy as np
import pandas as pd
import torch
import datetime
import matplotlib.pyplot as plt
import pyspiel
from open_spiel.python import rl_environment
import ppo_agent_self_new as ppo

"""
Self-Play-fähiges Training für OpenSpiel "president" mit PPO:
- Parameter-Sharing optional (Seat-ID-Feature)
- Opponent-Pool (eingefrorene Snapshots) + Mix aus aktueller Policy
- Echte PPO-Optimierung (Clipping, GAE) via ppo_agent.py
- Reward-Shaping (delta_hand, placement_bonus, env_reward)
"""

# ===================== REWARD-KONFIG ===================== #
STEP_REWARD = "none"      # "none" | "delta_hand" | "hand_penalty"
DELTA_WEIGHT = 0.0
HAND_PENALTY_COEFF = 0.0
FINAL_REWARD = "none"  # alias: "final_reward"
BONUS_WIN  = 0.0
BONUS_2ND  = 0.0
BONUS_3RD  = 0.0
BONUS_LAST = 0.0
ENV_REWARD = True

# ===================== SELF-PLAY-KONFIG ===================== #
PARAMETER_SHARING = True        # True = ein gemeinsames Netz (Seat-ID optional)
ADD_SEAT_ID = True              # fügt One-Hot Seat-ID zur Observation hinzu (empfohlen bei Sharing)
LEARNER_SEAT = 0                # welcher Seat sammelt Daten & lernt

SELFPLAY_MIX_CURRENT = 0.8      # Anteil Episoden/Seats gegen aktuelle Policy
SNAPSHOT_INTERVAL = 10_000      # alle X Episoden einen Snapshot in den Pool legen
OPP_POOL_CAP = 20               # maximale Poolgröße

# ===================== TRAIN/EVAL ===================== #
NUM_EPISODES = 500_000
EVAL_INTERVAL = 10_000
EVAL_EPISODES = 10_000

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
player_types = ["ppo", "opp", "opp", "opp"]  # Seat 0 = Lerner; 1–3 = Gegner aus Pool/aktuell

game_settings = {
    "num_players": 4,
    "deck_size": "64",          # "32" | "52" | "64"
    "shuffle_cards": True,
    "single_card_mode": False,
}

game = pyspiel.load_game("president", game_settings)
env = rl_environment.Environment(game)
info_state_size = env.observation_spec()["info_state"][0]
num_actions = env.action_spec()["num_actions"]
num_players = game_settings["num_players"]
seat_id_dim = num_players if ADD_SEAT_ID else 0

# === PPO-Agent(en) ===
if PARAMETER_SHARING:
    shared = ppo.PPOAgent(info_state_size, num_actions, seat_id_dim=seat_id_dim)
    agents = [shared]  # nur Referenz auf den Lerner; Gegner werden aus shared gelesen
else:
    agents = [ppo.PPOAgent(info_state_size, num_actions, seat_id_dim=seat_id_dim) for _ in range(num_players)]

# === RewardShaper (modular) ===
class RewardShaper:
    def __init__(self, step_mode, final_mode, env_flag,
                 delta_weight, hand_coeff,
                 bonus_win, bonus_2nd, bonus_3rd, bonus_last):
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

    def step_reward(self, *, mode, hand_before=None, hand_after=None, time_step=None,
                    player_id=None, deck_size=None) -> float:
        mode = mode or self.step_mode
        if mode == "none":
            return 0.0
        if mode == "delta_hand":
            if hand_before is None or hand_after is None:
                raise ValueError("delta_hand benötigt hand_before und hand_after")
            diff = float(hand_before - hand_after)
            diff = max(0.0, diff)
            return self.delta_weight * diff
        if mode == "hand_penalty":
            if time_step is None or player_id is None or deck_size is None:
                raise ValueError("hand_penalty benötigt time_step, player_id, deck_size")
            size = self.hand_size(time_step, player_id, deck_size)
            return self.hand_coeff * float(size)
        raise ValueError(f"Unbekannter STEP_REWARD: {mode}")

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

deck_size = int(game_settings["deck_size"])

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
    "observation_dim": info_state_size + seat_id_dim,
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
    "parameter_sharing": PARAMETER_SHARING,
    "add_seat_id": ADD_SEAT_ID,
}

csv_file = os.path.join(os.path.dirname(MODEL_BASE), "training_runs.csv")
columns_order = [
    "version", "timestamp", "agent_type", "num_episodes", "eval_interval", "eval_episodes",
    "num_players", "deck_size", "shuffle_cards", "single_card_mode",
    "observation_dim", "num_actions", "player_types", "model_path", "model_version_dir",
    "step_reward", "delta_weight", "final_reward", "env_reward",
    "hand_penalty_coeff", "bonus_win", "bonus_second", "bonus_third", "bonus_last",
    "parameter_sharing", "add_seat_id",
]

pd.DataFrame([metadata]).to_csv(csv_file, index=False, columns=columns_order)
print(f"📄 Konfiguration gespeichert unter: {csv_file}")

# === Gegner-Strategien (für optionale Evaluation) ===
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
macro_winrates = []  # Macro Average je Eval-Zeitpunkt

# === Opponent-Pool (Snapshots) ===
opponent_pool = []  # enthält state_dicts der Policy


def make_snapshot(policy_net):
    return copy.deepcopy(policy_net.state_dict())


class SnapshotPolicy:
    """Wrapper, der eine geladene Policy (nur Inferenz) hält."""

    def __init__(self, input_dim, num_actions, state_dict):
        self.net = ppo.PolicyNetwork(input_dim, num_actions)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self.num_actions = num_actions

    @torch.no_grad()
    def act(self, obs, legal, seat_one_hot=None):
        x = np.array(obs, dtype=np.float32)
        if seat_id_dim > 0:
            assert seat_one_hot is not None and len(seat_one_hot) == seat_id_dim
            x = np.concatenate([x, np.array(seat_one_hot, dtype=np.float32)], axis=0)
        device = next(self.net.parameters()).device
        x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        logits = self.net(x_t).squeeze(0)
        legal_mask = torch.zeros(self.num_actions, dtype=torch.float32, device=device)
        legal_mask[legal] = 1.0
        probs = ppo.masked_softmax(logits, legal_mask)
        dist = torch.distributions.Categorical(probs=probs)
        return int(dist.sample().item())


# === Trainingsloop ===
for episode in range(1, NUM_EPISODES + 1):
    time_step = env.reset()

    # --- Für Seats 1–3 festlegen: aktuelle Policy oder Snapshot? ---
    use_current = []
    tmp_snapshot_policies = {}
    for seat in [1, 2, 3]:
        if len(opponent_pool) == 0:
            use_current.append(True)
        else:
            use_current.append(np.random.rand() < SELFPLAY_MIX_CURRENT)
            if not use_current[-1]:
                idx = np.random.randint(len(opponent_pool))
                tmp_snapshot_policies[seat] = SnapshotPolicy(
                    input_dim=info_state_size + seat_id_dim,
                    num_actions=num_actions,
                    state_dict=opponent_pool[idx],
                )

    # --- Episoden-Rollout ---
    while not time_step.last():
        player = time_step.observations["current_player"]
        legal = time_step.observations["legal_actions"][player]
        info_state = time_step.observations["info_state"][player]

        # Seat-One-Hot (optional)
        seat_one_hot = None
        if ADD_SEAT_ID:
            seat_one_hot = np.zeros(num_players, dtype=np.float32)
            seat_one_hot[player] = 1.0

        # Handgröße vor dem Zug (für delta_hand)
        hand_before = shaper.hand_size(time_step, player_id=player, deck_size=deck_size)

        if player == LEARNER_SEAT:
            # Lernender Agent sammelt Daten
            if PARAMETER_SHARING:
                action = agents[0].step(info_state, legal, seat_one_hot=seat_one_hot)
            else:
                action = agents[player].step(info_state, legal, seat_one_hot=seat_one_hot)
        else:
            # Gegner handeln: aktuelle Policy (ohne Buffer) oder Snapshot
            if use_current[player - 1]:
                with torch.no_grad():
                    x = np.array(info_state, dtype=np.float32)
                    if ADD_SEAT_ID:
                        x = np.concatenate([x, seat_one_hot], axis=0)
                    device = next(agents[0]._policy.parameters()).device
                    x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
                    logits = agents[0]._policy(x_t).squeeze(0)
                    legal_mask = torch.zeros(num_actions, dtype=torch.float32, device=device)
                    legal_mask[legal] = 1.0
                    probs = ppo.masked_softmax(logits, legal_mask)
                    dist = torch.distributions.Categorical(probs=probs)
                    action = int(dist.sample().item())
            else:
                action = tmp_snapshot_policies[player].act(info_state, legal, seat_one_hot=seat_one_hot)

        # Schritt ausführen
        time_step = env.step([action])

        # Schritt-Reward für Lern-Seat berechnen & im Buffer aktualisieren
        if player == LEARNER_SEAT:
            hand_after = shaper.hand_size(time_step, player_id=player, deck_size=deck_size)
            r_step = shaper.step_reward(
                mode=STEP_REWARD,
                hand_before=hand_before,
                hand_after=hand_after,
                time_step=time_step,
                player_id=player,
                deck_size=deck_size,
            )
            if PARAMETER_SHARING:
                agents[0].post_step(r_step, done=time_step.last())
            else:
                agents[player].post_step(r_step, done=time_step.last())

    # === Episodenende: terminales Shaping & Training ===
    # Nur der Lern-Seat erhält finale Rewards & trainiert
    if shaper.include_env_reward():
        if PARAMETER_SHARING:
            agents[0]._buffer.finalize_last_reward(time_step.rewards[LEARNER_SEAT])
        else:
            agents[LEARNER_SEAT]._buffer.finalize_last_reward(time_step.rewards[LEARNER_SEAT])

    final_bonus = shaper.final_bonus(
        mode=FINAL_REWARD,
        final_rewards=time_step.rewards,
        player_id=LEARNER_SEAT,
    )
    if PARAMETER_SHARING:
        agents[0]._buffer.finalize_last_reward(final_bonus)
        agents[0].train()
    else:
        agents[LEARNER_SEAT]._buffer.finalize_last_reward(final_bonus)
        agents[LEARNER_SEAT].train()

    # === Snapshot-Logik ===
    if episode % SNAPSHOT_INTERVAL == 0:
        # nimm die aktuelle Policy des Lerners in den Pool (FIFO)
        learner_policy = agents[0]._policy if PARAMETER_SHARING else agents[LEARNER_SEAT]._policy
        opponent_pool.append(make_snapshot(learner_policy))
        if len(opponent_pool) > OPP_POOL_CAP:
            opponent_pool.pop(0)

    # === Evaluation & Logging ===
    if episode % EVAL_INTERVAL == 0:
        print(f"\n==== Evaluation nach {episode} Episoden ====")
        per_opponent_winrates = []

        for opponent_name in opponent_strategies:
            opponent_strategy = strategy_map[opponent_name]
            wins = 0
            for _ in range(EVAL_EPISODES):
                state = game.new_initial_state()
                while not state.is_terminal():
                    pid = state.current_player()
                    legal = state.legal_actions(pid)

                    if pid == 0:  # Lern-Seat
                        obs = state.information_state_tensor(pid)
                        # Seat-One-Hot ggf. anhängen
                        if ADD_SEAT_ID:
                            seat_oh = np.zeros(num_players, dtype=np.float32)
                            seat_oh[pid] = 1.0
                            x = np.concatenate([np.array(obs, dtype=np.float32), seat_oh], axis=0)
                        else:
                            x = np.array(obs, dtype=np.float32)
                        device = next(agents[0]._policy.parameters()).device
                        x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
                        with torch.no_grad():
                            logits = agents[0]._policy(x_t).squeeze(0)
                            legal_mask = torch.zeros(num_actions, dtype=torch.float32, device=device)
                            legal_mask[legal] = 1.0
                            probs_t = ppo.masked_softmax(logits, legal_mask)
                        probs = probs_t.detach().cpu().numpy()
                        action = int(np.random.choice(len(probs), p=probs))
                    else:
                        action = opponent_strategy(state)

                    state.apply_action(action)

                if state.returns()[0] == max(state.returns()):
                    wins += 1

            winrate = 100.0 * wins / EVAL_EPISODES
            winrates[opponent_name].append(winrate)
            per_opponent_winrates.append(winrate)
            print(f"✅ Winrate gegen {opponent_name}: {winrate:.1f}%")

        macro_avg = float(np.mean(per_opponent_winrates)) if per_opponent_winrates else 0.0
        macro_winrates.append(macro_avg)
        print(f"📊 Macro Average: {macro_avg:.2f}%")

        # Modelle speichern
        model_path_i = os.path.join(MODEL_BASE, f"ppo_model_{VERSION}_agent_p0_ep{episode:07d}")
        (agents[0] if PARAMETER_SHARING else agents[LEARNER_SEAT]).save(model_path_i)
        print(f"💾 Modell gespeichert unter: {model_path_i}")

        # Lernkurven plotten
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
