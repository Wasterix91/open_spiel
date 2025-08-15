# training/train_k4_shared_ppo.py
import os, re, datetime
import numpy as np
import pandas as pd
import torch
import pyspiel
from open_spiel.python import rl_environment
from agents import ppo_agent as ppo

# --------- K4: Shared Policy (alle Seats teilen 1 Netz) ---------

# Reward-Shaping
STEP_REWARD = "delta_hand"    # "none" | "delta_hand" | "hand_penalty"
DELTA_WEIGHT = 1.0
HAND_PENALTY_COEFF = 0.0
FINAL_REWARD = "none"         # oder "placement_bonus"
BONUS_WIN, BONUS_2ND, BONUS_3RD, BONUS_LAST = 0.0, 0.0, 0.0, 0.0
ENV_REWARD = True

# Shared-Policy
ADD_SEAT_ID = False           # optional One-Hot (empf. False bei echter Shared Policy)
NUM_EPISODES = 200_000
SAVE_INTERVAL = 10_000        # speichert episodierte Checkpoints (p0..p3 identisch)

# Spiel
GAME_SETTINGS = {
    "num_players": 4,
    "deck_size": "64",        # "32" | "52" | "64"
    "shuffle_cards": True,
    "single_card_mode": False,
}

# Pfade & Versionierung
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)  # Projekt-Root
MODELS_ROOT = os.path.join(ROOT, "models")

def find_next_version(base_dir):
    pattern = re.compile(r"ppo_model_(\d{2})$")
    os.makedirs(base_dir, exist_ok=True)
    existing = [int(m.group(1)) for m in (pattern.match(n) for n in os.listdir(base_dir)) if m]
    return f"{max(existing)+1:02d}" if existing else "01"

VERSION = find_next_version(MODELS_ROOT)
MODEL_DIR = os.path.join(MODELS_ROOT, f"ppo_model_{VERSION}", "train")
os.makedirs(MODEL_DIR, exist_ok=True)
print(f"📁 Neue Trainingsversion: {VERSION}")

# ---- Reward Shaper ----
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
        self.bonus_win, self.bonus_2nd = float(bonus_win), float(bonus_2nd)
        self.bonus_3rd, self.bonus_last = float(bonus_3rd), float(bonus_last)

    @staticmethod
    def _num_ranks_for_deck(ds):
        if ds in (32, 64): return 8
        if ds == 52: return 13
        raise NotImplementedError(f"Unsupported deck size: {ds}")

    def hand_size(self, time_step, player_id, deck_size: int) -> int:
        nr = self._num_ranks_for_deck(deck_size)
        hand = time_step.observations["info_state"][player_id]
        return int(sum(hand[:nr]))

    def step_reward(self, *, hand_before=None, hand_after=None,
                    time_step=None, player_id=None, deck_size=None) -> float:
        if self.step_mode == "none": return 0.0
        if self.step_mode == "delta_hand":
            diff = max(0.0, float(hand_before - hand_after))
            return self.delta_weight * diff
        if self.step_mode == "hand_penalty":
            size = self.hand_size(time_step, player_id, deck_size)
            return self.hand_coeff * float(size)
        raise ValueError(f"Unbekannter STEP_REWARD: {self.step_mode}")

    def final_bonus(self, final_rewards, player_id) -> float:
        mode = self.final_mode
        if mode == "none": return 0.0
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

# ---- Setup Game & Agent ----
game = pyspiel.load_game("president", GAME_SETTINGS)
env = rl_environment.Environment(game)
info_state_size = env.observation_spec()["info_state"][0]
num_actions = env.action_spec()["num_actions"]
num_players = GAME_SETTINGS["num_players"]
seat_id_dim = num_players if ADD_SEAT_ID else 0
deck_size_int = int(GAME_SETTINGS["deck_size"])

agent = ppo.PPOAgent(info_state_size, num_actions, seat_id_dim=seat_id_dim)
shaper = RewardShaper(
    STEP_REWARD, FINAL_REWARD, ENV_REWARD,
    DELTA_WEIGHT, HAND_PENALTY_COEFF,
    BONUS_WIN, BONUS_2ND, BONUS_3RD, BONUS_LAST
)

# ---- Konfig protokollieren ----
runs_csv = os.path.join(os.path.dirname(MODELS_ROOT), "training_runs.csv")
meta = {
    "version": VERSION,
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "agent_type": "PPO",
    "num_episodes": NUM_EPISODES,
    "save_interval": SAVE_INTERVAL,
    "num_players": GAME_SETTINGS["num_players"],
    "deck_size": GAME_SETTINGS["deck_size"],
    "shuffle_cards": GAME_SETTINGS["shuffle_cards"],
    "single_card_mode": GAME_SETTINGS["single_card_mode"],
    "observation_dim": info_state_size + seat_id_dim,
    "num_actions": num_actions,
    "shared_policy": True,
    "add_seat_id": ADD_SEAT_ID,
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
pd.DataFrame([meta]).to_csv(runs_csv, mode="a", index=False, header=not os.path.exists(runs_csv))
print(f"📄 Konfiguration geloggt: {runs_csv}")

# ---- Training ----
for episode in range(1, NUM_EPISODES + 1):
    ts = env.reset()
    # Tracke für jeden Seat die letzte Transition-Position im gemeinsamen Buffer
    last_idx_by_seat = {p: None for p in range(num_players)}

    while not ts.last():
        pid = ts.observations["current_player"]
        legal = ts.observations["legal_actions"][pid]
        info_state = ts.observations["info_state"][pid]

        seat_oh = None
        if ADD_SEAT_ID:
            seat_oh = np.zeros(num_players, dtype=np.float32); seat_oh[pid] = 1.0

        hand_before = shaper.hand_size(ts, player_id=pid, deck_size=deck_size_int)
        action = agent.step(info_state, legal, seat_one_hot=seat_oh)
        # merke Index der soeben hinzugefügten Transition
        last_idx_by_seat[pid] = len(agent._buffer.states) - 1

        ts = env.step([action])

        hand_after = shaper.hand_size(ts, player_id=pid, deck_size=deck_size_int)
        r_step = shaper.step_reward(
            hand_before=hand_before, hand_after=hand_after,
            time_step=ts, player_id=pid, deck_size=deck_size_int
        )
        agent.post_step(r_step, done=ts.last())

    # Terminal: Env-Reward + Finalbonus JE SEAT auf dessen letzte Transition buchen
    if shaper.include_env_reward():
        for p in range(num_players):
            li = last_idx_by_seat[p]
            if li is not None:
                agent._buffer.rewards[li] += ts.rewards[p]
    for p in range(num_players):
        li = last_idx_by_seat[p]
        if li is not None:
            agent._buffer.rewards[li] += shaper.final_bonus(ts.rewards, player_id=p)

    agent.train()

    # Speichern (einheitliches Netz als p0..p3, damit Eval pro Seat laden kann)
    if episode % SAVE_INTERVAL == 0:
        tag = f"{episode:07d}"
        for p in range(num_players):
            base = os.path.join(MODEL_DIR, f"ppo_model_{VERSION}_agent_p{p}_ep{tag}")
            torch.save(agent._policy.state_dict(), base + "_policy.pt")
            torch.save(agent._value.state_dict(),  base + "_value.pt")
        print(f"💾 Checkpoint gespeichert: Version {VERSION}, Episode {episode}")

print("✅ Training abgeschlossen.")
