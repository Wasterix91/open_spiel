# -*- coding: utf-8 -*-
# President/training/k2a2.py — DQN (K2): 4 simultan lernende Agents

import os, re, datetime, numpy as np, pandas as pd, torch
import pyspiel
from open_spiel.python import rl_environment
from agents import dqn_agent as dqn
from utils.fit_tensor import FeatureConfig, augment_observation

# ============== CONFIG (wie k1a1) ==============
CONFIG = {
    "EPISODES":        10_000,
    "EVAL_INTERVAL":   2000,
    "DECK_SIZE":       "64",       # "32" | "52" | "64"
    "SEED":            123,

    # DQN-Hyperparameter
    "DQN": {
        "learning_rate": 1e-3,
        "batch_size": 64,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay": 0.995,
        "buffer_size": 100_000,
        "target_update_freq": 1000,
        "soft_target_tau": 0.0,
        "max_grad_norm": 0.0,
        "use_double_dqn": True,
        "loss_huber_delta": 1.0,
    },

    # Reward-Shaping
    "REWARD": {
        "STEP": "delta_hand",
        "DELTA_WEIGHT": 1.0,
        "HAND_PENALTY_COEFF": 0.0,
        "FINAL": "none",
        "BONUS_WIN": 0.0, "BONUS_2ND": 0.0, "BONUS_3RD": 0.0, "BONUS_LAST": 0.0,
        "ENV_REWARD": True,
    },

    # Feature-Flags
    "FEATURES": {
        "NORMALIZE": False,
        "SEAT_ONEHOT": False,   # K2: getrennte Policies, daher i.d.R. False
    },
}

# ============== Helpers ==============
def find_next_version(models_root, prefix):
    pat = re.compile(rf"^{re.escape(prefix)}_(\d{{2}})$")
    os.makedirs(models_root, exist_ok=True)
    existing = [int(m.group(1)) for m in (pat.match(n) for n in os.listdir(models_root)) if m]
    return f"{max(existing)+1:02d}" if existing else "01"

class RewardShaper:
    def __init__(self, cfg):
        final = "placement_bonus" if cfg["FINAL"] in ("final_reward","placement_bonus") else cfg["FINAL"]
        self.step, self.final, self.env = cfg["STEP"], final, bool(cfg["ENV_REWARD"])
        self.dw, self.hp = float(cfg["DELTA_WEIGHT"]), float(cfg["HAND_PENALTY_COEFF"])
    @staticmethod
    def _ranks(deck): return 8 if deck in (32,64) else 13 if deck==52 else (_ for _ in ()).throw(ValueError("deck"))
    def hand_size(self, ts, pid, deck): return int(sum(ts.observations["info_state"][pid][:self._ranks(deck)]))
    def step_reward(self, **kw):
        if self.step=="none": return 0.0
        if self.step=="delta_hand": return self.dw*max(0.0, float(kw["hand_before"]-kw["hand_after"]))
        if self.step=="hand_penalty": return -self.hp*float(self.hand_size(kw["time_step"], kw["player_id"], kw["deck_size"]))
        raise ValueError(self.step)
    def include_env_reward(self): return self.env

# =========================== Training ===========================
def main():
    np.random.seed(CONFIG["SEED"]); torch.manual_seed(CONFIG["SEED"])
    ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); MODELS=os.path.join(ROOT,"models")

    game = pyspiel.load_game("president", {
        "num_players":4, "deck_size":CONFIG["DECK_SIZE"], "shuffle_cards":True, "single_card_mode":False,
    })
    env = rl_environment.Environment(game)
    A = env.action_spec()["num_actions"]

    deck_int = int(CONFIG["DECK_SIZE"])  # 32/52/64
    num_players = game.num_players()
    num_ranks   = 8 if deck_int in (32,64) else 13 if deck_int==52 else (_ for _ in ()).throw(ValueError("deck"))

    base_dim = game.observation_tensor_shape()[0]
    extra_dim = num_players if CONFIG["FEATURES"]["SEAT_ONEHOT"] else 0
    state_size = base_dim + extra_dim

    feat_cfg = FeatureConfig(num_players=num_players, num_ranks=num_ranks,
                             add_seat_onehot=bool(CONFIG["FEATURES"]["SEAT_ONEHOT"]),
                             normalize=bool(CONFIG["FEATURES"]["NORMALIZE"]))

    version = find_next_version(MODELS, "dqn_model")
    model_dir = os.path.join(MODELS, f"dqn_model_{version}", "train"); os.makedirs(model_dir, exist_ok=True)

    dqn_cfg = dqn.DQNConfig(**CONFIG["DQN"])
    agents = [dqn.DQNAgent(state_size=state_size, num_actions=A, config=dqn_cfg) for _ in range(num_players)]
    shaper = RewardShaper(CONFIG["REWARD"])

    # run log
    pd.DataFrame([{
        "version":version,"timestamp":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent_type":"DQN-K2","num_episodes":CONFIG["EPISODES"],"eval_interval":CONFIG["EVAL_INTERVAL"],
        "deck_size":CONFIG["DECK_SIZE"],
        "observation_dim":state_size,"num_actions":A,"model_version_dir":model_dir,
        "normalize":CONFIG["FEATURES"]["NORMALIZE"],"seat_onehot":CONFIG["FEATURES"]["SEAT_ONEHOT"],
    }]).to_csv(os.path.join(os.path.dirname(model_dir), "training_runs.csv"), index=False)

    EINT = CONFIG["EVAL_INTERVAL"]

    for ep in range(1, CONFIG["EPISODES"]+1):
        ts = env.reset()
        # Tracke letzten Buffer-Index pro Sitz (für Terminal-Env-Reward)
        last_idx = {p: None for p in range(num_players)}

        while not ts.last():
            p = ts.observations["current_player"]
            legal = ts.observations["legal_actions"][p]

            base_s = np.array(env._state.observation_tensor(p), dtype=np.float32)
            s = augment_observation(base_s, player_id=p, cfg=feat_cfg)
            a = int(agents[p].select_action(s, legal))

            ts_next = env.step([a])

            if not ts_next.last():
                base_s_next = np.array(env._state.observation_tensor(p), dtype=np.float32)
                s_next = augment_observation(base_s_next, player_id=p, cfg=feat_cfg)
            else:
                s_next = s

            hand_before = shaper.hand_size(ts, p, deck_int)
            hand_after  = shaper.hand_size(ts_next, p, deck_int)
            r = shaper.step_reward(hand_before=hand_before, hand_after=hand_after,
                                   time_step=ts_next, player_id=p, deck_size=deck_int)

            agents[p].buffer.add(s, a, float(r), s_next, bool(ts_next.last()),
                                 next_legal_actions=(ts_next.observations["legal_actions"][p] if not ts_next.last() else None))
            last_idx[p] = len(agents[p].buffer.buffer) - 1
            agents[p].train_step()

            ts = ts_next

        # Terminal: optional Env-Reward pro Agent auf letzte Transition addieren
        if shaper.include_env_reward():
            returns = env._state.returns()
            for p in range(num_players):
                li = last_idx[p]
                if li is None: continue
                buf = agents[p].buffer
                old = buf.buffer[li]
                new = buf.Experience(old.state, old.action, float(old.reward + returns[p]), old.next_state, old.done, old.next_legal_mask)
                buf.buffer[li] = new

        # Speichern
        if ep % EINT == 0:
            for i in range(num_players):
                base = os.path.join(model_dir, f"dqn_model_{version}_agent_p{i}_ep{ep:07d}")
                agents[i].save(base)
            print(f"💾 Modelle gespeichert nach Episode {ep}")

    print("✅ K2 DQN Training fertig.")

if __name__=="__main__": main()
