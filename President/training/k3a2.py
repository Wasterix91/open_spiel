# -*- coding: utf-8 -*-
# President/training/k3a2.py — DQN (K3): League Selfplay (Pool trainierbarer Policies)

import os, re, datetime, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
import pyspiel
from open_spiel.python import rl_environment
from agents import dqn_agent as dqn
from utils.fit_tensor import FeatureConfig, augment_observation

# ============== CONFIG (wie k1a1/k3a1) ==============
CONFIG = {
    "EPISODES":        10_000,
    "EVAL_INTERVAL":   2_000,
    "EVAL_EPISODES":   2_000,
    "DECK_SIZE":       "64",       # "32" | "52" | "64"
    "SEED":            123,

    # DQN-Hyperparameter
    "DQN": {
        "learning_rate": 1e-3,
        "batch_size": 64,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.997,
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

    # Feature-Flags (K3: One-Hot AN)
    "FEATURES": {
        "NORMALIZE": False,
        "SEAT_ONEHOT": True,
    },

    # League/Selfplay
    "LEAGUE": {
        "POOL_SIZE": 6,
        "SAVE_ALL": False,
        "MAIN_INDEX": 0,
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

    version = find_next_version(MODELS, "dqn_league")
    model_dir = os.path.join(MODELS, f"dqn_league_{version}", "train"); os.makedirs(model_dir, exist_ok=True)

    dqn_cfg = dqn.DQNConfig(**CONFIG["DQN"])
    agents = [dqn.DQNAgent(state_size=state_size, num_actions=A, config=dqn_cfg) for _ in range(CONFIG["LEAGUE"]["POOL_SIZE"])]

    shaper = RewardShaper(CONFIG["REWARD"])

    # Run‑Log
    pd.DataFrame([{
        "version":version,"timestamp":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent_type":"DQN-K3-LEAGUE","num_episodes":CONFIG["EPISODES"],"eval_interval":CONFIG["EVAL_INTERVAL"],
        "eval_episodes":CONFIG["EVAL_EPISODES"],"deck_size":CONFIG["DECK_SIZE"],
        "observation_dim":state_size,"num_actions":A,"model_version_dir":model_dir,
        "normalize":CONFIG["FEATURES"]["NORMALIZE"],"seat_onehot":CONFIG["FEATURES"]["SEAT_ONEHOT"],
        "pool_size":CONFIG["LEAGUE"]["POOL_SIZE"],
    }]).to_csv(os.path.join(os.path.dirname(model_dir), "training_runs.csv"), index=False)

    EINT, EEPS = CONFIG["EVAL_INTERVAL"], CONFIG["EVAL_EPISODES"]
    eval_wrs=[]

    for ep in range(1, CONFIG["EPISODES"]+1):
        # random Lineup ziehen
        lineup_idx = np.random.choice(len(agents), size=num_players, replace=False)
        lineup = [agents[i] for i in lineup_idx]

        ts = env.reset()
        last_idx = {p: None for p in range(num_players)}

        while not ts.last():
            p = ts.observations["current_player"]
            legal = ts.observations["legal_actions"][p]

            base_s = np.array(env._state.observation_tensor(p), dtype=np.float32)
            s = augment_observation(base_s, player_id=p, cfg=feat_cfg)
            a = int(lineup[p].select_action(s, legal))

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

            ag = lineup[p]
            ag.buffer.add(s, a, float(r), s_next, bool(ts_next.last()),
                          next_legal_actions=(ts_next.observations["legal_actions"][p] if not ts_next.last() else None))
            last_idx[p] = len(ag.buffer.buffer) - 1
            ag.train_step()

            ts = ts_next

        # Terminal: optional Env-Reward auf letzte Transition pro Sitz/Agent
        if shaper.include_env_reward():
            finals = env._state.returns()
            for seat in range(num_players):
                li = last_idx[seat]
                if li is None: continue
                ag = lineup[seat]
                buf = ag.buffer
                old = buf.buffer[li]
                new = buf.Experience(old.state, old.action, float(old.reward + finals[seat]), old.next_state, old.done, old.next_legal_mask)
                buf.buffer[li] = new

        # --- Eval & Save ---
        if ep % EINT == 0:
            MAIN = CONFIG["LEAGUE"]["MAIN_INDEX"]
            wins=0
            for _ in range(EEPS):
                st = game.new_initial_state()
                others = [i for i in range(len(agents)) if i != MAIN]
                sampled = np.random.choice(others, size=(num_players-1), replace=False)
                eval_lineup_idx = [MAIN] + list(sampled)
                eval_lineup = [agents[i] for i in eval_lineup_idx]

                # Exploration aus für Eval
                eps_old = [ag.epsilon for ag in eval_lineup]
                for ag in eval_lineup: ag.epsilon = 0.0

                while not st.is_terminal():
                    pid = st.current_player(); legal = st.legal_actions(pid)
                    ob_base = np.array(st.observation_tensor(pid), dtype=np.float32)
                    ob = augment_observation(ob_base, player_id=pid, cfg=feat_cfg)
                    a = int(eval_lineup[pid].select_action(ob, legal))
                    st.apply_action(a)
                if st.returns()[0] == max(st.returns()): wins += 1

                # Epsilon rücksetzen
                for ag, e in zip(eval_lineup, eps_old): ag.epsilon = e

            wr=100.0*wins/EEPS; eval_wrs.append(wr)
            print(f"✅ Eval (League/DQN) nach {ep} Episoden – MAIN Winrate: {wr:.1f}%")

            # Speichern
            if CONFIG["LEAGUE"]["SAVE_ALL"]:
                for i, ag in enumerate(agents):
                    base=os.path.join(model_dir, f"dqn_league_{version}_agent_{i}_ep{ep:07d}"); ag.save(base)
            else:
                base=os.path.join(model_dir, f"dqn_league_{version}_MAIN_ep{ep:07d}"); agents[MAIN].save(base)

    if eval_wrs:
        xs=list(range(EINT, CONFIG["EPISODES"]+1, EINT))
        plt.figure(figsize=(10,6)); plt.plot(xs, eval_wrs, marker="o"); plt.title("K3 – DQN League Selfplay (MAIN)")
        plt.xlabel("Episode"); plt.ylabel("Winrate (%)"); plt.grid(True)
        plots=os.path.join(os.path.dirname(model_dir),"plots"); os.makedirs(plots,exist_ok=True)
        out=os.path.join(plots,"lernkurven_k3_dqn.png"); plt.tight_layout(); plt.savefig(out); plt.close()
        print(f"📄 Lernkurve gespeichert unter: {out}")

if __name__=="__main__": main()
