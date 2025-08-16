# -*- coding: utf-8 -*-
# President/training/k1a2.py — DQN (K1): Single-Agent vs Heuristiken

import os, re, datetime, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
import pyspiel
from open_spiel.python import rl_environment
from agents import dqn_agent as dqn
from utils.fit_tensor import FeatureConfig, augment_observation
from utils.strategies import STRATS
from utils.training_eval_plots import EvalPlotter

# ============== CONFIG  ==============
CONFIG = {
    "EPISODES":        10_000,
    "EVAL_INTERVAL":   1_000,
    "EVAL_EPISODES":   2_000,
    "DECK_SIZE":       "64",      # "32" | "52" | "64"
    "SEED":            42,

    # Training-Gegner (Heuristiken)
    "OPPONENTS":       ["max_combo", "max_combo", "max_combo"],

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

    # Reward-Shaping (analog zu k1a1)
    "REWARD": {
        "STEP": "delta_hand",     # "none" | "delta_hand" | "hand_penalty"
        "DELTA_WEIGHT": 1.0,
        "HAND_PENALTY_COEFF": 0.0,
        "FINAL": "none",          # "none" | "placement_bonus"
        "BONUS_WIN": 0.0, "BONUS_2ND": 0.0, "BONUS_3RD": 0.0, "BONUS_LAST": 0.0,
        "ENV_REWARD": True,
    },

    # Feature-Flags: Normalisierung/Seat-One-Hot
    "FEATURES": {
        "NORMALIZE": False,
        "SEAT_ONEHOT": False,
    },

    # Eval-Kurven (identisch zu k1a1)
    "EVAL_CURVES": ["single_only", "max_combo", "random2"],
}

# ============== Helpers / Heuristiken ==============
def find_next_version(models_root, prefix):
    pat = re.compile(rf"^{re.escape(prefix)}_(\d{{2}})$"); os.makedirs(models_root, exist_ok=True)
    existing = [int(m.group(1)) for m in (pat.match(n) for n in os.listdir(models_root)) if m]
    return f"{max(existing)+1:02d}" if existing else "01"

class RewardShaper:
    def __init__(self, cfg):
        final = "placement_bonus" if cfg["FINAL"] in ("final_reward","placement_bonus") else cfg["FINAL"]
        self.step, self.final, self.env = cfg["STEP"], final, bool(cfg["ENV_REWARD"])
        self.dw, self.hp = float(cfg["DELTA_WEIGHT"]), float(cfg["HAND_PENALTY_COEFF"])
        self.b = (float(cfg["BONUS_WIN"]), float(cfg["BONUS_2ND"]), float(cfg["BONUS_3RD"]), float(cfg["BONUS_LAST"]))
    @staticmethod
    def _ranks(deck): return 8 if deck in (32,64) else 13 if deck==52 else (_ for _ in ()).throw(ValueError("deck"))
    def hand_size(self, ts, pid, deck): return int(sum(ts.observations["info_state"][pid][:self._ranks(deck)]))
    def step_reward(self, **kw):
        if self.step=="none": return 0.0
        if self.step=="delta_hand": return self.dw*max(0.0, float(kw["hand_before"]-kw["hand_after"]))
        if self.step=="hand_penalty": return -self.hp*float(self.hand_size(kw["time_step"], kw["player_id"], kw["deck_size"]))
        raise ValueError(self.step)
    def final_bonus(self, finals, pid):
        if self.final=="none": return 0.0
        order = sorted(range(len(finals)), key=lambda p: finals[p], reverse=True)
        place = order.index(pid)+1
        return (self.b[0],self.b[1],self.b[2],self.b[3])[place-1]
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

    # Basisdim aus Spiel holen und ggf. um Seat-One-Hot erweitern
    base_dim = game.observation_tensor_shape()[0]
    extra_dim = num_players if CONFIG["FEATURES"]["SEAT_ONEHOT"] else 0
    state_size = base_dim + extra_dim

    feat_cfg = FeatureConfig(num_players=num_players, num_ranks=num_ranks,
                             add_seat_onehot=bool(CONFIG["FEATURES"]["SEAT_ONEHOT"]),
                             normalize=bool(CONFIG["FEATURES"]["NORMALIZE"]))

    version = find_next_version(MODELS, "dqn_model")
    model_dir = os.path.join(MODELS, f"dqn_model_{version}", "train"); os.makedirs(model_dir, exist_ok=True)

    # Eval-Plotter wie in k1a1
    plotter = EvalPlotter(
        opponent_names=list(CONFIG["EVAL_CURVES"]),
        out_dir=model_dir,
        filename_prefix="lernkurve",
        csv_filename="eval_curves.csv",
        save_csv=True,
    )

    dqn_cfg = dqn.DQNConfig(**CONFIG["DQN"])
    agent = dqn.DQNAgent(state_size=state_size, num_actions=A, config=dqn_cfg)
    opponents = [STRATS[name] for name in CONFIG["OPPONENTS"]]
    shaper = RewardShaper(CONFIG["REWARD"])

    # run log
    pd.DataFrame([{
        "version":version,"timestamp":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent_type":"DQN","num_episodes":CONFIG["EPISODES"],"eval_interval":CONFIG["EVAL_INTERVAL"],
        "eval_episodes":CONFIG["EVAL_EPISODES"],"deck_size":CONFIG["DECK_SIZE"],
        "observation_dim":state_size,"num_actions":A,"model_version_dir":model_dir,
        "step_reward":CONFIG["REWARD"]["STEP"],"final_reward":CONFIG["REWARD"]["FINAL"],"env_reward":CONFIG["REWARD"]["ENV_REWARD"],
        "normalize":CONFIG["FEATURES"]["NORMALIZE"],"seat_onehot":CONFIG["FEATURES"]["SEAT_ONEHOT"],
        "opponents":",".join(CONFIG["OPPONENTS"]),
    }]).to_csv(os.path.join(os.path.dirname(model_dir), "training_runs.csv"), index=False)

    EINT, EEPS = CONFIG["EVAL_INTERVAL"], CONFIG["EVAL_EPISODES"]

    for ep in range(1, CONFIG["EPISODES"]+1):
        ts = env.reset()
        last_idx_p = {p: None for p in range(num_players)}

        while not ts.last():
            p = ts.observations["current_player"]; legal = ts.observations["legal_actions"][p]

            # Beobachtung pro Sitz
            base_s = np.array(env._state.observation_tensor(p), dtype=np.float32)
            s = augment_observation(base_s, player_id=p, cfg=feat_cfg)

            # Aktion (gleiche Policy auf allen Seats -> Shared Weights, aber K1-Szenario: nur p0 wird trainiert)
            if p == 0:
                a = int(agent.select_action(s, legal))
            else:
                # Heuristiken als Gegner
                a = int(opponents[p-1](env._state))

            ts_next = env.step([a])

            # Next-Obs bauen (auch im Terminal einen Vektor – s_next ignoriert sich über done)
            if not ts_next.last():
                base_s_next = np.array(env._state.observation_tensor(p), dtype=np.float32)
                s_next = augment_observation(base_s_next, player_id=p, cfg=feat_cfg)
                next_legals = ts_next.observations["legal_actions"][p]
            else:
                s_next = s
                # WICHTIG: immer eine Next-Legal-Maske liefern (auch im Terminal),
                # sonst mischt der Replay Buffer None/ndarrays → np.stack crasht.
                next_legals = list(range(A))

            # Nur p0 lernt (wie K1-Setup)
            if p == 0:
                hand_before = shaper.hand_size(ts, p, deck_int)
                hand_after  = shaper.hand_size(ts_next, p, deck_int)
                r = shaper.step_reward(hand_before=hand_before, hand_after=hand_after,
                                       time_step=ts_next, player_id=p, deck_size=deck_int)

                agent.buffer.add(s, a, float(r), s_next, bool(ts_next.last()),
                                 next_legal_actions=next_legals)
                last_idx_p[p] = len(agent.buffer.buffer) - 1
                agent.train_step()

            ts = ts_next

        # Terminal: optional Env-Reward + Finalbonus auf die letzte Transition von P0 addieren
        li = last_idx_p.get(0, None)
        if li is not None:
            bonus = 0.0
            if shaper.include_env_reward():
                bonus += ts.rewards[0]
            bonus += shaper.final_bonus(ts.rewards, 0)
            if abs(bonus) > 1e-8:
                buf = agent.buffer
                old = buf.buffer[li]
                new = buf.Experience(old.state, old.action, float(old.reward + bonus), old.next_state, old.done, old.next_legal_mask)
                buf.buffer[li] = new

        # Evaluation (identisch zu k1a1 – per Gegner + Macro + Plotter)
        if ep % EINT == 0:
            per_opponent = {}
            old_eps = agent.epsilon
            agent.epsilon = 0.0  # greedy Eval

            for opp_name in CONFIG["EVAL_CURVES"]:
                opp_fn = STRATS[opp_name]
                wins = 0
                for _ in range(EEPS):
                    st = game.new_initial_state()
                    while not st.is_terminal():
                        pid = st.current_player(); legal = st.legal_actions(pid)
                        if pid == 0:
                            obs_base = np.array(st.observation_tensor(pid), dtype=np.float32)
                            obs = augment_observation(obs_base, player_id=pid, cfg=feat_cfg)
                            a = int(agent.select_action(obs, legal))
                        else:
                            a = int(opp_fn(st))
                        st.apply_action(a)
                    if st.returns()[0] == max(st.returns()):
                        wins += 1
                wr = 100.0 * wins / EEPS
                per_opponent[opp_name] = wr
                print(f"✅ Eval nach {ep} – Winrate vs {opp_name}: {wr:.1f}%")

            agent.epsilon = old_eps

            macro = float(np.mean(list(per_opponent.values())))
            print(f"📊 Macro Average: {macro:.2f}%")

            # loggen & plotten
            plotter.add(ep, per_opponent)
            plotter.plot_all()

            # Checkpoints (latest + episodiert)
            base_latest = os.path.join(model_dir, f"dqn_model_{version}_agent_p0")
            base_ep     = os.path.join(model_dir, f"dqn_model_{version}_agent_p0_ep{ep:07d}")
            agent.save(base_latest); agent.save(base_ep)
            print(f"💾 Modelle gespeichert: {base_latest}_*, {base_ep}_*")

    print("✅ K1 DQN Training abgeschlossen.")

if __name__=="__main__": main()
