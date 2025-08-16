# -*- coding: utf-8 -*-
# President/training/k4a1.py — PPO (K4): Shared Policy + Eval vs Heuristiken
# Speichert in: models/k4a1/model_XX/{config.csv, plots/, models/}

import os, re, datetime, numpy as np, pandas as pd, torch
import pyspiel
from open_spiel.python import rl_environment
from agents import ppo_agent as ppo
from utils.fit_tensor import FeatureConfig, augment_observation
from utils.strategies import STRATS
from utils.training_eval_plots import EvalPlotter

# ============== CONFIG ==============
CONFIG = {
    "EPISODES":        14_000,
    "SAVE_INTERVAL":   2_000,
    "EVAL_INTERVAL":   2_000,
    "EVAL_EPISODES":   2_000,
    "DECK_SIZE":       "64",       # "32" | "52" | "64"
    "SEED":            123,

    # PPO-Hyperparameter
    "PPO": {
        "learning_rate": 3e-4,
        "num_epochs": 4,
        "batch_size": 256,
        "entropy_cost": 0.01,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "value_coef": 0.5,
        "max_grad_norm": 0.5,
    },

    # Reward-Shaping
    "REWARD": {
        "STEP": "delta_hand",     # "none" | "delta_hand" | "hand_penalty"
        "DELTA_WEIGHT": 1.0,
        "HAND_PENALTY_COEFF": 0.0,
        "FINAL": "none",          # "none" | "placement_bonus"
        "BONUS_WIN": 0.0, "BONUS_2ND": 0.0, "BONUS_3RD": 0.0, "BONUS_LAST": 0.0,
        "ENV_REWARD": True,
    },

    # Feature-Flags (Shared Policy: One-Hot i.d.R. False)
    "FEATURES": {
        "NORMALIZE": False,
        "SEAT_ONEHOT": False,
    },

    # Eval-Gegner (Heuristiken aus utils.strategies)
    "EVAL_CURVES": ["single_only", "max_combo", "random2"],
    "OPPONENTS": ["max_combo", "max_combo", "max_combo"],
}

# ============== Helpers ==============
def find_next_version(run_root):
    pat = re.compile(r"^model_(\d{2})$")
    os.makedirs(run_root, exist_ok=True)
    existing = [int(m.group(1)) for m in (pat.match(n) for n in os.listdir(run_root)) if m]
    return f"{max(existing)+1:02d}" if existing else "01"

class RewardShaper:
    def __init__(self, cfg):
        final = "placement_bonus" if cfg["FINAL"] in ("final_reward","placement_bonus") else cfg["FINAL"]
        self.step, self.final, self.env = cfg["STEP"], final, bool(cfg["ENV_REWARD"])
        self.dw, self.hp = float(cfg["DELTA_WEIGHT"]), float(cfg["HAND_PENALTY_COEFF"])
        self.b = (float(cfg["BONUS_WIN"]), float(cfg["BONUS_2ND"]),
                  float(cfg["BONUS_3RD"]), float(cfg["BONUS_LAST"]))
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

# =========================== Training + Eval ===========================
def main():
    np.random.seed(CONFIG["SEED"]); torch.manual_seed(CONFIG["SEED"])
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Neuer Speicherort: models/k4a1/model_XX/...
    RUN_ROOT = os.path.join(ROOT, "models", "k4a1")
    VERSION  = find_next_version(RUN_ROOT)
    RUN_DIR  = os.path.join(RUN_ROOT, f"model_{VERSION}")
    PLOTS_DIR  = os.path.join(RUN_DIR, "plots")
    MODELS_DIR = os.path.join(RUN_DIR, "models")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"📁 Output-Verzeichnis: {RUN_DIR}")

    # ---- Env ----
    game = pyspiel.load_game("president", {
        "num_players": 4,
        "deck_size":   CONFIG["DECK_SIZE"],
        "shuffle_cards": True,
        "single_card_mode": False,
    })
    env = rl_environment.Environment(game)
    info_dim = env.observation_spec()["info_state"][0]
    A        = env.action_spec()["num_actions"]

    # ---- Features ----
    deck_int    = int(CONFIG["DECK_SIZE"])
    num_players = game.num_players()
    num_ranks   = 8 if deck_int in (32, 64) else 13 if deck_int == 52 else (_ for _ in ()).throw(ValueError("deck"))
    feat_cfg = FeatureConfig(
        num_players=num_players,
        num_ranks=num_ranks,
        add_seat_onehot=False,                            # One-Hot nicht hier anhängen
        normalize=bool(CONFIG["FEATURES"]["NORMALIZE"]),
    )

    # ---- Plotter (legt Einzelkurven + alle(+macro) in PLOTS_DIR ab) ----
    plotter = EvalPlotter(
        opponent_names=list(CONFIG["EVAL_CURVES"]),
        out_dir=PLOTS_DIR,
        filename_prefix="lernkurve",
        csv_filename="eval_curves.csv",
        save_csv=True,
    )

    # ---- Agent ----
    ppo_cfg     = ppo.PPOConfig(**CONFIG["PPO"])
    seat_id_dim = (num_players if CONFIG["FEATURES"]["SEAT_ONEHOT"] else 0)
    agent       = ppo.PPOAgent(info_state_size=info_dim, num_actions=A, seat_id_dim=seat_id_dim, config=ppo_cfg)

    # ---- Reward + Gegner ----
    shaper    = RewardShaper(CONFIG["REWARD"])
    opponents = [STRATS[name] for name in CONFIG["OPPONENTS"]]

    # ---- config.csv schreiben ----
    cfg_row = {
        "version": VERSION,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent_type": "PPO-K4-SHARED",
        "num_episodes": CONFIG["EPISODES"],
        "save_interval": CONFIG["SAVE_INTERVAL"],
        "eval_interval": CONFIG["EVAL_INTERVAL"],
        "eval_episodes": CONFIG["EVAL_EPISODES"],
        "deck_size": CONFIG["DECK_SIZE"],
        "observation_dim": info_dim + seat_id_dim,
        "num_actions": A,
        "normalize": CONFIG["FEATURES"]["NORMALIZE"],
        "seat_onehot": CONFIG["FEATURES"]["SEAT_ONEHOT"],
        "opponents": ",".join(CONFIG["OPPONENTS"]),
        "run_dir": RUN_DIR,
    }
    pd.DataFrame([cfg_row]).to_csv(os.path.join(RUN_DIR, "config.csv"), index=False)
    print(f"📝 Konfiguration gespeichert: {os.path.join(RUN_DIR,'config.csv')}")

    SAVE_INT = int(CONFIG["SAVE_INTERVAL"])
    EVAL_INT = int(CONFIG["EVAL_INTERVAL"])
    EEPS     = int(CONFIG["EVAL_EPISODES"])

    for ep in range(1, CONFIG["EPISODES"] + 1):
        ts = env.reset()
        # letzte Transition je Sitz (für terminal shaping)
        last_idx = {p: None for p in range(num_players)}

        while not ts.last():
            p     = ts.observations["current_player"]
            legal = ts.observations["legal_actions"][p]

            base_obs = ts.observations["info_state"][p]
            obs = augment_observation(base_obs, player_id=p, cfg=feat_cfg)

            seat_oh = None
            if CONFIG["FEATURES"]["SEAT_ONEHOT"]:
                seat_oh = np.zeros((num_players,), dtype=np.float32); seat_oh[p] = 1.0

            hand_before = shaper.hand_size(ts, p, deck_int)
            a = int(agent.step(obs, legal, seat_one_hot=seat_oh))
            last_idx[p] = len(agent._buffer.states) - 1

            ts_next = env.step([a])

            hand_after = shaper.hand_size(ts_next, p, deck_int)
            r = shaper.step_reward(hand_before=hand_before, hand_after=hand_after,
                                   time_step=ts_next, player_id=p, deck_size=deck_int)
            agent.post_step(r, done=ts_next.last())

            ts = ts_next

        # Terminal: Env-Rewards + (optional) Platzierungsbonus nur auf letzte Transition pro Sitz
        if shaper.include_env_reward():
            finals = env._state.returns()
            for p in range(num_players):
                li = last_idx[p]
                if li is not None:
                    agent._buffer.rewards[li] += finals[p]
        for p in range(num_players):
            li = last_idx[p]
            if li is not None:
                agent._buffer.rewards[li] += shaper.final_bonus(env._state.returns(), p)

        # Ein gemeinsames Update (Shared Policy)
        agent.train()

        # ---------- Evaluation & Plots ----------
        if ep % EVAL_INT == 0:
            per_opponent = {}
            for opp_name in CONFIG["EVAL_CURVES"]:
                opp_fn = STRATS[opp_name]
                wins = 0
                for _ in range(EEPS):
                    st = game.new_initial_state()
                    while not st.is_terminal():
                        pid = st.current_player(); legal = st.legal_actions(pid)
                        if pid == 0:
                            ob = st.information_state_tensor(pid)
                            ob = augment_observation(ob, player_id=pid, cfg=feat_cfg)
                            seat_oh = None
                            if CONFIG["FEATURES"]["SEAT_ONEHOT"]:
                                seat_oh = np.zeros((num_players,), dtype=np.float32); seat_oh[pid] = 1.0
                            x = agent._make_input(ob, seat_one_hot=seat_oh)
                            with torch.no_grad():
                                logits = agent._policy(x)
                                mask = torch.zeros(A, device=logits.device); mask[legal] = 1.0
                                probs = ppo.masked_softmax(logits, mask)
                            a = int(torch.distributions.Categorical(probs=probs).sample().item())
                        else:
                            a = int(opp_fn(st))
                        st.apply_action(a)
                    if st.returns()[0] == max(st.returns()): wins += 1
                wr = 100.0 * wins / EEPS
                per_opponent[opp_name] = wr
                print(f"✅ Eval nach {ep} – Winrate vs {opp_name}: {wr:.1f}%")

            macro = float(np.mean(list(per_opponent.values())))
            print(f"📊 Macro Average: {macro:.2f}%")

            # Plot & CSV aktualisieren (legt alle gewünschten PNGs im PLOTS_DIR ab)
            plotter.add(ep, per_opponent)
            plotter.plot_all()

            # Modell speichern (ein Netz, kompatibel p0..p3)
            tag = f"{ep:07d}"
            for seat in range(num_players):
                base = os.path.join(MODELS_DIR, f"k4a1_model_{VERSION}_agent_p{seat}_ep{tag}")
                agent.save(base)
            print(f"💾 Modelle gespeichert (Eval): {MODELS_DIR} | Version {VERSION} | Episode {ep}")

        # ---------- Zwischen-Checkpoint ohne Eval ----------
        if ep % SAVE_INT == 0 and (ep % EVAL_INT != 0):
            tag = f"{ep:07d}"
            for seat in range(num_players):
                base = os.path.join(MODELS_DIR, f"k4a1_model_{VERSION}_agent_p{seat}_ep{tag}")
                agent.save(base)
            print(f"💾 Modelle gespeichert (Zwischenstand): {MODELS_DIR} | Version {VERSION} | Episode {ep}")

    print("✅ K4 Training abgeschlossen.")

if __name__=="__main__": main()
