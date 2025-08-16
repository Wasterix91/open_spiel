# -*- coding: utf-8 -*-
# President/training/k1a1.py — PPO (K1): Single-Agent vs Heuristiken
# Speichert jetzt in: models/k1a1/model_XX/{config.csv, plots/, models/}

import os, re, datetime, numpy as np, pandas as pd, torch
import pyspiel
from open_spiel.python import rl_environment
from agents import ppo_agent as ppo
from utils.strategies import STRATS
from utils.fit_tensor import FeatureConfig, augment_observation
from utils.training_eval_plots import EvalPlotter

# ============== CONFIG  ==============
CONFIG = {
    "EPISODES":        20_000,
    "EVAL_INTERVAL":   2_000,
    "EVAL_EPISODES":   2_000,
    "DECK_SIZE":       "64",      # "32" | "52" | "64"
    "SEED":            42,

    # Training-Gegner (Heuristiken)
    "OPPONENTS":       ["max_combo", "max_combo", "max_combo"],

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

    # Feature-Toggles
    "FEATURES": {
        "NORMALIZE": False,       # Du wolltest ohne Normalisierung trainieren
        "SEAT_ONEHOT": False,     # K1: typischerweise False
    },

    # Eval-Kurven
    "EVAL_CURVES": ["single_only", "max_combo", "random2"],  # + Macro wird automatisch geplottet
}

# ============== Helpers / Storage ==============
def find_next_version(dir_path, prefix):
    """
    Findet die nächste laufende Nummer für Ordner wie '<prefix>_01', '<prefix>_02', ...
    """
    pat = re.compile(rf"^{re.escape(prefix)}_(\d{{2}})$")
    os.makedirs(dir_path, exist_ok=True)
    existing = [int(m.group(1)) for m in (pat.match(n) for n in os.listdir(dir_path)) if m]
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
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ---------- NEU: Zielpfade gemäß Wunschstruktur ----------
    MODELS_ROOT = os.path.join(ROOT, "models", "k2a1")
    VERSION = find_next_version(MODELS_ROOT, "model")          # "01", "02", ...
    RUN_DIR = os.path.join(MODELS_ROOT, f"model_{VERSION}")    # models/k1a1/model_XX
    PLOTS_DIR = os.path.join(RUN_DIR, "plots")
    WEIGHTS_DIR = os.path.join(RUN_DIR, "models")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    print(f"📁 Neuer Run-Ordner: {RUN_DIR}")

    # ---------- Environment ----------
    game = pyspiel.load_game("president", {
        "num_players": 4,
        "deck_size":   CONFIG["DECK_SIZE"],
        "shuffle_cards": True,
        "single_card_mode": False,
    })
    env = rl_environment.Environment(game)
    info_dim = env.observation_spec()["info_state"][0]
    A = env.action_spec()["num_actions"]

    # ---------- Features ----------
    deck_int = int(CONFIG["DECK_SIZE"])
    num_players = game.num_players()
    num_ranks   = 8 if deck_int in (32,64) else 13 if deck_int==52 else (_ for _ in ()).throw(ValueError("deck"))
    feat_cfg = FeatureConfig(
        num_players=num_players, num_ranks=num_ranks,
        add_seat_onehot=bool(CONFIG["FEATURES"]["SEAT_ONEHOT"]),
        normalize=bool(CONFIG["FEATURES"]["NORMALIZE"]),
    )

    # ---------- Plotter schreibt direkt in PLOTS_DIR ----------
    plotter = EvalPlotter(
        opponent_names=list(CONFIG["EVAL_CURVES"]),
        out_dir=PLOTS_DIR,
        filename_prefix="lernkurve",
        csv_filename="eval_curves.csv",
        save_csv=True,
    )

    # ---------- Agent, Gegner, Shaper ----------
    ppo_cfg = ppo.PPOConfig(**CONFIG["PPO"])
    seat_id_dim = (num_players if CONFIG["FEATURES"]["SEAT_ONEHOT"] else 0)
    agent = ppo.PPOAgent(info_dim, A, seat_id_dim=seat_id_dim, config=ppo_cfg)
    opponents = [STRATS[name] for name in CONFIG["OPPONENTS"]]
    shaper = RewardShaper(CONFIG["REWARD"])

    # ---------- config.csv schreiben ----------
    pd.DataFrame([{
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "script": "k1a1",
        "version": VERSION,
        "agent_type": "PPO",
        "num_episodes": CONFIG["EPISODES"],
        "eval_interval": CONFIG["EVAL_INTERVAL"],
        "eval_episodes": CONFIG["EVAL_EPISODES"],
        "deck_size": CONFIG["DECK_SIZE"],
        "observation_dim": info_dim + seat_id_dim,
        "num_actions": A,
        "reward_step": CONFIG["REWARD"]["STEP"],
        "reward_final": CONFIG["REWARD"]["FINAL"],
        "env_reward": CONFIG["REWARD"]["ENV_REWARD"],
        "delta_weight": CONFIG["REWARD"]["DELTA_WEIGHT"],
        "hand_penalty_coeff": CONFIG["REWARD"]["HAND_PENALTY_COEFF"],
        "normalize": CONFIG["FEATURES"]["NORMALIZE"],
        "seat_onehot": CONFIG["FEATURES"]["SEAT_ONEHOT"],
        "opponents": ",".join(CONFIG["OPPONENTS"]),
        "learning_rate": CONFIG["PPO"]["learning_rate"],
        "num_epochs": CONFIG["PPO"]["num_epochs"],
        "batch_size": CONFIG["PPO"]["batch_size"],
        "entropy_cost": CONFIG["PPO"]["entropy_cost"],
        "gamma": CONFIG["PPO"]["gamma"],
        "gae_lambda": CONFIG["PPO"]["gae_lambda"],
        "clip_eps": CONFIG["PPO"]["clip_eps"],
        "value_coef": CONFIG["PPO"]["value_coef"],
        "max_grad_norm": CONFIG["PPO"]["max_grad_norm"],
    }]).to_csv(os.path.join(RUN_DIR, "config.csv"), index=False)
    print(f"📝 Konfiguration gespeichert: {os.path.join(RUN_DIR, 'config.csv')}")

    # ---------- Training ----------
    EINT, EEPS = CONFIG["EVAL_INTERVAL"], CONFIG["EVAL_EPISODES"]

    for ep in range(1, CONFIG["EPISODES"] + 1):
        ts = env.reset()
        while not ts.last():
            p = ts.observations["current_player"]
            legal = ts.observations["legal_actions"][p]
            hand_before = shaper.hand_size(ts, p, deck_int)

            if p == 0:
                base_obs = ts.observations["info_state"][p]
                obs = augment_observation(base_obs, player_id=p, cfg=feat_cfg)
                seat_oh = None
                if CONFIG["FEATURES"]["SEAT_ONEHOT"]:
                    seat_oh = np.zeros(num_players, dtype=np.float32); seat_oh[p] = 1.0
                a = int(agent.step(obs, legal, seat_one_hot=seat_oh))
            else:
                a = int(opponents[p - 1](env._state))

            ts_next = env.step([a])

            if p == 0:
                hand_after = shaper.hand_size(ts_next, p, deck_int)
                r = shaper.step_reward(
                    hand_before=hand_before, hand_after=hand_after,
                    time_step=ts_next, player_id=p, deck_size=deck_int
                )
                agent.post_step(r, done=ts_next.last())

            ts = ts_next

        if shaper.include_env_reward():
            agent._buffer.finalize_last_reward(ts.rewards[0])
        agent._buffer.finalize_last_reward(shaper.final_bonus(ts.rewards, 0))
        agent.train()

        # ---------- Evaluation ----------
        if ep % EINT == 0:
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
                            with torch.no_grad():
                                logits = agent._policy(torch.tensor(ob, dtype=torch.float32))
                                mask = torch.zeros(A); mask[legal] = 1.0
                                probs = ppo.masked_softmax(logits, mask)
                            a = int(torch.distributions.Categorical(probs=probs).sample().item())
                        else:
                            a = int(opp_fn(st))
                        st.apply_action(a)
                    if st.returns()[0] == max(st.returns()):
                        wins += 1
                wr = 100.0 * wins / EEPS
                per_opponent[opp_name] = wr
                print(f"✅ Eval nach {ep:7d} – Winrate vs {opp_name:11s}: {wr:5.1f}%")

            macro = float(np.mean(list(per_opponent.values())))
            print(f"📊 Macro Average (Ep {ep}): {macro:.2f}%")

            # Plots & CSV
            plotter.add(ep, per_opponent)
            plotter.plot_all()

            # ggf. Kombiplots auf die gewünschten Namen mappen
            src_all = os.path.join(PLOTS_DIR, "lernkurve_alle_strategien.png")
            src_all_macro = os.path.join(PLOTS_DIR, "lernkurve_alle_strategien_avg.png")
            dst_all = os.path.join(PLOTS_DIR, "lernkurve_alle.png")
            dst_all_macro = os.path.join(PLOTS_DIR, "lernkurve_alle_mit_macro.png")
            try:
                if os.path.exists(src_all):
                    if os.path.exists(dst_all): os.remove(dst_all)
                    os.replace(src_all, dst_all)
                if os.path.exists(src_all_macro):
                    if os.path.exists(dst_all_macro): os.remove(dst_all_macro)
                    os.replace(src_all_macro, dst_all_macro)
            except Exception as e:
                print(f"⚠️ Hinweis: Umbenennen der Kombiplots fehlgeschlagen: {e}")

            print(f"📈 Plots aktualisiert in: {PLOTS_DIR}")

            # Gewichte speichern
            base = os.path.join(WEIGHTS_DIR, f"k1a1_model_{VERSION}_agent_p0_ep{ep:07d}")
            agent.save(base)  # erzeugt ..._policy.pt und ..._value.pt
            print(f"💾 Modell gespeichert: {base}_[policy|value].pt")

    print("✅ K1 Training abgeschlossen.")

if __name__ == "__main__":
    main()
