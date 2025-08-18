# -*- coding: utf-8 -*-
# President/training/k4a2.py — DQN (K4): Shared Policy für alle Seats + Eval vs Heuristiken
# Speicherlayout wie bei k1a1/k1a2:
# models/k4a2/model_XX/
#   ├─ config.csv
#   ├─ timings.csv
#   ├─ plots/
#   │   ├─ lernkurve_single_only.png
#   │   ├─ lernkurve_max_combo.png
#   │   ├─ lernkurve_random2.png
#   │   ├─ lernkurve_alle.png
#   │   ├─ lernkurve_alle_mit_macro.png
#   │   └─ eval_curves.csv
#   └─ models/
#       └─ k4a2_model_XX_agent_p{0..3}_ep{NNNNNNN}_*.pt  (ein Netz, für Kompatibilität 4x gespeichert)

import os, re, time, datetime
import numpy as np, pandas as pd, torch
import pyspiel
from open_spiel.python import rl_environment

from agents import dqn_agent as dqn
from utils.fit_tensor import FeatureConfig, augment_observation
from utils.strategies import STRATS
from utils.training_eval_plots import EvalPlotter

# ===================== CONFIG ===================== #
CONFIG = {
    "EPISODES":        10_000,
    "SAVE_INTERVAL":   2_000,
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

    # Feature-Flags
    "FEATURES": {
        "NORMALIZE": False,
        "SEAT_ONEHOT": False,  # echte Shared Policy: False; testweise auf True schaltbar
    },

    # Eval-Kurven (einzeln + gemeinsamer Plot + Macro)
    "EVAL_CURVES": ["single_only", "max_combo", "random2"],

    # Gegner im Training (Seats 1–3)
    "OPPONENTS": ["max_combo", "max_combo", "max_combo"],
}

# ===================== Helpers ===================== #
def find_next_version(base_dir, prefix="model"):
    pat = re.compile(rf"^{re.escape(prefix)}_(\d{{2}})$")
    os.makedirs(base_dir, exist_ok=True)
    existing = [int(m.group(1)) for m in (pat.match(n) for n in os.listdir(base_dir)) if m]
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

def _alias_joint_plot_names(plots_dir: str):
    """Alias für konsistente Dateinamen wie bei k1a1."""
    mapping = {
        "lernkurve_alle_strategien.png": "lernkurve_alle.png",
        "lernkurve_alle_strategien_avg.png": "lernkurve_alle_mit_macro.png",
    }
    for src, dst in mapping.items():
        s = os.path.join(plots_dir, src)
        d = os.path.join(plots_dir, dst)
        if os.path.exists(s) and not os.path.exists(d):
            try: os.replace(s, d)
            except Exception: pass

# =========================== Training + Eval =========================== #
def main():
    np.random.seed(CONFIG["SEED"]); torch.manual_seed(CONFIG["SEED"])
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_ROOT = os.path.join(ROOT, "models")

    # ---- Run-Verzeichnis ----
    family_dir  = os.path.join(MODELS_ROOT, "k4a2")
    version     = find_next_version(family_dir, prefix="model")
    run_dir     = os.path.join(family_dir, f"model_{version}")
    plots_dir   = os.path.join(run_dir, "plots")
    weights_dir = os.path.join(run_dir, "models")
    os.makedirs(plots_dir, exist_ok=True); os.makedirs(weights_dir, exist_ok=True)
    print(f"📁 Neuer Lauf (k4a2): {run_dir}")

    # ---- Env ----
    game = pyspiel.load_game("president", {
        "num_players": 4,
        "deck_size":   CONFIG["DECK_SIZE"],
        "shuffle_cards": True,
        "single_card_mode": False,
    })
    env = rl_environment.Environment(game)
    A   = env.action_spec()["num_actions"]

    deck_int    = int(CONFIG["DECK_SIZE"])
    num_players = game.num_players()
    num_ranks   = 8 if deck_int in (32,64) else 13 if deck_int==52 else (_ for _ in ()).throw(ValueError("deck"))

    base_dim  = game.observation_tensor_shape()[0]
    extra_dim = num_players if CONFIG["FEATURES"]["SEAT_ONEHOT"] else 0
    state_size = base_dim + extra_dim

    feat_cfg = FeatureConfig(
        num_players=num_players, num_ranks=num_ranks,
        add_seat_onehot=bool(CONFIG["FEATURES"]["SEAT_ONEHOT"]),
        normalize=bool(CONFIG["FEATURES"]["NORMALIZE"]),
    )

    # ---- Plotter ----
    plotter = EvalPlotter(
        opponent_names=list(CONFIG["EVAL_CURVES"]),
        out_dir=plots_dir,
        filename_prefix="lernkurve",
        csv_filename="eval_curves.csv",
        save_csv=True,
    )

    # ---- Agent, Gegner, Shaper ----
    dqn_cfg  = dqn.DQNConfig(**CONFIG["DQN"])
    agent    = dqn.DQNAgent(state_size=state_size, num_actions=A, config=dqn_cfg)
    opponents = [STRATS[name] for name in CONFIG["OPPONENTS"]]
    shaper    = RewardShaper(CONFIG["REWARD"])

    # ---- config.csv ----
    pd.DataFrame([{
        "script":"k4a2","version":version,
        "timestamp":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent_type":"DQN-K4-SHARED",
        "num_episodes":CONFIG["EPISODES"],
        "save_interval":CONFIG["SAVE_INTERVAL"],
        "eval_interval":CONFIG["EVAL_INTERVAL"],
        "eval_episodes":CONFIG["EVAL_EPISODES"],
        "deck_size":CONFIG["DECK_SIZE"],
        "observation_dim":state_size,"num_actions":A,
        "normalize":CONFIG["FEATURES"]["NORMALIZE"],"seat_onehot":CONFIG["FEATURES"]["SEAT_ONEHOT"],
        "opponents":",".join(CONFIG["OPPONENTS"]),
        "models_dir":weights_dir,"plots_dir":plots_dir,
    }]).to_csv(os.path.join(run_dir, "config.csv"), index=False)
    print(f"📝 Konfiguration gespeichert: {os.path.join(run_dir, 'config.csv')}")

    # ---- Timing setup wie k1a1 ----
    timings_csv = os.path.join(run_dir, "timings.csv")
    timing_rows = []
    t0 = time.perf_counter()

    SAVE_INT = CONFIG["SAVE_INTERVAL"]
    EINT, EEPS = CONFIG["EVAL_INTERVAL"], CONFIG["EVAL_EPISODES"]

    # ===================== Loop ===================== #
    for ep in range(1, CONFIG["EPISODES"]+1):
        ep_start = time.perf_counter()
        steps = 0
        train_seconds_accum = 0.0

        ts = env.reset()
        last_idx = {p: None for p in range(num_players)}

        while not ts.last():
            steps += 1
            p = ts.observations["current_player"]
            legal = ts.observations["legal_actions"][p]

            base_s = np.array(env._state.observation_tensor(p), dtype=np.float32)
            s = augment_observation(base_s, player_id=p, cfg=feat_cfg)

            # Shared Policy spielt auf allen Seats (Selfplay)
            a = int(agent.select_action(s, legal))
            ts_next = env.step([a])

            # next state + legal mask (auch im Terminal robust setzen)
            if not ts_next.last():
                base_s_next = np.array(env._state.observation_tensor(p), dtype=np.float32)
                s_next = augment_observation(base_s_next, player_id=p, cfg=feat_cfg)
                next_legals = ts_next.observations["legal_actions"][p]
            else:
                s_next = s
                next_legals = list(range(A))

            hand_before = shaper.hand_size(ts, p, deck_int)
            hand_after  = shaper.hand_size(ts_next, p, deck_int)
            r = shaper.step_reward(hand_before=hand_before, hand_after=hand_after,
                                   time_step=ts_next, player_id=p, deck_size=deck_int)

            agent.buffer.add(s, a, float(r), s_next, bool(ts_next.last()), next_legal_actions=next_legals)
            last_idx[p] = len(agent.buffer.buffer) - 1

            tt0 = time.perf_counter()
            agent.train_step()
            train_seconds_accum += (time.perf_counter() - tt0)

            ts = ts_next

        # Terminal: Env-Rewards pro Sitz auf letzte Transition addieren
        if shaper.include_env_reward():
            finals = env._state.returns()
            buf = agent.buffer
            for p in range(num_players):
                li = last_idx[p]
                if li is None: continue
                old = buf.buffer[li]
                buf.buffer[li] = buf.Experience(old.state, old.action, float(old.reward + finals[p]),
                                                old.next_state, old.done, old.next_legal_mask)

        # Defaults für optionale Felder
        eval_seconds = 0.0
        plot_seconds = 0.0
        save_seconds = 0.0

        # ---------- Evaluation (pro Gegner + Macro) ----------
        if ep % EINT == 0:
            ev_start = time.perf_counter()

            per_opponent = {}
            old_eps = agent.epsilon
            agent.epsilon = 0.0  # greedy eval

            for opp_name in CONFIG["EVAL_CURVES"]:
                opp_fn = STRATS[opp_name]
                wins = 0
                for _ in range(EEPS):
                    st = game.new_initial_state()
                    while not st.is_terminal():
                        pid = st.current_player(); legal = st.legal_actions(pid)
                        if pid == 0:
                            ob_base = np.array(st.observation_tensor(pid), dtype=np.float32)
                            ob = augment_observation(ob_base, player_id=pid, cfg=feat_cfg)
                            a = int(agent.select_action(ob, legal))
                        else:
                            a = int(opp_fn(st))
                        st.apply_action(a)
                    if st.returns()[0] == max(st.returns()): wins += 1
                wr = 100.0 * wins / EEPS
                per_opponent[opp_name] = wr
                print(f"✅ Eval nach {ep:7d} – Winrate vs {opp_name:11s}: {wr:5.1f}%")

            agent.epsilon = old_eps
            macro = float(np.mean(list(per_opponent.values())))
            print(f"📊 Macro Average: {macro:.2f}%")

            eval_seconds = time.perf_counter() - ev_start

            # Plot & CSV via Plotter
            plot_start = time.perf_counter()
            plotter.add(ep, per_opponent)
            plotter.plot_all()
            _alias_joint_plot_names(plots_dir)
            plot_seconds = time.perf_counter() - plot_start

            # Speichern (ein Netz; der Name wird für p0..p3 dupliziert)
            save_start = time.perf_counter()
            tag = f"{ep:07d}"
            for p in range(num_players):
                base = os.path.join(weights_dir, f"k4a2_model_{version}_agent_p{p}_ep{tag}")
                agent.save(base)
            save_seconds = time.perf_counter() - save_start
            print(f"💾 Checkpoint gespeichert: Version {version}, Episode {ep}")

            # Konsolen-Timing (kompakt, wie k1a1)
            cum_seconds = time.perf_counter() - t0
            print(f"⏱ Timing @ep {ep}: episode {time.perf_counter()-ep_start:0.3f}s | "
                  f"train {train_seconds_accum:0.3f}s | eval {eval_seconds:0.3f}s | "
                  f"plot {plot_seconds:0.3f}s | save {save_seconds:0.3f}s | "
                  f"cum {cum_seconds/3600:0.2f}h")

        # ---------- Optionaler Zwischenspeicher ohne Eval ----------
        if ep % SAVE_INT == 0 and (ep % EINT != 0):
            save_start = time.perf_counter()
            tag = f"{ep:07d}"
            for p in range(num_players):
                base = os.path.join(weights_dir, f"k4a2_model_{version}_agent_p{p}_ep{tag}")
                agent.save(base)
            save_seconds = time.perf_counter() - save_start
            print(f"💾 Checkpoint gespeichert (ohne Eval): Version {version}, Episode {ep}")

        # Episode-Timing abschließen & loggen (k1a1-Format)
        ep_seconds = time.perf_counter() - ep_start
        cum_seconds = time.perf_counter() - t0
        timing_rows.append({
            "episode": ep,
            "steps": steps,
            "ep_seconds": ep_seconds,
            "train_seconds": train_seconds_accum,
            "eval_seconds": eval_seconds,
            "plot_seconds": plot_seconds,
            "save_seconds": save_seconds,
            "cum_seconds": cum_seconds,
        })

        # timings.csv bei jeder Eval und zusätzlich alle 1000 Episoden schreiben
        if ep % EINT == 0 or ep % 1000 == 0:
            pd.DataFrame(timing_rows).to_csv(timings_csv, index=False)
            eps_per_sec = ep / max(cum_seconds, 1e-9)
            print(f"🚀 Fortschritt: {ep}/{CONFIG['EPISODES']} Episoden | "
                  f"Durchsatz ~ {eps_per_sec:0.2f} eps/s")

    # Ende: finale CSV schreiben & Zusammenfassung
    total_seconds = time.perf_counter() - t0
    pd.DataFrame(timing_rows).to_csv(timings_csv, index=False)
    print(f"⏲️  Gesamtzeit: {total_seconds/3600:0.2f}h "
          f"(~ {CONFIG['EPISODES']/max(total_seconds,1e-9):0.2f} eps/s)")
    print("✅ K4 DQN Training abgeschlossen.")

if __name__=="__main__": main()
