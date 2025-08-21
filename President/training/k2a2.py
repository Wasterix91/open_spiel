# -*- coding: utf-8 -*-
# President/training/k2a2.py — DQN (K2): 4 simultan lernende Agents
# Neues Speicher-Layout:
#   models/k2a2/model_XX/
#     ├─ config.csv
#     ├─ timings.csv           (per-Episode-Timings)
#     ├─ plots/
#     │   ├─ lernkurve_single_only.png
#     │   ├─ lernkurve_max_combo.png
#     │   ├─ lernkurve_random2.png
#     │   ├─ lernkurve_alle.png
#     │   ├─ lernkurve_alle_mit_macro.png
#     │   └─ eval_curves.csv
#     └─ models/
#         └─ k2a2_model_XX_agent_p{i}_ep{NNNNNNN}_*.pt

import os, re, time, datetime, numpy as np, pandas as pd, torch
import pyspiel
from open_spiel.python import rl_environment
from agents import dqn_agent as dqn
from utils.fit_tensor import FeatureConfig, augment_observation
from utils.strategies import STRATS
from utils.plotter import EvalPlotter

# ============== CONFIG (wie k1a1) ==============
CONFIG = {
    "EPISODES":        200_000,
    "EVAL_INTERVAL":   10_000,     # auch Save-Intervall
    "EVAL_EPISODES":   2_000,
    "DECK_SIZE":       "16",       # "32" | "52" | "64"
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

    # Eval-Kurven (identisch zu k1a1/k2a1)
    "EVAL_CURVES": ["single_only", "max_combo", "random2"],
}

# ============== Helpers ==============
def find_next_version(base_dir, prefix="model"):
    """Scannt base_dir nach 'prefix_XX' und gibt das nächste XX zurück."""
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
    """Sorge für konsistente gemeinsame Plot-Namen (wie bei k1a1/k2a1)."""
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

# =========================== Training ===========================
def main():
    np.random.seed(CONFIG["SEED"]); torch.manual_seed(CONFIG["SEED"])
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_ROOT = os.path.join(ROOT, "models")

    # ---- Run-Verzeichnis anlegen ----
    family_dir  = os.path.join(MODELS_ROOT, "k2a2")
    version     = find_next_version(family_dir, prefix="model")
    run_dir     = os.path.join(family_dir, f"model_{version}")
    plots_dir   = os.path.join(run_dir, "plots")
    weights_dir = os.path.join(run_dir, "models")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    print(f"📁 Neuer Lauf (k2a2): {run_dir}")

    # ---- Env ----
    game = pyspiel.load_game("president", {
        "num_players":4, "deck_size":CONFIG["DECK_SIZE"], "shuffle_cards":True, "single_card_mode":False,
    })
    env = rl_environment.Environment(game)
    A = env.action_spec()["num_actions"]

    deck_int    = int(CONFIG["DECK_SIZE"])  # 32/52/64
    num_players = game.num_players()
    num_ranks   = 8 if deck_int in (32,64) else 13 if deck_int==52 else (_ for _ in ()).throw(ValueError("deck"))

    base_dim  = game.observation_tensor_shape()[0]
    extra_dim = num_players if CONFIG["FEATURES"]["SEAT_ONEHOT"] else 0
    state_size = base_dim + extra_dim

    feat_cfg = FeatureConfig(
        num_players=num_players, num_ranks=num_ranks,
        add_seat_onehot=bool(CONFIG["FEATURES"]["SEAT_ONEHOT"]),
        normalize=bool(CONFIG["FEATURES"]["NORMALIZE"])
    )

    # ---- Plotter ----
    plotter = EvalPlotter(
        opponent_names=list(CONFIG["EVAL_CURVES"]),
        out_dir=plots_dir,
        filename_prefix="lernkurve",
        csv_filename="eval_curves.csv",
        save_csv=True,
    )

    # ---- Agents & Reward ----
    dqn_cfg = dqn.DQNConfig(**CONFIG["DQN"])
    agents = [dqn.DQNAgent(state_size=state_size, num_actions=A, config=dqn_cfg) for _ in range(num_players)]
    shaper = RewardShaper(CONFIG["REWARD"])

    # ---- config.csv ----
    pd.DataFrame([{
        "script":"k2a2", "version":version,
        "timestamp":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent_type":"DQN-K2 (4x)", "num_episodes":CONFIG["EPISODES"],
        "eval_interval":CONFIG["EVAL_INTERVAL"], "eval_episodes":CONFIG["EVAL_EPISODES"],
        "deck_size":CONFIG["DECK_SIZE"],
        "observation_dim":state_size, "num_actions":A,
        "normalize":CONFIG["FEATURES"]["NORMALIZE"], "seat_onehot":CONFIG["FEATURES"]["SEAT_ONEHOT"],
        "models_dir":weights_dir, "plots_dir":plots_dir
    }]).to_csv(os.path.join(run_dir, "config.csv"), index=False)
    print(f"📝 Konfiguration gespeichert: {os.path.join(run_dir, 'config.csv')}")

    # ---- Timings (per Episode) ----
    timings_path = os.path.join(run_dir, "timings.csv")
    timings_cols = [
        "episode","wall_s","steps",
        "p0_transitions","p1_transitions","p2_transitions","p3_transitions",
        "p0_train_calls","p1_train_calls","p2_train_calls","p3_train_calls",
        "eps_p0","eps_p1","eps_p2","eps_p3"
    ]
    timings_buffer = []
    run_start = time.perf_counter()

    EINT, EEPS = CONFIG["EVAL_INTERVAL"], CONFIG["EVAL_EPISODES"]
    SAVE_INT = EINT  # speichern im Eval-Rhythmus

    # ================= Loop =================
    for ep in range(1, CONFIG["EPISODES"]+1):
        ep_t0 = time.perf_counter()
        ts = env.reset()
        # Tracke letzten Buffer-Index (für Terminal-Env-Reward) und Zähler
        last_idx = {p: None for p in range(num_players)}
        steps_in_ep = 0
        trans = [0,0,0,0]
        trains = [0,0,0,0]

        while not ts.last():
            p = ts.observations["current_player"]
            legal = ts.observations["legal_actions"][p]

            base_s = np.array(env._state.observation_tensor(p), dtype=np.float32)
            s = augment_observation(base_s, player_id=p, cfg=feat_cfg)
            a = int(agents[p].select_action(s, legal))

            ts_next = env.step([a])

            # Next-Obs & next_legals IMMER setzen (auch im Terminal) → stabiler Replay-Stack
            if not ts_next.last():
                base_s_next = np.array(env._state.observation_tensor(p), dtype=np.float32)
                s_next = augment_observation(base_s_next, player_id=p, cfg=feat_cfg)
                next_legals = ts_next.observations["legal_actions"][p]
            else:
                s_next = s
                next_legals = list(range(A))  # Fallback-Maske

            hand_before = shaper.hand_size(ts, p, deck_int)
            hand_after  = shaper.hand_size(ts_next, p, deck_int)
            r = shaper.step_reward(hand_before=hand_before, hand_after=hand_after,
                                   time_step=ts_next, player_id=p, deck_size=deck_int)

            ag = agents[p]
            ag.buffer.add(s, a, float(r), s_next, bool(ts_next.last()), next_legal_actions=next_legals)
            last_idx[p] = len(ag.buffer.buffer) - 1
            ag.train_step()

            trans[p]  += 1
            trains[p] += 1
            steps_in_ep += 1
            ts = ts_next

        # Terminal: optional Env-Reward pro Agent auf letzte Transition addieren
        if shaper.include_env_reward():
            returns = env._state.returns()
            for p in range(num_players):
                li = last_idx[p]
                if li is None: 
                    continue
                buf = agents[p].buffer
                old = buf.buffer[li]
                buf.buffer[li] = buf.Experience(
                    old.state, old.action, float(old.reward + returns[p]),
                    old.next_state, old.done, old.next_legal_mask
                )

        # ---- Episode-Timing erfassen & in CSV puffern ----
        ep_wall = time.perf_counter() - ep_t0
        timings_buffer.append({
            "episode": ep,
            "wall_s": round(ep_wall, 6),
            "steps": steps_in_ep,
            "p0_transitions": trans[0], "p1_transitions": trans[1], "p2_transitions": trans[2], "p3_transitions": trans[3],
            "p0_train_calls": trains[0], "p1_train_calls": trains[1], "p2_train_calls": trains[2], "p3_train_calls": trains[3],
            "eps_p0": getattr(agents[0], "epsilon", np.nan),
            "eps_p1": getattr(agents[1], "epsilon", np.nan),
            "eps_p2": getattr(agents[2], "epsilon", np.nan),
            "eps_p3": getattr(agents[3], "epsilon", np.nan),
        })
        if (ep % 100 == 0) or (ep == CONFIG["EPISODES"]):
            df = pd.DataFrame(timings_buffer, columns=timings_cols)
            write_header = not os.path.exists(timings_path)
            df.to_csv(timings_path, mode="a", index=False, header=write_header)
            timings_buffer.clear()

        # Optional: gelegentliche Fortschrittsmeldung zur Laufzeit
        if ep % 1000 == 0:
            print(f"⏱️  Episode {ep}: {ep_wall:.2f}s, steps={steps_in_ep}")

        # ---- Evaluation + Speichern ----
        if ep % EINT == 0:
            t_eval = time.perf_counter()

            per_opponent = {}
            # Für Eval: Agent P0 vs Heuristiken (epsilon=0)
            old_eps = agents[0].epsilon
            agents[0].epsilon = 0.0

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
                            a = int(agents[0].select_action(ob, legal))
                        else:
                            a = int(opp_fn(st))
                        st.apply_action(a)
                    if st.returns()[0] == max(st.returns()):
                        wins += 1
                wr = 100.0 * wins / EEPS
                per_opponent[opp_name] = wr
                print(f"✅ Eval nach {ep:7d} – Winrate (P0) vs {opp_name:11s}: {wr:5.1f}%")

            agents[0].epsilon = old_eps
            macro = float(np.mean(list(per_opponent.values())))
            print(f"📊 Macro Average (P0): {macro:.2f}%")

            # Plot & CSV updaten
            plotter.add(ep, per_opponent)
            plotter.plot_all()
            _alias_joint_plot_names(plots_dir)

            eval_wall = time.perf_counter() - t_eval
            print(f"⏱️  Eval-Dauer: {eval_wall:.2f}s")

            # Speichern aller vier Agents
            t_save = time.perf_counter()
            for i in range(num_players):
                base = os.path.join(weights_dir, f"k2a2_model_{version}_agent_p{i}_ep{ep:07d}")
                agents[i].save(base)
            print(f"💾 Modelle gespeichert (Episode {ep}) → {weights_dir}")
            print(f"⏱️  Save-Dauer: {time.perf_counter()-t_save:.2f}s")

    total_wall = time.perf_counter() - run_start
    print(f"⏱️  Gesamtlaufzeit: {total_wall/3600:.2f} h ({total_wall:.0f} s)")
    print("✅ K2 DQN Training fertig.")

if __name__=="__main__": main()
