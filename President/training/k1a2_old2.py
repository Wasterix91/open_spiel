# -*- coding: utf-8 -*-
# President/training/k1a2.py — DQN (K1): Single-Agent vs Heuristiken
# Neue Speicherstruktur:
#   models/k1a2/model_XX/
#     config.csv
#     timings.csv
#     plots/
#       lernkurve_single_only.png
#       lernkurve_max_combo.png
#       lernkurve_random2.png
#       lernkurve_alle.png
#       lernkurve_alle_mit_macro.png
#       eval_curves.csv
#     models/
#       k1a2_model_XX_agent_p0_ep0001000_*.pt
#
# WICHTIGER FIX:
#  - DQN-Transitions werden zwischen zwei aufeinanderfolgenden P0-Entscheidungen gebildet:
#    (s_P0, a_P0, r_step, s'_P0, done), wobei s'_P0 = Beobachtung, wenn P0 wieder am Zug ist
#    (oder Terminal). Gegnerzüge werden dazwischen "vorgespult", ohne neue P0-Transitions
#    zu erzeugen. Dadurch stimmen next_legal_actions & Bootstrapping-Zustände.

import os, re, datetime, time
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
import pyspiel
from open_spiel.python import rl_environment
from agents import dqn_agent_old_new as dqn
from utils.fit_tensor import FeatureConfig, augment_observation
from utils.strategies import STRATS
from utils.plotter import EvalPlotter

# ============== CONFIG  ==============
CONFIG = {
    "EPISODES":        200_000,
    "EVAL_INTERVAL":   5000,
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
        "BONUS_WIN": 10.0, "BONUS_2ND": 3.0, "BONUS_3RD": 2.0, "BONUS_LAST": 0.0,
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
def find_next_version(base_dir, prefix="model"):
    """Scans base_dir for 'prefix_XX' and returns next two-digit XX."""
    pat = re.compile(rf"^{re.escape(prefix)}_(\d{{2}})$")
    os.makedirs(base_dir, exist_ok=True)
    existing = [int(m.group(1)) for m in (pat.match(n) for n in os.listdir(base_dir)) if m]
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

def _alias_joint_plot_names(plots_dir):
    """
    Wenn der Plotter die alten gemeinsamen Namen schreibt, lege zusätzlich deine gewünschten Aliasse an:
      lernkurve_alle_strategien.png       -> lernkurve_alle.png
      lernkurve_alle_strategien_avg.png   -> lernkurve_alle_mit_macro.png
    """
    mapping = {
        "lernkurve_alle_strategien.png": "lernkurve_alle.png",
        "lernkurve_alle_strategien_avg.png": "lernkurve_alle_mit_macro.png",
    }
    for src, dst in mapping.items():
        src_path = os.path.join(plots_dir, src)
        dst_path = os.path.join(plots_dir, dst)
        if os.path.exists(src_path) and not os.path.exists(dst_path):
            try:
                os.replace(src_path, dst_path)
            except Exception:
                pass  # non-fatal

# =========================== Training ===========================
def main():
    np.random.seed(CONFIG["SEED"]); torch.manual_seed(CONFIG["SEED"])
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_ROOT = os.path.join(ROOT, "models")

    # ---- Neue Verzeichnisstruktur ----
    family_dir = os.path.join(MODELS_ROOT, "k1a2")
    version = find_next_version(family_dir, prefix="model")  # -> '01', '02', ...
    run_dir = os.path.join(family_dir, f"model_{version}")
    plots_dir = os.path.join(run_dir, "plots")
    weights_dir = os.path.join(run_dir, "models")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    print(f"📁 Neuer Lauf: {run_dir}")

    # ---- Game/Env ----
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

    # ---- Plotter in neues plots_dir ----
    plotter = EvalPlotter(
        opponent_names=list(CONFIG["EVAL_CURVES"]),
        out_dir=plots_dir,
        filename_prefix="lernkurve",
        csv_filename="eval_curves.csv",
        save_csv=True,
    )

    # ---- Agent/Reward/Gegner ----
    dqn_cfg = dqn.DQNConfig(**CONFIG["DQN"])
    agent = dqn.DQNAgent(state_size=state_size, num_actions=A, config=dqn_cfg)
    opponents = [STRATS[name] for name in CONFIG["OPPONENTS"]]
    shaper = RewardShaper(CONFIG["REWARD"])

    # ---- config.csv im Run-Root speichern ----
    pd.DataFrame([{
        "script":"k1a2","version":version,
        "timestamp":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent_type":"DQN","num_episodes":CONFIG["EPISODES"],
        "eval_interval":CONFIG["EVAL_INTERVAL"],"eval_episodes":CONFIG["EVAL_EPISODES"],
        "deck_size":CONFIG["DECK_SIZE"],
        "observation_dim":state_size,"num_actions":A,
        "normalize":CONFIG["FEATURES"]["NORMALIZE"],"seat_onehot":CONFIG["FEATURES"]["SEAT_ONEHOT"],
        "opponents":",".join(CONFIG["OPPONENTS"]),
        "models_dir":weights_dir,"plots_dir":plots_dir
    }]).to_csv(os.path.join(run_dir, "config.csv"), index=False)
    print(f"📝 Konfiguration gespeichert: {os.path.join(run_dir, 'config.csv')}")

    # ---- Timings-Setup ----
    timings_path = os.path.join(run_dir, "timings.csv")
    timings_cols = ["episode","wall_s","steps","p0_transitions","train_calls","epsilon"]
    timings_buffer = []
    run_start = time.perf_counter()

    EINT, EEPS = CONFIG["EVAL_INTERVAL"], CONFIG["EVAL_EPISODES"]

    # ---- Training loop (FIXED: P0->P0 Transitions) ----
    for ep in range(1, CONFIG["EPISODES"]+1):
        ep_t0 = time.perf_counter()

        ts = env.reset()
        last_idx_p0 = None
        steps_in_ep = 0
        p0_transitions = 0
        train_calls = 0

        while not ts.last():

            # 1) Vorspulen, bis P0 am Zug ist (Gegner spielen Heuristiken)
            while not ts.last() and ts.observations["current_player"] != 0:
                pid = ts.observations["current_player"]
                a_opp = int(opponents[pid-1](env._state))
                ts = env.step([a_opp])
                steps_in_ep += 1

            if ts.last():
                break  # Episode vorbei, kein P0-Zug mehr

            # 2) P0-Zug: s, a, r_step (Reward nur für diesen P0-Schritt)
            legal0 = ts.observations["legal_actions"][0]
            s_base = np.array(env._state.observation_tensor(0), dtype=np.float32)
            s      = augment_observation(s_base, player_id=0, cfg=feat_cfg)

            hand_before = shaper.hand_size(ts, 0, deck_int)
            a0 = int(agent.select_action(s, legal0))

            ts = env.step([a0])   # führt P0-Zug aus
            steps_in_ep += 1

            hand_after = shaper.hand_size(ts, 0, deck_int)
            r_step = shaper.step_reward(
                hand_before=hand_before, hand_after=hand_after,
                time_step=ts, player_id=0, deck_size=deck_int
            )

            # 3) Vorspulen bis P0 wieder am Zug ist oder Terminal
            while not ts.last() and ts.observations["current_player"] != 0:
                pid = ts.observations["current_player"]
                a_opp = int(opponents[pid-1](env._state))
                ts = env.step([a_opp])
                steps_in_ep += 1

            # 4) s' und next_legals für P0 definieren
            if ts.last():
                s_next = s
                next_legals = list(range(A))  # nie None in Buffer
                done = True
            else:
                next_legals = ts.observations["legal_actions"][0]
                s_next_base = np.array(env._state.observation_tensor(0), dtype=np.float32)
                s_next = augment_observation(s_next_base, player_id=0, cfg=feat_cfg)
                done = False

            # 5) Transition speichern & lernen
            agent.buffer.add(s, a0, float(r_step), s_next, done, next_legal_actions=next_legals)


            last_idx_p0 = len(agent.buffer.buffer) - 1
            agent.train_step()
            p0_transitions += 1
            train_calls += 1

        # Terminal: optional Env-Reward + Finalbonus auf letzte P0-Transition addieren
        if last_idx_p0 is not None:
            bonus = 0.0
            if shaper.include_env_reward():
                bonus += ts.rewards[0]
            bonus += shaper.final_bonus(ts.rewards, 0)
            if abs(bonus) > 1e-8:
                buf = agent.buffer
                old = buf.buffer[last_idx_p0]
                new = buf.Experience(old.state, old.action, float(old.reward + bonus),
                                    old.next_state, old.done, old.next_legal_mask)
                buf.buffer[last_idx_p0] = new


        # ---- Episode-Timing erfassen ----
        ep_wall = time.perf_counter() - ep_t0
        timings_buffer.append({
            "episode": ep,
            "wall_s": round(ep_wall, 6),
            "steps": steps_in_ep,
            "p0_transitions": p0_transitions,
            "train_calls": train_calls,
            "epsilon": getattr(agent, "epsilon", np.nan),
        })
        # Puffer regelmäßig schreiben (alle 100 Episoden + beim Ende)
        if (ep % 100 == 0) or (ep == CONFIG["EPISODES"]):
            df = pd.DataFrame(timings_buffer, columns=timings_cols)
            write_header = not os.path.exists(timings_path)
            df.to_csv(timings_path, mode="a", index=False, header=write_header)
            timings_buffer.clear()


        # ---- Evaluation (per Gegner + Macro + Plotter) ----
        if ep % EINT == 0:
            old_eps = agent.epsilon
            agent.epsilon = 0.0  # greedy Eval

            per_opponent_wr = {}  # nur Winrates für den (alten) EvalPlotter
            for opp_name in CONFIG["EVAL_CURVES"]:
                opp_fn = STRATS[opp_name]
                wins = 0
                place_counts = [0, 0, 0, 0]  # 1st..4th
                reward_sum = 0.0

                for _ in range(EEPS):
                    st = game.new_initial_state()
                    while not st.is_terminal():
                        pid = st.current_player()
                        legal = st.legal_actions(pid)
                        if pid == 0:
                            obs_base = np.array(st.observation_tensor(pid), dtype=np.float32)
                            obs = augment_observation(obs_base, player_id=pid, cfg=feat_cfg)
                            a = int(agent.select_action(obs, legal))
                        else:
                            a = int(opp_fn(st))
                        st.apply_action(a)

                    rets = st.returns()
                    reward_sum += float(rets[0])

                    # Platzierung aus Returns ableiten (höher = besser).
                    order = sorted(range(len(rets)), key=lambda i: rets[i], reverse=True)
                    place_idx = order.index(0)  # 0 => 1st, 1 => 2nd, ...
                    place_counts[place_idx] += 1
                    if place_idx == 0:
                        wins += 1

                wr = 100.0 * wins / EEPS
                per_opponent_wr[opp_name] = wr

                # Optional: hilfreiche Konsole-Ausgabe (zeigt auch Platzverteilung)
                p1, p2, p3, p4 = (c / EEPS for c in place_counts)
                print(f"✅ Eval nach {ep:7d} – Winrate vs {opp_name:11s}: {wr:5.1f}%")
                print(f"   Plätze: [1st {p1:.2f}, 2nd {p2:.2f}, 3rd {p3:.2f}, 4th {p4:.2f}]  Ø-Reward: {reward_sum/EEPS:.3f}")

            agent.epsilon = old_eps

            macro = float(np.mean(list(per_opponent_wr.values())))
            print(f"📊 Macro Average: {macro:.2f}%")

            # Plotter (alter EvalPlotter erwartet name->float)
            plotter.add(ep, per_opponent_wr)
            plotter.plot_all()
            _alias_joint_plot_names(plots_dir)

            # Checkpoint
            base_ep = os.path.join(weights_dir, f"k1a2_model_{version}_agent_p0_ep{ep:07d}")
            agent.save(base_ep)
            print(f"💾 Modell gespeichert: {base_ep}_*")


    total_wall = time.perf_counter() - run_start
    print(f"⏱️  Gesamtlaufzeit: {total_wall/3600:.2f} h ({total_wall:.0f} s)")
    print("✅ K1 DQN Training abgeschlossen.")

if __name__=="__main__": main()
