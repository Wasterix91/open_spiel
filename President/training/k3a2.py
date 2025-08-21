# -*- coding: utf-8 -*-
# President/training/k3a2.py — DQN (K3): Snapshot-Selfplay-Pool (Seat 0 lernt)
# Speicherlayout (wie k1a1/k3a1):
#   models/k3a2/model_XX/
#     ├─ config.csv
#     ├─ timings.csv
#     ├─ plots/
#     │   ├─ lernkurve_single_only.png
#     │   ├─ lernkurve_max_combo.png
#     │   ├─ lernkurve_random2.png
#     │   ├─ lernkurve_alle.png
#     │   ├─ lernkurve_alle_mit_macro.png
#     │   └─ lernkurve_league.png          (zusätzliche League-Eval)
#     └─ models/
#         └─ k3a2_model_XX_agent_p0_ep0001000_*.pt

import os, re, time, copy, datetime, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
import pyspiel
from open_spiel.python import rl_environment
from agents import dqn_agent as dqn
from utils.fit_tensor import FeatureConfig, augment_observation
from utils.strategies import STRATS
from utils.plotter import EvalPlotter

# ============== CONFIG (wie k1a1/k3a1) ==============
CONFIG = {
    "EPISODES":        200_000,
    "EVAL_INTERVAL":   10_000,
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

    # Snapshot-Selfplay (wie k3a1)
    "SNAPSHOT": {
        "MIX_CURRENT": 0.8,           # Anteil 'current' vs Snapshot bei Seats 1–3
        "SNAPSHOT_INTERVAL": 10_000,  # wie oft aktuelle Policy in den Pool
        "POOL_CAP": 20,               # FIFO-Größe
    },

    # Eval-Kurven (Heuristiken)
    "EVAL_CURVES": ["single_only", "max_combo", "random2"],

    # Optional: zusätzlich League-Eval (MAIN vs zufällige Snapshots, falls vorhanden)
    "EVAL_LEAGUE": True,
}

# ============== Helpers ==============
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
        self.b = (float(cfg.get("BONUS_WIN",0.0)), float(cfg.get("BONUS_2ND",0.0)),
                  float(cfg.get("BONUS_3RD",0.0)), float(cfg.get("BONUS_LAST",0.0)))
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
    mapping = {
        "lernkurve_alle_strategien.png": "lernkurve_alle.png",
        "lernkurve_alle_strategien_avg.png": "lernkurve_alle_mit_macro.png",
    }
    for src, dst in mapping.items():
        src_path = os.path.join(plots_dir, src)
        dst_path = os.path.join(plots_dir, dst)
        if os.path.exists(src_path) and not os.path.exists(dst_path):
            try: os.replace(src_path, dst_path)
            except Exception: pass

# --- Snapshot-Policy für DQN (frozen) ---
class SnapshotDQNA:
    """Eingefrorene DQN-Policy: epsilon=0, nur Vorwärtspfad."""
    def __init__(self, state_size, num_actions, dqn_cfg, q_state_dict):
        self.agent = dqn.DQNAgent(state_size=state_size, num_actions=num_actions, config=dqn_cfg)
        # Q-Netz auf Snapshot setzen; Exploration aus
        self.agent.q_network.load_state_dict(copy.deepcopy(q_state_dict))
        self.agent.epsilon = 0.0
    @torch.no_grad()
    def act(self, obs_vec: np.ndarray, legal_actions):
        return int(self.agent.select_action(obs_vec, legal_actions))

# =========================== Training ===========================
def main():
    np.random.seed(CONFIG["SEED"]); torch.manual_seed(CONFIG["SEED"])
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_ROOT = os.path.join(ROOT, "models")

    # ---- Verzeichnisstruktur wie k1a1 ----
    family_dir = os.path.join(MODELS_ROOT, "k3a2")
    version = find_next_version(family_dir, prefix="model")
    run_dir = os.path.join(family_dir, f"model_{version}")
    plots_dir = os.path.join(run_dir, "plots")
    weights_dir = os.path.join(run_dir, "models")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    print(f"📁 Neuer Lauf (k3a2): {run_dir}")

    # ---- Env ----
    game = pyspiel.load_game("president", {
        "num_players": 4,
        "deck_size":   CONFIG["DECK_SIZE"],
        "shuffle_cards": True,
        "single_card_mode": False,
    })
    env = rl_environment.Environment(game)
    A = env.action_spec()["num_actions"]

    deck_int = int(CONFIG["DECK_SIZE"])
    num_players = game.num_players()
    num_ranks   = 8 if deck_int in (32,64) else 13 if deck_int==52 else (_ for _ in ()).throw(ValueError("deck"))

    base_dim  = game.observation_tensor_shape()[0]
    extra_dim = num_players if CONFIG["FEATURES"]["SEAT_ONEHOT"] else 0
    state_size = base_dim + extra_dim

    feat_cfg = FeatureConfig(num_players=num_players, num_ranks=num_ranks,
                             add_seat_onehot=bool(CONFIG["FEATURES"]["SEAT_ONEHOT"]),
                             normalize=bool(CONFIG["FEATURES"]["NORMALIZE"]))

    # ---- Eval-Plotter (Heuristiken) ----
    plotter = EvalPlotter(
        opponent_names=list(CONFIG["EVAL_CURVES"]),
        out_dir=plots_dir,
        filename_prefix="lernkurve",
        csv_filename="eval_curves.csv",
        save_csv=True,
    )

    # ---- Agent/Reward ----
    dqn_cfg = dqn.DQNConfig(**CONFIG["DQN"])
    learner = dqn.DQNAgent(state_size=state_size, num_actions=A, config=dqn_cfg)
    shaper  = RewardShaper(CONFIG["REWARD"])

    # ---- Konfiguration speichern ----
    pd.DataFrame([{
        "script":"k3a2","version":version,
        "timestamp":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent_type":"DQN-K3-SNAPSHOT",
        "num_episodes":CONFIG["EPISODES"],
        "eval_interval":CONFIG["EVAL_INTERVAL"],
        "eval_episodes":CONFIG["EVAL_EPISODES"],
        "deck_size":CONFIG["DECK_SIZE"],
        "observation_dim":state_size,"num_actions":A,
        "normalize":CONFIG["FEATURES"]["NORMALIZE"], "seat_onehot":CONFIG["FEATURES"]["SEAT_ONEHOT"],
        "mix_current":CONFIG["SNAPSHOT"]["MIX_CURRENT"],
        "snapshot_interval":CONFIG["SNAPSHOT"]["SNAPSHOT_INTERVAL"],
        "pool_cap":CONFIG["SNAPSHOT"]["POOL_CAP"],
        "models_dir":weights_dir,"plots_dir":plots_dir
    }]).to_csv(os.path.join(run_dir, "config.csv"), index=False)
    print(f"📝 Konfiguration gespeichert: {os.path.join(run_dir, 'config.csv')}")

    # ---- Timing-Setup (wie k1a1) ----
    timings_csv = os.path.join(run_dir, "timings.csv")
    timing_rows = []
    t0 = time.perf_counter()

    # ---- Snapshot-Pool ----
    pool = []  # enthält state_dicts der q_networks
    MIX      = float(CONFIG["SNAPSHOT"]["MIX_CURRENT"])
    SNAPINT  = int(CONFIG["SNAPSHOT"]["SNAPSHOT_INTERVAL"])
    POOL_CAP = int(CONFIG["SNAPSHOT"]["POOL_CAP"])

    EINT, EEPS = CONFIG["EVAL_INTERVAL"], CONFIG["EVAL_EPISODES"]
    league_wrs = []  # optionale Zusatzkurve

    # ================= Loop =================
    for ep in range(1, CONFIG["EPISODES"]+1):
        ep_start = time.perf_counter()
        train_seconds_acc = 0.0
        steps = 0

        ts = env.reset()
        last_idx_p0 = None  # letzte Transition für Seat 0 (für Terminal-Boni)

        # Seats 1..3: aktuelle Policy oder Snapshot?
        use_current = {}
        snap_actor  = {}
        for seat in [1,2,3]:
            if len(pool) == 0:
                use_current[seat] = True
            else:
                use_current[seat] = (np.random.rand() < MIX)
                if not use_current[seat]:
                    state_dict = np.random.choice(pool)
                    snap_actor[seat] = SnapshotDQNA(
                        state_size=state_size, num_actions=A, dqn_cfg=dqn_cfg,
                        q_state_dict=state_dict
                    )

        # ---- Episode ----
        while not ts.last():
            steps += 1
            p = ts.observations["current_player"]
            legal = ts.observations["legal_actions"][p]

            base_s = np.array(env._state.observation_tensor(p), dtype=np.float32)
            s = augment_observation(base_s, player_id=p, cfg=feat_cfg)

            if p == 0:
                a = int(learner.select_action(s, legal))
            else:
                if use_current[p]:
                    a = int(learner.select_action(s, legal))   # aktuelles Netz, nur Vorwärts
                else:
                    a = int(snap_actor[p].act(s, legal))        # eingefrorener Snapshot

            ts_next = env.step([a])

            # Next-Obs & next_legals (auch im Terminal robust)
            if not ts_next.last():
                base_s_next = np.array(env._state.observation_tensor(p), dtype=np.float32)
                s_next = augment_observation(base_s_next, player_id=p, cfg=feat_cfg)
                next_legals = ts_next.observations["legal_actions"][p]
            else:
                s_next = s
                next_legals = list(range(A))

            # Nur Seat 0 lernt
            if p == 0:
                hand_before = shaper.hand_size(ts, p, deck_int)
                hand_after  = shaper.hand_size(ts_next, p, deck_int)
                r = shaper.step_reward(hand_before=hand_before, hand_after=hand_after,
                                       time_step=ts_next, player_id=p, deck_size=deck_int)

                learner.buffer.add(s, a, float(r), s_next, bool(ts_next.last()),
                                   next_legal_actions=next_legals)
                last_idx_p0 = len(learner.buffer.buffer) - 1

                t_train = time.perf_counter()
                learner.train_step()
                train_seconds_acc += (time.perf_counter() - t_train)

            ts = ts_next

        # Terminal: Env-Reward + (optional) Platzierungsbonus für P0 addieren
        if last_idx_p0 is not None:
            bonus = 0.0
            if shaper.include_env_reward():
                bonus += env._state.returns()[0]
            bonus += shaper.final_bonus(env._state.returns(), 0)
            if abs(bonus) > 1e-8:
                buf = learner.buffer
                old = buf.buffer[last_idx_p0]
                buf.buffer[last_idx_p0] = buf.Experience(
                    old.state, old.action, float(old.reward + bonus), old.next_state, old.done, old.next_legal_mask
                )

        # ---- Snapshot aufnehmen ----
        if ep % SNAPINT == 0:
            pool.append(copy.deepcopy(learner.q_network.state_dict()))
            if len(pool) > POOL_CAP:
                pool.pop(0)

        # Defaults für optionale Felder
        eval_seconds = 0.0
        plot_seconds = 0.0
        save_seconds = 0.0

        # ---------- Evaluation (wie k1a1) ----------
        if ep % EINT == 0:
            ev_start = time.perf_counter()

            per_opponent = {}
            old_eps = learner.epsilon
            learner.epsilon = 0.0  # greedy Eval

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
                            a = int(learner.select_action(ob, legal))
                        else:
                            a = int(opp_fn(st))
                        st.apply_action(a)
                    if st.returns()[0] == max(st.returns()):
                        wins += 1
                wr = 100.0 * wins / EEPS
                per_opponent[opp_name] = wr
                print(f"✅ Eval nach {ep:7d} – Winrate vs {opp_name:11s}: {wr:5.1f}%")

            learner.epsilon = old_eps

            macro = float(np.mean(list(per_opponent.values())))
            print(f"📊 Macro Average: {macro:.2f}%")

            eval_seconds = time.perf_counter() - ev_start

            # Optional: League-Eval (MAIN vs 3 zufällige Snapshots, falls vorhanden)
            if CONFIG.get("EVAL_LEAGUE", False) and len(pool) >= 3:
                wins = 0
                old_eps = learner.epsilon; learner.epsilon = 0.0
                for _ in range(EEPS):
                    st = game.new_initial_state()
                    # gegnerische 3 Snapshots ziehen
                    snap_idxs = np.random.choice(len(pool), size=3, replace=False)
                    snap_opps = [SnapshotDQNA(state_size, A, dqn_cfg, pool[i]) for i in snap_idxs]

                    while not st.is_terminal():
                        pid = st.current_player(); legal = st.legal_actions(pid)
                        ob_base = np.array(st.observation_tensor(pid), dtype=np.float32)
                        ob = augment_observation(ob_base, player_id=pid, cfg=feat_cfg)
                        if pid == 0:
                            a = int(learner.select_action(ob, legal))
                        else:
                            a = int(snap_opps[pid-1].act(ob, legal))
                        st.apply_action(a)
                    if st.returns()[0] == max(st.returns()): wins += 1
                wr_league = 100.0 * wins / EEPS
                league_wrs.append(wr_league)
                print(f"🤝 League-Eval (Snapshots) – MAIN Winrate: {wr_league:.1f}%")

            # Plot & CSV aktualisieren
            plot_start = time.perf_counter()
            plotter.add(ep, per_opponent)
            plotter.plot_all()
            _alias_joint_plot_names(plots_dir)

            # League-Kurve plotten (falls vorhanden)
            if league_wrs:
                xs = list(range(EINT, EINT*len(league_wrs)+1, EINT))
                plt.figure(figsize=(10,6))
                plt.plot(xs, league_wrs, marker="o")
                plt.title("K3 – DQN Snapshot Selfplay (MAIN vs Snapshot-Pool)")
                plt.xlabel("Episode"); plt.ylabel("Winrate (%)"); plt.grid(True)
                plt.tight_layout()
                out = os.path.join(plots_dir, "lernkurve_league.png")
                plt.savefig(out); plt.close()
                # keine extra Console-Zeile nötig
            plot_seconds = time.perf_counter() - plot_start

            # Checkpoint speichern
            save_start = time.perf_counter()
            base = os.path.join(weights_dir, f"k3a2_model_{version}_agent_p0_ep{ep:07d}")
            learner.save(base)  # erwartet ..._q.pt / ..._target.pt (abhängig von dqn_agent.py)
            save_seconds = time.perf_counter() - save_start
            print(f"💾 Modell gespeichert: {base}_*")

            # Konsolen-Timing (kompakt, wie k1a1)
            print(f"⏱ Timing @ep {ep}: episode {time.perf_counter()-ep_start:0.3f}s | "
                  f"train {train_seconds_acc:0.3f}s | eval {eval_seconds:0.3f}s | "
                  f"plot {plot_seconds:0.3f}s | save {save_seconds:0.3f}s | "
                  f"cum {(time.perf_counter()-t0)/3600:0.2f}h")

        # Episode-Timing abschließen & loggen (wie k1a1)
        ep_seconds = time.perf_counter() - ep_start
        cum_seconds = time.perf_counter() - t0
        timing_rows.append({
            "episode": ep,
            "steps": steps,
            "ep_seconds": ep_seconds,
            "train_seconds": train_seconds_acc,
            "eval_seconds": eval_seconds,
            "plot_seconds": plot_seconds,
            "save_seconds": save_seconds,
            "cum_seconds": cum_seconds,
        })

        # CSV bei jeder Eval schreiben (wie k1a1)
        if ep % EINT == 0:
            pd.DataFrame(timing_rows).to_csv(timings_csv, index=False)
            eps_per_sec = ep / max(cum_seconds, 1e-9)
            print(f"🚀 Fortschritt: {ep}/{CONFIG['EPISODES']} Episoden | "
                  f"Durchsatz ~ {eps_per_sec:0.2f} eps/s")

    # Ende: finale CSV schreiben & Zusammenfassung
    total_seconds = time.perf_counter() - t0
    pd.DataFrame(timing_rows).to_csv(timings_csv, index=False)
    print(f"⏲️  Gesamtzeit: {total_seconds/3600:0.2f}h "
          f"(~ {CONFIG['EPISODES']/max(total_seconds,1e-9):0.2f} eps/s)")
    print("✅ K3 DQN Snapshot-Selfplay Training abgeschlossen.")

if __name__=="__main__": main()
