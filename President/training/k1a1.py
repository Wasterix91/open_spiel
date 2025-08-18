# -*- coding: utf-8 -*-
import os, re, datetime, time, numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
import pyspiel
from open_spiel.python import rl_environment
from agents import ppo_agent as ppo
from utils.strategies import STRATS
from utils.fit_tensor import FeatureConfig, augment_observation
from utils.training_eval_plots import EvalPlotter

# ============== CONFIG  ==============
CONFIG = {
    "EPISODES":        1_200_000,
    "EVAL_INTERVAL":   10_000,
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
        "DELTA_WEIGHT": 0.0,
        "HAND_PENALTY_COEFF": 0.0,
        "FINAL": "none",          # "none" | "placement_bonus"
        "BONUS_WIN": 0.0, "BONUS_2ND": 0.0, "BONUS_3RD": 0.0, "BONUS_LAST": 0.0,
        "ENV_REWARD": True,
    },

    # Feature-Toggles
    "FEATURES": {
        "NORMALIZE": False,
        "SEAT_ONEHOT": False,     # K1: typischerweise False
    },

    # Eval-Kurven
    "EVAL_CURVES": ["single_only", "max_combo", "random2"],  # + Macro wird automatisch geplottet
}

# ================= Helpers / Heuristiken =================
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
    Falls der Plotter alte Joint-Namen erzeugt, legen wir deine gewünschten Aliase an:
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
                os.replace(src_path, dst_path)  # rename/alias
            except Exception:
                pass  # non-fatal

# =========================== Training ===========================
def main():
    np.random.seed(CONFIG["SEED"]); torch.manual_seed(CONFIG["SEED"])
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_ROOT = os.path.join(ROOT, "models")

    # ---- NEW: run directory structure ----
    family_dir = os.path.join(MODELS_ROOT, "k1a1")
    version = find_next_version(family_dir, prefix="model")   # -> '01', '02', ...
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
    info_dim = env.observation_spec()["info_state"][0]
    A = env.action_spec()["num_actions"]

    # ---- Features ----
    deck_int = int(CONFIG["DECK_SIZE"])
    num_players = game.num_players()
    num_ranks   = 8 if deck_int in (32,64) else 13 if deck_int==52 else (_ for _ in ()).throw(ValueError("deck"))
    feat_cfg = FeatureConfig(
        num_players=num_players, num_ranks=num_ranks,
        add_seat_onehot=bool(CONFIG["FEATURES"]["SEAT_ONEHOT"]),
        normalize=bool(CONFIG["FEATURES"]["NORMALIZE"]),
    )

    # ---- Plotter -> plots_dir ----
    plotter = EvalPlotter(
        opponent_names=list(CONFIG["EVAL_CURVES"]),
        out_dir=plots_dir,
        filename_prefix="lernkurve",
        csv_filename="eval_curves.csv",   # in plots_dir
        save_csv=True,
    )

    # ---- Agent/opp/reward ----
    ppo_cfg = ppo.PPOConfig(**CONFIG["PPO"])
    seat_id_dim = (num_players if CONFIG["FEATURES"]["SEAT_ONEHOT"] else 0)
    agent = ppo.PPOAgent(info_dim, A, seat_id_dim=seat_id_dim, config=ppo_cfg)
    opponents = [STRATS[name] for name in CONFIG["OPPONENTS"]]
    shaper = RewardShaper(CONFIG["REWARD"])

    # ---- Save config.csv at run root ----
    pd.DataFrame([{
        "script":"k1a1","version":version,
        "timestamp":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent_type":"PPO","num_episodes":CONFIG["EPISODES"],
        "eval_interval":CONFIG["EVAL_INTERVAL"],"eval_episodes":CONFIG["EVAL_EPISODES"],
        "deck_size":CONFIG["DECK_SIZE"],
        "observation_dim":info_dim + seat_id_dim,"num_actions":A,
        "normalize":CONFIG["FEATURES"]["NORMALIZE"], "seat_onehot":CONFIG["FEATURES"]["SEAT_ONEHOT"],
        "opponents":",".join(CONFIG["OPPONENTS"]),
        "models_dir":weights_dir,"plots_dir":plots_dir
    }]).to_csv(os.path.join(run_dir, "config.csv"), index=False)
    print(f"📝 Konfiguration gespeichert: {os.path.join(run_dir, 'config.csv')}")

    # ---- Timing setup ----
    timings_csv = os.path.join(run_dir, "timings.csv")
    timing_rows = []
    t0 = time.perf_counter()

    EINT, EEPS = CONFIG["EVAL_INTERVAL"], CONFIG["EVAL_EPISODES"]

    # ---- Training loop ----
    for ep in range(1, CONFIG["EPISODES"]+1):
        ep_start = time.perf_counter()
        steps = 0

        ts = env.reset()
        while not ts.last():
            steps += 1
            p = ts.observations["current_player"]; legal = ts.observations["legal_actions"][p]
            hand_before = shaper.hand_size(ts, p, deck_int)

            if p==0:
                base_obs = ts.observations["info_state"][p]
                obs = augment_observation(base_obs, player_id=p, cfg=feat_cfg)
                seat_oh = None
                if CONFIG["FEATURES"]["SEAT_ONEHOT"]:
                    seat_oh = np.zeros(num_players, dtype=np.float32); seat_oh[p] = 1.0
                a = int(agent.step(obs, legal, seat_one_hot=seat_oh))
            else:
                a = int(opponents[p-1](env._state))

            ts_next = env.step([a])

            if p==0:
                hand_after = shaper.hand_size(ts_next, p, deck_int)
                r = shaper.step_reward(hand_before=hand_before, hand_after=hand_after, time_step=ts_next, player_id=p, deck_size=deck_int)
                agent.post_step(r, done=ts_next.last())

            ts = ts_next

        # train() separat timen
        train_start = time.perf_counter()
        if shaper.include_env_reward(): agent._buffer.finalize_last_reward(ts.rewards[0])
        agent._buffer.finalize_last_reward(shaper.final_bonus(ts.rewards, 0))
        agent.train()
        train_seconds = time.perf_counter() - train_start

        # Defaults für optionale Felder
        eval_seconds = 0.0
        plot_seconds = 0.0
        save_seconds = 0.0

        # ---- Evaluation ----
        if ep % EINT == 0:
            ev_start = time.perf_counter()

            per_opponent = {}
            for opp_name in CONFIG["EVAL_CURVES"]:
                opp_fn = STRATS[opp_name]
                wins = 0
                for _ in range(EEPS):
                    st = game.new_initial_state()
                    while not st.is_terminal():
                        pid = st.current_player()
                        legal = st.legal_actions(pid)
                        if pid == 0:
                            obs = st.information_state_tensor(pid)
                            obs = augment_observation(obs, player_id=pid, cfg=feat_cfg)
                            with torch.no_grad():
                                logits = agent._policy(torch.tensor(obs, dtype=torch.float32))
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
            print(f"📊 Macro Average: {macro:.2f}%")

            eval_seconds = time.perf_counter() - ev_start

            # Plot & CSV aktualisieren (Plotter)
            plot_start = time.perf_counter()
            plotter.add(ep, per_opponent)
            plotter.plot_all()
            _alias_joint_plot_names(plots_dir)
            plot_seconds = time.perf_counter() - plot_start

            # Save weights
            save_start = time.perf_counter()
            base = os.path.join(weights_dir, f"k1a1_model_{version}_agent_p0_ep{ep:07d}")
            agent.save(base)  # ..._policy.pt / ..._value.pt
            save_seconds = time.perf_counter() - save_start
            print(f"💾 Modell gespeichert: {base}_policy.pt / {base}_value.pt")

            # Konsolen-Timing (kompakt)
            cum_seconds = time.perf_counter() - t0
            print(f"⏱ Timing @ep {ep}: episode {time.perf_counter()-ep_start:0.3f}s | "
                  f"train {train_seconds:0.3f}s | eval {eval_seconds:0.3f}s | "
                  f"plot {plot_seconds:0.3f}s | save {save_seconds:0.3f}s | "
                  f"cum {cum_seconds/3600:0.2f}h")

        # Episode-Timing abschließen & loggen
        ep_seconds = time.perf_counter() - ep_start
        cum_seconds = time.perf_counter() - t0
        timing_rows.append({
            "episode": ep,
            "steps": steps,
            "ep_seconds": ep_seconds,
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds,
            "plot_seconds": plot_seconds,
            "save_seconds": save_seconds,
            "cum_seconds": cum_seconds,
        })

        # Schreibintervall: bei jeder Eval und zusätzlich alle 1000 Episoden
        if ep % EINT == 0:
            pd.DataFrame(timing_rows).to_csv(timings_csv, index=False)
            # Optional kleine Progress-Kennzahl:
            eps_per_sec = ep / max(cum_seconds, 1e-9)
            print(f"🚀 Fortschritt: {ep}/{CONFIG['EPISODES']} Episoden | "
                  f"Durchsatz ~ {eps_per_sec:0.2f} eps/s")

    # Ende: finale CSV schreiben & Zusammenfassung
    total_seconds = time.perf_counter() - t0
    pd.DataFrame(timing_rows).to_csv(timings_csv, index=False)
    print(f"⏲️  Gesamtzeit: {total_seconds/3600:0.2f}h "
          f"(~ {CONFIG['EPISODES']/max(total_seconds,1e-9):0.2f} eps/s)")
    print("✅ K1 Training abgeschlossen.")

if __name__=="__main__": main()
