# -*- coding: utf-8 -*-
# President/training/k3a1.py — PPO (K3): Snapshot-Selfplay, 1 Learner + 3 Sparring
# Hinweis: minimale Änderungen gegenüber k3a1_test.py
#  - family-Name geändert (eigener Output-Ordner)
#  - Konfig getuned, damit Snapshots bei kurzen Läufen sichtbar sind (kürzeres SNAPSHOT_INTERVAL)

import os, datetime, time, copy, numpy as np, torch
import pyspiel
from open_spiel.python import rl_environment

from agents import ppo_agent as ppo
from utils.strategies import STRATS
from utils.fit_tensor import FeatureConfig, augment_observation
from utils.plotter import MetricsPlotter
from utils.timing import TimingMeter
from utils.reward_shaper import RewardShaper

from utils.load_save_common import find_next_version, prepare_run_dirs, save_config_csv, save_run_meta
from utils.load_save_a1_ppo import save_checkpoint_ppo

from utils.benchmark import run_benchmark
from utils.deck import ranks_for_deck

# ============== CONFIG ==============
CONFIG = {
    "EPISODES":         2000,
    "BENCH_INTERVAL":   500,
    "BENCH_EPISODES":   500,
    "TIMING_INTERVAL":  500,
    "DECK_SIZE":        "16",  # "12" | "16" | "20" | "24" | "32" | "52" | "64"
    "SEED":             42,


    # PPO
    "PPO": {
        "learning_rate": 3e-4,
        "num_epochs": 1,
        "batch_size": 256,
        "entropy_cost": 0.01,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "value_coef": 0.5,
        "max_grad_norm": 0.5,
    },

    # ======= Rewards (NEUES System) =======
    # STEP_MODE : "none" | "delta_weight_only" | "hand_penalty_coeff_only" | "combined"
    # FINAL_MODE: "none" | "env_only" | "rank_bonus" | "both"
    "REWARD": {
        "STEP_MODE": "none",
        "DELTA_WEIGHT": 0.0,
        "HAND_PENALTY_COEFF": 0.0,

        "FINAL_MODE": "env_only",
        "BONUS_WIN": 0.0, "BONUS_2ND": 0.0, "BONUS_3RD": 0.0, "BONUS_LAST": 0.0,
    },

    # Features
    "FEATURES": { "NORMALIZE": False, "SEAT_ONEHOT": True },

    # Snapshot-Selfplay
    "SNAPSHOT": {
        "LEARNER_SEAT": 0,
        "MIX_CURRENT": 0.8,            # Wahrscheinlichkeit, Seats 1–3 mit aktueller Policy spielen zu lassen
        "SNAPSHOT_INTERVAL": 200,      # kürzer als in *_test, damit bei 5k Eps viele Snapshots anfallen
        "POOL_CAP": 20,
    },

    # Benchmark-Gegner
    "BENCH_OPPONENTS": ["single_only", "max_combo", "random2"],
}

# ===== Helfer für Snapshot-Policies =====
class SnapshotPolicy(torch.nn.Module):
    """Eingefrorene Policy (nur Inferenz). Erwartet denselben Input wie PPOAgent._make_input."""
    def __init__(self, input_dim: int, num_actions: int, state_dict: dict):
        super().__init__()
        self.net = ppo.PolicyNetwork(input_dim, num_actions)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self.num_actions = num_actions

    @torch.no_grad()
    def act(self, x_vec: np.ndarray, legal_actions):
        device = next(self.net.parameters()).device
        x_t = torch.tensor(x_vec, dtype=torch.float32, device=device).unsqueeze(0)
        logits = self.net(x_t).squeeze(0)
        mask = torch.zeros(self.num_actions, dtype=torch.float32, device=device)
        mask[legal_actions] = 1.0
        probs = ppo.masked_softmax(logits, mask)
        return int(torch.distributions.Categorical(probs=probs).sample().item())

def main():
    np.random.seed(CONFIG["SEED"]); torch.manual_seed(CONFIG["SEED"])
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_ROOT = os.path.join(ROOT, "models")

    # ---- Verzeichnisse (k1a1-Stil) ----
    family = "k3a1"  # eigene Familie für recommended Variante
    family_dir = os.path.join(MODELS_ROOT, family)
    version = find_next_version(family_dir, prefix="model")
    paths = prepare_run_dirs(MODELS_ROOT, family, version, prefix="model")

    # ---- Plotter / Logger ----
    plotter = MetricsPlotter(
        out_dir=paths["plots_dir"],
        benchmark_opponents=list(CONFIG["BENCH_OPPONENTS"]),
        benchmark_csv="benchmark_curves.csv",
        train_csv="training_metrics.csv",
        save_csv=True, verbosity=1,
    )
    plotter.log("New Training (k3a1): Snapshot-Selfplay PPO")
    plotter.log(f"Deck_Size: {CONFIG['DECK_SIZE']}")
    plotter.log(f"Episodes: {CONFIG['EPISODES']}")
    plotter.log(f"Path: {paths['run_dir']}")

    # ---- Env ----
    game = pyspiel.load_game("president", {
        "num_players": 4,
        "deck_size":   CONFIG["DECK_SIZE"],
        "shuffle_cards": True,
        "single_card_mode": False,
    })
    env = rl_environment.Environment(game)
    info_dim = env.observation_spec()["info_state"][0]
    A = env.action_spec()["num_actions"]
    num_players = game.num_players()

    # ---- Features ----
    deck_int = int(CONFIG["DECK_SIZE"])
    num_ranks = ranks_for_deck(deck_int)
    # Seat-OneHot wird über agent._make_input angehängt → hier False
    feat_cfg = FeatureConfig(
        num_players=num_players, num_ranks=num_ranks,
        add_seat_onehot=False,
        normalize=CONFIG["FEATURES"]["NORMALIZE"],
    )
    seat_id_dim = (num_players if CONFIG["FEATURES"]["SEAT_ONEHOT"] else 0)

    # ---- Agent / Shaper ----
    ppo_cfg = ppo.PPOConfig(**CONFIG["PPO"]) 
    agent = ppo.PPOAgent(info_dim, A, seat_id_dim=seat_id_dim, config=ppo_cfg)
    shaper = RewardShaper(CONFIG["REWARD"])

    # ---- Config & Meta speichern ----
    save_config_csv({
        "script": family, "version": version,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent_type": "PPO_snapshot", "num_episodes": CONFIG["EPISODES"],
        "bench_interval": CONFIG["BENCH_INTERVAL"], "bench_episodes": CONFIG["BENCH_EPISODES"],
        "deck_size": CONFIG["DECK_SIZE"], "num_ranks": num_ranks,
        "observation_dim": info_dim + seat_id_dim, "num_actions": A,
        "normalize": CONFIG["FEATURES"]["NORMALIZE"], "seat_onehot": CONFIG["FEATURES"]["SEAT_ONEHOT"],
        "models_dir": paths["weights_dir"], "plots_dir": paths["plots_dir"],
        "mix_current": CONFIG["SNAPSHOT"]["MIX_CURRENT"],
        "snapshot_interval": CONFIG["SNAPSHOT"]["SNAPSHOT_INTERVAL"],
        "pool_cap": CONFIG["SNAPSHOT"]["POOL_CAP"],
        "learner_seat": CONFIG["SNAPSHOT"]["LEARNER_SEAT"],

        # Reward-Setup (neues System)
        "step_mode": shaper.step_mode,
        "delta_weight": shaper.dw,
        "hand_penalty_coeff": shaper.hp,
        "final_mode": shaper.final_mode,
        "bonus_win":   CONFIG["REWARD"]["BONUS_WIN"],
        "bonus_2nd":   CONFIG["REWARD"]["BONUS_2ND"],
        "bonus_3rd":   CONFIG["REWARD"]["BONUS_3RD"],
        "bonus_last":  CONFIG["REWARD"]["BONUS_LAST"],
    }, paths["config_csv"])
    save_run_meta({"family": family, "version": version, "algo": "ppo_snapshot", "deck": CONFIG["DECK_SIZE"]}, paths["run_meta_json"])

    # ---- Snapshot-Pool ----
    pool: list[dict] = []
    MIX = float(CONFIG["SNAPSHOT"]["MIX_CURRENT"]) 
    SNAPINT = int(CONFIG["SNAPSHOT"]["SNAPSHOT_INTERVAL"]) 
    POOL_CAP = int(CONFIG["SNAPSHOT"]["POOL_CAP"]) 
    LEARNER_SEAT = int(CONFIG["SNAPSHOT"]["LEARNER_SEAT"]) 

    # ---- Timing ----
    timer = TimingMeter(csv_path=paths["timings_csv"], interval=CONFIG["TIMING_INTERVAL"])
    t0 = time.perf_counter()
    BINT, BEPS = CONFIG["BENCH_INTERVAL"], CONFIG["BENCH_EPISODES"]

    # ---- Training ----
    for ep in range(1, CONFIG["EPISODES"] + 1):
        ep_start = time.perf_counter()
        ep_len = 0
        ep_shaping_return = 0.0

        ts = env.reset()
        last_idx_learner = None  # Index der letzten Transition des Learner-Seats

        # Seats != LEARNER_SEAT: aktuelle Policy oder Snapshot
        seat_actor = {}
        for seat in [s for s in range(num_players) if s != LEARNER_SEAT]:
            use_current = (len(pool) == 0) or (np.random.rand() < MIX)
            if use_current:
                seat_actor[seat] = "current"
            else:
                sd = pool[np.random.randint(len(pool))]
                seat_actor[seat] = SnapshotPolicy(info_dim + seat_id_dim, A, sd)

        while not ts.last():
            p = ts.observations["current_player"]
            legal = ts.observations["legal_actions"][p]

            base_obs = ts.observations["info_state"][p]
            obs = augment_observation(base_obs, player_id=p, cfg=feat_cfg)

            seat_oh = None
            if CONFIG["FEATURES"]["SEAT_ONEHOT"]:
                seat_oh = np.zeros(num_players, dtype=np.float32); seat_oh[p] = 1.0

            # Step-Reward-Prep
            if shaper.step_active():
                hand_before = shaper.hand_size(ts, p, deck_int)

            if p == LEARNER_SEAT:
                a = int(agent.step(obs, legal, seat_one_hot=seat_oh, player_id=p))
                last_idx_learner = len(agent._buffer.states) - 1
            else:
                actor = seat_actor[p]
                if actor == "current":
                    x = agent._make_input(obs, seat_one_hot=seat_oh)
                    with torch.no_grad():
                        logits = agent._policy(x)
                        mask = torch.zeros(A, device=logits.device); mask[legal] = 1.0
                        probs = ppo.masked_softmax(logits, mask)
                    a = int(torch.distributions.Categorical(probs=probs).sample().item())
                else:
                    x = agent._make_input(obs, seat_one_hot=seat_oh).cpu().numpy()
                    a = int(actor.act(x, legal))

            ts_next = env.step([a])
            ep_len += 1

            if p == LEARNER_SEAT and shaper.step_active():
                hand_after = shaper.hand_size(ts_next, p, deck_int)
                r = shaper.step_reward(hand_before=hand_before, hand_after=hand_after)
                ep_shaping_return += float(r)
                agent.post_step(r, done=ts_next.last())

            ts = ts_next

        # Finale Returns/Bonis für Lern-Sitz
        ep_env_return  = float(ts.rewards[LEARNER_SEAT])
        ep_final_bonus = float(shaper.final_bonus(ts.rewards, LEARNER_SEAT))

        if last_idx_learner is not None:
            if shaper.include_env_reward():
                agent._buffer.rewards[last_idx_learner] += ep_env_return
            agent._buffer.rewards[last_idx_learner] += ep_final_bonus
            agent._buffer.dones[last_idx_learner] = True

        # Train
        train_start = time.perf_counter()
        train_metrics = agent.train()
        train_seconds = time.perf_counter() - train_start

        if train_metrics is None:
            train_metrics = {}
        train_metrics.update({
            "train_seconds":      train_seconds,
            "ep_env_return":      ep_env_return,
            "ep_shaping_return":  ep_shaping_return,
            "ep_final_bonus":     ep_final_bonus,
            "ep_length":          ep_len,
        })
        plotter.add_train(ep, train_metrics)

        # Snapshot in Pool?
        if SNAPINT > 0 and (ep % SNAPINT == 0):
            pool.append(copy.deepcopy(agent._policy.state_dict()))
            if len(pool) > POOL_CAP:
                pool.pop(0)

        # Benchmark & Save (nur Learner-Policy)
        eval_seconds = plot_seconds = save_seconds = 0.0
        # Benchmark & Save (nur Learner-Policy)
        eval_seconds = plot_seconds = save_seconds = 0.0
        if BINT > 0 and (ep % BINT == 0):
            ev_start = time.perf_counter()
            per_opponent = run_benchmark(
                game=game,
                agent=agent,
                opponents_dict=STRATS,
                opponent_names=CONFIG["BENCH_OPPONENTS"],
                episodes=BEPS,
                feat_cfg=feat_cfg,
                num_actions=A,
            )

            plotter.log_bench_summary(ep, per_opponent)
            eval_seconds = time.perf_counter() - ev_start

            plot_start = time.perf_counter()
            plotter.add_benchmark(ep, per_opponent)
            plotter.plot_benchmark_rewards()
            plotter.plot_places_latest()

            # Einheitliche Titel für alle „lernkurve“-Plots:
            # - Einzelplots:  "Lernkurve - KxAy vs <gegner>"
            # - Multi/Macro:  "Lernkurve - KxAy vs feste Heuristiken"
            title_multi = f"Lernkurve - {family.upper()} vs feste Heuristiken"
            plotter.plot_benchmark(
                filename_prefix="lernkurve",
                with_macro=True,
                family_title=family.upper(),   # für Einzelplots
                multi_title=title_multi,       # für Multi- & Macro-Plot (identischer Titel)
            )

            plotter.plot_train(filename_prefix="training_metrics", separate=True)
            plot_seconds = time.perf_counter() - plot_start

            save_start = time.perf_counter()
            tag = f"{family}_model_{version}_agent_p{LEARNER_SEAT}_ep{ep:07d}"
            save_checkpoint_ppo(agent, paths["weights_dir"], tag)
            save_seconds = time.perf_counter() - save_start

            cum_seconds = time.perf_counter() - t0
            plotter.log_timing(
                ep,
                ep_seconds=(time.perf_counter() - ep_start),
                train_seconds=train_seconds,
                eval_seconds=eval_seconds,
                plot_seconds=plot_seconds,
                save_seconds=save_seconds,
                cum_seconds=cum_seconds,
            )


        # Timing CSV
        ep_seconds = time.perf_counter() - ep_start
        timer.maybe_log(ep, {
            "steps": ep_len,
            "ep_seconds": ep_seconds,
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds,
            "plot_seconds": plot_seconds,
            "save_seconds": save_seconds,
        })

    total_seconds = time.perf_counter() - t0
    plotter.log("")
    plotter.log(f"Gesamtzeit: {total_seconds/3600:0.2f}h (~ {CONFIG['EPISODES']/max(total_seconds,1e-9):0.2f} eps/s)")  
    plotter.log(f"{family}, Snapshot Selfplay (1 Learner, 3 Sparring). Training abgeschlossen.")
    plotter.log(f"Path: {paths['run_dir']}")

if __name__ == "__main__":
    main()
