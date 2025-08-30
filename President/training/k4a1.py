# President/training/k4a1.py
# -*- coding: utf-8 -*-
# PPO (K4): Shared Policy + In-Proc "External" Trainer (Bundle-Updates)

import os, datetime, time, numpy as np, torch
import pyspiel
from open_spiel.python import rl_environment

from agents import ppo_agent as ppo
from utils.fit_tensor import FeatureConfig, augment_observation, expected_feature_len
from utils.plotter import MetricsPlotter
from utils.reward_shaper import RewardShaper
from utils.strategies import STRATS
from utils.benchmark import run_benchmark
from utils.deck import ranks_for_deck

from utils.load_save_common import find_next_version, prepare_run_dirs, save_config_csv, save_run_meta
from utils.load_save_a1_ppo import save_checkpoint_ppo

# ============== CONFIG ==============
CONFIG = {
    "EPISODES":         500_000,
    "BENCH_INTERVAL":   10_000,
    "BENCH_EPISODES":   2000,
    "DECK_SIZE":        "64",  # "12" | "16" | "20" | "24" | "32" | "52" | "64"
    "SEED":             42,
    # PPO
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

    # In-Proc „External“ Trainer (Bündel-Updates)
    "INPROC_TRAINER": {
        "EPISODES_PER_UPDATE": 50,   # Bundle-Größe
        "UPDATES_PER_CALL":     2,   # wie oft agent.train() pro Bundle
        "MIN_SAMPLES_TO_TRAIN": 1000 # Guard, um Mini-Bundles zu vermeiden
    },

    # Rewards
    "REWARD": {
        "STEP_MODE": "none",         # "none" | "delta_weight_only" | "hand_penalty_coeff_only" | "combined"
        "DELTA_WEIGHT": 0.0,
        "HAND_PENALTY_COEFF": 0.0,

        "FINAL_MODE": "env_only",    # "none" | "env_only" | "rank_bonus" | "both"
        "BONUS_WIN": 0.0, "BONUS_2ND": 0.0, "BONUS_3RD": 0.0, "BONUS_LAST": 0.0,
    },

    # Features
    "FEATURES": {
        "USE_HISTORY": True,
        "SEAT_ONEHOT": True,
        "PLOT_METRICS": False,
        "SAVE_METRICS_TO_CSV": False,   # <- Flag ist nun maßgeblich für Plotter-CSV
    },

    # Benchmark-Gegner
    "BENCH_OPPONENTS": ["single_only", "max_combo", "random2"],
}

def _with_seat_onehot(vec: np.ndarray, p: int, num_players: int, enabled: bool) -> np.ndarray:
    if not enabled:
        return vec
    seat_oh = np.zeros(num_players, dtype=np.float32); seat_oh[p] = 1.0
    return np.concatenate([vec, seat_oh], axis=0)

def main():
    np.random.seed(CONFIG["SEED"]); torch.manual_seed(CONFIG["SEED"])
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_ROOT = os.path.join(ROOT, "models")

    family = "k4a1"
    family_dir = os.path.join(MODELS_ROOT, family)
    version = find_next_version(family_dir, prefix="model")
    paths = prepare_run_dirs(MODELS_ROOT, family, version, prefix="model")

    # Plotter mit korrektem save_csv-Flag
    plotter = MetricsPlotter(
        out_dir=paths["plots_dir"],
        benchmark_opponents=list(CONFIG["BENCH_OPPONENTS"]),
        benchmark_csv="benchmark_curves.csv",
        train_csv="training_metrics.csv",
        save_csv=CONFIG["FEATURES"].get("SAVE_METRICS_TO_CSV", False),
        verbosity=1,
    )
    plotter.log("New Training (k4a1): Shared-Policy PPO — In-Proc External Trainer (Bundle-Updates)")
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
    A = env.action_spec()["num_actions"]
    num_players = game.num_players()

    # ---- Features ----
    deck_int  = int(CONFIG["DECK_SIZE"])
    num_ranks = ranks_for_deck(deck_int)
    feat_cfg = FeatureConfig(
        num_players=num_players,
        num_ranks=num_ranks,
        add_seat_onehot=False,  # Seat-1hot NICHT in augment_observation angehängt
        include_history=CONFIG["FEATURES"]["USE_HISTORY"],
    )
    seat_id_dim = num_players if CONFIG["FEATURES"]["SEAT_ONEHOT"] else 0
    # korrektes observation_dim (inkl. optionalem Seat-1hot)
    observation_dim = expected_feature_len(feat_cfg) + seat_id_dim

    # ---- Agent / Shaper ----
    ppo_cfg = ppo.PPOConfig(**CONFIG["PPO"])
    agent = ppo.PPOAgent(
        info_state_size=expected_feature_len(feat_cfg),
        num_actions=A,
        seat_id_dim=seat_id_dim,
        config=ppo_cfg,
        segmented_gae_mode="jump"
    )
    shaper = RewardShaper(CONFIG["REWARD"])

    # ---- Config & Meta ----
    save_config_csv({
        "script": family, "version": version,
        "deck_size": CONFIG["DECK_SIZE"], "num_ranks": num_ranks,
        "use_history": CONFIG["FEATURES"]["USE_HISTORY"],

        # Reward-Setup
        "step_mode": shaper.step_mode,
        "delta_weight": shaper.dw,
        "hand_penalty_coeff": shaper.hp,
        "final_mode": shaper.final_mode,
        "bonus_win":   CONFIG["REWARD"]["BONUS_WIN"],
        "bonus_2nd":   CONFIG["REWARD"]["BONUS_2ND"],
        "bonus_3rd":   CONFIG["REWARD"]["BONUS_3RD"],
        "bonus_last":  CONFIG["REWARD"]["BONUS_LAST"],

        "agent_type": "PPO_shared_inproc", "num_episodes": CONFIG["EPISODES"],
        "bench_interval": CONFIG["BENCH_INTERVAL"], "bench_episodes": CONFIG["BENCH_EPISODES"],
        "observation_dim": observation_dim,
        "num_actions": A,
        "seat_onehot": CONFIG["FEATURES"]["SEAT_ONEHOT"],

        # In-Proc Trainer
        "episodes_per_update": CONFIG["INPROC_TRAINER"]["EPISODES_PER_UPDATE"],
        "updates_per_call":    CONFIG["INPROC_TRAINER"]["UPDATES_PER_CALL"],
        "min_samples_to_train": CONFIG["INPROC_TRAINER"]["MIN_SAMPLES_TO_TRAIN"],

        "models_dir": paths["weights_dir"], "plots_dir": paths["plots_dir"],
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_opponents": ",".join(CONFIG["BENCH_OPPONENTS"]),
    }, paths["config_csv"])
    save_run_meta({"family": family, "version": version, "algo": "ppo_shared_inproc", "deck": CONFIG["DECK_SIZE"]},
                  paths["run_meta_json"])

    t0 = time.perf_counter()
    BINT, BEPS = CONFIG["BENCH_INTERVAL"], CONFIG["BENCH_EPISODES"]

    EPISODES_PER_UPDATE = int(CONFIG["INPROC_TRAINER"]["EPISODES_PER_UPDATE"])
    UPDATES_PER_CALL    = int(CONFIG["INPROC_TRAINER"]["UPDATES_PER_CALL"])
    MIN_SAMPLES         = int(CONFIG["INPROC_TRAINER"]["MIN_SAMPLES_TO_TRAIN"])
    ep_in_bundle = 0

    collect_metrics = CONFIG["FEATURES"].get("SAVE_METRICS_TO_CSV", False) or CONFIG["FEATURES"].get("PLOT_METRICS", False)

    # ---- Training (Sammeln → Bündel-Updates) ----
    for ep in range(1, CONFIG["EPISODES"] + 1):
        ep_start = time.perf_counter()
        ep_len = 0

        ts = env.reset()
        last_idx = {p: None for p in range(num_players)}  # letzte Transition je Sitz (für Finals)

        while not ts.last():
            p = ts.observations["current_player"]
            legal = ts.observations["legal_actions"][p]

            # Einheitliche Basis: observation_tensor
            base_obs = np.asarray(env._state.observation_tensor(p), dtype=np.float32)
            obs = augment_observation(base_obs, player_id=p, cfg=feat_cfg)

            seat_oh = None
            if CONFIG["FEATURES"]["SEAT_ONEHOT"]:
                seat_oh = np.zeros(num_players, dtype=np.float32); seat_oh[p] = 1.0

            # optionaler Step-Reward
            if shaper.step_active():
                hand_before = shaper.hand_size(ts, p, deck_int)

            a = int(agent.step(obs, legal, seat_one_hot=seat_oh, player_id=p))
            last_idx[p] = len(agent._buffer.states) - 1

            ts = env.step([a])
            ep_len += 1

            if shaper.step_active():
                hand_after = shaper.hand_size(ts, p, deck_int)
                step_r = shaper.step_reward(hand_before=hand_before, hand_after=hand_after)
                agent.post_step(step_r, done=ts.last())

        # Episodenende: Finals/Bonis je Sitz in die *letzte* Transition buchen + done markieren
        finals = env._state.returns()
        for p in range(num_players):
            li = last_idx[p]
            if li is None: continue
            if shaper.include_env_reward():
                agent._buffer.rewards[li] += float(finals[p])
            agent._buffer.rewards[li] += float(shaper.final_bonus(finals, p))
            agent._buffer.dones[li] = True

        # Sammelzähler
        ep_in_bundle += 1

        # Episoden-Metriken (einheitlich)
        ep_metrics = {"ep_length": int(ep_len), "ep_env_return_p0": float(finals[0])}
        if collect_metrics:
            if CONFIG["FEATURES"].get("SAVE_METRICS_TO_CSV", False):
                plotter.add_train(ep, ep_metrics)
            else:
                plotter.train_rows.append({"episode": int(ep), **ep_metrics})
                if plotter.train_keys is None:
                    plotter.train_keys = ["episode"] + list(ep_metrics.keys())

        # ======= Bündel-Update =======
        train_seconds = 0.0
        if ep_in_bundle >= EPISODES_PER_UPDATE:
            buf_len = len(agent._buffer.states)
            if buf_len >= MIN_SAMPLES:
                t_train = time.perf_counter()
                for _ in range(UPDATES_PER_CALL):
                    train_metrics = agent.train()
                train_seconds = time.perf_counter() - t_train

                if collect_metrics:
                    # Logging der Update-Größe/Qualität
                    bs = int(CONFIG["PPO"]["batch_size"])
                    batches_per_epoch = max(1, buf_len // bs)
                    upd_metrics = {
                        "train_seconds": float(train_seconds),
                        "buffer_len": int(buf_len),
                        "batches_per_epoch": int(batches_per_epoch),
                        "num_epochs": int(CONFIG["PPO"]["num_epochs"]),
                    }
                    # PPO-eigene Metriken, falls vorhanden
                    if train_metrics:
                        upd_metrics.update({
                            "policy_loss": train_metrics.get("policy_loss", 0.0),
                            "value_loss":  train_metrics.get("value_loss", 0.0),
                            "entropy":     train_metrics.get("entropy", 0.0),
                            "approx_kl":   train_metrics.get("approx_kl", 0.0),
                            "clip_frac":   train_metrics.get("clip_frac", 0.0),
                        })
                    if CONFIG["FEATURES"].get("SAVE_METRICS_TO_CSV", False):
                        plotter.add_train(ep, upd_metrics)
                    else:
                        plotter.train_rows.append({"episode": int(ep), **upd_metrics})
                        if plotter.train_keys is None:
                            plotter.train_keys = ["episode"] + list(upd_metrics.keys())
            # Buffer leeren (on-policy Rhythmus)
            agent._buffer.clear()
            ep_in_bundle = 0

        # ===== Benchmark & Save =====
        eval_seconds = plot_seconds = save_seconds = 0.0
        if BINT > 0 and (ep % BINT == 0):
            ev_start = time.perf_counter()
            per_opponent = run_benchmark(
                game=game, agent=agent, opponents_dict=STRATS,
                opponent_names=CONFIG["BENCH_OPPONENTS"], episodes=BEPS,
                feat_cfg=feat_cfg, num_actions=A
            )
            plotter.log_bench_summary(ep, per_opponent)
            eval_seconds = time.perf_counter() - ev_start

            plot_start = time.perf_counter()
            plotter.add_benchmark(ep, per_opponent)
            plotter.plot_benchmark_rewards()
            plotter.plot_places_latest()

            title_multi = f"Lernkurve - {family.upper()} vs feste Heuristiken"
            plotter.plot_benchmark(
                filename_prefix="lernkurve",
                with_macro=True,
                family_title=family.upper(),
                multi_title=title_multi,
            )
            if CONFIG["FEATURES"].get("PLOT_METRICS", False):
                plotter.plot_train(filename_prefix="training_metrics", separate=True)
            plot_seconds = time.perf_counter() - plot_start

            save_start = time.perf_counter()
            tag = f"{family}_model_{version}_agent_p0_ep{ep:07d}"
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

    total_seconds = time.perf_counter() - t0
    plotter.log("")
    plotter.log(f"Gesamtzeit: {total_seconds/3600:0.2f}h (~ {CONFIG['EPISODES']/max(total_seconds,1e-9):0.2f} eps/s)")
    plotter.log(f"{family}, Shared Policy Selfplay (PPO, Bundle-Updates, 1 Prozess). Training abgeschlossen.")
    plotter.log(f"Path: {paths['run_dir']}")

if __name__ == "__main__":
    main()
