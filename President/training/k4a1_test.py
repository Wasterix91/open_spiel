# President/training/k4a1_rec.py
# -*- coding: utf-8 -*-
# PPO (K4 Recommended): Shared Policy (4 Sammler) + In-Proc "External" Trainer (Bundle-Updates)

import os, datetime, time, numpy as np, torch
import pyspiel
from open_spiel.python import rl_environment

from agents import ppo_agent as ppo
from utils.fit_tensor import FeatureConfig, augment_observation
from utils.plotter import MetricsPlotter
from utils.timing import TimingMeter
from utils.reward_shaper import RewardShaper
from utils.strategies import STRATS

from utils.load_save_common import find_next_version, prepare_run_dirs, save_config_csv, save_run_meta
from utils.load_save_a1_ppo import save_checkpoint_ppo

from utils.benchmark import run_benchmark
from utils.deck import ranks_for_deck


# ============== CONFIG ==============
CONFIG = {
    "EPISODES":         10_000,
    "BENCH_INTERVAL":   500,
    "BENCH_EPISODES":   2_000,
    "TIMING_INTERVAL":  500,
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


    # ===== In-Proc „External“ Trainer =====
    # Sammle N Episoden → mache K PPO-Updates → Buffer clear → weiter sammeln
    "INPROC_TRAINER": {
        "EPISODES_PER_UPDATE": 20,  # Bundle-Größe
        "UPDATES_PER_CALL":     1,    # wie oft agent.train() pro Bundle
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

    # Features (Shared Policy: Seat-OneHot optional)
    "FEATURES": { "NORMALIZE": False, "SEAT_ONEHOT": True },

    # Benchmark-Gegner
    "BENCH_OPPONENTS": ["single_only", "max_combo", "random2"],
}


def main():
    np.random.seed(CONFIG["SEED"]); torch.manual_seed(CONFIG["SEED"])
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_ROOT = os.path.join(ROOT, "models")

    # ---- Verzeichnisse (k1a1-Stil) ----
    family = "k4a1"
    family_dir = os.path.join(MODELS_ROOT, family)
    version = find_next_version(family_dir, prefix="model")
    paths = prepare_run_dirs(MODELS_ROOT, family, version, prefix="model")

    # ---- Plotter ----
    plotter = MetricsPlotter(
        out_dir=paths["plots_dir"],
        benchmark_opponents=list(CONFIG["BENCH_OPPONENTS"]),
        benchmark_csv="benchmark_curves.csv",
        train_csv="training_metrics.csv",
        save_csv=True, verbosity=1,
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
    info_dim = env.observation_spec()["info_state"][0]
    A = env.action_spec()["num_actions"]
    num_players = game.num_players()

    # ---- Features ----
    deck_int = int(CONFIG["DECK_SIZE"])
    num_ranks = ranks_for_deck(deck_int)
    feat_cfg = FeatureConfig(
        num_players=num_players, num_ranks=num_ranks,
        add_seat_onehot=False,                             
        normalize=CONFIG["FEATURES"]["NORMALIZE"],
    )
    seat_id_dim = num_players if CONFIG["FEATURES"]["SEAT_ONEHOT"] else 0

    # ---- Agent / Shaper ----
    ppo_cfg = ppo.PPOConfig(**CONFIG["PPO"])
    agent = ppo.PPOAgent(info_dim, A, seat_id_dim=seat_id_dim, config=ppo_cfg)
    shaper = RewardShaper(CONFIG["REWARD"])

    # ---- Config & Meta ----
    from utils.load_save_common import save_config_csv, save_run_meta
    save_config_csv({
        "script": family, "version": version,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent_type": "PPO_shared_inproc", "num_episodes": CONFIG["EPISODES"],
        "bench_interval": CONFIG["BENCH_INTERVAL"], "bench_episodes": CONFIG["BENCH_EPISODES"],
        "deck_size": CONFIG["DECK_SIZE"], "num_ranks": num_ranks,
        "observation_dim": info_dim + seat_id_dim, "num_actions": A,
        "normalize": CONFIG["FEATURES"]["NORMALIZE"], "seat_onehot": CONFIG["FEATURES"]["SEAT_ONEHOT"],
        "models_dir": paths["weights_dir"], "plots_dir": paths["plots_dir"],
        # Reward-Setup
        "step_mode": shaper.step_mode,
        "delta_weight": shaper.dw,
        "hand_penalty_coeff": shaper.hp,
        "final_mode": shaper.final_mode,
        "bonus_win":   CONFIG["REWARD"]["BONUS_WIN"],
        "bonus_2nd":   CONFIG["REWARD"]["BONUS_2ND"],
        "bonus_3rd":   CONFIG["REWARD"]["BONUS_3RD"],
        "bonus_last":  CONFIG["REWARD"]["BONUS_LAST"],
        # In-Proc Trainer
        "episodes_per_update": CONFIG["INPROC_TRAINER"]["EPISODES_PER_UPDATE"],
        "updates_per_call":    CONFIG["INPROC_TRAINER"]["UPDATES_PER_CALL"],
    }, paths["config_csv"])
    save_run_meta({"family": family, "version": version, "algo": "ppo_shared_inproc", "deck": CONFIG["DECK_SIZE"]},
                  paths["run_meta_json"])

    # ---- Timing ----
    timer = TimingMeter(csv_path=paths["timings_csv"], interval=CONFIG["TIMING_INTERVAL"])
    t0 = time.perf_counter()
    BINT, BEPS = CONFIG["BENCH_INTERVAL"], CONFIG["BENCH_EPISODES"]

    # ---- In-Proc External Trainer Steuerung ----
    EPISODES_PER_UPDATE = int(CONFIG["INPROC_TRAINER"]["EPISODES_PER_UPDATE"])
    UPDATES_PER_CALL    = int(CONFIG["INPROC_TRAINER"]["UPDATES_PER_CALL"])
    ep_in_bundle = 0  # zählt gesammelte Episoden seit letztem Update

    # ---- Training (Sammeln → Bündel-Update) ----
    for ep in range(1, CONFIG["EPISODES"] + 1):
        ep_start = time.perf_counter()
        ep_len = 0

        ts = env.reset()
        last_idx = {p: None for p in range(num_players)}  # letzte Transition je Sitz (für Finals)

        while not ts.last():
            p = ts.observations["current_player"]
            legal = ts.observations["legal_actions"][p]
            base_obs = ts.observations["info_state"][p]
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

        # Sammelzähler hoch
        ep_in_bundle += 1

        # Logging (nur Sammelmetriken; Training erst beim Bundle-Update)
        plotter.add_train(ep, {"ep_length": ep_len, "ep_env_return": float(finals[0])})

        # ======= In-Proc „External“ Trainer: Update nach N Episoden =======
        train_seconds = 0.0
        if ep_in_bundle >= EPISODES_PER_UPDATE:
            t_train = time.perf_counter()

            # Mehrfach-Update auf dem gesammelten Bundle
            for _ in range(UPDATES_PER_CALL):
                agent.train()

            train_seconds = time.perf_counter() - t_train
            ep_in_bundle = 0
            # Nach dem Update den Rollout-Buffer leeren (klassische Actor→Learner-Übergabe)
            agent._buffer.clear()

            # Trainingszeit nachtragen
            plotter.add_train(ep, {"train_seconds": train_seconds})

        # ===== Benchmark & Save (Policy von Player 0) =====
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
            plotter.plot_benchmark(filename_prefix="lernkurve", with_macro=True)
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
    plotter.log("K4 (Shared-Policy PPO, In-Proc External Trainer) abgeschlossen.")

if __name__ == "__main__":
    main()
