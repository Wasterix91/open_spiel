# -*- coding: utf-8 -*-
# k1a2.py — K1: Single-Agent DQN vs 3 Heuristiken (k1a1-Stil, neues Reward-System)

import os, time, datetime, numpy as np, torch
import pyspiel
from open_spiel.python import rl_environment

from agents.dqn_agent import DQNAgent, DQNConfig
from utils.fit_tensor import FeatureConfig, augment_observation
from utils.deck import ranks_for_deck
from utils.plotter import MetricsPlotter
from utils.timing import TimingMeter
from utils.load_save_common import find_next_version, prepare_run_dirs, save_config_csv, save_run_meta
from utils.benchmark import run_benchmark
from utils.strategies import STRATS
from utils.reward_shaper import RewardShaper

CONFIG = {
    "EPISODES":         1000,
    "BENCH_INTERVAL":   200,
    "BENCH_EPISODES":   500,
    "TIMING_INTERVAL":  250,
    # Erlaubt: "12" | "16" | "20" | "24" | "32" | "52" | "64"
    "DECK_SIZE":        "16",
    "SEED":             42,

    "OPPONENTS":        ["max_combo", "max_combo", "max_combo"],

    # DQN (dqn_agent2)
    "DQN": {
        "learning_rate": 3e-4, "batch_size": 128, "gamma": 0.995,
        "buffer_size": 200_000, "target_update_freq": 5000, "soft_target_tau": 0.0,
        "max_grad_norm": 1.0, "n_step": 3, "per_alpha": 0.6,
        "per_beta_start": 0.4, "per_beta_frames": 1_000_000,
        "eps_start": 1.0, "eps_end": 0.05, "eps_decay_frames": 500_000,
        "loss_huber_delta": 1.0, "dueling": True, "device": "cpu",
    },

    # ======= Rewards (NEUES System wie k1a1) =======
    # STEP_MODE : "none" | "delta_weight_only" | "hand_penalty_coeff_only" | "combined"
    # FINAL_MODE: "none" | "env_only" | "rank_bonus" | "both"
    "REWARD": {
        "STEP_MODE": "delta_weight_only",
        "DELTA_WEIGHT": 0.5,
        "HAND_PENALTY_COEFF": 0.0,

        "FINAL_MODE": "env_only",
        "BONUS_WIN": 10.0, "BONUS_2ND": 0.0, "BONUS_3RD": 0.0, "BONUS_LAST": 0.0,
    },

    "FEATURES": { "NORMALIZE": False, "SEAT_ONEHOT": False },
    "BENCH_OPPONENTS": ["single_only", "max_combo", "random2"],
}

def main():
    np.random.seed(CONFIG["SEED"]); torch.manual_seed(CONFIG["SEED"])
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_ROOT = os.path.join(ROOT, "models")

    family = "k1a2"
    family_dir = os.path.join(MODELS_ROOT, family)
    version = find_next_version(family_dir, prefix="model")
    paths = prepare_run_dirs(MODELS_ROOT, family, version, prefix="model")

    plotter = MetricsPlotter(
        out_dir=paths["plots_dir"],
        benchmark_opponents=list(CONFIG["BENCH_OPPONENTS"]),
        benchmark_csv="benchmark_curves.csv",
        train_csv="training_metrics.csv",
        save_csv=True, verbosity=1
    )
    plotter.log("New Training (k1a2): DQN Single-Agent vs Heuristiken")
    plotter.log(f"Deck_Size: {CONFIG['DECK_SIZE']}")
    plotter.log(f"Episodes: {CONFIG['EPISODES']}")
    plotter.log(f"Path: {paths['run_dir']}")

    # Env
    game = pyspiel.load_game("president", {
        "num_players": 4, "deck_size": CONFIG["DECK_SIZE"],
        "shuffle_cards": True, "single_card_mode": False
    })
    env = rl_environment.Environment(game)
    info_dim = env.observation_spec()["info_state"][0]
    A = env.action_spec()["num_actions"]
    num_players = game.num_players()

    # Features
    deck_int = int(CONFIG["DECK_SIZE"])
    num_ranks = ranks_for_deck(deck_int)
    feat_cfg = FeatureConfig(
        num_players=num_players, num_ranks=num_ranks,
        add_seat_onehot=CONFIG["FEATURES"]["SEAT_ONEHOT"],
        normalize=CONFIG["FEATURES"]["NORMALIZE"],
    )

    # Agent, Gegner, Reward-Shaper
    agent = DQNAgent(info_dim, A, DQNConfig(**CONFIG["DQN"]))
    opponents = [STRATS[name] for name in CONFIG["OPPONENTS"]]
    shaper = RewardShaper(CONFIG["REWARD"])

    # Config/Meta
    save_config_csv({
        "script": family, "version": version,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent_type": "DQN", "num_episodes": CONFIG["EPISODES"],
        "bench_interval": CONFIG["BENCH_INTERVAL"], "bench_episodes": CONFIG["BENCH_EPISODES"],
        "deck_size": CONFIG["DECK_SIZE"], "num_ranks": num_ranks,
        "observation_dim": info_dim, "num_actions": A,
        "normalize": CONFIG["FEATURES"]["NORMALIZE"], "seat_onehot": CONFIG["FEATURES"]["SEAT_ONEHOT"],
        "opponents": ",".join(CONFIG["OPPONENTS"]),
        "models_dir": paths["weights_dir"], "plots_dir": paths["plots_dir"],
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
    save_run_meta({"family": family, "version": version, "algo": "dqn", "deck": CONFIG["DECK_SIZE"]}, paths["run_meta_json"])

    # Timing
    timer = TimingMeter(csv_path=paths["timings_csv"], interval=CONFIG["TIMING_INTERVAL"])
    t0 = time.perf_counter()
    BINT, BEPS = CONFIG["BENCH_INTERVAL"], CONFIG["BENCH_EPISODES"]

    for ep in range(1, CONFIG["EPISODES"] + 1):
        ep_start = time.perf_counter()
        ep_len = 0
        ep_shaping_return = 0.0
        ep_final_bonus = 0.0

        ts = env.reset()

        while not ts.last():
            ep_len += 1
            p = ts.observations["current_player"]
            legal = ts.observations["legal_actions"][p]

            if p == 0:
                if shaper.step_active():
                    hand_before = shaper.hand_size(ts, p, deck_int)

                ob = augment_observation(ts.observations["info_state"][p], player_id=p, cfg=feat_cfg)
                a = agent.select_action(ob, legal)
            else:
                a = int(opponents[p-1](env._state))

            ts_next = env.step([int(a)])

            # P0 speichert: Reward = nur Shaping-Teil (ENV erst am Ende auf letzte Transition)
            if p == 0:
                if shaper.step_active():
                    hand_after = shaper.hand_size(ts_next, p, deck_int)
                    r_shape = shaper.step_reward(hand_before=hand_before, hand_after=hand_after)
                    ep_shaping_return += float(r_shape)
                else:
                    r_shape = 0.0

                ob_next = augment_observation(ts_next.observations["info_state"][p], player_id=p, cfg=feat_cfg)
                next_legal = None if ts_next.last() else ts_next.observations["legal_actions"][ts_next.observations["current_player"]]
                agent.store(ob, int(a), float(r_shape), ob_next, bool(ts_next.last()), next_legal)

            ts = ts_next

        # final env score & platzierungsbonus (nur für Logging/letzte Transition)
        ep_env_score = float(ts.rewards[0])
        ep_final_bonus = float(shaper.final_bonus(ts.rewards, 0))

        # Letzte Transition im Replay (der Episode) um ENV/Final-Bonus anreichern.
        # Hinweis: DQNAgent2 nutzt N-step PER. Am Episodenende wird die Rest-N-Step Transition
        # noch gepusht → wir können den letzten Eintrag modifizieren.
        buf = agent.buffer
        if len(buf.buffer) > 0:
            li = len(buf.buffer) - 1
            last = buf.buffer[li]
            # addiere ENV (falls konfiguriert) + Platzierungsbonus
            add_env = ep_env_score if shaper.include_env_reward() else 0.0
            new_r = float(last.r + add_env + ep_final_bonus)
            buf.buffer[li] = buf.Exp(last.s, last.a, new_r, last.ns, last.done, last.next_mask)

        # Train
        st = time.perf_counter()
        train_out = agent.train_step() or {}
        train_seconds = time.perf_counter() - st

        # Log train metrics (P0)
        metrics = {
            "ep_length": ep_len,
            "train_seconds": train_seconds,
            "ep_env_score": ep_env_score,
            "ep_shaping_return": ep_shaping_return,
            "ep_final_bonus": ep_final_bonus,
        }
        # optional aus dem Agenten übernehmen (loss/epsilon, falls vorhanden)
        for k in ("loss", "epsilon", "beta"):
            if k in train_out:
                metrics[k] = float(train_out[k])

        plotter.add_train(ep, metrics)

        # Benchmark+Save
        eval_seconds = plot_seconds = save_seconds = 0.0
        if ep % BINT == 0:
            evs = time.perf_counter()
            per_opponent = run_benchmark(
                game=game, agent=agent, opponents_dict=STRATS,
                opponent_names=CONFIG["BENCH_OPPONENTS"],
                episodes=BEPS, feat_cfg=feat_cfg, num_actions=A
            )
            plotter.log_bench_summary(ep, per_opponent)
            eval_seconds = time.perf_counter() - evs

            ps = time.perf_counter()
            plotter.add_benchmark(ep, per_opponent)
            plotter.plot_benchmark_rewards()
            plotter.plot_places_latest()
            plotter.plot_benchmark(filename_prefix="lernkurve", with_macro=True)
            plotter.plot_train(filename_prefix="training_metrics", separate=True)
            plot_seconds = time.perf_counter() - ps

            ss = time.perf_counter()
            tag = f"{family}_model_{version}_agent_p0_ep{ep:07d}"
            agent.save(os.path.join(paths["weights_dir"], tag))
            save_seconds = time.perf_counter() - ss

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

        # timing csv
        ep_seconds = time.perf_counter() - ep_start
        timer.maybe_log(ep, {
            "steps": ep_len, "ep_seconds": ep_seconds,
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds, "plot_seconds": plot_seconds, "save_seconds": save_seconds
        })

    total_seconds = time.perf_counter() - t0
    plotter.log("")
    plotter.log(f"Gesamtzeit: {total_seconds/3600:0.2f}h (~ {CONFIG['EPISODES']/max(total_seconds,1e-9):0.2f} eps/s)")
    plotter.log("K1 (DQN) Training abgeschlossen.")

if __name__ == "__main__":
    main()
