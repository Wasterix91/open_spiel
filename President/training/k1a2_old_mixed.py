# -*- coding: utf-8 -*-
# k1a2.py — K1: Single-Agent DQN vs 3 Heuristiken (P0→P0-Transitions, Light-Agent)

import os, time, datetime
import numpy as np, torch
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
    "EPISODES":         100_000,
    "BENCH_INTERVAL":   5000,
    "BENCH_EPISODES":   2_000,
    "TIMING_INTERVAL":  500,
    "DECK_SIZE":        "64",  # "12" | "16" | "20" | "24" | "32" | "52" | "64"
    "SEED":             42,

    # Training-Gegner (fix; wenn du sampeln willst, ersetze das unten episodisch)
    "OPPONENTS":        ["max_combo", "max_combo", "max_combo"],

    # ===== DQN (Light+) – Keys passend zu agents/dqn_agent.py =====
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

    # ======= Rewards (kompatibel zu utils.reward_shaper) =======
    # STEP_MODE: "none" | "delta_weight_only" | "hand_penalty_coeff_only" | "combined"
    # FINAL_MODE: "none" | "env_only" | "rank_bonus" | "both"
    "REWARD": {
        "STEP_MODE": "none",
        "DELTA_WEIGHT": 0.5,
        "HAND_PENALTY_COEFF": 0.0,

        "FINAL_MODE": "env_only",
        "BONUS_WIN": 10.0, "BONUS_2ND": 0.0, "BONUS_3RD": 0.0, "BONUS_LAST": 0.0,
    },

    "FEATURES": { "NORMALIZE": False, "SEAT_ONEHOT": False },

    # Benchmark-Gegner (für Plotter/Reports)
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
    plotter.log("New Training (k1a2): DQN Single-Agent vs Heuristiken (P0→P0)")
    plotter.log(f"Deck_Size: {CONFIG['DECK_SIZE']}")
    plotter.log(f"Episodes: {CONFIG['EPISODES']}")
    plotter.log(f"Path: {paths['run_dir']}")

    # ---- Env ----
    game = pyspiel.load_game("president", {
        "num_players": 4, "deck_size": CONFIG["DECK_SIZE"],
        "shuffle_cards": True, "single_card_mode": False
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
        add_seat_onehot=CONFIG["FEATURES"]["SEAT_ONEHOT"],
        normalize=CONFIG["FEATURES"]["NORMALIZE"],
    )

    # ---- Agent, Gegner, Reward-Shaper ----
    agent = DQNAgent(info_dim + (num_players if CONFIG["FEATURES"]["SEAT_ONEHOT"] else 0), A, DQNConfig(**CONFIG["DQN"]))
    opponents = [STRATS[name] for name in CONFIG["OPPONENTS"]]
    shaper = RewardShaper(CONFIG["REWARD"])

    # ---- Config/Meta speichern ----
    save_config_csv({
        "script": family, "version": version,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent_type": "DQN", "num_episodes": CONFIG["EPISODES"],
        "bench_interval": CONFIG["BENCH_INTERVAL"], "bench_episodes": CONFIG["BENCH_EPISODES"],
        "deck_size": CONFIG["DECK_SIZE"], "num_ranks": num_ranks,
        "observation_dim": info_dim + (num_players if CONFIG["FEATURES"]["SEAT_ONEHOT"] else 0),
        "num_actions": A,
        "normalize": CONFIG["FEATURES"]["NORMALIZE"], "seat_onehot": CONFIG["FEATURES"]["SEAT_ONEHOT"],
        "opponents": ",".join(CONFIG["OPPONENTS"]),
        "models_dir": paths["weights_dir"], "plots_dir": paths["plots_dir"],
        # Reward-Setup
        "step_mode": getattr(shaper, "step_mode", "n/a"),
        "delta_weight": getattr(shaper, "dw", "n/a"),
        "hand_penalty_coeff": getattr(shaper, "hp", "n/a"),
        "final_mode": getattr(shaper, "final_mode", "n/a"),
        "bonus_win":   CONFIG["REWARD"]["BONUS_WIN"],
        "bonus_2nd":   CONFIG["REWARD"]["BONUS_2ND"],
        "bonus_3rd":   CONFIG["REWARD"]["BONUS_3RD"],
        "bonus_last":  CONFIG["REWARD"]["BONUS_LAST"],
    }, paths["config_csv"])
    save_run_meta({"family": family, "version": version, "algo": "dqn", "deck": CONFIG["DECK_SIZE"]}, paths["run_meta_json"])

    # ---- Timing ----
    timer = TimingMeter(csv_path=paths["timings_csv"], interval=CONFIG["TIMING_INTERVAL"])
    t0 = time.perf_counter()
    BINT, BEPS = CONFIG["BENCH_INTERVAL"], CONFIG["BENCH_EPISODES"]

    for ep in range(1, CONFIG["EPISODES"] + 1):
        ep_start = time.perf_counter()
        ep_len = 0
        ep_shaping_return = 0.0
        ep_final_bonus = 0.0
        train_seconds_accum = 0.0

        ts = env.reset()
        last_p0_idx = None

        while not ts.last():

            # (1) Vorspulen bis P0 am Zug ist (Gegner spielen Heuristiken)
            while not ts.last() and ts.observations["current_player"] != 0:
                pid = ts.observations["current_player"]
                a_opp = int(opponents[pid-1](env._state))
                ts = env.step([a_opp])
                ep_len += 1

            if ts.last():
                break  # Episode vorbei

            # (2) P0-Zug: s, a, shaping reward für diesen Schritt
            legal0 = ts.observations["legal_actions"][0]
            ob_base = np.array(ts.observations["info_state"][0], dtype=np.float32)
            ob = augment_observation(ob_base, player_id=0, cfg=feat_cfg)

            if hasattr(shaper, "step_active") and shaper.step_active():
                hand_before = shaper.hand_size(ts, 0, deck_int)

            a0 = int(agent.select_action(ob, legal0))
            ts = env.step([a0])
            ep_len += 1

            if hasattr(shaper, "step_active") and shaper.step_active():
                hand_after = shaper.hand_size(ts, 0, deck_int)
                r_shape = float(shaper.step_reward(hand_before=hand_before, hand_after=hand_after))
                ep_shaping_return += r_shape
            else:
                r_shape = 0.0

            # (3) Vorspulen bis P0 wieder am Zug ist oder Terminal
            while not ts.last() and ts.observations["current_player"] != 0:
                pid = ts.observations["current_player"]
                a_opp = int(opponents[pid-1](env._state))
                ts = env.step([a_opp])
                ep_len += 1

            # (4) s' und next_legals definieren (für P0)
            if ts.last():
                ob_next = ob
                next_legals = list(range(A))
                done = True
            else:
                ob_next_base = np.array(ts.observations["info_state"][0], dtype=np.float32)
                ob_next = augment_observation(ob_next_base, player_id=0, cfg=feat_cfg)
                next_legals = ts.observations["legal_actions"][0]
                done = False

            # (5) Transition speichern & einmal trainieren
            agent.buffer.add(ob, a0, r_shape, ob_next, done, next_legal_actions=next_legals)
            last_p0_idx = len(agent.buffer.buffer) - 1

            tts = time.perf_counter()
            agent.train_step()
            train_seconds_accum += (time.perf_counter() - tts)

        # (6) Terminal: ENV-Reward + Finalbonus auf letzte P0-Transition addieren
        ep_env_score = float(ts.rewards[0])
        ep_final_bonus = float(shaper.final_bonus(ts.rewards, 0)) if hasattr(shaper, "final_bonus") else 0.0

        if last_p0_idx is not None:
            add_env = ep_env_score if (hasattr(shaper, "include_env_reward") and shaper.include_env_reward()) else 0.0
            bonus = add_env + ep_final_bonus

            if abs(bonus) > 1e-8:
                buf = agent.buffer
                old = buf.buffer[last_p0_idx]
                new = buf.Experience(old.state, old.action, float(old.reward + bonus),
                                     old.next_state, old.done, old.next_legal_mask)
                buf.buffer[last_p0_idx] = new

        # ---- Train-Metriken einsammeln
        metrics = {
            "ep_length": ep_len,
            "train_seconds": train_seconds_accum,
            "ep_env_score": ep_env_score,
            "ep_shaping_return": ep_shaping_return,
            "ep_final_bonus": ep_final_bonus,
            "epsilon": float(getattr(agent, "epsilon", np.nan)),
        }
        plotter.add_train(ep, metrics)

        # ---- Benchmark + Save in Intervallen
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
                train_seconds=train_seconds_accum,
                eval_seconds=eval_seconds,
                plot_seconds=plot_seconds,
                save_seconds=save_seconds,
                cum_seconds=cum_seconds,
            )

        # ---- Timing CSV
        ep_seconds = time.perf_counter() - ep_start
        timer.maybe_log(ep, {
            "steps": ep_len, "ep_seconds": ep_seconds,
            "train_seconds": train_seconds_accum,
            "eval_seconds": eval_seconds, "plot_seconds": plot_seconds, "save_seconds": save_seconds
        })

    total_seconds = time.perf_counter() - t0
    plotter.log("")
    plotter.log(f"Gesamtzeit: {total_seconds/3600:0.2f}h (~ {CONFIG['EPISODES']/max(total_seconds,1e-9):0.2f} eps/s)")
    plotter.log("K1 (DQN) Training abgeschlossen.")

if __name__ == "__main__":
    main()
