# President/training/k4a2_test.py
# -*- coding: utf-8 -*-
# DQN (K4): Shared Policy (1 Netz für alle Seats) + Benchmark vs Heuristiken

import os, time, datetime, numpy as np, torch
import pyspiel
from open_spiel.python import rl_environment

from agents import dqn_agent as dqn
from utils.fit_tensor import FeatureConfig, augment_observation
from utils.strategies import STRATS
from utils.plotter import MetricsPlotter
from utils.timing import TimingMeter
from utils.reward_shaper import RewardShaper
from utils.benchmark import run_benchmark
from utils.deck import ranks_for_deck
from utils.load_save_common import find_next_version, prepare_run_dirs, save_config_csv, save_run_meta

# ============== CONFIG ==============
CONFIG = {
    "EPISODES":         1000,
    "BENCH_INTERVAL":   200,
    "BENCH_EPISODES":   500,
    "TIMING_INTERVAL":  250,
    # Erlaubt: "12" | "16" | "20" | "24" | "32" | "52" | "64"
    "DECK_SIZE":        "16",
    "SEED":             42,

    # DQNConfig-kompatibel (siehe dqn_agent.py)
    "DQN": {
        "learning_rate": 3e-4,
        "batch_size": 128,
        "gamma": 0.995,
        "buffer_size": 200_000,
        "target_update_freq": 5000,
        "soft_target_tau": 0.0,
        "max_grad_norm": 1.0,
        "n_step": 3,
        "per_alpha": 0.6,
        "per_beta_start": 0.4,
        "per_beta_frames": 1_000_000,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_decay_frames": 500_000,
        "loss_huber_delta": 1.0,
        "dueling": True,
        "device": "cpu",
    },

    # ======= Rewards (neues System wie k1a1/k3a1/k4a1) =======
    "REWARD": {
        "STEP_MODE": "delta_weight_only",
        "DELTA_WEIGHT": 1.0,
        "HAND_PENALTY_COEFF": 0.0,

        "FINAL_MODE": "env_only",
        "BONUS_WIN": 0.0, "BONUS_2ND": 0.0, "BONUS_3RD": 0.0, "BONUS_LAST": 0.0,
    },

    # Features (Shared Policy: Seat-OneHot sinnvoll → True)
    "FEATURES": { "NORMALIZE": False, "SEAT_ONEHOT": True },

    # Gegner: Selfplay im Training (ein Netz), Benchmark separat
    "OPPONENTS": ["max_combo", "max_combo", "max_combo"],
    "BENCH_OPPONENTS": ["single_only", "max_combo", "random2"],
}

# =========================== Training ===========================
def main():
    np.random.seed(CONFIG["SEED"]); torch.manual_seed(CONFIG["SEED"])
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_ROOT = os.path.join(ROOT, "models")

    # ---- Verzeichnisse (k1-Stil) ----
    family = "k4a2"
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
    plotter.log("New Training (k4a2): Shared-Policy DQN")
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
        add_seat_onehot=CONFIG["FEATURES"]["SEAT_ONEHOT"],
        normalize=CONFIG["FEATURES"]["NORMALIZE"],
    )
    state_size = info_dim + (num_players if CONFIG["FEATURES"]["SEAT_ONEHOT"] else 0)

    # ---- Agent / Shaper ----
    dqn_cfg = dqn.DQNConfig(**CONFIG["DQN"])
    agent = dqn.DQNAgent(state_size=state_size, num_actions=A, cfg=dqn_cfg)
    shaper = RewardShaper(CONFIG["REWARD"])

    # ---- Config & Meta speichern ----
    save_config_csv({
        "script": family, "version": version,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent_type": "DQN_shared", "num_episodes": CONFIG["EPISODES"],
        "bench_interval": CONFIG["BENCH_INTERVAL"], "bench_episodes": CONFIG["BENCH_EPISODES"],
        "deck_size": CONFIG["DECK_SIZE"], "num_ranks": num_ranks,
        "observation_dim": state_size, "num_actions": A,
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
    save_run_meta({"family": family, "version": version, "algo": "dqn_shared", "deck": CONFIG["DECK_SIZE"]},
                  paths["run_meta_json"])

    # ---- Timing ----
    timer = TimingMeter(csv_path=paths["timings_csv"], interval=CONFIG["TIMING_INTERVAL"])
    t0 = time.perf_counter()
    BINT, BEPS = CONFIG["BENCH_INTERVAL"], CONFIG["BENCH_EPISODES"]

    # ================= Loop =================
    for ep in range(1, CONFIG["EPISODES"] + 1):
        ep_start = time.perf_counter()
        ep_len = 0
        ep_shaping_returns = [0.0 for _ in range(num_players)]

        ts = env.reset()
        last_idx = {p: None for p in range(num_players)}  # Index der letzten Transition je Sitz

        while not ts.last():
            p = ts.observations["current_player"]
            legal = ts.observations["legal_actions"][p]

            # Info-State + ggf. Seat-OneHot (passt zu state_size)
            base_obs = ts.observations["info_state"][p]
            obs = augment_observation(base_obs, player_id=p, cfg=feat_cfg)

            # Step-Shaping vorbereiten
            if shaper.step_active():
                hand_before = shaper.hand_size(ts, p, deck_int)

            # Shared Policy spielt auf allen Seats (ε-greedy bei select_action)
            a = int(agent.select_action(obs, legal))
            ts_next = env.step([a])
            ep_len += 1

            # Next-Obs & Legal-Maske (Maske für *nächsten* current_player!)
            if not ts_next.last():
                base_next = ts_next.observations["info_state"][p]
                obs_next = augment_observation(base_next, player_id=p, cfg=feat_cfg)
                npid = ts_next.observations["current_player"]
                next_legals = ts_next.observations["legal_actions"][npid]
            else:
                obs_next = obs
                next_legals = None  # None ⇒ im Trainer volle Maske

            # Step-Reward
            r = 0.0
            if shaper.step_active():
                hand_after = shaper.hand_size(ts_next, p, deck_int)
                r = float(shaper.step_reward(hand_before=hand_before, hand_after=hand_after))
                ep_shaping_returns[p] += r

            # Replay speichern + Online-Train
            agent.store(state=obs, action=int(a), reward=float(r),
                        next_state=obs_next, done=bool(ts_next.last()),
                        next_legal_actions=next_legals)
            last_idx[p] = len(agent.buffer.buffer) - 1
            agent.train_step()

            ts = ts_next

        # ===== Episodenende: Final-Rewards/Bonis auf letzte Transition je Sitz =====
        finals = [float(ts.rewards[i]) for i in range(num_players)]
        for p in range(num_players):
            li = last_idx[p]
            if li is None:
                continue
            bonus = (finals[p] if shaper.include_env_reward() else 0.0) + float(shaper.final_bonus(finals, p))
            if abs(bonus) > 1e-8:
                buf = agent.buffer
                old = buf.buffer[li]
                buf.buffer[li] = buf.Exp(old.s, old.a, float(old.r + bonus), old.ns, old.done, old.next_mask)

        # ===== Trainingsmetriken loggen =====
        avg_env_return   = float(np.mean(finals))
        avg_shape_return = float(np.mean(ep_shaping_returns))
        avg_final_bonus  = float(np.mean([shaper.final_bonus(finals, i) for i in range(num_players)]))
        plotter.add_train(ep, {
            "ep_length":             ep_len,
            "ep_env_return_p0":      finals[0],
            "ep_shaping_return_p0":  ep_shaping_returns[0],
            "ep_final_bonus_p0":     float(shaper.final_bonus(finals, 0)),
            "ep_env_return_avg":     avg_env_return,
            "ep_shaping_return_avg": avg_shape_return,
            "ep_final_bonus_avg":    avg_final_bonus,
            "epsilon":               getattr(agent, "_epsilon", lambda: np.nan)(),
        })

        # ===== Benchmark & Save =====
        eval_seconds = plot_seconds = save_seconds = 0.0
        if BINT > 0 and (ep % BINT == 0):
            ev_start = time.perf_counter()
            per_opponent = run_benchmark(
                game=game,
                agent=agent,                      # Shared Policy (Seat 0 wird evaluiert)
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
            plotter.plot_benchmark(filename_prefix="lernkurve", with_macro=True)
            plotter.plot_train(filename_prefix="training_metrics", separate=True)
            plot_seconds = time.perf_counter() - plot_start

            save_start = time.perf_counter()
            # Ein Netz – für Kompatibilität 4x als p0..p3 speichern
            tag = f"{family}_model_{version}_agent_p{{seat}}_ep{ep:07d}"
            for seat in range(num_players):
                base = os.path.join(paths["weights_dir"], tag.format(seat=seat))
                agent.save(base)
            save_seconds = time.perf_counter() - save_start

            cum_seconds = time.perf_counter() - t0
            plotter.log_timing(
                ep,
                ep_seconds=(time.perf_counter() - ep_start),
                train_seconds=0.0,               # Online-Train oben
                eval_seconds=eval_seconds,
                plot_seconds=plot_seconds,
                save_seconds=save_seconds,
                cum_seconds=cum_seconds,
            )

        # ===== Timing CSV (streaming) =====
        ep_seconds = time.perf_counter() - ep_start
        timer.maybe_log(ep, {
            "steps": ep_len,
            "ep_seconds": ep_seconds,
            "train_seconds": 0.0,
            "eval_seconds": eval_seconds,
            "plot_seconds": plot_seconds,
            "save_seconds": save_seconds,
        })

    # Ende
    total_seconds = time.perf_counter() - t0
    plotter.log("")
    plotter.log(f"Gesamtzeit: {total_seconds/3600:0.2f}h (~ {CONFIG['EPISODES']/max(total_seconds,1e-9):0.2f} eps/s)")
    plotter.log("K4 (Shared-Policy DQN) Training abgeschlossen.")

if __name__ == "__main__":
    main()
