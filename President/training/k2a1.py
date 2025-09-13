# -*- coding: utf-8 -*-
# President/training/k2a1.py — PPO (K2): Vier getrennte Agents, simultanes Lernen, k1a1-Stil
# Struktur: models/k2a1/model_XX/{config.csv, plots/, models/}

import os, datetime, time, numpy as np, torch
import pyspiel
from open_spiel.python import rl_environment

from agents import ppo_agent as ppo
from utils.strategies import STRATS
from utils.fit_tensor import FeatureConfig, augment_observation
from utils.plotter import MetricsPlotter
from utils.reward_shaper import RewardShaper

from utils.load_save_common import find_next_version, prepare_run_dirs, save_config_csv, save_run_meta
from utils.load_save_a1_ppo import save_checkpoint_ppo

from utils.benchmark import run_benchmark
from utils.deck import ranks_for_deck


# ============== CONFIG  ==============
CONFIG = {
    "EPISODES":         100_000,
    "BENCH_INTERVAL":   2_000,
    "BENCH_EPISODES":   2_000,
    "DECK_SIZE":        "16",  # "12" | "16" | "20" | "24" | "32" | "52" | "64"
    "SEED":             42,

    # PPO-Hyperparameter (für alle vier Agents identisch)
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
    # FINAL_MODE: "none" | "env_only" | "rank_only" | "both"
    "REWARD": {
        "STEP_MODE": "none",
        "DELTA_WEIGHT": 0.0,
        "HAND_PENALTY_COEFF": 0.0,

        "FINAL_MODE": "env_only",
        "BONUS_WIN": 0.0, "BONUS_2ND": 0.0, "BONUS_3RD": 0.0, "BONUS_LAST": 0.0,
    },

    "FEATURES": {
        "USE_HISTORY": True,
        "SEAT_ONEHOT": False,
        "NORMALIZE": False,
        "DEBUG_FEATURES": True,
        "PLOT_METRICS": True,
        "SAVE_METRICS_TO_CSV": False,
        "RET_SMOOTH_WINDOW": 150,   # Fenstergröße für Moving Average der Rewards
        "PLOT_KEYS": [              # steuert plot_train(); mögliche Keys:
            # PPO-Metriken:
            #   reward_mean, reward_std, return_mean,
            #   adv_mean_raw, adv_std_raw,
            #   policy_loss, value_loss,
            #   entropy, approx_kl, clip_frac
            # Trainings-/Umgebungsmetriken:
            #   train_seconds, ep_env_score, ep_shaping_return,
            #   ep_final_bonus, ep_length
            # Sonderplots (aus Memory, nicht aus metrics):
            #   ep_return_raw, ep_return_components,
            #   ep_return_env, ep_return_shaping, ep_return_final
            "return_mean",
            "reward_mean",
            "entropy",
            "approx_kl",
            "clip_frac",
            "policy_loss",
            "value_loss",
            "ep_return_raw",
            "ep_return_components",
            "ep_return_env",         # Einzelplot env_score
            "ep_return_shaping",     # Einzelplot shaping
            "ep_return_final",       # Einzelplot final_bonus
            "ep_return_training",
        ],
    },


    # Benchmark-Gegner (wie in k1a1)
    "BENCH_OPPONENTS": ["single_only", "max_combo", "random2"],
}


# =========================== Training ===========================
def main():

        # ================== DEBUG: Feature-Vektor Check ==================
    if CONFIG["FEATURES"].get("DEBUG_FEATURES", False):  # nur temporär!
        for deck_size in [16, 64]:
            for use_history in [True, False]:
                for seat_onehot in [True, False]:
                    for normalize in [True, False]:
                        CONFIG["DECK_SIZE"] = str(deck_size)
                        CONFIG["FEATURES"]["USE_HISTORY"] = use_history
                        CONFIG["FEATURES"]["SEAT_ONEHOT"] = seat_onehot
                        CONFIG["FEATURES"]["NORMALIZE"] = normalize

                        # ---- Env / FeatureConfig bauen ----
                        game = pyspiel.load_game("president", {
                            "num_players": 4,
                            "deck_size": CONFIG["DECK_SIZE"],
                            "shuffle_cards": True,
                            "single_card_mode": False,
                        })
                        env = rl_environment.Environment(game)
                        num_players = game.num_players()
                        deck_int = int(CONFIG["DECK_SIZE"])
                        num_ranks = ranks_for_deck(deck_int)
                        feat_cfg = FeatureConfig(
                            num_players=num_players,
                            num_ranks=num_ranks,
                            add_seat_onehot=CONFIG["FEATURES"]["SEAT_ONEHOT"],
                            include_history=CONFIG["FEATURES"]["USE_HISTORY"],
                            normalize=CONFIG["FEATURES"]["NORMALIZE"],
                            deck_size=deck_int,
                        )
                        state_size = feat_cfg.input_dim()  # KEIN zusätzliches + seat_id_dim

                        ts = env.reset()
                        p = ts.observations["current_player"]
                        base_obs = ts.observations["info_state"][p]
                        obs = augment_observation(base_obs, player_id=p, cfg=feat_cfg)



                        print("Deck Size:  ", CONFIG["DECK_SIZE"])
                        print("Use History:", CONFIG["FEATURES"]["USE_HISTORY"])
                        print("Seat 1-Hot: ", CONFIG["FEATURES"]["SEAT_ONEHOT"])
                        print("Normalize:  ", CONFIG["FEATURES"]["NORMALIZE"])
                        print(f"Tensor length={len(obs)}  (model input_dim={feat_cfg.input_dim()})")
                        print(f"Tensor: {np.round(obs, 3)}")
                        print("-" * 60)

        return
    # ================== DEBUG Ende ==================

    np.random.seed(CONFIG["SEED"]); torch.manual_seed(CONFIG["SEED"])
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_ROOT = os.path.join(ROOT, "models")

    # ---- Lauf-Verzeichnisse ----
    family = "k2a1"  # 4 gleichzeitige Agents im k1a1-Stil
    family_dir = os.path.join(MODELS_ROOT, family)
    version = find_next_version(family_dir, prefix="model")
    paths = prepare_run_dirs(MODELS_ROOT, family, version, prefix="model")

    # ---- Plotter ----
    plotter = MetricsPlotter(
        out_dir=paths["plots_dir"],
        benchmark_opponents=list(CONFIG["BENCH_OPPONENTS"]),
        benchmark_csv="benchmark_curves.csv",
        train_csv="training_metrics.csv",
        save_csv=CONFIG["FEATURES"]["SAVE_METRICS_TO_CSV"],
        verbosity=1,
        smooth_window=CONFIG["FEATURES"].get("RET_SMOOTH_WINDOW", 150),
    )

    plotter.log("New Training (k2a1): 4 getrennte PPO-Agents — simultanes Lernen")
    plotter.log(f"Deck_Size: {CONFIG['DECK_SIZE']}")
    plotter.log(f"Episodes: {CONFIG['EPISODES']}")
    plotter.log(f"Path: {paths['run_dir']}")

    # ---- Game/Env ----
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
        add_seat_onehot=CONFIG["FEATURES"]["SEAT_ONEHOT"],
        include_history=CONFIG["FEATURES"]["USE_HISTORY"],
        normalize=CONFIG["FEATURES"].get("NORMALIZE", False),
        deck_size=deck_int,
    )
    state_size = feat_cfg.input_dim()


    # ---- Vier Agents + Reward ----
    ppo_cfg = ppo.PPOConfig(**CONFIG["PPO"])
    agents = [
        ppo.PPOAgent(state_size, A, config=ppo_cfg)
        for _ in range(num_players)
    ]

    shaper = RewardShaper(CONFIG["REWARD"])

    # ---- Run-Metadaten & config ----
    save_config_csv({
        "script": family, "version": version,
        "deck_size": CONFIG["DECK_SIZE"], "num_ranks": num_ranks,
        "use_history": CONFIG["FEATURES"]["USE_HISTORY"],
        # Reward-Setup (neues System)
        "step_mode": shaper.step_mode,
        "delta_weight": shaper.dw,
        "hand_penalty_coeff": shaper.hp,
        "final_mode": shaper.final_mode,
        "bonus_win":   CONFIG["REWARD"]["BONUS_WIN"],
        "bonus_2nd":   CONFIG["REWARD"]["BONUS_2ND"],
        "bonus_3rd":   CONFIG["REWARD"]["BONUS_3RD"],
        "bonus_last":  CONFIG["REWARD"]["BONUS_LAST"],

        "agent_type": "PPOx4_simultaneous", "num_episodes": CONFIG["EPISODES"],
        "bench_interval": CONFIG["BENCH_INTERVAL"], "bench_episodes": CONFIG["BENCH_EPISODES"],
        "observation_dim": feat_cfg.input_dim(),
        "num_actions": A,
        "seat_onehot": CONFIG["FEATURES"]["SEAT_ONEHOT"],

        "ret_smooth_window": CONFIG["FEATURES"].get("RET_SMOOTH_WINDOW", 150),

        "opponents": ",".join(CONFIG["BENCH_OPPONENTS"]),
        "models_dir": paths["weights_dir"], "plots_dir": paths["plots_dir"],
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }, paths["config_csv"])

    save_run_meta({
        "family": family, "version": version,
        "algo": "ppo_x4_simul", "deck": CONFIG["DECK_SIZE"]
    }, paths["run_meta_json"])

    t0 = time.perf_counter()
    BINT, BEPS = CONFIG["BENCH_INTERVAL"], CONFIG["BENCH_EPISODES"]

    # Metrik-Collection wie in k1a1: optional speichern/plotten
    collect_metrics = (
        CONFIG["FEATURES"].get("SAVE_METRICS_TO_CSV", False)
        or CONFIG["FEATURES"].get("PLOT_METRICS", False)
    )

    # ---- Training loop ----
    for ep in range(1, CONFIG["EPISODES"] + 1):
        ep_start = time.perf_counter()
        ep_len = 0
        ep_shaping_returns = [0.0 for _ in range(num_players)]  # pro Agent

        ts = env.reset()
        last_idx = {p: None for p in range(num_players)}  # Index der letzten Transition je Agent

        while not ts.last():
            p = ts.observations["current_player"]
            legal = ts.observations["legal_actions"][p]

            # Beobachtung für aktuellen Spieler p (Seat-One-Hot wird – falls aktiv –
            # bereits in augment_observation/feat_cfg berücksichtigt)
            base_obs = ts.observations["info_state"][p]
            obs = augment_observation(base_obs, player_id=p, cfg=feat_cfg)


            # Handgröße vor der Aktion (nur wenn Step-Rewards aktiv)
            if shaper.step_active():
                hand_before = shaper.hand_size(ts, p, deck_int)

            # Action aus eigenem Netz des aktuellen Sitzes
            a = int(agents[p].step(obs, legal, player_id=p))

            last_idx[p] = len(agents[p]._buffer.states) - 1

            # Schritt in der Env
            ts_next = env.step([a])
            ep_len += 1

            # Step-Shaping direkt auf die zuletzt gespeicherte Transition des aktiven Agents
            if shaper.step_active():
                hand_after = shaper.hand_size(ts_next, p, deck_int)
                r = shaper.step_reward(hand_before=hand_before, hand_after=hand_after)
                ep_shaping_returns[p] += float(r)
                agents[p].post_step(r, done=ts_next.last())

            ts = ts_next

        # ===== Episodenende: Final-Rewards & done-Markierung je Agent =====
        finals = [float(ts.rewards[i]) for i in range(num_players)]

        for p in range(num_players):
            li = last_idx[p]
            if li is None:
                continue  # (sollte praktisch nie passieren)

            # ENV-Return nur, wenn der Shaper es vorsieht
            if shaper.include_env_reward():
                agents[p]._buffer.rewards[li] += finals[p]

            # Benutzerdefinierter Platzierungsbonus
            agents[p]._buffer.rewards[li] += float(shaper.final_bonus(finals, p))

            # Terminal markieren (wichtig für GAE)
            agents[p]._buffer.dones[li] = True

        # ===== Training (ein Update pro Agent am Episodenende) =====
        train_seconds_sum = 0.0
        train_seconds_p0 = 0.0
        for i in range(num_players):
            t_start = time.perf_counter()
            _metrics = agents[i].train()  # kann None sein
            dt = time.perf_counter() - t_start
            train_seconds_sum += dt
            if i == 0:
                train_seconds_p0 = dt

        # Trainingsmetriken (aggregiert + p0-spezifisch für Plot-Kompatibilität)
        avg_env_return   = float(np.mean(finals))
        avg_shape_return = float(np.mean(ep_shaping_returns))
        avg_final_bonus  = float(np.mean([shaper.final_bonus(finals, i) for i in range(num_players)]))
        train_metrics = {
            "train_seconds_total":   train_seconds_sum,
            "train_seconds_p0":      train_seconds_p0,
            "ep_env_return_p0":      finals[0],
            "ep_shaping_return_p0":  ep_shaping_returns[0],
            "ep_final_bonus_p0":     float(shaper.final_bonus(finals, 0)),
            "ep_env_return_avg":     avg_env_return,
            "ep_shaping_return_avg": avg_shape_return,
            "ep_final_bonus_avg":    avg_final_bonus,
            "ep_length":             ep_len,
        }

        # --- Trainingsäquivalenter Return (P0) exakt wie in k1a1 ---
        env_part_p0     = finals[0] if shaper.include_env_reward() else 0.0
        shaping_part_p0 = ep_shaping_returns[0] if shaper.step_active() else 0.0
        ep_return_training_p0 = shaping_part_p0 + env_part_p0 + float(shaper.final_bonus(finals, 0))
        # In Metrics mitschreiben (damit plot_train() dieselben Keys findet)
        train_metrics["ep_return_training"] = ep_return_training_p0

        # Sonderplot: Episoden-Return (P0) + Komponenten (mit Gating)
        plotter.add_ep_returns(
            global_episode=ep,
            ep_returns=[ep_return_training_p0],
            components={
                "env_score":   [env_part_p0],      # 0.0, wenn nicht aktiv
                "shaping":     [shaping_part_p0],  # 0.0, wenn nicht aktiv
                "final_bonus": [float(shaper.final_bonus(finals, 0))],
            },
        )

        # Speicherung/Buffering analog k1a1
        if collect_metrics:
            if CONFIG["FEATURES"].get("SAVE_METRICS_TO_CSV", False):
                plotter.add_train(ep, train_metrics)
            else:
                plotter.train_rows.append({"episode": int(ep), **train_metrics})
                if plotter.train_keys is None:
                    plotter.train_keys = ["episode"] + list(train_metrics.keys())

        # ---- Benchmark (nur Agent von Player 0, wie k1a1) ----
        eval_seconds = 0.0
        plot_seconds = 0.0
        save_seconds = 0.0

        if ep % BINT == 0:
            ev_start = time.perf_counter()
            per_opponent = run_benchmark(
                game=game,
                agent=agents[0],                    # nur Player-0-Policy wird evaluiert
                opponents_dict=STRATS,
                opponent_names=CONFIG["BENCH_OPPONENTS"],
                episodes=BEPS,
                feat_cfg=feat_cfg,
                num_actions=A,
            )
            plotter.log_bench_summary(ep, per_opponent)
            eval_seconds = time.perf_counter() - ev_start

            # Plots & CSV
            plot_start = time.perf_counter()
            plotter.add_benchmark(ep, per_opponent)
            plotter.plot_benchmark_rewards()
            plotter.plot_places_latest()

            # Einheitliche Titel:
            # - Einzelplots:  "Lernkurve - K2A1 vs <gegner>"
            # - Multi/Macro:  "Lernkurve - K2A1 vs feste Heuristiken"
            title_multi = f"Lernkurve - {family.upper()} vs feste Heuristiken"
            plotter.plot_benchmark(
                filename_prefix="lernkurve",
                with_macro=True,
                family_title=family.upper(),   # für Einzelplots
                multi_title=title_multi,       # für Multi- & Macro-Plot (gleicher Titel)
            )

            if CONFIG["FEATURES"].get("PLOT_METRICS", False):
                plotter.plot_train(
                    include_keys=CONFIG["FEATURES"].get("PLOT_KEYS"),
                    separate=True,
                )

            plot_seconds = time.perf_counter() - plot_start

            # Save weights (alle vier Agents)
            save_start = time.perf_counter()
            for seat_id, ag in enumerate(agents):
                tag = f"{family}_model_{version}_agent_p{seat_id}_ep{ep:07d}"
                save_checkpoint_ppo(ag, paths["weights_dir"], tag)
            save_seconds = time.perf_counter() - save_start

            # Kompakt-Timing
            cum_seconds = time.perf_counter() - t0
            plotter.log_timing(
                ep,
                ep_seconds=(time.perf_counter() - ep_start),
                train_seconds=train_seconds_p0,     # p0 als Referenz
                eval_seconds=eval_seconds,
                plot_seconds=plot_seconds,
                save_seconds=save_seconds,
                cum_seconds=cum_seconds,
            )

    total_seconds = time.perf_counter() - t0
    plotter.log("")
    plotter.log(f"Gesamtzeit: {total_seconds/3600:0.2f}h (~ {CONFIG['EPISODES']/max(total_seconds,1e-9):0.2f} eps/s)")
    plotter.log(f"{family}, Multi Agent RL (IQL). Training abgeschlossen.")
    plotter.log(f"Path: {paths['run_dir']}")



if __name__ == "__main__":
    main()
