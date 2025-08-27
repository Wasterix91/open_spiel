# -*- coding: utf-8 -*-
# President/training/k2a1.py — PPO (K2): Vier getrennte Agents, simultanes Lernen, k1a1-Stil
# Struktur: models/k2a1/model_XX/{config.csv, plots/, models/}

import os, datetime, time, numpy as np, torch
import pyspiel
from open_spiel.python import rl_environment

from agents import ppo_agent as ppo
from utils.strategies import STRATS
from utils.fit_tensor import FeatureConfig, augment_observation, expected_feature_len, expected_input_dim
from utils.plotter import MetricsPlotter
from utils.reward_shaper import RewardShaper

from utils.load_save_common import find_next_version, prepare_run_dirs, save_config_csv, save_run_meta
from utils.load_save_a1_ppo import save_checkpoint_ppo

from utils.benchmark import run_benchmark
from utils.deck import ranks_for_deck


# ============== CONFIG  ==============
CONFIG = {
    "EPISODES":         500_000,
    "BENCH_INTERVAL":   10_000,
    "BENCH_EPISODES":   2000,
    "DECK_SIZE":        "64",  # "12" | "16" | "20" | "24" | "32" | "52" | "64"
    "SEED":             42,

    # PPO-Hyperparameter (für alle vier Agents identisch)
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

    # Feature-Toggles (analog k1a1)
    "FEATURES": {
        "USE_HISTORY": False,    # ✅ True = Variante 2 (mit Historie), False = Variante 1 (ohne)
        "SEAT_ONEHOT": False,    # optional: Sitz-One-Hot im Agent verwenden
        "PLOT_METRICS": False,   # Trainingsplots erzeugen?
        "SAVE_METRICS_TO_CSV": False,  # Trainingsmetriken persistent speichern?
    },

    # Benchmark-Gegner (wie in k1a1)
    "BENCH_OPPONENTS": ["single_only", "max_combo", "random2"],
}


# =========================== Training ===========================
def main():
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
        save_csv=True,
        verbosity=1,
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

    # Wichtig: Seat-One-Hot NICHT im augment_observation anhängen (damit kein doppeltes One-Hot)
    feat_cfg = FeatureConfig(
        num_players=num_players,
        num_ranks=num_ranks,
        add_seat_onehot=False,                          # <- immer False lassen
        include_history=CONFIG["FEATURES"]["USE_HISTORY"],
    )
    seat_id_dim = (num_players if CONFIG["FEATURES"]["SEAT_ONEHOT"] else 0)

    # ✅ Agent-Inputgrößen sauber bestimmen
    info_dim = expected_feature_len(feat_cfg)  # Basis-Features (ohne Seat-One-Hot)

    # ---- Vier Agents + Reward ----
    ppo_cfg = ppo.PPOConfig(**CONFIG["PPO"])
    agents = [
        ppo.PPOAgent(info_dim, A, seat_id_dim=seat_id_dim, config=ppo_cfg)
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
        "observation_dim": expected_input_dim(feat_cfg),  # inkl. Seat-One-Hot, falls aktiv
        "num_actions": A,
        "seat_onehot": CONFIG["FEATURES"]["SEAT_ONEHOT"],

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

            # Beobachtung für aktuellen Spieler p
            base_obs = ts.observations["info_state"][p]
            obs = augment_observation(base_obs, player_id=p, cfg=feat_cfg)

            seat_oh = None
            if CONFIG["FEATURES"]["SEAT_ONEHOT"]:
                seat_oh = np.zeros(num_players, dtype=np.float32); seat_oh[p] = 1.0

            # Handgröße vor der Aktion (nur wenn Step-Rewards aktiv)
            if shaper.step_active():
                hand_before = shaper.hand_size(ts, p, deck_int)

            # Action aus eigenem Netz des aktuellen Sitzes
            a = int(agents[p].step(obs, legal, seat_one_hot=seat_oh, player_id=p))
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
            # - Einzelplots:  "Lernkurve - K1A1 vs <gegner>"
            # - Multi/Macro:  "Lernkurve - K1A1 vs feste Heuristiken"
            title_multi = f"Lernkurve - {family.upper()} vs feste Heuristiken"
            plotter.plot_benchmark(
                filename_prefix="lernkurve",
                with_macro=True,
                family_title=family.upper(),   # für Einzelplots
                multi_title=title_multi,       # für Multi- & Macro-Plot (gleicher Titel)
            )

            if CONFIG["FEATURES"].get("PLOT_METRICS", False):
                plotter.plot_train(filename_prefix="training_metrics", separate=True)

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
