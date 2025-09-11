# -*- coding: utf-8 -*-
# k2a2.py — K2: Vier getrennte DQN-Agents, simultanes Lernen (IQL)
#
# Ausrichtung wie k1a2 / k1a1:
# - augment_observation ohne Seat-1hot; optionaler Seat-1hot wird extern angehängt
# - state_size = expected_feature_len(feat_cfg) + seat_id_dim
# - Beobachtungen basieren auf env._state.observation_tensor(p)
# - RewardShaper & Benchmark wie k1a2
# - Plot-Äquivalenz zu k1a1/k1a2: smooth_window, SAVE_METRICS_TO_CSV, ep_return_training, Komponenten-Gating, PLOT_KEYS

import os, time, datetime
import numpy as np, torch
import pyspiel
from open_spiel.python import rl_environment

from agents.dqn_agent import DQNAgent, DQNConfig
from utils.fit_tensor import FeatureConfig, augment_observation, expected_feature_len
from utils.deck import ranks_for_deck
from utils.plotter import MetricsPlotter
from utils.load_save_common import find_next_version, prepare_run_dirs, save_config_csv, save_run_meta
from utils.benchmark import run_benchmark
from utils.strategies import STRATS
from utils.reward_shaper import RewardShaper

CONFIG = {
    "EPISODES":         100_000,
    "BENCH_INTERVAL":   2_000,
    "BENCH_EPISODES":   2_000,
    "DECK_SIZE":        "16",  # "12" | "16" | "20" | "24" | "32" | "52" | "64"
    "SEED":             42,

    # DQN
    "DQN": {
        "learning_rate":     3e-4,
        "batch_size":        128,
        "gamma":             0.995,
        "epsilon_start":     1.0,
        "epsilon_end":       0.05,
        "epsilon_decay":     0.9997,
        "buffer_size":       200_000,
        "target_update_freq": 5000,
        "soft_target_tau":   0.0,
        "max_grad_norm":     1.0,
        "use_double_dqn":    True,
        "loss_huber_delta":  1.0,
    },

    # Rewards (neues System)
    # STEP_MODE : "none" | "delta_weight_only" | "hand_penalty_coeff_only" | "combined"
    # FINAL_MODE: "none" | "env_only" | "rank_only" | "both"
    "REWARD": {
        "STEP_MODE": "none",
        "DELTA_WEIGHT": 0.0,
        "HAND_PENALTY_COEFF": 0.0,
        "FINAL_MODE": "env_only",
        "BONUS_WIN": 0.0, "BONUS_2ND": 0.0, "BONUS_3RD": 0.0, "BONUS_LAST": 0.0,
    },

    # Feature-Toggles wie k1a2 / k1a1 (für Plot-Parität ergänzt)
    "FEATURES": {
        "USE_HISTORY": True,
        "SEAT_ONEHOT": False,
        "PLOT_METRICS": True,
        "SAVE_METRICS_TO_CSV": False,
        "RET_SMOOTH_WINDOW": 150,
        "PLOT_KEYS": [
            # DQN/Train:
            "epsilon_p0", "ep_length",
            # Sonderplots (aus Memory; Episoden-Return und Komponenten):
            "ep_return_raw", "ep_return_components",
            "ep_return_env", "ep_return_shaping", "ep_return_final",
            "ep_return_training",
        ],
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

    family = "k2a2"
    version = find_next_version(os.path.join(MODELS_ROOT, family), prefix="model")
    paths = prepare_run_dirs(MODELS_ROOT, family, version, prefix="model")

    plotter = MetricsPlotter(
        out_dir=paths["plots_dir"],
        benchmark_opponents=list(CONFIG["BENCH_OPPONENTS"]),
        benchmark_csv="benchmark_curves.csv",
        train_csv="training_metrics.csv",
        save_csv=CONFIG["FEATURES"]["SAVE_METRICS_TO_CSV"],
        verbosity=1,
        smooth_window=CONFIG["FEATURES"].get("RET_SMOOTH_WINDOW", 150),
    )
    plotter.log("New Training (k2a2): 4 getrennte DQN-Agents — simultanes Lernen")
    plotter.log(f"Deck_Size: {CONFIG['DECK_SIZE']}")
    plotter.log(f"Episodes: {CONFIG['EPISODES']}")
    plotter.log(f"Path: {paths['run_dir']}")

    # ---- Env ----
    game = pyspiel.load_game("president", {
        "num_players": 4, "deck_size": CONFIG["DECK_SIZE"],
        "shuffle_cards": True, "single_card_mode": False
    })
    env = rl_environment.Environment(game)
    A = env.action_spec()["num_actions"]
    num_players = game.num_players()

    # ---- Features ----
    deck_int = int(CONFIG["DECK_SIZE"])
    num_ranks = ranks_for_deck(deck_int)
    feat_cfg = FeatureConfig(
        num_players=num_players,
        num_ranks=num_ranks,
        add_seat_onehot=False,
        include_history=CONFIG["FEATURES"]["USE_HISTORY"]
    )

    seat_id_dim = (num_players if CONFIG["FEATURES"]["SEAT_ONEHOT"] else 0)
    state_size = expected_feature_len(feat_cfg) + seat_id_dim

    # ---- Agents & Reward ----
    agents = [DQNAgent(state_size, A, DQNConfig(**CONFIG["DQN"])) for _ in range(num_players)]
    shaper = RewardShaper(CONFIG["REWARD"])

    # ---- Config speichern ----
    save_config_csv({
        "script": family, "version": version,
        "deck_size": CONFIG["DECK_SIZE"], "num_ranks": num_ranks,
        "use_history": CONFIG["FEATURES"]["USE_HISTORY"],
        "agent_type": "DQNx4_simultaneous", "num_episodes": CONFIG["EPISODES"],
        "bench_interval": CONFIG["BENCH_INTERVAL"], "bench_episodes": CONFIG["BENCH_EPISODES"],
        "observation_dim": state_size, "num_actions": A,
        "seat_onehot": CONFIG["FEATURES"]["SEAT_ONEHOT"],
        "ret_smooth_window": CONFIG["FEATURES"].get("RET_SMOOTH_WINDOW", 150),
        "models_dir": paths["weights_dir"], "plots_dir": paths["plots_dir"],
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "step_mode": shaper.step_mode, "delta_weight": shaper.dw,
        "hand_penalty_coeff": shaper.hp, "final_mode": shaper.final_mode,
        "bonus_win": CONFIG["REWARD"]["BONUS_WIN"], "bonus_2nd": CONFIG["REWARD"]["BONUS_2ND"],
        "bonus_3rd": CONFIG["REWARD"]["BONUS_3RD"], "bonus_last": CONFIG["REWARD"]["BONUS_LAST"],
    }, paths["config_csv"])
    save_run_meta({"family": family, "version": version, "algo": "dqn_x4_simul", "deck": CONFIG["DECK_SIZE"]},
                  paths["run_meta_json"])

    t0 = time.perf_counter()
    BINT, BEPS = CONFIG["BENCH_INTERVAL"], CONFIG["BENCH_EPISODES"]

    # Sammeln nur, wenn wir es brauchen (CSV oder Plot)
    collect_metrics = (
        CONFIG["FEATURES"].get("SAVE_METRICS_TO_CSV", False)
        or CONFIG["FEATURES"].get("PLOT_METRICS", False)
    )

    for ep in range(1, CONFIG["EPISODES"] + 1):
        ep_start = time.perf_counter()
        ep_len = 0
        ep_shape = [0.0 for _ in range(num_players)]
        ep_final_bonus = [0.0 for _ in range(num_players)]

        ts = env.reset()
        pending = {p: None for p in range(num_players)}   # {"s":..., "a":..., "r":...}
        last_idx = {p: None for p in range(num_players)}  # Index letzte Transition je Sitz

        while not ts.last():
            p = ts.observations["current_player"]
            legal = ts.observations["legal_actions"][p]

            # (A) Falls p wieder dran ist: offene Transition schließen (decision-to-decision)
            if pending[p] is not None:
                s_now_base = np.asarray(env._state.observation_tensor(p), dtype=np.float32)
                s_now = augment_observation(s_now_base, player_id=p, cfg=feat_cfg)
                s_now = _with_seat_onehot(s_now, p, num_players, CONFIG["FEATURES"]["SEAT_ONEHOT"])
                rec = pending[p]; pending[p] = None

                agents[p].buffer.add(rec["s"], rec["a"], rec["r"], s_now, False,
                                     next_legal_actions=legal)
                last_idx[p] = len(agents[p].buffer.buffer) - 1
                agents[p].train_step()

            # (B) aktuelle Beobachtung für p
            s_base = np.asarray(env._state.observation_tensor(p), dtype=np.float32)
            s = augment_observation(s_base, player_id=p, cfg=feat_cfg)
            s = _with_seat_onehot(s, p, num_players, CONFIG["FEATURES"]["SEAT_ONEHOT"])

            # (C) optionaler Step-Reward
            r_step = 0.0
            if shaper.step_active():
                hand_before = shaper.hand_size(ts, p, deck_int)

            # (D) Aktion wählen & ausführen
            a = int(agents[p].select_action(s, legal))
            ts_next = env.step([a]); ep_len += 1

            if shaper.step_active():
                hand_after = shaper.hand_size(ts_next, p, deck_int)
                r_step = float(shaper.step_reward(hand_before=hand_before, hand_after=hand_after))
                ep_shape[p] += r_step

            # (E) Pending anlegen
            assert pending[p] is None
            pending[p] = {"s": s, "a": a, "r": r_step}
            ts = ts_next

        # ---- Episodenende: alle offenen Pendings finalisieren (done=True) ----
        for p in range(num_players):
            if pending[p] is not None:
                rec = pending[p]; pending[p] = None
                s_next = rec["s"]
                agents[p].buffer.add(rec["s"], rec["a"], rec["r"], s_next, True,
                                     next_legal_actions=list(range(A)))
                last_idx[p] = len(agents[p].buffer.buffer) - 1
                agents[p].train_step()

        # ---- Finals/Bonus auf letzte Transition buchen ----
        finals = env._state.returns()
        for p in range(num_players):
            li = last_idx[p]
            if li is None:
                continue
            add_env = float(finals[p]) if shaper.include_env_reward() else 0.0
            add_bonus = float(shaper.final_bonus(finals, p))
            ep_final_bonus[p] = add_bonus

            buf = agents[p].buffer
            old = buf.buffer[li]
            buf.buffer[li] = buf.Experience(
                old.state, old.action, float(old.reward + add_env + add_bonus),
                old.next_state, old.done, old.next_legal_mask
            )

        # ---- Metriken (P0-zentriert + Aggregat) ----
        metrics = {
            "ep_length":             ep_len,
            "ep_env_return_p0":      float(finals[0]),
            "ep_shaping_return_p0":  ep_shape[0],
            "ep_final_bonus_p0":     ep_final_bonus[0],
            "ep_env_return_avg":     float(np.mean(finals)),
            "ep_shaping_return_avg": float(np.mean(ep_shape)),
            "ep_final_bonus_avg":    float(np.mean(ep_final_bonus)),
            "epsilon_p0":            float(agents[0].epsilon),
        }

        # --- Trainingsäquivalenter Return (P0) exakt wie k1a1/k1a2 ---
        env_part_p0     = float(finals[0]) if shaper.include_env_reward() else 0.0
        shaping_part_p0 = float(ep_shape[0]) if shaper.step_active() else 0.0
        ep_return_training_p0 = shaping_part_p0 + env_part_p0 + float(shaper.final_bonus(finals, 0))
        metrics["ep_return_training"] = ep_return_training_p0

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

        # Trainingsmetriken speichern/merken (Parität zu k1a1/k1a2)
        if collect_metrics:
            if CONFIG["FEATURES"].get("SAVE_METRICS_TO_CSV", False):
                plotter.add_train(ep, metrics)
            else:
                plotter.train_rows.append({"episode": int(ep), **metrics})
                if plotter.train_keys is None:
                    plotter.train_keys = ["episode"] + list(metrics.keys())

        # ---- Benchmark (Agent 0) & Save ----
        eval_seconds = plot_seconds = save_seconds = 0.0
        if ep % BINT == 0:
            evs = time.perf_counter()
            per_opponent = run_benchmark(
                game=game,
                agent=agents[0],
                opponents_dict=STRATS,
                opponent_names=CONFIG["BENCH_OPPONENTS"],
                episodes=BEPS,
                feat_cfg=feat_cfg,
                num_actions=A,
            )
            plotter.log_bench_summary(ep, per_opponent)
            eval_seconds = time.perf_counter() - evs

            ps = time.perf_counter()
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
                plotter.plot_train(
                    include_keys=CONFIG["FEATURES"].get("PLOT_KEYS"),
                    separate=True,
                )
            plot_seconds = time.perf_counter() - ps

            ss = time.perf_counter()
            for seat_id, ag in enumerate(agents):
                tag = f"{family}_model_{version}_agent_p{seat_id}_ep{ep:07d}"
                ag.save(os.path.join(paths["weights_dir"], tag))
            save_seconds = time.perf_counter() - ss

            plotter.log_timing(
                ep,
                ep_seconds=(time.perf_counter() - ep_start),
                train_seconds=0.0,
                eval_seconds=eval_seconds,
                plot_seconds=plot_seconds,
                save_seconds=save_seconds,
                cum_seconds=(time.perf_counter() - t0),
            )

    total_seconds = time.perf_counter() - t0
    plotter.log("")
    plotter.log(f"Gesamtzeit: {total_seconds/3600:0.2f}h (~ {CONFIG['EPISODES']/max(total_seconds,1e-9):0.2f} eps/s)")
    plotter.log(f"{family}, Multi Agent RL (IQL). Training abgeschlossen.")
    plotter.log(f"Path: {paths['run_dir']}")

if __name__ == "__main__":
    main()
