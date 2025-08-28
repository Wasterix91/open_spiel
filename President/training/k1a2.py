# -*- coding: utf-8 -*-
# k1a2.py — K1: Single-Agent DQN vs Heuristiken
# Training: P0→P0-Transitions
#
# Kombi:
# - state_size wie im ursprünglichen, funktionierenden Skript (observation_tensor_shape + optional Seat-1hot)
# - restliche Logik aus der neuen Version (info_state + augment_observation ohne Seat-1hot, optionales Seat-1hot extern)

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
    "EPISODES":         500_000,
    "BENCH_INTERVAL":   10_000,
    "BENCH_EPISODES":   2000,
    "DECK_SIZE":        "64",  # "12" | "16" | "20" | "24" | "32" | "52" | "64"
    "SEED":             42,

    # Training-Gegner (Heuristiken)
    "OPPONENTS":        ["max_combo", "max_combo", "max_combo"],

    # DQN (kompatibel zu agents/dqn_agent.DQNConfig)
    "DQN": {
        "learning_rate":     3e-4,
        "batch_size":        128,
        "gamma":             0.995,
        "epsilon_start":     1.0,
        "epsilon_end":       0.05,
        "epsilon_decay":     0.9997,   # multiplikativ pro train_step
        "buffer_size":       200_000,
        "target_update_freq": 5000,
        "soft_target_tau":   0.0,      # z.B. 0.005 für Polyak
        "max_grad_norm":     1.0,
        "use_double_dqn":    True,
        "loss_huber_delta":  1.0,
    },

    # ======= Rewards (neues System) =======
    # STEP_MODE : "none" | "delta_weight_only" | "hand_penalty_coeff_only" | "combined"
    # FINAL_MODE: "none" | "env_only" | "rank_bonus" | "both"
    "REWARD": {
        "STEP_MODE": "none",
        "DELTA_WEIGHT": 0.0,
        "HAND_PENALTY_COEFF": 0.0,

        "FINAL_MODE": "env_only",
        "BONUS_WIN": 0.0, "BONUS_2ND": 0.0, "BONUS_3RD": 0.0, "BONUS_LAST": 0.0,
    },

    # Feature-Toggles (analog "neu")
    "FEATURES": {
        "USE_HISTORY": False,    # Historie in Features einbetten?
        "SEAT_ONEHOT": False,    # Sitz-One-Hot optional separat anhängen
        "PLOT_METRICS": False,   # Trainingsplots erzeugen?
        "SAVE_METRICS_TO_CSV": False,  # Trainingsmetriken persistent speichern?
    },

    # Benchmark-Gegner (für Plotter/Reports)
    "BENCH_OPPONENTS": ["single_only", "max_combo", "random2"],
}

def _with_seat_onehot(vec: np.ndarray, p: int, num_players: int, enabled: bool) -> np.ndarray:
    if not enabled:
        return vec
    seat_oh = np.zeros(num_players, dtype=np.float32)
    seat_oh[p] = 1.0
    return np.concatenate([vec, seat_oh], axis=0)

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
    A = env.action_spec()["num_actions"]
    num_players = game.num_players()

    # ---- Features (wie "neu": augment_observation ohne Seat-1hot; Seat-1hot extern anhängen) ----
    deck_int = int(CONFIG["DECK_SIZE"])
    num_ranks = ranks_for_deck(deck_int)
    feat_cfg = FeatureConfig(
        num_players=num_players,
        num_ranks=num_ranks,
        add_seat_onehot=False,                           # Seat-One-Hot NICHT in augment_observation
        include_history=CONFIG["FEATURES"]["USE_HISTORY"]
    )

    # ==== state_size WIE URSPRÜNGLICH (funktionierend) ====
    # Basierend auf observation_tensor_shape + optionalem externen Seat-1hot
    base_dim = game.observation_tensor_shape()[0]
    seat_id_dim = num_players if CONFIG["FEATURES"]["SEAT_ONEHOT"] else 0
    state_size = expected_feature_len(feat_cfg) + seat_id_dim

    # ---- Agent, Gegner, Reward-Shaper ----
    agent = DQNAgent(state_size, A, DQNConfig(**CONFIG["DQN"]))
    opponents = [STRATS[name] for name in CONFIG["OPPONENTS"]]
    shaper = RewardShaper(CONFIG["REWARD"])

    # ---- Config/Meta speichern ----
    save_config_csv({
        "script": family, "version": version,
        "deck_size": CONFIG["DECK_SIZE"], "num_ranks": num_ranks,
        "use_history": CONFIG["FEATURES"]["USE_HISTORY"],

        "agent_type": "DQN", "num_episodes": CONFIG["EPISODES"],
        "bench_interval": CONFIG["BENCH_INTERVAL"], "bench_episodes": CONFIG["BENCH_EPISODES"],
        "observation_dim": state_size, "num_actions": A,
        "seat_onehot": CONFIG["FEATURES"]["SEAT_ONEHOT"],
        "opponents": ",".join(CONFIG["OPPONENTS"]),
        "models_dir": paths["weights_dir"], "plots_dir": paths["plots_dir"],
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        # Reward-Setup (neues System)
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

    t0 = time.perf_counter()
    BINT, BEPS = CONFIG["BENCH_INTERVAL"], CONFIG["BENCH_EPISODES"]

    # Optionales Sammeln von Metriken (analog "neu")
    collect_metrics = (
        CONFIG["FEATURES"].get("SAVE_METRICS_TO_CSV", False)
        or CONFIG["FEATURES"].get("PLOT_METRICS", False)
    )

    for ep in range(1, CONFIG["EPISODES"] + 1):
        ep_start = time.perf_counter()
        ep_len = 0
        ep_shaping_return = 0.0
        ep_final_bonus = 0.0
        train_seconds_accum = 0.0

        ts = env.reset()
        last_p0_idx = None

        while not ts.last():

            # 1) Vorspulen, bis P0 am Zug ist (Gegner spielen Heuristiken)
            while not ts.last() and ts.observations["current_player"] != 0:
                pid = ts.observations["current_player"]
                a_opp = int(opponents[pid-1](env._state))  # Heuristiken erwarten pyspiel.State
                ts = env.step([a_opp])
                ep_len += 1

            if ts.last():
                break  # Episode vorbei

            # 2) P0-Zug: s, a, r_step (Reward nur für diesen P0-Schritt)
            legal0 = ts.observations["legal_actions"][0]
            s_base = np.array(env._state.observation_tensor(0), dtype=np.float32)
            s = augment_observation(s_base, player_id=0, cfg=feat_cfg)
            s = _with_seat_onehot(s, p=0, num_players=num_players, enabled=CONFIG["FEATURES"]["SEAT_ONEHOT"])

            # Shaping-Step (falls aktiv)
            if hasattr(shaper, "step_active") and shaper.step_active():
                hand_before = shaper.hand_size(ts, 0, deck_int)

            a0 = int(agent.select_action(s, legal0))
            ts = env.step([a0])  # führt P0-Zug aus
            ep_len += 1

            if hasattr(shaper, "step_active") and shaper.step_active():
                hand_after = shaper.hand_size(ts, 0, deck_int)
                r_step = float(shaper.step_reward(hand_before=hand_before, hand_after=hand_after))
                ep_shaping_return += r_step
            else:
                r_step = 0.0

            # 3) Vorspulen bis P0 wieder am Zug ist oder Terminal
            while not ts.last() and ts.observations["current_player"] != 0:
                pid = ts.observations["current_player"]
                a_opp = int(opponents[pid-1](env._state))
                ts = env.step([a_opp])
                ep_len += 1

            # 4) s' und next_legals für P0 definieren
            if ts.last():
                s_next = s
                next_legals = list(range(A))  # nie None im Buffer
                done = True
            else:
                next_legals = ts.observations["legal_actions"][0]
                s_next_base = np.array(env._state.observation_tensor(0), dtype=np.float32)
                s_next = augment_observation(s_next_base, player_id=0, cfg=feat_cfg)
                s_next = _with_seat_onehot(s_next, p=0, num_players=num_players, enabled=CONFIG["FEATURES"]["SEAT_ONEHOT"])
                done = False

            # 5) Transition speichern & lernen
            agent.buffer.add(s, a0, float(r_step), s_next, done, next_legal_actions=next_legals)
            last_p0_idx = len(agent.buffer.buffer) - 1

            tts = time.perf_counter()
            agent.train_step()
            train_seconds_accum += (time.perf_counter() - tts)

        # 6) Terminal: ENV-Reward + Finalbonus auf letzte P0-Transition addieren
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

        # ---- Train-Metriken
        metrics = {
            "ep_length": ep_len,
            "train_seconds": train_seconds_accum,
            "ep_env_score": ep_env_score,
            "ep_shaping_return": ep_shaping_return,
            "ep_final_bonus": ep_final_bonus,
            "epsilon": float(getattr(agent, "epsilon", np.nan)),
        }

        if collect_metrics:
            if CONFIG["FEATURES"].get("SAVE_METRICS_TO_CSV", False):
                plotter.add_train(ep, metrics)
            else:
                plotter.train_rows.append({"episode": int(ep), **metrics})
                if plotter.train_keys is None:
                    plotter.train_keys = ["episode"] + list(metrics.keys())
        else:
            # minimal: trotzdem in CSV, falls save_csv=True
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

            # Einheitliche Titel
            title_multi = f"Lernkurve - {family.upper()} vs feste Heuristiken"
            plotter.plot_benchmark(
                filename_prefix="lernkurve",
                with_macro=True,
                family_title=family.upper(),
                multi_title=title_multi,
            )

            if CONFIG["FEATURES"].get("PLOT_METRICS", False):
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

    total_seconds = time.perf_counter() - t0
    plotter.log("")
    plotter.log(f"Gesamtzeit: {total_seconds/3600:0.2f}h (~ {CONFIG['EPISODES']/max(total_seconds,1e-9):0.2f} eps/s)")
    plotter.log(f"{family}, Single Agent vs Heuristiken (max_combo). Training abgeschlossen.")
    plotter.log(f"Path: {paths['run_dir']}")

if __name__ == "__main__":
    main()
