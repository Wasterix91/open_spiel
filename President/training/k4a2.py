# -*- coding: utf-8 -*-
# President/training/k4a2.py — DQN (K4): Shared Policy (1 Netz für alle Seats)
# Fixes:
# - Decision-to-Decision per Seat (pending-Pattern, wie PPO k4a1)
# - next_legal_actions nur, wenn derselbe Spieler wieder am Zug ist
# - robuster DQNConfig-Build aus DEFAULT_CONFIG
# - Import auf agents/dqn_agent

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
    "EPISODES":         2000,
    "BENCH_INTERVAL":   500,
    "BENCH_EPISODES":   500,
    "TIMING_INTERVAL":  500,
    "DECK_SIZE":        "16",  # "12" | "16" | "20" | "24" | "32" | "52" | "64"
    "SEED":             42,

    # DQNConfig-kompatibel (siehe agents/dqn_agent.py)
    "DQN": {
        "learning_rate":     3e-4,
        "batch_size":        128,
        "gamma":             0.995,
        "epsilon_start":     1.0,
        "epsilon_end":       0.05,
        "epsilon_decay":     0.9997,        # multiplikativ pro train_step
        "buffer_size":       200_000,
        "target_update_freq": 5000,         # oder soft_target_tau > 0 für Polyak
        "soft_target_tau":   0.0,
        "max_grad_norm":     1.0,
        "use_double_dqn":    True,
        "loss_huber_delta":  1.0,
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

    # Shared Policy: Seat-OneHot sinnvoll → True
    "FEATURES": { "NORMALIZE": False, "SEAT_ONEHOT": True },

    # Gegner/Benchmark
    "BENCH_OPPONENTS": ["single_only", "max_combo", "random2"],
}

# =========================== Training ===========================
def main():
    np.random.seed(CONFIG["SEED"]); torch.manual_seed(CONFIG["SEED"])
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_ROOT = os.path.join(ROOT, "models")

    # ---- Verzeichnisse ----
    family = "k4a2"
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
    base_cfg = dqn.DEFAULT_CONFIG
    overrides = {k: CONFIG["DQN"][k] for k in CONFIG["DQN"] if k in base_cfg._fields}
    dqn_cfg = base_cfg._replace(**overrides)

    agent = dqn.DQNAgent(state_size=state_size, num_actions=A, config=dqn_cfg, device="cpu")
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
        # Pending-Transitions pro Sitz
        pending = {p: None for p in range(num_players)}   # {"s":..., "a":..., "r":...}
        last_idx = {p: None for p in range(num_players)}  # Index der letzten gespeicherten Transition je Sitz

        while not ts.last():
            p = ts.observations["current_player"]
            legal = ts.observations["legal_actions"][p]

            # (A) Falls p wieder dran ist, offene Transition von p schließen (decision-to-decision)
            if pending[p] is not None:
                base_now = np.array(ts.observations["info_state"][p], dtype=np.float32)
                s_now = augment_observation(base_now, player_id=p, cfg=feat_cfg)
                rec = pending[p]; pending[p] = None

                agent.buffer.add(rec["s"], rec["a"], rec["r"], s_now, False,
                                 next_legal_actions=legal)  # Legal-Maske gehört JETZT p
                last_idx[p] = len(agent.buffer.buffer) - 1
                agent.train_step()

            # (B) aktuelle Beobachtung für p
            base_obs = np.array(ts.observations["info_state"][p], dtype=np.float32)
            obs = augment_observation(base_obs, player_id=p, cfg=feat_cfg)

            # (C) optionaler Step-Reward
            r_step = 0.0
            if shaper.step_active():
                hand_before = shaper.hand_size(ts, p, deck_int)

            # (D) Aktion wählen & ausführen
            a = int(agent.select_action(obs, legal))
            ts_next = env.step([a]); ep_len += 1

            if shaper.step_active():
                hand_after = shaper.hand_size(ts_next, p, deck_int)
                r_step = float(shaper.step_reward(hand_before=hand_before, hand_after=hand_after))
                ep_shaping_returns[p] += r_step

            # (E) Pending für p anlegen (wird geschlossen, wenn p wieder dran ist oder am Ende)
            assert pending[p] is None
            pending[p] = {"s": obs, "a": a, "r": r_step}

            ts = ts_next

        # ---- Episodenende: alle offenen Transitions finalisieren (done=True) ----
        for p in range(num_players):
            if pending[p] is not None:
                rec = pending[p]; pending[p] = None
                s_next = rec["s"]  # Dummy-Next-State ist ok
                agent.buffer.add(rec["s"], rec["a"], rec["r"], s_next, True,
                                 next_legal_actions=list(range(A)))  # nie leere Maske
                last_idx[p] = len(agent.buffer.buffer) - 1
                agent.train_step()

        # ===== Finals/Bonis auf jeweils letzte Transition buchen =====
        finals = [float(ts.rewards[i]) for i in range(num_players)]
        for p in range(num_players):
            li = last_idx[p]
            if li is None:
                continue
            bonus = (finals[p] if shaper.include_env_reward() else 0.0) + float(shaper.final_bonus(finals, p))
            if abs(bonus) > 1e-8:
                buf = agent.buffer
                old = buf.buffer[li]
                buf.buffer[li] = buf.Experience(
                    old.state, old.action, float(old.reward + bonus),
                    old.next_state, old.done, old.next_legal_mask
                )

        # ===== Trainingsmetriken =====
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
            "epsilon":               float(agent.epsilon),
        })

        # ===== Benchmark & Save =====
        eval_seconds = plot_seconds = save_seconds = 0.0
        if BINT > 0 and (ep % BINT == 0):
            ev_start = time.perf_counter()
            per_opponent = run_benchmark(
                game=game,
                agent=agent,                      # Shared Policy; eval auf Seat 0
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

            # Einheitliche Titel
            title_multi = f"Lernkurve - {family.upper()} vs feste Heuristiken"
            plotter.plot_benchmark(
                filename_prefix="lernkurve",
                with_macro=True,
                family_title=family.upper(),   # Einzelplots: „Lernkurve - KxAy vs <gegner>“
                multi_title=title_multi,       # Multi & Macro: „Lernkurve - KxAy vs feste Heuristiken“
            )

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
                train_seconds=0.0,               # on-the-fly Training
                eval_seconds=eval_seconds,
                plot_seconds=plot_seconds,
                save_seconds=save_seconds,
                cum_seconds=cum_seconds,
            )


        # ---- Timing CSV ----
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
    plotter.log(f"{family}, Shared Policy Selfplay (4 Rollout, external Training). Training abgeschlossen.")
    plotter.log(f"Path: {paths['run_dir']}")

if __name__ == "__main__":
    main()
