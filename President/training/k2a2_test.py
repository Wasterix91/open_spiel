# -*- coding: utf-8 -*-
# President/training/k2a2.py — DQN (K2): Vier getrennte Agents, simultanes Lernen (k1a1-Stil)
# Struktur: models/k2a2/model_XX/{config.csv, plots/, models/}

import os, time, datetime, numpy as np, torch
import pyspiel
from open_spiel.python import rl_environment

from agents import dqn_agent as dqn
from utils.strategies import STRATS
from utils.fit_tensor import FeatureConfig, augment_observation
from utils.plotter import MetricsPlotter
from utils.timing import TimingMeter
from utils.reward_shaper import RewardShaper

from utils.load_save_common import find_next_version, prepare_run_dirs, save_config_csv, save_run_meta
from utils.benchmark import run_benchmark
from utils.deck import ranks_for_deck


# ============== CONFIG  ==============
CONFIG = {
    "EPISODES":         10_000,
    "BENCH_INTERVAL":   500,
    "BENCH_EPISODES":   2_000,
    "TIMING_INTERVAL":  500,
    "DECK_SIZE":        "64",  # "12" | "16" | "20" | "24" | "32" | "52" | "64"
    "SEED":             42,

    "DEVICE":           "cpu",

    # DQN (exakt die Keys aus agents/dqn_agent.py::DQNConfig)
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

    # ======= Rewards (NEUES System wie k1a1/k1a2) =======
    # STEP_MODE : "none" | "delta_weight_only" | "hand_penalty_coeff_only" | "combined"
    # FINAL_MODE: "none" | "env_only" | "rank_bonus" | "both"
    "REWARD": {
        "STEP_MODE": "none",
        "DELTA_WEIGHT": 0.0,
        "HAND_PENALTY_COEFF": 0.0,

        "FINAL_MODE": "env_only",
        "BONUS_WIN": 0.0, "BONUS_2ND": 0.0, "BONUS_3RD": 0.0, "BONUS_LAST": 0.0,
    },

    # Feature-Toggles
    "FEATURES": {
        "NORMALIZE": False,
        "SEAT_ONEHOT": False,   # getrennte Policies → meist False
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
    family = "k2a2"  # 4 gleichzeitige DQN-Agents
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

    plotter.log("New Training (k2a2): 4 getrennte DQN-Agents — simultanes Lernen")
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

    # ---- Beobachtung/Features (wie k1a2: observation_tensor) ----
    deck_int = int(CONFIG["DECK_SIZE"])
    num_ranks = ranks_for_deck(deck_int)
    base_dim  = game.observation_tensor_shape()[0]
    extra_dim = num_players if CONFIG["FEATURES"]["SEAT_ONEHOT"] else 0
    state_size = base_dim + extra_dim

    feat_cfg = FeatureConfig(
        num_players=num_players, num_ranks=num_ranks,
        add_seat_onehot=CONFIG["FEATURES"]["SEAT_ONEHOT"],
        normalize=CONFIG["FEATURES"]["NORMALIZE"],
    )

    # ---- Vier Agents (je Sitz ein eigenes DQN) + Reward ----
    # Robust: nur Felder überschreiben, die die DQNConfig wirklich hat
    base_cfg = dqn.DEFAULT_CONFIG
    overrides = {k: CONFIG["DQN"][k] for k in CONFIG["DQN"] if k in base_cfg._fields}
    dqn_cfg = base_cfg._replace(**overrides)

    agents = [
        dqn.DQNAgent(state_size=state_size, num_actions=A, config=dqn_cfg, device=CONFIG["DEVICE"])
        for _ in range(num_players)
    ]
    shaper = RewardShaper(CONFIG["REWARD"])

    # ---- Run-Metadaten & config ----
    save_config_csv({
        "script": family, "version": version,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent_type": "DQNx4_simultaneous", "num_episodes": CONFIG["EPISODES"],
        "bench_interval": CONFIG["BENCH_INTERVAL"], "bench_episodes": CONFIG["BENCH_EPISODES"],
        "deck_size": CONFIG["DECK_SIZE"], "num_ranks": num_ranks,
        "observation_dim": state_size, "num_actions": A,
        "normalize": CONFIG["FEATURES"]["NORMALIZE"], "seat_onehot": CONFIG["FEATURES"]["SEAT_ONEHOT"],
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

    save_run_meta({
        "family": family, "version": version,
        "algo": "dqn_x4_simul", "deck": CONFIG["DECK_SIZE"]
    }, paths["run_meta_json"])

    # ---- Timing (streaming) ----
    timer = TimingMeter(csv_path=paths["timings_csv"], interval=CONFIG["TIMING_INTERVAL"])

    t0 = time.perf_counter()
    BINT, BEPS = CONFIG["BENCH_INTERVAL"], CONFIG["BENCH_EPISODES"]

    # =========================== Training loop ===========================
    for ep in range(1, CONFIG["EPISODES"] + 1):
        ep_start = time.perf_counter()
        ep_len = 0
        ep_shape = [0.0 for _ in range(num_players)]   # Step-Shaping-Return je Sitz
        ep_final_bonus = [0.0 for _ in range(num_players)]

        ts = env.reset()
        # Pending-Transitions pro Sitz (decision-to-decision)
        pending = {p: None for p in range(num_players)}  # {"s":..., "a":..., "r":...}
        last_idx = {p: None for p in range(num_players)} # Index der letzten gespeicherten Transition

        while not ts.last():
            p = ts.observations["current_player"]
            legal = ts.observations["legal_actions"][p]

            # (A) Falls von p noch eine offene Transition existiert, jetzt abschließen:
            #     p ist wieder am Zug ⇒ next_state = aktueller Zustand von p, next_legals = legal von p
            if pending[p] is not None:
                base_now = np.array(env._state.observation_tensor(p), dtype=np.float32)
                s_now = augment_observation(base_now, player_id=p, cfg=feat_cfg)
                rec = pending[p]; pending[p] = None

                agents[p].buffer.add(rec["s"], rec["a"], rec["r"], s_now, False,
                                     next_legal_actions=legal)
                last_idx[p] = len(agents[p].buffer.buffer) - 1
                agents[p].train_step()

            # (B) Aktuelle Beobachtung für p
            s_base = np.array(env._state.observation_tensor(p), dtype=np.float32)
            s = augment_observation(s_base, player_id=p, cfg=feat_cfg)

            # Step-Shaping (optional)
            r_step = 0.0
            if shaper.step_active():
                hand_before = shaper.hand_size(ts, p, deck_int)

            # (C) Aktion wählen & ausführen
            a = int(agents[p].select_action(s, legal))
            ts_next = env.step([a])
            ep_len += 1

            if shaper.step_active():
                hand_after = shaper.hand_size(ts_next, p, deck_int)
                r_step = float(shaper.step_reward(hand_before=hand_before, hand_after=hand_after))
                ep_shape[p] += r_step

            # (D) Pending-Record anlegen (wird geschlossen, wenn p wieder dran ist oder Terminal)
            assert pending[p] is None
            pending[p] = {"s": s, "a": a, "r": r_step}

            ts = ts_next

        # ---- Episodenende: alle offenen Pendings finalisieren (done=True) ----
        for p in range(num_players):
            if pending[p] is not None:
                rec = pending[p]; pending[p] = None
                s_next = rec["s"]  # Dummy-Next-State ist ok
                agents[p].buffer.add(rec["s"], rec["a"], rec["r"], s_next, True,
                                     next_legal_actions=list(range(A)))
                last_idx[p] = len(agents[p].buffer.buffer) - 1
                agents[p].train_step()

        # ===== Episodenende: ENV & Bonus auf letzte Transition addieren =====
        finals = env._state.returns()  # returns nach Terminal
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

        # ===== Trainingsmetriken (P0 & Aggregate) =====
        train_metrics = {
            "ep_length":             ep_len,
            "ep_env_return_p0":      float(finals[0]),
            "ep_shaping_return_p0":  ep_shape[0],
            "ep_final_bonus_p0":     ep_final_bonus[0],
            "ep_env_return_avg":     float(np.mean(finals)),
            "ep_shaping_return_avg": float(np.mean(ep_shape)),
            "ep_final_bonus_avg":    float(np.mean(ep_final_bonus)),
            "epsilon_p0":            float(agents[0].epsilon),
        }
        plotter.add_train(ep, train_metrics)

        # ===== Benchmark (nur Agent von Player 0) & Save =====
        eval_seconds = plot_seconds = save_seconds = 0.0
        if ep % BINT == 0:
            evs = time.perf_counter()
            per_opponent = run_benchmark(
                game=game,
                agent=agents[0],                     # p0-Policy evaluieren
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
            plotter.plot_benchmark(filename_prefix="lernkurve", with_macro=True)
            plotter.plot_train(filename_prefix="training_metrics", separate=True)
            plot_seconds = time.perf_counter() - ps

            ss = time.perf_counter()
            for seat_id, ag in enumerate(agents):
                tag = f"{family}_model_{version}_agent_p{seat_id}_ep{ep:07d}"
                ag.save(os.path.join(paths["weights_dir"], tag))
            save_seconds = time.perf_counter() - ss

            # Kompakt-Timing
            cum_seconds = time.perf_counter() - t0
            plotter.log_timing(
                ep,
                ep_seconds=(time.perf_counter() - ep_start),
                train_seconds=0.0,  # DQN train_step on-the-fly
                eval_seconds=eval_seconds,
                plot_seconds=plot_seconds,
                save_seconds=save_seconds,
                cum_seconds=cum_seconds,
            )

        # ---- Episoden-Timing -> CSV ----
        ep_seconds = time.perf_counter() - ep_start
        timer.maybe_log(ep, {
            "steps": ep_len,
            "ep_seconds": ep_seconds,
            "train_seconds": 0.0,          # wir trainieren on-the-fly
            "eval_seconds": eval_seconds,
            "plot_seconds": plot_seconds,
            "save_seconds": save_seconds,
        })

    # Ende
    total_seconds = time.perf_counter() - t0
    plotter.log("")
    plotter.log(
        f"Gesamtzeit: {total_seconds/3600:0.2f}h "
        f"(~ {CONFIG['EPISODES']/max(total_seconds,1e-9):0.2f} eps/s)"
    )
    plotter.log("K2 (4 DQN Agents simultan) Training abgeschlossen.")


if __name__ == "__main__":
    main()
