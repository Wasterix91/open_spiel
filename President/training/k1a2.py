# -*- coding: utf-8 -*-
# President/training/k1a2.py — K1: Single-Agent DQN vs Heuristiken/Population (P0→P0)

import os, time, datetime, re, csv
import numpy as np, torch
import pyspiel
from open_spiel.python import rl_environment

from agents.dqn_agent import DQNAgent, DQNConfig
from agents import v_table_agent
from utils.fit_tensor import FeatureConfig, augment_observation
from utils.deck import ranks_for_deck
from utils.plotter import MetricsPlotter
from utils.load_save_common import find_next_version, prepare_run_dirs, save_config_csv, save_run_meta
from utils.benchmark import run_benchmark
from utils.strategies import STRATS
from utils.reward_shaper import RewardShaper
from collections import defaultdict

CONFIG = {
    "EPISODES":         100_000,
    "BENCH_INTERVAL":   5000,
    "BENCH_EPISODES":   2000,
    "DECK_SIZE":        "64",  # "12" | "16" | "20" | "24" | "32" | "52" | "64"
    "SEED":             42,

    # Pfadpräfix der Wertetabelle (ohne *_params.json/_index.bin/_data.bin)
    "V_TABLE_PATH": "agents/tables/v_table_4_4_4",

    # --- Gegner ---
    # Fixed (Fallback, wenn POOL leer oder alle Gewichte 0)
    "OPPONENTS": ["max_combo", "single_only", "random2"],

    # Population: aktiviert, sobald irgendein Gewicht > 0 ist
    # Tabellengegner einfach als "v_table" referenzieren
    "OPPONENT_POOL": {
        "max_combo": 1.0,
        "single_only": 0.0,
        "random2": 0.0,
        "v_table": 0.0
    },

        # >0: Wechsel alle n Episoden; 0/negativ: nie wechseln
    "SWITCH_INTERVAL": 0,

    # DQN (kompatibel zu agents/dqn_agent.DQNConfig)
    "DQN": {
        "learning_rate": 3e-4,
        "batch_size": 128,
        "gamma": 0.995,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_type": "multiplicative",   # "linear" | "multiplicative"
        "epsilon_decay": 0.9997,      # für multiplicative
        #"epsilon_decay_frames": 100_000,  # für linear
        "buffer_size": 200_000,
        "target_update_freq": 5000,
        "soft_target_tau": 0.0,
        "max_grad_norm": 1.0,
        "use_double_dqn": True,
        "loss_huber_delta": 1.0,
        "optimizer": "adam"
    },


    # ======= Rewards (neues System) =======
    # STEP_MODE : "none" | "delta_weight_only" | "hand_penalty_coeff_only" | "combined"
    # FINAL_MODE: "none" | "env_only" | "rank_only" | "both"
    "REWARD": {
        "STEP_MODE": "combined",
        "DELTA_WEIGHT": 0.1,
        "HAND_PENALTY_COEFF": 0.0,

        "FINAL_MODE": "env_only",
        "BONUS_WIN": 0.0, "BONUS_2ND": 0.0, "BONUS_3RD": 0.0, "BONUS_LAST": 0.0,
    },

    # Feature-Toggles
    "FEATURES": {
        "USE_HISTORY": True,
        "SEAT_ONEHOT": False,
        "NORMALIZE": False,
        "DEBUG_FEATURES": False,
        "PLOT_METRICS": True,
        "SAVE_METRICS_TO_CSV": False,

        # Fenstergröße für Moving Average der Episode-Returns
        "RET_SMOOTH_WINDOW": 150,

        # Winrate-Plot (wie k4a2): Rolling-Window & Konfidenzintervall
        "WR_SMOOTH_WINDOW": 3,
        "WR_SHOW_CI": True,
        "WR_CI_Z": 1.96,

        # Ausgabeformate für alle Plots
        "PLOT_FORMATS": ["pdf"],

        # Steuert plot_train(); ep_return_* triggert die In-Memory-Return-Plots
        "PLOT_KEYS": [
            #"return_mean", "reward_mean", "entropy", "approx_kl",
            "epsilon", "ep_length", "train_seconds",
            #"clip_frac", "policy_loss", "value_loss",
            #"ep_return_raw", "ep_return_components",
            #"ep_return_env", "ep_return_shaping", "ep_return_final",
            #"ep_return_training",
        ],
    },

    # Benchmark-Gegner (für Plotter/Reports)
    "BENCH_OPPONENTS": ["single_only", "max_combo", "random2"],
}

# ------------------------------ Helfer ------------------------------
def resolve_opponent(name: str):
    """callable(state)->action. 'v_table' lädt die Wertetabelle via CONFIG['V_TABLE_PATH']."""
    if name == "v_table":
        path = CONFIG.get("V_TABLE_PATH")
        if not path or not isinstance(path, str):
            raise ValueError("CONFIG['V_TABLE_PATH'] ist nicht gesetzt oder ungültig.")
        return v_table_agent.ValueTableAgent(path)
    if name not in STRATS:
        raise KeyError(f"Unbekannter Gegner-Name: {name}")
    return STRATS[name]

def sample_lineup_from_pool(pool_dict, n_seats=3, rng=np.random):
    """Gewichtetes Sampling pro Seat (mit Zurücklegen). Erwartet >=1 Gewicht > 0."""
    items = [(k, float(w)) for k, w in pool_dict.items() if float(w) > 0.0]
    if not items:
        raise ValueError("OPPONENT_POOL hat keine positiven Gewichte.")
    names, weights = zip(*items)
    probs = np.asarray(weights, dtype=np.float64); probs /= probs.sum()
    chooser = getattr(rng, "choice", np.random.choice)
    return [chooser(names, p=probs) for _ in range(n_seats)]

def _safe_name(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(name))

def _safe_token_for_col(token: str) -> str:
    return token.replace(":", "_").replace("/", "_").replace("\\", "_")

def _save_opponent_usage_csv(path_csv: str, counts: dict, total_seat_samples: int):
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    rows = []
    for token, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        pct = (100.0 * c / total_seat_samples) if total_seat_samples > 0 else 0.0
        rows.append({"token": token, "count": int(c), "percent": f"{pct:.4f}"})
    with open(path_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["token", "count", "percent"])
        w.writeheader()
        w.writerows(rows)

# ------------------------------ Training ------------------------------
def main():
    # ================== DEBUG: Feature-Vektor Check ==================
    if CONFIG["FEATURES"].get("DEBUG_FEATURES", False):
        for deck_size in [16, 64]:
            for use_history in [True, False]:
                for seat_onehot in [True, False]:
                    for normalize in [True, False]:
                        # Konfiguration temporär setzen
                        CONFIG["DECK_SIZE"] = str(deck_size)
                        CONFIG["FEATURES"]["USE_HISTORY"] = use_history
                        CONFIG["FEATURES"]["SEAT_ONEHOT"] = seat_onehot
                        CONFIG["FEATURES"]["NORMALIZE"] = normalize

                        # Env + FeatureConfig
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
                        state_size = feat_cfg.input_dim()


                        # Beobachtung genau wie im DQN-Training
                        ts = env.reset()
                        p = 0  # wir inspizieren P0
                        base_obs = np.asarray(env._state.observation_tensor(p), dtype=np.float32)
                        obs = augment_observation(base_obs, player_id=p, cfg=feat_cfg)

                        # Ausgabe im gewünschten Format
                        print("Deck Size:  ", CONFIG["DECK_SIZE"])
                        print("Use History:", CONFIG["FEATURES"]["USE_HISTORY"])
                        print("Seat 1-Hot: ", CONFIG["FEATURES"]["SEAT_ONEHOT"])
                        print("Normalize:  ", CONFIG["FEATURES"]["NORMALIZE"])
                        print(f"Tensor length={len(obs)}")
                        print(f"Tensor: {np.round(obs, 3)}")
                        print("-" * 60)

        return  # Debug beendet, nicht trainieren
    # ================== DEBUG Ende ==================

    np.random.seed(CONFIG["SEED"]); torch.manual_seed(CONFIG["SEED"])
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_ROOT = os.path.join(ROOT, "models")

    family = "k1a2"
    family_dir = os.path.join(MODELS_ROOT, family)
    version = find_next_version(family_dir, prefix="model")
    paths = prepare_run_dirs(MODELS_ROOT, family, version, prefix="model")

    # Plotter (nur Labels weitergeben)
    bench_tokens = list(CONFIG["BENCH_OPPONENTS"])
    bench_labels = [_safe_name(n) for n in bench_tokens]
    plotter = MetricsPlotter(
        out_dir=paths["plots_dir"],
        benchmark_opponents=bench_labels,
        benchmark_csv="benchmark_curves.csv",
        train_csv="training_metrics.csv",
        save_csv=CONFIG["FEATURES"]["SAVE_METRICS_TO_CSV"],
        verbosity=1,
        smooth_window=CONFIG["FEATURES"].get("RET_SMOOTH_WINDOW", 150),
        out_formats=CONFIG["FEATURES"].get("PLOT_FORMATS", ["png", "svg", "pdf"]),
        name_prefix=f"{family}_{version}",   # <--- NEU: z.B. "k1a2_51"
    )


    plotter.log("New Training (k1a2): DQN Single-Agent vs Heuristiken/Population (P0→P0)")
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
        add_seat_onehot=CONFIG["FEATURES"]["SEAT_ONEHOT"],
        include_history=CONFIG["FEATURES"]["USE_HISTORY"],
        normalize=CONFIG["FEATURES"]["NORMALIZE"],
        deck_size=deck_int, 
    )
    state_size = feat_cfg.input_dim()
   # KEIN extra seat_id_dim mehr

    # ---- Agent, Reward-Shaper ----
    agent = DQNAgent(state_size, A, DQNConfig(**CONFIG["DQN"]))
    shaper = RewardShaper(CONFIG["REWARD"])

    # ---- Gegner-Setup (Fixed vs Population) ----
    pool = CONFIG.get("OPPONENT_POOL", {}) or {}
    use_population = any(float(w) > 0.0 for w in pool.values())
    val = int(CONFIG.get("SWITCH_INTERVAL", 1))
    switch_interval = val if (use_population and val > 0) else None
    rng = np.random.default_rng(CONFIG["SEED"])

    if use_population:
        opponents_names_current = sample_lineup_from_pool(pool, n_seats=3, rng=rng)
        opponents = [resolve_opponent(n) for n in opponents_names_current]
        #plotter.log(f"[Population] Initiales Lineup: {[str(n) for n in opponents_names_current]}")
    else:
        fixed = CONFIG.get("OPPONENTS", ["max_combo"] * 3)
        opponents_names_current = list(fixed)
        opponents = [resolve_opponent(n) for n in opponents_names_current]

    # ---- Zähler für realisierte Gegner ----
    realized_counts = defaultdict(int)
    seat_samples_total = 0

    # ---- Config/Meta speichern (inkl. Pool-Spalten) ----
    opp_pool_weights = {}
    if use_population:
        for k, v in pool.items():
            opp_pool_weights[f"pool_w_{_safe_token_for_col(k)}"] = float(v)
    else:
        for i, name in enumerate(opponents_names_current, start=1):
            opp_pool_weights[f"fixed_seat{i}"] = name

    cfg_row = {
        "script": family, "version": version,
        "deck_size": CONFIG["DECK_SIZE"], "num_ranks": num_ranks,
        "use_history": CONFIG["FEATURES"]["USE_HISTORY"],
        "agent_type": "DQN", "num_episodes": CONFIG["EPISODES"],
        "bench_interval": CONFIG["BENCH_INTERVAL"], "bench_episodes": CONFIG["BENCH_EPISODES"],
        "observation_dim": state_size, "num_actions": A,
        "seat_onehot": CONFIG["FEATURES"]["SEAT_ONEHOT"],
        "use_population": use_population,
        "opponents_switch_interval": CONFIG.get("SWITCH_INTERVAL", None),
        "v_table_path": CONFIG.get("V_TABLE_PATH", ""),
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
        **opp_pool_weights,
    }
    save_config_csv(cfg_row, paths["config_csv"])
    save_run_meta({"family": family, "version": version, "algo": "dqn", "deck": CONFIG["DECK_SIZE"]}, paths["run_meta_json"])

    t0 = time.perf_counter()
    BINT, BEPS = CONFIG["BENCH_INTERVAL"], CONFIG["BENCH_EPISODES"]
    collect_metrics = (
        CONFIG["FEATURES"].get("SAVE_METRICS_TO_CSV", False)
        or CONFIG["FEATURES"].get("PLOT_METRICS", False)
    )

    for ep in range(1, CONFIG["EPISODES"] + 1):
        ep_start = time.perf_counter()
        ep_len = 0
        ep_shaping_return = 0.0
        ep_final_bonus = 0.0
        ep_step_delta_return = 0.0   # <— NEU
        ep_step_penalty_return = 0.0 # <— NEU

        train_seconds_accum = 0.0

        # Timing-Teilmessungen
        eval_seconds = 0.0
        plot_seconds = 0.0
        save_seconds = 0.0

        # Gegnerwechsel (nur bei Population & Intervall>0)
        if use_population and (switch_interval is not None) and ((ep - 1) % switch_interval == 0):
            opponents_names_current = sample_lineup_from_pool(pool, n_seats=3, rng=rng)
            opponents = [resolve_opponent(n) for n in opponents_names_current]
            #plotter.log(f"[Population] Episode {ep}: Lineup -> {[str(n) for n in opponents_names_current]}")

        # --- realisierte Gegner dieser Episode zählen ---
        for name in opponents_names_current:
            realized_counts[name] += 1
        seat_samples_total += len(opponents_names_current)  # i.d.R. 3

        ts = env.reset()
        last_p0_idx = None

        while not ts.last():

            # 1) Vorspulen, bis P0 am Zug ist
            while not ts.last() and ts.observations["current_player"] != 0:
                pid = ts.observations["current_player"]
                a_opp = int(opponents[pid-1](env._state))
                ts = env.step([a_opp])
                ep_len += 1

            if ts.last():
                break

            # 2) P0-Zug
            legal0 = ts.observations["legal_actions"][0]
            s_base = np.array(env._state.observation_tensor(0), dtype=np.float32)
            # augment_observation hängt den Sitz-One-Hot an, falls in feat_cfg aktiviert.
            s = augment_observation(s_base, player_id=0, cfg=feat_cfg)

            if hasattr(shaper, "step_active") and shaper.step_active():
                hand_before = shaper.hand_size(ts, 0, deck_int)

            a0 = int(agent.select_action(s, legal0))
            ts = env.step([a0])
            ep_len += 1

            if hasattr(shaper, "step_active") and shaper.step_active():
                hand_after = shaper.hand_size(ts, 0, deck_int)
                # wie in k4a2: Komponenten separat
                delta_r, penalty_r, r_step = shaper.step_reward_components(
                    hand_before=hand_before, hand_after=hand_after
                )
                ep_shaping_return       += float(r_step)
                ep_step_delta_return    += float(delta_r)
                ep_step_penalty_return  += float(penalty_r)
            else:
                r_step = 0.0

            # 3) Vorspulen bis P0 wieder am Zug ist oder Terminal
            while not ts.last() and ts.observations["current_player"] != 0:
                pid = ts.observations["current_player"]
                a_opp = int(opponents[pid-1](env._state))
                ts = env.step([a_opp])
                ep_len += 1

            # 4) s' / next_legals
            if ts.last():
                s_next = s
                next_legals = list(range(A))
                done = True
            else:
                next_legals = ts.observations["legal_actions"][0]
                s_next_base = np.array(env._state.observation_tensor(0), dtype=np.float32)
                s_next = augment_observation(s_next_base, player_id=0, cfg=feat_cfg)
                done = False

            # 5) Store & train step
            agent.buffer.add(s, a0, float(r_step), s_next, done, next_legal_actions=next_legals)
            last_p0_idx = len(agent.buffer.buffer) - 1
            tts = time.perf_counter()
            agent.train_step()
            train_seconds_accum += (time.perf_counter() - tts)

        # 6) Terminal-Reward/Bonus auf letzte Transition
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

        # Train-Metriken
        metrics = {
            "ep_length": ep_len,
            "train_seconds": train_seconds_accum,
            "ep_env_score": ep_env_score,
            "ep_shaping_return": ep_shaping_return,
            "ep_final_bonus": ep_final_bonus,
            "epsilon": float(getattr(agent, "epsilon", np.nan)),
        }

        # --- Trainingsäquivalenter Return exakt wie in k1a1/k1a2 ---
        env_part     = ep_env_score if (hasattr(shaper, "include_env_reward") and shaper.include_env_reward()) else 0.0
        shaping_part = ep_shaping_return if (hasattr(shaper, "step_active") and shaper.step_active()) else 0.0
        ep_return_training = shaping_part + env_part + ep_final_bonus
        metrics["ep_return_training"] = ep_return_training

        # Sonderplot: Episoden-Return + Komponenten (mit Gating)
        plotter.add_ep_returns(
            global_episode=ep,
            ep_returns=[ep_return_training],
            components={
                "env_score":   [env_part],
                "shaping":     [shaping_part],
                "final_bonus": [ep_final_bonus],
                "step_delta":  [ep_step_delta_return],   # <— NEU
                "step_penalty":[ep_step_penalty_return], # <— NEU
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

        # ---- Benchmark + Save in Intervallen ----
        if ep % BINT == 0:
            evs = time.perf_counter()
            bench_map = dict(STRATS)
            for tok in bench_tokens:
                if tok == "v_table":
                    bench_map[tok] = resolve_opponent(tok)

            per_opponent_tokens = run_benchmark(
                game=game, agent=agent, opponents_dict=bench_map,
                opponent_names=bench_tokens,
                episodes=BEPS, feat_cfg=feat_cfg, num_actions=A
            )
            eval_seconds = time.perf_counter() - evs

            ps = time.perf_counter()
            per_opponent = {_safe_name(k): v for k, v in per_opponent_tokens.items()}
            plotter.log_bench_summary(ep, per_opponent)
            plotter.add_benchmark(ep, per_opponent)

            # identische Gruppenplots wie k4a2
            plotter.plot_reward_groups(window=plotter.smooth_window)

            plotter.plot_places_latest()
            title_multi = f"Lernkurve (Benchmark) - {family.upper()} vs. feste Heuristiken"
            plotter.plot_benchmark(
                filename_prefix="lernkurve",
                with_macro=True,
                family_title=family.upper(),
                multi_title=f"Lernkurve - {family.upper()} vs feste Heuristiken",
                variants=["03"], 
            )


            # OPTIONAL: Wenn du die reinen Reward-Kurven nicht brauchst, kannst du den Aufruf weglassen
            # plotter.plot_benchmark_rewards()


            if CONFIG["FEATURES"].get("PLOT_METRICS", False):
                plotter.plot_train(
                    include_keys=CONFIG["FEATURES"].get("PLOT_KEYS"),
                    separate=True,
                )
            plot_seconds = time.perf_counter() - ps

            ss = time.perf_counter()
            tag = f"{family}_model_{version}_agent_p0_ep{ep:07d}"
            agent.save(os.path.join(paths["weights_dir"], tag))
            save_seconds = time.perf_counter() - ss

            plotter.log_timing(
                ep,
                ep_seconds=(time.perf_counter() - ep_start),
                train_seconds=train_seconds_accum,
                eval_seconds=eval_seconds,
                plot_seconds=plot_seconds,
                save_seconds=save_seconds,
                cum_seconds=(time.perf_counter() - t0),
            )

    # Ende
    usage_csv = os.path.join(paths["plots_dir"], "opponent_usage.csv")
    _save_opponent_usage_csv(usage_csv, realized_counts, seat_samples_total)
    plotter.log(f"Opponent-Usage gespeichert: {usage_csv}")

    total_seconds = time.perf_counter() - t0
    mode_txt = "Population" if use_population else "Fixed"
    plotter.log("")
    plotter.log(f"Gesamtzeit: {total_seconds/3600:0.2f}h (~ {CONFIG['EPISODES']/max(total_seconds,1e-9):0.2f} eps/s)")
    plotter.log(f"{family}, Single Agent vs {mode_txt}. Training abgeschlossen.")
    plotter.log(f"Path: {paths['run_dir']}")

if __name__ == "__main__":
    main()
