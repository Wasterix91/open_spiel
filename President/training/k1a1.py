# President/training/k1a1.py

import os, datetime, time, numpy as np, torch, re, csv
import pyspiel
from open_spiel.python import rl_environment

from agents import ppo_agent as ppo
from agents import v_table_agent
from utils.strategies import STRATS
from utils.fit_tensor import FeatureConfig, augment_observation
from utils.deck import ranks_for_deck
from utils.plotter import MetricsPlotter
from utils.reward_shaper import RewardShaper
from utils.load_save_common import find_next_version, prepare_run_dirs, save_config_csv, save_run_meta
from utils.load_save_a1_ppo import save_checkpoint_ppo
from utils.benchmark import run_benchmark
from collections import defaultdict

# ============================ CONFIG ============================
CONFIG = {
    "EPISODES":         1_000_000,
    "BENCH_INTERVAL":   10_000,
    "BENCH_EPISODES":   2_000,
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
        "max_combo": 0.0,
        "single_only": 1.0,
        "random2": 0.0,
        "v_table": 0.0
    },

    # >0: Wechsel alle n Episoden; 0/negativ: nie wechseln
    "SWITCH_INTERVAL": 0,

    # PPO-Hyperparameter
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

    # Rewards
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
        "DEBUG_FEATURES": False,
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

    # Benchmark-Gegner (nur Labels/Namen; "v_table" erlaubt)
    "BENCH_OPPONENTS": ["single_only", "max_combo", "random2"],
}

# ===================== Helfer =====================
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
    # für Dateinamen/Plot-Labels
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(name))

def _safe_token_for_col(token: str) -> str:
    # für Spaltennamen in config.csv
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

# =============================== Training ===============================
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
                        base_obs = env._state.observation_tensor(p)
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
    family = "k1a1"
    family_dir = os.path.join(MODELS_ROOT, family)
    version = find_next_version(family_dir, prefix="model")
    paths = prepare_run_dirs(MODELS_ROOT, family, version, prefix="model")

    # ---- Plotter ----
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
    )

    plotter.log("New Training (k1a1)")
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
        normalize=CONFIG["FEATURES"]["NORMALIZE"],
        deck_size=deck_int,
    )

    state_size = feat_cfg.input_dim()

    # ---- Agent & Reward ----
    ppo_cfg = ppo.PPOConfig(**CONFIG["PPO"])
    agent = ppo.PPOAgent(state_size, A, config=ppo_cfg)
    shaper  = RewardShaper(CONFIG["REWARD"])

    # ---- Gegner-Setup ----
    pool = CONFIG.get("OPPONENT_POOL", {}) or {}
    use_population = any(float(w) > 0.0 for w in pool.values())
    # SWITCH_INTERVAL: >0 ⇒ alle n Episoden, 0/negativ ⇒ nie wechseln
    val = int(CONFIG.get("SWITCH_INTERVAL", 1))
    switch_interval = val if (use_population and val > 0) else None
    rng = np.random.default_rng(CONFIG["SEED"])

    if use_population:
        opponents_names_current = sample_lineup_from_pool(pool, n_seats=3, rng=rng)
        opponents = [resolve_opponent(n) for n in opponents_names_current]
        plotter.log(f"[Population] Initiales Lineup: {[str(n) for n in opponents_names_current]}")
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
        "step_mode": shaper.step_mode, "delta_weight": shaper.dw,
        "hand_penalty_coeff": shaper.hp, "final_mode": shaper.final_mode,
        "bonus_win":   CONFIG["REWARD"]["BONUS_WIN"],
        "bonus_2nd":   CONFIG["REWARD"]["BONUS_2ND"],
        "bonus_3rd":   CONFIG["REWARD"]["BONUS_3RD"],
        "bonus_last":  CONFIG["REWARD"]["BONUS_LAST"],
        "agent_type": "PPO", "num_episodes": CONFIG["EPISODES"],
        "bench_interval": CONFIG["BENCH_INTERVAL"], "bench_episodes": CONFIG["BENCH_EPISODES"],
        "observation_dim": feat_cfg.input_dim(),
        "use_population": use_population,
        "opponents_switch_interval": CONFIG.get("SWITCH_INTERVAL", None),
        "v_table_path": CONFIG.get("V_TABLE_PATH", ""),
        "models_dir": paths["weights_dir"], "plots_dir": paths["plots_dir"],
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **opp_pool_weights,
    }
    save_config_csv(cfg_row, paths["config_csv"])
    save_run_meta({"family": family, "version": version, "algo": "ppo", "deck": CONFIG["DECK_SIZE"]}, paths["run_meta_json"])

    t0 = time.perf_counter()
    BINT, BEPS = CONFIG["BENCH_INTERVAL"], CONFIG["BENCH_EPISODES"]
    collect_metrics = CONFIG["FEATURES"].get("SAVE_METRICS_TO_CSV", False) or CONFIG["FEATURES"].get("PLOT_METRICS", False)

    # ========================== Training Loop ==========================
    for ep in range(1, CONFIG["EPISODES"] + 1):
        ep_start = time.perf_counter()
        ep_len = 0
        ep_shaping_return = 0.0

        # Timing-Teilmessungen
        eval_seconds = 0.0
        plot_seconds = 0.0
        save_seconds = 0.0

        # Gegnerwechsel (bei Population & Intervall>0)
        if use_population and (switch_interval is not None) and ((ep - 1) % switch_interval == 0):
            opponents_names_current = sample_lineup_from_pool(pool, n_seats=3, rng=rng)
            opponents = [resolve_opponent(n) for n in opponents_names_current]
            plotter.log(f"[Population] Episode {ep}: Lineup -> {[str(n) for n in opponents_names_current]}")

        # --- realisierte Gegner dieser Episode zählen ---
        for name in opponents_names_current:
            realized_counts[name] += 1
        seat_samples_total += len(opponents_names_current)  # i.d.R. 3

        ts = env.reset()
        last_idx_p0 = None

        while not ts.last():
            p = ts.observations["current_player"]
            legal = ts.observations["legal_actions"][p]

            if p == 0:
                if shaper.step_active():
                    hand_before = shaper.hand_size(ts, p, deck_int)

                # Beobachtung aufbereiten (Seat-1hot wird – falls aktiviert – in augment_observation gehandhabt)
                base_obs = env._state.observation_tensor(p)
                obs = augment_observation(base_obs, player_id=p, cfg=feat_cfg)

                a = int(agent.step(obs, legal, player_id=0))  # kein seat_one_hot mehr
                last_idx_p0 = len(agent._buffer.states) - 1
            else:
                a = int(opponents[p - 1](env._state))

            ts_next = env.step([a])
            ep_len += 1

            if p == 0 and shaper.step_active():
                hand_after = shaper.hand_size(ts_next, p, deck_int)
                r = shaper.step_reward(hand_before=hand_before, hand_after=hand_after)
                ep_shaping_return += float(r)
                agent.post_step(r, done=ts_next.last())

            ts = ts_next


        # final env return / bonus
        ep_env_score = float(ts.rewards[0])
        ep_final_bonus = float(shaper.final_bonus(ts.rewards, 0))
        if last_idx_p0 is not None:
            if shaper.include_env_reward():
                agent._buffer.rewards[last_idx_p0] += float(ts.rewards[0])
            agent._buffer.rewards[last_idx_p0] += float(shaper.final_bonus(ts.rewards, 0))
            agent._buffer.dones[last_idx_p0] = True

        # --- Trainingsäquivalenter Return (P0) wie in k1a2/k2a2 ---
        env_part     = ep_env_score if shaper.include_env_reward() else 0.0
        shaping_part = ep_shaping_return if shaper.step_active() else 0.0
        ep_return_training = shaping_part + env_part + ep_final_bonus

        # Sonderplot: Episoden-Return + Komponenten (mit Gating)
        plotter.add_ep_returns(
            global_episode=ep,
            ep_returns=[ep_return_training],
            components={
                "env_score":   [env_part],
                "shaping":     [shaping_part],
                "final_bonus": [ep_final_bonus],
            },
        )

        # Update
        train_start = time.perf_counter()
        train_metrics = agent.train()
        train_seconds = time.perf_counter() - train_start

        # Trainingsmetriken
        if collect_metrics:
            metrics = train_metrics or {}
            metrics.update({
                "train_seconds":      train_seconds,
                "ep_env_score":       ep_env_score,
                "ep_shaping_return":  ep_shaping_return,
                "ep_final_bonus":     ep_final_bonus,
                "ep_length":          ep_len,
                "ep_return_training": ep_return_training,
            })
            if CONFIG["FEATURES"].get("SAVE_METRICS_TO_CSV", False):
                plotter.add_train(ep, metrics)
            else:
                plotter.train_rows.append({"episode": int(ep), **metrics})
                if plotter.train_keys is None:
                    plotter.train_keys = ["episode"] + list(metrics.keys())

        # ---- Benchmark/Checkpoint ----
        if ep % BINT == 0:
            ev_start = time.perf_counter()
            bench_dict = dict(STRATS)
            for tok in bench_tokens:
                if tok == "v_table":
                    bench_dict[tok] = resolve_opponent(tok)

            per_opponent_tokens = run_benchmark(
                game=game, agent=agent, opponents_dict=bench_dict,
                opponent_names=bench_tokens, episodes=BEPS,
                feat_cfg=feat_cfg, num_actions=A,
            )
            per_opponent_labels = {_safe_name(k): v for k, v in per_opponent_tokens.items()}
            eval_seconds = time.perf_counter() - ev_start

            plot_start = time.perf_counter()
            plotter.log_bench_summary(ep, per_opponent_labels)
            plotter.add_benchmark(ep, per_opponent_labels)
            plotter.plot_benchmark_rewards()
            plotter.plot_places_latest()
            title_multi = f"Lernkurve - {family.upper()} vs feste Heuristiken"
            plotter.plot_benchmark(filename_prefix="lernkurve", with_macro=True,
                                   family_title=family.upper(), multi_title=title_multi)
            if CONFIG["FEATURES"].get("PLOT_METRICS", False):
                plotter.plot_train(
                    include_keys=CONFIG["FEATURES"].get("PLOT_KEYS"),
                    separate=True,
                )
            plot_seconds = time.perf_counter() - plot_start

            save_start = time.perf_counter()
            tag = f"{family}_model_{version}_agent_p0_ep{ep:07d}"
            save_checkpoint_ppo(agent, paths["weights_dir"], tag)
            save_seconds = time.perf_counter() - save_start

            plotter.log_timing(
                ep,
                ep_seconds=(time.perf_counter() - ep_start),
                train_seconds=train_seconds,
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
