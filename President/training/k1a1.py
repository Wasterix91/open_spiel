import os, datetime, time, numpy as np, torch
import pyspiel
from open_spiel.python import rl_environment

from agents import ppo_agent as ppo
from utils.strategies import STRATS
from utils.fit_tensor import FeatureConfig, augment_observation
from utils.plotter import MetricsPlotter
from utils.timing import TimingMeter
from utils.reward_shaper import RewardShaper

from utils.load_save_common import find_next_version, prepare_run_dirs, save_config_csv, save_run_meta
from utils.load_save_a1_ppo import save_checkpoint_ppo

from utils.benchmark import run_benchmark
from utils.deck import ranks_for_deck


# ============== CONFIG  ==============
CONFIG = {
    "EPISODES":         200,
    "BENCH_INTERVAL":   100,
    "BENCH_EPISODES":   500,
    "TIMING_INTERVAL":  50,
    "DECK_SIZE":        "64",  # "12" | "16" | "20" | "24" | "32" | "52" | "64"
    "SEED":             42,

    # Training-Gegner (Heuristiken)
    "OPPONENTS":        ["max_combo", "max_combo", "max_combo"],

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

    # Reward-Shaping
    "REWARD": {
        "STEP": "delta_hand",
        "DELTA_WEIGHT": 0.5,
        "HAND_PENALTY_COEFF": 0.0,
        "FINAL": "none",
        "BONUS_WIN": 10.0, "BONUS_2ND": 0.0, "BONUS_3RD": 0.0, "BONUS_LAST": 0.0,
        "ENV_REWARD": True,
    },

    # Feature-Toggles
    "FEATURES": {
        "NORMALIZE": False,
        "SEAT_ONEHOT": False,
    },

    # Benchmark-Gegner
    "BENCH_OPPONENTS": ["single_only", "max_combo", "random2"],
}


# =========================== Training ===========================
def main():
    np.random.seed(CONFIG["SEED"]); torch.manual_seed(CONFIG["SEED"])
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_ROOT = os.path.join(ROOT, "models")

    # ---- Lauf-Verzeichnisse ----
    family = "k1a1"  # K1-Konzept + A1 (PPO)
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

    plotter.log("New Training")
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
    info_dim = env.observation_spec()["info_state"][0]
    A = env.action_spec()["num_actions"]

    # ---- Features ----
    deck_int = int(CONFIG["DECK_SIZE"])
    num_players = game.num_players()

    num_ranks = ranks_for_deck(deck_int)
    feat_cfg = FeatureConfig(
        num_players=num_players, num_ranks=num_ranks,
        add_seat_onehot=CONFIG["FEATURES"]["SEAT_ONEHOT"],
        normalize=CONFIG["FEATURES"]["NORMALIZE"],
    )

    # ---- Agent/opp/reward ----
    ppo_cfg = ppo.PPOConfig(**CONFIG["PPO"])
    seat_id_dim = (num_players if CONFIG["FEATURES"]["SEAT_ONEHOT"] else 0)
    agent = ppo.PPOAgent(info_dim, A, seat_id_dim=seat_id_dim, config=ppo_cfg)
    opponents = [STRATS[name] for name in CONFIG["OPPONENTS"]]
    shaper = RewardShaper(CONFIG["REWARD"])

    # ---- Run-Metadaten & config ----
    save_config_csv({
        "script": family, "version": version,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent_type": "PPO", "num_episodes": CONFIG["EPISODES"],
        "bench_interval": CONFIG["BENCH_INTERVAL"], "bench_episodes": CONFIG["BENCH_EPISODES"],
        "deck_size": CONFIG["DECK_SIZE"], "num_ranks": num_ranks,
        "observation_dim": info_dim + seat_id_dim, "num_actions": A,
        "normalize": CONFIG["FEATURES"]["NORMALIZE"], "seat_onehot": CONFIG["FEATURES"]["SEAT_ONEHOT"],
        "opponents": ",".join(CONFIG["OPPONENTS"]),
        "models_dir": paths["weights_dir"], "plots_dir": paths["plots_dir"],
    }, paths["config_csv"])

    save_run_meta({
        "family": family, "version": version,
        "algo": "ppo", "deck": CONFIG["DECK_SIZE"]
    }, paths["run_meta_json"])

    # ---- Timing (streaming) ----
    timer = TimingMeter(csv_path=paths["timings_csv"], interval=CONFIG["TIMING_INTERVAL"])

    t0 = time.perf_counter()
    BINT, BEPS = CONFIG["BENCH_INTERVAL"], CONFIG["BENCH_EPISODES"]

    # ---- Training loop ----
    for ep in range(1, CONFIG["EPISODES"] + 1):
        ep_start = time.perf_counter()
        ep_len = 0
        ep_shaping_return = 0.0

        ts = env.reset()

        while not ts.last():
            ep_len += 1
            p = ts.observations["current_player"]
            legal = ts.observations["legal_actions"][p]
            hand_before = shaper.hand_size(ts, p, deck_int)

            if p == 0:
                base_obs = ts.observations["info_state"][p]
                obs = augment_observation(base_obs, player_id=p, cfg=feat_cfg)
                seat_oh = None
                if CONFIG["FEATURES"]["SEAT_ONEHOT"]:
                    seat_oh = np.zeros(num_players, dtype=np.float32); seat_oh[p] = 1.0
                a = int(agent.step(obs, legal, seat_one_hot=seat_oh))
            else:
                a = int(opponents[p - 1](env._state))

            ts_next = env.step([a])

            if p == 0:
                hand_after = shaper.hand_size(ts_next, p, deck_int)
                r = shaper.step_reward(
                    hand_before=hand_before,
                    hand_after=hand_after,
                    time_step=ts_next,
                    player_id=p,
                    deck_size=deck_int,
                )
                ep_shaping_return += float(r)
                agent.post_step(r, done=ts_next.last())

            ts = ts_next

        # final env return for P0:
        ep_env_return = float(ts.rewards[0])
        ep_final_bonus = float(shaper.final_bonus(ts.rewards, 0))

        # ---- train() separat timen ----
        train_start = time.perf_counter()
        if shaper.include_env_reward():
            agent._buffer.finalize_last_reward(ts.rewards[0])
        agent._buffer.finalize_last_reward(shaper.final_bonus(ts.rewards, 0))
        train_metrics = agent.train()
        train_seconds = time.perf_counter() - train_start

        # Trainingsmetriken
        if train_metrics is None:
            train_metrics = {}
        train_metrics.update({
            "train_seconds":      train_seconds,
            "ep_env_return":      ep_env_return,
            "ep_shaping_return":  ep_shaping_return,
            "ep_final_bonus":     ep_final_bonus,
            "ep_length":          ep_len,
        })
        plotter.add_train(ep, train_metrics)

        # ---- Benchmark (früher Eval) ----
        eval_seconds = 0.0
        plot_seconds = 0.0
        save_seconds = 0.0

        if ep % BINT == 0:
            ev_start = time.perf_counter()

            # NEU: ausgelagert
            per_opponent = run_benchmark(
                game=game,
                agent=agent,
                opponents_dict=STRATS,
                opponent_names=CONFIG["BENCH_OPPONENTS"],
                episodes=BEPS,
                feat_cfg=feat_cfg,
                num_actions=A,
            )
            plotter.log_bench_summary(ep, per_opponent)

            eval_seconds = time.perf_counter() - ev_start

            # Plot & CSV
            plot_start = time.perf_counter()
            plotter.add_benchmark(ep, per_opponent)             # CSV (Winrate, Reward, Plätze)
            plotter.plot_benchmark_rewards()                    # Reward-Kurven
            plotter.plot_places_latest()                        # letzte Verteilung als Balken pro Gegner
            plotter.plot_benchmark(filename_prefix="lernkurve", with_macro=True)
            plotter.plot_train(filename_prefix="training_metrics", separate=True)
            plot_seconds = time.perf_counter() - plot_start

            # Save weights (PPO)
            save_start = time.perf_counter()
            tag = f"{family}_model_{version}_agent_p0_ep{ep:07d}"
            save_checkpoint_ppo(agent, paths["weights_dir"], tag)
            save_seconds = time.perf_counter() - save_start

            # Kompakt-Timing
            cum_seconds = time.perf_counter() - t0
            plotter.log_timing(
                ep,
                ep_seconds=(time.perf_counter() - ep_start),
                train_seconds=train_seconds,
                eval_seconds=eval_seconds,
                plot_seconds=plot_seconds,
                save_seconds=save_seconds,
                cum_seconds=cum_seconds,
            )

        # ---- Episoden-Timing -> CSV (verschlankt) ----
        ep_seconds = time.perf_counter() - ep_start
        timer.maybe_log(ep, {
            "steps": ep_len,
            "ep_seconds": ep_seconds,
            "train_seconds": train_seconds,
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
    plotter.log("K1 Training abgeschlossen.")


if __name__ == "__main__":
    main()
