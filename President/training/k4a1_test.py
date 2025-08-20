# -*- coding: utf-8 -*-
# President/training/k4a1.py — PPO (K4): Shared Policy, k1a1-Stil
import os, datetime, time, numpy as np, torch
import pyspiel
from open_spiel.python import rl_environment

from agents import ppo_agent as ppo
from utils.fit_tensor import FeatureConfig, augment_observation
from utils.plotter import MetricsPlotter
from utils.timing import TimingMeter
from utils.reward_shaper import RewardShaper
from utils.strategies import STRATS

from utils.load_save_common import find_next_version, prepare_run_dirs, save_config_csv, save_run_meta
from utils.load_save_a1_ppo import save_checkpoint_ppo

from utils.benchmark import run_benchmark
from utils.deck import ranks_for_deck

# ============== CONFIG ==============
CONFIG = {
    "EPISODES":         200,
    "BENCH_INTERVAL":   100,
    "BENCH_EPISODES":   500,
    "TIMING_INTERVAL":  50,
    "DECK_SIZE":        "64",  # "12" | "16" | "20" | "24" | "32" | "52" | "64"
    "SEED":             42,

    # PPO
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

    "EXTERNAL": {
    "ENABLED": True,               # externes Training aktiv
    "EPISODES_PER_BUNDLE": 10,     # nach X Episoden Rollouts schreiben
    "POLL_NEW_POLICY": True,       # nach jedem Bundle nach neuen Gewichten schauen
    "LATEST_TAG_FILE": "LATEST_POLICY.txt",   # Tag-Datei vom Trainer
    },

    # ======= Rewards =======
    # STEP-Varianten: "none" | "delta_weight_only" | "hand_penalty_coeff_only" | "combined"
    # FINAL-Varianten: "none" | "env_only" | "rank_bonus" | "both"
    "REWARD": {
        "STEP_MODE": "delta_weight_only",
        "DELTA_WEIGHT": 1.0,
        "HAND_PENALTY_COEFF": 1.0,

        "FINAL_MODE": "none",
        "BONUS_WIN": -10.0, "BONUS_2ND": 0.0, "BONUS_3RD": 0.0, "BONUS_LAST": 0.0,
    },

    # Features (Shared Policy: Seat-OneHot optional)
    "FEATURES": { "NORMALIZE": False, "SEAT_ONEHOT": False },

    # Benchmark-Gegner
    "BENCH_OPPONENTS": ["single_only", "max_combo", "random2"],
}

def _bundle_path(paths, bundle_idx):
    roll_dir = os.path.join(paths["run_dir"], "rollouts")
    os.makedirs(roll_dir, exist_ok=True)
    return os.path.join(roll_dir, f"bundle_{bundle_idx:06d}.npz")

def _dump_rollouts_npz(agent, path, meta):
    import numpy as np, os, uuid
    buf = agent._buffer

    # Meta als UTF-8 Bytes (NumPy 2.0: np.bytes_ statt np.string_)
    meta_bytes = meta.encode("utf-8")

    # Optional: atomar schreiben (tmp → rename), verhindert halbe Dateien
    tmp = f"{path}.tmp.{uuid.uuid4().hex}"

    np.savez_compressed(
        tmp,
        states=np.array(buf.states, dtype=np.float32),
        actions=np.array(buf.actions, dtype=np.int64),
        rewards=np.array(buf.rewards, dtype=np.float32),
        dones=np.array(buf.dones, dtype=np.bool_),
        old_log_probs=np.array(buf.log_probs, dtype=np.float32),
        values=np.array(buf.values, dtype=np.float32),
        legal_masks=np.array(buf.legal_masks, dtype=np.float32),
        player_ids=np.array([(-1 if pid is None else pid) for pid in buf.player_ids], dtype=np.int64),
        meta=np.array(meta_bytes, dtype=np.bytes_),  # <-- wichtig
    )

    os.replace(tmp, path)  # atomar ins Ziel verschieben



def main():
    np.random.seed(CONFIG["SEED"]); torch.manual_seed(CONFIG["SEED"])
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_ROOT = os.path.join(ROOT, "models")

    # ---- Verzeichnisse (k1a1-Stil) ----
    family = "k4a1"
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
    plotter.log("New Training (k4a1): Shared-Policy PPO")
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
    seat_id_dim = (num_players if CONFIG["FEATURES"]["SEAT_ONEHOT"] else 0)

    # ---- Agent / Shaper ----
    ppo_cfg = ppo.PPOConfig(**CONFIG["PPO"])
    agent = ppo.PPOAgent(info_dim, A, seat_id_dim=seat_id_dim, config=ppo_cfg)
    shaper = RewardShaper(CONFIG["REWARD"])


    # ---- Config & Meta ----
    save_config_csv({
        "script": family, "version": version,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent_type": "PPO_shared", "num_episodes": CONFIG["EPISODES"],
        "bench_interval": CONFIG["BENCH_INTERVAL"], "bench_episodes": CONFIG["BENCH_EPISODES"],
        "deck_size": CONFIG["DECK_SIZE"], "num_ranks": num_ranks,
        "observation_dim": info_dim + seat_id_dim, "num_actions": A,
        "normalize": CONFIG["FEATURES"]["NORMALIZE"], "seat_onehot": CONFIG["FEATURES"]["SEAT_ONEHOT"],
        "models_dir": paths["weights_dir"], "plots_dir": paths["plots_dir"],

        # Reward-Setup (nur neues System, direkt aus dem Shaper/CONFIG)
        "step_mode": shaper.step_mode,
        "delta_weight": shaper.dw,
        "hand_penalty_coeff": shaper.hp,
        "final_mode": shaper.final_mode,
        "bonus_win":   CONFIG["REWARD"]["BONUS_WIN"],
        "bonus_2nd":   CONFIG["REWARD"]["BONUS_2ND"],
        "bonus_3rd":   CONFIG["REWARD"]["BONUS_3RD"],
        "bonus_last":  CONFIG["REWARD"]["BONUS_LAST"],
    }, paths["config_csv"])

    save_run_meta({"family": family, "version": version, "algo": "ppo_shared", "deck": CONFIG["DECK_SIZE"]}, paths["run_meta_json"])

    # ---- Timing ----
    timer = TimingMeter(csv_path=paths["timings_csv"], interval=CONFIG["TIMING_INTERVAL"])
    t0 = time.perf_counter()
    BINT, BEPS = CONFIG["BENCH_INTERVAL"], CONFIG["BENCH_EPISODES"]

    external_cfg = CONFIG.get("EXTERNAL", {})
    EXT_ENABLED = bool(external_cfg.get("ENABLED", False))
    EPISODES_PER_BUNDLE = int(external_cfg.get("EPISODES_PER_BUNDLE", 10))
    POLL_NEW_POLICY = bool(external_cfg.get("POLL_NEW_POLICY", True))
    LATEST_TAG_FILE = str(external_cfg.get("LATEST_TAG_FILE", "LATEST_POLICY.txt"))

    current_policy_tag = "init"  # beliebiger Start-Tag
    bundle_ep_count = 0
    bundle_idx = 0

    # ---- Training ----
    for ep in range(1, CONFIG["EPISODES"] + 1):
        ep_start = time.perf_counter()
        ep_len = 0

        ts = env.reset()
        last_idx = {p: None for p in range(num_players)}  # letzte Transition je Sitz

        while not ts.last():
            p = ts.observations["current_player"]
            legal = ts.observations["legal_actions"][p]
            base_obs = ts.observations["info_state"][p]
            obs = augment_observation(base_obs, player_id=p, cfg=feat_cfg)

            seat_oh = None
            if CONFIG["FEATURES"]["SEAT_ONEHOT"]:
                seat_oh = np.zeros(num_players, dtype=np.float32); seat_oh[p] = 1.0

            # ✅ Handgröße VOR dem Schritt nur dann merken, wenn Step-Rewards aktiv sind
            if shaper.step_active():
                hand_before = shaper.hand_size(ts, p, deck_int)

            a = int(agent.step(obs, legal, seat_one_hot=seat_oh, player_id=p))
            last_idx[p] = len(agent._buffer.states) - 1

            ts = env.step([a])
            ep_len += 1

            # ✅ Step-Reward ausschließlich über den Shaper berechnen & verbuchen
            if shaper.step_active():
                hand_after = shaper.hand_size(ts, p, deck_int)
                step_r = shaper.step_reward(hand_before=hand_before, hand_after=hand_after)
                agent.post_step(step_r, done=ts.last())


        # ===== Episodenende: Final-Rewards auf letzte Transition je Spieler addieren =====
        finals = env._state.returns()

        for p in range(num_players):
            li = last_idx[p]
            if li is None:
                continue

            # ✅ ENV-Return nur, wenn der Shaper das sagt
            if shaper.include_env_reward():
                agent._buffer.rewards[li] += float(finals[p])

            # ✅ Benutzerdefinierter Rank-Bonus nur via Shaper
            agent._buffer.rewards[li] += float(shaper.final_bonus(finals, p))

            # ✅ Terminal markieren (wichtig für GAE)
            agent._buffer.dones[li] = True


        # ===== Rollout-Bundling (externes Training) =====
        train_start = time.perf_counter()

        if EXT_ENABLED:
            bundle_ep_count += 1

            if bundle_ep_count >= EPISODES_PER_BUNDLE:
                # 1) Rollouts -> NPZ
                import json
                meta = json.dumps({
                    "algo": "ppo",
                    "family": family,
                    "version": version,
                    "deck_size": CONFIG["DECK_SIZE"],
                    "num_actions": A,
                    "obs_dim": info_dim + seat_id_dim,
                    "num_players": num_players,
                    "ppo_config": CONFIG["PPO"],
                    "reward_config": CONFIG["REWARD"],
                    "policy_tag_used": current_policy_tag,
                    "bundle_idx": bundle_idx,
                })
                out_path = _bundle_path(paths, bundle_idx)
                _dump_rollouts_npz(agent, out_path, meta)
                # 2) Buffer leeren (wichtig: NICHT trainieren!)
                agent._buffer.clear()
                bundle_ep_count = 0
                bundle_idx += 1
                print(f"[k4] wrote rollout bundle: {out_path}")

                # 3) Optional: neue Policy laden
                if POLL_NEW_POLICY:
                    tag_file = os.path.join(paths["weights_dir"], LATEST_TAG_FILE)
                    if os.path.isfile(tag_file):
                        try:
                            with open(tag_file, "r") as f:
                                new_tag = f.read().strip()
                            if new_tag and new_tag != current_policy_tag:
                                base = os.path.join(paths["weights_dir"], new_tag)
                                agent.restore(base)
                                current_policy_tag = new_tag
                                print(f"[k4] loaded new policy tag: {new_tag}")
                        except Exception as e:
                            print(f"[k4] WARNING: failed to load new policy: {e}")

            # Keine Trainingsmetriken, wir haben nicht aktualisiert
            train_metrics = {}
            train_seconds = 0.0
        else:
            # Fallback: interner Trainer (wie früher)
            train_metrics = agent.train()
            train_seconds = time.perf_counter() - train_start


        if train_metrics is None:
            train_metrics = {}
        train_metrics.update({
            "train_seconds": train_seconds,
            "ep_length": ep_len,
            "ep_env_return": float(finals[0]),  # Player-0 als Referenz
        })
        plotter.add_train(ep, train_metrics)

        # ===== Benchmark & Save (nur Player-0) =====
        eval_seconds = plot_seconds = save_seconds = 0.0
        if ep % BINT == 0:
            ev_start = time.perf_counter()
            per_opponent = run_benchmark(
                game=game, agent=agent, opponents_dict=STRATS,
                opponent_names=CONFIG["BENCH_OPPONENTS"], episodes=BEPS,
                feat_cfg=feat_cfg, num_actions=A
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
            tag = f"{family}_model_{version}_agent_p0_ep{ep:07d}"
            save_checkpoint_ppo(agent, paths["weights_dir"], tag)
            save_seconds = time.perf_counter() - save_start

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

        # Timing CSV
        ep_seconds = time.perf_counter() - ep_start
        timer.maybe_log(ep, {
            "steps": ep_len,
            "ep_seconds": ep_seconds,
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds,
            "plot_seconds": plot_seconds,
            "save_seconds": save_seconds,
        })

    total_seconds = time.perf_counter() - t0
    plotter.log("")
    plotter.log(f"Gesamtzeit: {total_seconds/3600:0.2f}h (~ {CONFIG['EPISODES']/max(total_seconds,1e-9):0.2f} eps/s)")
    plotter.log("K4 (Shared-Policy) Training abgeschlossen.")

if __name__ == "__main__":
    main()
