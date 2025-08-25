# -*- coding: utf-8 -*-
# President/training/k3a1.py — PPO (K3): Snapshot-Selfplay, 1 Learner + 3 Sparring
# Hinweis: analog zu k1a1-Änderungen:
#  - TimingMeter entfernt, Metrics/Plots optional per Flags
#  - FeatureConfig mit include_history; Seat-One-Hot nicht in augment_observation
#  - Dimensionsberechnung über expected_feature_len/expected_input_dim
#  - umfangreicheres Config-Logging
#  - größeres Standard-Setup (Episoden/Bench/Deck)

import os, datetime, time, copy, numpy as np, torch
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

# ============== CONFIG ==============
CONFIG = {
    "EPISODES":         10000,
    "BENCH_INTERVAL":   2000,
    "BENCH_EPISODES":   2000,
    "DECK_SIZE":        "64",  # "12" | "16" | "20" | "24" | "32" | "52" | "64"
    "SEED":             42,

    # PPO
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
    # FINAL_MODE: "none" | "env_only" | "rank_bonus" | "both"
    "REWARD": {
        "STEP_MODE": "none",
        "DELTA_WEIGHT": 0.5,
        "HAND_PENALTY_COEFF": 0.0,

        "FINAL_MODE": "env_only",
        "BONUS_WIN": 10.0, "BONUS_2ND": 0.0, "BONUS_3RD": 0.0, "BONUS_LAST": 0.0,
    },

    # Features (analog k1a1)
    "FEATURES": {
        "USE_HISTORY": False,     # True = mit Historie
        "SEAT_ONEHOT": False,     # optional: Sitz-One-Hot dem Agent geben
        "PLOT_METRICS": False,    # Trainingsplots erzeugen?
        "SAVE_METRICS_TO_CSV": False,  # Trainingsmetriken persistent speichern?
    },

    # Snapshot-Selfplay
    "SNAPSHOT": {
        "LEARNER_SEAT": 0,
        "MIX_CURRENT": 0.8,            # Wahrscheinlichkeit, Seats 1–3 mit aktueller Policy spielen zu lassen
        "SNAPSHOT_INTERVAL": 200,      # relativ kurz, damit viele Snapshots entstehen
        "POOL_CAP": 20,
    },

    # Benchmark-Gegner
    "BENCH_OPPONENTS": ["single_only", "max_combo", "random2"],
}

# ===== Helfer für Snapshot-Policies =====
class SnapshotPolicy(torch.nn.Module):
    """Eingefrorene Policy (nur Inferenz). Erwartet denselben Input wie PPOAgent._make_input."""
    def __init__(self, input_dim: int, num_actions: int, state_dict: dict):
        super().__init__()
        self.net = ppo.PolicyNetwork(input_dim, num_actions)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self.num_actions = num_actions

    @torch.no_grad()
    def act(self, x_vec: np.ndarray, legal_actions):
        device = next(self.net.parameters()).device
        x_t = torch.tensor(x_vec, dtype=torch.float32, device=device).unsqueeze(0)
        logits = self.net(x_t).squeeze(0)
        mask = torch.zeros(self.num_actions, dtype=torch.float32, device=device)
        mask[legal_actions] = 1.0
        probs = ppo.masked_softmax(logits, mask)
        return int(torch.distributions.Categorical(probs=probs).sample().item())

def main():
    np.random.seed(CONFIG["SEED"]); torch.manual_seed(CONFIG["SEED"])
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_ROOT = os.path.join(ROOT, "models")

    # ---- Verzeichnisse (k1a1-Stil) ----
    family = "k3a1"  # eigene Familie für Snapshot-Selfplay
    family_dir = os.path.join(MODELS_ROOT, family)
    version = find_next_version(family_dir, prefix="model")
    paths = prepare_run_dirs(MODELS_ROOT, family, version, prefix="model")

    # ---- Plotter / Logger ----
    plotter = MetricsPlotter(
        out_dir=paths["plots_dir"],
        benchmark_opponents=list(CONFIG["BENCH_OPPONENTS"]),
        benchmark_csv="benchmark_curves.csv",
        train_csv="training_metrics.csv",
        save_csv=True, verbosity=1,
    )
    plotter.log("New Training (k3a1): Snapshot-Selfplay PPO")
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
    A = env.action_spec()["num_actions"]
    num_players = game.num_players()

    # ---- Features ----
    deck_int  = int(CONFIG["DECK_SIZE"])
    num_ranks = ranks_for_deck(deck_int)

    # Seat-One-Hot NICHT in augment_observation anhängen; optional separat über seat_one_hot
    feat_cfg = FeatureConfig(
        num_players=num_players,
        num_ranks=num_ranks,
        add_seat_onehot=False,                          # <- immer False lassen
        include_history=CONFIG["FEATURES"]["USE_HISTORY"],
    )
    seat_id_dim = (num_players if CONFIG["FEATURES"]["SEAT_ONEHOT"] else 0)

    # ✅ Agent-Inputgrößen sauber bestimmen
    info_dim = expected_feature_len(feat_cfg)  # Basis-Features (ohne Seat-One-Hot)

    # ---- Agent / Shaper ----
    ppo_cfg = ppo.PPOConfig(**CONFIG["PPO"])
    agent = ppo.PPOAgent(info_dim, A, seat_id_dim=seat_id_dim, config=ppo_cfg)
    shaper = RewardShaper(CONFIG["REWARD"])

    # ---- Config & Meta speichern ----
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

        "agent_type": "PPO_snapshot", "num_episodes": CONFIG["EPISODES"],
        "bench_interval": CONFIG["BENCH_INTERVAL"], "bench_episodes": CONFIG["BENCH_EPISODES"],
        "observation_dim": expected_input_dim(feat_cfg),  # inkl. Seat-One-Hot, falls aktiv
        "num_actions": A,
        "seat_onehot": CONFIG["FEATURES"]["SEAT_ONEHOT"],

        "models_dir": paths["weights_dir"], "plots_dir": paths["plots_dir"],
        "mix_current": CONFIG["SNAPSHOT"]["MIX_CURRENT"],
        "snapshot_interval": CONFIG["SNAPSHOT"]["SNAPSHOT_INTERVAL"],
        "pool_cap": CONFIG["SNAPSHOT"]["POOL_CAP"],
        "learner_seat": CONFIG["SNAPSHOT"]["LEARNER_SEAT"],
        "benchmark_opponents": ",".join(CONFIG["BENCH_OPPONENTS"]),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }, paths["config_csv"])
    save_run_meta({"family": family, "version": version, "algo": "ppo_snapshot", "deck": CONFIG["DECK_SIZE"]}, paths["run_meta_json"])

    # ---- Snapshot-Pool ----
    pool: list[dict] = []
    MIX = float(CONFIG["SNAPSHOT"]["MIX_CURRENT"])
    SNAPINT = int(CONFIG["SNAPSHOT"]["SNAPSHOT_INTERVAL"])
    POOL_CAP = int(CONFIG["SNAPSHOT"]["POOL_CAP"])
    LEARNER_SEAT = int(CONFIG["SNAPSHOT"]["LEARNER_SEAT"])

    t0 = time.perf_counter()
    BINT, BEPS = CONFIG["BENCH_INTERVAL"], CONFIG["BENCH_EPISODES"]

    # Optionales Sammeln von Metriken (analog k1a1)
    collect_metrics = (
        CONFIG["FEATURES"].get("SAVE_METRICS_TO_CSV", False)
        or CONFIG["FEATURES"].get("PLOT_METRICS", False)
    )

    # ---- Training ----
    for ep in range(1, CONFIG["EPISODES"] + 1):
        ep_start = time.perf_counter()
        ep_len = 0
        ep_shaping_return = 0.0

        ts = env.reset()
        last_idx_learner = None  # Index der letzten Transition des Learner-Seats

        # Seats != LEARNER_SEAT: aktuelle Policy oder Snapshot
        seat_actor = {}
        for seat in [s for s in range(num_players) if s != LEARNER_SEAT]:
            use_current = (len(pool) == 0) or (np.random.rand() < MIX)
            if use_current:
                seat_actor[seat] = "current"
            else:
                sd = pool[np.random.randint(len(pool))]
                # Wichtig: Eingabedimension = expected_input_dim(feat_cfg)
                seat_actor[seat] = SnapshotPolicy(expected_input_dim(feat_cfg), A, sd)

        while not ts.last():
            p = ts.observations["current_player"]
            legal = ts.observations["legal_actions"][p]

            base_obs = ts.observations["info_state"][p]
            obs = augment_observation(base_obs, player_id=p, cfg=feat_cfg)

            seat_oh = None
            if CONFIG["FEATURES"]["SEAT_ONEHOT"]:
                seat_oh = np.zeros(num_players, dtype=np.float32); seat_oh[p] = 1.0

            # Step-Reward-Prep
            if shaper.step_active():
                hand_before = shaper.hand_size(ts, p, deck_int)

            if p == LEARNER_SEAT:
                a = int(agent.step(obs, legal, seat_one_hot=seat_oh, player_id=p))
                last_idx_learner = len(agent._buffer.states) - 1
            else:
                actor = seat_actor[p]
                if actor == "current":
                    x = agent._make_input(obs, seat_one_hot=seat_oh)
                    with torch.no_grad():
                        logits = agent._policy(x)
                        mask = torch.zeros(A, device=logits.device); mask[legal] = 1.0
                        probs = ppo.masked_softmax(logits, mask)
                    a = int(torch.distributions.Categorical(probs=probs).sample().item())
                else:
                    x = agent._make_input(obs, seat_one_hot=seat_oh).cpu().numpy()
                    a = int(actor.act(x, legal))

            ts_next = env.step([a])
            ep_len += 1

            if p == LEARNER_SEAT and shaper.step_active():
                hand_after = shaper.hand_size(ts_next, p, deck_int)
                r = shaper.step_reward(hand_before=hand_before, hand_after=hand_after)
                ep_shaping_return += float(r)
                agent.post_step(r, done=ts_next.last())

            ts = ts_next

        # Finale Returns/Bonis für Lern-Sitz
        ep_env_return  = float(ts.rewards[LEARNER_SEAT])
        ep_final_bonus = float(shaper.final_bonus(ts.rewards, LEARNER_SEAT))

        if last_idx_learner is not None:
            if shaper.include_env_reward():
                agent._buffer.rewards[last_idx_learner] += ep_env_return
            agent._buffer.rewards[last_idx_learner] += ep_final_bonus
            agent._buffer.dones[last_idx_learner] = True

        # Train
        train_start = time.perf_counter()
        train_metrics = agent.train()
        train_seconds = time.perf_counter() - train_start

        # Trainingsmetriken sammeln / speichern analog k1a1
        metrics = train_metrics or {}
        metrics.update({
            "train_seconds":      train_seconds,
            "ep_env_return":      ep_env_return,
            "ep_shaping_return":  ep_shaping_return,
            "ep_final_bonus":     ep_final_bonus,
            "ep_length":          ep_len,
        })
        if collect_metrics:
            if CONFIG["FEATURES"].get("SAVE_METRICS_TO_CSV", False):
                plotter.add_train(ep, metrics)
            else:
                plotter.train_rows.append({"episode": int(ep), **metrics})
                if plotter.train_keys is None:
                    plotter.train_keys = ["episode"] + list(metrics.keys())

        # Snapshot in Pool?
        if SNAPINT > 0 and (ep % SNAPINT == 0):
            pool.append(copy.deepcopy(agent._policy.state_dict()))
            if len(pool) > POOL_CAP:
                pool.pop(0)

        # Benchmark & Save (nur Learner-Policy)
        eval_seconds = plot_seconds = save_seconds = 0.0
        if BINT > 0 and (ep % BINT == 0):
            ev_start = time.perf_counter()
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

            plot_start = time.perf_counter()
            plotter.add_benchmark(ep, per_opponent)
            plotter.plot_benchmark_rewards()
            plotter.plot_places_latest()

            # Einheitliche Titel:
            # - Einzelplots:  "Lernkurve - KxAy vs <gegner>"
            # - Multi/Macro:  "Lernkurve - KxAy vs feste Heuristiken"
            title_multi = f"Lernkurve - {family.upper()} vs feste Heuristiken"
            plotter.plot_benchmark(
                filename_prefix="lernkurve",
                with_macro=True,
                family_title=family.upper(),   # für Einzelplots
                multi_title=title_multi,       # für Multi- & Macro-Plot
            )

            if CONFIG["FEATURES"].get("PLOT_METRICS", False):
                plotter.plot_train(filename_prefix="training_metrics", separate=True)

            plot_seconds = time.perf_counter() - plot_start

            save_start = time.perf_counter()
            tag = f"{family}_model_{version}_agent_p{LEARNER_SEAT}_ep{ep:07d}"
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

    total_seconds = time.perf_counter() - t0
    plotter.log("")
    plotter.log(f"Gesamtzeit: {total_seconds/3600:0.2f}h (~ {CONFIG['EPISODES']/max(total_seconds,1e-9):0.2f} eps/s)")
    plotter.log(f"{family}, Snapshot Selfplay (1 Learner, 3 Sparring). Training abgeschlossen.")
    plotter.log(f"Path: {paths['run_dir']}")

if __name__ == "__main__":
    main()
