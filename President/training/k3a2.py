# -*- coding: utf-8 -*-
# President/training/k3a2_test.py
# DQN (K3): Snapshot-Selfplay – 1 Learner + 3 Sparring (analog zu k3a1)

import os, datetime, time, copy, numpy as np, torch
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

# ============== CONFIG ==============
CONFIG = {
    "EPISODES":         10_000,
    "BENCH_INTERVAL":   500,
    "BENCH_EPISODES":   2000,
    "TIMING_INTERVAL":  200,
    "DECK_SIZE":        "64",  # "12" | "16" | "20" | "24" | "32" | "52" | "64"
    "SEED":             42,

    # DQN (Schlüssel passen zu DQNConfig in dqn_agent.py)
    "DQN": {
        "learning_rate": 3e-4,
        "batch_size": 128,
        "gamma": 0.995,
        "buffer_size": 200_000,
        "target_update_freq": 5000,
        "soft_target_tau": 0.0,
        "max_grad_norm": 1.0,
        "n_step": 3,
        "per_alpha": 0.6,
        "per_beta_start": 0.4,
        "per_beta_frames": 1_000_000,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_decay_frames": 500_000,
        "loss_huber_delta": 1.0,
        "dueling": True,
        "device": "cpu",
    },

    # ======= Rewards (NEUES System, wie k3a1) =======
    # STEP_MODE : "none" | "delta_weight_only" | "hand_penalty_coeff_only" | "combined"
    # FINAL_MODE: "none" | "env_only" | "rank_bonus" | "both"
    "REWARD": {
        "STEP_MODE": "none",
        "DELTA_WEIGHT": 1.0,
        "HAND_PENALTY_COEFF": 0.0,

        "FINAL_MODE": "env_only",
        "BONUS_WIN": 0.0, "BONUS_2ND": 0.0, "BONUS_3RD": 0.0, "BONUS_LAST": 0.0,
    },

    # Features (K3: OneHot oft AN; wird in die Observation augmentiert)
    "FEATURES": { "NORMALIZE": False, "SEAT_ONEHOT": True },

    # Snapshot-Selfplay (wie k3a1)
    "SNAPSHOT": {
        "LEARNER_SEAT": 0,
        "MIX_CURRENT": 0.8,           # Anteil „current“ vs. Snapshot bei Seats 1–3
        "SNAPSHOT_INTERVAL": 200,     # bewusst kurz für sichtbare Pool-Entwicklung
        "POOL_CAP": 20,
    },

    # Benchmark-Gegner
    "BENCH_OPPONENTS": ["single_only", "max_combo", "random2"],
}

# ===== Greedy-Aktion (ohne ε, mit Legal-Maske & Tie-Break) =====
@torch.no_grad()
def greedy_action(q_module: torch.nn.Module, obs_vec: np.ndarray, legal_actions, device=None) -> int:
    if device is None:
        try:
            device = next(q_module.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    s = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
    qvals = q_module(s).squeeze(0).detach().cpu().numpy()
    masked = np.full_like(qvals, -1e9, dtype=np.float32)
    idx = list(legal_actions)
    masked[idx] = qvals[idx]
    maxv = masked.max()
    cands = np.flatnonzero(masked == maxv)
    return int(np.random.choice(cands))

# ===== Eingefrorene Snapshot-Policy für DQN (immer greedy) =====
class SnapshotDQNA:
    """Frozen DQN-Policy: lädt Q-Snapshot und agiert greedy (argmax) über legal actions."""
    def __init__(self, state_size, num_actions, dqn_cfg, q_state_dict):
        self.agent = dqn.DQNAgent(state_size=state_size, num_actions=num_actions, cfg=dqn_cfg)
        # WICHTIG: aktueller Code nutzt .q (nicht .q_network)
        self.agent.q.load_state_dict(copy.deepcopy(q_state_dict))
        self.agent.tgt.load_state_dict(self.agent.q.state_dict())
        self.agent.tgt.eval()

    @torch.no_grad()
    def act(self, obs_vec: np.ndarray, legal_actions):
        return greedy_action(self.agent.q, obs_vec, legal_actions)

# =========================== Training ===========================
def main():
    np.random.seed(CONFIG["SEED"]); torch.manual_seed(CONFIG["SEED"])
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_ROOT = os.path.join(ROOT, "models")

    # ---- Verzeichnisse (k1-Stil) ----
    family = "k3a2"
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
    plotter.log("New Training (k3a2): Snapshot-Selfplay DQN")
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
    deck_int = int(CONFIG["DECK_SIZE"])
    num_ranks = ranks_for_deck(deck_int)
    base_dim  = game.observation_tensor_shape()[0]
    extra_dim = (num_players if CONFIG["FEATURES"]["SEAT_ONEHOT"] else 0)
    state_size = base_dim + extra_dim

    feat_cfg = FeatureConfig(
        num_players=num_players, num_ranks=num_ranks,
        add_seat_onehot=CONFIG["FEATURES"]["SEAT_ONEHOT"],
        normalize=CONFIG["FEATURES"]["NORMALIZE"],
    )

    # ---- Agent & Reward ----
    dqn_cfg = dqn.DQNConfig(**CONFIG["DQN"])
    learner = dqn.DQNAgent(state_size=state_size, num_actions=A, cfg=dqn_cfg)
    shaper  = RewardShaper(CONFIG["REWARD"])

    # ---- Config & Meta speichern ----
    save_config_csv({
        "script": family, "version": version,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent_type": "DQN_snapshot", "num_episodes": CONFIG["EPISODES"],
        "bench_interval": CONFIG["BENCH_INTERVAL"], "bench_episodes": CONFIG["BENCH_EPISODES"],
        "deck_size": CONFIG["DECK_SIZE"], "num_ranks": num_ranks,
        "observation_dim": state_size, "num_actions": A,
        "normalize": CONFIG["FEATURES"]["NORMALIZE"], "seat_onehot": CONFIG["FEATURES"]["SEAT_ONEHOT"],
        "models_dir": paths["weights_dir"], "plots_dir": paths["plots_dir"],
        "mix_current": CONFIG["SNAPSHOT"]["MIX_CURRENT"],
        "snapshot_interval": CONFIG["SNAPSHOT"]["SNAPSHOT_INTERVAL"],
        "pool_cap": CONFIG["SNAPSHOT"]["POOL_CAP"],
        "learner_seat": CONFIG["SNAPSHOT"]["LEARNER_SEAT"],

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
    save_run_meta({"family": family, "version": version, "algo": "dqn_snapshot", "deck": CONFIG["DECK_SIZE"]},
                  paths["run_meta_json"])

    # ---- Snapshot-Pool ----
    pool: list[dict] = []
    MIX       = float(CONFIG["SNAPSHOT"]["MIX_CURRENT"])
    SNAPINT   = int(CONFIG["SNAPSHOT"]["SNAPSHOT_INTERVAL"])
    POOL_CAP  = int(CONFIG["SNAPSHOT"]["POOL_CAP"])
    LEARN_SEAT= int(CONFIG["SNAPSHOT"]["LEARNER_SEAT"])

    # ---- Timing ----
    timer = TimingMeter(csv_path=paths["timings_csv"], interval=CONFIG["TIMING_INTERVAL"])
    t0 = time.perf_counter()
    BINT, BEPS = CONFIG["BENCH_INTERVAL"], CONFIG["BENCH_EPISODES"]

    # ================= Loop =================
    for ep in range(1, CONFIG["EPISODES"] + 1):
        ep_start = time.perf_counter()
        ep_len = 0
        ep_shaping_return = 0.0

        ts = env.reset()
        last_idx_learner = None  # Index der letzten Transition (für Terminal-Boni)

        # Seats != Learner: aktuelle Policy (greedy) oder Snapshot (greedy)
        seat_actor = {}
        for seat in [s for s in range(num_players) if s != LEARN_SEAT]:
            use_current = (len(pool) == 0) or (np.random.rand() < MIX)
            if use_current:
                seat_actor[seat] = "current"
            else:
                sd = pool[np.random.randint(len(pool))]
                seat_actor[seat] = SnapshotDQNA(state_size, A, dqn_cfg, sd)

        while not ts.last():
            p = ts.observations["current_player"]
            legal = ts.observations["legal_actions"][p]

            # C++-Observation (konsistent zu state_size)
            base_obs = np.array(env._state.observation_tensor(p), dtype=np.float32)
            obs = augment_observation(base_obs, player_id=p, cfg=feat_cfg)

            # Step-Reward-Prep (nur, wenn aktiv)
            if shaper.step_active():
                hand_before = shaper.hand_size(ts, p, deck_int)

            if p == LEARN_SEAT:
                # Learner nutzt ε-greedy
                a = int(learner.select_action(obs, legal))
            else:
                # Sparring: immer greedy (kein ε)
                if seat_actor[p] == "current":
                    a = int(greedy_action(learner.q, obs, legal))
                else:
                    a = int(seat_actor[p].act(obs, legal))

            ts_next = env.step([a])
            ep_len += 1

            if p == LEARN_SEAT:
                # Next-Obs & optionale Legal-Maske (robust auch im Terminal)
                if not ts_next.last():
                    base_next = np.array(env._state.observation_tensor(p), dtype=np.float32)
                    obs_next = augment_observation(base_next, player_id=p, cfg=feat_cfg)
                    next_legals = ts_next.observations["legal_actions"][p]
                else:
                    obs_next = obs
                    next_legals = None  # Maske optional

                # Step-Reward anwenden (falls aktiv)
                r = 0.0
                if shaper.step_active():
                    hand_after = shaper.hand_size(ts_next, p, deck_int)
                    r = float(shaper.step_reward(hand_before=hand_before, hand_after=hand_after))
                    ep_shaping_return += r

                # Transition speichern (inkl. optionaler Legal-Maske)
                learner.store(
                    state=obs, action=int(a), reward=float(r),
                    next_state=obs_next, done=bool(ts_next.last()),
                    next_legal_actions=next_legals
                )
                last_idx_learner = len(learner.buffer.buffer) - 1

                # Online-Train auf Learner-Transitions
                learner.train_step()

            ts = ts_next

        # ---- Episodenende: ENV/Bonus auf letzte Learner-Transition addieren ----
        ep_env_return  = float(ts.rewards[LEARN_SEAT])
        ep_final_bonus = float(shaper.final_bonus(ts.rewards, LEARN_SEAT))
        if last_idx_learner is not None:
            bonus = 0.0
            if shaper.include_env_reward():
                bonus += ep_env_return
            bonus += ep_final_bonus
            if abs(bonus) > 1e-8:
                buf = learner.buffer
                old = buf.buffer[last_idx_learner]
                # PERBuffer.Exp Felder: ("s","a","r","ns","done","next_mask")
                buf.buffer[last_idx_learner] = buf.Exp(
                    old.s, old.a, float(old.r + bonus), old.ns, old.done, old.next_mask
                )

        # ---- Snapshot in Pool? ----
        if SNAPINT > 0 and (ep % SNAPINT == 0):
            pool.append(copy.deepcopy(learner.q.state_dict()))
            if len(pool) > POOL_CAP:
                pool.pop(0)

        # ---- Training-Metriken loggen ----
        plotter.add_train(ep, {
            "ep_length":         ep_len,
            "ep_env_return":     ep_env_return,
            "ep_shaping_return": ep_shaping_return,
            "ep_final_bonus":    ep_final_bonus,
        })

        # ---- Benchmark & Save (nur Learner-Policy) ----
        eval_seconds = plot_seconds = save_seconds = 0.0
        if BINT > 0 and (ep % BINT == 0):
            ev_start = time.perf_counter()
            per_opponent = run_benchmark(
                game=game,
                agent=learner,
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
            plotter.plot_benchmark(filename_prefix="lernkurve", with_macro=True)
            plotter.plot_train(filename_prefix="training_metrics", separate=True)
            plot_seconds = time.perf_counter() - plot_start

            save_start = time.perf_counter()
            tag = f"{family}_model_{version}_agent_p{LEARN_SEAT}_ep{ep:07d}"
            learner.save(os.path.join(paths["weights_dir"], tag))
            save_seconds = time.perf_counter() - save_start

            cum_seconds = time.perf_counter() - t0
            plotter.log_timing(
                ep,
                ep_seconds=(time.perf_counter() - ep_start),
                train_seconds=0.0,  # Online-Train oben eingerechnet, hier optional separat loggen
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
    plotter.log("K3 (Snapshot-Selfplay, DQN) Training abgeschlossen.")

if __name__ == "__main__":
    main()
