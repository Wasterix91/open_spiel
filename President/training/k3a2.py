# -*- coding: utf-8 -*-
# President/training/k3a2.py — DQN Snapshot-Selfplay (1 Learner + 3 Sparring)
# Fixes:
#  - Decision-to-Decision: Learner-Transitions werden erst geschlossen, wenn der Learner wieder am Zug ist
#  - next_legal_actions nur dann setzen, wenn DERSELBE Spieler wieder zieht (sonst done/Terminal)
#  - robuster DQNConfig-Build aus DEFAULT_CONFIG (kompatibel zu agents/dqn_agent.py)

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
    "EPISODES":         2000,
    "BENCH_INTERVAL":   500,
    "BENCH_EPISODES":   500,
    "TIMING_INTERVAL":  500,
    "DECK_SIZE":        "16",  # "12" | "16" | "20" | "24" | "32" | "52" | "64"
    "SEED":             42,

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

    # Learner bekommt Seat-OneHot via augment_observation (hier True)
    "FEATURES": { "NORMALIZE": False, "SEAT_ONEHOT": True },

    "SNAPSHOT": { "LEARNER_SEAT": 0, "MIX_CURRENT": 0.8, "SNAPSHOT_INTERVAL": 200, "POOL_CAP": 20 },

    "BENCH_OPPONENTS": ["single_only", "max_combo", "random2"],
}

# ===== Greedy-Aktion (Q argmax über legal actions) =====
@torch.no_grad()
def greedy_action(q_module: torch.nn.Module, obs_vec: np.ndarray, legal_actions, device=None) -> int:
    if device is None:
        try:
            device = next(q_module.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    s = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
    qvals = q_module(s).squeeze(0).detach().cpu().numpy()
    masked = np.full_like(qvals, -np.inf, dtype=np.float32)
    idx = list(legal_actions)
    masked[idx] = qvals[idx]
    maxv = masked.max()
    cands = np.flatnonzero(masked == maxv)
    return int(np.random.choice(cands))

# ===== Eingefrorene Snapshot-DQN (immer greedy) =====
class SnapshotDQNA:
    def __init__(self, state_size, num_actions, dqn_cfg, q_state_dict):
        self.agent = dqn.DQNAgent(state_size=state_size, num_actions=num_actions, config=dqn_cfg)
        self.agent.target_network.eval()
        self.agent.q_network.load_state_dict(copy.deepcopy(q_state_dict))
        self.agent.target_network.load_state_dict(self.agent.q_network.state_dict())
        self.agent.target_network.eval()

    @torch.no_grad()
    def act(self, obs_vec: np.ndarray, legal_actions):
        return greedy_action(self.agent.q_network, obs_vec, legal_actions)

# =========================== Training ===========================
def main():
    np.random.seed(CONFIG["SEED"]); torch.manual_seed(CONFIG["SEED"])
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_ROOT = os.path.join(ROOT, "models")

    family = "k3a2"
    version = find_next_version(os.path.join(MODELS_ROOT, family), prefix="model")
    paths = prepare_run_dirs(MODELS_ROOT, family, version, prefix="model")

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
    extra_dim = num_players if CONFIG["FEATURES"]["SEAT_ONEHOT"] else 0
    state_size = base_dim + extra_dim

    feat_cfg = FeatureConfig(
        num_players=num_players, num_ranks=num_ranks,
        add_seat_onehot=CONFIG["FEATURES"]["SEAT_ONEHOT"],
        normalize=CONFIG["FEATURES"]["NORMALIZE"],
    )

    # ---- Agent & Shaper ----
    base_cfg = dqn.DEFAULT_CONFIG
    overrides = {k: CONFIG["DQN"][k] for k in CONFIG["DQN"] if k in base_cfg._fields}
    dqn_cfg = base_cfg._replace(**overrides)

    learner = dqn.DQNAgent(state_size=state_size, num_actions=A, config=dqn_cfg, device="cpu")
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
        "step_mode": shaper.step_mode, "delta_weight": shaper.dw,
        "hand_penalty_coeff": shaper.hp, "final_mode": shaper.final_mode,
        "bonus_win": CONFIG["REWARD"]["BONUS_WIN"], "bonus_2nd": CONFIG["REWARD"]["BONUS_2ND"],
        "bonus_3rd": CONFIG["REWARD"]["BONUS_3RD"], "bonus_last": CONFIG["REWARD"]["BONUS_LAST"],
    }, paths["config_csv"])
    save_run_meta({"family": family, "version": version, "algo": "dqn_snapshot", "deck": CONFIG["DECK_SIZE"]},
                  paths["run_meta_json"])

    # ---- Snapshot-Pool ----
    pool: list[dict] = []
    MIX       = float(CONFIG["SNAPSHOT"]["MIX_CURRENT"])
    SNAPINT   = int(CONFIG["SNAPSHOT"]["SNAPSHOT_INTERVAL"])
    POOL_CAP  = int(CONFIG["SNAPSHOT"]["POOL_CAP"])
    LEARN     = int(CONFIG["SNAPSHOT"]["LEARNER_SEAT"])

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
        last_idx_learner = None
        pending = None   # offene Transition des Learners: {"s":..., "a":..., "r":...}

        # Gegner Seats auswählen (current vs Snapshot)
        seat_actor = {}
        for seat in [s for s in range(num_players) if s != LEARN]:
            if (len(pool) == 0) or (np.random.rand() < MIX):
                seat_actor[seat] = "current"
            else:
                sd = pool[np.random.randint(len(pool))]
                seat_actor[seat] = SnapshotDQNA(state_size, A, dqn_cfg, sd)

        while not ts.last():
            p = ts.observations["current_player"]
            legal = ts.observations["legal_actions"][p]

            # Wenn der Learner wieder dran ist, eine evtl. offene Transition schließen (decision-to-decision)
            if (p == LEARN) and (pending is not None):
                base_now = np.array(env._state.observation_tensor(LEARN), dtype=np.float32)
                s_now = augment_observation(base_now, player_id=LEARN, cfg=feat_cfg)
                learner.buffer.add(pending["s"], pending["a"], pending["r"], s_now, False,
                                   next_legal_actions=legal)  # legal gehört jetzt dem Learner!
                last_idx_learner = len(learner.buffer.buffer) - 1
                learner.train_step()
                pending = None

            # aktuelle Beobachtung des aktiven Spielers
            base_obs = np.array(env._state.observation_tensor(p), dtype=np.float32)
            obs = augment_observation(base_obs, player_id=p, cfg=feat_cfg)

            # Step-Shaping Vorbereitung
            if shaper.step_active():
                hand_before = shaper.hand_size(ts, p, deck_int)

            # Aktion wählen
            if p == LEARN:
                a = int(learner.select_action(obs, legal))
            else:
                a = int(greedy_action(learner.q_network, obs, legal)) if seat_actor[p] == "current" \
                    else int(seat_actor[p].act(obs, legal))

            # Schritt in der Env
            ts_next = env.step([a]); ep_len += 1

            # Step-Shaping nur für den Learner speichern
            if p == LEARN:
                r = 0.0
                if shaper.step_active():
                    hand_after = shaper.hand_size(ts_next, p, deck_int)
                    r = float(shaper.step_reward(hand_before=hand_before, hand_after=hand_after))
                    ep_shaping_return += r

                # Learner-Transition offen halten bis er wieder am Zug ist oder Terminal
                assert pending is None
                pending = {"s": obs, "a": a, "r": r}

            ts = ts_next

        # Episodenende: evtl. offene Learner-Transition finalisieren (done=True)
        if pending is not None:
            s_next = pending["s"]  # Dummy-Next-State ok; Maske „alle legal“ (oder None)
            learner.buffer.add(pending["s"], pending["a"], pending["r"], s_next, True,
                               next_legal_actions=list(range(A)))
            last_idx_learner = len(learner.buffer.buffer) - 1
            learner.train_step()
            pending = None

        # Finale Rewards / Bonus auf die letzte Learner-Transition addieren
        ep_env = float(ts.rewards[LEARN])
        ep_bonus = float(shaper.final_bonus(ts.rewards, LEARN))
        if last_idx_learner is not None:
            add = (ep_env if shaper.include_env_reward() else 0.0) + ep_bonus
            if abs(add) > 1e-8:
                buf = learner.buffer
                old = buf.buffer[last_idx_learner]
                buf.buffer[last_idx_learner] = buf.Experience(
                    old.state, old.action, float(old.reward + add),
                    old.next_state, old.done, old.next_legal_mask
                )

        # Snapshot in Pool?
        if SNAPINT > 0 and (ep % SNAPINT == 0):
            pool.append(copy.deepcopy(learner.q_network.state_dict()))
            if len(pool) > POOL_CAP:
                pool.pop(0)

        # Training-Metriken
        plotter.add_train(ep, {
            "ep_length":         ep_len,
            "ep_env_return":     ep_env,
            "ep_shaping_return": ep_shaping_return,
            "ep_final_bonus":    ep_bonus,
        })

        # Benchmark & Save (nur Learner-Policy)
        eval_seconds = plot_seconds = save_seconds = 0.0
        if BINT > 0 and (ep % BINT == 0):
            evs = time.perf_counter()
            per_opponent = run_benchmark(
                game=game, agent=learner, opponents_dict=STRATS,
                opponent_names=CONFIG["BENCH_OPPONENTS"], episodes=BEPS,
                feat_cfg=feat_cfg, num_actions=A
            )
            plotter.log_bench_summary(ep, per_opponent)
            eval_seconds = time.perf_counter() - evs

            ps = time.perf_counter()
            plotter.add_benchmark(ep, per_opponent)
            plotter.plot_benchmark_rewards()
            plotter.plot_places_latest()

            # Einheitliche Titel für alle "lernkurve"-Plots
            title_multi = f"Lernkurve - {family.upper()} vs feste Heuristiken"
            plotter.plot_benchmark(
                filename_prefix="lernkurve",
                with_macro=True,
                family_title=family.upper(),   # Einzelplots: „Lernkurve - KxAy vs <gegner>“
                multi_title=title_multi,       # Multi & Macro: „Lernkurve - KxAy vs feste Heuristiken“
            )

            plotter.plot_train(filename_prefix="training_metrics", separate=True)
            plot_seconds = time.perf_counter() - ps

            ss = time.perf_counter()
            tag = f"{family}_model_{version}_agent_p{LEARN}_ep{ep:07d}"
            learner.save(os.path.join(paths["weights_dir"], tag))
            save_seconds = time.perf_counter() - ss

            cum = time.perf_counter() - t0
            plotter.log_timing(
                ep,
                ep_seconds=(time.perf_counter() - ep_start),
                train_seconds=0.0,
                eval_seconds=eval_seconds,
                plot_seconds=plot_seconds,
                save_seconds=save_seconds,
                cum_seconds=cum,
            )


        # Timing CSV
        ep_seconds = time.perf_counter() - ep_start
        timer.maybe_log(ep, {
            "steps": ep_len,
            "ep_seconds": ep_seconds,
            "train_seconds": 0.0,
            "eval_seconds": eval_seconds,
            "plot_seconds": plot_seconds,
            "save_seconds": save_seconds,
        })

    total_seconds = time.perf_counter() - t0
    plotter.log("")
    plotter.log(f"Gesamtzeit: {total_seconds/3600:0.2f}h (~ {CONFIG['EPISODES']/max(total_seconds,1e-9):0.2f} eps/s)")  
    plotter.log(f"{family}, Snapshot Selfplay (1 Learner, 3 Sparring). Training abgeschlossen.")
    plotter.log(f"Path: {paths['run_dir']}")

if __name__ == "__main__":
    main()
