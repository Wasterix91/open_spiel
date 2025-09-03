# -*- coding: utf-8 -*-
# k3a2.py — DQN Snapshot-Selfplay (1 Learner + 3 Sparring)
#
# Ausrichtung wie k1a2:
# - augment_observation ohne Seat-1hot; optionaler Seat-1hot wird extern angehängt
# - state_size = expected_feature_len(feat_cfg) + seat_id_dim
# - Beobachtungen basieren auf env._state.observation_tensor(p)
# - RewardShaper & Benchmark wie k1a2

import os, datetime, time, copy, numpy as np, torch
import pyspiel
from open_spiel.python import rl_environment

from agents.dqn_agent import DQNAgent, DQNConfig
from utils.strategies import STRATS
from utils.fit_tensor import FeatureConfig, augment_observation, expected_feature_len
from utils.plotter import MetricsPlotter
from utils.reward_shaper import RewardShaper
from utils.load_save_common import find_next_version, prepare_run_dirs, save_config_csv, save_run_meta
from utils.benchmark import run_benchmark
from utils.deck import ranks_for_deck

CONFIG = {
    "EPISODES":         500_000,
    "BENCH_INTERVAL":   10_000,
    "BENCH_EPISODES":   2000,
    "DECK_SIZE":        "64",  # "12" | "16" | "20" | "24" | "32" | "52" | "64"
    "SEED":             42,


    # DQN
    "DQN": {
        "learning_rate":     3e-4,
        "batch_size":        128,
        "gamma":             0.995,
        "epsilon_start":     1.0,
        "epsilon_end":       0.05,
        "epsilon_decay":     0.9997,
        "buffer_size":       200_000,
        "target_update_freq": 5000,
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

    # Features (wie k1a2)
    "FEATURES": {
        "USE_HISTORY": True,
        "SEAT_ONEHOT": False,
        "PLOT_METRICS": False,
        "SAVE_METRICS_TO_CSV": False,
    },

    "SNAPSHOT": { "LEARNER_SEAT": 0, "MIX_CURRENT": 0.8, "SNAPSHOT_INTERVAL": 200, "POOL_CAP": 20 },

    "BENCH_OPPONENTS": ["single_only", "max_combo", "random2"],
}

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

def _with_seat_onehot(vec: np.ndarray, p: int, num_players: int, enabled: bool) -> np.ndarray:
    if not enabled:
        return vec
    seat_oh = np.zeros(num_players, dtype=np.float32); seat_oh[p] = 1.0
    return np.concatenate([vec, seat_oh], axis=0)

class SnapshotDQNA:
    def __init__(self, state_size, num_actions, dqn_cfg, q_state_dict):
        # ---- FIX: dqn_cfg ist bereits ein DQNConfig (namedtuple). Direkt durchreichen.
        # Falls mal ein dict reinkommt, in DQNConfig umwandeln.
        cfg = DQNConfig(**dqn_cfg) if isinstance(dqn_cfg, dict) else dqn_cfg
        self.agent = DQNAgent(state_size, num_actions, cfg, device="cpu")
        self.agent.target_network.eval()
        self.agent.q_network.load_state_dict(copy.deepcopy(q_state_dict))
        self.agent.target_network.load_state_dict(self.agent.q_network.state_dict())
        self.agent.target_network.eval()

    @torch.no_grad()
    def act(self, obs_vec: np.ndarray, legal_actions):
        return greedy_action(self.agent.q_network, obs_vec, legal_actions)

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
    deck_int  = int(CONFIG["DECK_SIZE"])
    num_ranks = ranks_for_deck(deck_int)
    feat_cfg = FeatureConfig(
        num_players=num_players,
        num_ranks=num_ranks,
        add_seat_onehot=False,
        include_history=CONFIG["FEATURES"]["USE_HISTORY"],
    )
    seat_id_dim = (num_players if CONFIG["FEATURES"]["SEAT_ONEHOT"] else 0)
    state_size = expected_feature_len(feat_cfg) + seat_id_dim

    # ---- Agent & Shaper ----
    dqn_cfg = DQNConfig(**CONFIG["DQN"])
    learner = DQNAgent(state_size=state_size, num_actions=A, config=dqn_cfg, device="cpu")
    shaper  = RewardShaper(CONFIG["REWARD"])

    # ---- Config & Meta ----
    save_config_csv({
        "script": family, "version": version,
        "deck_size": CONFIG["DECK_SIZE"], "num_ranks": num_ranks,
        "use_history": CONFIG["FEATURES"]["USE_HISTORY"],
        "agent_type": "DQN_snapshot", "num_episodes": CONFIG["EPISODES"],
        "bench_interval": CONFIG["BENCH_INTERVAL"], "bench_episodes": CONFIG["BENCH_EPISODES"],
        "observation_dim": state_size, "num_actions": A,
        "seat_onehot": CONFIG["FEATURES"]["SEAT_ONEHOT"],
        "models_dir": paths["weights_dir"], "plots_dir": paths["plots_dir"],
        "mix_current": CONFIG["SNAPSHOT"]["MIX_CURRENT"],
        "snapshot_interval": CONFIG["SNAPSHOT"]["SNAPSHOT_INTERVAL"],
        "pool_cap": CONFIG["SNAPSHOT"]["POOL_CAP"],
        "learner_seat": CONFIG["SNAPSHOT"]["LEARNER_SEAT"],
        "step_mode": shaper.step_mode, "delta_weight": shaper.dw,
        "hand_penalty_coeff": shaper.hp, "final_mode": shaper.final_mode,
        "bonus_win": CONFIG["REWARD"]["BONUS_WIN"], "bonus_2nd": CONFIG["REWARD"]["BONUS_2ND"],
        "bonus_3rd": CONFIG["REWARD"]["BONUS_3RD"], "bonus_last": CONFIG["REWARD"]["BONUS_LAST"],
        "benchmark_opponents": ",".join(CONFIG["BENCH_OPPONENTS"]),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }, paths["config_csv"])
    save_run_meta({"family": family, "version": version, "algo": "dqn_snapshot", "deck": CONFIG["DECK_SIZE"]},
                  paths["run_meta_json"])

    # ---- Snapshot-Pool ----
    pool: list[dict] = []
    MIX       = float(CONFIG["SNAPSHOT"]["MIX_CURRENT"])
    SNAPINT   = int(CONFIG["SNAPSHOT"]["SNAPSHOT_INTERVAL"])
    POOL_CAP  = int(CONFIG["SNAPSHOT"]["POOL_CAP"])
    LEARN     = int(CONFIG["SNAPSHOT"]["LEARNER_SEAT"])

    t0 = time.perf_counter()
    BINT, BEPS = CONFIG["BENCH_INTERVAL"], CONFIG["BENCH_EPISODES"]

    # Optional: Sammeln wie k1a2
    collect_metrics = (
        CONFIG["FEATURES"].get("SAVE_METRICS_TO_CSV", False)
        or CONFIG["FEATURES"].get("PLOT_METRICS", False)
    )

    for ep in range(1, CONFIG["EPISODES"] + 1):
        ep_start = time.perf_counter()
        ep_len = 0
        ep_shaping_return = 0.0

        ts = env.reset()
        last_idx_learner = None
        pending = None  # {"s":..., "a":..., "r":...}

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

            # Learner wieder dran? offene Transition schließen (decision-to-decision)
            if (p == LEARN) and (pending is not None):
                base_now = np.asarray(env._state.observation_tensor(LEARN), dtype=np.float32)
                s_now = augment_observation(base_now, player_id=LEARN, cfg=feat_cfg)
                s_now = _with_seat_onehot(s_now, LEARN, num_players, CONFIG["FEATURES"]["SEAT_ONEHOT"])
                learner.buffer.add(pending["s"], pending["a"], pending["r"], s_now, False,
                                   next_legal_actions=legal)
                last_idx_learner = len(learner.buffer.buffer) - 1
                learner.train_step()
                pending = None

            # aktuelle Beobachtung
            base_obs = np.asarray(env._state.observation_tensor(p), dtype=np.float32)
            obs = augment_observation(base_obs, player_id=p, cfg=feat_cfg)
            obs = _with_seat_onehot(obs, p, num_players, CONFIG["FEATURES"]["SEAT_ONEHOT"])

            # Step-Shaping Vorbereitung
            if shaper.step_active():
                hand_before = shaper.hand_size(ts, p, deck_int)

            # Aktion wählen
            if p == LEARN:
                a = int(learner.select_action(obs, legal))
            else:
                a = int(greedy_action(learner.q_network, obs, legal)) if seat_actor[p] == "current" \
                    else int(seat_actor[p].act(obs, legal))

            # Schritt in Env
            ts_next = env.step([a]); ep_len += 1

            # Step-Shaping nur für den Learner
            if p == LEARN:
                r = 0.0
                if shaper.step_active():
                    hand_after = shaper.hand_size(ts_next, p, deck_int)
                    r = float(shaper.step_reward(hand_before=hand_before, hand_after=hand_after))
                    ep_shaping_return += r

                assert pending is None
                pending = {"s": obs, "a": a, "r": r}

            ts = ts_next

        # Episodenende: offene Learner-Transition finalisieren
        if pending is not None:
            s_next = pending["s"]
            learner.buffer.add(pending["s"], pending["a"], pending["r"], s_next, True,
                               next_legal_actions=list(range(A)))
            last_idx_learner = len(learner.buffer.buffer) - 1
            learner.train_step()
            pending = None

        # Finale Rewards / Bonus
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

        # Metriken
        metrics = {
            "ep_length":         ep_len,
            "ep_env_return":     ep_env,
            "ep_shaping_return": ep_shaping_return,
            "ep_final_bonus":    ep_bonus,
        }
        plotter.add_train(ep, metrics)

        # Benchmark & Save
        if CONFIG["BENCH_INTERVAL"] > 0 and (ep % BINT == 0):
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
            title_multi = f"Lernkurve - {family.upper()} vs feste Heuristiken"
            plotter.plot_benchmark("lernkurve", with_macro=True,
                                   family_title=family.upper(), multi_title=title_multi)
            if CONFIG["FEATURES"].get("PLOT_METRICS", False):
                plotter.plot_train(filename_prefix="training_metrics", separate=True)
            plot_seconds = time.perf_counter() - ps

            ss = time.perf_counter()
            tag = f"{family}_model_{version}_agent_p{LEARN}_ep{ep:07d}"
            learner.save(os.path.join(paths["weights_dir"], tag))
            save_seconds = time.perf_counter() - ss

            plotter.log_timing(
                ep,
                ep_seconds=(time.perf_counter() - ep_start),
                train_seconds=0.0,
                eval_seconds=eval_seconds,
                plot_seconds=plot_seconds,
                save_seconds=save_seconds,
                cum_seconds=(time.perf_counter() - t0),
            )

    total_seconds = time.perf_counter() - t0
    plotter.log("")
    plotter.log(f"Gesamtzeit: {total_seconds/3600:0.2f}h (~ {CONFIG['EPISODES']/max(total_seconds,1e-9):0.2f} eps/s)")
    plotter.log(f"{family}, Snapshot Selfplay (1 Learner, 3 Sparring). Training abgeschlossen.")
    plotter.log(f"Path: {paths['run_dir']}")

if __name__ == "__main__":
    main()
