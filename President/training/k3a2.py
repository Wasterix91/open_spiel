# -*- coding: utf-8 -*-
# k3a2.py — DQN Snapshot-Selfplay (1 Learner + 3 Sparring)
#
# Plot-Parität zu k1a1/k1a2:
# - smooth_window via FEATURES.RET_SMOOTH_WINDOW
# - SAVE_METRICS_TO_CSV steuert CSV-Schreiben (Train); Benchmark-CSV immer
# - ep_return_training = shaping(+optional) + env(+optional) + final_bonus
# - Komponenten-Gating in add_ep_returns
# - plot_train(include_keys=FEATURES.PLOT_KEYS, separate=True)

import os, datetime, time, copy, numpy as np, torch
import pyspiel
from open_spiel.python import rl_environment

from agents.dqn_agent import DQNAgent, DQNConfig
from utils.strategies import STRATS
from agents import v_table_agent
from utils.fit_tensor import FeatureConfig, augment_observation
from utils.plotter import MetricsPlotter
from utils.reward_shaper import RewardShaper
from utils.load_save_common import find_next_version, prepare_run_dirs, save_config_csv, save_run_meta
from utils.benchmark import run_benchmark
from utils.deck import ranks_for_deck

CONFIG = {
    "EPISODES":         1_000_000,
    "BENCH_INTERVAL":   10_000,
    "BENCH_EPISODES":   2_000,
    "DECK_SIZE":        "64",  # "12" | "16" | "20" | "24" | "32" | "52" | "64"
    "SEED":             42,

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

    # ======= Rewards (NEUES System) =======
    # STEP_MODE : "none" | "delta_weight_only" | "hand_penalty_coeff_only" | "combined"
    # FINAL_MODE: "none" | "env_only" | "rank_only" | "both"
    "REWARD": {
        "STEP_MODE": "none",
        "DELTA_WEIGHT": 0.0,
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

    "SNAPSHOT": { "LEARNER_SEAT": 0, "MIX_CURRENT": 0.8, "SNAPSHOT_INTERVAL": 200, "POOL_CAP": 20 },

    "V_TABLE_PATH": "agents/tables/v_table_4_4_4",  # Pfadpräfix ohne *_params.json etc.

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

class SnapshotDQNA:
    def __init__(self, state_size, num_actions, dqn_cfg, q_state_dict):
        cfg = DQNConfig(**dqn_cfg) if isinstance(dqn_cfg, dict) else dqn_cfg
        self.agent = DQNAgent(state_size, num_actions, cfg, device="cpu")
        self.agent.target_network.eval()
        self.agent.q_network.load_state_dict(copy.deepcopy(q_state_dict))
        self.agent.target_network.load_state_dict(self.agent.q_network.state_dict())
        self.agent.target_network.eval()

    @torch.no_grad()
    def act(self, obs_vec: np.ndarray, legal_actions):
        return greedy_action(self.agent.q_network, obs_vec, legal_actions)

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
                        state_size = feat_cfg.input_dim()  # KEIN extra seat_id_dim mehr


                        # Beobachtung genau wie im DQN-Training
                        ts = env.reset()
                        p = 0  # wir inspizieren P0
                        base_obs = np.asarray(env._state.observation_tensor(p), dtype=np.float32)

                        obs = augment_observation(base_obs, player_id=p, cfg=feat_cfg)


                        print("Deck Size:  ", CONFIG["DECK_SIZE"])
                        print("Use History:", CONFIG["FEATURES"]["USE_HISTORY"])
                        print("Seat 1-Hot: ", CONFIG["FEATURES"]["SEAT_ONEHOT"])
                        print("Normalize:  ", CONFIG["FEATURES"]["NORMALIZE"])
                        print(f"Tensor length={len(obs)}  (model input_dim={feat_cfg.input_dim()})")
                        print(f"Tensor: {np.round(obs, 3)}")
                        print("-" * 60)


        return  # Debug beendet, nicht trainieren
    # ================== DEBUG Ende ==================

        # --- Frühzeitige Validierung ---
    if "v_table" in CONFIG["BENCH_OPPONENTS"]:
        path = CONFIG.get("V_TABLE_PATH")
        if not isinstance(path, str) or not path:
            raise ValueError(
                "BENCH_OPPONENTS enthält 'v_table', aber V_TABLE_PATH ist nicht gesetzt oder ungültig."
            )

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
        save_csv=CONFIG["FEATURES"]["SAVE_METRICS_TO_CSV"],
        verbosity=1,
        smooth_window=CONFIG["FEATURES"].get("RET_SMOOTH_WINDOW", 150),
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
        add_seat_onehot=CONFIG["FEATURES"]["SEAT_ONEHOT"],
        include_history=CONFIG["FEATURES"]["USE_HISTORY"],
        normalize=CONFIG["FEATURES"]["NORMALIZE"],
        deck_size=deck_int,
    )
    state_size = feat_cfg.input_dim()   # KEIN extra seat_id_dim mehr


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
        "ret_smooth_window": CONFIG["FEATURES"].get("RET_SMOOTH_WINDOW", 150),
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

    # Sammeln nur, wenn wir es brauchen (CSV oder Plot)
    collect_metrics = (
        CONFIG["FEATURES"].get("SAVE_METRICS_TO_CSV", False)
        or CONFIG["FEATURES"].get("PLOT_METRICS", False)
    )

    for ep in range(1, CONFIG["EPISODES"] + 1):
        ep_start = time.perf_counter()
        ep_len = 0
        ep_shaping_return = 0.0
        train_seconds_accum = 0.0

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
            base_obs = np.asarray(env._state.observation_tensor(p), dtype=np.float32)
            obs = augment_observation(base_obs, player_id=p, cfg=feat_cfg)


            # Learner wieder dran? offene Transition schließen (decision-to-decision)
            if (p == LEARN) and (pending is not None):
                base_now = np.asarray(env._state.observation_tensor(LEARN), dtype=np.float32)
                s_now = augment_observation(base_now, player_id=LEARN, cfg=feat_cfg)
                learner.buffer.add(pending["s"], pending["a"], pending["r"], s_now, False,
                                next_legal_actions=legal)
                last_idx_learner = len(learner.buffer.buffer) - 1

                tts = time.perf_counter()
                learner.train_step()
                train_seconds_accum += (time.perf_counter() - tts)

                pending = None



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
                pending = {"s": obs, "a": a, "r": r}

            ts = ts_next

        # Episodenende: offene Learner-Transition finalisieren
        if pending is not None:
            s_next = pending["s"]
            learner.buffer.add(pending["s"], pending["a"], pending["r"], s_next, True,
                               next_legal_actions=list(range(A)))
            last_idx_learner = len(learner.buffer.buffer) - 1
            tts = time.perf_counter()
            learner.train_step()
            train_seconds_accum += (time.perf_counter() - tts)
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

        # --- Trainingsäquivalenter Return (Learner) wie k1a1/k1a2 ---
        env_part     = ep_env if shaper.include_env_reward() else 0.0
        shaping_part = ep_shaping_return if shaper.step_active() else 0.0
        ep_return_training = shaping_part + env_part + ep_bonus

        # Metriken (nur P0/Learner — konsistent zur Linie)
        metrics = {
            "ep_length":         ep_len,
            "train_seconds":     train_seconds_accum,
            "ep_env_return":     ep_env,
            "ep_shaping_return": ep_shaping_return,
            "ep_final_bonus":    ep_bonus,
            "epsilon":           float(getattr(learner, "epsilon", np.nan)),
            "ep_return_training": ep_return_training,
        }

        # Sonderplot: Episoden-Return + Komponenten (mit Gating)
        plotter.add_ep_returns(
            global_episode=ep,
            ep_returns=[ep_return_training],
            components={
                "env_score":   [env_part],      # 0.0, wenn nicht aktiv
                "shaping":     [shaping_part],  # 0.0, wenn nicht aktiv
                "final_bonus": [ep_bonus],
            },
        )

        # Train-Metriken ggf. persistieren oder nur puffern
        if collect_metrics:
            if CONFIG["FEATURES"].get("SAVE_METRICS_TO_CSV", False):
                plotter.add_train(ep, metrics)
            else:
                plotter.train_rows.append({"episode": int(ep), **metrics})
                if plotter.train_keys is None:
                    plotter.train_keys = ["episode"] + list(metrics.keys())

        # Benchmark & Save
        eval_seconds = plot_seconds = save_seconds = 0.0
        if CONFIG["BENCH_INTERVAL"] > 0 and (ep % BINT == 0):
            evs = time.perf_counter()
            # STRATS kopieren und 'v_table' nur dann hinzufügen, wenn er im BENCH_OPPONENTS-Set steht
            bench_map = dict(STRATS)
            for tok in CONFIG["BENCH_OPPONENTS"]:
                if tok == "v_table":
                    bench_map[tok] = resolve_opponent(tok)

            per_opponent = run_benchmark(
                game=game, agent=learner, opponents_dict=bench_map,  # <-- RICHTIG
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
            plotter.plot_benchmark(
                filename_prefix="lernkurve",
                with_macro=True,
                family_title=family.upper(),
                multi_title=title_multi,
            )
            if CONFIG["FEATURES"].get("PLOT_METRICS", False):
                plotter.plot_train(
                    include_keys=CONFIG["FEATURES"].get("PLOT_KEYS"),
                    separate=True,
                )
            plot_seconds = time.perf_counter() - ps

            ss = time.perf_counter()
            tag = f"{family}_model_{version}_agent_p{LEARN}_ep{ep:07d}"
            learner.save(os.path.join(paths["weights_dir"], tag))
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

    total_seconds = time.perf_counter() - t0
    plotter.log("")
    plotter.log(f"Gesamtzeit: {total_seconds/3600:0.2f}h (~ {CONFIG['EPISODES']/max(total_seconds,1e-9):0.2f} eps/s)")
    plotter.log(f"{family}, Snapshot Selfplay (1 Learner, 3 Sparring). Training abgeschlossen.")
    plotter.log(f"Path: {paths['run_dir']}")

if __name__ == "__main__":
    main()
