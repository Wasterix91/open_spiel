# -*- coding: utf-8 -*-
# President/training/k3a1_snapshot.py — PPO (K3-Style) mit Snapshot-Selfplay-Pool
# - Seat 0 lernt (Parameter-Sharing optional)
# - Seats 1–3: mix aus aktueller Policy und eingefrorenen Snapshots
# - Eval vs Heuristiken (single_only, max_combo, random2) + Macro über EvalPlotter

import os, re, copy, datetime
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import pyspiel
from open_spiel.python import rl_environment

from agents import ppo_agent as ppo
from utils.strategies import STRATS
from utils.fit_tensor import FeatureConfig, augment_observation
from utils.training_eval_plots import EvalPlotter

# ===================== CONFIG ===================== #
CONFIG = {
    "EPISODES":        200_000,
    "EVAL_INTERVAL":   10_000,
    "EVAL_EPISODES":   2_000,
    "DECK_SIZE":       "64",
    "SEED":            123,

    # PPO-Hyperparameter
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

    # Reward-Shaping
    "REWARD": {
        "STEP": "delta_hand",     # "none" | "delta_hand" | "hand_penalty"
        "DELTA_WEIGHT": 1.0,
        "HAND_PENALTY_COEFF": 0.0,
        "FINAL": "none",          # "none" | "placement_bonus"
        "BONUS_WIN": 0.0, "BONUS_2ND": 0.0, "BONUS_3RD": 0.0, "BONUS_LAST": 0.0,
        "ENV_REWARD": True,
    },

    # Feature-Flags (wie k3a1)
    "FEATURES": {
        "NORMALIZE": False,
        "SEAT_ONEHOT": True,    # One-Hot wird beim Agent-Input (_make_input) angehängt
    },

    # Snapshot-Selfplay
    "SNAPSHOT": {
        "PARAMETER_SHARING": True,     # ein gemeinsames Netz; Seat 0 lernt
        "LEARNER_SEAT": 0,
        "MIX_CURRENT": 0.8,            # Anteil 'current' vs Snapshot bei Seats 1–3
        "SNAPSHOT_INTERVAL": 10_000,   # wie oft aktuelle Policy in den Pool
        "POOL_CAP": 20,                # FIFO-Größe
    },

    # Heuristik-Evalkurven (für Plotter)
    "EVAL_CURVES": ["single_only", "max_combo", "random2"],
}

# ===================== Helpers ===================== #
def find_next_version(models_root, prefix):
    pat = re.compile(rf"^{re.escape(prefix)}_(\d{{2}})$")
    os.makedirs(models_root, exist_ok=True)
    existing = [int(m.group(1)) for m in (pat.match(n) for n in os.listdir(models_root)) if m]
    return f"{max(existing)+1:02d}" if existing else "01"

class RewardShaper:
    def __init__(self, cfg):
        final = "placement_bonus" if cfg["FINAL"] in ("final_reward","placement_bonus") else cfg["FINAL"]
        self.step, self.final, self.env = cfg["STEP"], final, bool(cfg["ENV_REWARD"])
        self.dw, self.hp = float(cfg["DELTA_WEIGHT"]), float(cfg["HAND_PENALTY_COEFF"])
        self.b = (float(cfg.get("BONUS_WIN",0.0)), float(cfg.get("BONUS_2ND",0.0)),
                  float(cfg.get("BONUS_3RD",0.0)), float(cfg.get("BONUS_LAST",0.0)))
    @staticmethod
    def _ranks(deck): return 8 if deck in (32,64) else 13 if deck==52 else (_ for _ in ()).throw(ValueError("deck"))
    def hand_size(self, ts, pid, deck): return int(sum(ts.observations["info_state"][pid][:self._ranks(deck)]))
    def step_reward(self, **kw):
        if self.step=="none": return 0.0
        if self.step=="delta_hand": return self.dw*max(0.0, float(kw["hand_before"]-kw["hand_after"]))
        if self.step=="hand_penalty": return -self.hp*float(self.hand_size(kw["time_step"], kw["player_id"], kw["deck_size"]))
        raise ValueError(self.step)
    def final_bonus(self, finals, pid):
        if self.final=="none": return 0.0
        order = sorted(range(len(finals)), key=lambda p: finals[p], reverse=True)
        place = order.index(pid)+1
        return (self.b[0],self.b[1],self.b[2],self.b[3])[place-1]
    def include_env_reward(self): return self.env

def make_snapshot_state_dict(policy_net: torch.nn.Module):
    return copy.deepcopy(policy_net.state_dict())

class SnapshotPolicy:
    """Eingefrorene Policy nur für Inferenz. Erwartet identischen Input wie die aktuelle Policy."""
    def __init__(self, input_dim: int, num_actions: int, state_dict: dict):
        self.net = ppo.PolicyNetwork(input_dim, num_actions)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self.num_actions = num_actions

    @torch.no_grad()
    def act(self, obs_vec: np.ndarray, legal_actions, seat_one_hot=None):
        # obs_vec: bereits augment_observation(...) (ohne seat-one-hot)
        x = np.array(obs_vec, dtype=np.float32)
        if seat_one_hot is not None:
            x = np.concatenate([x, np.asarray(seat_one_hot, dtype=np.float32)], axis=0)
        device = next(self.net.parameters()).device
        x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        logits = self.net(x_t).squeeze(0)

        legal_mask = torch.zeros(self.num_actions, dtype=torch.float32, device=device)
        legal_mask[legal_actions] = 1.0
        probs = ppo.masked_softmax(logits, legal_mask)
        return int(torch.distributions.Categorical(probs=probs).sample().item())

# ===================== Training ===================== #
def main():
    np.random.seed(CONFIG["SEED"]); torch.manual_seed(CONFIG["SEED"])
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS = os.path.join(ROOT, "models")

    # ---- Env ----
    game = pyspiel.load_game("president", {
        "num_players": 4,
        "deck_size": CONFIG["DECK_SIZE"],
        "shuffle_cards": True,
        "single_card_mode": False,
    })
    env = rl_environment.Environment(game)
    info_dim = env.observation_spec()["info_state"][0]
    A = env.action_spec()["num_actions"]
    num_players = game.num_players()
    deck_int = int(CONFIG["DECK_SIZE"])
    num_ranks = 8 if deck_int in (32, 64) else 13 if deck_int==52 else (_ for _ in ()).throw(ValueError("deck"))

    # ---- Features & Agent ----
    feat_cfg = FeatureConfig(
        num_players=num_players,
        num_ranks=num_ranks,
        add_seat_onehot=False,                            # Seat-One-Hot kommt über Agent._make_input
        normalize=bool(CONFIG["FEATURES"]["NORMALIZE"]),
    )
    seat_id_dim = num_players if CONFIG["FEATURES"]["SEAT_ONEHOT"] else 0

    version = find_next_version(MODELS, "ppo_snapshot")
    model_dir = os.path.join(MODELS, f"ppo_snapshot_{version}", "train"); os.makedirs(model_dir, exist_ok=True)

    ppo_cfg = ppo.PPOConfig(**CONFIG["PPO"])
    learner = ppo.PPOAgent(info_state_size=info_dim, num_actions=A,
                           seat_id_dim=seat_id_dim, config=ppo_cfg)

    # ---- EvalPlotter (Heuristiken) ----
    plotter = EvalPlotter(
        opponent_names=list(CONFIG["EVAL_CURVES"]),
        out_dir=model_dir,
        filename_prefix="lernkurve",
        csv_filename="eval_curves.csv",
        save_csv=True,
    )

    # ---- Log Config ----
    pd.DataFrame([{
        "version": version,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent_type": "PPO-K3-SNAPSHOT",
        "num_episodes": CONFIG["EPISODES"],
        "eval_interval": CONFIG["EVAL_INTERVAL"],
        "eval_episodes": CONFIG["EVAL_EPISODES"],
        "deck_size": CONFIG["DECK_SIZE"],
        "observation_dim": info_dim + seat_id_dim,
        "num_actions": A,
        "model_version_dir": model_dir,
        "normalize": CONFIG["FEATURES"]["NORMALIZE"],
        "seat_onehot": CONFIG["FEATURES"]["SEAT_ONEHOT"],
        "param_sharing": CONFIG["SNAPSHOT"]["PARAMETER_SHARING"],
        "mix_current": CONFIG["SNAPSHOT"]["MIX_CURRENT"],
        "snapshot_interval": CONFIG["SNAPSHOT"]["SNAPSHOT_INTERVAL"],
        "pool_cap": CONFIG["SNAPSHOT"]["POOL_CAP"],
    }]).to_csv(os.path.join(os.path.dirname(model_dir), "training_runs.csv"), index=False)

    # ---- Snapshot-Pool ----
    pool: list[dict] = []  # list of state_dicts der Policy

    shaper = RewardShaper(CONFIG["REWARD"])

    EINT, EEPS = CONFIG["EVAL_INTERVAL"], CONFIG["EVAL_EPISODES"]
    SNAPINT = CONFIG["SNAPSHOT"]["SNAPSHOT_INTERVAL"]
    MIX = float(CONFIG["SNAPSHOT"]["MIX_CURRENT"])
    POOL_CAP = int(CONFIG["SNAPSHOT"]["POOL_CAP"])
    LEARNER_SEAT = int(CONFIG["SNAPSHOT"]["LEARNER_SEAT"])
    assert LEARNER_SEAT == 0, "Dieses Skript nimmt Seat 0 als Lerner an."

    for ep in range(1, CONFIG["EPISODES"] + 1):
        ts = env.reset()

        # Seats 1..3: entscheiden, ob aktuelle Policy oder Snapshot genutzt wird
        use_current = {}
        snap_actor = {}
        for seat in [1, 2, 3]:
            if len(pool) == 0:
                use_current[seat] = True
            else:
                use_current[seat] = (np.random.rand() < MIX)
                if not use_current[seat]:
                    idx = np.random.randint(len(pool))
                    snap_actor[seat] = SnapshotPolicy(
                        input_dim=info_dim + seat_id_dim,
                        num_actions=A,
                        state_dict=pool[idx],
                    )

        # ---- Episode ----
        while not ts.last():
            p = ts.observations["current_player"]
            legal = ts.observations["legal_actions"][p]

            base_obs = ts.observations["info_state"][p]
            obs = augment_observation(base_obs, player_id=p, cfg=feat_cfg)

            seat_oh = None
            if CONFIG["FEATURES"]["SEAT_ONEHOT"]:
                seat_oh = np.zeros((num_players,), dtype=np.float32); seat_oh[p] = 1.0

            hand_before = shaper.hand_size(ts, p, deck_int)

            if p == LEARNER_SEAT:
                # Lernender sammelt Daten (Buffer) und wählt Aktion
                a = int(learner.step(obs, legal, seat_one_hot=seat_oh))
            else:
                # Gegner: aktuelle Policy (vorwärts) oder Snapshot
                if use_current[p]:
                    x = learner._make_input(obs, seat_one_hot=seat_oh)
                    with torch.no_grad():
                        logits = learner._policy(x)
                        mask = torch.zeros(A, device=logits.device); mask[legal] = 1.0
                        probs = ppo.masked_softmax(logits, mask)
                    a = int(torch.distributions.Categorical(probs=probs).sample().item())
                else:
                    a = int(snap_actor[p].act(obs, legal, seat_one_hot=seat_oh))

            ts_next = env.step([a])

            # Reward shaping nur für Lern-Sitz
            if p == LEARNER_SEAT:
                hand_after = shaper.hand_size(ts_next, p, deck_int)
                r = shaper.step_reward(hand_before=hand_before, hand_after=hand_after,
                                       time_step=ts_next, player_id=p, deck_size=deck_int)
                learner.post_step(r, done=ts_next.last())

            ts = ts_next

        # ---- Episodenende: terminale Rewards & Training ----
        if shaper.include_env_reward():
            learner._buffer.finalize_last_reward(env._state.returns()[LEARNER_SEAT])
        learner._buffer.finalize_last_reward(
            shaper.final_bonus(env._state.returns(), LEARNER_SEAT)
        )
        learner.train()

        # ---- Snapshot in Pool legen ----
        if ep % SNAPINT == 0:
            pool.append(make_snapshot_state_dict(learner._policy))
            if len(pool) > POOL_CAP:
                pool.pop(0)

        # ---- Evaluation & Speichern ----
        if ep % EINT == 0:
            per_opponent = {}
            for opp_name in CONFIG["EVAL_CURVES"]:
                opp_fn = STRATS[opp_name]
                wins = 0
                for _ in range(EEPS):
                    st = game.new_initial_state()
                    while not st.is_terminal():
                        pid = st.current_player(); legal = st.legal_actions(pid)
                        if pid == 0:
                            ob = st.information_state_tensor(pid)
                            ob = augment_observation(ob, player_id=pid, cfg=feat_cfg)
                            seat_oh = None
                            if CONFIG["FEATURES"]["SEAT_ONEHOT"]:
                                seat_oh = np.zeros((num_players,), dtype=np.float32); seat_oh[pid] = 1.0
                            x = learner._make_input(ob, seat_one_hot=seat_oh)
                            with torch.no_grad():
                                logits = learner._policy(x)
                                mask = torch.zeros(A, device=logits.device); mask[legal] = 1.0
                                probs = ppo.masked_softmax(logits, mask)
                            a = int(torch.distributions.Categorical(probs=probs).sample().item())
                        else:
                            a = int(opp_fn(st))
                        st.apply_action(a)
                    if st.returns()[0] == max(st.returns()):
                        wins += 1
                per_opponent[opp_name] = 100.0 * wins / EEPS
                print(f"✅ Eval nach {ep} – Winrate vs {opp_name}: {per_opponent[opp_name]:.1f}%")

            macro = float(np.mean(list(per_opponent.values())))
            print(f"📊 Macro Average: {macro:.2f}%")
            plotter.add(ep, per_opponent)
            plotter.plot_all()

            base = os.path.join(model_dir, f"ppo_snapshot_{version}_agent_p0_ep{ep:07d}")
            learner.save(base)
            print(f"💾 Modell gespeichert: {base}_*.pt")

    print("✅ K3-Snapshot Training abgeschlossen.")

if __name__ == "__main__":
    main()
