# President/training/train_k2_multi_simul.py
import os
import re
import argparse
import datetime
import numpy as np
import pandas as pd
import torch

import pyspiel
from open_spiel.python import rl_environment

from agents import ppo_agent as ppo
from agents import dqn_agent as dqn


class RewardShaper:
    def __init__(self, step_mode="delta_hand", final_mode="none", env_flag=True, delta_weight=1.0, hand_coeff=0.0):
        self.step_mode = step_mode; self.final_mode = final_mode; self.env_flag = bool(env_flag)
        self.delta_weight = float(delta_weight); self.hand_coeff = float(hand_coeff)

    @staticmethod
    def _num_ranks_for_deck(deck_size):
        if deck_size in (32, 64): return 8
        if deck_size == 52: return 13
        raise NotImplementedError

    def hand_size(self, ts, pid, deck_size: int) -> int:
        nr = self._num_ranks_for_deck(deck_size); hand = ts.observations["info_state"][pid]
        return int(sum(hand[:nr]))

    def step_reward(self, *, hand_before=None, hand_after=None, time_step=None, player_id=None, deck_size=None):
        if self.step_mode == "none": return 0.0
        if self.step_mode == "delta_hand":
            diff = max(0.0, float(hand_before - hand_after))
            return self.delta_weight * diff
        if self.step_mode == "hand_penalty":
            size = self.hand_size(time_step, player_id, deck_size)
            return -self.hand_coeff * float(size)
        raise ValueError

    def include_env_reward(self): return self.env_flag


def find_next_version(base_dir, prefix):
    pat = re.compile(fr"{re.escape(prefix)}_(\d{{2}})$")
    os.makedirs(base_dir, exist_ok=True)
    existing = [int(m.group(1)) for m in (pat.match(n) for n in os.listdir(base_dir)) if m]
    return f"{max(existing) + 1:02d}" if existing else "01"


def main():
    ap = argparse.ArgumentParser("K2: Multi simultan (4 Lernende)")
    ap.add_argument("--algo", choices=["ppo","dqn"], default="ppo")
    ap.add_argument("--episodes", type=int, default=1_000_000)
    ap.add_argument("--eval_interval", type=int, default=10_000)
    ap.add_argument("--deck_size", choices=["32","52","64"], default="64")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--models_root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)

    game = pyspiel.load_game("president", {
        "num_players": 4, "deck_size": args.deck_size, "shuffle_cards": True, "single_card_mode": False
    })
    env = rl_environment.Environment(game)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]
    deck_size_int = int(args.deck_size)

    if args.algo == "ppo":
        version = find_next_version(os.path.join(args.models_root, "ppo_model"), "ppo_model")
        model_dir = os.path.join(args.models_root, f"ppo_model_{version}", "train")
        agents = [ppo.PPOAgent(info_state_size, num_actions) for _ in range(4)]
    else:
        version = find_next_version(os.path.join(args.models_root, "dqn_model"), "dqn_model")
        model_dir = os.path.join(args.models_root, f"dqn_model_{version}", "train")
        agents = [dqn.DQNAgent(info_state_size, num_actions) for _ in range(4)]
    os.makedirs(model_dir, exist_ok=True)

    shaper = RewardShaper(step_mode="delta_hand", final_mode="none", env_flag=True, delta_weight=1.0, hand_coeff=0.0)

    # Log
    runs_csv = os.path.join(os.path.dirname(model_dir), "training_runs.csv")
    pd.DataFrame([{
        "version": version, "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent_type": args.algo.upper(), "num_episodes": args.episodes, "eval_interval": args.eval_interval,
        "num_players": 4, "deck_size": args.deck_size, "observation_dim": info_state_size,
        "num_actions": num_actions, "model_version_dir": model_dir, "step_reward": "delta_hand", "final_reward": "none",
    }]).to_csv(runs_csv, index=False)
    print(f"📄 Konfiguration gespeichert unter: {runs_csv}")

    EVAL_INTERVAL = args.eval_interval

    for ep in range(1, args.episodes + 1):
        ts = env.reset()
        while not ts.last():
            pid = ts.observations["current_player"]
            legal = ts.observations["legal_actions"][pid]
            hand_before = shaper.hand_size(ts, pid, deck_size_int)

            if args.algo == "ppo":
                action = agents[pid].step(ts.observations["info_state"][pid], legal)
            else:
                s = np.array(ts.observations["info_state"][pid], dtype=np.float32)
                action = agents[pid].select_action(s, legal)

            ts_next = env.step([int(action)])

            # reward shaping pro-Seat
            hand_after = shaper.hand_size(ts_next, pid, deck_size_int)
            r = shaper.step_reward(hand_before=hand_before, hand_after=hand_after,
                                   time_step=ts_next, player_id=pid, deck_size=deck_size_int)

            if args.algo == "ppo":
                agents[pid].post_step(r, done=ts_next.last())
            else:
                s = np.array(ts.observations["info_state"][pid], dtype=np.float32)
                ns = np.array(ts_next.observations["info_state"][pid], dtype=np.float32) if not ts_next.last() else s
                agents[pid].store_transition(s, int(action), float(r), ns, bool(ts_next.last()),
                                             next_legal_actions=ts_next.observations["legal_actions"][pid] if not ts_next.last() else [])
                agents[pid].train_step()

            ts = ts_next

        # terminal rewards & train
        if args.algo == "ppo":
            for i in range(4):
                if shaper.include_env_reward():
                    agents[i]._buffer.finalize_last_reward(env._state.returns()[i])
                agents[i].train()

        # Save
        if ep % EVAL_INTERVAL == 0:
            if args.algo == "ppo":
                for i in range(4):
                    base = os.path.join(model_dir, f"ppo_model_{version}_agent_p{i}_ep{ep:07d}")
                    agents[i].save(base)
            else:
                for i in range(4):
                    base = os.path.join(model_dir, f"dqn_model_{version}_agent_p{i}_ep{ep:07d}")
                    agents[i].save(base)
            print(f"💾 Modelle gespeichert nach Episode {ep}")

    print("✅ K2 Training fertig.")


if __name__ == "__main__":
    main()
