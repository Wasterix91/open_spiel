# President/training/train_k3_selfplay_league.py
import os
import re
import copy
import argparse
import datetime
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import pyspiel
from open_spiel.python import rl_environment

from agents import ppo_agent as ppo


# ===== Helpers =====
def find_next_version(base_dir, prefix):
    pat = re.compile(fr"{re.escape(prefix)}_(\d{{2}})$")
    os.makedirs(base_dir, exist_ok=True)
    existing = [int(m.group(1)) for m in (pat.match(n) for n in os.listdir(base_dir)) if m]
    return f"{max(existing) + 1:02d}" if existing else "01"


class RewardShaper:
    def __init__(self, step_mode="delta_hand", final_mode="none", env_flag=True, delta_weight=1.0):
        self.step_mode = step_mode; self.final_mode = final_mode; self.env_flag = bool(env_flag); self.delta_weight = float(delta_weight)

    @staticmethod
    def _num_ranks_for_deck(deck_size):
        return 8 if deck_size in (32,64) else 13

    def hand_size(self, ts, pid, deck_size: int) -> int:
        nr = self._num_ranks_for_deck(deck_size); hand = ts.observations["info_state"][pid]
        return int(sum(hand[:nr]))

    def step_reward(self, *, hand_before=None, hand_after=None, **_):
        if self.step_mode == "none": return 0.0
        diff = max(0.0, float(hand_before - hand_after))
        return self.delta_weight * diff

    def include_env_reward(self): return self.env_flag


# ===== Self-Play League (K3) =====
def main():
    ap = argparse.ArgumentParser("K3: Self-Play League (P0 lernt; Gegner = aktuelle Policy oder Snapshots)")
    ap.add_argument("--episodes", type=int, default=20_000)
    ap.add_argument("--eval_interval", type=int, default=2000)
    ap.add_argument("--eval_episodes", type=int, default=2000)
    ap.add_argument("--deck_size", choices=["32","52","64"], default="64")
    ap.add_argument("--param_sharing", action="store_true", default=True)
    ap.add_argument("--add_seat_id", action="store_true", default=True)
    ap.add_argument("--mix_current", type=float, default=0.8)
    ap.add_argument("--snapshot_interval", type=int, default=10_000)
    ap.add_argument("--pool_cap", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--models_root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)

    game = pyspiel.load_game("president", {
        "num_players": 4, "deck_size": args.deck_size, "shuffle_cards": True, "single_card_mode": False
    })
    env = rl_environment.Environment(game)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]
    num_players = 4
    seat_id_dim = num_players if args.add_seat_id else 0
    deck_size_int = int(args.deck_size)

    # Models
    version = find_next_version(os.path.join(args.models_root, "ppo_model"), "ppo_model")
    model_dir = os.path.join(args.models_root, f"ppo_model_{version}", "train")
    os.makedirs(model_dir, exist_ok=True)

    # Agent
    shared = ppo.PPOAgent(info_state_size, num_actions, seat_id_dim=seat_id_dim)
    agents = [shared]  # nur der Lerner (P0); Gegner leiten von shared ab
    shaper = RewardShaper(step_mode="delta_hand", final_mode="none", env_flag=True, delta_weight=1.0)

    # Logging
    runs_csv = os.path.join(os.path.dirname(model_dir), "training_runs.csv")
    pd.DataFrame([{
        "version": version, "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent_type": "PPO", "num_episodes": args.episodes, "eval_interval": args.eval_interval,
        "eval_episodes": args.eval_episodes, "num_players": 4, "deck_size": args.deck_size,
        "observation_dim": info_state_size + seat_id_dim, "num_actions": num_actions,
        "model_version_dir": model_dir, "parameter_sharing": True, "add_seat_id": args.add_seat_id,
    }]).to_csv(runs_csv, index=False)
    print(f"📄 Konfiguration gespeichert unter: {runs_csv}")

    # Opponent Pool
    pool = []

    def make_snapshot(policy_net):
        return copy.deepcopy(policy_net.state_dict())

    class SnapshotPolicy:
        def __init__(self, input_dim, num_actions, state_dict):
            self.net = ppo.PolicyNetwork(input_dim, num_actions)
            self.net.load_state_dict(state_dict)
            self.net.eval()
            self.num_actions = num_actions

        @torch.no_grad()
        def act(self, obs, legal, seat_one_hot=None):
            x = np.array(obs, dtype=np.float32)
            if seat_id_dim > 0:
                assert seat_one_hot is not None and len(seat_one_hot) == seat_id_dim
                x = np.concatenate([x, np.array(seat_one_hot, dtype=np.float32)], axis=0)
            device = next(self.net.parameters()).device
            x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
            logits = self.net(x_t).squeeze(0)
            legal_mask = torch.zeros(self.num_actions, dtype=torch.float32, device=device)
            legal_mask[legal] = 1.0
            probs = ppo.masked_softmax(logits, legal_mask)
            dist = torch.distributions.Categorical(probs=probs)
            return int(dist.sample().item())

    # Train
    for ep in range(1, args.episodes + 1):
        ts = env.reset()

        # Decide opponents for seats 1–3
        use_current = []
        snap_policies = {}
        for seat in [1,2,3]:
            if len(pool) == 0:
                use_current.append(True)
            else:
                use = (np.random.rand() < args.mix_current)
                use_current.append(use)
                if not use:
                    idx = np.random.randint(len(pool))
                    snap_policies[seat] = SnapshotPolicy(info_state_size + seat_id_dim, num_actions, pool[idx])

        while not ts.last():
            pid = ts.observations["current_player"]
            legal = ts.observations["legal_actions"][pid]
            obs = ts.observations["info_state"][pid]

            seat_oh = None
            if args.add_seat_id:
                seat_oh = np.zeros(num_players, dtype=np.float32); seat_oh[pid] = 1.0

            hand_before = shaper.hand_size(ts, pid, deck_size_int)

            if pid == 0:
                action = agents[0].step(obs, legal, seat_one_hot=seat_oh)
            else:
                if len(pool) == 0 or use_current[pid - 1]:
                    with torch.no_grad():
                        x = np.array(obs, dtype=np.float32)
                        if args.add_seat_id: x = np.concatenate([x, seat_oh], axis=0)
                        device = next(agents[0]._policy.parameters()).device
                        x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
                        logits = agents[0]._policy(x_t).squeeze(0)
                        legal_mask = torch.zeros(num_actions, dtype=torch.float32, device=device); legal_mask[legal] = 1.0
                        probs = ppo.masked_softmax(logits, legal_mask)
                        action = int(torch.distributions.Categorical(probs=probs).sample().item())
                else:
                    action = snap_policies[pid].act(obs, legal, seat_one_hot=seat_oh)

            ts_next = env.step([int(action)])

            if pid == 0:
                hand_after = shaper.hand_size(ts_next, pid, deck_size_int)
                r = shaper.step_reward(hand_before=hand_before, hand_after=hand_after)
                agents[0].post_step(r, done=ts_next.last())

            ts = ts_next

        # terminal reward + train
        if shaper.include_env_reward():
            agents[0]._buffer.finalize_last_reward(env._state.returns()[0])
        agents[0].train()

        # snapshots
        if ep % args.snapshot_interval == 0:
            pool.append(make_snapshot(agents[0]._policy))
            if len(pool) > args.pool_cap:
                pool.pop(0)

        # eval + save
        if ep % args.eval_interval == 0:
            wins = 0
            for _ in range(args.eval_episodes):
                st = game.new_initial_state()
                while not st.is_terminal():
                    pid = st.current_player()
                    legal = st.legal_actions(pid)
                    if pid == 0:
                        o = st.information_state_tensor(pid)
                        if args.add_seat_id:
                            seat_oh = np.zeros(num_players, dtype=np.float32); seat_oh[pid] = 1.0
                            x = np.concatenate([np.array(o, dtype=np.float32), seat_oh], axis=0)
                        else:
                            x = np.array(o, dtype=np.float32)
                        with torch.no_grad():
                            logits = agents[0]._policy(torch.tensor(x, dtype=torch.float32)).squeeze(0)
                            legal_mask = torch.zeros(num_actions, dtype=torch.float32); legal_mask[legal] = 1.0
                            probs = ppo.masked_softmax(logits, legal_mask)
                            a = int(torch.distributions.Categorical(probs=probs).sample().item())
                    else:
                        # simple heuristic opponents in eval to track progress
                        a = max(legal) if len(legal) > 1 else legal[0]
                    st.apply_action(a)
                if st.returns()[0] == max(st.returns()):
                    wins += 1
            wr = 100.0 * wins / args.eval_episodes
            print(f"✅ Eval nach {ep} – Winrate P0: {wr:.1f}%")

            base = os.path.join(model_dir, f"ppo_model_{version}_agent_p0_ep{ep:07d}")
            agents[0].save(base)
            print(f"💾 Modell gespeichert unter: {base}")

    print("✅ K3 Training fertig.")


if __name__ == "__main__":
    main()
