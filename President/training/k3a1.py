# -*- coding: utf-8 -*-
# President/training/k3a1.py — PPO (K3: League Selfplay)
# League-Selfplay + Eval vs Heuristiken (Single_Only/Max_Combo/Random2 + Macro)

import os, re, datetime, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
import pyspiel
from open_spiel.python import rl_environment
from agents import ppo_agent as ppo
from utils.strategies import STRATS                              # <-- NEU (Heuristiken)
from utils.fit_tensor import FeatureConfig, augment_observation
from utils.training_eval_plots import EvalPlotter               # <-- NEU (Plotter)

# ============== CONFIG ==============
CONFIG = {
    "EPISODES":        200_000,
    "EVAL_INTERVAL":   10_000,
    "EVAL_EPISODES":   2_000,
    "DECK_SIZE":       "64",
    "SEED":            123,
    "PPO": {
        "learning_rate": 3e-4, "num_epochs": 4, "batch_size": 256,
        "entropy_cost": 0.01, "gamma": 0.99, "gae_lambda": 0.95,
        "clip_eps": 0.2, "value_coef": 0.5, "max_grad_norm": 0.5,
    },
    "REWARD": {
        "STEP": "delta_hand", "DELTA_WEIGHT": 1.0, "HAND_PENALTY_COEFF": 0.0,
        "FINAL": "none", "BONUS_WIN": 0.0, "BONUS_2ND": 0.0, "BONUS_3RD": 0.0, "BONUS_LAST": 0.0,
        "ENV_REWARD": True,
    },
    "FEATURES": {
        "NORMALIZE": False,
        "SEAT_ONEHOT": True,     # K3: absolute Seat-One-Hot an den Agent anhängen
    },
    "LEAGUE": {
        "POOL_SIZE": 6, "SAVE_ALL": False, "MAIN_INDEX": 0,
    },
    # Heuristik-Evalkurven (analog K1)
    "EVAL_CURVES": ["single_only", "max_combo", "random2"],
}

# ============== Helpers ==============
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
        self.b = (float(cfg["BONUS_WIN"]), float(cfg["BONUS_2ND"]), float(cfg["BONUS_3RD"]), float(cfg["BONUS_LAST"]))
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

# =========================== Training ===========================
def main():
    np.random.seed(CONFIG["SEED"]); torch.manual_seed(CONFIG["SEED"])
    ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); MODELS=os.path.join(ROOT,"models")

    game = pyspiel.load_game("president", {
        "num_players":4, "deck_size":CONFIG["DECK_SIZE"], "shuffle_cards":True, "single_card_mode":False,
    })
    env = rl_environment.Environment(game)
    info_dim = env.observation_spec()["info_state"][0]
    A = env.action_spec()["num_actions"]

    deck_int = int(CONFIG["DECK_SIZE"])
    num_players = game.num_players()
    num_ranks   = 8 if deck_int in (32,64) else 13 if deck_int==52 else (_ for _ in ()).throw(ValueError("deck"))
    # Normierung via augment_observation; Seat-One-Hot hängt der Agent intern an (seat_id_dim)
    feat_cfg = FeatureConfig(
        num_players=num_players, num_ranks=num_ranks,
        add_seat_onehot=False,                             #  <<< wichtig: hier FALSE
        normalize=bool(CONFIG["FEATURES"]["NORMALIZE"]),
    )

    POOL = CONFIG["LEAGUE"]["POOL_SIZE"]
    MAIN = CONFIG["LEAGUE"]["MAIN_INDEX"]

    version = find_next_version(MODELS, "ppo_league")
    model_dir = os.path.join(MODELS, f"ppo_league_{version}", "train"); os.makedirs(model_dir, exist_ok=True)

    # EvalPlotter für Heuristik-Kurven
    plotter = EvalPlotter(
        opponent_names=list(CONFIG["EVAL_CURVES"]),
        out_dir=model_dir,
        filename_prefix="lernkurve",          # -> lernkurve_single_only.png etc.
        csv_filename="eval_curves.csv",
        save_csv=True,
    )

    ppo_cfg = ppo.PPOConfig(**CONFIG["PPO"])
    agents = [ppo.PPOAgent(info_state_size=info_dim, num_actions=A,
                           seat_id_dim=(num_players if CONFIG["FEATURES"]["SEAT_ONEHOT"] else 0),
                           config=ppo_cfg)
              for _ in range(POOL)]

    shaper = RewardShaper(CONFIG["REWARD"])

    # Run-Log
    pd.DataFrame([{
        "version":version,"timestamp":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent_type":"PPO-K3-LEAGUE","num_episodes":CONFIG["EPISODES"],"eval_interval":CONFIG["EVAL_INTERVAL"],
        "eval_episodes":CONFIG["EVAL_EPISODES"],"deck_size":CONFIG["DECK_SIZE"],
        "observation_dim":info_dim,"num_actions":A,"model_version_dir":model_dir,
        "normalize":CONFIG["FEATURES"]["NORMALIZE"],"seat_onehot":CONFIG["FEATURES"]["SEAT_ONEHOT"],
        "pool_size":POOL,
    }]).to_csv(os.path.join(os.path.dirname(model_dir), "training_runs.csv"), index=False)

    EINT, EEPS = CONFIG["EVAL_INTERVAL"], CONFIG["EVAL_EPISODES"]
    league_wrs=[]  # MAIN vs Pool (wie bisher)

    for ep in range(1, CONFIG["EPISODES"]+1):
        # —— Matchmaking: zufällige 4 Policies aus dem Pool ——
        lineup_idx = np.random.choice(POOL, size=num_players, replace=False)
        lineup = [agents[i] for i in lineup_idx]

        ts = env.reset()
        while not ts.last():
            p = ts.observations["current_player"]; legal = ts.observations["legal_actions"][p]
            base_obs = ts.observations["info_state"][p]
            obs = augment_observation(base_obs, player_id=p, cfg=feat_cfg)

            seat_oh = None
            if CONFIG["FEATURES"]["SEAT_ONEHOT"]:
                seat_oh = np.zeros((num_players,), dtype=np.float32); seat_oh[p] = 1.0

            a = int(lineup[p].step(obs, legal, seat_one_hot=seat_oh))
            ts_next = env.step([a])

            # Reward shaping pro Sitz
            hand_before = int(sum(base_obs[:num_ranks]))
            hand_after  = int(sum(ts_next.observations["info_state"][p][:num_ranks]))
            r = shaper.step_reward(hand_before=hand_before, hand_after=hand_after, time_step=ts_next, player_id=p, deck_size=deck_int)
            lineup[p].post_step(r, done=ts_next.last())
            ts = ts_next

        # Terminal: Env-Rewards für alle Policies, die gespielt haben
        finals = env._state.returns()
        for seat in range(num_players):
            lineup[seat]._buffer.finalize_last_reward(finals[seat])
            # Optional: lineup[seat]._buffer.finalize_last_reward(shaper.final_bonus(finals, seat))
        for seat in range(num_players):
            lineup[seat].train()

        # —— Evaluation & Speichern ——
        if ep % EINT == 0:
            # 1) League-Eval (MAIN vs random Pool-Lineup) – bestehende Kurve
            wins=0
            for _ in range(EEPS):
                st = game.new_initial_state()
                others = [i for i in range(POOL) if i != MAIN]
                sampled = np.random.choice(others, size=(num_players-1), replace=False)
                eval_lineup = [agents[MAIN]] + [agents[i] for i in sampled]
                while not st.is_terminal():
                    pid = st.current_player(); legal = st.legal_actions(pid)
                    ob = st.information_state_tensor(pid)
                    ob = augment_observation(ob, player_id=pid, cfg=feat_cfg)
                    seat_oh = None
                    if CONFIG["FEATURES"]["SEAT_ONEHOT"]:
                        seat_oh = np.zeros((num_players,), dtype=np.float32); seat_oh[pid] = 1.0
                    # <<< FIX: Input korrekt bauen (inkl. Seat-One-Hot) >>>
                    x = eval_lineup[pid]._make_input(ob, seat_one_hot=seat_oh)
                    with torch.no_grad():
                        logits = eval_lineup[pid]._policy(x)
                        mask = torch.zeros(A, device=logits.device); mask[legal]=1.0
                        probs = ppo.masked_softmax(logits, mask)
                    a = int(torch.distributions.Categorical(probs=probs).sample().item())
                    st.apply_action(a)
                if st.returns()[0] == max(st.returns()): wins += 1
            wr_league = 100.0*wins/EEPS; league_wrs.append(wr_league)
            print(f"✅ Eval (League) nach {ep} – MAIN Winrate vs Pool: {wr_league:.1f}%")

            # 2) Heuristik-Eval (analog K1): MAIN auf Seat0 vs 3× gleiche Heuristik
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
                            # <<< FIX: auch hier _make_input nutzen >>>
                            x = agents[MAIN]._make_input(ob, seat_one_hot=seat_oh)
                            with torch.no_grad():
                                logits = agents[MAIN]._policy(x)
                                mask = torch.zeros(A, device=logits.device); mask[legal]=1.0
                                probs = ppo.masked_softmax(logits, mask)
                            a = int(torch.distributions.Categorical(probs=probs).sample().item())
                        else:
                            a = int(opp_fn(st))
                        st.apply_action(a)
                    if st.returns()[0] == max(st.returns()): wins += 1
                wr = 100.0 * wins / EEPS
                per_opponent[opp_name] = wr
                print(f"✅ Eval nach {ep} – Winrate vs {opp_name}: {wr:.1f}%")

            # Macro + Plotten
            macro = float(np.mean(list(per_opponent.values())))
            print(f"📊 Macro Average: {macro:.2f}%")
            plotter.add(ep, per_opponent)
            plotter.plot_all()

            # Speichern
            if CONFIG["LEAGUE"]["SAVE_ALL"]:
                for i, ag in enumerate(agents):
                    base=os.path.join(model_dir, f"ppo_league_{version}_agent_{i}_ep{ep:07d}"); ag.save(base)
            else:
                base=os.path.join(model_dir, f"ppo_league_{version}_MAIN_ep{ep:07d}"); agents[MAIN].save(base)

    # Separater Plot für League-Selfplay-Kurve (MAIN vs Pool)
    if league_wrs:
        xs=list(range(EINT, CONFIG["EPISODES"]+1, EINT))
        plt.figure(figsize=(10,6)); plt.plot(xs, league_wrs, marker="o")
        plt.title("K3 – PPO League Selfplay (MAIN vs Pool)")
        plt.xlabel("Episode"); plt.ylabel("Winrate (%)"); plt.grid(True)
        plots=os.path.join(os.path.dirname(model_dir),"plots"); os.makedirs(plots,exist_ok=True)
        out=os.path.join(plots,"lernkurven_k3_league.png"); plt.tight_layout(); plt.savefig(out); plt.close()
        print(f"📄 League-Kurve gespeichert unter: {out}")

if __name__=="__main__": main()
