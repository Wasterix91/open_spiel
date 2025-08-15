# -*- coding: utf-8 -*-
import os, re, datetime, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
import pyspiel
from open_spiel.python import rl_environment
from agents import ppo_agent as ppo
from utils import STRATS

# ============== CONFIG  ==============
CONFIG = {
    "EPISODES":        20_000,
    "EVAL_INTERVAL":   2000,
    "EVAL_EPISODES":   2_000,
    "DECK_SIZE":       "64",      # "32" | "52" | "64"
    "SEED":            42,
    "OPPONENTS":       ["max_combo", "max_combo", "max_combo"],

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
}

# ================= Helpers / Heuristiken =================
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

""" def random2(s):
    legal=s.legal_actions(); 
    if len(legal)>1 and 0 in legal: legal=[a for a in legal if a!=0]
    return int(np.random.choice(legal))
def max_combo(s):
    pid=s.current_player(); dec=[(a,s.action_to_string(pid,a)) for a in s.legal_actions()]
    if not dec: return 0
    def prio(t): return 4 if "Quad" in t else 3 if "Triple" in t else 2 if "Pair" in t else 1
    return max(dec, key=lambda x:(prio(x[1]),-x[0]))[0]
def single_only(s):
    pid=s.current_player(); dec=[(a,s.action_to_string(pid,a)) for a in s.legal_actions()]
    singles=[x for x in dec if "Single" in x[1]]; return singles[0][0] if singles else 0
STRATS={"random2":random2,"max_combo":max_combo,"single_only":single_only} """

# =========================== Training ===========================
def main():
    np.random.seed(CONFIG["SEED"]); torch.manual_seed(CONFIG["SEED"])
    ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); MODELS=os.path.join(ROOT,"models")

    game = pyspiel.load_game("president", {
        "num_players":4, "deck_size":CONFIG["DECK_SIZE"], "shuffle_cards":True, "single_card_mode":False,
    })
    env = rl_environment.Environment(game)
    info_dim = env.observation_spec()["info_state"][0]; A = env.action_spec()["num_actions"]
    deck_int = int(CONFIG["DECK_SIZE"])

    version = find_next_version(MODELS, "ppo_model")
    model_dir = os.path.join(MODELS, f"ppo_model_{version}", "train"); os.makedirs(model_dir, exist_ok=True)

    ppo_cfg = ppo.PPOConfig(**CONFIG["PPO"])
    agent = ppo.PPOAgent(info_dim, A, seat_id_dim=0, config=ppo_cfg)
    opponents = [STRATS[name] for name in CONFIG["OPPONENTS"]]
    shaper = RewardShaper(CONFIG["REWARD"])

    # run log
    pd.DataFrame([{
        "version":version,"timestamp":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent_type":"PPO","num_episodes":CONFIG["EPISODES"],"eval_interval":CONFIG["EVAL_INTERVAL"],
        "eval_episodes":CONFIG["EVAL_EPISODES"],"deck_size":CONFIG["DECK_SIZE"],
        "observation_dim":info_dim,"num_actions":A,"model_version_dir":model_dir,
        "step_reward":CONFIG["REWARD"]["STEP"],"final_reward":CONFIG["REWARD"]["FINAL"],"env_reward":CONFIG["REWARD"]["ENV_REWARD"],
        "opponents":",".join(CONFIG["OPPONENTS"]),
    }]).to_csv(os.path.join(os.path.dirname(model_dir), "training_runs.csv"), index=False)

    eval_wrs=[]; EINT, EEPS = CONFIG["EVAL_INTERVAL"], CONFIG["EVAL_EPISODES"]

    for ep in range(1, CONFIG["EPISODES"]+1):
        ts = env.reset()
        while not ts.last():
            p = ts.observations["current_player"]; legal = ts.observations["legal_actions"][p]
            hand_before = shaper.hand_size(ts, p, deck_int)
            if p==0:
                a = int(agent.step(ts.observations["info_state"][p], legal))
            else:
                a = int(opponents[p-1](env._state))
            ts_next = env.step([a])
            if p==0:
                hand_after = shaper.hand_size(ts_next, p, deck_int)
                r = shaper.step_reward(hand_before=hand_before, hand_after=hand_after, time_step=ts_next, player_id=p, deck_size=deck_int)
                agent.post_step(r, done=ts_next.last())
            ts = ts_next

        if shaper.include_env_reward(): agent._buffer.finalize_last_reward(ts.rewards[0])
        agent._buffer.finalize_last_reward(shaper.final_bonus(ts.rewards, 0))
        agent.train()

        if ep % EINT == 0:
            wins=0
            for _ in range(EEPS):
                st=game.new_initial_state()
                while not st.is_terminal():
                    pid=st.current_player(); legal=st.legal_actions(pid)
                    if pid==0:
                        obs=st.information_state_tensor(pid)
                        with torch.no_grad():
                            logits=agent._policy(torch.tensor(obs,dtype=torch.float32))
                            mask=torch.zeros(A); mask[legal]=1.0
                            probs=ppo.masked_softmax(logits, mask)
                        a=int(torch.distributions.Categorical(probs=probs).sample().item())
                    else:
                        a=int(opponents[pid-1](st))
                    st.apply_action(a)
                if st.returns()[0]==max(st.returns()): wins+=1
            wr=100.0*wins/EEPS; eval_wrs.append(wr)
            print(f"✅ Eval nach {ep} Episoden – Winrate: {wr:.1f}%")
            base=os.path.join(model_dir, f"ppo_model_{version}_agent_p0_ep{ep:07d}"); agent.save(base)

    if eval_wrs:
        xs=list(range(EINT, CONFIG["EPISODES"]+1, EINT))
        plt.figure(figsize=(10,6)); plt.plot(xs, eval_wrs, marker="o"); plt.title("K1 – PPO vs Heuristiken")
        plt.xlabel("Episode"); plt.ylabel("Winrate (%)"); plt.grid(True)
        plots=os.path.join(os.path.dirname(model_dir),"plots"); os.makedirs(plots,exist_ok=True)
        out=os.path.join(plots,"lernkurven.png"); plt.tight_layout(); plt.savefig(out); plt.close()
        print(f"📄 Lernkurve gespeichert unter: {out}")

if __name__=="__main__": main()
