# -*- coding: utf-8 -*-
import os, re, datetime, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
import pyspiel
from open_spiel.python import rl_environment
from agents import dqn_agent as dqn
from utils import STRATS

# ============== CONFIG (im Skript änderbar) ==============
CONFIG = {
    "EPISODES":        10_000,
    "EVAL_INTERVAL":   1000,
    "EVAL_EPISODES":   2_000,
    "DECK_SIZE":       "64",
    "SEED":            42,
    "OPPONENTS":       ["max_combo","max_combo","max_combo"],

    # DQN-Hyperparameter (alle Pflichtfelder vorhanden)
    "DQN": {
        "learning_rate": 1e-3,
        "batch_size": 64,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay": 0.995,
        "buffer_size": 100_000,
        "target_update_freq": 1000,
        "soft_target_tau": 0.005,
        "max_grad_norm": 5.0,
        "use_double_dqn": True,
        "loss_huber_delta": 1.0,
    },

    # Reward-Shaping (für Konsistenz mit PPO)
    "REWARD": {
        "STEP": "delta_hand",
        "DELTA_WEIGHT": 1.0,
        "HAND_PENALTY_COEFF": 0.0,
        "FINAL": "none",
        "BONUS_WIN": 0.0, "BONUS_2ND": 0.0, "BONUS_3RD": 0.0, "BONUS_LAST": 0.0,
        "ENV_REWARD": True,
    },
}

# ===== Helpers / Heuristiken (gleich wie oben, gekürzt) =====
def find_next_version(models_root, prefix):
    pat=re.compile(rf"^{re.escape(prefix)}_(\d{{2}})$"); os.makedirs(models_root,exist_ok=True)
    existing=[int(m.group(1)) for m in (pat.match(n) for n in os.listdir(models_root)) if m]
    return f"{max(existing)+1:02d}" if existing else "01"

class RewardShaper:
    def __init__(self,cfg): self.step=cfg["STEP"]; self.dw=float(cfg["DELTA_WEIGHT"])
    @staticmethod
    def _ranks(deck): return 8 if deck in (32,64) else 13
    def hand_size(self, ts, pid, deck): return int(sum(ts.observations["info_state"][pid][:self._ranks(deck)]))
    def step_reward(self, *, hand_before=None, hand_after=None, **_): 
        return 0.0 if self.step=="none" else self.dw*max(0.0,float(hand_before-hand_after))

def random2(s):
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
STRATS={"random2":random2,"max_combo":max_combo,"single_only":single_only}

# =========================== Training ===========================
def main():
    np.random.seed(CONFIG["SEED"]); torch.manual_seed(CONFIG["SEED"])
    ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); MODELS=os.path.join(ROOT,"models")

    game = pyspiel.load_game("president", {
        "num_players":4, "deck_size":CONFIG["DECK_SIZE"], "shuffle_cards":True, "single_card_mode":False,
    })
    env = rl_environment.Environment(game)
    obs_dim = game.observation_tensor_shape()[0]       # WICHTIG: DQN nutzt observation_tensor
    A = env.action_spec()["num_actions"]
    deck_int = int(CONFIG["DECK_SIZE"])

    version = find_next_version(MODELS, "dqn_model")
    model_dir = os.path.join(MODELS, f"dqn_model_{version}", "train"); os.makedirs(model_dir, exist_ok=True)

    dqn_cfg = dqn.DQNConfig(**CONFIG["DQN"])
    agent = dqn.DQNAgent(state_size=obs_dim, num_actions=A, config=dqn_cfg)
    opponents = [STRATS[name] for name in CONFIG["OPPONENTS"]]
    shaper = RewardShaper(CONFIG["REWARD"])

    pd.DataFrame([{
        "version":version,"timestamp":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent_type":"DQN","num_episodes":CONFIG["EPISODES"],"eval_interval":CONFIG["EVAL_INTERVAL"],
        "eval_episodes":CONFIG["EVAL_EPISODES"],"deck_size":CONFIG["DECK_SIZE"],
        "observation_dim":obs_dim,"num_actions":A,"model_version_dir":model_dir,
        "opponents":",".join(CONFIG["OPPONENTS"]),
    }]).to_csv(os.path.join(os.path.dirname(model_dir), "training_runs.csv"), index=False)

    eval_wrs=[]; EINT, EEPS = CONFIG["EVAL_INTERVAL"], CONFIG["EVAL_EPISODES"]

    for ep in range(1, CONFIG["EPISODES"]+1):
        ts = env.reset()
        while not ts.last():
            p = ts.observations["current_player"]; legal = ts.observations["legal_actions"][p]
            hand_before = shaper.hand_size(ts, p, deck_int)

            if p==0:
                s = np.array(env._state.observation_tensor(0), dtype=np.float32)
                a = int(agent.select_action(s, legal))
            else:
                a = int(opponents[p-1](env._state))

            ts_next = env.step([a])

            if p==0:
                s_next = np.array(env._state.observation_tensor(0), dtype=np.float32) if not ts_next.last() else s
                hand_after = shaper.hand_size(ts_next, p, deck_int)
                r = shaper.step_reward(hand_before=hand_before, hand_after=hand_after)
                agent.buffer.add(s, a, float(r), s_next, bool(ts_next.last()))
                agent.train_step()

            ts = ts_next

        if ep % EINT == 0:
            wins=0
            for _ in range(EEPS):
                st=game.new_initial_state()
                while not st.is_terminal():
                    pid=st.current_player(); legal=st.legal_actions(pid)
                    if pid==0:
                        obs=np.array(st.observation_tensor(pid),dtype=np.float32)
                        a=int(agent.select_action(obs, legal))
                    else:
                        a=int(opponents[pid-1](st))
                    st.apply_action(a)
                if st.returns()[0]==max(st.returns()): wins+=1
            wr=100.0*wins/EEPS; eval_wrs.append(wr)
            print(f"✅ Eval nach {ep} Episoden – Winrate: {wr:.1f}%")
            base_latest=os.path.join(model_dir, f"dqn_model_{version}_agent_p0")
            base_ep    =os.path.join(model_dir, f"dqn_model_{version}_agent_p0_ep{ep:07d}")
            agent.save(base_latest); agent.save(base_ep)

    if eval_wrs:
        xs=list(range(EINT, CONFIG["EPISODES"]+1, EINT))
        plt.figure(figsize=(10,6)); plt.plot(xs, eval_wrs, marker="o"); plt.title("K1 – DQN vs Heuristiken")
        plt.xlabel("Episode"); plt.ylabel("Winrate (%)"); plt.grid(True)
        plots=os.path.join(os.path.dirname(model_dir),"plots"); os.makedirs(plots,exist_ok=True)
        out=os.path.join(plots,"lernkurven.png"); plt.tight_layout(); plt.savefig(out); plt.close()
        print(f"📄 Lernkurve gespeichert unter: {out}")

if __name__=="__main__": main()
