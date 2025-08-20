# external_trainer.py
import os, glob, json, numpy as np, torch
from agents.ppo_agent import PPOAgent, PPOConfig  # nutzt dieselbe Klasse!

def load_bundle(npz_path):
    data = np.load(npz_path, allow_pickle=False)
    raw = data["meta"]
    v = raw.item() if isinstance(raw, np.ndarray) else raw
    if isinstance(v, bytes):
        v = v.decode("utf-8")
    meta = json.loads(v)

    return data, meta


def compute_advantages_segmented(rewards, values, dones, player_ids, gamma, lam):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        same_next = (t < T-1) and (player_ids[t] == player_ids[t+1]) and (not dones[t])
        next_nonterm = 1.0 if same_next else 0.0
        next_v = values[t+1] if same_next else 0.0
        delta = rewards[t] + gamma * next_v * next_nonterm - values[t]
        lastgaelam = delta + gamma * lam * next_nonterm * lastgaelam
        adv[t] = lastgaelam
    ret = adv + values
    return adv, ret

def train_on_bundles(bundles_dir, weights_dir, latest_tag_file, train_iters=1):
    # Finde alle npz-Bundles, die noch nicht verarbeitet wurden (z.B. per .done Marker)
    paths = sorted(glob.glob(os.path.join(bundles_dir, "*.npz")))
    if not paths:
        print("[trainer] no bundles")
        return

    # Lade Meta von erstem Bundle, initialisiere Agent
    data0, meta0 = load_bundle(paths[0])
    obs_dim = int(meta0["obs_dim"])
    num_actions = int(meta0["num_actions"])
    ppo_cfg = PPOConfig(**meta0["ppo_config"])
    agent = PPOAgent(obs_dim, num_actions, seat_id_dim=0, config=ppo_cfg, device="cpu")

    # Optional: Startgewichte laden, falls vorhanden
    tag_file = os.path.join(weights_dir, latest_tag_file)
    if os.path.isfile(tag_file):
        with open(tag_file, "r") as f:
            tag = f.read().strip()
        if tag:
            base = os.path.join(weights_dir, tag)
            try:
                agent.restore(base)
                print(f"[trainer] restored weights: {tag}")
            except Exception as e:
                print(f"[trainer] restore failed: {e}")

    # Sammle alle Bundles in einen großen Batch
    buf = {"states":[], "actions":[], "rewards":[], "dones":[], "old_log_probs":[],
           "values":[], "legal_masks":[], "player_ids":[]}

    for p in paths:
        d, m = load_bundle(p)
        for k in buf.keys():
            buf[k].append(d[k])
    for k in buf.keys():
        buf[k] = np.concatenate(buf[k], axis=0)

    # GAE (segmentiert)
    adv, ret = compute_advantages_segmented(
        buf["rewards"], buf["values"], buf["dones"], buf["player_ids"],
        gamma=ppo_cfg.gamma, lam=ppo_cfg.gae_lambda
    )

    # Mache ein paar PPO-Updates auf diesem “Mega-Batch”
    # (Wir benutzen die vorhandene trainer-Logik, indem wir den internen Buffer direkt befüllen)
    agent._buffer.states = buf["states"].tolist()
    agent._buffer.actions = buf["actions"].tolist()
    agent._buffer.rewards = buf["rewards"].tolist()
    agent._buffer.dones = buf["dones"].tolist()
    agent._buffer.log_probs = buf["old_log_probs"].tolist()
    agent._buffer.values = buf["values"].tolist()
    agent._buffer.legal_masks = buf["legal_masks"].tolist()
    agent._buffer.player_ids = buf["player_ids"].tolist()

    # Override: falls du lieber die neu berechneten returns/advantages nutzen willst,
    # könntest du die agent.train()-Methode leicht anpassen. Für einen ersten Durchlauf
    # reicht es, die Standard-Train-Logik zu nutzen (sie rechnet GAE selbst).

    for _ in range(train_iters):
        agent.train()

    # Speichere neue Gewichte mit neuem Tag und schreibe LATEST_POLICY.txt
    # (Tag z.B. als laufender Zähler)
    import time
    tag = f"external_v{int(time.time())}"
    base = os.path.join(weights_dir, tag)
    agent.save(base)
    with open(os.path.join(weights_dir, latest_tag_file), "w") as f:
        f.write(tag)
    print(f"[trainer] wrote new weights: {tag}")

    # Markiere Bundles als verarbeitet (optional: verschieben/löschen)
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass

if __name__ == "__main__":
    import argparse, time

    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles_dir", required=True)
    ap.add_argument("--weights_dir", required=True)
    ap.add_argument("--latest_tag_file", default="LATEST_POLICY.txt")
    ap.add_argument("--train_iters", type=int, default=1)
    ap.add_argument("--poll_every", type=float, default=2.0)  # Sekunden
    args = ap.parse_args()

    print("[trainer] starting…")
    while True:
        train_on_bundles(
            bundles_dir=args.bundles_dir,
            weights_dir=args.weights_dir,
            latest_tag_file=args.latest_tag_file,
            train_iters=args.train_iters,
        )
        time.sleep(args.poll_every)

