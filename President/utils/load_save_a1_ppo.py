# utils/load_save_a1_ppo.py
import os, glob, re
import torch

def save_checkpoint_ppo(agent, weights_dir: str, tag: str):
    """
    Speichert PPO-Policy/Value unter:
      <weights_dir>/<tag>_policy.pt
      <weights_dir>/<tag>_value.pt
    """
    os.makedirs(weights_dir, exist_ok=True)
    base = os.path.join(weights_dir, tag)
    torch.save(agent._policy.state_dict(), base + "_policy.pt")
    torch.save(agent._value.state_dict(),  base + "_value.pt")
    return base

def load_checkpoint_ppo(agent, weights_dir: str, tag: str):
    base = os.path.join(weights_dir, tag)
    agent._policy.load_state_dict(torch.load(base + "_policy.pt", map_location="cpu"))
    agent._value.load_state_dict(torch.load(base + "_value.pt",  map_location="cpu"))

def latest_checkpoint_tag(weights_dir: str, pattern_prefix: str) -> str | None:
    """
    Findet den höchsten ep im Schema:
      '<pattern_prefix>_epXXXXXXX_policy.pt'
    und gibt den Basistag ohne Suffix zurück (z.B. 'k1a1_model_01_agent_p0_ep0001000').
    """
    glob_pat = os.path.join(weights_dir, f"{pattern_prefix}_ep*_policy.pt")
    files = glob.glob(glob_pat)
    best = None; best_ep = -1
    rx = re.compile(rf"{re.escape(pattern_prefix)}_ep(\d+)_policy\.pt$")
    for f in files:
        m = rx.search(os.path.basename(f))
        if m:
            ep = int(m.group(1))
            if ep > best_ep:
                best_ep = ep
                best = os.path.splitext(f)[0][:-len("_policy")]  # Basistag ohne _policy
    return best if best_ep >= 0 else None
