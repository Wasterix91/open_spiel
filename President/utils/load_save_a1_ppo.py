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

# utils/load_save_a2_dqn.py
def latest_checkpoint_tag(*args, **kwargs):
    raise RuntimeError(
        "latest_checkpoint_tag ist deaktiviert: Episoden müssen explizit angegeben werden."
    )

