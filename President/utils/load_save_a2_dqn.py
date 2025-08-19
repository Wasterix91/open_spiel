# utils/load_save_a2_dqn.py
import os, glob, re
import torch

def save_checkpoint_dqn(agent, weights_dir: str, tag: str, save_target: bool = True):
    """
    Speichert DQN unter:
      <weights_dir>/<tag>_q.pt
      <weights_dir>/<tag>_target.pt (optional, falls vorhanden)
    """
    os.makedirs(weights_dir, exist_ok=True)
    base = os.path.join(weights_dir, tag)
    torch.save(agent.q_net.state_dict(), base + "_q.pt")
    if save_target and hasattr(agent, "target_net") and agent.target_net is not None:
        torch.save(agent.target_net.state_dict(), base + "_target.pt")
    return base

def load_checkpoint_dqn(agent, weights_dir: str, tag: str, load_target: bool = True):
    base = os.path.join(weights_dir, tag)
    agent.q_net.load_state_dict(torch.load(base + "_q.pt", map_location="cpu"))
    if load_target and hasattr(agent, "target_net") and agent.target_net is not None:
        agent.target_net.load_state_dict(torch.load(base + "_target.pt", map_location="cpu"))

def latest_checkpoint_tag(weights_dir: str, pattern_prefix: str) -> str | None:
    glob_pat = os.path.join(weights_dir, f"{pattern_prefix}_ep*_q.pt")
    files = glob.glob(glob_pat)
    best = None; best_ep = -1
    rx = re.compile(rf"{re.escape(pattern_prefix)}_ep(\d+)_q\.pt$")
    for f in files:
        m = rx.search(os.path.basename(f))
        if m:
            ep = int(m.group(1))
            if ep > best_ep:
                best_ep = ep
                best = os.path.splitext(f)[0][:-len("_q")]  # Basistag ohne _q
    return best if best_ep >= 0 else None
