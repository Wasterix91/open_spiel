# utils/load_save_a2_dqn.py
import os, glob, re
import torch

def save_checkpoint_dqn(agent, weights_dir: str, tag: str, save_target: bool = True):
    """
    Speichert DQN unter neuen Namen und (für Kompatibilität) zusätzlich Legacy:
      <weights_dir>/<tag>_qnet.pt   (+ Legacy _q.pt)
      <weights_dir>/<tag>_tgt.pt    (+ Legacy _target.pt)
    """
    os.makedirs(weights_dir, exist_ok=True)
    base = os.path.join(weights_dir, tag)

    # Bevorzugte neuen Namen
    torch.save(agent.q_net.state_dict(), base + "_qnet.pt")
    if save_target and hasattr(agent, "target_net") and agent.target_net is not None:
        torch.save(agent.target_net.state_dict(), base + "_tgt.pt")

    # Kompatibilität: zusätzlich Legacy-Dateien schreiben (optional)
    try:
        torch.save(agent.q_net.state_dict(), base + "_q.pt")
        if save_target and hasattr(agent, "target_net") and agent.target_net is not None:
            torch.save(agent.target_net.state_dict(), base + "_target.pt")
    except Exception:
        pass

    return base


def load_checkpoint_dqn(agent, weights_dir: str, tag: str, load_target: bool = True):
    """
    Lädt zuerst neue Namen (_qnet/_tgt), sonst Legacy (_q/_target).
    """
    base = os.path.join(weights_dir, tag)
    qnet_p = base + "_qnet.pt"
    tgt_p  = base + "_tgt.pt"
    legacy_q = base + "_q.pt"
    legacy_t = base + "_target.pt"

    map_loc = getattr(agent, "device", "cpu")

    if os.path.exists(qnet_p):
        agent.q_net.load_state_dict(torch.load(qnet_p, map_location=map_loc))
        if load_target and hasattr(agent, "target_net") and agent.target_net is not None and os.path.exists(tgt_p):
            agent.target_net.load_state_dict(torch.load(tgt_p, map_location=map_loc))
    else:
        agent.q_net.load_state_dict(torch.load(legacy_q, map_location=map_loc))
        if load_target and hasattr(agent, "target_net") and agent.target_net is not None and os.path.exists(legacy_t):
            agent.target_net.load_state_dict(torch.load(legacy_t, map_location=map_loc))


# utils/load_save_a2_dqn.py
def latest_checkpoint_tag(*args, **kwargs):
    raise RuntimeError(
        "latest_checkpoint_tag ist deaktiviert: Episoden müssen explizit angegeben werden."
    )
