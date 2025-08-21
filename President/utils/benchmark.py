# utils/benchmark.py
import numpy as np
import torch
import pyspiel
from agents import ppo_agent as ppo  # für masked_softmax
from utils.fit_tensor import augment_observation

def _get_policy_module(agent):
    """
    Liefert ein nn.Module, das aus einem (1D/2D) State-Input Action-Scores erzeugt.
    - PPO:   agent._policy
    - DQN v1 (dqn_agent.py):   agent.q_network
    - DQN v2 (dqn_agent2.py):  agent.q
    - Fallbacks:                agent.q_net
    """
    if hasattr(agent, "_policy"):
        return agent._policy
    for name in ("q_network", "q", "q_net"):
        if hasattr(agent, name):
            return getattr(agent, name)
    raise AttributeError("Kein kompatibles Policy-/Q-Modul am Agent gefunden "
                         "(erwartet _policy / q_network / q / q_net).")

@torch.no_grad()
def _policy_logits_for_benchmark(agent, obs_vec, *, seat_id: int, num_players: int):
    policy_mod = _get_policy_module(agent)
    try:
        device = next(policy_mod.parameters()).device
    except Exception:
        device = torch.device("cpu")

    if hasattr(agent, "_make_input"):
        seat_oh = np.zeros(num_players, dtype=np.float32)
        seat_oh[seat_id] = 1.0
        x = agent._make_input(obs_vec, seat_one_hot=seat_oh)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(device)
    else:
        x = torch.tensor(obs_vec, dtype=torch.float32, device=device)

    # >>> NEU: Batch-Dimension sicherstellen
    if x.ndim == 1:
        x = x.unsqueeze(0)   # [1, F]

    out = policy_mod(x)

    # >>> NEU: wieder auf 1D zurück
    if out.ndim == 2 and out.size(0) == 1:
        out = out.squeeze(0)

    return out  # [A]


def run_benchmark(game, agent, opponents_dict, opponent_names, episodes, feat_cfg, num_actions):
    """
    Simuliert 'episodes' Spiele pro Gegner in 'opponent_names' gegen den aktuellen Agenten.
    Liefert pro Gegner:
      - winrate    : in %
      - reward     : mittlerer ENV-Reward von P0
      - places     : Anteile [1st, 2nd, 3rd, 4th]
      - episodes   : Anzahl getesteter Episoden
    """
    results = {}
    num_players = game.num_players()

    for name in opponent_names:
        opp_fn = opponents_dict[name]
        wins = 0
        rewards = []
        place_counts = [0, 0, 0, 0]

        for _ in range(episodes):
            st = game.new_initial_state()
            while not st.is_terminal():
                pid = st.current_player()
                legal = st.legal_actions(pid)

                if pid == 0:
                    obs = st.information_state_tensor(pid)
                    obs = augment_observation(obs, player_id=pid, cfg=feat_cfg)
                    logits = _policy_logits_for_benchmark(agent, obs_vec=obs, seat_id=0, num_players=num_players)

                    # Maske der legalen Aktionen
                    mask = torch.zeros(num_actions, dtype=torch.float32, device=logits.device)
                    mask[legal] = 1.0

                    # DQN erkennen (hat typischerweise q / q_network / q_net)
                    is_dqn = any(hasattr(agent, n) for n in ("q", "q_network", "q_net"))
                    if is_dqn:
                        # Greedy: argmax über Q nur auf legalen Actions
                        neg = torch.finfo(logits.dtype).min / 2
                        masked_q = torch.where(mask > 0, logits, neg)
                        a = int(torch.argmax(masked_q).item())
                    else:
                        # PPO: Sampling aus maskierter Policy
                        probs = ppo.masked_softmax(logits, mask)
                        a = int(torch.distributions.Categorical(probs=probs).sample().item())
                else:
                    a = int(opp_fn(st))

                st.apply_action(a)

            ret0 = float(st.returns()[0])
            rewards.append(ret0)
            place_idx = int(num_players - 1 - ret0)
            place_idx = max(0, min(num_players - 1, place_idx))
            place_counts[place_idx] += 1
            if place_idx == 0:
                wins += 1

        results[name] = {
            "winrate": 100.0 * wins / episodes,
            "reward": float(np.mean(rewards)),
            "places": [c / episodes for c in place_counts],
            "episodes": int(episodes),
        }

    return results
