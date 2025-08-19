# utils/benchmark.py
import numpy as np
import torch
import pyspiel
from agents import ppo_agent as ppo  # für masked_softmax
from utils.fit_tensor import augment_observation

def _policy_logits_for_benchmark(agent, obs_vec, *, seat_id: int, num_players: int):
    """
    Liefert Policy-Logits für obs_vec.
    Nutzt agent._make_input(), falls vorhanden, und hängt IMMER die Seat-OneHot an.
    Der Agent ignoriert sie intern, wenn _seat_id_dim == 0.
    """
    try:
        policy_device = next(agent._policy.parameters()).device
    except Exception:
        policy_device = torch.device("cpu")

    if hasattr(agent, "_make_input"):
        seat_oh = np.zeros(num_players, dtype=np.float32)
        seat_oh[seat_id] = 1.0
        x = agent._make_input(obs_vec, seat_one_hot=seat_oh)  # np->Tensor handled intern
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(policy_device)
        return agent._policy(x)

    # Fallback: kein _make_input vorhanden → direkte Übergabe (ohne OneHot)
    x = torch.tensor(obs_vec, dtype=torch.float32, device=policy_device)
    return agent._policy(x)



def run_benchmark(game, agent, opponents_dict, opponent_names, episodes, feat_cfg, num_actions):
    """
    Simuliert 'episodes' Spiele pro Gegner in 'opponent_names' gegen den aktuellen Agenten.
    Liefert pro Gegner:
      - winrate    : in %
      - reward     : mittlerer ENV-Reward von P0 (entspricht terminalem return von P0)
      - places     : relative Häufigkeit [1st, 2nd, 3rd, 4th] (Summe ~ 1.0)
      - episodes   : Anzahl getesteter Episoden
    """
    results = {}
    num_players = game.num_players()

    for name in opponent_names:
        opp_fn = opponents_dict[name]
        wins = 0
        rewards = []
        place_counts = [0, 0, 0, 0]  # Indizes: 0→1st, 1→2nd, 2→3rd, 3→4th

        for _ in range(episodes):
            st = game.new_initial_state()
            while not st.is_terminal():
                pid = st.current_player()
                legal = st.legal_actions(pid)

                if pid == 0:
                    obs = st.information_state_tensor(pid)
                    obs = augment_observation(obs, player_id=pid, cfg=feat_cfg)
                    with torch.no_grad():
                        logits = _policy_logits_for_benchmark(
                            agent, obs_vec=obs, seat_id=0, num_players=num_players
                        )
                        mask = torch.zeros(num_actions, dtype=torch.float32, device=logits.device)
                        mask[legal] = 1.0
                        probs = ppo.masked_softmax(logits, mask)
                        a = int(torch.distributions.Categorical(probs=probs).sample().item())

                else:
                    a = int(opp_fn(st))

                st.apply_action(a)

            # --- Terminal: returns() enthält den ENV-Reward (kTerminal) ---
            ret0 = float(st.returns()[0])      # 3/2/1/0 bei 4 Spielern
            rewards.append(ret0)

            # Platz ableiten: bei N Spielern -> Platzindex = (N-1 - return)
            place_idx = int(num_players - 1 - ret0)
            place_idx = max(0, min(num_players - 1, place_idx))
            place_counts[place_idx] += 1

            if place_idx == 0:
                wins += 1

        results[name] = {
            "winrate": 100.0 * wins / episodes,
            "reward":  float(np.mean(rewards)),
            "places":  [c / episodes for c in place_counts],
            "episodes": int(episodes),
        }

    return results
