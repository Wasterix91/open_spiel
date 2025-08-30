# utils/benchmark.py
import numpy as np
import torch
import pyspiel
from agents import ppo_agent as ppo  # für masked_softmax
from utils.fit_tensor import augment_observation


def _get_policy_module(agent):
    """
    Liefert ein nn.Module, das aus einem (1D/2D) State-Input Action-Scores erzeugt.
    - PPO:                         agent._policy
    - DQN v1 (dqn_agent.py):       agent.q_network
    - DQN v2 (dqn_agent2.py):      agent.q
    - Weitere Fallbacks:           agent.q_net
    - Letzter Fallback:            der Agent selbst, wenn er ein nn.Module ist
    """
    if hasattr(agent, "_policy"):
        return agent._policy
    for name in ("q_network", "q", "q_net"):
        if hasattr(agent, name):
            return getattr(agent, name)

    # Letzter Fallback: Agent selbst ist ein nn.Module mit forward()
    if isinstance(agent, torch.nn.Module):
        return agent

    raise AttributeError(
        "Kein kompatibles Policy-/Q-Modul am Agent gefunden "
        "(erwartet _policy / q_network / q / q_net oder nn.Module-Agent)."
    )


def _expected_in_features(mod, agent) -> int | None:
    """
    Versucht, die erwartete Input-Dimension der Policy/Q-Networks zu ermitteln.
    """
    # häufig: mod.net[0] ist nn.Linear
    net = getattr(mod, "net", None)
    if hasattr(net, "__getitem__") and hasattr(net[0], "in_features"):
        return int(net[0].in_features)

    # manchmal ist das Modul selbst ein Sequential
    if isinstance(mod, torch.nn.Sequential) and hasattr(mod[0], "in_features"):
        return int(mod[0].in_features)

    # weitere gängige Aliasse
    for name in ("encoder", "feature", "backbone", "fc_in", "in"):
        layer = getattr(mod, name, None)
        if hasattr(layer, "in_features"):
            return int(layer.in_features)

    # Fallback: Agent speichert evtl. die Größe selbst
    for name in ("state_size", "obs_dim", "input_dim"):
        if hasattr(agent, name):
            try:
                return int(getattr(agent, name))
            except Exception:
                pass

    return None


@torch.no_grad()
def _policy_logits_for_benchmark(agent, obs_vec, *, seat_id: int, num_players: int, feat_cfg=None):
    """
    Ermittelt automatisch die erwartete Input-Dimension des Netzes und hängt die
    Sitz-One-Hot NUR an, wenn sie erforderlich ist. So bleiben DQN/PPO und beide
    Feature-Varianten (mit/ohne Sitz-One-Hot) kompatibel.
    """
    policy_mod = _get_policy_module(agent)
    try:
        device = next(policy_mod.parameters()).device
    except Exception:
        device = torch.device("cpu")

    # 1) Spezialfall PPO: dessen _make_input kapselt die richtige Feature-Zusammenstellung.
    if hasattr(agent, "_make_input"):
        seat_oh = np.zeros(num_players, dtype=np.float32)
        seat_oh[seat_id] = 1.0
        x = agent._make_input(obs_vec, seat_one_hot=seat_oh)
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        out = policy_mod(x)
        return out.squeeze(0) if out.ndim == 2 and out.size(0) == 1 else out

    # 2) Allgemeinfall (DQN etc.): erwartete Eingabedimension lesen.
    expected_in = _expected_in_features(policy_mod, agent)

    # Basis-Input aus obs_vec
    x = torch.tensor(np.asarray(obs_vec, dtype=np.float32), dtype=torch.float32, device=device)
    if x.ndim == 1:
        x = x.unsqueeze(0)  # [1, D]

    if expected_in is not None:
        cur = x.shape[1]
        if cur == expected_in:
            pass  # genau passend
        elif cur + num_players == expected_in:
            # Seat-One-Hot anhängen
            seat_oh = torch.zeros((1, num_players), dtype=torch.float32, device=device)
            seat_oh[0, seat_id] = 1.0
            x = torch.cat([x, seat_oh], dim=1)
        else:
            raise RuntimeError(
                f"[benchmark] Feature-Dim {cur} passt nicht zum Model-Input {expected_in} "
                f"(cur + num_players = {cur + num_players}). "
                "Bitte Seat-One-Hot/Feature-Konfig angleichen."
            )
    else:
        # Letzter Fallback, falls das Modell seine Eingabedim nicht preisgibt:
        # - Wenn feat_cfg.add_seat_onehot True ist → nichts tun (Seat-OH steckt schon drin)
        # - Sonst anhängen
        if not (feat_cfg is not None and getattr(feat_cfg, "add_seat_onehot", False)):
            seat_oh = torch.zeros((1, num_players), dtype=torch.float32, device=device)
            seat_oh[0, seat_id] = 1.0
            x = torch.cat([x, seat_oh], dim=1)

    out = policy_mod(x)
    return out.squeeze(0) if out.ndim == 2 and out.size(0) == 1 else out


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
                    # Beobachtung bauen und ggf. (je nach feat_cfg) augmentieren
                    obs = st.information_state_tensor(pid)
                    obs = augment_observation(obs, player_id=pid, cfg=feat_cfg)

                    logits = _policy_logits_for_benchmark(
                        agent, obs_vec=obs, seat_id=0, num_players=num_players, feat_cfg=feat_cfg
                    )

                    # Maske der legalen Aktionen
                    mask = torch.zeros(num_actions, dtype=torch.float32, device=logits.device)
                    mask[legal] = 1.0

                    # PPO erkennen (hat typischerweise _policy / _make_input)
                    is_ppo = hasattr(agent, "_policy") or hasattr(agent, "_make_input")
                    if is_ppo:
                        # Sampling aus maskierter Policy
                        probs = ppo.masked_softmax(logits, mask)
                        a = int(torch.distributions.Categorical(probs=probs).sample().item())
                    else:
                        # DQN: Greedy-Argmax nur über legale Actions
                        neg = torch.finfo(logits.dtype).min / 2
                        masked_q = torch.where(mask > 0, logits, neg)
                        a = int(torch.argmax(masked_q).item())
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
