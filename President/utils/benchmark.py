# utils/benchmark.py
import numpy as np
import torch
import pyspiel

from agents import ppo_agent as ppo  # für masked_softmax
from utils.fit_tensor import FeatureConfig, augment_observation
from utils.deck import ranks_for_deck


# -------------------- Hilfen: Policy-Modul & Input-Dimension --------------------
def _get_policy_module(agent):
    if hasattr(agent, "_policy"):
        return agent._policy
    for name in ("q_network", "q", "q_net"):
        if hasattr(agent, name):
            return getattr(agent, name)
    if isinstance(agent, torch.nn.Module):
        return agent
    raise AttributeError("Kein kompatibles Policy-/Q-Modul am Agent (_policy/q_network/q/q_net).")

def _expected_in_features(mod, agent) -> int | None:
    net = getattr(mod, "net", None)
    if hasattr(net, "__getitem__") and hasattr(net[0], "in_features"):
        return int(net[0].in_features)
    if isinstance(mod, torch.nn.Sequential) and hasattr(mod[0], "in_features"):
        return int(mod[0].in_features)
    for name in ("encoder", "feature", "backbone", "fc_in", "in"):
        layer = getattr(mod, name, None)
        if hasattr(layer, "in_features"):
            return int(layer.in_features)
    for name in ("state_size", "obs_dim", "input_dim"):
        if hasattr(agent, name):
            try:
                return int(getattr(agent, name))
            except Exception:
                pass
    return None


# -------------------- Einheitliche Feature-Brücke (DQN & PPO) --------------------
def _mk_obs_for_agent(game, state, player, agent, feat_cfg: FeatureConfig | None):
    """Baut den Input-Vektor so, dass er zur erwarteten Model-Input-Dimension passt.
       - PPO: nutzt _make_input (falls vorhanden) auf Basis des *Info*-Vektors.
       - DQN/sonstige: entscheidet RAW vs BASE(+History) und optional Seat-One-Hot
                       per Inputdimension, analog zu eval_macro._mk_obs_for_dqn.
    """
    num_players = game.num_players()
    deck_int = int(game.get_parameters()["deck_size"])
    num_ranks = ranks_for_deck(deck_int)

    # PPO-Spezialweg: Delegation an _make_input falls vorhanden (verlässlichste Quelle)
    if hasattr(agent, "_make_input"):
        info = np.asarray(state.information_state_tensor(player), dtype=np.float32)
        seat_oh = np.zeros(num_players, dtype=np.float32); seat_oh[player] = 1.0
        x = agent._make_input(info, seat_one_hot=seat_oh)  # PPO kennt seine Basislänge selbst
        return np.asarray(x, dtype=np.float32)

    # DQN/sonstige: wir arbeiten mit OBS (wie im Training für K1–K4 vorgesehen)
    obs_raw = np.asarray(state.observation_tensor(player), dtype=np.float32)

    # Default feat_cfg, falls None (History aus, Seat-OH extern bei Bedarf)
    if feat_cfg is None:
        feat_cfg = FeatureConfig(
            num_players=num_players,
            num_ranks=num_ranks,
            add_seat_onehot=False,
            include_history=False
        )

    base_aug = augment_observation(obs_raw, player_id=player, cfg=feat_cfg)
    base_len = base_aug.shape[0]

    # erwartete Inputdimension lesen
    policy_mod = _get_policy_module(agent)
    expected_in = _expected_in_features(policy_mod, agent)

    def add_seat(vec: np.ndarray) -> np.ndarray:
        seat_oh = np.zeros(num_players, dtype=np.float32)
        seat_oh[player] = 1.0
        return np.concatenate([vec, seat_oh], axis=0)

    # Entscheidung wie in eval_macro:
    if expected_in is None:
        # Fallback: Wenn feat_cfg.add_seat_onehot True → Seat steckt schon drin; sonst anhängen.
        x = base_aug  # bevorzugt BASE, weil augment_observation deck-/spielerzahlstabil ist
        if not getattr(feat_cfg, "add_seat_onehot", False):
            x = add_seat(x)
        return x.astype(np.float32, copy=False)

    # 1) passt exakt zu BASE(+History) ohne Seat
    if base_len == expected_in:
        return base_aug.astype(np.float32, copy=False)
    # 2) BASE + Seat-One-Hot
    if base_len + num_players == expected_in:
        return add_seat(base_aug).astype(np.float32, copy=False)
    # 3) RAW ohne Seat
    if obs_raw.shape[0] == expected_in:
        return obs_raw.astype(np.float32, copy=False)
    # 4) RAW + Seat-One-Hot
    if obs_raw.shape[0] + num_players == expected_in:
        return add_seat(obs_raw).astype(np.float32, copy=False)

    # 5) Notnagel: Pad/Trunc (laut Log sichtbar machen!)
    x = obs_raw
    if x.shape[0] < expected_in:
        x = np.concatenate([x, np.zeros(expected_in - x.shape[0], dtype=np.float32)], axis=0)
    else:
        x = x[:expected_in]
    return x.astype(np.float32, copy=False)


@torch.no_grad()
def _policy_logits(agent, x_vec, *, seat_id: int, num_players: int, feat_cfg: FeatureConfig | None):
    """Vorwärtsdurchlauf, ohne Sampling/Greedy. PPO nutzt ggf. schon _make_input in _mk_obs_for_agent."""
    policy_mod = _get_policy_module(agent)
    try:
        device = next(policy_mod.parameters()).device
    except Exception:
        device = torch.device("cpu")
    x = torch.tensor(np.asarray(x_vec, dtype=np.float32), dtype=torch.float32, device=device)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    out = policy_mod(x)
    return out.squeeze(0) if out.ndim == 2 and out.size(0) == 1 else out


# -------------------- Hauptfunktion --------------------
def run_benchmark(game, agent, opponents_dict, opponent_names, episodes, feat_cfg, num_actions):
    """
    Simuliert 'episodes' Spiele pro Gegner.
    - Nutzt dieselbe Feature-Logik wie eval_macro (RAW/BASE(+History), optional Seat-OH).
    - PPO: Sampling aus maskierter Policy; DQN: masked greedy argmax (legal only).
    """
    results = {}
    num_players = game.num_players()

    for name in opponent_names:
        opp_fn = opponents_dict[name]
        wins = 0
        rewards = []
        place_counts = [0, 0, 0, 0]

        # Einmaliges Logging zur Sicherheit (erste Episode)
        did_log_shape = False

        for ep in range(episodes):
            st = game.new_initial_state()
            while not st.is_terminal():
                pid = st.current_player()
                legal = st.legal_actions(pid)

                if pid == 0:
                    x_vec = _mk_obs_for_agent(game, st, pid, agent, feat_cfg)

                    logits = _policy_logits(
                        agent, x_vec, seat_id=0, num_players=num_players, feat_cfg=feat_cfg
                    )

                    # Maske der legalen Aktionen
                    mask = torch.zeros(num_actions, dtype=torch.float32, device=logits.device)
                    if len(legal) > 0:
                        mask[legal] = 1.0

                    # PPO erkennen (hat typischerweise _policy / _make_input)
                    is_ppo = hasattr(agent, "_policy") or hasattr(agent, "_make_input")
                    if is_ppo:
                        probs = ppo.masked_softmax(logits, mask)
                        a = int(torch.distributions.Categorical(probs=probs).sample().item())
                    else:
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
            "wins": int(wins),
            "reward_std": float(np.std(rewards, ddof=1)) if episodes > 1 else 0.0,
            "place_counts": [int(c) for c in place_counts],
        }

    return results
