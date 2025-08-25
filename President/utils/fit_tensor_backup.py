# utils/fit_tensor.py
from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np

@dataclass
class FeatureConfig:
    num_players: int
    num_ranks: int                     # kNumRanks aus dem Spiel
    add_seat_onehot: bool = False      # für K3=True, sonst False
    normalize: bool = True
    max_combo_size: int = 4            # President: Singles/Pairs/Trips/Quads
    empty_top_rank_value: float = -1.0 # wie in C++: -1 wenn kein Stapel

    def extra_dim(self) -> int:
        return self.num_players if self.add_seat_onehot else 0

def seat_one_hot(player_id: int, num_players: int) -> np.ndarray:
    v = np.zeros((num_players,), dtype=np.float32)
    v[player_id] = 1.0
    return v

def augment_observation(
    base_obs: Sequence[float],
    player_id: int,
    cfg: FeatureConfig
) -> np.ndarray:
    """
    Erwartete Base-Observation (aus C++), Layout:
      [0 : num_ranks)                         -> eigene Hand (counts je Rang)
      [num_ranks : num_ranks + (P-1))        -> Gegner-Kartensummen
      next                                    -> last_played_relative (0..P-1)
      next                                    -> current_combo_size (0..4, 0 wenn last_rel==0)
      next                                    -> top_rank (0..num_ranks-1 oder -1 wenn leer)

    Gibt ein np.float32 zurück, optional mit angehängter Sitz-One-Hot und Normalisierung.
    """
    x = np.asarray(base_obs, dtype=np.float32).copy()
    assert x.ndim == 1, "augment_observation erwartet einen 1D-Vektor"
    expected_len = cfg.num_ranks + (cfg.num_players - 1) + 3 + cfg.num_ranks
    if len(x) != expected_len:
        raise ValueError(
            f"Observation length mismatch: got {len(x)}, expected {expected_len} "
            f"(num_ranks={cfg.num_ranks}, num_players={cfg.num_players})"
        )

    if cfg.normalize:
        # Indizes berechnen
        i_hand_end = cfg.num_ranks
        i_opp_end  = i_hand_end + (cfg.num_players - 1)
        i_last_rel = i_opp_end
        i_combo    = i_last_rel + 1
        i_top      = i_last_rel + 2

        # last_rel in [0,1]
        if cfg.num_players > 1:
            x[i_last_rel] = x[i_last_rel] / float(cfg.num_players - 1)
        else:
            x[i_last_rel] = 0.0

        # combo_size in [0,1]
        if cfg.max_combo_size > 0:
            x[i_combo] = np.clip(x[i_combo] / float(cfg.max_combo_size), 0.0, 1.0)

        # top_rank in [0,1], -1 -> 0
        if x[i_top] <= cfg.empty_top_rank_value + 1e-6:
            x[i_top] = 0.0
        else:
            denom = max(1, cfg.num_ranks - 1)
            x[i_top] = np.clip(x[i_top] / float(denom), 0.0, 1.0)

        # (Optional) Handcounts und Gegnerkarten grob skalieren
        # -> durch max. Handgröße teilen (President: 52 / num_players)
        # max_hand = 52 // cfg.num_players
        # x[:i_hand_end] = x[:i_hand_end] / float(max_hand)
        # x[i_hand_end:i_opp_end] = x[i_hand_end:i_opp_end] / float(max_hand)

    if cfg.add_seat_onehot:
        x = np.concatenate([x, seat_one_hot(player_id, cfg.num_players)]).astype(np.float32, copy=False)

    return x

def augmented_dim(base_dim: int, cfg: FeatureConfig) -> int:
    """Gibt die neue Dimensionalität zurück, wenn `add_seat_onehot` aktiv ist."""
    return base_dim + cfg.extra_dim()

def expected_base_dim(cfg: FeatureConfig) -> int:
    """Dimensionalität der C++-Observation ohne Zusätze."""
    return cfg.num_ranks + (cfg.num_players - 1) + 3