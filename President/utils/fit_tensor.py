# utils/fit_tensor.py
from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass
class FeatureConfig:
    num_players: int
    num_ranks: int                      # kNumRanks aus dem Spiel
    add_seat_onehot: bool = False       # optional: Sitz-One-Hot anhängen
    include_history: bool = True        # True = mit Historie (Variante 2), False = ohne (Variante 1)
    normalize: bool = False
    deck_size: int = 0
    max_combo_size: int = 4             # President: Singles/Pairs/Trips/Quads
    empty_top_rank_value: float = -1.0  # wie in C++: -1 wenn kein Stapel

    def __post_init__(self) -> None:
        if self.num_players <= 0:
            raise ValueError("num_players must be > 0")
        if self.num_ranks <= 0:
            raise ValueError("num_ranks must be > 0")
        if self.max_combo_size < 0:
            raise ValueError("max_combo_size must be >= 0")

    # -------- Dimensionen (einzige Quelle der Wahrheit) -------- #
    def extra_dim(self) -> int:
        return self.num_players if self.add_seat_onehot else 0

    def core_dim(self) -> int:
        """
        Länge der vom Agenten genutzten Features OHNE Sitz-One-Hot.
        (Variante 1/2 über include_history)
        """
        base = self.num_ranks + (self.num_players - 1) + 3
        if self.include_history:
            base += self.num_ranks
        return base

    def input_dim(self) -> int:
        """Länge des Agent-Inputs INKLUSIVE optionaler Sitz-One-Hot."""
        return self.core_dim() + self.extra_dim()

    def env_obs_len(self) -> int:
        """
        Länge des vom C++-Spiel gelieferten Vektors (immer Variante 2 = mit History):
          [Hand k], [Opp (P-1)], [last_rel, combo, top], [History k]
        """
        return self.num_ranks + (self.num_players - 1) + 3 + self.num_ranks


def seat_one_hot(player_id: int, num_players: int) -> np.ndarray:
    v = np.zeros((num_players,), dtype=np.float32)
    if not (0 <= player_id < num_players):
        raise ValueError(f"player_id {player_id} out of range [0,{num_players-1}]")
    v[player_id] = 1.0
    return v


# ----------------- Augmentierung ----------------- #
def augment_observation(
    base_obs: Sequence[float],
    player_id: int,
    cfg: FeatureConfig
) -> np.ndarray:
    """
    Erwartete Base-Observation (aus C++, immer Variante 2 = inkl. Historie):

      Indizes im vollen Vektor:
        0 .. k-1                          -> eigene Hand (counts je Rang)
        k .. k+(P-1)-1                    -> Gegner-Gesamtkarten je Spieler
        next (k+(P-1))                    -> last_played_relative (0..P-1)
        next                               -> current_combo_size (0..4; 0 wenn last_rel==0)
        next                               -> top_rank (0..k-1 oder -1 wenn leer)
        rest: k Werte                      -> Karten-Historie (pro Rang: bereits ausgespielt)

    Option cfg.include_history schneidet die Historie ab (Variante 1) oder behält sie (Variante 2).
    """
    x_full = np.asarray(base_obs, dtype=np.float32)
    exp_len = cfg.env_obs_len()
    if x_full.ndim != 1 or len(x_full) != exp_len:
        raise ValueError(
            f"Observation length mismatch: got {len(x_full)}, expected {exp_len} "
            f"(num_ranks={cfg.num_ranks}, num_players={cfg.num_players})"
        )

    # Indizes (basieren auf Variante 2 = voller Vektor)
    i_hand_end = cfg.num_ranks
    i_opp_end  = i_hand_end + (cfg.num_players - 1)
    i_last_rel = i_opp_end
    i_combo    = i_last_rel + 1
    i_top      = i_last_rel + 2
    i_hist_beg = i_top + 1
    i_hist_end = i_hist_beg + cfg.num_ranks  # exklusiv

    # Kopf bis inkl. top_rank
    head = x_full[:i_hist_beg]

    # ggf. Historie anfügen
    if cfg.include_history:
        core = np.concatenate([head, x_full[i_hist_beg:i_hist_end]], axis=0)
    else:
        core = head.copy()  # head nicht aliasen

    # Normalisierung nur auf die Kopf-Features (Indices bleiben gültig)
    if cfg.normalize:
        # last_rel in [0,1]
        if cfg.num_players > 1:
            core[i_last_rel] = core[i_last_rel] / float(cfg.num_players - 1)
        else:
            core[i_last_rel] = 0.0

        # combo_size in [0,1]
        if cfg.max_combo_size > 0:
            core[i_combo] = np.clip(core[i_combo] / float(cfg.max_combo_size), 0.0, 1.0)

        # top_rank in [0,1], -1 -> 0
        if core[i_top] <= cfg.empty_top_rank_value + 1e-6:
            core[i_top] = 0.0
        else:
            denom = max(1, cfg.num_ranks - 1)
            core[i_top] = np.clip(core[i_top] / float(denom), 0.0, 1.0)

        # Hand- & Gegneranzahlen skalieren (0..1)
        if cfg.deck_size > 0:
            max_hand = max(1, cfg.deck_size // cfg.num_players)
            core[:i_hand_end] = core[:i_hand_end] / float(max_hand)
            core[i_hand_end:i_opp_end] = core[i_hand_end:i_opp_end] / float(max_hand)

    # optional Sitz-One-Hot anhängen
    if cfg.add_seat_onehot:
        core = np.concatenate([core, seat_one_hot(player_id, cfg.num_players)], axis=0)

    return core.astype(np.float32, copy=False)
