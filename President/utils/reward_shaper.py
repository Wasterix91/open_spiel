class RewardShaper:
    """
    Neues Reward-System (ohne Abwärtskompatibilität).

    Erwartete CFG-Keys:
      STEP_MODE  : "none" | "delta_weight_only" | "hand_penalty_coeff_only" | "combined"
      DELTA_WEIGHT : float
      HAND_PENALTY_COEFF : float

      FINAL_MODE : "none" | "env_only" | "rank_only" | "both"
      BONUS_WIN, BONUS_2ND, BONUS_3RD, BONUS_LAST : float

    Nutzung:
      - Step-Reward pro Zug:
            hb = shaper.hand_size(ts_before, p, deck_int)
            ts_after = env.step([a])
            ha = shaper.hand_size(ts_after, p, deck_int)
            r_step = shaper.step_reward(hand_before=hb, hand_after=ha)
            agent.post_step(r_step, done=ts_after.last())

      - Am Episodenende:
            if shaper.include_env_reward():
                buffer[last_of_p].reward += env_returns[p]
            buffer[last_of_p].reward += shaper.final_bonus(env_returns, p)
            buffer[last_of_p].done = True
    """
    # ---- erlaubte Modi ----
    _STEP_CHOICES  = {"none", "delta_weight_only", "hand_penalty_coeff_only", "combined"}
    _FINAL_CHOICES = {"none", "env_only", "rank_only", "both"}

    def __init__(self, cfg: dict):
        # STEP
        self.step_mode: str = str(cfg["STEP_MODE"])
        if self.step_mode not in self._STEP_CHOICES:
            raise ValueError(f"Invalid STEP_MODE={self.step_mode!r}. "
                             f"Expected one of {sorted(self._STEP_CHOICES)}.")
        self.dw: float = float(cfg["DELTA_WEIGHT"])
        self.hp: float = float(cfg["HAND_PENALTY_COEFF"])

        # FINAL
        self.final_mode: str = str(cfg["FINAL_MODE"])
        if self.final_mode not in self._FINAL_CHOICES:
            raise ValueError(f"Invalid FINAL_MODE={self.final_mode!r}. "
                             f"Expected one of {sorted(self._FINAL_CHOICES)}.")
        self.bonus = (
            float(cfg["BONUS_WIN"]),
            float(cfg["BONUS_2ND"]),
            float(cfg["BONUS_3RD"]),
            float(cfg["BONUS_LAST"]),
        )

    # ---------- Hilfen ----------
    @staticmethod
    def _ranks(total_cards: int) -> int:
        mapping = {12: 3, 16: 4, 20: 5, 24: 6, 32: 8, 52: 13, 64: 8}
        try:
            return mapping[int(total_cards)]
        except (KeyError, ValueError):
            raise ValueError(f"Unsupported deck size: {total_cards} "
                             f"(expected one of {sorted(mapping.keys())})")

    def hand_size(self, ts, pid: int, total_cards: int) -> int:
        nr = self._ranks(total_cards)
        return int(sum(ts.observations["info_state"][pid][:nr]))

    # ---------- STEP ----------
    def step_active(self) -> bool:
        return self.step_mode != "none"

    def step_reward(self, *, hand_before: int, hand_after: int) -> float:
        mode = self.step_mode
        if mode == "none":
            return 0.0

        r = 0.0
        if mode in ("delta_weight_only", "combined"):
            delta_cards = max(0.0, float(hand_before - hand_after))  # 0 bei Pass
            # >>> Quadratische Belohnung für Kombos
            r += self.dw * (delta_cards ** 2)

            # >>> Alternative (optional): Dreiecksbelohnung für Kombos
            # tri = delta_cards * (delta_cards + 1.0) / 2.0
            # r += self.dw * tri
            # Hinweis: Falls du die Dreiecksformel nutzt, kommentiere die
            # quadratische Zeile oben aus, damit nicht doppelt addiert wird.

        if mode in ("hand_penalty_coeff_only", "combined"):
            r += -self.hp * float(hand_after)

        return r


    # ---------- FINAL ----------
    def include_env_reward(self) -> bool:
        """Ob ENV-Return am Episodenende addiert werden soll."""
        return self.final_mode in ("env_only", "both")

    def final_bonus(self, finals, pid: int) -> float:
        """Benutzerdefinierter Platzierungsbonus (nur bei rank_only/both)."""
        if self.final_mode not in ("rank_only", "both"):
            return 0.0
        order = sorted(range(len(finals)), key=lambda p: finals[p], reverse=True)
        place = order.index(pid) + 1  # 1..N
        return (self.bonus[0], self.bonus[1], self.bonus[2], self.bonus[3])[place - 1]
