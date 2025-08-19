class RewardShaper:
    def __init__(self, cfg):
        # Normalisiere evtl. "final_reward" -> "placement_bonus"
        final = "placement_bonus" if cfg["FINAL"] in ("final_reward", "placement_bonus") else cfg["FINAL"]
        self.step, self.final, self.env = cfg["STEP"], final, bool(cfg["ENV_REWARD"])
        self.dw  = float(cfg["DELTA_WEIGHT"])
        self.hp  = float(cfg["HAND_PENALTY_COEFF"])
        self.b   = (
            float(cfg["BONUS_WIN"]),
            float(cfg["BONUS_2ND"]),
            float(cfg["BONUS_3RD"]),
            float(cfg["BONUS_LAST"]),
        )

    @staticmethod
    def _ranks(total_cards: int) -> int:
        """
        Mappe Gesamtanzahl Karten -> Anzahl Ränge.
        Muss konsistent zu president.cc sein:

          12 -> 3   (Q,K,A)           mit 4 Farben
          16 -> 4   (J,Q,K,A)         mit 4 Farben
          20 -> 5   (10,J,Q,K,A)      mit 4 Farben
          24 -> 6   (9,10,J,Q,K,A)    mit 4 Farben
          32 -> 8   (7..A)            mit 4 Farben
          52 -> 13  (2..A)            mit 4 Farben
          64 -> 8   (7..A)            mit 8 Farben
        """
        mapping = {
            12: 3,
            16: 4,
            20: 5,
            24: 6,
            32: 8,
            52: 13,
            64: 8,
        }
        try:
            return mapping[int(total_cards)]
        except (KeyError, ValueError):
            raise ValueError(f"Unsupported deck size: {total_cards} (expected one of {sorted(mapping.keys())})")

    def hand_size(self, ts, pid, total_cards: int) -> int:
        nr = self._ranks(total_cards)
        # Annahme: In der Info-State-Tensor liegt zuerst ein Block mit Handcounts pro Rang (Länge = nr)
        return int(sum(ts.observations["info_state"][pid][:nr]))

    def step_reward(self, **kw):
        if self.step == "none":
            return 0.0
        if self.step == "delta_hand":
            return self.dw * max(0.0, float(kw["hand_before"] - kw["hand_after"]))
        if self.step == "hand_penalty":
            return -self.hp * float(self.hand_size(kw["time_step"], kw["player_id"], kw["deck_size"]))
        raise ValueError(self.step)

    def final_bonus(self, finals, pid):
        if self.final == "none":
            return 0.0
        order = sorted(range(len(finals)), key=lambda p: finals[p], reverse=True)
        place = order.index(pid) + 1
        return (self.b[0], self.b[1], self.b[2], self.b[3])[place - 1]

    def include_env_reward(self):
        return self.env
