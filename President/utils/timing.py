# utils/timing.py
from dataclasses import dataclass
import time, os, csv

COLUMNS = [
    "episode",
    "steps",          # = ep_length
    "ep_seconds",
    "train_seconds",
    "eval_seconds",
    "plot_seconds",
    "save_seconds",
    "cum_seconds",
]

@dataclass
class TimingMeter:
    csv_path: str
    interval: int = 500

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        if not os.path.exists(self.csv_path):
            # nur Header schreiben (keine Dummy-Zeile)
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(COLUMNS)
        self.t0 = time.perf_counter()

    def maybe_log(self, ep: int, fields: dict):
        if ep % self.interval != 0:
            return
        now = time.perf_counter()
        cum = now - self.t0

        # Feste Spaltenreihenfolge erzwingen, fehlende Felder als 0.0/0
        row = [
            ep,
            fields.get("steps", 0),
            float(fields.get("ep_seconds", 0.0)),
            float(fields.get("train_seconds", 0.0)),
            float(fields.get("eval_seconds", 0.0)),
            float(fields.get("plot_seconds", 0.0)),
            float(fields.get("save_seconds", 0.0)),
            float(cum),
        ]

        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
