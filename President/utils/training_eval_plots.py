# utils/training_eval_plots.py
import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

class EvalPlotter:
    def __init__(
        self,
        opponent_names: List[str],
        out_dir: str,
        filename_prefix: str = "lernkurve",
        csv_filename: str = "eval_curves.csv",
        save_csv: bool = True,
    ):
        self.opponent_names = list(opponent_names)
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.filename_prefix = filename_prefix
        self.csv_filename = os.path.join(self.out_dir, csv_filename)
        self.save_csv = bool(save_csv)

        self.episodes: List[int] = []
        self.history: Dict[str, List[float]] = {name: [] for name in self.opponent_names}
        self.macro: List[float] = []

        # Falls CSV noch nicht existiert, Header anlegen
        if self.save_csv and (not os.path.exists(self.csv_filename)):
            with open(self.csv_filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode"] + self.opponent_names + ["macro"])

    def add(self, episode: int, scores: Dict[str, float], macro: Optional[float] = None):
        """
        episode: int
        scores:  dict name->winrate (in %)
        macro:   optionaler Macro-Durchschnitt. Wenn None, wird er aus 'scores'
                 über self.opponent_names berechnet (naNs ignoriert).
        """
        self.episodes.append(int(episode))

        row_vals = []
        vals_for_macro = []
        for name in self.opponent_names:
            v = float(scores.get(name, float("nan")))
            self.history[name].append(v)
            row_vals.append(v)
            if math.isfinite(v):
                vals_for_macro.append(v)

        if macro is None:
            macro = float(np.mean(vals_for_macro)) if vals_for_macro else float("nan")
        self.macro.append(macro)

        if self.save_csv:
            with open(self.csv_filename, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([episode] + row_vals + [macro])

    def plot_all(self):
        if not self.episodes:
            return

        # Einzelplots pro Gegner
        for name in self.opponent_names:
            plt.figure(figsize=(10, 6))
            plt.plot(self.episodes, self.history[name], marker="o")
            plt.title(f"{self.filename_prefix} – Winrate vs {name}")
            plt.xlabel("Episode")
            plt.ylabel("Winrate (%)")
            plt.grid(True)
            plt.tight_layout()
            out = os.path.join(self.out_dir, f"{self.filename_prefix}_{name}.png")
            plt.savefig(out)
            plt.close()

        # Gemeinsamer Plot (alle Gegner)
        plt.figure(figsize=(12, 8))
        for name in self.opponent_names:
            plt.plot(self.episodes, self.history[name], marker="o", label=name)
        plt.title(f"{self.filename_prefix} – Winrate vs Gegner")
        plt.xlabel("Episode")
        plt.ylabel("Winrate (%)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        out_all = os.path.join(self.out_dir, f"{self.filename_prefix}_alle.png")
        plt.savefig(out_all)
        plt.close()

        # Gemeinsamer Plot + Macro Average
        plt.figure(figsize=(12, 8))
        for name in self.opponent_names:
            plt.plot(self.episodes, self.history[name], marker="o", label=name)
        plt.plot(self.episodes, self.macro, marker="o", linestyle="--", label="macro_avg")
        plt.title(f"{self.filename_prefix} – Winrate (mit Macro Average)")
        plt.xlabel("Episode")
        plt.ylabel("Winrate (%)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        out_macro = os.path.join(self.out_dir, f"{self.filename_prefix}_alle_mit_macro.png")
        plt.savefig(out_macro)
        plt.close()
