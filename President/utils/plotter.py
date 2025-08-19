# utils/plotter.py
import os, csv, math, time
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


class MetricsPlotter:
    """
    Ein Plotter für zwei Kanäle:
      - Benchmark: Winrates, Ø-Reward und Platzierungsverteilung vs. feste Gegner
      - Train:     Trainingsmetriken (loss, entropy, ...)

    CSVs (wide):
      - benchmark_csv: episode, [<name>_wr,<name>_reward,<name>_p1.._p4]*, macro_wr, macro_reward
      - train_csv    : episode, <metric1>, <metric2>, ...
    """

    # ---------- Konstruktor ----------
    def __init__(
        self,
        out_dir: str,
        benchmark_opponents: Optional[List[str]] = None,
        benchmark_csv: str = "benchmark_curves.csv",
        train_csv: str = "train_metrics.csv",
        save_csv: bool = True,
        verbosity: int = 1,
    ):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        # Logging
        self.verbosity = verbosity

        # Benchmark
        self.bench_names: List[str] = list(benchmark_opponents or [])
        self.bench_csv_path = os.path.join(self.out_dir, benchmark_csv)
        self._bench_save_csv = bool(save_csv)

        self.bench_episodes: List[int] = []
        # Historien getrennt nach Kennzahl
        self.bench_hist_wr: Dict[str, List[float]] = {n: [] for n in self.bench_names}
        self.bench_hist_reward: Dict[str, List[float]] = {n: [] for n in self.bench_names}
        # Platzierungsverteilung je Gegner: Liste von (p1,p2,p3,p4) je Episode
        self.bench_hist_places: Dict[str, List[Tuple[float, float, float, float]]] = {
            n: [] for n in self.bench_names
        }
        self.bench_macro_wr: List[float] = []
        self.bench_macro_reward: List[float] = []

        # Train
        self.train_csv_path = os.path.join(self.out_dir, train_csv)
        self._train_save_csv = bool(save_csv)
        self.train_keys: Optional[List[str]] = None
        self.train_rows: List[dict] = []

        # CSV-Header für Benchmark: dynamisch beim ersten add_benchmark, da Gegnerliste auch aus den Daten kommen kann
        # train_csv: Header dynamisch beim ersten add_train


        # ---------- Benchmark: add ----------
    def add_benchmark(
        self,
        episode: int,
        results: Dict[str, dict],
        macro_wr: Optional[float] = None,
        macro_reward: Optional[float] = None,
    ):
        """
        results-Format (neu):
        {
            "<opp>": {
            "winrate": float,             # in %
            "reward": float,              # mittlerer ENV-Reward von P0
            "places": [p1,p2,p3,p4],      # Anteile (sum ~ 1.0)
            "episodes": int               # Anzahl Episoden
            },
            ...
        }

        Abwärtskompatibel:
        - Wenn results[name] nur eine Zahl ist → als Winrate interpretiert.
        - Wenn reward fehlt → Fallback auf mean_reward.
        - Wenn places fehlt → Fallback auf place_dist.
        """
        # Falls bench_names noch leer sind: aus results ableiten
        if not self.bench_names:
            self.bench_names = list(results.keys())
            self.bench_hist_wr = {n: [] for n in self.bench_names}
            self.bench_hist_reward = {n: [] for n in self.bench_names}
            self.bench_hist_places = {n: [] for n in self.bench_names}
            # CSV-Header schreiben
            if self._bench_save_csv and (not os.path.exists(self.bench_csv_path)):
                header = ["episode"]
                for name in self.bench_names:
                    header += [
                        f"{name}_wr",
                        f"{name}_reward",
                        f"{name}_p1",
                        f"{name}_p2",
                        f"{name}_p3",
                        f"{name}_p4",
                    ]
                header += ["macro_wr", "macro_reward"]
                with open(self.bench_csv_path, "w", newline="") as f:
                    csv.writer(f).writerow(header)

        self.bench_episodes.append(int(episode))

        row = [episode]
        wr_for_macro, rw_for_macro = [], []
        for name in self.bench_names:
            val = results.get(name, float("nan"))

            # Legacy: nur Winrate als float
            if isinstance(val, (int, float)):
                wr = float(val)
                rw = float("nan")
                p1 = p2 = p3 = p4 = float("nan")
            else:
                wr = float(val.get("winrate", float("nan")))
                rw = float(val.get("reward", val.get("mean_reward", float("nan"))))
                place = val.get("places") or val.get("place_dist", [float("nan")] * 4)
                # Robust extrahieren
                p1 = float(place[0]) if len(place) > 0 else float("nan")
                p2 = float(place[1]) if len(place) > 1 else float("nan")
                p3 = float(place[2]) if len(place) > 2 else float("nan")
                p4 = float(place[3]) if len(place) > 3 else float("nan")

            # Historie
            self.bench_hist_wr[name].append(wr)
            self.bench_hist_reward[name].append(rw)
            self.bench_hist_places[name].append((p1, p2, p3, p4))

            row += [wr, rw, p1, p2, p3, p4]
            if math.isfinite(wr):
                wr_for_macro.append(wr)
            if math.isfinite(rw):
                rw_for_macro.append(rw)

        # Macros
        macro_wr = float(np.mean(wr_for_macro)) if (macro_wr is None) else float(macro_wr)
        macro_reward = (
            float(np.mean(rw_for_macro)) if (macro_reward is None) else float(macro_reward)
        )
        self.bench_macro_wr.append(macro_wr)
        self.bench_macro_reward.append(macro_reward)
        row += [macro_wr, macro_reward]

        if self._bench_save_csv:
            with open(self.bench_csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row)


    # ---------- Benchmark: plot ----------
    def plot_benchmark(self, filename_prefix: str = "lernkurve", with_macro: bool = True):
        """Winrate-Kurven (wie früher)."""
        if not self.bench_episodes:
            return

        # Einzelplots pro Gegner (Winrate)
        for name in self.bench_names:
            plt.figure(figsize=(10, 6))
            plt.plot(self.bench_episodes, self.bench_hist_wr[name], marker="o")
            plt.title(f"{filename_prefix} – Winrate vs {name}")
            plt.xlabel("Episode")
            plt.ylabel("Winrate (%)")
            plt.grid(True)
            plt.tight_layout()
            out = os.path.join(self.out_dir, f"{filename_prefix}_{name}.png")
            plt.savefig(out)
            plt.close()

        # Gemeinsamer Plot (alle Gegner, Winrate)
        plt.figure(figsize=(12, 8))
        for name in self.bench_names:
            plt.plot(self.bench_episodes, self.bench_hist_wr[name], marker="o", label=name)
        plt.title(f"{filename_prefix} – Winrate vs Gegner")
        plt.xlabel("Episode")
        plt.ylabel("Winrate (%)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        out_all = os.path.join(self.out_dir, f"{filename_prefix}_alle_strategien.png")
        plt.savefig(out_all)
        plt.close()

        # Gemeinsamer Plot + Macro Average (optional, Winrate)
        if with_macro:
            plt.figure(figsize=(12, 8))
            for name in self.bench_names:
                plt.plot(self.bench_episodes, self.bench_hist_wr[name], marker="o", label=name)
            plt.plot(self.bench_episodes, self.bench_macro_wr, marker="o", linestyle="--", label="macro_wr")
            plt.title(f"{filename_prefix} – Winrate (mit Macro Average)")
            plt.xlabel("Episode")
            plt.ylabel("Winrate (%)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            out_macro = os.path.join(self.out_dir, f"{filename_prefix}_alle_strategien_incl_macro.png")
            plt.savefig(out_macro)
            plt.close()

    def plot_benchmark_rewards(self, filename_prefix: str = "benchmark_rewards", with_macro: bool = True):
        """Ø-Reward-Kurven aus dem Benchmark."""
        if not self.bench_episodes:
            return

        # Einzelplots pro Gegner (Reward)
        for name in self.bench_names:
            plt.figure(figsize=(10, 6))
            plt.plot(self.bench_episodes, self.bench_hist_reward[name], marker="o")
            plt.title(f"{filename_prefix} – Ø-Reward vs {name}")
            plt.xlabel("Episode")
            plt.ylabel("Ø-Reward (P0)")
            plt.grid(True)
            plt.tight_layout()
            out = os.path.join(self.out_dir, f"{filename_prefix}_{name}.png")
            plt.savefig(out)
            plt.close()

        # Gemeinsamer Plot (alle Gegner, Reward)
        plt.figure(figsize=(12, 8))
        for name in self.bench_names:
            plt.plot(self.bench_episodes, self.bench_hist_reward[name], marker="o", label=name)
        if with_macro:
            plt.plot(self.bench_episodes, self.bench_macro_reward, marker="o", linestyle="--", label="macro_reward")
        plt.title(f"{filename_prefix} – Ø-Reward vs Gegner")
        plt.xlabel("Episode")
        plt.ylabel("Ø-Reward (P0)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        out_all = os.path.join(self.out_dir, f"{filename_prefix}_alle_strategien.png")
        plt.savefig(out_all)
        plt.close()

    def plot_places_latest(self, filename_prefix: str = "places"):
        """
        Zeichnet für jede Gegnerstrategie ein Balkendiagramm der Platzierungsverteilung
        für die *letzte* Benchmark-Episode (praktisch als Snapshot).
        """
        if not self.bench_episodes:
            return
        last_idx = len(self.bench_episodes) - 1
        ep = self.bench_episodes[last_idx]

        labels = ["1st", "2nd", "3rd", "4th"]
        for name in self.bench_names:
            p1, p2, p3, p4 = self.bench_hist_places[name][last_idx] if self.bench_hist_places[name] else (np.nan, np.nan, np.nan, np.nan)
            plt.figure(figsize=(6, 4))
            plt.bar(labels, [p1, p2, p3, p4])
            plt.title(f"Platzierungsverteilung vs {name} @Ep{ep}")
            plt.ylabel("Anteil")
            plt.ylim(0, 1)
            out = os.path.join(self.out_dir, f"{filename_prefix}_{name}_ep{ep}.png")
            plt.savefig(out)
            plt.close()

    # ---------- Train: add & plot ----------
    def add_train(self, episode: int, metrics: Dict[str, float]):
        if not metrics:
            return
        row = {"episode": int(episode), **{k: float(v) for k, v in metrics.items()}}
        self.train_rows.append(row)

        if self.train_keys is None:
            self.train_keys = ["episode"] + list(metrics.keys())
            if self._train_save_csv and (not os.path.exists(self.train_csv_path)):
                with open(self.train_csv_path, "w", newline="") as f:
                    csv.writer(f).writerow(self.train_keys)

        if self._train_save_csv:
            with open(self.train_csv_path, "a", newline="") as f:
                csv.writer(f).writerow([row.get(k, "") for k in self.train_keys])

    def plot_train(self, filename_prefix: str = "training_metrics", separate: bool = True):
        if not self.train_rows or not self.train_keys:
            return
        episodes = [r["episode"] for r in self.train_rows]
        metric_keys = [k for k in self.train_keys if k != "episode"]

        if separate:
            for k in metric_keys:
                vals = [r.get(k, float("nan")) for r in self.train_rows]
                plt.figure(figsize=(10, 6))
                plt.plot(episodes, vals, marker="o")
                plt.title(f"Training – {k}")
                plt.xlabel("Episode")
                plt.ylabel(k)
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(self.out_dir, f"{filename_prefix}_{k}.png"))
                plt.close()
        else:
            plt.figure(figsize=(12, 8))
            for k in metric_keys:
                vals = [r.get(k, float("nan")) for r in self.train_rows]
                plt.plot(episodes, vals, marker="o", label=k)
            plt.title("Training – Metrics")
            plt.xlabel("Episode")
            plt.ylabel("Value")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.out_dir, f"{filename_prefix}_all.png"))
            plt.close()

    # ---------- Logging (Konsole + run.log) ----------
    def log(self, msg: str, level: int = 1):
        if level > self.verbosity:
            return
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        try:
            with open(os.path.join(self.out_dir, "run.log"), "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def log_bench_summary(self, episode: int, per_opponent: Dict[str, dict]):
        wrs, rws = [], []
        bench_eps = None
        for val in per_opponent.values():
            if isinstance(val, (int, float)):
                wrs.append(float(val))
            else:
                wr = val.get("winrate")
                rw = val.get("reward", val.get("mean_reward"))
                if wr is not None: wrs.append(float(wr))
                if rw is not None: rws.append(float(rw))
                if bench_eps is None:
                    bench_eps = val.get("episodes")

        macro_wr = float(np.mean(wrs)) if wrs else float("nan")
        macro_reward = float(np.mean(rws)) if rws else float("nan")

        self.log("")
        if bench_eps is not None:
            self.log(f"Benchmark @train_ep {episode:7d} (bench_eps: {bench_eps})")
        else:
            self.log(f"Benchmark @train_ep {episode:7d}")

        self.log(f"{'Macro Winrate':<23s}: {macro_wr:5.1f}%")
        self.log(f"{'Macro Ø-Reward':<23s}: {macro_reward: .3f}")

        for name in sorted(per_opponent.keys()):
            val = per_opponent[name]
            if isinstance(val, (int, float)):
                self.log(f"Winrate vs {name:12s}: {float(val):5.1f}%")
                continue

            wr = val.get("winrate", float("nan"))
            rw_mean = val.get("reward", val.get("mean_reward", float("nan")))
            eps = val.get("episodes", 1)
            rw_sum = rw_mean * eps if np.isfinite(rw_mean) else float("nan")

            places = val.get("places") or val.get("place_dist", [float('nan')]*4)
            # absolute/relative sauber formatiert
            abs_places = [int(round(p * eps)) if np.isfinite(p) else 0 for p in places]
            place_str = "[" + ", ".join(f"{abs_places[i]} ({places[i]:.2f})" if np.isfinite(places[i]) else f"{abs_places[i]} (nan)"
                                        for i in range(len(abs_places))) + "]"

            self.log(
                f"Winrate vs {name:12s}: {wr:5.1f}% | "
                f"Σ-Reward: {rw_sum: .1f} | "
                f"Ø-Reward: {rw_mean: .3f} | "
                f"Places: {place_str}"
            )




    def log_timing(
        self,
        ep: int,
        ep_seconds: float,
        train_seconds: float,
        eval_seconds: float,
        plot_seconds: float,
        save_seconds: float,
        cum_seconds: float,
    ):
        self.log(
            f"Timing @ep {ep}: episode {ep_seconds:0.1f}s | "
            f"train {train_seconds:0.3f}s | eval {eval_seconds:0.1f}s | "
            f"plot {plot_seconds:0.1f}s | save {save_seconds:0.3f}s | "
            f"cum {cum_seconds/3600:0.2f}h"
        )
