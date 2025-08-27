# utils/plotter.py
import os, csv, math, time
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino Linotype", "Book Antiqua", "Palatino", "DejaVu Serif"]
})


class MetricsPlotter:
    """
    Ein Plotter für zwei Kanäle:
      - Benchmark: Winrates, Ø-Reward
      - Train:     Trainingsmetriken (loss, entropy, ...)

    CSVs (wide):
      - benchmark_csv: episode, [<name>_wr,<name>_reward,<name>_p0.._p3]*, macro_wr, macro_reward
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
        # Platzierungsverteilung je Gegner: Liste von (p0,p1,p2,p3) je Episode
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
            "places": [p0,p1,p2,p3],      # Anteile (sum ~ 1.0)
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

        # --- Header sicherstellen (neu oder fehlend -> ergänzen, ohne Daten zu verlieren) ---
        if self._bench_save_csv:
            header_fields = ["episode"]
            for name in self.bench_names:
                header_fields += [
                    f"{name}_wr",
                    f"{name}_reward",
                    f"{name}_p0",
                    f"{name}_p1",
                    f"{name}_p2",
                    f"{name}_p3",
                ]
            header_fields += ["macro_wr", "macro_reward"]
            header_line = ",".join(header_fields) + "\n"

            need_header = False
            if not os.path.exists(self.bench_csv_path) or os.path.getsize(self.bench_csv_path) == 0:
                need_header = True
            else:
                with open(self.bench_csv_path, "r", encoding="utf-8") as f:
                    first = f.readline()
                    if not first.strip().startswith("episode"):
                        # Datei hat keinen Header -> wir fügen ihn vorne an
                        rest = f.read()
                        with open(self.bench_csv_path, "w", newline="", encoding="utf-8") as fw:
                            fw.write(header_line)
                            fw.write(rest)
                    # falls first bereits mit "episode" beginnt: nichts tun

            if need_header:
                with open(self.bench_csv_path, "w", newline="", encoding="utf-8") as f:
                    f.write(header_line)

        # --- Werte einsammeln ---
        self.bench_episodes.append(int(episode))

        row = [episode]
        wr_for_macro, rw_for_macro = [], []
        for name in self.bench_names:
            val = results.get(name, float("nan"))

            # Legacy: nur Winrate als float
            if isinstance(val, (int, float)):
                wr = float(val)
                rw = float("nan")
                p0 = p1 = p2 = p3 = float("nan")
            else:
                wr = float(val.get("winrate", float("nan")))
                rw = float(val.get("reward", val.get("mean_reward", float("nan"))))
                place = val.get("places") or val.get("place_dist", [float("nan")] * 4)
                # robust extrahieren
                p0 = float(place[0]) if len(place) > 0 else float("nan")
                p1 = float(place[1]) if len(place) > 1 else float("nan")
                p2 = float(place[2]) if len(place) > 2 else float("nan")
                p3 = float(place[3]) if len(place) > 3 else float("nan")

            # Historie
            self.bench_hist_wr[name].append(wr)
            self.bench_hist_reward[name].append(rw)
            self.bench_hist_places[name].append((p0, p1, p2, p3))

            row += [wr, rw, p0, p1, p2, p3]
            if math.isfinite(wr):
                wr_for_macro.append(wr)
            if math.isfinite(rw):
                rw_for_macro.append(rw)

        # Macros
        macro_wr = float(np.mean(wr_for_macro)) if (macro_wr is None) else float(macro_wr)
        macro_reward = float(np.mean(rw_for_macro)) if (macro_reward is None) else float(macro_reward)
        self.bench_macro_wr.append(macro_wr)
        self.bench_macro_reward.append(macro_reward)
        row += [macro_wr, macro_reward]

        if self._bench_save_csv:
            with open(self.bench_csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

    def plot_benchmark(
        self,
        filename_prefix: str = "lernkurve",
        with_macro: bool = True,
        family_title: str | None = None,
        multi_title: str | None = None,
    ):
        """Winrate-Kurven mit konsistenten Titeln + spezifizierte Dateinamen."""
        if not self.bench_episodes:
            return

        # ---- Pretty-Namen für Gegner ----
        def _pretty_opp(name: str) -> str:
            mapping = {
                "max_combo": "Max Combo",
                "random2": "Random2",
                "single_only": "Single Only",
            }
            return mapping.get(name, name)

        # ---- Titel-Helper ----
        def _title_single(opp_pretty: str) -> str:
            return f"Lernkurve - {family_title} vs {opp_pretty}" if family_title else f"{filename_prefix}"

        def _title_multi() -> str:
            if multi_title:
                return multi_title
            if family_title:
                return f"Lernkurve - {family_title} vs feste Heuristiken"
            return f"{filename_prefix}"

        # ---- Farbpalette aus aktuellem Prop-Cycle ----
        base_colors = plt.rcParams.get("axes.prop_cycle", None)
        if base_colors is not None:
            base_colors = base_colors.by_key().get("color", [])
        if not base_colors:
            base_colors = [plt.cm.tab10(i) for i in range(10)]
        colors = {name: base_colors[i % len(base_colors)] for i, name in enumerate(self.bench_names)}

        # ---- Macro-Farbe fix: grau ----
        macro_color = "#444444"

        # ---- gemeinsame Achsen + optionale 25%-Linie (mit Legendeneintrag) ----
        def _apply_common_axes(title: str, add_25: bool = False):
            plt.title(title)
            plt.xlabel("Episode")
            plt.ylabel("Winrate (%)")
            plt.ylim(0, 100)
            plt.grid(True)

            ax = plt.gca()
            ax.set_xlim(left=0)

            ticks = ax.get_xticks()
            if 0.0 not in ticks:
                ax.set_xticks(np.unique(np.concatenate(([0.0], ticks))))

            if add_25:
                # dünne rote Referenzlinie + Label für Legende
                plt.axhline(25, color="r", linewidth=1, label="25% Linie")

        def _plot_single(name: str, out_path: str, add_25: bool = False, title_override: str | None = None):
            plt.figure(figsize=(10, 6))
            plt.plot(
                self.bench_episodes,
                self.bench_hist_wr[name],
                marker="o",
                markersize=4,
                color=colors[name],
            )
            title = title_override or _title_single(_pretty_opp(name))
            _apply_common_axes(title, add_25=add_25)
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()

        def _plot_multi(out_path: str, include_macro: bool, add_25: bool = False):
            plt.figure(figsize=(12, 8))
            for name in self.bench_names:
                plt.plot(
                    self.bench_episodes,
                    self.bench_hist_wr[name],
                    marker="o",
                    markersize=4,
                    label=name,
                    color=colors[name],
                )
            if include_macro:
                plt.plot(
                    self.bench_episodes,
                    self.bench_macro_wr,
                    marker="o",
                    markersize=4,
                    linestyle="--",
                    label="avg_macro",
                    color=macro_color,
                )
            _apply_common_axes(_title_multi(), add_25=add_25)
            plt.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()

        # ---- Ausgaben gemäß Vorgaben ----
        os.makedirs(self.out_dir, exist_ok=True)

        # 01: Heuristiken (alle Gegner, ohne Macro, ohne 25%-Linie)
        _plot_multi(
            out_path=os.path.join(self.out_dir, "01_Lernkurve_Heuristiken.png"),
            include_macro=False,
            add_25=False,
        )

        # 02: Heuristiken inkl. avg (mit Macro, ohne 25%-Linie)
        _plot_multi(
            out_path=os.path.join(self.out_dir, "02_Lernkurve_Heuristiken_incl_avg.png"),
            include_macro=True,
            add_25=False,
        )

        # 03: Heuristiken inkl. avg + 25%-Linie (rot, dünn) — Linie in Legende
        _plot_multi(
            out_path=os.path.join(self.out_dir, "03_Lernkurve_Heuristiken_incl_avg_25.png"),
            include_macro=True,
            add_25=True,
        )

        # 04: Heuristiken + 25%-Linie (ohne Macro) — Linie in Legende
        _plot_multi(
            out_path=os.path.join(self.out_dir, "04_Lernkurve_Heuristiken_incl_25.png"),
            include_macro=False,
            add_25=True,
        )

        # 05: Einzelplot Max Combo (Titel fix)
        if "max_combo" in self.bench_names:
            _plot_single(
                "max_combo",
                os.path.join(self.out_dir, "05_Lernkurve_max_combo.png"),
                add_25=False,
                title_override=_title_single(_pretty_opp("max_combo")),
            )

        # 06: Einzelplot Random2 (Titel fix)
        if "random2" in self.bench_names:
            _plot_single(
                "random2",
                os.path.join(self.out_dir, "06_Lernkurve_random2.png"),
                add_25=False,
                title_override=_title_single(_pretty_opp("random2")),
            )

        # 07: Einzelplot "single_only" → erster Gegner, Titel fix auf "Single Only"
        if self.bench_names:
            first_name = self.bench_names[0]
            _plot_single(
                first_name,
                os.path.join(self.out_dir, "07_Lernkurve_single_only.png"),
                add_25=False,
                title_override=_title_single(_pretty_opp("single_only")),
            )

    def plot_benchmark_rewards(self, filename_prefix: str = "benchmark_rewards", with_macro: bool = True, title_prefix: str | None = None):
        """Ø-Reward-Kurven aus dem Benchmark."""
        if not self.bench_episodes:
            return
        title = title_prefix or filename_prefix

        # Einzelplots pro Gegner (Reward)
        for name in self.bench_names:
            plt.figure(figsize=(10, 6))
            plt.plot(self.bench_episodes, self.bench_hist_reward[name], marker="o")
            plt.title(f"{title} – Ø-Reward vs {name}")
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
        plt.title(f"{title} – Ø-Reward vs Gegner")
        plt.xlabel("Episode")
        plt.ylabel("Ø-Reward (P0)")
        plt.grid(True)
        plt.legend(loc="upper right")
        plt.tight_layout()
        out_all = os.path.join(self.out_dir, f"{filename_prefix}_alle_strategien.png")
        plt.savefig(out_all)
        plt.close()

    def plot_places_latest(self, filename_prefix: str = "places"):
        """
        Schreibt für jede Gegnerstrategie die Platzierungsverteilung (p0..p3)
        der letzten Benchmark-Episode ANS ENDE EINER SAMMEL-CSV.

        Datei: <out_dir>/<filename_prefix>_latest.csv
        Spalten: opponent, episode, p0, p1, p2, p3  (Anteile, Summe ~ 1.0)
        - Header wird nur einmal geschrieben.
        - Bereits vorhandene (opponent, episode)-Kombinationen werden nicht doppelt angefügt.
        """
        if not self.bench_episodes:
            return

        last_idx = len(self.bench_episodes) - 1
        ep = int(self.bench_episodes[last_idx])

        # Zeilen aufbauen
        rows = []
        for name in self.bench_names:
            if self.bench_hist_places.get(name):
                p0, p1, p2, p3  = self.bench_hist_places[name][last_idx]
            else:
                p0 = p1 = p2 = p3 = float("nan")
            rows.append({
                "opponent": name,
                "episode": ep,
                "p0": float(p0),
                "p1": float(p1),
                "p2": float(p2),
                "p3": float(p3),
            })

        os.makedirs(self.out_dir, exist_ok=True)
        out_csv = os.path.join(self.out_dir, f"{filename_prefix}.csv")
        fieldnames = ["opponent", "episode", "p0", "p1", "p2", "p3"]

        file_exists = os.path.isfile(out_csv) and os.path.getsize(out_csv) > 0

        # existierende (opponent, episode) einlesen, um Duplikate zu vermeiden
        existing_keys = set()
        if file_exists:
            with open(out_csv, "r", newline="", encoding="utf-8") as f_in:
                try:
                    r = csv.DictReader(f_in)
                    for rec in r:
                        opp = rec.get("opponent")
                        try:
                            epi = int(rec.get("episode")) if rec.get("episode") is not None else None
                        except (TypeError, ValueError):
                            epi = None
                        if opp is not None and epi is not None:
                            existing_keys.add((opp, epi))
                except csv.Error:
                    # Falls die Datei korrupt ist, hängen wir trotzdem an.
                    pass

        # nur neue Zeilen behalten
        rows_to_write = [r for r in rows if (r["opponent"], r["episode"]) not in existing_keys]
        if not rows_to_write:
            return  # nichts Neues

        # anhängen; Header nur schreiben, wenn Datei neu/leer
        with open(out_csv, "a", newline="", encoding="utf-8") as f_out:
            w = csv.DictWriter(f_out, fieldnames=fieldnames)
            if not file_exists:
                w.writeheader()
            w.writerows(rows_to_write)


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
                plt.plot(episodes, vals,
                        marker="o", markersize=3,
                        linestyle="None")  # nur Punkte
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
                plt.plot(episodes, vals,
                        marker="o", markersize=3,
                        linestyle="None", label=k)
            plt.title("Training – Metrics")
            plt.xlabel("Episode")
            plt.ylabel("Value")
            plt.grid(True)
            plt.legend(loc="upper left")
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

class EvalPlotter:
    """
    Minimaler Plotter für k2a2/k3a2:
      - add(episode, {"opp": winrate_float, ...}) schreibt CSV (wide)
      - plot_all() erzeugt:
          lernkurve_<opp>.png
          lernkurve_alle_strategien.png
          lernkurve_alle_strategien_avg.png
    """
    def __init__(self, opponent_names, out_dir,
                 filename_prefix="Lernkurve",
                 csv_filename="eval_curves.csv",
                 save_csv=True):
        self.opps = list(opponent_names or [])
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.prefix = filename_prefix
        self.csv_path = os.path.join(self.out_dir, csv_filename)
        self.save_csv = bool(save_csv)
        self.episodes = []
        self.hist = {n: [] for n in self.opps}
        if self.save_csv and (not os.path.exists(self.csv_path)):
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["episode"] + self.opps)

    def add(self, episode: int, per_opponent: dict):
        self.episodes.append(int(episode))
        row = [episode]
        for name in self.opps:
            val = float(per_opponent.get(name, float("nan")))
            self.hist[name].append(val)
            row.append(val)
        if self.save_csv:
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row)

    def plot_all(self):
        if not self.episodes:
            return

        # Einzelplots
        for name in self.opps:
            ys = self.hist[name]
            plt.figure(figsize=(10,6))
            plt.plot(self.episodes, ys, marker="o")
            plt.title(f"{self.prefix} – Winrate vs {name}")
            plt.xlabel("Episode"); plt.ylabel("Winrate (%)"); plt.grid(True)
            plt.ylim(0, 100)
            plt.tight_layout()
            plt.savefig(os.path.join(self.out_dir, f"{self.prefix}_{name}.png"))
            plt.close()

        # Alle Strategien
        plt.figure(figsize=(12,8))
        for name in self.opps:
            plt.plot(self.episodes, self.hist[name], marker="o", label=name)
        plt.title(f"{self.prefix} – Winrate vs Gegner")
        plt.xlabel("Episode"); plt.ylabel("Winrate (%)"); plt.grid(True); plt.legend(loc="upper right")
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, f"{self.prefix}_alle_strategien.png"))
        plt.close()

        # + Macro Average (Dateiname endet auf *_avg.png für Kompatibilität mit deinen Aliasen)
        macro = []
        for t in range(len(self.episodes)):
            vals = [self.hist[name][t] for name in self.opps]
            macro.append(float(np.nanmean(vals)))
        plt.figure(figsize=(12,8))
        for name in self.opps:
            plt.plot(self.episodes, self.hist[name], marker="o", label=name)
        plt.plot(self.episodes, macro, marker="o", linestyle="--", label="macro_wr")
        plt.title(f"{self.prefix} – Winrate (mit Macro Average)")
        plt.xlabel("Episode"); plt.ylabel("Winrate (%)"); plt.grid(True); plt.legend(loc="upper right")
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, f"{self.prefix}_alle_strategien_avg.png"))
        plt.close()
