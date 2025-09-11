# utils/plotter.py
import os, csv, math, time
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino Linotype", "Book Antiqua", "Palatino", "DejaVu Serif"]
})


class MetricsPlotter:
    def __init__(
        self,
        out_dir: str,
        benchmark_opponents: Optional[List[str]] = None,
        benchmark_csv: str = "benchmark_curves.csv",
        train_csv: str = "train_metrics.csv",
        save_csv: bool = True,
        verbosity: int = 1,
        smooth_window: int | None = None,   # NEU
    ):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        # Logging
        self.verbosity = verbosity
        self.smooth_window = int(smooth_window) if smooth_window is not None else 150  # NEU


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
                # ---- Konsistente Farben für alle Return-Plots ----
        # Passe hier an, wenn du andere Farben willst.
        self._ret_colors = {
            "env_score":    "tab:blue",
            "shaping":      "tab:green",
            "final_bonus":  "tab:orange",
            # Total-Return (ep_return_raw):
            "total_scatter": "tab:gray",    # Punktewolke
            "total_ma":      "tab:red",     # Moving Average
        }

        # dezenter, einheitlicher Scatter-Ton für alle Komponenten
        self._ret_colors["component_scatter"] = "#B0B0B0"  # helles Grau
        # optionale Sichtbarkeits-Parameter
        self._ret_scatter_alpha = 0.18
        self._ret_scatter_size  = 6
        self._ret_line_width    = 2.2
        self._ret_use_outline   = True   # weiße Kontur für Kurven (bessere Sichtbarkeit)



        # --- In-Memory Episode-Returns (kein CSV, keine Pandas) ---
        # Pro globaler Trainingsepisode genau ein Return (z. B. P0), plus optionale Komponenten
        self.ep_ret_eps: List[int] = []            # global_ep
        self.ep_ret_vals: List[float] = []         # trainingsäquivalenter Return (genaues Buffersignal)
        self.ep_ret_components: Dict[str, List[float]] = {}  # z.B. {"env_score": [...], "shaping": [...], "final_bonus": [...]}


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


        def _plot_multi(out_path: str, include_macro: bool, add_25: bool = False):
            plt.figure(figsize=(12, 8))
            for name in self.bench_names:
                plt.plot(
                    self.bench_episodes,
                    self.bench_hist_wr[name],
                    marker="o",
                    markersize=4,
                    label=_pretty_opp(name),   # statt label=name
                    color=colors[name],
                )
            if include_macro:
                plt.plot(
                    self.bench_episodes,
                    self.bench_macro_wr,
                    marker="o",
                    markersize=4,
                    linestyle="--",
                    label="Avg Macro",        # schöner Name
                    color=macro_color,
                )
            _apply_common_axes(_title_multi(), add_25=add_25)
            plt.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()


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
        if not self._bench_save_csv:
            return

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

    def plot_train(
        self,
        filename_prefix: str = "training_metrics",
        separate: bool = True,
        include_keys: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        # Sonder-Keys, die nicht aus train_rows kommen, sondern aus Memory:
        special_keys = {
            "ep_return_raw", "ep_return_components",
            "ep_return_env", "ep_return_shaping", "ep_return_final"
        }

        wants_special = False
        if include_keys is not None:
            wants_special = any(k in special_keys for k in include_keys)

        # Nur abbrechen, wenn weder Trainingsmetriken noch Sonderplots gewünscht sind
        if (not self.train_rows or not self.train_keys) and not wants_special:
            return


        episodes = [r["episode"] for r in self.train_rows]
        # alle möglichen Keys (ohne 'episode')
        all_keys = [k for k in self.train_keys if k != "episode"]

        # Filter anwenden
        keys = all_keys
        if include_keys is not None:
            allow = set(include_keys)
            keys = [k for k in keys if k in allow]
        if exclude_keys is not None:
            deny = set(exclude_keys)
            keys = [k for k in keys if k not in deny]

                # --- Sonderplots über PLOT_KEYS steuern ---
        # Wenn "ep_return_raw" oder "ep_return_components" angefragt sind,
        # zeichnen wir sie über plot_ep_returns() und entfernen die Pseudo-Keys
        # aus der normalen Metrikenliste.
        # --- Sonderplots über PLOT_KEYS steuern ---
        if include_keys is not None:
            req = set(include_keys)

            # Gesamt + Komponenten-Mehrfachplot
            if ("ep_return_raw" in req) or ("ep_return_components" in req):
                self.plot_ep_returns(window=self.smooth_window)

            # Einzelplots je Komponente (Mapping Pseudo-Key -> Komponentenname)
            mapping = {
                "ep_return_env":     "env_score",
                "ep_return_shaping": "shaping",
                "ep_return_final":   "final_bonus",
            }
            for pseudo, comp in mapping.items():
                if pseudo in req:
                    self.plot_ep_return_component(comp, window=self.smooth_window)

            # Pseudo-Keys aus der normalen Metrikenliste entfernen
            keys = [k for k in keys if k not in {"ep_return_raw", "ep_return_components",
                                                 "ep_return_env", "ep_return_shaping", "ep_return_final"}]



        if not keys:
            return  # nichts zu plotten

        if separate:
            for k in keys:
                vals = [r.get(k, float("nan")) for r in self.train_rows]
                plt.figure(figsize=(10, 6))
                plt.plot(episodes, vals, marker="o", markersize=3, linestyle="None")
                plt.title(f"Training – {k}")
                plt.xlabel("Episode"); plt.ylabel(k)
                plt.grid(True); plt.tight_layout()
                plt.savefig(os.path.join(self.out_dir, f"{filename_prefix}_{k}.png"))
                plt.close()
        else:
            plt.figure(figsize=(12, 8))
            for k in keys:
                vals = [r.get(k, float("nan")) for r in self.train_rows]
                plt.plot(episodes, vals, marker="o", markersize=3, linestyle="None", label=k)
            plt.title("Training – Metrics")
            plt.xlabel("Episode"); plt.ylabel("Value")
            plt.grid(True); plt.legend(loc="upper left"); plt.tight_layout()
            plt.savefig(os.path.join(self.out_dir, f"{filename_prefix}_all.png"))
            plt.close()

    def add_ep_returns(
        self,
        global_episode: int,
        ep_returns: List[float],
        components: Optional[Dict[str, List[float]]] = None
    ):

        """
        Nimmt Episode-Return(s) entgegen und speichert sie NUR im Speicher.
        - Keine CSV, keine Pandas.
        - Üblicher Use-Case: genau 1 Return pro globaler Episode.
        """
        if not ep_returns:
            return

        comp_keys = list(components.keys()) if components else []
        for k in comp_keys:
            if k not in self.ep_ret_components:
                self.ep_ret_components[k] = []

        for i, r in enumerate(ep_returns):
            self.ep_ret_eps.append(int(global_episode))
            self.ep_ret_vals.append(float(r))
            for k in comp_keys:
                vals = components.get(k) or []
                self.ep_ret_components[k].append(float(vals[i]) if i < len(vals) else float("nan"))

    def plot_ep_returns(self, window: int | None = None):
        """
        Plottet die Episode-Returns aus dem IN-MEMORY-Puffer.
        - Erzeugt:
            - ep_return_raw.png (roh + Moving Avg, in 2 festen Farben)
            - ep_return_components.png (alle Komponenten gemeinsam)
            - ep_return_<komponente>.png (je Komponente einzeln)
        """
        if not self.ep_ret_eps or not self.ep_ret_vals:
            return
        window = self.smooth_window if (window is None) else int(window)



        x = np.asarray(self.ep_ret_eps, dtype=float)
        y = np.asarray(self.ep_ret_vals, dtype=float)

        def _movavg(v: np.ndarray, w: int) -> np.ndarray:
            if v.size == 0:
                return v
            w = max(1, int(w))
            c = np.cumsum(np.insert(v, 0, 0.0))
            tail = (c[w:] - c[:-w]) / float(w)                       # volle Fenster
            head = np.array([np.mean(v[:i+1]) for i in range(min(w-1, v.size))], dtype=float)
            return np.concatenate([head, tail])

        y_ma = _movavg(y, window)

        # --- ep_return_raw.png (2 feste Farben, konsistent) ---
        plt.figure(figsize=(12, 7))

        # Punkte in dezentem Grau
        plt.scatter(
            x, y,
            s=self._ret_scatter_size,
            alpha=self._ret_scatter_alpha,
            label="Rewards (roh)",
            color=self._ret_colors["total_scatter"],
            zorder=1,
        )

        # Moving Average in Kontrastfarbe (über den Punkten)
        line_ma, = plt.plot(
            x, y_ma,
            linewidth=self._ret_line_width,
            label=f"Moving Avg (w={window})",
            color=self._ret_colors["total_ma"],
            zorder=3,
        )

        # optional weiße Outline für bessere Lesbarkeit
        if self._ret_use_outline:

            line_ma.set_path_effects([pe.Stroke(linewidth=self._ret_line_width+1.4, foreground="white"), pe.Normal()])

        plt.title("Training – Rewards (roh)")
        plt.xlabel("Trainings-Episode (global)")
        plt.ylabel("Rewards")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "ep_return_raw.png"))
        plt.close()

        # --- ep_return_components.png (alle Komponenten gemeinsam) ---
        comp_keys = [k for k in ("env_score", "shaping", "final_bonus") if k in self.ep_ret_components]
        if comp_keys:
            plt.figure(figsize=(12, 7))
            for k in comp_keys:
                v = np.asarray(self.ep_ret_components[k], dtype=float)
                col = self._ret_colors.get(k, None)
                plt.plot(x, _movavg(v, window), label=k, color=col)
            plt.title("Training – Return-Komponenten (Moving Avg)")
            plt.xlabel("Trainings-Episode (global)")
            plt.ylabel("Rewards")

            plt.grid(True); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(self.out_dir, "ep_return_components.png"))
            plt.close()

            # --- Einzelplots pro Komponente (gleiche Farben) ---
            for k in comp_keys:
                v  = np.asarray(self.ep_ret_components[k], dtype=float)
                mv = _movavg(v, window)
                col_line = self._ret_colors.get(k, None)
                col_scatter = self._ret_colors["component_scatter"]

                plt.figure(figsize=(10, 6))
                # dezente Punkte
                plt.scatter(
                    x, v,
                    s=self._ret_scatter_size,
                    alpha=self._ret_scatter_alpha,
                    label="roh",
                    color=col_scatter,
                    zorder=1,
                )
                # farbige MA-Linie drüber
                line_comp, = plt.plot(
                    x, mv,
                    linewidth=self._ret_line_width,
                    label=f"{k} (MA)",
                    color=col_line,
                    zorder=3,
                )
                if self._ret_use_outline:
                    
                    line_comp.set_path_effects([pe.Stroke(linewidth=self._ret_line_width+1.2, foreground="white"), pe.Normal()])

                plt.title(f"Training – {k} (Moving Avg)")
                plt.xlabel("Trainings-Episode (global)")
                plt.ylabel("Rewards")

                plt.grid(True); plt.legend(); plt.tight_layout()
                plt.savefig(os.path.join(self.out_dir, f"ep_return_{k}.png"))
                plt.close()



    def plot_ep_return_component(self, component_key: str, window: int | None = None,
                                filename: Optional[str] = None, title: Optional[str] = None):
        """
        Einzelplot für eine Return-Komponente aus dem In-Memory-Puffer.
        component_key: "env_score" | "shaping" | "final_bonus"
        Erzeugt standardmäßig: ep_return_<component_key>.png
        """
        if not self.ep_ret_eps or component_key not in self.ep_ret_components:
            return

        

        x = np.asarray(self.ep_ret_eps, dtype=float)
        v = np.asarray(self.ep_ret_components[component_key], dtype=float)

        def _movavg(arr: np.ndarray, w: int) -> np.ndarray:
            if arr.size == 0:
                return arr
            w = max(1, int(w))
            w = min(w, arr.size)
            c = np.cumsum(np.insert(arr, 0, 0.0))
            tail = (c[w:] - c[:-w]) / float(w)
            head = np.array([np.mean(arr[:i+1]) for i in range(w-1)], dtype=float)
            return np.concatenate([head, tail])

        window = self.smooth_window if (window is None) else int(window)
        mv = _movavg(v, window)
        out_name = filename or f"ep_return_{component_key}.png"
        ttl = title or f"Training – {component_key} (Moving Avg)"
        col = self._ret_colors.get(component_key, None)

        plt.figure(figsize=(12, 7))

        # dezente, einheitliche Punkte
        plt.scatter(
            x, v,
            s=self._ret_scatter_size,
            alpha=self._ret_scatter_alpha,
            label="roh",
            color=self._ret_colors["component_scatter"],
            zorder=1,
        )

        # farbige MA-Linie
        line_comp, = plt.plot(
            x, mv,
            linewidth=self._ret_line_width,
            label=f"Moving Avg (w={window})",
            color=col,
            zorder=3,
        )
        if self._ret_use_outline:
            
            line_comp.set_path_effects([pe.Stroke(linewidth=self._ret_line_width+1.2, foreground="white"), pe.Normal()])

        plt.title(ttl)
        plt.xlabel("Trainings-Episode (global)")
        plt.ylabel("Rewards")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, out_name))
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

