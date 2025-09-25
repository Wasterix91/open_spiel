# utils/plotter.py
import os, csv, math, time
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors 
import colorsys


# 0% Rand links/rechts auf allen Achsen
mpl.rcParams["axes.xmargin"] = 0.0

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino Linotype", "Book Antiqua", "Palatino", "DejaVu Serif"],
    "pdf.fonttype": 42,     # Text in PDFs bleibt editierbarer Text
    "ps.fonttype": 42,
    "svg.fonttype": "none", # Text in SVG bleibt Text (nicht in Pfade konvertieren)
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
        smooth_window: int | None = None,
        out_formats: Optional[List[str]] = None,   # <--- NEU
    ):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)


        self._out_formats = list(out_formats or ["png"])  # <--- NEU
        # Logging
        self.verbosity = verbosity
        self.smooth_window = int(smooth_window) if smooth_window is not None else 150

        # Benchmark
        self.bench_names: List[str] = list(benchmark_opponents or [])
        self.bench_csv_path = os.path.join(self.out_dir, benchmark_csv)
        self._bench_save_csv = bool(save_csv)

        self.bench_episodes: List[int] = []
        self.bench_hist_wr: Dict[str, List[float]] = {n: [] for n in self.bench_names}
        self.bench_hist_reward: Dict[str, List[float]] = {n: [] for n in self.bench_names}
        self.bench_hist_places: Dict[str, List[Tuple[float, float, float, float]]] = {
            n: [] for n in self.bench_names
        }
        self.bench_hist_n: Dict[str, List[int]] = {n: [] for n in self.bench_names}
        self.bench_hist_k: Dict[str, List[int]] = {n: [] for n in self.bench_names}
        self.bench_macro_wr: List[float] = []
        self.bench_macro_reward: List[float] = []

        # Train
        self.train_csv_path = os.path.join(self.out_dir, train_csv)
        self._train_save_csv = bool(save_csv)
        self.train_keys: Optional[List[str]] = None
        self.train_rows: List[dict] = []

        # ---- Konsistente Farben ----
        self._ret_colors = {
            # Totals / Linienfarben
            "total_scatter": "tab:gray",
            "total_ma":      "tab:red",
            "total_all":     "#222222",

            # Final-Gruppe (Blautöne)
            "final_total":   "#1f77b4",
            "final_env":     "#6baed6",
            "final_rank":    "#08519c",

            # Step-Gruppe (Grüntöne)
            "step_total":    "#2ca02c",
            "step_delta":    "#74c476",
            "step_penalty":  "#006d2c",

            # Komponenten-Aliasse → gleiche Farben wie Gruppen
            "env_score":     "#6baed6",  # Final (Env)
            "final_bonus":   "#08519c",  # Final (Rank)
            "shaping":       "#2ca02c",  # Step (Total)

            # Scatter-Grundfarbe (falls benötigt)
            "component_scatter": "#B0B0B0",
        }

        # Plot-Styling (zentral)
        self._ret_line_width   = 1.4      # dicker: MA-Linie
        self._ret_outline_add  = 0.5      # weißer Rand um Linie (Kontrast)
        self._sc_total_size    = 14.0     # Punkte im Rohplot
        self._sc_comp_size     = 8.0      # Punkte in Komponentenplots
        self._sc_marker        = "o"      # skalierbar
        self._sc_edge          = "none"
        self._sc_linewidths    = 0.0

        # Sichtbarkeit/“Kräftigkeit” der Punkte (Alpha wird in _fade_rgba genutzt)
        self._fade_alpha_total = 0.35     # Rohplot-Punkte
        self._fade_alpha_comp  = 0.40     # Komponenten-Punkte
        self._fade_mix_blue    = 0.16     # Blau weniger aufhellen (satter)
        self._fade_mix_other   = 0.26     # andere Farben
        self._ret_use_outline  = True




        # In-Memory Episode-Returns
        self.ep_ret_eps: List[int] = []     # global_ep
        self.ep_ret_vals: List[float] = []  # Trainings-Reward (Buffersignal)
        self.ep_ret_components: Dict[str, List[float]] = {}  # z.B. {"env_score":[...], "shaping":[...], "final_bonus":[...]}

    # ---------- Hilfsfunktionen ----------
    def _fade_rgba(self, col, alpha=None):
        a = self._fade_alpha_total if alpha is None else float(alpha)
        if col is None:
            return (0.0, 0.0, 0.0, a)
        rgb = np.array(mcolors.to_rgb(col), dtype=float)
        h, s, v = colorsys.rgb_to_hsv(*rgb)
        mix = self._fade_mix_blue if (0.55 <= h <= 0.72) else self._fade_mix_other
        light = mix * np.ones(3) + (1.0 - mix) * rgb
        return (float(light[0]), float(light[1]), float(light[2]), a)
    
    def _savefig(self, out_path_no_change: str, *, dpi: int = 300):
        """
        Nimmt einen Pfad (z. B. .../plot.png) und speichert in allen
        Formaten aus self._out_formats (png/svg/pdf).
        """
        base, _ = os.path.splitext(out_path_no_change)
        for fmt in self._out_formats:
            path = f"{base}.{fmt}"
            plt.savefig(path, bbox_inches="tight", transparent=False, dpi=dpi)




    # ---------- Benchmark: add ----------
    def add_benchmark(
        self,
        episode: int,
        results: Dict[str, dict],
        macro_wr: Optional[float] = None,
        macro_reward: Optional[float] = None,
    ):
        if not self.bench_names:
            self.bench_names = list(results.keys())
            self.bench_hist_wr = {n: [] for n in self.bench_names}
            self.bench_hist_reward = {n: [] for n in self.bench_names}
            self.bench_hist_places = {n: [] for n in self.bench_names}

        if not hasattr(self, "bench_hist_n"):
            self.bench_hist_n = {n: [] for n in self.bench_names}
        if not hasattr(self, "bench_hist_k"):
            self.bench_hist_k = {n: [] for n in self.bench_names}

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
                        rest = f.read()
                        with open(self.bench_csv_path, "w", newline="", encoding="utf-8") as fw:
                            fw.write(header_line)
                            fw.write(rest)
            if need_header:
                with open(self.bench_csv_path, "w", newline="", encoding="utf-8") as f:
                    f.write(header_line)

        self.bench_episodes.append(int(episode))

        row = [episode]
        wr_for_macro, rw_for_macro = [], []
        for name in self.bench_names:
            val = results.get(name, float("nan"))

            if isinstance(val, (int, float)):
                wr = float(val)
                rw = float("nan")
                p0 = p1 = p2 = p3 = float("nan")
                n_i = 0
                k_i = 0
            else:
                wr = float(val.get("winrate", float("nan")))
                rw = float(val.get("reward", val.get("mean_reward", float("nan"))))
                place = val.get("places") or val.get("place_dist", [float("nan")] * 4)
                p0 = float(place[0]) if len(place) > 0 else float("nan")
                p1 = float(place[1]) if len(place) > 1 else float("nan")
                p2 = float(place[2]) if len(place) > 2 else float("nan")
                p3 = float(place[3]) if len(place) > 3 else float("nan")

                n_i = int(val.get("episodes", 0) or 0)
                if "wins" in val and val.get("wins") is not None:
                    k_i = int(val.get("wins", 0) or 0)
                else:
                    if math.isfinite(wr) and n_i > 0:
                        k_i = int(round(wr / 100.0 * n_i))
                        k_i = max(0, min(k_i, n_i))
                    else:
                        k_i = 0

            self.bench_hist_wr[name].append(wr)
            self.bench_hist_reward[name].append(rw)
            self.bench_hist_places[name].append((p0, p1, p2, p3))
            self.bench_hist_n.setdefault(name, []).append(n_i)
            self.bench_hist_k.setdefault(name, []).append(k_i)

            row += [wr, rw, p0, p1, p2, p3]
            if math.isfinite(wr):
                wr_for_macro.append(wr)
            if math.isfinite(rw):
                rw_for_macro.append(rw)

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
        smooth_window: int = 1,
        show_ci: bool = True,
        ci_z: float = 1.96,
        variants: Optional[List[str]] = None,   # <--- NEU
    ):
        """Winrate-Kurven (ggf. geglättet) + optionale Konfidenzbänder."""
        if not self.bench_episodes:
            return

        def _pretty_opp(name: str) -> str:
            mapping = {
                "max_combo": "Max Combo",
                "random2": "Random 2",
                "single_only": "Single Only",
            }
            return mapping.get(name, name)

        def _title_single(opp_pretty: str) -> str:
            return f"Lernkurve - {family_title} vs {opp_pretty}" if family_title else f"{filename_prefix}"

        def _title_multi() -> str:
            if multi_title:
                return multi_title
            if family_title:
                return f"Lernkurve (Benchmark) - {family_title} vs. feste Heuristiken"
            return f"{filename_prefix}"

        # Farben
        base_colors = plt.rcParams.get("axes.prop_cycle", None)
        if base_colors is not None:
            base_colors = base_colors.by_key().get("color", [])
        if not base_colors:
            base_colors = [plt.cm.tab10(i) for i in range(10)]
        colors = {name: base_colors[i % len(base_colors)] for i, name in enumerate(self.bench_names)}
        macro_color = "#444444"

        # Achsen/Styling
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
                plt.axhline(25, color="r", linewidth=1, label="25% Linie")

        # Helfer: Wilson-CI & rollierende Summen (lokal für diesen Plot)
        def _wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
            if n <= 0:
                return (float("nan"), float("nan"))
            p = k / n
            denom = 1.0 + (z * z) / n
            center = (p + (z * z) / (2 * n)) / denom
            delta = z * math.sqrt(p * (1 - p) / n + (z * z) / (4 * n * n)) / denom
            lo = max(0.0, center - delta)
            hi = min(1.0, center + delta)
            return lo, hi

        def _roll_sum(xs: List[int], w: int) -> List[int]:
            if w <= 1 or not xs:
                return list(xs)
            out = []
            from collections import deque
            q = deque()
            s = 0
            for x in xs:
                q.append(int(x)); s += int(x)
                if len(q) > w:
                    s -= int(q.popleft())
                out.append(int(s))
            return out

        def _movavg(v: np.ndarray, w: int) -> np.ndarray:
            if v.size == 0 or w <= 1:
                return v
            w = max(1, int(w))
            c = np.cumsum(np.insert(v, 0, 0.0))
            tail = (c[w:] - c[:-w]) / float(w)
            head = np.array([np.mean(v[:i + 1]) for i in range(min(w - 1, v.size))], dtype=float)
            return np.concatenate([head, tail])

        def _series_pct_and_ci(name: str):
            wr_raw = self.bench_hist_wr[name]
            has_counts = hasattr(self, "bench_hist_k") and hasattr(self, "bench_hist_n") \
                        and (name in self.bench_hist_k) and (name in self.bench_hist_n) \
                        and (len(self.bench_hist_k[name]) == len(wr_raw)) \
                        and any(int(n) > 0 for n in self.bench_hist_n[name])

            if has_counts:
                k = list(self.bench_hist_k[name])
                n = list(self.bench_hist_n[name])
                k_use = _roll_sum(k, smooth_window) if smooth_window and smooth_window > 1 else k
                n_use = _roll_sum(n, smooth_window) if smooth_window and smooth_window > 1 else n

                y, lo, hi = [], [], []
                T = len(self.bench_episodes)
                for t in range(T):
                    nn = n_use[t] if t < len(n_use) else 0
                    kk = k_use[t] if t < len(k_use) else 0
                    if nn > 0:
                        l, h = _wilson_interval(kk, nn, z=ci_z)
                        p = kk / nn
                        y.append(100.0 * p)
                        lo.append(100.0 * l)
                        hi.append(100.0 * h)
                    else:
                        v = wr_raw[t] if t < len(wr_raw) else float("nan")
                        y.append(v); lo.append(float("nan")); hi.append(float("nan"))
                return np.array(y), np.array(lo), np.array(hi)
            else:
                y = np.array(wr_raw, dtype=float)
                y = _movavg(y, smooth_window) if smooth_window and smooth_window > 1 else y
                lo = np.full_like(y, np.nan, dtype=float)
                hi = np.full_like(y, np.nan, dtype=float)
                return y, lo, hi

        def _plot_multi(out_path: str, include_macro: bool, add_25: bool = False):
            plt.figure(figsize=(12, 8))
            for name in self.bench_names:
                y, lo, hi = _series_pct_and_ci(name)
                plt.plot(
                    self.bench_episodes,
                    y,
                    marker="o",
                    markersize=3,
                    label=_pretty_opp(name),
                    color=colors[name],
                )
                if show_ci and np.isfinite(lo).any():
                    plt.fill_between(self.bench_episodes, lo, hi, alpha=0.15, color=colors[name], linewidth=0)
            if include_macro:
                ys_macro = []
                T = len(self.bench_episodes)
                for t in range(T):
                    vals_t = []
                    for nm in self.bench_names:
                        yy, _, _ = _series_pct_and_ci(nm)
                        if t < len(yy) and math.isfinite(yy[t]):
                            vals_t.append(yy[t])
                    ys_macro.append(float(np.mean(vals_t)) if vals_t else float("nan"))
                plt.plot(
                    self.bench_episodes,
                    ys_macro,
                    marker="o",
                    markersize=3,
                    linestyle="--",
                    label="Avg Macro",
                    color=macro_color,
                )
            _apply_common_axes(_title_multi(), add_25=add_25)
            plt.legend(loc="upper right")
            plt.tight_layout()
            self._savefig(out_path)
            plt.close()

        def _plot_single(name: str, out_path: str, add_25: bool = False, title_override: str | None = None):
            plt.figure(figsize=(10, 6))
            y, lo, hi = _series_pct_and_ci(name)
            plt.plot(
                self.bench_episodes,
                y,
                marker="o",
                markersize=4,
                color=colors[name],
            )
            if show_ci and np.isfinite(lo).any():
                plt.fill_between(self.bench_episodes, lo, hi, alpha=0.2, color=colors[name], linewidth=0)
            title = title_override or _title_single(_pretty_opp(name))
            _apply_common_axes(title, add_25=add_25)
            plt.tight_layout()
            self._savefig(out_path)
            plt.close()

        os.makedirs(self.out_dir, exist_ok=True)
        variants = set(variants or ("03",))  # Default: nur 03 rendern

        # --- Multi-Plots nach Variante ---
        if "01" in variants:
            _plot_multi(
                out_path=os.path.join(self.out_dir, "01_Lernkurve_Heuristiken.png"),
                include_macro=False, add_25=False,
            )
        if "02" in variants:
            _plot_multi(
                out_path=os.path.join(self.out_dir, "02_Lernkurve_Heuristiken_incl_avg.png"),
                include_macro=True, add_25=False,
            )
        if "03" in variants:
            _plot_multi(
                out_path=os.path.join(self.out_dir, "03_Lernkurve_Heuristiken_incl_avg_25.png"),
                include_macro=True, add_25=True,
            )
        if "04" in variants:
            _plot_multi(
                out_path=os.path.join(self.out_dir, "04_Lernkurve_Heuristiken_incl_25.png"),
                include_macro=False, add_25=True,
            )

        # --- Einzelplots nur wenn explizit angefordert ---
        if "05" in variants and "max_combo" in self.bench_names:
            _plot_single(
                "max_combo",
                os.path.join(self.out_dir, "05_Lernkurve_max_combo.png"),
                add_25=False,
                title_override=_title_single(_pretty_opp("max_combo")),
            )

        if "06" in variants and "random2" in self.bench_names:
            _plot_single(
                "random2",
                os.path.join(self.out_dir, "06_Lernkurve_random2.png"),
                add_25=False,
                title_override=_title_single(_pretty_opp("random2")),
            )

        if "07" in variants and self.bench_names:
            first_name = self.bench_names[0]
            _plot_single(
                first_name,
                os.path.join(self.out_dir, "07_Lernkurve_single_only.png"),
                add_25=False,
                title_override=_title_single(_pretty_opp(first_name)),
            )


    def plot_benchmark_rewards(self, filename_prefix: str = "benchmark_rewards", with_macro: bool = True, title_prefix: str | None = None):
        """Ø-Reward-Kurven aus dem Benchmark."""
        if not self.bench_episodes:
            return
        title = title_prefix or filename_prefix

        for name in self.bench_names:
            plt.figure(figsize=(10, 6))
            plt.plot(self.bench_episodes, self.bench_hist_reward[name], marker="o")
            plt.title(f"{title} – Ø-Reward vs {name}")
            plt.xlabel("Episode")
            plt.ylabel("Ø-Reward (P0)")
            plt.grid(True)
            plt.tight_layout()
            out = os.path.join(self.out_dir, f"{filename_prefix}_{name}.png")
            self._savefig(out)
            plt.close()

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
        self._savefig(out_all)
        plt.close()

    def plot_places_latest(self, filename_prefix: str = "places"):
        """Schreibt Platzierungsverteilung der letzten Benchmark-Episode in CSV (append)."""
        if not self._bench_save_csv:
            return
        if not self.bench_episodes:
            return

        last_idx = len(self.bench_episodes) - 1
        ep = int(self.bench_episodes[last_idx])

        rows = []
        for name in self.bench_names:
            if self.bench_hist_places.get(name):
                p0, p1, p2, p3 = self.bench_hist_places[name][last_idx]
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
        out_csv = os.path.join(self.out_dir, f"{filename_prefix}_latest.csv")
        fieldnames = ["opponent", "episode", "p0", "p1", "p2", "p3"]

        file_exists = os.path.isfile(out_csv) and os.path.getsize(out_csv) > 0

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
                    pass

        rows_to_write = [r for r in rows if (r["opponent"], r["episode"]) not in existing_keys]
        if not rows_to_write:
            return

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
        special_keys = {
            "ep_return_raw", "ep_return_components",
            "ep_return_env", "ep_return_shaping", "ep_return_final",
            "ep_return_training",
        }

        wants_special = False
        if include_keys is not None:
            wants_special = any(k in special_keys for k in include_keys)

        if (not self.train_rows or not self.train_keys) and not wants_special:
            return

        episodes = [r["episode"] for r in self.train_rows]
        all_keys = [k for k in self.train_keys if k != "episode"]

        keys = all_keys
        if include_keys is not None:
            allow = set(include_keys)
            keys = [k for k in keys if k in allow]
        if exclude_keys is not None:
            deny = set(exclude_keys)
            keys = [k for k in keys if k not in deny]

        if include_keys is not None:
            req = set(include_keys)

            if ("ep_return_raw" in req) or ("ep_return_components" in req) or ("ep_return_training" in req):
                self.plot_ep_returns(window=self.smooth_window)

            mapping = {
                "ep_return_env":     "env_score",
                "ep_return_shaping": "shaping",
                "ep_return_final":   "final_bonus",
            }
            for pseudo, comp in mapping.items():
                if pseudo in req:
                    self.plot_ep_return_component(comp, window=self.smooth_window)

            keys = [k for k in keys if k not in {"ep_return_raw", "ep_return_components",
                                                 "ep_return_env", "ep_return_shaping",
                                                 "ep_return_final", "ep_return_training"}]

        if not keys:
            return

        if separate:
            for k in keys:
                vals = [r.get(k, float("nan")) for r in self.train_rows]
                plt.figure(figsize=(10, 6))
                plt.plot(episodes, vals, marker="o", markersize=3, linestyle="None")
                plt.title(f"Training – {k}")
                plt.xlabel("Episode"); plt.ylabel(k)
                plt.grid(True); plt.tight_layout()
                self._savefig(os.path.join(self.out_dir, f"{filename_prefix}_{k}.png"))
                plt.close()
        else:
            plt.figure(figsize=(12, 8))
            for k in keys:
                vals = [r.get(k, float("nan")) for r in self.train_rows]
                plt.plot(episodes, vals, marker="o", markersize=3, linestyle="None", label=k)
            plt.title("Training – Metrics")
            plt.xlabel("Episode"); plt.ylabel("Value")
            plt.grid(True); plt.legend(loc="upper left"); plt.tight_layout()
            self._savefig(os.path.join(self.out_dir, f"{filename_prefix}_all.png"))
            plt.close()

    def add_ep_returns(
        self,
        global_episode: int,
        ep_returns: List[float],
        components: Optional[Dict[str, List[float]]] = None
    ):
        """Nimmt Episode-Return(s) entgegen und speichert sie nur im Speicher."""
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
        Erzeugt:
          - ep_return_raw.png (Rohpunkte + Moving Average)
          - ep_return_components.png (Env/Step/Rank gemeinsam)
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
            tail = (c[w:] - c[:-w]) / float(w)
            head = np.array([np.mean(v[:i+1]) for i in range(min(w-1, v.size))], dtype=float)
            return np.concatenate([head, tail])

        y_ma = _movavg(y, window)

        # --- ep_return_raw.png ---
        plt.figure(figsize=(12, 7))
        base_col = self._ret_colors.get("total_ma", "tab:red")

        color = self._fade_rgba(base_col, alpha=self._fade_alpha_total)

        plt.scatter(
            x, y,
            s=self._sc_total_size,
            marker=self._sc_marker,
            color=color,
            edgecolors=self._sc_edge,
            linewidths=self._sc_linewidths,
            zorder=1,
            rasterized=True,  # <--- hinzufügen
        )



        line_ma, = plt.plot(
            x, y_ma,
            linewidth=self._ret_line_width,
            label=f"Rewards gesamt (MA, w={window})",
            color=base_col,
            zorder=3,
        )
        if self._ret_use_outline:
            line_ma.set_path_effects([
                pe.Stroke(linewidth=self._ret_line_width + self._ret_outline_add, foreground="white"),
                pe.Normal()
            ])


        plt.title("Training – Rewards (roh)")
        plt.xlabel("Trainings-Episode")
        plt.ylabel("Rewards")
        plt.grid(True); plt.legend(); plt.tight_layout()
        self._savefig(os.path.join(self.out_dir, "ep_return_raw.png"))
        plt.close()

        # --- ep_return_components.png ---
        comp_keys = ["env_score", "shaping", "final_bonus"]
        for k in comp_keys:
            if k not in self.ep_ret_components:
                self.ep_ret_components[k] = [0.0] * len(self.ep_ret_eps)

        legend_names = {
            "env_score":   "Final Reward (Env)",
            "shaping":     "Step Reward",
            "final_bonus": "Final Reward (Rank)",
        }

        plt.figure(figsize=(12, 7))
        for k in comp_keys:
            v = np.asarray(self.ep_ret_components[k], dtype=float)
            col = self._ret_colors.get(k, None)
            plt.plot(x, _movavg(v, window), label=legend_names.get(k, k), color=col)

        plt.title("Reward-Komponenten")
        plt.xlabel("Trainings-Episode")
        plt.ylabel("Rewards")
        plt.grid(True); plt.legend(); plt.tight_layout()
        self._savefig(os.path.join(self.out_dir, "ep_return_components.png"))
        plt.close()

    def plot_ep_return_component(self, component_key: str, window: int | None = None,
                                filename: Optional[str] = None, title: Optional[str] = None,
                                show_legend: bool = True):
        """
        Einzelplot für eine Return-Komponente.
        - Scatter: sehr klein & kräftig (aufgehellte Linienfarbe), ohne Label.
        - Linie: Moving Average mit optionaler Legende.
        - Schreibt die Datei immer, auch wenn (noch) keine Daten da sind.
        """
        os.makedirs(self.out_dir, exist_ok=True)

        legend_names = {
            "env_score":    "Final Reward (Env)",
            "final_bonus":  "Final Reward (Rank)",
            "shaping":      "Step Reward (Total)",
            "step_delta":   "Step Reward (Combo)",
            "step_penalty": "Step Reward (Hand-Penalty)",
        }
        nice_name = legend_names.get(component_key, component_key)

        window   = self.smooth_window if (window is None) else int(window)
        out_name = filename or f"ep_return_{component_key}.png"
        ttl      = title or f"Training – {nice_name}"
        base_col = self._ret_colors.get(component_key, None)

        x = np.asarray(self.ep_ret_eps, dtype=float)
        v = np.asarray(self.ep_ret_components.get(component_key, []), dtype=float)

        def _movavg(arr: np.ndarray, w: int) -> np.ndarray:
            if arr.size == 0 or w <= 1:
                return arr
            w = max(1, int(w)); w = min(w, arr.size)
            c = np.cumsum(np.insert(arr, 0, 0.0))
            tail = (c[w:] - c[:-w]) / float(w)
            head = np.array([np.mean(arr[:i+1]) for i in range(w-1)], dtype=float)
            return np.concatenate([head, tail])

        plt.figure(figsize=(12, 7))

        if v.size > 0 and x.size > 0:
            L = min(x.size, v.size)

            # >>> NEU: Final-Scatter kräftiger und größer
            is_final = component_key in ("env_score", "final_bonus")

            alpha = (0.22 if is_final else self._fade_alpha_comp)         # Transparenz
            size  = (self._sc_comp_size * 1.35 if is_final else self._sc_comp_size)  # Größe

            # Final: kein Aufhellen → sattes Original mit Alpha
            # Sonst: aufgehellte Variante wie bisher
            color = (mcolors.to_rgba(base_col, alpha=alpha)
                    if is_final else self._fade_rgba(base_col, alpha=alpha))

            plt.scatter(
                x[:L], v[:L],
                s=size,
                marker=self._sc_marker,
                color=color,
                edgecolors=self._sc_edge,
                linewidths=self._sc_linewidths,
                zorder=1,
                rasterized=True,  # <--- hinzufügen
            )



            mv = _movavg(v[:L], window)
            line_comp, = plt.plot(
                x[:L], mv,
                linewidth=self._ret_line_width,
                label=(f"{nice_name} (MA)" if show_legend else None),
                color=base_col,
                zorder=3,
            )
            if self._ret_use_outline:
                line_comp.set_path_effects([
                    pe.Stroke(linewidth=self._ret_line_width + self._ret_outline_add, foreground="white"),
                    pe.Normal()
                ])

            if show_legend:
                plt.legend()

        plt.title(ttl)
        plt.xlabel("Trainings-Episode")
        plt.ylabel("Rewards")
        plt.grid(True); plt.tight_layout()
        self._savefig(os.path.join(self.out_dir, out_name))
        plt.close()

    def plot_reward_groups(
        self,
        window: int | None = None,
        out_all: str   = "08_rewards_all_components.png",
        out_final: str = "09_rewards_final_components.png",
        out_step: str  = "12_rewards_step_components.png",
        also_individual: bool = True,
    ):
        """
        Erzeugt:
          - rewards_all.png   (Total, Final-Total, Env, Rank, Step-Total, Combo, Hand-Penalty)
          - rewards_final.png (Final-Total, Env, Rank)
          - rewards_step.png  (Step-Total, Combo, Hand-Penalty)
          Optional zusätzlich die 4 Einzelplots (s. Dateinamen unten).
        """
        os.makedirs(self.out_dir, exist_ok=True)
        window = self.smooth_window if (window is None) else int(window)

        # --- Daten ---
        x = np.asarray(self.ep_ret_eps, dtype=float)
        total_all   = np.asarray(self.ep_ret_vals, dtype=float)

        def comp(k): return np.asarray(self.ep_ret_components.get(k, []), dtype=float)
        env        = comp("env_score")
        rank       = comp("final_bonus")
        step_total = comp("shaping")
        step_delta = comp("step_delta")
        step_pen   = comp("step_penalty")

        L_final = min(len(env), len(rank))
        final_total = env[:L_final] + rank[:L_final] if L_final > 0 else np.array([], dtype=float)

        # Farben absichern
        self._ret_colors.setdefault("total_all",   "#222222")
        self._ret_colors.setdefault("final_total", "#1f77b4")
        self._ret_colors.setdefault("final_env",   "#6baed6")
        self._ret_colors.setdefault("final_rank",  "#08519c")
        self._ret_colors.setdefault("step_total",  "#2ca02c")
        self._ret_colors.setdefault("step_delta",  "#74c476")
        self._ret_colors.setdefault("step_penalty","#006d2c")

        styles = {
            "Total Rewards":                 dict(color=self._ret_colors["total_all"],   lw=2.6, ls="-"),
            "Final Reward (Total)":          dict(color=self._ret_colors["final_total"], lw=2.2, ls="-"),
            "Final Reward (Env)":            dict(color=self._ret_colors["final_env"],   lw=1.8, ls="--"),
            "Final Reward (Rank)":           dict(color=self._ret_colors["final_rank"],  lw=1.8, ls=":"),
            "Step Reward (Total)":           dict(color=self._ret_colors["step_total"],  lw=2.2, ls="-"),
            "Step Reward (Combo)":           dict(color=self._ret_colors["step_delta"],  lw=1.8, ls="--"),
            "Step Reward (Hand-Penalty)":    dict(color=self._ret_colors["step_penalty"],lw=1.8, ls=":"),
        }

        series = [
            ("Total Rewards",              total_all),
            ("Final Reward (Total)",       final_total),
            ("Final Reward (Env)",         env),
            ("Final Reward (Rank)",        rank),
            ("Step Reward (Total)",        step_total),
            ("Step Reward (Combo)",        step_delta),
            ("Step Reward (Hand-Penalty)", step_pen),
        ]

        def _movavg(arr: np.ndarray, w: int) -> np.ndarray:
            if arr.size == 0 or w <= 1: return arr
            w = max(1, int(w))
            c = np.cumsum(np.insert(arr, 0, 0.0))
            tail = (c[w:] - c[:-w]) / float(w)
            head = np.array([np.mean(arr[:i+1]) for i in range(min(w-1, arr.size))], dtype=float)
            return np.concatenate([head, tail])

        def _plot_one(x, y, style):
            if x.size == 0 or y.size == 0: return None
            L = min(x.size, y.size)
            (h,) = plt.plot(x[:L], _movavg(y[:L], window), **style)
            return h

        def _legend_all(labels, handles_map):
            hs = []
            for name in labels:
                if name in handles_map:
                    hs.append(handles_map[name])
                else:
                    st = styles[name]
                    hs.append(Line2D([0], [0], color=st["color"], lw=st["lw"], ls=st["ls"]))
            plt.legend(hs, labels)

        # --- (A) Alle ---
        plt.figure(figsize=(12, 7))
        handles = {}
        for name, y in series:
            h = _plot_one(x, y, styles[name])
            if h is not None: handles[name] = h
        plt.title("Rewards (Training) – Alle Komponenten")
        plt.xlabel("Trainings-Episode"); plt.ylabel("Rewards"); plt.grid(True)
        _legend_all([n for n,_ in series], handles)
        plt.tight_layout(); self._savefig(os.path.join(self.out_dir, out_all)); plt.close()

        # --- (B) Final ---
        plt.figure(figsize=(12, 7))
        handles = {}
        for name in ["Final Reward (Total)", "Final Reward (Env)", "Final Reward (Rank)"]:
            y = dict(series)[name]
            h = _plot_one(x, y, styles[name])
            if h is not None: handles[name] = h
        plt.title("Rewards (Training) – Final Komponenten")
        plt.xlabel("Trainings-Episode"); plt.ylabel("Rewards"); plt.grid(True)
        _legend_all(["Final Reward (Total)", "Final Reward (Env)", "Final Reward (Rank)"], handles)
        plt.tight_layout(); self._savefig(os.path.join(self.out_dir, out_final)); plt.close()

        # --- (C) Step ---
        plt.figure(figsize=(12, 7))
        handles = {}
        for name in ["Step Reward (Total)", "Step Reward (Combo)", "Step Reward (Hand-Penalty)"]:
            y = dict(series)[name]
            h = _plot_one(x, y, styles[name])
            if h is not None: handles[name] = h
        plt.title("Rewards (Training) – Step Komponenten")
        plt.xlabel("Trainings-Episode"); plt.ylabel("Rewards"); plt.grid(True)
        _legend_all(["Step Reward (Total)", "Step Reward (Combo)", "Step Reward (Hand-Penalty)"], handles)
        plt.tight_layout(); self._savefig(os.path.join(self.out_dir, out_step)); plt.close()

        # --- (D) Einzelplots (ohne Legende) ---
        if also_individual:
            self.plot_ep_return_component("env_score",   window=window,
                                          filename="10_final_reward_env.png",
                                          title="Rewards (Training) – Final Reward (Env)",
                                          show_legend=False)
            self.plot_ep_return_component("final_bonus", window=window,
                                          filename="11_final_reward_rank.png",
                                          title="Rewards (Training) – Final Reward (Rank-Bonus)",
                                          show_legend=False)
            self.plot_ep_return_component("step_delta",  window=window,
                                          filename="13_step_reward_combo.png",
                                          title="Rewards (Training) – Step Reward (Combo-Bonus)",
                                          show_legend=False)
            self.plot_ep_return_component("step_penalty",window=window,
                                          filename="14_step_reward_hand_penalty.png",
                                          title="Rewards (Training) – Step Reward (Hand-Penalty)",
                                          show_legend=False)

    # ---------- Logging ----------
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
            abs_places = [int(round(p * eps)) if np.isfinite(p) else 0 for p in places]
            place_str = "[" + ", ".join(
                f"{abs_places[i]} ({places[i]:.2f})" if np.isfinite(places[i]) else f"{abs_places[i]} (nan)"
                for i in range(len(abs_places))
            ) + "]"

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

        # + Macro Average
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
