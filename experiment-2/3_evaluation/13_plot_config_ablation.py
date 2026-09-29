"""
Stage 13 — Config-Ablation des Zellindex (Nachtrag 2026-09-11).

Trennt die beiden Faktoren, in denen sich config1 (overlap, max_res 13) und
config2 (center, max_res 10) zugleich unterscheiden. Drei zusaetzliche
Belegungen weichen von config1 in genau einem Feld ab:

    config3  overlap / 11      config4  overlap / 12      config5  center / 13

Gerechnet nur Inferenz auf M3, M4, M5 (M1/M2 nicht). Leitmetrik
acc1_ambiguous_only (n = 23'848).

Eingaben : results/summary.csv            (E_c1, E_c2, E_default — Referenz)
           results/config_ablation/summary_e8.csv  (E_c3, E_c4, E_c5)
Ausgaben : results/tables/config_ablation.csv          alle Metriken, 15 Zellen
           plots/config_ablation.png                   Arbeitsfassung
           visuals/results_kap6_print/fig_config_ablation.{pdf,png}
           results/fig_config_ablation__data.csv (Eingabe von 5_figures/abb_10)
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import config as C

HERE = Path(__file__).resolve().parent
RES = C.RESULTS_DIR
OUT_PRINT = C.RESULTS_DIR          # fig_config_ablation__data.csv (+ Druckfassungen)
OUT_PRINT.mkdir(parents=True, exist_ok=True)

MODELS = [("M3_default_finetuned", "M3"), ("M4_spatial_config1", "M4"),
          ("M5_spatial_config2", "M5")]
# Arm -> (Zuordnungsmodus, feinste Stufe)
ARMS = {"E_c2": ("center", 10), "E_c5": ("center", 13),
        "E_c3": ("overlap", 11), "E_c4": ("overlap", 12), "E_c1": ("overlap", 13)}
BASELINE = ("M3_default_finetuned", "E_default")

# Hausstil der Druckabbildungen: Helligkeit traegt die Reihenfolge (graustufentauglich),
# Identitaet zusaetzlich ueber Markerform und Direktbeschriftung.
COL = {"M3": "#8f8f8f", "M4": "#2f6fb8", "M5": "#1b2f45"}
MRK = {"M3": "o", "M4": "s", "M5": "^"}
C_GRID = "#dcdee3"
C_REF = "#9a9a9a"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5, "axes.labelsize": 8.5,
    "axes.linewidth": 0.6, "axes.edgecolor": "#4a4a4a",
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "legend.frameon": False, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def load(path):
    with open(path, encoding="utf-8") as fh:
        return {(r["model"], r["eval_resolver"]): r for r in csv.DictReader(fh)}


def main():
    ref = load(RES / "summary.csv")
    e8 = load(RES / "config_ablation" / "summary_e8.csv")
    rows = {}
    for m, _ in MODELS:
        for arm in ARMS:
            src = e8 if arm in ("E_c3", "E_c4", "E_c5") else ref
            rows[(m, arm)] = src[(m, arm)]
    # Invarianten: identische Kandidatenmengen in allen Zellen
    for r in rows.values():
        assert int(r["no_candidates"]) == 8159 and int(r["ambiguous_total"]) == 23848
        assert int(r["correct_unambiguous"]) == 43544
    base = float(ref[BASELINE]["acc1_ambiguous_only"])

    # Tabelle mit allen Metriken
    keys = ["acc1_ambiguous_only", "acc1", "acc3", "mrr", "doc_accuracy"]
    tab = RES / "tables" / "config_ablation.csv"
    with open(tab, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "arm", "containment_mode", "max_resolution"] + keys)
        for m, short in MODELS:
            for arm, (mode, res) in ARMS.items():
                w.writerow([short, arm, mode, res] + [rows[(m, arm)][k] for k in keys])
    print("Tabelle:", tab)

    # Abbildung
    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=C_GRID, linewidth=0.5, zorder=0); ax.set_axisbelow(True)
    ax.axhline(base, color=C_REF, lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax.text(9.72, base + 0.004, f"M3/E$_\\mathrm{{default}}$ (baseline) {base:.3f}",
            color="#5a5a5a", fontsize=7, va="bottom")
    data = []
    for m, short in MODELS:
        for mode, ls, fill in (("overlap", "-", True), ("center", "--", False)):
            arms = [a for a, (md, _) in ARMS.items() if md == mode]
            xs = [ARMS[a][1] for a in arms]
            ys = [float(rows[(m, a)]["acc1_ambiguous_only"]) for a in arms]
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            xs = [xs[i] for i in order]; ys = [ys[i] for i in order]
            ax.plot(xs, ys, color=COL[short], lw=1.6 if mode == "overlap" else 1.2,
                    ls=ls, marker=MRK[short], ms=5.5,
                    mfc=COL[short] if fill else "white", mec=COL[short], mew=1.1,
                    zorder=3)
            for a in arms:
                data.append([short, a, mode, ARMS[a][1], rows[(m, a)]["acc1_ambiguous_only"]])
    # Direktbeschriftung am rechten Ende der overlap-Linie (M3/M4 fallen bei 13 zusammen)
    nudge = {"M3": -0.011, "M4": +0.001, "M5": +0.006}
    for m, short in MODELS:
        y = float(rows[(m, "E_c1")]["acc1_ambiguous_only"])
        ax.text(13.12, y + nudge[short], short, color=COL[short], fontsize=8,
                fontweight="bold", va="center")
    ax.set_xlim(9.7, 13.45)
    ax.set_xticks([10, 11, 12, 13])
    ax.set_xticklabels(["10\n≈15 000 m²", "11\n≈2 150 m²", "12\n≈307 m²", "13\n≈44 m²"])
    ax.set_xlabel("finest permitted cell resolution (H3), mean cell area")
    ax.set_ylabel("Acc@1 ambiguous ($n = 23 848$)")
    ax.set_ylim(0.60, 0.87)
    handles = [Line2D([], [], color="#333333", lw=1.6, ls="-", marker="o", ms=4.5,
                      mfc="#333333", label="overlap mode"),
               Line2D([], [], color="#333333", lw=1.2, ls="--", marker="o", ms=4.5,
                      mfc="white", label="centre-point mode (center)")]
    handles += [Line2D([], [], color=COL[s], lw=0, marker=MRK[s], ms=5, label=s)
                for _, s in MODELS]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=3,
              handlelength=2.2, columnspacing=1.4, borderaxespad=0.0)
    C.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.PLOTS_DIR / "config_ablation.png")
    for ext in ("pdf", "png"):
        fig.savefig(C.PLOTS_DIR / f"fig_config_ablation.{ext}")
    with open(OUT_PRINT / "fig_config_ablation__data.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["model", "arm", "containment_mode", "max_resolution",
                                        "acc1_ambiguous_only"]); w.writerows(data)
        w.writerow(["M3", "E_default", "—", "—", f"{base:.4f}"])
    print("Abbildung:", C.PLOTS_DIR / "config_ablation.png", "| Daten:", OUT_PRINT)


if __name__ == "__main__":
    main()
