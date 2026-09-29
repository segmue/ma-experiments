"""
Stage 08 — Ambiguity-Zerlegung visualisieren.

Erzeugt aus results/matrix/*.json (keine Geoparser-Laufzeit noetig):
  plots/ambiguity_breakdown.png       — links: error_breakdown mit gesplittetem
                                        correct-Band (1 Kandidat vs. disambiguiert),
                                        rechts: 5×3-Heatmap Acc@1 nur-ambig
  results/tables/ambiguity_breakdown.csv — errors-Felder aller 15 Zellen

Verwendung:
    python 08_plot_ambiguity.py
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config as C

MODEL_ORDER = list(C.MODELS)
RESOLVER_ORDER = list(C.MATRIX_RESOLVERS)

# Reihenfolge + Farben der fuenf Baender (Anteile an total; gruen→rot wie
# error_breakdown.png, aber correct zweigeteilt).
BANDS = [
    ("correct_unambiguous", "correct — 1 candidate (trivial)", "#1a7837"),
    ("correct_ambiguous", "correct — disambiguated (>1 candidate)", "#8fce8f"),
    ("wrong_rank", "wrong_rank", "#fee08b"),
    ("gold_not_in_candidates", "gold_not_in_candidates", "#f4a45c"),
    ("no_candidate", "no_candidate", "#d73027"),
]


def load_matrix() -> dict:
    cells = {}
    for path in C.MATRIX_DIR.glob("*.json"):
        c = json.loads(path.read_text(encoding="utf-8"))
        cells[(c["model"], c["eval_resolver"])] = c
    return cells


def main():
    cells = load_matrix()
    if not cells:
        print("Keine Matrix-JSONs gefunden — nichts zu tun.")
        return

    # Tabelle: errors-Felder aller Zellen.
    rows = []
    for m in MODEL_ORDER:
        for e in RESOLVER_ORDER:
            if (m, e) in cells:
                rows.append({"model": m, "eval_resolver": e,
                             **cells[(m, e)]["errors"]})
    df_all = pd.DataFrame(rows)
    df_all.to_csv(C.TABLES_DIR / "ambiguity_breakdown.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                   gridspec_kw={"width_ratios": [1.2, 1]})

    # Panel 1: gestapelte Anteile je Haupt-System, correct-Band gesplittet.
    keys = [k for k, _, _ in BANDS]
    sys_rows = []
    for m, e in C.MAIN_SYSTEMS:
        if (m, e) in cells:
            errs = cells[(m, e)]["errors"]
            sys_rows.append((f"{C.SHORT_MODEL[m]}+{e}", *[errs.get(k, 0) for k in keys]))
    df = pd.DataFrame(sys_rows, columns=["system", *keys]).set_index("system")
    df_pct = df.div(df.sum(axis=1), axis=0)
    bottom = pd.Series(0.0, index=df_pct.index)
    for key, label, color in BANDS:
        ax1.bar(df_pct.index, df_pct[key], bottom=bottom, label=label,
                color=color, width=0.7)
        bottom += df_pct[key]
    ax1.set_ylabel("Share of toponyms")
    ax1.set_title("Error analysis with ambiguity split (location level)")
    ax1.legend(fontsize=8, loc="lower left", framealpha=0.9)
    ax1.set_xticks(range(len(df_pct.index)))
    ax1.set_xticklabels(df_pct.index, rotation=30, ha="right", fontsize=8)

    # Panel 2: Acc@1 nur auf ambigen Items (>1 Kandidat), alle 15 Zellen.
    heat = pd.DataFrame(index=MODEL_ORDER, columns=RESOLVER_ORDER, dtype=float)
    for (m, e), c in cells.items():
        if m in heat.index and e in heat.columns:
            heat.loc[m, e] = c["errors"]["acc1_ambiguous_only"]
    sns.heatmap(heat.rename(index=C.SHORT_MODEL).astype(float), annot=True, fmt=".3f",
                cmap="viridis", vmin=0, vmax=1,
                cbar_kws={"label": "Acc@1 (ambiguous only)"}, ax=ax2)
    ax2.set_title("Acc@1 on ambiguous items only (>1 candidate)")
    ax2.set_ylabel("Model")
    ax2.set_xlabel("Description variant")

    plt.tight_layout()
    plt.savefig(C.PLOTS_DIR / "ambiguity_breakdown.png", dpi=150)
    plt.close()
    print(f"OK: {C.PLOTS_DIR / 'ambiguity_breakdown.png'}")
    print(f"OK: {C.TABLES_DIR / 'ambiguity_breakdown.csv'}")


if __name__ == "__main__":
    main()
