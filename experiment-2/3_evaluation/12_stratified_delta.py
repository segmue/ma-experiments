"""
Stage 12 — Stratifizierung nach Objektklasse als DELTA-Heatmap (Variante B).

Ergaenzt plots/stratified_objektart_acc1.png (absolute Werte). Dieselben Zeilen
und Spalten, aber jede Zelle zeigt die Differenz zur fairen Baseline
M3_default_finetuned/E_default derselben Klasse. Damit ist direkt ablesbar,
WO das System gegenueber der Verwaltungshierarchie etwas leistet und wo es ihr
unterliegt — in der absoluten Heatmap muss man das aus zwei Zellen im Kopf
subtrahieren.

Divergierende Farbskala, symmetrisch um null zentriert. Zeilenbeschriftung
traegt Korpushaeufigkeit und Gazetteer-Featurezahl mit, weil beide Groessen
die Werte erst lesbar machen.

Output:
  plots/stratified_objektart_delta.png
  results/tables/stratified_objektart_delta.csv   (die geplotteten Werte)

Verwendung (nach 05_stratified_objektart.py und 11_class_profile.py):
    python 12_stratified_delta.py
    python 12_stratified_delta.py --min-n 150     # engere Auswahl
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config as C

BASELINE = ("M3_default_finetuned", "E_default")
SHORT = {"M1_dguzh": "M1", "M2_distiluse_base": "M2", "M3_default_finetuned": "M3",
         "M4_spatial_config1": "M4", "M5_spatial_config2": "M5"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=50,
                    help="Mindest-Fallzahl je Klasse (Vorgabe 50, wie Stage 05)")
    args = ap.parse_args()

    strat = pd.read_csv(C.TABLES_DIR / "stratified_objektart.csv")
    strat = strat[strat.n >= args.min_n].copy()
    strat["system"] = strat.model.map(SHORT) + "+" + strat.eval_resolver

    base = strat[(strat.model == BASELINE[0]) & (strat.eval_resolver == BASELINE[1])]
    base = base.set_index("objektart")["acc1"]

    wide = strat.pivot(index="objektart", columns="system", values="acc1")
    delta = wide.sub(base, axis=0)

    # Spaltenreihenfolge wie in der absoluten Heatmap: modellweise, Resolver innen.
    cols = [f"{SHORT[m]}+{e}" for m in C.MODELS for e in C.MATRIX_RESOLVERS
            if f"{SHORT[m]}+{e}" in delta.columns]
    delta = delta[cols]

    # Zeilen nach Korpushaeufigkeit, "Other" ans Ende.
    n_by_class = strat.groupby("objektart")["n"].max()
    order = [c for c in n_by_class.sort_values(ascending=False).index if c != "Other"]
    if "Other" in n_by_class.index:
        order.append("Other")
    delta = delta.loc[order]

    # Gazetteer-Featurezahl aus dem Klassenprofil, falls vorhanden.
    gaz = {}
    prof = C.TABLES_DIR / "class_profile.csv"
    if prof.exists():
        pf = pd.read_csv(prof)
        gaz = dict(zip(pf.objektart, pf.n_gazetteer))

    uml = {"Huegelzug": "Hügelzug", "Haupthuegel": "Haupthügel", "Uebrige Bahnen": "Übrige Bahnen"}

    def label(cls: str) -> str:
        disp = uml.get(cls, cls)
        n = f"{int(n_by_class[cls]):,}".replace(",", " ")
        g = gaz.get(cls)
        if pd.notna(g) and g != "":
            return f"{disp}  (n={n} · gaz. {int(g):,})".replace(",", " ")
        return f"{disp}  (n={n})"

    delta.index = [label(c) for c in delta.index]

    full = float(max(abs(delta.min().min()), abs(delta.max().max())))
    # Farbskala kappen: eine einzelne Klasse (Huegelzug, +0.84) wuerde sonst die
    # gesamte Skala belegen und alle uebrigen Zellen fast weiss erscheinen lassen.
    # Die Zahlen bleiben vollstaendig annotiert, nur die Farbe ist gekappt.
    lim = min(full, 0.50)
    clipped = full > lim
    plt.figure(figsize=(16, 11))
    sns.heatmap(delta, annot=True, fmt="+.2f", cmap="RdBu", center=0,
                vmin=-lim, vmax=lim, linewidths=.4, linecolor="white",
                annot_kws={"fontsize": 8},
                cbar_kws={"label": "Δ Accuracy@1 vs. M3+E_default"})
    note = (f" Colour scale clipped at ±{lim:.2f} (maximum {full:+.2f}), "
            "numbers complete." if clipped else "")
    plt.title("Δ Accuracy@1 vs. the fair baseline (M3+E_default), "
              f"by object class (OBJEKTART) × system (classes with n≥{args.min_n})\n"
              "Blue = system better than the administrative hierarchy, red = worse. "
              "The baseline column is zero by definition.\n" + note.strip())
    plt.ylabel("Object class (OBJEKTART, gold)")
    plt.xlabel("System (model+description variant)")
    plt.tight_layout()
    out = C.PLOTS_DIR / "stratified_objektart_delta.png"
    plt.savefig(out, dpi=150)
    plt.close()

    delta.round(4).to_csv(C.TABLES_DIR / "stratified_objektart_delta.csv")
    print(f"{delta.shape[0]} Klassen × {delta.shape[1]} Systeme, "
          f"Spanne {-lim:+.3f} bis {lim:+.3f}")
    print("->", out)
    print("->", C.TABLES_DIR / "stratified_objektart_delta.csv")


if __name__ == "__main__":
    main()
