# -*- coding: utf-8 -*-
"""
ERZEUGNIS  Anhangsabbildung: die Verteilung der Assoziationswerte von B1 und
           D1 als zwei Histogramme untereinander, ueber alle 11'990 gerichteten Zellen
           ausserhalb der Diagonale (keine Schwelle, keine Kappung). Beide Felder
           haben dieselbe x-Achse (symlog, linear innerhalb +-0.01), dieselben 42
           Klassen und dieselbe y-Skala. Es sind die Randhistogramme der frueheren
           Hexbin-Fassung, nur untereinander gestellt. Ergaenzt die klassierte
           Heatmap abb_A_assoziationswerte_b1_d1_klassiert.py.
           Ausgabe: 5_figures/output/
                    abb_A_histogramme_b1_d1_en_<Datum>.png / .pdf
                    ...__daten.csv (Zellen je Histogrammklasse, B1 und D1)
EINGABE    ma-experiments/matrices/config1/{b1,d1}_matrix.csv
AUFRUF     python3 abb_A_histogramme_b1_d1.py [--out ZIELORDNER]
Datum: 30.09.2026
"""
from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import stil_kap6 as S
import heatmap_b1_d1_stil_kap6 as H

S.setze_stil()
NAME = "abb_A_histogramme_b1_d1_en"
LT = 0.01                                   # linearer Bereich der symlog-Achse
FARBE = "#93aec6"
TICKS = [-1, -0.3, -0.1, -0.03, 0, 0.03, 0.1, 0.3]
TLAB = ["−1", "−0.3", "−0.1", "−0.03", "0", "0.03", "0.1", "0.3"]
NEBEN = [-0.01, 0.01]


def sym(v):
    return np.sign(v) * np.log10(1 + np.abs(v) / LT)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=S.AUSGABE)
    args = p.parse_args()
    b1 = H.matrix_lesen(S.MATRICES / "config1" / "b1_matrix.csv")
    d1 = H.matrix_lesen(S.MATRICES / "config1" / "d1_matrix.csv")
    assert list(b1.index) == list(d1.index)
    off = ~np.eye(len(b1), dtype=bool)
    werte = {"B1": b1.to_numpy(float)[off], "D1": d1.loc[b1.index, b1.index].to_numpy(float)[off]}

    lo, hi = sym(-1.0) - 0.08, sym(0.75) + 0.08
    kanten = np.linspace(lo, hi, 43)
    zaehl = {k: np.histogram(sym(v), bins=kanten)[0] for k, v in werte.items()}
    ymax = max(z.max() for z in zaehl.values()) * 1.08

    fig, achsen = plt.subplots(2, 1, figsize=(6.3, 4.4), sharex=True, gridspec_kw=dict(hspace=0.28))
    titel = {"B1": "(a)  B₁ — footprint overlap", "D1": "(b)  D₁ — ring adjacency"}
    for ax, k in zip(achsen, ("B1", "D1")):
        ax.bar(kanten[:-1], zaehl[k], width=np.diff(kanten), align="edge", color=FARBE,
               edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="#555555", linewidth=0.6, zorder=0)
        ax.set_ylim(0, ymax)
        ax.set_xlim(lo, hi)
        ax.set_ylabel("cells", fontsize=8)
        ax.set_title(titel[k], fontsize=9, loc="left", pad=4)
        ax.tick_params(labelsize=7, length=2.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: S.tausender(v)))
    ax = achsen[-1]
    ax.set_xticks(sym(np.array(TICKS)))
    ax.set_xticklabels(TLAB)
    ax.set_xticks(sym(np.array(NEBEN)), minor=True)
    ax.set_xlabel("Association value (symmetric log scale, linear within ±0.01)", fontsize=8)
    achsen[0].tick_params(labelbottom=True)
    achsen[0].set_xticks(sym(np.array(TICKS)))
    achsen[0].set_xticklabels(TLAB)

    heute = _dt.date.today().isoformat()
    S.speichern(fig, args.out, NAME, heute)
    zeilen = [[f"{kanten[i]:.4f}", f"{kanten[i + 1]:.4f}", int(zaehl["B1"][i]), int(zaehl["D1"][i])]
              for i in range(len(kanten) - 1)]
    S.daten_ablegen(args.out, NAME, ["klasse_von_symlog", "klasse_bis_symlog", "b1_zellen", "d1_zellen"],
                    zeilen, heute)


if __name__ == "__main__":
    main()
