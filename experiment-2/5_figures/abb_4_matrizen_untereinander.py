# -*- coding: utf-8 -*-
"""
ERZEUGNIS  Abbildung 4 der Arbeit, neu gesetzt: die beiden Assoziationsmatrizen
           untereinander statt nebeneinander, als ganzseitige Abbildung.
           Oben das Ueberlagerungsmass B1, unten das Adjazenzmass D1, beide
           110 x 110 in derselben Ordnung nach Objektklasse. Ohne Titel und
           ohne Fusszeile: was dort stand, tragen Bildunterschrift und Text
           von Abschnitt 6.1. Die Gruppennamen stehen nur links.
           Ausgabe: 5_figures/output/
                    abb_4_matrizen_untereinander_en_<Datum>.pdf / .png
                    und ...__daten.csv (die gezeichneten Zellen)

EINGABE    dieselben Quellen wie heatmap_b1_d1_stil_kap6.py, dessen Funktionen
           hier eingebunden werden (Matrizen, Objektklassen, Ordnung, Maske,
           Feldzeichnung). Kein Wert ist geaendert.

AUFRUF     python3 abb_4_matrizen_untereinander.py [--out ZIELORDNER] [--dpi 400]

Datum: 26.09.2026
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

import heatmap_b1_d1_stil_kap6 as H
import stil_kap6 as S

S.setze_stil()

NAME = "abb_4_matrizen_untereinander_en"

# Kuerzere Gruppennamen fuer die linke Beschriftung; die langen Namen stehen in
# der Ordnungstabelle (Kategorie_Objektklasse_Gruppe_stil_kap6_2026-09-18.csv).
KURZ = {
    "Single objects and minor landforms": "Single objects, minor landforms",
    "Water bodies and water structures": "Water bodies and structures",
    "Leisure and sports facilities": "Leisure and sports",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=S.AUSGABE)
    p.add_argument("--dpi", type=int, default=400)
    args = p.parse_args()
    b1 = H.matrix_lesen(S.MATRICES / "config1" / "b1_matrix.csv")
    d1 = H.matrix_lesen(S.MATRICES / "config1" / "d1_matrix.csv")
    if list(b1.index) != list(d1.index):
        raise SystemExit("Die beiden Matrizen fuehren nicht dieselbe Kategorienordnung")
    shp = S.SWISSNAMES3D
    art2klasse, _ = H.klassenzuordnung(shp)
    reihenfolge, grenzen, beschriftung, _ = H.ordnung_bilden(list(b1.index), art2klasse)

    wb = b1.loc[reihenfolge, reihenfolge].to_numpy(dtype=float)
    wd = d1.loc[reihenfolge, reihenfolge].to_numpy(dtype=float)
    mb, md = H.resolversicht(wb), H.resolversicht(wd)
    norm = LogNorm(vmin=H.SKALA_UNTEN, vmax=H.SKALA_OBEN)
    spannen = [(start, laenge) for _, start, laenge in beschriftung]

    # Hochformat: zwei Quadrate uebereinander. Bei 16 cm Satzbreite bleibt die
    # Abbildung mit diesem Seitenverhaeltnis unter 23 cm Hoehe, also auf einer
    # Seite samt Unterschrift.
    fig = plt.figure(figsize=(6.3, 8.2))
    gs = fig.add_gridspec(2, 1, hspace=0.24,
                          left=0.235, right=0.985, top=0.975, bottom=0.170)
    axo = fig.add_subplot(gs[0, 0])
    axu = fig.add_subplot(gs[1, 0])
    axc = fig.add_axes((0.41, 0.100, 0.42, 0.012))

    bild = H.feld_zeichnen(axo, wb, mb, norm, grenzen, spannen,
                           f"(a)  B₁ — footprint overlap   "
                           f"({H._tsd(int(mb.sum()))} partner pairs)")
    H.feld_zeichnen(axu, wd, md, norm, grenzen, spannen,
                    f"(b)  D₁ — ring adjacency   "
                    f"({H._tsd(int(md.sum()))} partner pairs)")
    for ax in (axo, axu):
        ax.title.set_fontsize(9.0)
        ax.title.set_ha("left")
        ax.title.set_position((0.0, 1.0))

    n = len(reihenfolge)
    mitten = [start + laenge / 2 - 0.5 for _, start, laenge in beschriftung]
    namen = [KURZ.get(name.split(" ", 1)[1], name.split(" ", 1)[1]) for name, _, _ in beschriftung]
    namen = [f"{name.split(' ', 1)[0]} {kurz}" for name, kurz in zip((b[0] for b in beschriftung), namen)]
    nummern = [name.split(" ", 1)[0] for name in namen]

    for ax in (axo, axu):
        ax.set_xticks(mitten)
        ax.set_xticklabels(nummern, fontsize=7.0)
        ax.set_yticks([])
        ax.tick_params(axis="x", length=2.5, width=0.6, pad=2)
        ax.set_xlabel("Partner category (columns), groups 1 to 14; rows: source category",
                      fontsize=8.0, labelpad=4)

    fig.canvas.draw()
    for ax in (axo, axu):
        hoehe_punkte = ax.get_window_extent().height / fig.dpi * 72.0
        mindest = 8.6 / hoehe_punkte * n
        gezogen = H.auseinanderziehen(mitten, mindest, -0.5, n - 0.5)
        misch = mtrans.blended_transform_factory(ax.transAxes, ax.transData)
        for name, wahr, gez in zip(namen, mitten, gezogen):
            ax.text(-0.040, gez, name, transform=misch, ha="right", va="center",
                    fontsize=7.0, clip_on=False)
            ax.plot([-0.034, -0.009], [gez, wahr], transform=misch,
                    color="#888888", linewidth=0.5, clip_on=False)
        pass

    balken = fig.colorbar(bild, cax=axc, extend="max", orientation="horizontal")
    balken.set_label("Association value of the selected partner (B₁ or D₁), log scale",
                     fontsize=7.8, labelpad=3)
    balken.ax.tick_params(labelsize=7.0, length=2.5, width=0.6, pad=2)
    balken.set_ticks([0.001, 0.003, 0.01, 0.03, 0.1, 0.3])
    balken.ax.set_xticklabels(["0.001", "0.003", "0.01", "0.03", "0.10", "0.30"])
    balken.ax.minorticks_off()

    zeichen = [
        Line2D([0], [0], marker="s", linestyle="none", markersize=6.5,
               markerfacecolor=H.FARBE_REST, markeredgecolor="#555555",
               markeredgewidth=0.5, label="not read by the resolver"),
        Line2D([0], [0], marker="s", linestyle="none", markersize=6.5,
               markerfacecolor=H.FARBE_DIAGONALE, markeredgecolor="#555555",
               markeredgewidth=0.5, label="diagonal, left out"),
        Line2D([0], [0], color=H.FARBE_GITTER, linewidth=1.0, label="group boundary"),
        Line2D([0], [0], color=H.FARBE_EIGENGRUPPE, linewidth=1.2,
               label="own group (block diagonal)"),
    ]
    fig.legend(handles=zeichen, loc="lower center", ncol=2, frameon=False,
               fontsize=7.6, bbox_to_anchor=(0.62, 0.000), handletextpad=0.7,
               columnspacing=2.0)

    ziel = args.out
    heute = _dt.date.today().isoformat()
    S.speichern(fig, ziel, NAME, heute)

    zeilen = []
    for kuerzel, w, m in (("B1", wb, mb), ("D1", wd, md)):
        for i, j in zip(*np.where(m)):
            zeilen.append([kuerzel, reihenfolge[i], reihenfolge[j], f"{w[i, j]:.6f}"])
    S.daten_ablegen(ziel, NAME, ["mass", "quelle", "partner", "wert"], zeilen, heute)


if __name__ == "__main__":
    main()
