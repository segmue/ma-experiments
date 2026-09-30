# -*- coding: utf-8 -*-
"""
ERZEUGNIS  Anhangsabbildung: die beiden Assoziationsmatrizen B1 und D1 untereinander,
           110 x 110, nach Objektklasse geordnet (dieselbe Ordnung wie Abbildung 4).
           Anders als Abbildung 4 zeigt sie NICHT die Resolversicht, sondern die
           Assoziationswerte selbst: keine Schwelle (assoc_threshold), keine Kappung
           (max_categories); gezeichnet sind alle 11'990 gerichteten Zellen ausserhalb
           der Diagonale.
           Farbskala: klassiert und divergierend, fuer beide Felder dieselben Grenzen
           (-0.60, -0.30, -0.10, -0.03, -0.01 | 0.01, 0.03, 0.10, 0.30). Die unterste
           Klasse (<= -0.60) enthaelt auch die Untergrenze -1; eine eigene Klasse dafuer
           gibt es nicht mehr (Festlegung 30.09.2026).
           Unter dem Farbbalken steht die Besetzung jeder Klasse in Prozent je Mass.
           Ausgabe: 5_figures/output/
                    abb_A_assoziationswerte_b1_d1_klassiert_en_<Datum>.png / .pdf
                    ...__daten.csv          (alle gezeichneten Zellen, B1 und D1)
                    ...__klassen__daten.csv (Klassenbesetzung je Mass)

EINGABE    ma-experiments/matrices/config1/{b1,d1}_matrix.csv
           swissNAMES3D_{PKT,LIN,PLY}.dbf (Attribut OBJEKTKLASSE) fuer die Ordnung;
           Funktionen aus heatmap_b1_d1_stil_kap6.py, Bildsprache aus stil_kap6.py.
           Liegen die DBF nicht unter stil_kap6.SWISSNAMES3D, wird die Kopie in
           experiment2_evaluation_textberg/1_preprocessing/ verwendet.

AUFRUF     python3 abb_A_assoziationswerte_b1_d1_klassiert.py [--out ZIELORDNER]

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
import matplotlib.transforms as mtrans
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

import stil_kap6 as S

_ERSATZ = (S.REPO.parent / "experiment2_evaluation_textberg" / "1_preprocessing"
           / "swissnames3d_2024_2056.shp")
if not (S.SWISSNAMES3D / "swissNAMES3D_PKT.dbf").exists() and _ERSATZ.exists():
    S.SWISSNAMES3D = _ERSATZ

import heatmap_b1_d1_stil_kap6 as H

S.setze_stil()
NAME = "abb_A_assoziationswerte_b1_d1_klassiert_en"

# ── Klassen ────────────────────────────────────────────────────────────────
GRENZEN = [-1.0001, -0.6, -0.3, -0.1, -0.03, -0.01, 0.01, 0.03, 0.1, 0.3, 1.0]
FARBEN = ["#6e1a12", "#9e3a28", "#c8664f", "#e39f8a", "#f5d6cc",  # negativ, dunkel -> hell (unterste Klasse <= -0.60)
          "#f7f5f1",                                              # |v| < 0.01
          "#d3dde8", "#93aec6", "#4c7098", "#1b3350"]             # positiv, hell -> dunkel
KLASSEN = ["≤ −0.60", "−0.60 to −0.30", "−0.30 to −0.10", "−0.10 to −0.03",
           "−0.03 to −0.01", "−0.01 to 0.01", "0.01 to 0.03", "0.03 to 0.10",
           "0.10 to 0.30", "≥ 0.30"]
NK = len(FARBEN)
KARTE = ListedColormap(FARBEN)
NORM = BoundaryNorm(GRENZEN, KARTE.N)
FARBE_DIAG = "#ffffff"
FARBE_GITTER = "#8a8a8a"
FARBE_EIGEN = "#000000"
KURZ = {"Single objects and minor landforms": "Single objects, minor landforms",
        "Water bodies and water structures": "Water bodies and structures",
        "Leisure and sports facilities": "Leisure and sports"}


def prozent(a):
    return "<1" if 0 < a < 0.005 else f"{a * 100:.0f}"


def klasse(v):
    return np.digitize(v, GRENZEN[1:-1], right=False)  # wie BoundaryNorm


def lesen():
    b1 = H.matrix_lesen(S.MATRICES / "config1" / "b1_matrix.csv")
    d1 = H.matrix_lesen(S.MATRICES / "config1" / "d1_matrix.csv")
    assert list(b1.index) == list(d1.index)
    art2klasse, _ = H.klassenzuordnung(S.SWISSNAMES3D)
    reihenfolge, grenzen, beschriftung, _ = H.ordnung_bilden(list(b1.index), art2klasse)
    wb = b1.loc[reihenfolge, reihenfolge].to_numpy(float)
    wd = d1.loc[reihenfolge, reihenfolge].to_numpy(float)
    return reihenfolge, grenzen, beschriftung, wb, wd


def feld(ax, w, grenzen, spannen, titel):
    n = w.shape[0]
    diag = np.eye(n, dtype=bool)
    bild = ax.imshow(np.ma.masked_where(diag, w), cmap=KARTE, norm=NORM,
                     interpolation="nearest", aspect="equal")
    ax.imshow(np.ma.masked_where(~diag, np.zeros_like(w)), cmap=ListedColormap([FARBE_DIAG]),
              interpolation="nearest", aspect="equal", zorder=3)
    for g in grenzen:
        ax.axhline(g - 0.5, color=FARBE_GITTER, linewidth=0.4, zorder=4)
        ax.axvline(g - 0.5, color=FARBE_GITTER, linewidth=0.4, zorder=4)
    for start, laenge in spannen:
        ax.add_patch(plt.Rectangle((start - 0.5, start - 0.5), laenge, laenge, fill=False,
                                   edgecolor=FARBE_EIGEN, linewidth=0.9, zorder=5))
    for r in ax.spines.values():
        r.set_color("#555555"); r.set_linewidth(0.8)
    ax.set_title(titel, fontsize=9.0, loc="left", pad=6)
    return bild


def heatmap(reihenfolge, grenzen, beschriftung, wb, wd, ziel: Path, heute: str):
    spannen = [(s, l) for _, s, l in beschriftung]
    n = len(reihenfolge)
    off = ~np.eye(n, dtype=bool)
    fig = plt.figure(figsize=(6.3, 8.6))
    gs = fig.add_gridspec(2, 1, hspace=0.20, left=0.235, right=0.985, top=0.975, bottom=0.175)
    axo, axu = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    axc = fig.add_axes((0.235, 0.085, 0.75, 0.014))
    bild = feld(axo, wb, grenzen, spannen, "(a)  B₁ — footprint overlap")
    feld(axu, wd, grenzen, spannen, "(b)  D₁ — ring adjacency")
    mitten = [s + l / 2 - 0.5 for _, s, l in beschriftung]
    namen = []
    for name, _, _ in beschriftung:
        nr, rest = name.split(" ", 1)
        namen.append(f"{nr} {KURZ.get(rest, rest)}")
    nummern = [x.split(" ", 1)[0] for x in namen]
    for ax in (axo, axu):
        ax.set_xticks(mitten); ax.set_xticklabels(nummern, fontsize=7.0)
        ax.set_yticks([])
        ax.tick_params(axis="x", length=2.5, width=0.6, pad=2)
    axu.set_xlabel("Partner category (columns), groups 1 to 14; rows: source category",
                   fontsize=8.0, labelpad=4)
    fig.canvas.draw()
    for ax in (axo, axu):
        hp = ax.get_window_extent().height / fig.dpi * 72.0
        gez = H.auseinanderziehen(mitten, 8.6 / hp * n, -0.5, n - 0.5)
        misch = mtrans.blended_transform_factory(ax.transAxes, ax.transData)
        for name, wahr, g in zip(namen, mitten, gez):
            ax.text(-0.040, g, name, transform=misch, ha="right", va="center", fontsize=7.0, clip_on=False)
            ax.plot([-0.034, -0.009], [g, wahr], transform=misch, color="#888888", linewidth=0.5, clip_on=False)
    # Farbbalken: Klassen gleich breit, Anteil je Mass darunter
    kb, kd = klasse(wb[off]), klasse(wd[off])
    cb = fig.colorbar(bild, cax=axc, orientation="horizontal", spacing="uniform",
                      boundaries=GRENZEN, ticks=[])
    axc.set_xlim(GRENZEN[0], GRENZEN[-1])
    # Beschriftung in Klassenmitten (uniform spacing -> Achse laeuft 0..10)
    axc.cla()
    for i, f in enumerate(FARBEN):
        axc.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=f, edgecolor="#555555", linewidth=0.4))
    axc.set_xlim(0, NK); axc.set_ylim(0, 1); axc.set_yticks([])
    axc.set_xticks(list(range(1, NK)))
    axc.set_xticklabels(["−0.60", "−0.30", "−0.10", "−0.03", "−0.01", "0.01", "0.03", "0.10", "0.30"], fontsize=6.6)
    axc.tick_params(axis="x", length=2.5, width=0.6, pad=2)
    for r in axc.spines.values():
        r.set_visible(False)
    tot = off.sum()
    for i in range(NK):
        axc.text(i + 0.5, -1.55, prozent((kb == i).sum() / tot), ha="center", va="top", fontsize=6.3)
        axc.text(i + 0.5, -2.55, prozent((kd == i).sum() / tot), ha="center", va="top", fontsize=6.3)
    axc.text(-0.15, -1.55, "B₁ %", ha="right", va="top", fontsize=6.5)
    axc.text(-0.15, -2.55, "D₁ %", ha="right", va="top", fontsize=6.5)
    axc.text(6.5, 1.25, "Association value (B₁ or D₁), classed",
             ha="center", va="bottom", fontsize=7.6)
    zeichen = [Line2D([0], [0], marker="s", linestyle="none", markersize=6, markerfacecolor=FARBE_DIAG,
                      markeredgecolor="#555555", markeredgewidth=0.5, label="diagonal, left out"),
               Line2D([0], [0], color=FARBE_GITTER, linewidth=1.0, label="group boundary"),
               Line2D([0], [0], color=FARBE_EIGEN, linewidth=1.2, label="own group (block diagonal)")]
    fig.legend(handles=zeichen, loc="lower center", ncol=3, frameon=False, fontsize=7.4,
               bbox_to_anchor=(0.61, -0.005), handletextpad=0.6, columnspacing=1.8)
    S.speichern(fig, ziel, NAME, heute)
    # Klassenbesetzung je Mass
    zeilen = [[k, int((kb == i).sum()), f"{(kb == i).sum() / tot:.4f}",
               int((kd == i).sum()), f"{(kd == i).sum() / tot:.4f}"] for i, k in enumerate(KLASSEN)]
    S.daten_ablegen(ziel, NAME + "__klassen", ["klasse", "b1_zellen", "b1_anteil", "d1_zellen", "d1_anteil"],
                    zeilen, heute)
    # alle gezeichneten Zellen
    ii, jj = np.where(off)
    zeilen = [[reihenfolge[i], reihenfolge[j], f"{wb[i, j]:.6f}", f"{wd[i, j]:.6f}"] for i, j in zip(ii, jj)]
    S.daten_ablegen(ziel, NAME, ["quelle", "partner", "b1", "d1"], zeilen, heute)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=S.AUSGABE)
    args = p.parse_args()
    reihenfolge, grenzen, beschriftung, wb, wd = lesen()
    heatmap(reihenfolge, grenzen, beschriftung, wb, wd, args.out, _dt.date.today().isoformat())


if __name__ == "__main__":
    main()
