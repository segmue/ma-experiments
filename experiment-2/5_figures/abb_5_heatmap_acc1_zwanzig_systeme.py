#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Erzeugt: Abbildung 5 — Acc@1 der zwanzig Systeme (fuenf Encoder-Modelle mal
         vier Sentence Generators). Fassung vom 26.09.2026: ohne Titel und
         Fusszeile, Zeilen M1 bis M5 und Spalten E_default bis E_d1 ohne
         Zusatz, wie in Abbildung 6 (b). Die Zahlen sind unveraendert.

Eingabedateien: keine. Alle Zahlen stehen als Literal im Abschnitt ZAHLEN und
         stammen ausschliesslich aus
         `04 Writing/revision/fixes/Register/Zahlenregister_A_Aufloesungsergebnisse.md`
         Abschnitt 2 (die fuenfzehn Zellen E_default / E_c1 / E_c2) und
         Abschnitt 7.1 (die fuenf Zellen E_d1, Lauf vom 17.09.2026); im Repo
         entsprechen sie results/summary.csv und results/summary_E_d1.csv.
         Die Bildsprache kommt ueber das Modul `stil_kap6.py` im selben Ordner.

Aufruf:  python3 abb_5_heatmap_acc1_zwanzig_systeme.py [--out <Zielordner>]
         Vorgabe fuer --out ist 5_figures/output/.
         Feste Pfade gibt es nicht; --out ist der einzige Ortsbezug.

Datum:   2026-09-26
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import stil_kap6 as S

DATUM = "2026-09-26"
# Englische Fassung; die deutsche bleibt unter dem Namen ohne _en liegen.
NAME = "abb_5_heatmap_acc1_zwanzig_systeme_en"

# ── ZAHLEN ───────────────────────────────────────────────────────────────────
# Zahlenregister A, Abschnitt 2 (Spalten E_default, E_c1, E_c2) und
# Abschnitt 7.1 (Spalte E_d1). Bezugsmenge 75'741 Items in jeder Zelle.
MODELLE = ["M1", "M2", "M3", "M4", "M5"]
VARIANTEN = ["E_default", "E_c1", "E_c2", "E_d1"]
ACC1 = {
    "M1": {"E_default": .6167, "E_c1": .7595, "E_c2": .6811, "E_d1": .7834},
    "M2": {"E_default": .7116, "E_c1": .8015, "E_c2": .7430, "E_d1": .8101},
    "M3": {"E_default": .8210, "E_c1": .8367, "E_c2": .8037, "E_d1": .8299},
    "M4": {"E_default": .7375, "E_c1": .8369, "E_c2": .7723, "E_d1": .8358},
    "M5": {"E_default": .7829, "E_c1": .8403, "E_c2": .8128, "E_d1": .8438},
}
N_ITEMS = 75741

MODELLZEILE = {m: m for m in MODELLE}
SPALTE = {"E_default": "E$_\\mathrm{default}$", "E_c1": "E$_\\mathrm{c1}$",
          "E_c2": "E$_\\mathrm{c2}$", "E_d1": "E$_\\mathrm{d1}$"}

# Helligkeitsmonotone Folge: in Graustufen bleibt die Ordnung erhalten.
CMAP = LinearSegmentedColormap.from_list(
    "kap6", ["#f4f5f7", S.C_BASE, S.C_DIAG, S.C_BEST, "#0d1c2b"])


def zeichne(out: Path) -> None:
    S.setze_stil()
    M = np.array([[ACC1[m][v] for v in VARIANTEN] for m in MODELLE])

    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    bild = ax.imshow(M, cmap=CMAP, vmin=0.55, vmax=0.88, aspect="auto")

    for i, m in enumerate(MODELLE):
        for j, v in enumerate(VARIANTEN):
            w = M[i, j]
            hell = "#ffffff" if w > 0.775 else "#1a1a1a"
            ax.text(j, i, S.zahl(w), ha="center", va="center",
                    fontsize=8.6, color=hell,
                    fontweight="bold" if w == M.max() else "normal")

    # Beste Zelle umranden — sie wandert mit der vierten Variante auf E_d1.
    bi, bj = divmod(int(M.argmax()), M.shape[1])
    ax.add_patch(plt.Rectangle((bj - .5, bi - .5), 1, 1, fill=False,
                               edgecolor=S.C_WARN, linewidth=1.6, zorder=5))

    # Trennlinie vor der neuen Spalte.
    ax.axvline(2.5, color="#ffffff", linewidth=2.4, zorder=4)

    ax.set_xticks(range(len(VARIANTEN)))
    ax.set_xticklabels([SPALTE[v] for v in VARIANTEN], fontsize=8, linespacing=1.4)
    ax.set_yticks(range(len(MODELLE)))
    ax.set_yticklabels([MODELLZEILE[m] for m in MODELLE], fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    cb = fig.colorbar(bild, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label("Acc@1 (share of items)", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    cb.outline.set_linewidth(0.4)


    S.daten_ablegen(out, NAME,
                    ["modell", "variante", "acc1", "n_items"],
                    [[m, v, f"{ACC1[m][v]:.4f}", N_ITEMS]
                     for m in MODELLE for v in VARIANTEN], DATUM)
    S.speichern(fig, out, NAME, DATUM)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(S.AUSGABE),
                   help="Zielordner der Bilddateien (Vorgabe: 5_figures/output/)")
    zeichne(Path(p.parse_args().out))


if __name__ == "__main__":
    main()
