#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Erzeugt: Abbildung 10 — Genauigkeit auf den ambigen Toponymen ueber die feinste
         zugelassene Zellstufe (Ablation der Diskretisierung), fuer M3, M4, M5.
         Fassung vom 26.09.2026 im Stil von Kapitel 6 (stil_kap6.py) und mit
         derselben Figurbreite wie Abbildung 9, damit die Schrift bei 16 cm
         Satzbreite dieselbe Groesse hat wie in den uebrigen Abbildungen.
         Gegenueber der alten Fassung (3_evaluation/13_plot_config_ablation.py):
         Schreibweise
         «centre-point mode» ohne den Klammerzusatz, Baseline-Linie ohne Wert
         (er steht im Text), sonst gleicher Aufbau.

Eingabe: results/fig_config_ablation__data.csv — genau die Werte, die die alte
         Fassung gezeichnet hat (3_evaluation/13_plot_config_ablation.py).
         Alle fuenfzehn Zellen werden gegen die Werte von Tabelle 12 geprueft;
         bei Abweichung bricht das Skript ab.

Aufruf:  python3 abb_10_aufloesungsreihe.py [--wurzel RESULTS_ORDNER] [--out ZIELORDNER]
Datum:   2026-09-26
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D

import stil_kap6 as S

DATUM = "2026-09-26"
NAME = "abb_10_aufloesungsreihe_en"
N_AMBIG = 23848

# Tabelle 12 der Arbeit (Kontrollwerte; Register A / Anhang E).
TABELLE_12 = {
    ("M3", "E_c2"): .727, ("M3", "E_c5"): .776, ("M3", "E_c3"): .805, ("M3", "E_c4"): .785, ("M3", "E_c1"): .832,
    ("M4", "E_c2"): .627, ("M4", "E_c5"): .725, ("M4", "E_c3"): .818, ("M4", "E_c4"): .796, ("M4", "E_c1"): .832,
    ("M5", "E_c2"): .756, ("M5", "E_c5"): .803, ("M5", "E_c3"): .845, ("M5", "E_c4"): .826, ("M5", "E_c1"): .843,
}
BASELINE_SOLL = .782

FARBE = {"M3": "#9a9a9a", "M4": S.C_DIAG, "M5": S.C_BEST}
MARKER = {"M3": "o", "M4": "s", "M5": "^"}
STUFEN = {10: "≈15’000 m²", 11: "≈2’150 m²", 12: "≈307 m²", 13: "≈44 m²"}


def lesen(pfad: Path):
    zellen, baseline = {}, None
    with open(pfad, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["arm"] == "E_default":
                baseline = float(r["acc1_ambiguous_only"])
                continue
            zellen[(r["model"], r["arm"])] = (r["containment_mode"], int(r["max_resolution"]),
                                              float(r["acc1_ambiguous_only"]))
    if baseline is None or len(zellen) != 15:
        raise SystemExit(f"{pfad.name}: erwartet 15 Zellen und eine Baseline")
    for k, (_, _, w) in zellen.items():
        if round(w, 3) != TABELLE_12[k]:
            raise SystemExit(f"{k}: {w:.4f} weicht von Tabelle 12 ({TABELLE_12[k]}) ab")
    if round(baseline, 3) != BASELINE_SOLL:
        raise SystemExit(f"Baseline {baseline:.4f} weicht von {BASELINE_SOLL} ab")
    return zellen, baseline


def zeichne(wurzel: Path, out: Path) -> None:
    S.setze_stil()
    zellen, base = lesen(wurzel / "fig_config_ablation__data.csv")

    fig, ax = plt.subplots(figsize=(6.3, 3.9))
    S.style(ax, "y")
    ax.axhline(base, color="#9a9a9a", lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax.text(9.72, base + 0.004, "fair baseline M3/E$_\\mathrm{default}$",
            color="#5a5a5a", fontsize=7.2, va="bottom")

    zeilen = []
    for m in ("M3", "M4", "M5"):
        for modus, ls, voll in (("overlap", "-", True), ("center", "--", False)):
            punkte = sorted((res, w, arm) for (mm, arm), (md, res, w) in zellen.items()
                            if mm == m and md == modus)
            ax.plot([p[0] for p in punkte], [p[1] for p in punkte], color=FARBE[m],
                    lw=1.6 if voll else 1.2, ls=ls, marker=MARKER[m], ms=5.5,
                    mfc=FARBE[m] if voll else "white", mec=FARBE[m], mew=1.1, zorder=3)
            zeilen += [[m, arm, modus, res, f"{w:.4f}"] for res, w, arm in punkte]
    # Direktbeschriftung am rechten Ende der Ueberlagerungslinie.
    versatz = {"M3": -0.011, "M4": +0.001, "M5": +0.006}
    for m in ("M3", "M4", "M5"):
        ax.text(13.12, zellen[(m, "E_c1")][2] + versatz[m], m, color=FARBE[m],
                fontsize=8, fontweight="bold", va="center")

    ax.set_xlim(9.7, 13.45)
    ax.set_xticks(sorted(STUFEN))
    ax.set_xticklabels([f"{s}\n{STUFEN[s]}" for s in sorted(STUFEN)])
    ax.set_xlabel("Finest permitted cell resolution (H3), mean cell area")
    ax.set_ylabel(f"Acc@1 on {S.tausender(N_AMBIG)} ambiguous toponyms")
    ax.set_ylim(0.60, 0.87)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: S.zahl(v, 2)))
    zeichen = [Line2D([], [], color="#333333", lw=1.6, ls="-", marker="o", ms=4.5,
                      mfc="#333333", label="overlap mode"),
               Line2D([], [], color="#333333", lw=1.2, ls="--", marker="o", ms=4.5,
                      mfc="white", label="centre-point mode")]
    zeichen += [Line2D([], [], color=FARBE[m], lw=0, marker=MARKER[m], ms=5, label=m)
                for m in ("M3", "M4", "M5")]
    ax.legend(handles=zeichen, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=5,
              handlelength=2.2, columnspacing=1.4, borderaxespad=0.0, fontsize=7.5)

    zeilen.append(["M3", "E_default", "—", "—", f"{base:.4f}"])
    S.daten_ablegen(out, NAME, ["model", "arm", "containment_mode", "max_resolution",
                                "acc1_ambiguous_only"], zeilen, DATUM)
    S.speichern(fig, out, NAME, DATUM)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wurzel", type=Path, default=S.RESULTS,
                   help="Ordner mit fig_config_ablation__data.csv (Default: experiment-2/results)")
    p.add_argument("--out", type=Path, default=S.AUSGABE)
    a = p.parse_args()
    zeichne(a.wurzel, a.out)


if __name__ == "__main__":
    main()
