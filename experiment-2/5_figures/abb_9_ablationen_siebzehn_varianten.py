#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Erzeugt: Abbildung 9 (Ablationen der Kandidatenbeschreibung) auf der Leitmetrik,
         also Acc@1 ueber die 23'848 ambigen Toponyme. Fassung vom 26.09.2026:
         SIEBZEHN Varianten (neu die dritte Kontrollvariante Kategorienkappung 10
         auf B1, wertgleich mit der Basis), sprechende englische Balkenlabels
         statt der Maschinennamen, ohne Titel, ohne die Marke «(RQ2)».
         Vorlage: abb_9_ablationen_leitmetrik.py (Zahlen unveraendert).

Rechnung: Acc@1 ambig = (round(acc1 * 75'741) - 43'544) / 23'848. Der Sockel der
         43'544 korrekt geloesten eindeutigen Toponyme ist in jeder Variante gleich
         (Kandidatenmenge identisch); dieselbe Nachrechnung verwendet Anhang D.
         Die Gesamtwerte acc1 stammen unveraendert aus dem Schwesterskript
         (Register A, Abschnitte 2.1/2.2). Kontrolle: alle sechzehn Werte stimmen
         mit Tabelle 12 (vierstellig) ueberein, sonst bricht das Skript ab.

Aufruf:  python3 abb_9_ablationen_siebzehn_varianten.py [--out <Zielordner>]
Datum:   2026-09-26 (Fassung siebzehn Varianten)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib, matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import stil_kap6 as S

DATUM = "2026-09-26"
NAME = "abb_9_ablationen_siebzehn_varianten_leitmetrik_en"
N_ITEMS, N_AMBIG, SOCKEL = 75741, 23848, 43544
BASIS_GESAMT = .840298          # M5/E_c1
BASELINE_FAIR_GESAMT = .821008  # M3/E_default

def ambig(acc1: float) -> float:
    return (round(acc1 * N_ITEMS) - SOCKEL) / N_AMBIG

BASIS = ambig(BASIS_GESAMT)                 # .8429
BASELINE_FAIR = ambig(BASELINE_FAIR_GESAMT) # .7816

# (Schluessel, Beschriftung, acc1 gesamt, Mass, Bemerkung, Tabelle-12-Wert ambig)
VARIANTEN = [
    ("no_dynamic",   .821167, "B1", "",        .7821),
    ("no_static",    .834053, "B1", "",        .8230),
    ("uniform_b1",   .827504, "B1", "RQ2",     .8022),
    ("assoc_0.001",  .840298, "B1", "control", .8429),
    ("assoc_0.01",   .843004, "B1", "",        .8515),
    ("assoc_0.1",    .822355, "B1", "",        .7859),
    ("max_slots_10", .840298, "B1", "control", .8429),
    ("max_slots_5",  .837552, "B1", "",        .8342),
    ("max_slots_3",  .835122, "B1", "",        .8264),
    ("E_b1_c10",     .840298, "B1", "control", .8429),
    ("E_b1_c6",      .834832, "B1", "",        .8255),
    ("E_b1_c20",     .835928, "B1", "",        .8290),
    ("E_d1_c6",      .831109, "D1", "",        .8137),
    ("E_d1_c20",     .849646, "D1", "",        .8726),
    ("E_d1",         .843849, "D1", "",        .8542),
    ("E_d1_s01",     .845645, "D1", "",        .8599),
    ("E_d1_s1",      .844061, "D1", "",        .8548),
]

# Sprechende Labels (englisch wie alle Abbildungen); die deutschen Namen stehen
# in Tabelle 11 bzw. Anhang D.
LABEL = {
    "no_dynamic":   "B₁ · no dynamic context",
    "no_static":    "B₁ · no static context",
    "uniform_b1":   "matrix of ones (no association)",
    "assoc_0.001":  "B₁ · threshold 0.001",
    "assoc_0.01":   "B₁ · threshold 0.01",
    "assoc_0.1":    "B₁ · threshold 0.1",
    "max_slots_10": "B₁ · 10 slots",
    "max_slots_5":  "B₁ · 5 slots",
    "max_slots_3":  "B₁ · 3 slots",
    "E_b1_c10":     "B₁ · cap 10 categories",
    "E_b1_c6":      "B₁ · cap 6 categories",
    "E_b1_c20":     "B₁ · cap 20 categories",
    "E_d1_c6":      "D₁ · cap 6 categories",
    "E_d1_c20":     "D₁ · cap 20 categories",
    "E_d1":         "D₁ · cap 10 categories",
    "E_d1_s01":     "D₁ · threshold 0.01",
    "E_d1_s1":      "D₁ · threshold 0.1",
}


def zeichne(out: Path) -> None:
    S.setze_stil()
    posten = []
    for k, acc1, mass, bem, soll in VARIANTEN:
        a = ambig(acc1)
        assert abs(round(a, 4) - soll) < 1e-9, (k, round(a, 4), soll)
        posten.append((k, a, a - BASIS, mass, bem))
    assert abs(round(BASIS, 4) - .8429) < 1e-9 and abs(round(BASELINE_FAIR, 4) - .7816) < 1e-9
    posten.sort(key=lambda t: t[2])

    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    farben = {"B1": S.C_DIAG, "D1": S.C_BEST}
    schraff = {"B1": None, "D1": "///"}
    for i, (k, a, d, mass, bem) in enumerate(posten):
        col = S.C_WARN if k == "uniform_b1" else farben[mass]
        ax.barh(i, d, color=col, height=0.62, zorder=3, hatch=schraff[mass],
                edgecolor="white", linewidth=0.4)
        off = -0.0011 if d < 0 else 0.0011
        ax.text(d + off, i, S.zahl(a), va="center",
                ha="right" if d < 0 else "left", fontsize=7.3, zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
    ax.set_yticks(range(len(posten)))
    ax.set_yticklabels([LABEL[k] + ("  (control)" if bem == "control" else "")
                        for k, *_ , bem in posten], fontsize=7.6)
    ax.axvline(0, color="#4a4a4a", linewidth=0.7, zorder=4)
    d_fair = BASELINE_FAIR - BASIS
    ax.axvline(d_fair, color=S.C_LINE, linewidth=0.9, linestyle=(0, (4, 2)), zorder=4)
    ax.text(d_fair + 0.0015, len(posten) - 0.55, "fair baseline M3/E$_\\mathrm{default}$",
            fontsize=7.0, ha="left", va="top", rotation=90, color=S.C_LINE)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: S.zahl(v, 3)))
    ax.set_xlabel("$\\Delta$ Acc@1 on ambiguous toponyms vs. ablation base M5/E$_\\mathrm{c1}$ "
                  f"({S.zahl(BASIS)})\nat the end of each bar: Acc@1 (ambiguous) of the variant")
    lo = min(p[2] for p in posten) - 0.017
    hi = max(p[2] for p in posten) + 0.014
    ax.set_xlim(min(lo, d_fair - 0.007), hi)
    ax.set_ylim(-0.72, len(posten) - 0.35)
    S.style(ax, "x")
    ax.legend(handles=[
        Patch(facecolor=S.C_DIAG, label="B$_1$ — footprint overlap"),
        Patch(facecolor=S.C_WARN, label="matrix of ones — no selection by association"),
        Patch(facecolor=S.C_BEST, hatch="///", edgecolor="white", label="D$_1$ — ring adjacency")],
        loc="upper center", bbox_to_anchor=(0.5, -0.185), ncol=1, handlelength=1.3, fontsize=7.4)
    S.daten_ablegen(out, NAME,
                    ["variante", "mass", "acc1_gesamt_voll", "acc1_ambig", "delta_ambig_gegen_basis"],
                    [[k, mass, f"{dict((v[0], v[1]) for v in VARIANTEN)[k]:.6f}", f"{a:.4f}", f"{d:+.4f}"]
                     for k, a, d, mass, _ in posten]
                    + [["baseline_M5_E_c1", "B1", f"{BASIS_GESAMT:.6f}", f"{BASIS:.4f}", "+0.0000"],
                       ["baseline_M3_E_default", "-", f"{BASELINE_FAIR_GESAMT:.6f}", f"{BASELINE_FAIR:.4f}", f"{d_fair:+.4f}"]],
                    DATUM)
    S.speichern(fig, out, NAME, DATUM)

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(S.AUSGABE))
    zeichne(Path(p.parse_args().out))

if __name__ == "__main__":
    main()
