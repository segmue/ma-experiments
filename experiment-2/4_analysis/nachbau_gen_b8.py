#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nachbau_gen_b8.py — Nachbau des verlorenen Skripts `$HOME/luecken/gen_b8.py`
(Zahlenregister B, Abschnitt 8, Nachtrag 18.09.2026: Verteilung des Jaccard der
Zehnermengen). Das Original ist nicht auffindbar (MA-Ordner, Zips, Vault-Git geprüft
am 29.09.2026). Laut Register rechnete es «nach derselben Vorschrift wie
neurechnung_kennzahlen_b1_d1.py»; diese Vorschrift ist hier übernommen.

Rechenweg
  Zehnermenge einer Zeile (Kategorie) = Partner ohne Selbstbezug, erst Schwellwert
  >= 0.001, dann die zehn stärksten (Lesart «mit Schwelle»); Lesart «ohne Schwelle»:
  nur Kappung auf zehn. Jaccard je Zeile = |B1 ∩ D1| / |B1 ∪ D1| der beiden
  Zehnermengen. Kennzahlen über die 110 Zeilen: Median, erstes/drittes Quartil
  (lineare Interpolation, numpy-Standard), Minimum, Maximum samt Kategorie,
  Zahl der Zeilen ohne gemeinsamen Partner, Zahl der Zeilen mit gleichem
  stärkstem Partner.

Eingaben (Standard = versionierte Matrizen in ma-experiments/matrices/config1/)
  b1_matrix.csv  (Überlagerungsmass B1, config1; NICHT npmi_matrix.csv)
  d1_matrix.csv  (Adjazenzmass D1, Lauf T-141 = npmi_dist_matrix_D1.csv)
  Abweichend per --b1 / --d1.
Ausgabe
  <exp2>/4_analysis/out/nachbau_gen_b8_check.csv (nur Aggregatzahlen)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCHWELLE, KAPPUNG = 0.001, 10
T = "chapters/06_results.tex:157"


def lesen(p: Path) -> pd.DataFrame:
    m = pd.read_csv(p, sep=";", index_col=0)
    m.index = [str(i) for i in m.index]
    m.columns = [str(c) for c in m.columns]
    assert list(m.index) == list(m.columns) and not m.isna().any().any()
    return m


def zehnermengen(m: pd.DataFrame, mit_schwelle: bool) -> dict:
    aus = {}
    for kat in m.index:
        s = m.loc[kat].drop(labels=[kat])
        if mit_schwelle:
            s = s[s >= SCHWELLE]
        aus[kat] = [str(k) for k in s.sort_values(ascending=False).head(KAPPUNG).index]
    return aus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1", type=Path, default=REPO / "matrices" / "config1" / "b1_matrix.csv")
    ap.add_argument("--d1", type=Path, default=REPO / "matrices" / "config1" / "d1_matrix.csv")
    ap.add_argument("--out", type=Path, default=HERE / "out" / "nachbau_gen_b8_check.csv")
    a = ap.parse_args()
    b1, d1 = lesen(a.b1), lesen(a.d1)
    assert list(b1.index) == list(d1.index) and b1.shape == (110, 110)

    rows = []

    def add(rid, b, reg, th, neu, g):
        rows.append({"register_id": rid, "beschreibung": b, "wert_register": reg,
                     "wert_thesis": th, "wert_nachgerechnet": neu, "gleich": "ja" if g else "nein"})

    for mit in (True, False):
        zb, zd = zehnermengen(b1, mit), zehnermengen(d1, mit)
        jac, gem, uni, gleich, ohne = {}, {}, {}, 0, 0
        for k in b1.index:
            X, Y = set(zb[k]), set(zd[k])
            jac[k] = len(X & Y) / len(X | Y)
            gem[k], uni[k] = len(X & Y), len(X | Y)
            gleich += int(zb[k][0] == zd[k][0])
            ohne += int(not (X & Y))
        j = pd.Series(jac)
        q1, med, q3 = (float(np.quantile(j.to_numpy(), q)) for q in (.25, .5, .75))
        kmax = j.idxmax()
        les = "mit Schwelle 0.001" if mit else "ohne Schwelle"
        add("B 8 (Nachtrag 18.09.)", f"Jaccard Zehnermengen, erstes Quartil ({les})", "0.1111",
            ("0.1111 (" + T + ")") if mit else "nicht in der Arbeit", f"{q1:.6f}", f"{q1:.4f}" == "0.1111")
        add("B 8", f"Jaccard Zehnermengen, Median ({les})", "0.1765", ("0.1765 (" + T + ")") if mit else "nicht in der Arbeit",
            f"{med:.6f}", f"{med:.4f}" == "0.1765")
        add("B 8 (Nachtrag 18.09.)", f"Jaccard Zehnermengen, drittes Quartil ({les})", "0.2500",
            ("0.2500 (" + T + ")") if mit else "nicht in der Arbeit", f"{q3:.6f}", f"{q3:.4f}" == "0.2500")
        add("B 8 (Nachtrag 18.09.)", f"Jaccard Zehnermengen, Maximum ({les})",
            "0.5385 (Haltestelle Schiff, 7 von 13)", "nicht in der Arbeit",
            f"{j.max():.6f} ({kmax}, {gem[kmax]} von {uni[kmax]})",
            f"{j.max():.4f}" == "0.5385" and kmax == "Haltestelle Schiff" and (gem[kmax], uni[kmax]) == (7, 13))
        add("B 8 (Nachtrag 18.09.)", f"Jaccard Zehnermengen, Minimum ({les})", "0.0000",
            "nicht in der Arbeit", f"{j.min():.6f}", f"{j.min():.4f}" == "0.0000")
        soll = 11 if mit else 10
        add("B 8", f"Zeilen ohne gemeinsamen Partner ({les})", f"{soll} von 110",
            ("eleven of the 110 rows (" + T + ")") if mit else "nicht in der Arbeit",
            f"{ohne} von 110", ohne == soll)
        if mit:
            add("B 8", "Zeilen mit gleichem stärkstem Partner (mit Schwelle)", "19 von 110",
                "19 of the 110 rows, 91 change (" + T + ")", f"{gleich} von 110", gleich == 19)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["register_id", "beschreibung", "wert_register",
                                           "wert_thesis", "wert_nachgerechnet", "gleich"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"geschrieben: {a.out}  ({sum(r['gleich'] == 'ja' for r in rows)} von {len(rows)} gleich)")
    for r in rows:
        print(" ", r["gleich"], r["beschreibung"], "->", r["wert_nachgerechnet"])


if __name__ == "__main__":
    main()
