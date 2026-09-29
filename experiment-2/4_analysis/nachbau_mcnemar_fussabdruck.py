#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nachbau_mcnemar_fussabdruck.py — Nachbau des verlorenen Skripts
`$HOME/ausw_ed1/mcnemar_fussabdruck.py` (Zahlenregister A, Abschnitt 7.4 und
Lücke 12.8). Das Original ist nicht auffindbar (MA-Ordner, Zips, Vault-Git
geprüft am 29.09.2026); dieser Nachbau folgt der Verfahrensbeschreibung im
gelöschten Bericht `AUSWERTUNG_E_d1.md` (Vault-Commit 1c7c507^), Abschnitt 4.

Verfahren
  1. Gepaarter exakter McNemar-Test: zweiseitige Binomialprobe (p0 = 0.5) auf den
     diskordanten Items zweier Zellen desselben Modells (korrekt = rank == 1).
  2. Kippersaldo = (nur b korrekt) − (nur a korrekt), a = Ausgangsarm, b = neuer Arm.
  3. Zerlegung des Saldos nach Fussabdruckklasse der Gold-Kategorie:
     relative Fläche je Kategorie aus w = B1/NPMI auf config1 (Zeile 0,
     w/(1−w) = p_j/p_0), obere Hälfte der 110 Kategorien (55) = «grossflächig»,
     Rest = «punktförmig» (Regel aus B2_Gegenprobe.py, Z. 89–101).
  4. Gold-Kategorie = Kategoriefeld der Kandidatenbeschreibung des Gold-Eintrags
     in descriptions_E_c1.pkl (erstes Komma-Feld nach dem Namen, das eine der
     110 Matrixkategorien ist).

Eingaben (Standard = Orte, an denen die Pipeline von ma-experiments sie erzeugt)
  <exp2>/results/matrix/per_item/<Modell>_<Arm>.pkl.gz   Per-Item-Dumps (Stage 02)
  <exp2>/3_evaluation/cache/descriptions_E_c1.pkl         Kandidatenbeschreibungen
  <repo>/output/config1/b1_matrix.csv, npmi_matrix.csv   B1 und rohes NPMI (config1)
  Abweichende Orte per --zellen DIR [DIR ...], --beschreibungen DATEI, --matrixdir DIR.

Ausgabe
  <exp2>/4_analysis/out/nachbau_mcnemar_fussabdruck_check.csv  (nur Aggregatzahlen)

Aufruf
  python3 nachbau_mcnemar_fussabdruck.py [--zellen D1 D2 ...] [--beschreibungen F]
                                         [--matrixdir D] [--out F]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import math
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent          # .../experiment-2/4_analysis
EXP2 = HERE.parent
REPO = EXP2.parent

M = {"M1": "M1_dguzh", "M2": "M2_distiluse_base", "M3": "M3_default_finetuned",
     "M4": "M4_spatial_config1", "M5": "M5_spatial_config2"}

# ---------------------------------------------------------------- Vergleichswerte
# Register A, Abschnitt 7.1 und 7.4 (Stand 29.09.2026) sowie thesis_en.
# Schlüssel: (Modell, Arm a, Arm b). Werte: nur_a, nur_b, p, netto, gross, punkt.
REG = {
    ("M1", "E_c1", "E_d1"): (1861, 3666, "1.92e-132", 1805, 373, 1432),
    ("M2", "E_c1", "E_d1"): (1129, 1783, "5.58e-34", 654, 526, 128),
    ("M3", "E_c1", "E_d1"): (1430, 916, "2.07e-26", -514, 116, -630),
    ("M4", "E_c1", "E_d1"): (980, 900, "6.84e-02", -80, 119, -199),
    ("M5", "E_c1", "E_d1"): (684, 953, "3.17e-11", 269, 350, -81),
    ("M5", "E_c1", "E_b1_c6"): (857, 443, "7.06e-31", -414, 136, -550),   # Eichung K2 gegen K
    ("M5", "E_d1", "E_d1_c6"): (1384, 419, "3.69e-120", -965, -130, -835),
    ("M5", "E_d1", "E_d1_c20"): (429, 868, "1.13e-34", 439, 183, 256),
    ("M5", "E_c1", "E_b1_c20"): (871, 540, "1.09e-18", -331, -126, -205),
    ("M5", "E_b1_c6", "E_d1_c6"): (1567, 1285, "1.40e-07", -282, 84, -366),
    ("M5", "E_b1_c20", "E_d1_c20"): (327, 1366, "9.99e-151", 1039, 659, 380),
    ("M5", "E_d1", "E_d1_s01"): (30, 166, "5.15e-24", 136, 1, 135),
    ("M5", "E_d1", "E_d1_s1"): (322, 338, "5.59e-01", 16, -172, 188),
    ("M4", "E_d1", "E_d1_c6"): (1761, 331, "6.09e-235", -1430, -180, -1250),
    ("M4", "E_d1", "E_d1_c20"): (377, 1334, "3.99e-125", 957, 219, 738),
    ("M4", "E_c1", "E_b1_c6"): (1356, 469, "1.30e-99", -887, -26, -861),
    ("M4", "E_c1", "E_b1_c20"): (883, 687, "8.33e-07", -196, -344, 148),
    ("M4", "E_b1_c6", "E_d1_c6"): (2013, 1390, "1.11e-26", -623, -35, -588),
    ("M4", "E_b1_c20", "E_d1_c20"): (264, 1337, "1.47e-172", 1073, 682, 391),
    ("M4", "E_d1", "E_d1_s01"): (8, 127, "1.08e-28", 119, 5, 114),
    ("M4", "E_d1", "E_d1_s1"): (207, 314, "3.17e-06", 107, -40, 147),
}
# Werte, die in thesis_en stehen (Kap. 6, Abschnitt 6.3.3 und Tabelle 9):
# chapters/06_results.tex, Z. 254 und Z. 276/282.
THESIS = {
    ("M5", "E_c1", "E_d1"): {"nur_a": "684", "nur_b": "953", "p": "3.2e-11"},
    ("M4", "E_c1", "E_d1"): {"nur_a": "980", "nur_b": "900", "p": "6.8e-02"},
}
# Einzelposten je Gold-Kategorie (Register A 7.4 bzw. AUSWERTUNG 4.1/4.3)
REG_KAT = {
    ("M5", "E_c1", "E_b1_c6", "Hauptgipfel"): -405,
    ("M5", "E_c1", "E_b1_c6", "Gipfel"): -215,
    ("M5", "E_c1", "E_b1_c6", "Grat"): 82,
    ("M5", "E_d1", "E_d1_c6", "Alpiner Gipfel"): -1044,
    ("M4", "E_d1", "E_d1_c6", "Alpiner Gipfel"): -1348,
    ("M5", "E_d1", "E_d1_s01", "Hauptgipfel"): 125,
    ("M5", "E_d1", "E_d1_s01", "Gipfel"): 6,
    ("M5", "E_d1", "E_d1_s01", "Alpiner Gipfel"): 4,
    ("M5", "E_d1", "E_d1_s01", "Skilift"): 3,
    ("M5", "E_d1", "E_d1_s01", "Flurname swisstopo"): -2,
    ("M4", "E_d1", "E_d1_s01", "Hauptgipfel"): 110,
    ("M5", "E_c1", "E_d1", "Hauptgipfel"): -207,
}
# Zusatzvergleich aus AUSWERTUNG 4.1 (nicht im Register A 7.4): D1x20 gegen B1x10
REG_EXTRA = {("M5", "E_c1", "E_d1_c20"): (None, None, None, 708, 533, 175)}


# ---------------------------------------------------------------- Rechnen
def exact_mcnemar_p(b: int, c: int) -> float:
    """Zweiseitiger exakter Binomialtest (p0 = 0.5), wie 12_mcnemar_main.py."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            + n * math.log(0.5) for i in range(k + 1)]
    m = max(logs)
    return min(1.0, 2.0 * math.exp(m) * sum(math.exp(x - m) for x in logs))


def finde(dirs: list[Path], name: str) -> Path:
    for d in dirs:
        for p in (d / name, d / "per_item" / name):
            if p.is_file():
                return p
    raise SystemExit(f"fehlt: {name} in {[str(d) for d in dirs]}")


def lade_zelle(dirs, modell, arm):
    items = pickle.load(gzip.open(finde(dirs, f"{M[modell]}_{arm}.pkl.gz"), "rb"))
    return items


def fussabdruckklassen(matrixdir: Path):
    npmi = pd.read_csv(matrixdir / "npmi_matrix.csv", sep=";", index_col=0)
    b1 = pd.read_csv(matrixdir / "b1_matrix.csv", sep=";", index_col=0)
    assert list(npmi.index) == list(b1.index)
    cats = list(b1.index)
    n1, bb = npmi.to_numpy(float), b1.to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = bb / n1
    w[np.abs(n1) < 1e-12] = np.nan
    np.fill_diagonal(w, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = w / (1.0 - w)
    p = ratio[0, :].copy()
    p[0] = 1.0
    ra = pd.Series(p, index=cats)
    ra = ra / ra.max()
    gross = set(ra.sort_values(ascending=False).head(len(cats) // 2).index)
    return cats, gross


def gold_kategorie(beschreibungen: dict, cats: set):
    cache = {}

    def f(gid):
        if gid not in cache:
            txt = beschreibungen.get(gid)
            kat = None
            if txt is not None:
                for teil in txt.split(", ")[1:]:
                    if teil in cats:
                        kat = teil
                        break
            cache[gid] = kat
        return cache[gid]
    return f


def vergleich(a_items, b_items, gkat, gross):
    assert len(a_items) == len(b_items)
    nur_a = nur_b = 0
    saldo = Counter()
    kat_saldo = Counter()
    alle_ambig = True
    ohne_kat = 0
    for x, y in zip(a_items, b_items):
        assert (x["doc_id"], x["start"], x["end"], x["gold_id"]) == \
               (y["doc_id"], y["start"], y["end"], y["gold_id"])
        ca, cb = x["rank"] == 1, y["rank"] == 1
        if ca == cb:
            continue
        s = 1 if cb else -1
        if cb:
            nur_b += 1
        else:
            nur_a += 1
        if x["n_candidates"] <= 1:
            alle_ambig = False
        k = gkat(x["gold_id"])
        if k is None:
            ohne_kat += 1
        saldo["netto"] += s
        saldo["gross" if k in gross else "punkt"] += s
        kat_saldo[k] += s
    return {"nur_a": nur_a, "nur_b": nur_b, "p": exact_mcnemar_p(nur_a, nur_b),
            "netto": saldo["netto"], "gross": saldo["gross"], "punkt": saldo["punkt"],
            "kat": kat_saldo, "alle_ambig": alle_ambig, "ohne_kat": ohne_kat}


def p_fmt(p: float, stellen: int) -> str:
    return f"{p:.{stellen - 1}e}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--zellen", nargs="+", type=Path,
                    default=[EXP2 / "results" / "matrix"])
    ap.add_argument("--beschreibungen", type=Path,
                    default=EXP2 / "3_evaluation" / "cache" / "descriptions_E_c1.pkl")
    ap.add_argument("--matrixdir", type=Path, default=REPO / "output" / "config1")
    ap.add_argument("--out", type=Path,
                    default=HERE / "out" / "nachbau_mcnemar_fussabdruck_check.csv")
    a = ap.parse_args()

    cats, gross = fussabdruckklassen(a.matrixdir)
    gkat = gold_kategorie(pickle.load(open(a.beschreibungen, "rb")), set(cats))
    print(f"Fussabdruckklassen: {len(gross)} grossflächig von {len(cats)}")

    zellen = {}

    def z(mod, arm):
        if (mod, arm) not in zellen:
            zellen[(mod, arm)] = lade_zelle(a.zellen, mod, arm)
        return zellen[(mod, arm)]

    rows = []

    def add(rid, besch, reg, thesis, neu, gleich):
        rows.append({"register_id": rid, "beschreibung": besch, "wert_register": reg,
                     "wert_thesis": thesis, "wert_nachgerechnet": neu,
                     "gleich": gleich})

    alle = dict(REG)
    alle.update(REG_EXTRA)
    for key, soll in alle.items():
        mod, arm_a, arm_b = key
        r = vergleich(z(mod, arm_a), z(mod, arm_b), gkat, gross)
        rid = "A 7.1/7.4" if key in REG else "AUSWERTUNG_E_d1 4.1 (nicht im Register)"
        lab = f"{mod}: {arm_a} gegen {arm_b}"
        th = THESIS.get(key, {})
        felder = [("nur_a", f"nur {arm_a} korrekt", str(r["nur_a"])),
                  ("nur_b", f"nur {arm_b} korrekt", str(r["nur_b"])),
                  ("p", "McNemar p exakt zweiseitig", p_fmt(r["p"], 3)),
                  ("netto", "Kippersaldo netto", f"{r['netto']:+d}"),
                  ("gross", "Kippersaldo grossflächig", f"{r['gross']:+d}"),
                  ("punkt", "Kippersaldo punktförmig", f"{r['punkt']:+d}")]
        for i, (feld, besch, neu) in enumerate(felder):
            sw = soll[i]
            if sw is None:
                continue
            if feld == "p":
                reg_s = sw
                g_reg = p_fmt(r["p"], 3) == p_fmt(float(sw), 3)
            else:
                reg_s = f"{sw:+d}" if feld in ("netto", "gross", "punkt") else str(sw)
                g_reg = int(neu) == int(sw)
            t = th.get(feld)
            if t is None:
                t_s, g_t = "nicht in der Arbeit", True
            elif feld == "p":
                t_s, g_t = t, p_fmt(r["p"], 2) == p_fmt(float(t), 2)
            else:
                t_s, g_t = t, int(neu) == int(t)
            add(rid, f"{lab}; {besch}", reg_s, t_s, neu, "ja" if (g_reg and g_t) else "nein")
        add(rid, f"{lab}; alle diskordanten Items ambig", "True", "nicht in der Arbeit",
            str(r["alle_ambig"]), "ja" if r["alle_ambig"] else "nein")
        if r["ohne_kat"]:
            print(f"  Hinweis {lab}: {r['ohne_kat']} diskordante Items ohne Gold-Kategorie")
        for (km, ka, kb, kat), sw in REG_KAT.items():
            if (km, ka, kb) == key:
                neu = r["kat"].get(kat, 0)
                add("A 7.4 / AUSWERTUNG 4", f"{lab}; Kippersaldo Gold-Kategorie {kat}",
                    f"{sw:+d}", "nicht in der Arbeit", f"{neu:+d}",
                    "ja" if neu == sw else "nein")
        print(f"{lab:36s} {r['nur_a']:5d}/{r['nur_b']:5d} p={r['p']:.3g} "
              f"netto {r['netto']:+d} gross {r['gross']:+d} punkt {r['punkt']:+d}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, lineterminator="\n", fieldnames=["register_id", "beschreibung", "wert_register",
                                          "wert_thesis", "wert_nachgerechnet", "gleich"])
        w.writeheader()
        w.writerows(rows)
    n_ja = sum(r["gleich"] == "ja" for r in rows)
    print(f"geschrieben: {a.out}  ({n_ja} von {len(rows)} gleich)")


if __name__ == "__main__":
    main()
