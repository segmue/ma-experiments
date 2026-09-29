#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nachbau_ablationen_ambig.py — Nachbau der verlorenen Rechnung `$HOME/abl/calc.py`
(K-14_N-3_Ablationen_ambig.md, im Vault am 21.09.2026 gelöscht, Commit 1c7c507):
Acc@1 und Acc@3 der Ablationsvarianten auf den 23'848 ambigen Toponymen.
Das Original ist nicht auffindbar (MA-Ordner, Zips, Vault-Git geprüft am 29.09.2026).

Rechenweg
  Ablationen vom 12.09.2026 (04_ablations.py schreibt nur Aggregate, keine Per-Item-Dumps):
    korrekt_gesamt(v) = round(acc(v) * 75'741)
    acc_ambig(v)      = (korrekt_gesamt(v) - 43'544) / 23'848        (acc = Acc@1 oder Acc@3)
  Der Sockel 43'544 (eindeutige Toponyme, n_candidates == 1, korrekt) ist in jeder
  Variante gleich, weil die Kandidatenmenge nur vom Oberflächentext abhängt und bei
  einem Kandidaten rank == 1 unabhängig vom Encoder ist; bei n == 1 ist rank <= 3
  dasselbe wie rank == 1, also gilt derselbe Sockel für Acc@3. Der Sockel wird hier aus
  den Per-Item-Dumps nachgezählt, nicht vorausgesetzt.
  Arme des E_d1-Laufs (17.09.2026, 02_evaluate_matrix.py mit Per-Item-Dumps):
    acc_ambig direkt aus den Per-Item-Dumps gezählt (Items mit n_candidates > 1),
    zusätzlich gegen die Sockelformel gekreuzt.
  Deltas: gegen die Ablationsbasis E_c1 desselben Modells bzw. innerhalb D1 gegen E_d1.

Eingaben (Standard = Ablageort der Pipeline in ma-experiments, config.py)
  <exp2>/results/ablations.json                 M5 (04_ablations.py, Default)
  <exp2>/results/ablations_M4.json              M4 (04_ablations.py --out ablations_M4.json;
                                                im Altbestand: ablations_M4_backup.json)
  <exp2>/results/matrix/per_item/<Modell>_<SG>.pkl.gz   E_c1, E_default, E_d1-Arme
  Abweichend per --abl-m5, --abl-m4, --zellen DIR [DIR ...].
Ausgabe
  <exp2>/4_analysis/out/nachbau_ablationen_ambig_check.csv (nur Aggregatzahlen)
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP2 = HERE.parent
N_ITEMS, N_AMBIG = 75741, 23848
M = {"M3": "M3_default_finetuned", "M4": "M4_spatial_config1", "M5": "M5_spatial_config2"}
T, A, FIG = "chapters/06_results.tex", "appendix/appendix.tex", "Abb. 9 (Bildwert laut abb_9_ablationen_siebzehn_varianten.py)"

# Variante (Anhangbeschriftung) -> Quelle: ("abl", Schlüssel) oder ("zelle", Sentence Generator)
VAR = [
    ("Category cap 6 / B1", ("zelle", "E_b1_c6")), ("Category cap 6 / D1", ("zelle", "E_d1_c6")),
    ("Category cap 10 / B1", ("zelle", "E_c1")), ("Category cap 10 / D1", ("zelle", "E_d1")),
    ("Category cap 20 / B1", ("zelle", "E_b1_c20")), ("Category cap 20 / D1", ("zelle", "E_d1_c20")),
    ("Threshold 0.001 / B1", ("abl", "assoc_0.001")), ("Threshold 0.01 / B1", ("abl", "assoc_0.01")),
    ("Threshold 0.01 / D1", ("zelle", "E_d1_s01")), ("Threshold 0.1 / B1", ("abl", "assoc_0.1")),
    ("Threshold 0.1 / D1", ("zelle", "E_d1_s1")), ("Slot limit 10 / B1", ("abl", "max_slots_10")),
    ("Slot limit 5 / B1", ("abl", "max_slots_5")), ("Slot limit 3 / B1", ("abl", "max_slots_3")),
    ("without dynamic context / B1", ("abl", "no_dynamic")),
    ("without static context / B1", ("abl", "no_static")),
    ("all-ones matrix", ("abl", "uniform_b1")), ("Base", ("abl", "baseline_c1")),
]
# Register A, Abschnitte 5, 7.2, 7.3: Acc@1 ambig und Delta ambig gegen E_c1
REG = {
 "M5": {"Category cap 6 / B1": (.8255, -.0174), "Category cap 6 / D1": (.8137, -.0292),
        "Category cap 10 / B1": (.8429, 0), "Category cap 10 / D1": (.8542, .0113),
        "Category cap 20 / B1": (.8290, -.0139), "Category cap 20 / D1": (.8726, .0297),
        "Threshold 0.001 / B1": (.8429, 0), "Threshold 0.01 / B1": (.8515, .0086),
        "Threshold 0.01 / D1": (.8599, .0170), "Threshold 0.1 / B1": (.7859, -.0570),
        "Threshold 0.1 / D1": (.8548, .0120), "Slot limit 10 / B1": (.8429, 0),
        "Slot limit 5 / B1": (.8342, -.0087), "Slot limit 3 / B1": (.8264, -.0164),
        "without dynamic context / B1": (.7821, -.0608), "without static context / B1": (.8230, -.0198),
        "all-ones matrix": (.8022, -.0406), "Base": (.8429, None)},
 "M4": {"Category cap 6 / B1": (.7948, -.0372), "Category cap 6 / D1": (.7687, -.0633),
        "Category cap 10 / B1": (.8320, 0), "Category cap 10 / D1": (.8287, -.0034),
        "Category cap 20 / B1": (.8238, -.0082), "Category cap 20 / D1": (.8688, .0368),
        "Threshold 0.001 / B1": (.8320, 0), "Threshold 0.01 / B1": (.8397, .0077),
        "Threshold 0.01 / D1": (.8337, .0016), "Threshold 0.1 / B1": (.7341, -.0980),
        "Threshold 0.1 / D1": (.8332, .0011), "Slot limit 10 / B1": (.8320, 0),
        "Slot limit 5 / B1": (.8057, -.0263), "Slot limit 3 / B1": (.7840, -.0480),
        "without dynamic context / B1": (.7213, -.1107), "without static context / B1": (.8069, -.0251),
        "all-ones matrix": (.7638, -.0683), "Base": (.8320, None)},
}
# thesis_en Anhang Tab. A8 (M4): acc1, d_total, ambig, d_ambig
A8 = {"Category cap 6 / B1": ".8252 −.0117 .7948 −.0372", "Category cap 6 / D1": ".8169 −.0199 .7687 −.0633",
      "Category cap 10 / B1": ".8369 ±.0000 .8320 ±.0000", "Category cap 10 / D1": ".8358 −.0011 .8287 −.0034",
      "Category cap 20 / B1": ".8343 −.0026 .8238 −.0082", "Category cap 20 / D1": ".8485 +.0116 .8688 +.0368",
      "Threshold 0.001 / B1": ".8369 ±.0000 .8320 ±.0000", "Threshold 0.01 / B1": ".8393 +.0024 .8397 +.0077",
      "Threshold 0.01 / D1": ".8374 +.0005 .8337 +.0016", "Threshold 0.1 / B1": ".8060 −.0308 .7341 −.0980",
      "Threshold 0.1 / D1": ".8372 +.0004 .8332 +.0011", "Slot limit 10 / B1": ".8369 ±.0000 .8320 ±.0000",
      "Slot limit 5 / B1": ".8286 −.0083 .8057 −.0263", "Slot limit 3 / B1": ".8218 −.0151 .7840 −.0480",
      "without dynamic context / B1": ".8020 −.0349 .7213 −.1107",
      "without static context / B1": ".8290 −.0079 .8069 −.0251",
      "all-ones matrix": ".8154 −.0215 .7638 −.0683", "Base": ".8369 — .8320 —"}
# thesis_en Anhang Tab. A9: M5 acc3, M5 acc3 ambig, M4 acc3, M4 acc3 ambig
A9 = {"Category cap 6 / B1": ".8701 .9376 .8676 .9297", "Category cap 6 / D1": ".8714 .9416 .8669 .9273",
      "Category cap 10 / B1": ".8710 .9403 .8691 .9345", "Category cap 10 / D1": ".8716 .9424 .8680 .9307",
      "Category cap 20 / B1": ".8702 .9378 .8701 .9376", "Category cap 20 / D1": ".8724 .9449 .8714 .9418",
      "Threshold 0.001 / B1": ".8710 .9403 .8691 .9345", "Threshold 0.01 / B1": ".8723 .9446 .8699 .9368",
      "Threshold 0.01 / D1": ".8717 .9428 .8681 .9312", "Threshold 0.1 / B1": ".8677 .9299 .8614 .9100",
      "Threshold 0.1 / D1": ".8719 .9431 .8679 .9305", "Slot limit 10 / B1": ".8710 .9403 .8691 .9345",
      "Slot limit 5 / B1": ".8712 .9411 .8692 .9345", "Slot limit 3 / B1": ".8715 .9420 .8684 .9323",
      "without dynamic context / B1": ".8674 .9288 .8610 .9086",
      "without static context / B1": ".8715 .9419 .8703 .9381",
      "all-ones matrix": ".8673 .9285 .8611 .9090", "Base": ".8710 .9403 .8691 .9345"}
# Abb. 9: Acc@1 ambig je Variante auf M5 (Kontrollwerte in abb_9_ablationen_siebzehn_varianten.py)
FIG9 = {"without dynamic context / B1": ".7821", "without static context / B1": ".8230",
        "all-ones matrix": ".8022", "Threshold 0.001 / B1": ".8429", "Threshold 0.01 / B1": ".8515",
        "Threshold 0.1 / B1": ".7859", "Slot limit 10 / B1": ".8429", "Slot limit 5 / B1": ".8342",
        "Slot limit 3 / B1": ".8264", "Category cap 10 / B1": ".8429", "Category cap 6 / B1": ".8255",
        "Category cap 20 / B1": ".8290", "Category cap 6 / D1": ".8137", "Category cap 20 / D1": ".8726",
        "Category cap 10 / D1": ".8542", "Threshold 0.01 / D1": ".8599", "Threshold 0.1 / D1": ".8548"}
# Fliesstext 6.3.5 (Z. 320–330) und Kap. 8: (Beschreibung, Formel, Thesiswert, Fundstelle, Registerwert)
# Formel: ("wert", Modell, Variante) | ("delta", Modell, Variante b, Variante a) -> b - a (Betrag, wenn abs)
TEXT = [
    ("M5/E_d1 ambig", ("wert", "M5", "Category cap 10 / D1"), ".8542", T + ":320", ".8542"),
    ("B1: Kappung 6 kostet", ("abs", "M5", "Category cap 6 / B1", "Base"), ".0174", T + ":322", ".0174"),
    ("B1: Kappung 20 kostet", ("abs", "M5", "Category cap 20 / B1", "Base"), ".0139", T + ":322", ".0139"),
    ("D1: Kappung 6 kostet (gegen E_d1)", ("abs", "M5", "Category cap 6 / D1", "Category cap 10 / D1"), ".0405", T + ":322", ".0405"),
    ("D1: Kappung 20 bringt (gegen E_d1)", ("abs", "M5", "Category cap 20 / D1", "Category cap 10 / D1"), ".0184", T + ":322", ".0184"),
    ("D1 x 20 ambig (höchster Ablationswert)", ("wert", "M5", "Category cap 20 / D1"), ".8726", T + ":322; chapters/08_conclusion.tex:7", ".8726"),
    ("M4: D1 x 10 hinter B1-Basis", ("abs", "M4", "Category cap 10 / D1", "Base"), ".0034", T + ":322", ".0034"),
    ("M4: D1 x 20 gegen B1-Basis", ("abs", "M4", "Category cap 20 / D1", "Base"), ".0368", T + ":322", ".0368"),
    ("B1: Schwelle 0.01 bringt", ("abs", "M5", "Threshold 0.01 / B1", "Base"), ".0086", T + ":324", ".0086"),
    ("D1: Schwelle 0.01 bringt (gegen E_d1)", ("abs", "M5", "Threshold 0.01 / D1", "Category cap 10 / D1"), ".0057", T + ":324", ".0057"),
    ("B1: Schwelle 0.1 kostet", ("abs", "M5", "Threshold 0.1 / B1", "Base"), ".0570", T + ":324", ".0570"),
    ("D1: Schwelle 0.1 (gegen E_d1)", ("abs", "M5", "Threshold 0.1 / D1", "Category cap 10 / D1"), ".0006", T + ":324", ".0007 (A 7.3: +0.000671)"),
    ("Basis M5/E_c1 ambig", ("wert", "M5", "Base"), ".8429", T + ":326", ".8429"),
    ("Einsermatrix ambig M5", ("wert", "M5", "all-ones matrix"), ".8022", T + ":326/328", ".8022"),
    ("Verlust Einsermatrix M5", ("abs", "M5", "all-ones matrix", "Base"), ".0406", T + ":326", ".0406"),
    ("ohne dynamischen Kontext M5", ("wert", "M5", "without dynamic context / B1"), ".7821", T + ":326", ".7821"),
    ("ohne statischen Kontext M5", ("wert", "M5", "without static context / B1"), ".8230", T + ":326", ".8230"),
    ("Verlust ohne dynamischen Kontext M5", ("abs", "M5", "without dynamic context / B1", "Base"), ".0608", T + ":326/328", ".0608"),
    ("Verlust ohne statischen Kontext M5", ("abs", "M5", "without static context / B1", "Base"), ".0198", T + ":326", ".0198"),
    ("5 Slots kosten M5", ("abs", "M5", "Slot limit 5 / B1", "Base"), ".0087", T + ":326", ".0087"),
    ("3 Slots kosten M5", ("abs", "M5", "Slot limit 3 / B1", "Base"), ".0164", T + ":326/328", ".0164"),
    ("M4 ohne dynamischen Kontext", ("abs", "M4", "without dynamic context / B1", "Base"), ".1107", T + ":328", ".1107"),
    ("M4 Schwelle 0.1", ("abs", "M4", "Threshold 0.1 / B1", "Base"), ".0980", T + ":328", ".0980"),
    ("M4 3 Slots", ("abs", "M4", "Slot limit 3 / B1", "Base"), ".0480", T + ":328", ".0480"),
    ("M4 Einsermatrix ambig", ("wert", "M4", "all-ones matrix"), ".7638", T + ":328", ".7638"),
]


def finde(dirs, name):
    for d in dirs:
        for p in (d / name, d / "per_item" / name):
            if p.is_file():
                return p
    raise SystemExit(f"fehlt: {name}")


def ambig_formel(acc, sockel):
    return (round(acc * N_ITEMS) - sockel) / N_AMBIG


def fmt(x, st=4, sign=False):
    s = f"{x:+.{st}f}" if sign else f"{x:.{st}f}"
    return s.replace("0.", ".", 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abl-m5", type=Path, default=EXP2 / "results" / "ablations.json")
    ap.add_argument("--abl-m4", type=Path, default=EXP2 / "results" / "ablations_M4.json")
    ap.add_argument("--zellen", nargs="+", type=Path, default=[EXP2 / "results" / "matrix"])
    ap.add_argument("--out", type=Path, default=HERE / "out" / "nachbau_ablationen_ambig_check.csv")
    a = ap.parse_args()
    abl = {"M5": json.loads(a.abl_m5.read_text()), "M4": json.loads(a.abl_m4.read_text())}
    rows = []

    def add(rid, b, reg, th, neu, g):
        rows.append({"register_id": rid, "beschreibung": b, "wert_register": reg,
                     "wert_thesis": th, "wert_nachgerechnet": neu, "gleich": "ja" if g else "nein"})

    W = {}   # (Modell, Variante) -> dict acc1, ambig, acc3, acc3_ambig
    for mod in ("M5", "M4"):
        assert abl[mod]["model"] == M[mod]
        # Sockel aus dem Per-Item-Dump der Basiszelle nachzählen
        base_items = pickle.load(gzip.open(finde(a.zellen, f"{M[mod]}_E_c1.pkl.gz"), "rb"))
        s1 = sum(p["n_candidates"] == 1 and p["rank"] == 1 for p in base_items)
        s3 = sum(p["n_candidates"] == 1 and p["rank"] is not None and p["rank"] <= 3 for p in base_items)
        nam = sum(p["n_candidates"] > 1 for p in base_items)
        add("A 5 / K-14 1.3", f"{mod}: Sockel Acc@1 / Acc@3 und Zahl ambiger Items (Per-Item E_c1)",
            "43544 / 43544 / 23848", "43,544 (Anhang Ablationen)", f"{s1} / {s3} / {nam}",
            s1 == s3 == 43544 and nam == N_AMBIG)
        for v in abl[mod]["variants"].values():
            assert v["total"] == N_ITEMS and v["no_candidates"] == 8159
        for lab, (art, key) in VAR:
            if art == "abl":
                v = abl[mod]["variants"][key]
                acc1, acc3 = v["accuracy_at_1"], v["accuracy_at_3"]
                am1, am3 = ambig_formel(acc1, s1), ambig_formel(acc3, s3)
                # Kreuzprobe: beide Nenner liefern denselben ganzzahligen Zähler
                wc = v["accuracy_at_1_with_candidates"] * (N_ITEMS - 8159)
                assert abs(wc - round(wc)) < 1e-6 and round(wc) == round(acc1 * N_ITEMS)
            else:
                it = pickle.load(gzip.open(finde(a.zellen, f"{M[mod]}_{key}.pkl.gz"), "rb"))
                amb = [p for p in it if p["n_candidates"] > 1]
                acc1 = sum(p["rank"] == 1 for p in it) / len(it)
                acc3 = sum(p["rank"] is not None and p["rank"] <= 3 for p in it) / len(it)
                am1 = sum(p["rank"] == 1 for p in amb) / len(amb)
                am3 = sum(p["rank"] is not None and p["rank"] <= 3 for p in amb) / len(amb)
                assert abs(am1 - ambig_formel(acc1, s1)) < 1e-12 and abs(am3 - ambig_formel(acc3, s3)) < 1e-12
            W[(mod, lab)] = {"acc1": acc1, "ambig": am1, "acc3": acc3, "acc3_ambig": am3}

    for mod in ("M5", "M4"):
        b = W[(mod, "Base")]
        for lab, _ in VAR:
            w = W[(mod, lab)]
            r_amb, r_d = REG[mod][lab]
            d_amb, d_tot = w["ambig"] - b["ambig"], w["acc1"] - b["acc1"]
            # Acc@1 ambig
            th, g = "nicht in der Arbeit", True
            if mod == "M4":
                t = A8[lab].split()
                th = f"{t[2]} ({A} Tab. A8)"
                g &= fmt(w["ambig"]) == t[2]
            elif lab in FIG9:
                th = f"{FIG9[lab]} ({FIG})"
                g &= fmt(w["ambig"]) == FIG9[lab]
            g &= fmt(w["ambig"]) == fmt(r_amb)
            add("A 5 / 7.2 / 7.3", f"{mod} {lab}: Acc@1 ambig", fmt(r_amb), th, f"{w['ambig']:.6f}", g)
            if r_d is not None:
                th, g = "nicht in der Arbeit", True
                if mod == "M4":
                    t = A8[lab].split()[3].replace("−", "-").replace("±", "+")
                    th = f"{A8[lab].split()[3]} ({A} Tab. A8)"
                    g &= fmt(d_amb, sign=True).replace("-.0000", "+.0000") == t
                g &= fmt(d_amb, sign=True).replace("-.0000", "+.0000") == fmt(r_d, sign=True)
                add("A 5 / 7.2 / 7.3", f"{mod} {lab}: Δ Acc@1 ambig gegen E_c1", fmt(r_d, sign=True),
                    th, f"{d_amb:+.6f}", g)
            if mod == "M4":
                t = A8[lab].split()
                add("A 5 / 7.2 / 7.3", f"M4 {lab}: Acc@1 gesamt", "—", f"{t[0]} ({A} Tab. A8)",
                    f"{w['acc1']:.6f}", fmt(w["acc1"]) == t[0])
                if t[1] != "—":
                    add("A 5 / 7.2 / 7.3", f"M4 {lab}: Δ Acc@1 gesamt", "—", f"{t[1]} ({A} Tab. A8)",
                        f"{d_tot:+.6f}", fmt(d_tot, sign=True).replace("-.0000", "+.0000")
                        == t[1].replace("−", "-").replace("±", "+"))
            # Acc@3 gesamt und ambig (Tab. A9)
            t9 = A9[lab].split()
            i = 0 if mod == "M5" else 2
            add("A 5 (nur Spannweite)", f"{mod} {lab}: Acc@3 gesamt", "—", f"{t9[i]} ({A} Tab. A9)",
                f"{w['acc3']:.6f}", fmt(w["acc3"]) == t9[i])
            add("A 5 (nur Spannweite)", f"{mod} {lab}: Acc@3 ambig", "—", f"{t9[i+1]} ({A} Tab. A9)",
                f"{w['acc3_ambig']:.6f}", fmt(w["acc3_ambig"]) == t9[i + 1])

    for besch, f, tv, wo, rv in TEXT:
        if f[0] == "wert":
            x = W[(f[1], f[2])]["ambig"]
        else:
            x = abs(W[(f[1], f[2])]["ambig"] - W[(f[1], f[3])]["ambig"])
        add("A 5 / 7.2 / 7.3", f"Fliesstext: {besch}", rv, f"{tv} ({wo})", f"{x:.6f}", fmt(x) == tv)

    # Spannweiten 6.3.5 (Feld ohne die D1-Zeilen, M5)
    b1 = [lab for lab, _ in VAR if not lab.endswith("D1")]
    r1 = [W[("M5", l)]["ambig"] for l in b1]
    r3 = [W[("M5", l)]["acc3_ambig"] for l in b1]
    for besch, x, tv in (("Acc@3 ambig Minimum", min(r3), ".9285"), ("Acc@3 ambig Maximum", max(r3), ".9446"),
                         ("Acc@3 ambig Spannweite", max(r3) - min(r3), ".0161"),
                         ("Acc@1 ambig Spannweite", max(r1) - min(r1), ".0694")):
        add("A 5", f"Fliesstext 6.3.5: {besch}, M5 ohne D1-Zeilen", tv, f"{tv} ({T}:330)",
            f"{x:.6f}", fmt(x) == tv)
    q = W[("M5", "all-ones matrix")]["ambig"] - W[("M5", "Base")]["ambig"]
    q /= W[("M5", "without dynamic context / B1")]["ambig"] - W[("M5", "Base")]["ambig"]
    add("A 5", "Quotient Verlust Einsermatrix / Verlust ohne dynamischen Kontext («two-thirds»)",
        ".668", f"two-thirds ({T}:326)", f"{q:.4f}", f"{q:.3f}" == "0.668")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["register_id", "beschreibung", "wert_register",
                                           "wert_thesis", "wert_nachgerechnet", "gleich"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    n_ja = sum(r["gleich"] == "ja" for r in rows)
    print(f"geschrieben: {a.out}  ({n_ja} von {len(rows)} gleich)")
    for r in rows:
        if r["gleich"] == "nein":
            print("  NEIN:", r["beschreibung"], "| Register", r["wert_register"], "| Thesis",
                  r["wert_thesis"], "| neu", r["wert_nachgerechnet"])


if __name__ == "__main__":
    main()
