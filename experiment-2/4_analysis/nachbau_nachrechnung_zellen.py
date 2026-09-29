#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nachbau_nachrechnung_zellen.py — Nachbau des verlorenen Skripts
`$HOME/ausw_ed1/nachrechnung_zellen.py` (Zahlenregister A, Lücke 12.8). Das
Original ist nicht auffindbar (MA-Ordner, Zips, Vault-Git geprüft am 29.09.2026).
Laut gelöschtem Bericht `AUSWERTUNG_E_d1.md` (Vault-Commit 1c7c507^) zählte es
die Zellen des E_d1-Laufs vom 17.09.2026 aus den Per-Item-Abzügen neu aus.

Was gerechnet wird (je Zelle Modell × Sentence Generator, Definitionen wie
eval_core.location_metrics / document_metrics und 02_evaluate_matrix.error_counts):
  acc1  = Anteil rank == 1 über alle Items (75'741, ohne Kandidat = Fehler)
  acc3  = Anteil rank <= 3;  mrr = Mittel von rr
  acc1_wc / acc3_wc = dasselbe über Items mit mindestens einem Kandidaten
  doc   = Makromittel der Dokumentgenauigkeit (6'323 Dokumente)
  ambig = Acc@1 über Items mit n_candidates > 1 (23'848)
Jede Zahl wird (a) gegen das JSON der Pipeline, (b) gegen Register A 7.1–7.3 und
(c) gegen den Wert in thesis_en (auf dessen Rundung) gestellt.

Zellen: die 22 Zellen des E_d1-Laufs (M1–M5 × E_c1/E_d1, M4/M5 × sieben Achsenarme)
plus die fünf E_default-Zellen der Hauptmatrix (für die Differenzen in 6.3.1/7.1).

Eingaben (Standard = Ablageort der Pipeline in ma-experiments, config.py)
  <exp2>/results/matrix/<Modell>_<SG>.json und per_item/<Modell>_<SG>.pkl.gz
  Abweichende Orte per --zellen DIR [DIR ...] (je DIR und DIR/per_item wird gesucht).
Ausgabe
  <exp2>/4_analysis/out/nachbau_nachrechnung_zellen_check.csv (nur Aggregatzahlen)
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
M = {"M1": "M1_dguzh", "M2": "M2_distiluse_base", "M3": "M3_default_finetuned",
     "M4": "M4_spatial_config1", "M5": "M5_spatial_config2"}
ARME = ["E_b1_c6", "E_b1_c20", "E_d1_c6", "E_d1_c20", "E_d1_s01", "E_d1_s1"]
ZELLEN = ([(m, "E_default") for m in M] + [(m, sg) for m in M for sg in ("E_c1", "E_d1")]
          + [(m, a) for m in ("M4", "M5") for a in ARME])
FELDER = ["acc1", "ambig", "acc3", "mrr", "doc", "acc1_wc", "acc3_wc"]
JSONFELD = {"acc1": ("location", "accuracy_at_1"), "acc3": ("location", "accuracy_at_3"),
            "mrr": ("location", "mrr"), "acc1_wc": ("location", "accuracy_at_1_with_candidates"),
            "acc3_wc": ("location", "accuracy_at_3_with_candidates"),
            "doc": ("document", "document_accuracy"), "ambig": ("errors", "acc1_ambiguous_only")}

# Register A, Abschnitte 7.1–7.3: acc1, ambig, acc3, mrr, doc (vierstellig)
REG = {
    ("M1", "E_c1"): (.7595, .5863, .8498, .8085, .7111), ("M1", "E_d1"): (.7834, .6620, .8523, .8213, .7624),
    ("M2", "E_c1"): (.8015, .7196, .8616, .8335, .7745), ("M2", "E_d1"): (.8101, .7470, .8622, .8375, .7903),
    ("M3", "E_c1"): (.8367, .8315, .8713, .8550, .8156), ("M3", "E_d1"): (.8299, .8100, .8696, .8510, .8139),
    ("M4", "E_c1"): (.8369, .8320, .8691, .8539, .8190), ("M4", "E_d1"): (.8358, .8287, .8680, .8533, .8211),
    ("M5", "E_c1"): (.8403, .8429, .8710, .8568, .8217), ("M5", "E_d1"): (.8438, .8542, .8716, .8592, .8314),
    ("M5", "E_b1_c6"): (.8348, .8255, .8701, .8536, .8186), ("M5", "E_b1_c20"): (.8359, .8290, .8702, .8545, .8176),
    ("M5", "E_d1_c6"): (.8311, .8137, .8714, .8525, .8209), ("M5", "E_d1_c20"): (.8496, .8726, .8724, .8622, .8381),
    ("M4", "E_b1_c6"): (.8252, .7948, .8676, .8474, .8095), ("M4", "E_b1_c20"): (.8343, .8238, .8701, .8534, .8151),
    ("M4", "E_d1_c6"): (.8169, .7687, .8669, .8434, .8030), ("M4", "E_d1_c20"): (.8485, .8688, .8714, .8610, .8353),
    ("M5", "E_d1_s01"): (.8456, .8599, .8717, .8601, .8324), ("M5", "E_d1_s1"): (.8441, .8548, .8719, .8592, .8303),
    ("M4", "E_d1_s01"): (.8374, .8337, .8681, .8541, .8221), ("M4", "E_d1_s1"): (.8372, .8332, .8679, .8540, .8215),
}
# thesis_en: (Zelle, Feld) -> (Wert wie gedruckt, Fundstelle)
T = "chapters/06_results.tex"
A = "appendix/appendix.tex"
THESIS = {
    (("M5", "E_d1"), "acc1"): (".844", T + ":210"), (("M5", "E_d1"), "ambig"): (".8542", T + ":320"),
    (("M5", "E_c1"), "acc1"): (".8403", T + ":214"), (("M5", "E_c1"), "ambig"): (".8429", T + ":326"),
    (("M4", "E_c1"), "acc1"): (".8369", T + ":214"), (("M4", "E_c1"), "ambig"): (".832", T + ":210"),
    (("M3", "E_c1"), "acc1"): (".8367", T + ":214"), (("M4", "E_d1"), "ambig"): (".829", T + ":281 (Tab. 9)"),
    (("M1", "E_d1"), "ambig"): (".662", T + ":230"), (("M2", "E_d1"), "ambig"): (".747", T + ":230"),
    (("M3", "E_default"), "acc1"): (".821", T + ":208"), (("M3", "E_default"), "ambig"): (".782", T + ":228"),
    (("M5", "E_d1"), "mrr"): (".859", T + ":214 (Maximum MRR) / " + A + " Tab. A6"),
}
# Anhang Tab. A5 (acc3, doc, acc1_wc, acc3_wc) und A6 (mrr), dreistellig
A5 = {("M1", "E_c1"): ".850 .711 .851 .952", ("M1", "E_d1"): ".852 .762 .878 .955",
      ("M2", "E_c1"): ".862 .774 .898 .966", ("M2", "E_d1"): ".862 .790 .908 .966",
      ("M3", "E_c1"): ".871 .816 .938 .976", ("M3", "E_d1"): ".870 .814 .930 .975",
      ("M4", "E_c1"): ".869 .819 .938 .974", ("M4", "E_d1"): ".868 .821 .937 .973",
      ("M5", "E_c1"): ".871 .822 .942 .976", ("M5", "E_d1"): ".872 .831 .946 .977"}
A6 = {("M1", "E_c1"): ".808", ("M1", "E_d1"): ".821", ("M2", "E_c1"): ".834", ("M2", "E_d1"): ".838",
      ("M3", "E_c1"): ".855", ("M3", "E_d1"): ".851", ("M4", "E_c1"): ".854", ("M4", "E_d1"): ".853",
      ("M5", "E_c1"): ".857"}
for k, s in A5.items():
    for f, v in zip(("acc3", "doc", "acc1_wc", "acc3_wc"), s.split()):
        THESIS[(k, f)] = (v, A + " Tab. A5")
for k, v in A6.items():
    THESIS.setdefault((k, "mrr"), (v, A + " Tab. A6"))
# Differenzen in Prozentpunkten (Zelle b minus Zelle a): Thesiswert, Fundstelle
DIFF = [
    (("M3", "E_default"), ("M3", "E_d1"), "acc1", "0.9", T + ":208"),
    (("M1", "E_default"), ("M1", "E_d1"), "acc1", "16.7", T + ":208"),
    (("M3", "E_default"), ("M3", "E_d1"), "ambig", "2.8", T + ":208"),
    (("M1", "E_default"), ("M1", "E_d1"), "ambig", "52.9", T + ":208"),
    (("M3", "E_default"), ("M5", "E_d1"), "ambig", "7.3", T + ":230"),
    (("M3", "E_default"), ("M5", "E_d1"), "acc1", "2.3", "chapters/07_discussion.tex:12"),
    (("M3", "E_default"), ("M5", "E_c1"), "ambig", "6.1", T + ":230"),
    (("M3", "E_default"), ("M5", "E_c1"), "acc1", "1.9", T + ":230"),
]


def finde(dirs, name):
    for d in dirs:
        for p in (d / name, d / "per_item" / name):
            if p.is_file():
                return p
    return None


def zaehle(items):
    n = len(items)
    wc = [p for p in items if p["n_candidates"] > 0]
    amb = [p for p in items if p["n_candidates"] > 1]
    docs = {}
    for p in items:
        docs.setdefault(p["doc_id"], []).append(p["rank"] == 1)
    return {"acc1": sum(p["rank"] == 1 for p in items) / n,
            "acc3": sum(p["rank"] is not None and p["rank"] <= 3 for p in items) / n,
            "mrr": sum(p["rr"] for p in items) / n,
            "acc1_wc": sum(p["rank"] == 1 for p in wc) / len(wc),
            "acc3_wc": sum(p["rank"] is not None and p["rank"] <= 3 for p in wc) / len(wc),
            "doc": sum(sum(v) / len(v) for v in docs.values()) / len(docs),
            "ambig": sum(p["rank"] == 1 for p in amb) / len(amb),
            "_n": n, "_amb": len(amb), "_docs": len(docs),
            "_sockel": sum(p["rank"] == 1 and p["n_candidates"] == 1 for p in items)}


def stellen(s: str) -> int:
    return len(s.split(".")[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zellen", nargs="+", type=Path, default=[EXP2 / "results" / "matrix"])
    ap.add_argument("--out", type=Path,
                    default=HERE / "out" / "nachbau_nachrechnung_zellen_check.csv")
    a = ap.parse_args()

    rows, R = [], {}
    for (m, sg) in ZELLEN:
        name = f"{M[m]}_{sg}"
        pk = finde(a.zellen, name + ".pkl.gz")
        if pk is None:
            raise SystemExit(f"fehlt: {name}.pkl.gz")
        r = zaehle(pickle.load(gzip.open(pk, "rb")))
        R[(m, sg)] = r
        js = finde(a.zellen, name + ".json")
        j = json.loads(js.read_text(encoding="utf-8")) if js else None
        reg = REG.get((m, sg))
        for f in FELDER:
            v = r[f]
            reg_s, g = "—", True
            if reg and f in ("acc1", "ambig", "acc3", "mrr", "doc"):
                rv = reg[("acc1", "ambig", "acc3", "mrr", "doc").index(f)]
                reg_s = f"{rv:.4f}"
                g &= f"{v:.4f}" == reg_s
            if j is not None:
                jv = j[JSONFELD[f][0]][JSONFELD[f][1]]
                g &= abs(jv - v) < 1e-12
            th = THESIS.get(((m, sg), f))
            if th:
                t_s = f"{th[0]} ({th[1]})"
                g &= f"{v:.{stellen(th[0])}f}".lstrip("0") == th[0]
            else:
                t_s = "nicht in der Arbeit"
            if reg_s == "—" and not th:
                continue    # weder Register- noch Thesiswert: nur JSON-Abgleich, unten gezählt
            rows.append({"register_id": "A 7.1–7.3" if reg else "A 2 (Hauptmatrix)",
                         "beschreibung": f"{m}/{sg} {f}" + (" [= JSON]" if j is not None and
                                         abs(j[JSONFELD[f][0]][JSONFELD[f][1]] - v) < 1e-12 else ""),
                         "wert_register": reg_s, "wert_thesis": t_s,
                         "wert_nachgerechnet": f"{v:.6f}", "gleich": "ja" if g else "nein"})
        if j is not None:
            abw = [f for f in FELDER if abs(j[JSONFELD[f][0]][JSONFELD[f][1]] - r[f]) >= 1e-12]
            rows.append({"register_id": "JSON-Abgleich", "beschreibung":
                         f"{m}/{sg}: alle 7 Kennzahlen gleich dem Pipeline-JSON",
                         "wert_register": "—", "wert_thesis": "—",
                         "wert_nachgerechnet": "abweichend: " + ",".join(abw) if abw else "alle gleich",
                         "gleich": "nein" if abw else "ja"})
        print(f"{m}/{sg:9s} acc1 {r['acc1']:.4f} ambig {r['ambig']:.4f} acc3 {r['acc3']:.4f} "
              f"mrr {r['mrr']:.4f} doc {r['doc']:.4f} n={r['_n']} amb={r['_amb']} "
              f"docs={r['_docs']} sockel={r['_sockel']} json={'ja' if j else 'nein'}")

    sockel = {R[z]["_sockel"] for z in ZELLEN}
    rows.append({"register_id": "A 7 (Nachtrag 19.09.)", "beschreibung":
                 "eindeutiger Sockel korrekt (n_candidates == 1), alle Zellen",
                 "wert_register": "43544", "wert_thesis": "43,544 (" + T + ":226)",
                 "wert_nachgerechnet": "/".join(map(str, sorted(sockel))),
                 "gleich": "ja" if sockel == {43544} else "nein"})
    for za, zb, f, tv, wo in DIFF:
        d = 100 * (R[zb][f] - R[za][f])
        rows.append({"register_id": "A 2/3 bzw. 7.1", "beschreibung":
                     f"Differenz {zb[0]}/{zb[1]} minus {za[0]}/{za[1]}, {f}, Prozentpunkte",
                     "wert_register": "—", "wert_thesis": f"{tv} ({wo})",
                     "wert_nachgerechnet": f"{d:.3f}", "gleich": "ja" if f"{d:.1f}" == tv else "nein"})

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["register_id", "beschreibung", "wert_register",
                                           "wert_thesis", "wert_nachgerechnet", "gleich"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"geschrieben: {a.out}  ({sum(r['gleich'] == 'ja' for r in rows)} von {len(rows)} gleich)")


if __name__ == "__main__":
    main()
