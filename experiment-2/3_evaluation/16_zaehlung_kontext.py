"""Stage 16 (Zusatzlauf all_categories, 29.09.2026): Sanity, Acc@1/Acc@3 ambig und Mengenkontrolle je Variante.

Liest results/<--json> (Ausgabe von 04_ablations.py), die Beschreibungscaches
cache/descriptions_<key>.pkl und die ambigen Items aus cache/items_E_c1.pkl
(len(candidate_ids) >= 2). Gewichtung: jede Kandidatenbeschreibung zaehlt so oft,
wie sie in ambigen Items vorkommt.

Kennzahlen je Variante:
  anteil_bei   Anteil Beschreibungen mit Kontextobjekten (enthaelt ", bei ")
  obj_mittel   mittlere Zahl genannter Kontextobjekt-NAMEN je Beschreibung (so, wie
               sie im Satz stehen: mehrsprachige Namen eines Objekts zaehlen einzeln,
               daher sind > max_slots Namen moeglich)
  kat_mittel   mittlere Zahl genannter Kategorien je Beschreibung
Namen/Kategorien werden aus dem "bei"-Teil geparst (Kategorielabels = Zeilen der
B1-Matrix); Namen, die selbst ", " oder " und " enthalten, koennen die Zahl leicht
verfaelschen. Gruppen ohne abschliessendes Label zaehlen als parse_warnungen.

Aufruf (aus 3_evaluation):  python 16_zaehlung_kontext.py --json ablations_all.json --out ../results/zaehlung_all.json
Acc@k ambig = (round(acc_k x 75741) - 43544) / 23848 (konstanter Sockel der eindeutigen Toponyme, Rang 1 und 3).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

N_TOTAL, N_OK_UNAMBIG, N_AMBIG = 75741, 43544, 23848
SOLL = {"baseline_c1": 0.8429, "no_dynamic": 0.7821, "uniform_b1": 0.8022}
REIHENFOLGE = ["baseline_c1", "no_dynamic", "uniform_b1", "all_uniform", "all_b1", "all_d1"]


def acc_ambig(acc1: float) -> float:
    return (round(acc1 * N_TOTAL) - N_OK_UNAMBIG) / N_AMBIG


def parse_kontext(satz: str, labels: set[str]) -> tuple[int, int, bool]:
    """-> (objekte, kategorien, warnung) des 'bei'-Teils."""
    toks = satz.split(", ")
    start = next((i for i, t in enumerate(toks) if i >= 2 and t.startswith("bei ")), None)
    if start is None:
        return 0, 0, False
    toks = toks[start:]
    toks[0] = toks[0][4:]
    obj = kat = 0
    gruppe: list[str] = []
    for t in toks:
        if not gruppe and (t.startswith("in ") or t.startswith("nahe ")):
            break  # statischer bzw. Filler-Teil beginnt
        if t in labels and gruppe:
            obj += len(gruppe) + (1 if " und " in gruppe[-1] else 0)
            kat += 1
            gruppe = []
        else:
            gruppe.append(t)
    return obj, kat, bool(gruppe)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="ablations_all.json", help="relativ zu results/")
    ap.add_argument("--eval-dir", default=".", help="Pfad zu 3_evaluation")
    ap.add_argument("--matrix", default=None,
                    help="Matrix fuer die Kategorielabels (Default config1/b1_matrix.csv)")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="nur Zaehlung dieser Varianten, ohne Sanity (Test)")
    ap.add_argument("--out", default=None, help="Ergebnis zusaetzlich als JSON")
    a = ap.parse_args()

    ev = Path(a.eval_dir).resolve()
    sys.path.insert(0, str(ev))
    cache = ev / "cache"

    if a.matrix:
        mpath = Path(a.matrix)
    else:
        import config as C  # noqa: E402
        mpath = C.matrix_path(C.ABLATION_BASE_CONFIG)
    labels = set(pd.read_csv(mpath, sep=";", index_col=0).index.astype(str))

    items = pickle.loads((cache / "items_E_c1.pkl").read_bytes())["items"]
    ambig = [it for it in items if len(it["candidate_ids"]) >= 2]
    cand = [u for it in ambig for u in it["candidate_ids"]]
    print(f"ambige Items: {len(ambig)} (Soll {N_AMBIG}), Kandidatenbeschreibungen: {len(cand)}")
    if len(ambig) != N_AMBIG:
        print("WARNUNG: Zahl ambiger Items weicht ab")

    res: dict = {}
    gueltig = True
    if a.variants is None:
        data = json.loads((ev / "results" / a.json).read_text(encoding="utf-8"))
        var = data["variants"]
        print(f"\nModell {data.get('model')}, {data.get('n_documents')} Docs")
        for vid, loc in var.items():
            if loc.get("total") not in (None, N_TOTAL):
                print(f"UNGUELTIG: {vid} total={loc.get('total')} statt {N_TOTAL}")
                gueltig = False
        print("\nSanity (Acc@1 ambig, 4 Stellen):")
        for vid, soll in SOLL.items():
            ist = round(acc_ambig(var[vid]["accuracy_at_1"]), 4) if vid in var else None
            ok = ist == soll
            gueltig &= ok
            print(f"  {vid:12s} ist {ist}  soll {soll}  {'OK' if ok else 'ABWEICHUNG'}")
        print("LAUF GUELTIG" if gueltig else "LAUF UNGUELTIG - nicht auswerten, melden")
        vids = [v for v in REIHENFOLGE if v in var] + [v for v in var if v not in REIHENFOLGE]
    else:
        var, vids = {}, a.variants

    nd = acc_ambig(var["no_dynamic"]["accuracy_at_1"]) if "no_dynamic" in var else None
    print(f"\n{'Variante':12s} | {'Acc@1 amb':>9s} | {'d no_dyn':>8s} | "
          f"{'Anteil bei':>10s} | {'Namen/Beschr':>12s} | {'Kat/Beschr':>10s} | Warn")
    for vid in vids:
        key = "E_c1" if vid == "baseline_c1" else f"abl_{vid}"
        p = cache / f"descriptions_{key}.pkl"
        if not p.exists():
            print(f"{vid:12s} | Cache fehlt: {p.name}")
            continue
        desc = pickle.loads(p.read_bytes())
        fehlt = sum(1 for u in cand if u not in desc)
        n_bei = s_obj = s_kat = warn = 0
        for u in cand:
            s = desc.get(u, "")
            n_bei += ", bei " in s
            o, k, w = parse_kontext(s, labels)
            s_obj += o
            s_kat += k
            warn += w
        n = len(cand)
        r = {"anteil_bei": n_bei / n, "obj_mittel": s_obj / n, "kat_mittel": s_kat / n,
             "parse_warnungen": warn, "fehlende_beschreibungen": fehlt}
        acc = d = None
        if vid in var:
            acc = acc_ambig(var[vid]["accuracy_at_1"])
            d = acc - nd if nd is not None else None
            acc3 = acc_ambig(var[vid]["accuracy_at_3"]) if "accuracy_at_3" in var[vid] else None
            r.update(acc1_ambig=acc, delta_no_dynamic=d, acc3_ambig=acc3)
        res[vid] = r
        f = lambda x, fmt: format(x, fmt) if x is not None else "-"
        print(f"{vid:12s} | {f(acc, '9.4f')} | {f(d, '+8.4f')} | {r['anteil_bei']:10.3f} | "
              f"{r['obj_mittel']:12.2f} | {r['kat_mittel']:10.2f} | {warn}"
              + (f"  (FEHLEN {fehlt})" if fehlt else ""))

    if a.out:
        Path(a.out).write_text(json.dumps({"gueltig": gueltig, "varianten": res}, indent=2,
                                          ensure_ascii=False), encoding="utf-8")
        print(f"\nGespeichert: {a.out}")
    sys.exit(0 if gueltig else 2)


if __name__ == "__main__":
    main()
