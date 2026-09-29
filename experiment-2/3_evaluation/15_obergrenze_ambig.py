"""
Stage 15 — Obergrenze der Leitmetrik (Acc@1 ueber die ambigen Toponyme).

Ambig = mindestens zwei Kandidaten (wie 02_evaluate_matrix.py). Erreichbar ist ein
ambiges Toponym nur, wenn das Gold unter seinen Kandidaten steht:
    Obergrenze_ambig = #{n_i >= 2 und gold_i in C_i} / #{n_i >= 2}
Die Kandidatenmengen haengen nicht vom Sentence Generator ab; gerechnet auf
cache/items_E_default.pkl, Gegenprobe auf items_E_c1/E_d1/E_d1_c20.pkl und auf den
per_item-Dateien (rank None bei n_candidates >= 2) von M5/E_c1, M5/E_d1, M5/E_d1_c20.
Zur Einordnung zusaetzlich die Obergrenzen gesamt (.885) und mit Kandidat (.992)
sowie der ausgeschoepfte Anteil der Obergrenze fuer drei M5-Zellen.

Output: results/obergrenze_ambig.csv
"""
import csv
import gzip
import pickle

import config as C

ITEMS = ["items_E_default.pkl", "items_E_c1.pkl", "items_E_d1.pkl", "items_E_d1_c20.pkl"]
PER_ITEM = {
    "M5/E_c1": C.PER_ITEM_DIR / "M5_spatial_config2_E_c1.pkl.gz",
    "M5/E_d1": C.PER_ITEM_DIR / "M5_spatial_config2_E_d1.pkl.gz",
    "M5/E_d1_c20": C.PER_ITEM_DIR / "M5_spatial_config2_E_d1_c20.pkl.gz",
}


def grenzen(items):
    n_all = len(items)
    n_cand = n_amb = reach_cand = reach_amb = 0
    for x in items:
        c = x["candidate_ids"]
        n = len(c)
        if n == 0:
            continue
        n_cand += 1
        hit = x["gold_id"] in c
        reach_cand += hit
        if n > 1:
            n_amb += 1
            reach_amb += hit
    return {"total": n_all, "with_candidates": n_cand, "ambiguous_total": n_amb,
            "reachable_all": reach_cand, "reachable_ambiguous": reach_amb,
            "gold_missing_ambiguous": n_amb - reach_amb,
            "bound_acc1": reach_cand / n_all,
            "bound_with_cand": reach_cand / n_cand,
            "bound_ambiguous_only": reach_amb / n_amb}


rows = []
for f in ITEMS:
    r = grenzen(pickle.load(open(C.CACHE_DIR / f, "rb"))["items"])
    rows.append({"source": f, **r})
    print(f, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
assert len({(r["ambiguous_total"], r["reachable_ambiguous"]) for r in rows}) == 1
ref = rows[0]

for sys_, path in PER_ITEM.items():
    p = pickle.load(gzip.open(path))
    amb = [x for x in p if x["n_candidates"] > 1]
    missing = sum(1 for x in amb if x["rank"] is None)
    correct = sum(1 for x in amb if x["rank"] == 1)
    assert len(amb) == ref["ambiguous_total"], sys_
    assert missing == ref["gold_missing_ambiguous"], sys_
    acc = correct / len(amb)
    share = correct / (len(amb) - missing)
    print(sys_, {"ambig": len(amb), "gold_missing": missing, "correct_ambig": correct,
                 "acc1_ambig": round(acc, 6), "share_of_bound": round(share, 6)})
    rows.append({"source": sys_, "ambiguous_total": len(amb),
                 "gold_missing_ambiguous": missing,
                 "reachable_ambiguous": len(amb) - missing,
                 "bound_ambiguous_only": (len(amb) - missing) / len(amb),
                 "acc1_ambiguous_only": acc, "correct_ambiguous": correct,
                 "share_of_bound": share})

fields = []
for r in rows:
    fields += [k for k in r if k not in fields]
with open(C.RESULTS_DIR / "obergrenze_ambig.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
