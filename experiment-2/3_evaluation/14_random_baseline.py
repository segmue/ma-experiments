"""
Stage 14 — Erwartete Acc@1 einer gleichverteilten Zufallswahl (random baseline).

E[Acc@1_rand] = 1/N * sum_i [gold_i in C_i] / |C_i|
Toponyme ohne Kandidat oder ohne Gold in C_i tragen 0 bei.
Die Kandidatenmengen haengen nicht vom Sentence Generator ab; gerechnet wird
auf cache/items_E_default.pkl, Gegenprobe auf items_E_c1.pkl und items_E_d1.pkl.

Output: results/random_baseline.csv
"""
import csv
import pickle

import config as C

FILES = ["items_E_default.pkl", "items_E_c1.pkl", "items_E_d1.pkl"]


def baseline(items):
    N = len(items)
    s_all = s_amb = 0.0
    n_amb = n_cand = 0
    for x in items:
        c = x["candidate_ids"]
        n = len(c)
        if n == 0:
            continue
        n_cand += 1
        p = (1.0 / n) if x["gold_id"] in c else 0.0
        s_all += p
        if n > 1:
            n_amb += 1
            s_amb += p
    return {"total": N, "with_candidates": n_cand, "ambiguous_total": n_amb,
            "exp_acc1": s_all / N,
            "exp_acc1_with_cand": s_all / n_cand,
            "exp_acc1_ambiguous_only": s_amb / n_amb}


rows = []
for f in FILES:
    r = baseline(pickle.load(open(C.CACHE_DIR / f, "rb"))["items"])
    rows.append({"source": f, **r})
    print(f, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

assert len({(r["exp_acc1"], r["exp_acc1_ambiguous_only"]) for r in rows}) == 1

with open(C.RESULTS_DIR / "random_baseline.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
