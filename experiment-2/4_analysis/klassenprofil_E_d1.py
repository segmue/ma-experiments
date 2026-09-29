#!/usr/bin/env python3
"""Runde 063: Fehlerspalten von Tabelle 11 fuer M5/E_d1 (statt M5/E_c1).
Liest die Einzelwerte M5/E_d1 und M5/E_c1, joint OBJEKTART per UUID aus dem
GeoPackage (sqlite3), zaehlt je Klasse: n, acc1, wrong_rank_rate, no_candidate_rate,
gold_not_in_candidates_rate. Schreibt results/tables/klassenprofil_E_d1.csv."""
import pickle, gzip, sqlite3, csv, collections, sys
from pathlib import Path
# Pfade des Repos: 3_evaluation/config.py (nur pathlib, keine Rechenabhaengigkeiten).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3_evaluation"))
import config as C  # noqa: E402
gpkg = C.CORPUS_GPKG
con = sqlite3.connect(str(gpkg))
art = dict(con.execute("SELECT sn3d_uuid, sn3d_objektart FROM corpus_toponyms").fetchall())
files = {"E_d1": C.PER_ITEM_DIR / "M5_spatial_config2_E_d1.pkl.gz",
         "E_c1": C.PER_ITEM_DIR / "M5_spatial_config2_E_c1.pkl.gz"}
out = {}
for k, f in files.items():
    d = pickle.load(gzip.open(f))
    agg = collections.defaultdict(lambda: collections.Counter())
    for x in d:
        a = agg[art.get(x["gold_id"], "?")]
        a["n"] += 1
        if x["n_candidates"] == 0: a["nocand"] += 1
        elif x["rank"] is None: a["goldnot"] += 1
        elif x["rank"] == 1: a["c1"] += 1
        else: a["wrong"] += 1
    out[k] = agg
rows = []
for kl, a in sorted(out["E_d1"].items(), key=lambda t: -t[1]["n"]):
    b = out["E_c1"][kl]
    rows.append([kl, a["n"], f'{a["c1"]/a["n"]:.4f}', f'{a["wrong"]/a["n"]:.4f}', f'{b["wrong"]/b["n"]:.4f}',
                 f'{a["nocand"]/a["n"]:.4f}', f'{a["goldnot"]/a["n"]:.4f}'])
with open(C.TABLES_DIR / "klassenprofil_E_d1.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["objektart","n","acc1_M5_E_d1","wrong_rank_M5_E_d1","wrong_rank_M5_E_c1","no_candidate_rate","gold_not_in_candidates_rate"]); w.writerows(rows)
for r in rows[:22]: print(r)
print("total", sum(a["n"] for a in out["E_d1"].values()), "unknown", out["E_d1"]["?"]["n"])
