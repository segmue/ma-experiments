"""
Verifikation des dritten Systems `geo_strtree` gegen die bestehende
DuckDB-Referenz `geo` — VOR den eigentlichen Messlaeufen auszufuehren.

Prueft drei Dinge:

  1. Ergebnisgleichheit. Fuer eine stratifizierte Stichprobe (PKT/LIN/PLY)
     liefern beide Engines fuer alle vier Methoden dieselben Zeilen in
     derselben Reihenfolge (feature_id-Sequenzen zeichengleich; das
     Overlap-Mass zusaetzlich numerisch verglichen).
  2. Indexwirksamkeit. Der STRtree-Ersatz fuer den EXPLAIN-Check: Wie viele
     der n Objekte beruehrt eine Abfrage ueberhaupt? Berichtet wird die
     Bounding-Box-Vorauswahl des Baums (tree.query ohne Praedikat) gegen die
     Gesamtzahl. Ein Wert weit unter 100 % belegt, dass der Index greift —
     das Gegenstueck zu `rtree_index_scan: true`.
  3. Beschreibungsparitaet. Fuer eine Stichprobe des Stage-03-Workloads
     erzeugen beide Engines ueber denselben BatchSentenceGenerator Saetze;
     gefordert ist eine Rate zeichengleicher Saetze von 1.0.

Aufruf (aus 2_benchmark/):
    python verify_geo_strtree.py
    python verify_geo_strtree.py --config config1 --n-single 10 --n-batch 50
    python verify_geo_strtree.py --skip-sentences

Output: <RESULTS_DIR>/geo_strtree_verification.json (RESULTS_DIR folgt
BENCH_VARIANT). Exit-Code 0 nur, wenn alles besteht.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time

import duckdb
import numpy as np

import bench_config as C

C.add_paths()

from engine_geo import GeometricEngine  # noqa: E402
from geo_strtree import StrtreeGeometricEngine  # noqa: E402
from geoparser_h3_resolver.pipeline.build_config import BuildConfig  # noqa: E402
from geoparser_h3_resolver.sentence_generator import (  # noqa: E402
    CandidateSentenceGenerator,
    FeatureInput,
)
from geoparser_h3_resolver.sentence_generator.batch_generator import (  # noqa: E402
    BatchSentenceGenerator,
)

ITEMS_BY_CFG = {"config1": "items_E_c1.pkl", "config2": "items_E_c2.pkl"}


# ---------------------------------------------------------------- Helfer ----
def _strata(conn, n_per_type: int) -> list[int]:
    ids: list[int] = []
    for pat in ("%PKT%", "%LIN%", "%PLY%"):
        rows = conn.execute(f"""
            SELECT feature_id FROM features_geo
            WHERE source LIKE '{pat}' AND NAME IS NOT NULL
            ORDER BY hash(feature_id + {C.SEED}) LIMIT {n_per_type}
        """).fetchall()
        ids += [int(r[0]) for r in rows]
    return ids


def _top_objektarten(conn, n: int = 2) -> list[str]:
    rows = conn.execute(
        "SELECT OBJEKTART FROM features_geo WHERE OBJEKTART IS NOT NULL "
        f"GROUP BY OBJEKTART ORDER BY COUNT(*) DESC, OBJEKTART LIMIT {n}"
    ).fetchall()
    return [r[0] for r in rows]


def _key(df, cols) -> list[tuple]:
    """Zeilenfolge als vergleichbare Tupel (Reihenfolge ist Teil des Vertrags)."""
    out = []
    for row in df[cols].itertuples(index=False, name=None):
        out.append(tuple(int(v) if isinstance(v, (int, np.integer)) else
                         (None if v is None or v != v else str(v)) for v in row))
    return out


def _cmp(label: str, a_df, b_df, cols, report: dict) -> bool:
    ka, kb = _key(a_df, cols), _key(b_df, cols)
    ok = ka == kb
    entry = {"ok": ok, "n_geo": len(ka), "n_geo_strtree": len(kb)}
    if not ok:
        sa, sb = set(ka), set(kb)
        entry["only_geo"] = [list(x) for x in sorted(sa - sb)[:5]]
        entry["only_geo_strtree"] = [list(x) for x in sorted(sb - sa)[:5]]
        entry["same_set_wrong_order"] = sa == sb
    report[label] = entry
    print(f"    {'ok  ' if ok else 'FAIL'} {label}: {len(ka)} vs {len(kb)} Zeilen")
    return ok


def _cmp_measure(label: str, a_df, b_df, report: dict) -> bool:
    if a_df.empty and b_df.empty:
        report[label] = {"ok": True, "max_rel_diff": 0.0}
        return True
    if len(a_df) != len(b_df):
        report[label] = {"ok": False, "reason": "Zeilenzahl"}
        return False
    a = a_df["overlap_cells"].to_numpy(dtype=float)
    b = b_df["overlap_cells"].to_numpy(dtype=float)
    denom = np.maximum(np.abs(a), 1e-12)
    rel = float(np.max(np.abs(a - b) / denom)) if len(a) else 0.0
    ok = rel < 1e-6
    report[label] = {"ok": ok, "max_rel_diff": rel}
    print(f"    {'ok  ' if ok else 'FAIL'} {label}: max. rel. Abweichung {rel:.3e}")
    return ok


# ------------------------------------------------------------- Workload ----
def workload_features(cfg: str) -> list[FeatureInput]:
    """Identisch zu 03_descriptions_parity.workload_features."""
    with open(C.EVAL3_DIR / "cache" / ITEMS_BY_CFG[cfg], "rb") as f:
        items = pickle.load(f)["items"]
    doc_ids = C.sample_doc_ids(it["doc_id"] for it in items)
    items = [it for it in items if it["doc_id"] in doc_ids]
    uuids = sorted({u for it in items for u in it["candidate_ids"]})
    conn = duckdb.connect(str(C.h3_db(cfg)), read_only=True)
    feats = []
    for i in range(0, len(uuids), 1000):
        chunk = uuids[i:i + 1000]
        ph = ", ".join("?" for _ in chunk)
        rows = conn.execute(f"""
            SELECT UUID, feature_id, NAME, OBJEKTART FROM features
            WHERE UUID IN ({ph})
            QUALIFY ROW_NUMBER() OVER (PARTITION BY UUID ORDER BY feature_id) = 1
        """, chunk).fetchall()
        feats += [FeatureInput(feature_id=r[1], name=r[2], objektart=r[3]) for r in rows]
    conn.close()
    feats.sort(key=lambda f: f.feature_id)
    return feats


def sentence_cfg(cfg: str):
    matrix = C.b1_matrix(cfg)
    return BuildConfig.from_yaml(C.config_yaml(cfg)).to_sentence_generator_config(matrix)


# ---------------------------------------------------------------- Ablauf ----
def verify(cfg: str, args) -> dict:
    print(f"[verify] {cfg}")
    report: dict = {"config": cfg}
    meta = duckdb.connect(str(C.geo_db(cfg)), read_only=True)
    single_ids = _strata(meta, args.n_single)
    objektarten = _top_objektarten(meta)
    batch_ids = [int(r[0]) for r in meta.execute(f"""
        SELECT feature_id FROM features_geo WHERE NAME IS NOT NULL
        ORDER BY hash(feature_id + {C.SEED + 1}) LIMIT {args.n_batch}
    """).fetchall()]
    meta.close()
    report["objektart_filter"] = objektarten
    report["n_single_ids"] = len(single_ids)
    report["n_batch_ids"] = len(batch_ids)

    ok = True
    t0 = time.perf_counter()
    with StrtreeGeometricEngine(C.geo_db(cfg)) as tree_eng, \
            GeometricEngine(C.geo_db(cfg), mode="buffered") as duck_eng:
        report["build_stats"] = tree_eng.build_stats

        # 1 — Ergebnisgleichheit, Einzelpfad
        eq: dict = {}
        for fid in single_ids:
            for label, call in (
                ("intersect_filtered", lambda e, f=None: e.find_intersecting_features(
                    f, objektart_list=objektarten, exclude_id=f)),
                ("intersect_plain", lambda e, f=None: e.find_intersecting_features(
                    f, exclude_id=f, max_results=20)),
                ("overlap_gemeinde", lambda e, f=None: e.find_overlapping_features(
                    f, objektart="Gemeindegebiet", max_results=2)),
            ):
                a = call(duck_eng, fid).df()
                b = call(tree_eng, fid).df()
                cols = (["feature_id", "NAME", "OBJEKTART"]
                        if label == "overlap_gemeinde"
                        else ["feature_id", "NAME", "OBJEKTART", "source", "UUID"])
                ok &= _cmp(f"single/{label}/fid={fid}", a, b, cols, eq)
                if label == "overlap_gemeinde":
                    ok &= _cmp_measure(f"single/{label}/fid={fid}/measure", a, b, eq)
        report["single_path"] = eq

        # 2 — Ergebnisgleichheit, Batchpfad (der strittige Messpunkt)
        eqb: dict = {}
        print(f"    Batch ueber {len(batch_ids)} Quellen (geo-Seite dauert Minuten) ...",
              flush=True)
        a = duck_eng.find_intersecting_features_batch(
            batch_ids, objektart_list=objektarten, exclude_self=True).df()
        b = tree_eng.find_intersecting_features_batch(
            batch_ids, objektart_list=objektarten, exclude_self=True).df()
        ok &= _cmp("batch/intersect_filtered", a, b,
                   ["src_id", "feature_id", "NAME", "OBJEKTART", "source", "UUID"], eqb)
        a = duck_eng.find_overlapping_features_batch(
            batch_ids, objektart="Gemeindegebiet", max_results=2).df()
        b = tree_eng.find_overlapping_features_batch(
            batch_ids, objektart="Gemeindegebiet", max_results=2).df()
        ok &= _cmp("batch/overlap_gemeinde", a, b,
                   ["src_id", "feature_id", "NAME", "OBJEKTART"], eqb)
        ok &= _cmp_measure("batch/overlap_gemeinde/measure", a, b, eqb)
        report["batch_path"] = eqb

        # 3 — Indexwirksamkeit (STRtree-Gegenstueck zum EXPLAIN-Check)
        n_total = tree_eng.build_stats["n_features"]
        fractions = []
        for fid in single_ids:
            pos = tree_eng._pos_of(fid)
            bbox_hits = len(tree_eng._tree.query(tree_eng._geoms[pos]))
            fractions.append(bbox_hits / n_total)
        report["index_evidence"] = {
            "n_features": n_total,
            "bbox_candidates_median": float(np.median(fractions)) * n_total,
            "scanned_fraction_median": round(float(np.median(fractions)), 8),
            "scanned_fraction_max": round(float(np.max(fractions)), 8),
            "note": ("Anteil der Objekte, die die Baumabfrage ueberhaupt beruehrt. "
                     "Ein Wert weit unter 1.0 ist der STRtree-Beleg dafuer, dass der "
                     "Index greift; das Gegenstueck zu explain_checks.rtree_index_scan."),
        }
        print(f"    Indexwirksamkeit: Median {report['index_evidence']['scanned_fraction_median']:.2e} "
              f"der {n_total} Objekte beruehrt")
    report["equality_seconds"] = round(time.perf_counter() - t0, 1)

    # 4 — Beschreibungsparitaet
    if not args.skip_sentences:
        feats = workload_features(cfg)
        step = max(1, len(feats) // args.n_sentences)
        sample = feats[::step][:args.n_sentences]
        scfg = sentence_cfg(cfg)
        sents = {}
        for name, mk in (("geo", lambda: GeometricEngine(C.geo_db(cfg), mode="buffered")),
                         ("geo_strtree", lambda: StrtreeGeometricEngine(C.geo_db(cfg)))):
            print(f"    Saetze ({name}, {len(sample)} Objekte) ...", flush=True)
            t1 = time.perf_counter()
            with mk() as engine:
                gen = CandidateSentenceGenerator(engine, scfg)
                BatchSentenceGenerator(gen).precompute(sample)
                sents[name] = {fid: g.sentence for fid, g in gen._cache.items()}
            print(f"      {time.perf_counter() - t1:.1f}s")
        ids = sorted(set(sents["geo"]) & set(sents["geo_strtree"]))
        identical = sum(1 for i in ids if sents["geo"][i] == sents["geo_strtree"][i])
        rate = identical / len(ids) if ids else 0.0
        diffs = [{"feature_id": i, "geo": sents["geo"][i],
                  "geo_strtree": sents["geo_strtree"][i]}
                 for i in ids if sents["geo"][i] != sents["geo_strtree"][i]][:5]
        report["sentence_parity"] = {
            "n_features": len(ids), "identical_sentence_rate": round(rate, 4),
            "examples_differing": diffs,
        }
        ok &= (rate == 1.0)
        print(f"    {'ok  ' if rate == 1.0 else 'FAIL'} Satzparitaet geo vs geo_strtree: "
              f"{rate:.4f} ueber {len(ids)} Objekte")

    report["all_ok"] = bool(ok)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, choices=list(C.CONFIGS))
    ap.add_argument("--n-single", type=int, default=7, help="Features je Stratum")
    ap.add_argument("--n-batch", type=int, default=50)
    ap.add_argument("--n-sentences", type=int, default=100)
    ap.add_argument("--skip-sentences", action="store_true")
    args = ap.parse_args()

    C.tee_log("verify_geo_strtree")
    configs = [args.config] if args.config else list(C.CONFIGS)
    out = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "configs": {}}
    all_ok = True
    for cfg in configs:
        rep = verify(cfg, args)
        out["configs"][cfg] = rep
        all_ok &= rep["all_ok"]
    out["all_ok"] = bool(all_ok)

    path = C.RESULTS_DIR / "geo_strtree_verification.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[verify] {'BESTANDEN' if all_ok else 'FEHLGESCHLAGEN'} -> {path}")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
