"""
Validierung der V2-Optimierung (Equi-Join-Umformulierung im Einzelpfad der H3Engine).

Verfahren (gleich wie bei der Batch-Validierung am 11.6.):
  1. VOR dem Umbau:  python validate_singlepath_v2.py snapshot
     -> zieht 2x50 zufaellige Features (config1+config2) und speichert die
        Ergebnisse von find_intersecting_features / find_overlapping_features
        (mehrere Parameter-Varianten) als Baseline-Pickle.
  2. NACH dem Umbau: python validate_singlepath_v2.py compare
     -> berechnet dieselben Aufrufe neu und vergleicht zeilenweise
        (inkl. overlap_cells-Werte). Gefordert: 0 Mismatches.

Beide Modi messen die Zeit pro Aufruf -> Vorher/Nachher-Timing im selben Log.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import config as C
from h3_multi_resolution_engine.engine import H3Engine

BASELINE = C.CACHE_DIR / "v2_singlepath_baseline.pkl"
N_SAMPLES = 50


def _sample_features(engine: H3Engine, n: int) -> list[int]:
    rows = engine.conn.execute(
        f"SELECT feature_id FROM features ORDER BY hash(feature_id + 42) LIMIT {n}"
    ).fetchall()
    return [r[0] for r in rows]


def _top_objektarten(engine: H3Engine, n: int = 2) -> list[str]:
    rows = engine.conn.execute(
        "SELECT OBJEKTART FROM features WHERE OBJEKTART IS NOT NULL "
        "GROUP BY OBJEKTART ORDER BY COUNT(*) DESC, OBJEKTART LIMIT ?",
        [n],
    ).fetchall()
    return [r[0] for r in rows]


def collect(cfg_name: str) -> tuple[dict, dict]:
    """Liefert ({key: [row-tuples]}, {methode: (anzahl_calls, total_sekunden)})."""
    db = C.duckdb_path(cfg_name)
    results: dict = {}
    timings: dict = {}

    with H3Engine(db) as engine:
        fids = _sample_features(engine, N_SAMPLES)
        objektarten = _top_objektarten(engine)

        def run(key_suffix: str, fn):
            t0 = time.perf_counter()
            df = fn().df()
            dt = time.perf_counter() - t0
            calls, total = timings.get(key_suffix, (0, 0.0))
            timings[key_suffix] = (calls + 1, total + dt)
            return [tuple(r) for r in df.itertuples(index=False, name=None)]

        for fid in fids:
            results[(cfg_name, fid, "overlap_gemeinde")] = run(
                "overlap_gemeinde",
                lambda: engine.find_overlapping_features(
                    fid, objektart="Gemeindegebiet", max_results=2
                ),
            )
            results[(cfg_name, fid, "overlap_unfiltered")] = run(
                "overlap_unfiltered",
                lambda: engine.find_overlapping_features(fid, max_results=5),
            )
            results[(cfg_name, fid, "intersect_plain")] = run(
                "intersect_plain",
                lambda: engine.find_intersecting_features(
                    fid, exclude_id=fid, max_results=20
                ),
            )
            results[(cfg_name, fid, "intersect_filtered_sized")] = run(
                "intersect_filtered_sized",
                lambda: engine.find_intersecting_features(
                    fid,
                    objektart_list=objektarten,
                    exclude_id=fid,
                    order_by_size=True,
                    max_results=10,
                ),
            )

    return results, timings


def print_timings(cfg_name: str, timings: dict) -> None:
    print(f"  Timing {cfg_name}:")
    for method, (calls, total) in sorted(timings.items()):
        print(f"    {method:28s} {calls:4d} Calls, {1000*total/calls:8.1f} ms/Call")


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("snapshot", "compare"):
        print("Usage: python validate_singlepath_v2.py {snapshot|compare}")
        return 2

    all_results: dict = {}
    for cfg_name in ("config1", "config2"):
        print(f"[{mode}] {cfg_name} ...")
        results, timings = collect(cfg_name)
        print_timings(cfg_name, timings)
        all_results.update(results)

    if mode == "snapshot":
        C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(BASELINE, "wb") as f:
            pickle.dump(all_results, f)
        print(f"\nBaseline gespeichert: {BASELINE} ({len(all_results)} Aufrufe)")
        return 0

    # compare
    with open(BASELINE, "rb") as f:
        baseline = pickle.load(f)

    assert set(baseline.keys()) == set(all_results.keys()), "Key-Mengen differieren!"
    mismatches = 0
    for key in sorted(baseline.keys(), key=str):
        if baseline[key] != all_results[key]:
            mismatches += 1
            print(f"MISMATCH {key}:")
            print(f"  alt: {baseline[key]}")
            print(f"  neu: {all_results[key]}")

    print(f"\nVerglichen: {len(baseline)} Aufrufe, Mismatches: {mismatches}")
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
