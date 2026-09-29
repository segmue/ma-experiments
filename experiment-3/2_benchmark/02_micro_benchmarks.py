"""
Stage 02: Mikro-Benchmarks der Engine-Methoden.

Vergleicht pro Config (config1/config2) drei Systeme:
  - h3           : H3Engine (Multi-Resolution-Zell-Lookup)
  - geo          : GeometricEngine, buffered-Modus (RTREE + ST_Intersects)
  - geo_dwithin  : GeometricEngine, dwithin-Modus (Rohgeometrien + ST_DWithin)

Messpunkte (alle warm, nach Warm-up):
  Einzelpfad, pro Geometrietyp-Stratum (PKT/LIN/PLY, je STRATA_N Features):
    - overlap_gemeinde   : find_overlapping_features(objektart='Gemeindegebiet', max_results=2)
    - intersect_filtered : find_intersecting_features(objektart_list=Top-2-OBJEKTARTen,
                           exclude_id) -- entspricht dem Generator-Hot-Path
    - intersect_plain    : find_intersecting_features(exclude_id, max_results=20) (ungefiltert)
  Batch (BATCH_SIZE_BENCH gemischte Features, N_REPEATS Wiederholungen):
    - batch_overlap_gemeinde, batch_intersect_filtered

Latenz-Statistik: Median/IQR (p25/p75) ueber die Einzel-Calls bzw. Repeats.
Output: results/micro.json + results/tables/micro.csv
"""

from __future__ import annotations

import json
import statistics
import time

import duckdb

import bench_config as C

C.add_paths()

from engine_geo import GeometricEngine  # noqa: E402
from geo_strtree import StrtreeGeometricEngine  # noqa: E402
from h3_multi_resolution_engine.engine import H3Engine  # noqa: E402


def _percentiles(samples: list[float]) -> dict:
    s = sorted(samples)
    n = len(s)
    return {
        "n": n,
        "median_ms": round(1000 * statistics.median(s), 2),
        "p25_ms": round(1000 * s[max(0, int(0.25 * n) - 1)], 2),
        "p75_ms": round(1000 * s[min(n - 1, int(0.75 * n))], 2),
        "mean_ms": round(1000 * statistics.fmean(s), 2),
    }


def _strata(meta_conn) -> dict[str, list[int]]:
    out = {}
    for label, pat in (("PKT", "%PKT%"), ("LIN", "%LIN%"), ("PLY", "%PLY%")):
        rows = meta_conn.execute(f"""
            SELECT feature_id FROM features
            WHERE source LIKE '{pat}' AND NAME IS NOT NULL
            ORDER BY hash(feature_id + {C.SEED}) LIMIT {C.STRATA_N}
        """).fetchall()
        out[label] = [r[0] for r in rows]
    return out


def _batch_ids(meta_conn) -> list[int]:
    rows = meta_conn.execute(f"""
        SELECT feature_id FROM features
        WHERE NAME IS NOT NULL
        ORDER BY hash(feature_id + {C.SEED + 1}) LIMIT {C.BATCH_SIZE_BENCH}
    """).fetchall()
    return [r[0] for r in rows]


def _top_objektarten(meta_conn, n: int = 2) -> list[str]:
    rows = meta_conn.execute(
        "SELECT OBJEKTART FROM features WHERE OBJEKTART IS NOT NULL "
        f"GROUP BY OBJEKTART ORDER BY COUNT(*) DESC, OBJEKTART LIMIT {n}"
    ).fetchall()
    return [r[0] for r in rows]


def bench_engine(engine, strata, batch_ids, objektarten) -> dict:
    res: dict = {}

    def timed(fn) -> float:
        t0 = time.perf_counter()
        fn().df()
        return time.perf_counter() - t0

    single_points = {
        "overlap_gemeinde": lambda fid: engine.find_overlapping_features(
            fid, objektart="Gemeindegebiet", max_results=2
        ),
        "intersect_filtered": lambda fid: engine.find_intersecting_features(
            fid, objektart_list=objektarten, exclude_id=fid
        ),
        "intersect_plain": lambda fid: engine.find_intersecting_features(
            fid, exclude_id=fid, max_results=20
        ),
    }
    for name, call in single_points.items():
        for stratum, fids in strata.items():
            t_point = time.perf_counter()
            for fid in fids[: C.N_WARMUP]:  # Warm-up (zaehlt aufs Budget)
                try:
                    call(fid).df()
                except Exception:
                    pass
                if time.perf_counter() - t_point > C.POINT_TIME_BUDGET_S:
                    break
            samples = []
            for fid in fids:
                if (len(samples) >= C.MIN_POINT_SAMPLES
                        and time.perf_counter() - t_point > C.POINT_TIME_BUDGET_S):
                    break
                try:
                    samples.append(timed(lambda f=fid: call(f)))
                except Exception as err:
                    print(f"    Fehler {name}/{stratum}/fid={fid}: {err}")
            if samples:
                stats = _percentiles(samples)
                if len(samples) < len(fids):
                    stats["truncated"] = True  # Zeitbudget erreicht
                res[f"{name}__{stratum}"] = stats
            print(f"    {name}__{stratum}: {res.get(f'{name}__{stratum}')}")

    batch_points = {
        "batch_overlap_gemeinde": lambda: engine.find_overlapping_features_batch(
            batch_ids, objektart="Gemeindegebiet", max_results=2
        ),
        "batch_intersect_filtered": lambda: engine.find_intersecting_features_batch(
            batch_ids, objektart_list=objektarten, exclude_self=True
        ),
    }
    for name, call in batch_points.items():
        try:
            call().df()  # Warm-up
            t_point = time.perf_counter()
            samples = []
            for _ in range(C.N_REPEATS):
                samples.append(timed(call))
                if (len(samples) >= C.MIN_BATCH_REPEATS
                        and time.perf_counter() - t_point > C.POINT_TIME_BUDGET_S):
                    break
            stats = _percentiles(samples)
            stats["per_feature_ms"] = round(stats["median_ms"] / len(batch_ids), 3)
            if len(samples) < C.N_REPEATS:
                stats["truncated"] = True  # Zeitbudget erreicht
            res[name] = stats
            print(f"    {name}: {stats}")
        except Exception as err:
            print(f"    Fehler {name}: {err}")
            res[name] = {"error": str(err)}
    return res


def main() -> None:
    C.tee_log("02_micro")
    results: dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": {
            "strata_n": C.STRATA_N, "n_repeats": C.N_REPEATS,
            "batch_size": C.BATCH_SIZE_BENCH, "seed": C.SEED,
        },
        "configs": {},
    }

    for cfg in C.CONFIGS:
        print(f"[02] {cfg}")
        meta = duckdb.connect(str(C.h3_db(cfg)), read_only=True)
        strata = _strata(meta)
        batch_ids = _batch_ids(meta)
        objektarten = _top_objektarten(meta)
        meta.close()
        results["configs"][cfg] = {"objektart_filter": objektarten, "systems": {}}

        systems = {
            "h3": lambda: H3Engine(C.h3_db(cfg)),
            "geo": lambda: GeometricEngine(C.geo_db(cfg), mode="buffered"),
            "geo_strtree": lambda: StrtreeGeometricEngine(C.geo_db(cfg)),
            "geo_dwithin": lambda: GeometricEngine(C.geo_db(cfg), mode="dwithin"),
        }
        for sys_name, mk in systems.items():
            ckpt = C.CACHE_DIR / f"micro_{cfg}_{sys_name}.json"
            cached = C.load_json(ckpt)
            if cached is not None:
                print(f"  System {sys_name}: [skip] Checkpoint {ckpt.name}")
                results["configs"][cfg]["systems"][sys_name] = cached
                continue
            print(f"  System {sys_name}:")
            with mk() as engine:
                res = bench_engine(engine, strata, batch_ids, objektarten)
            C.save_json_atomic(ckpt, res)
            results["configs"][cfg]["systems"][sys_name] = res

    out = C.RESULTS_DIR / "micro.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Flache CSV-Tabelle
    import csv
    with open(C.TABLES_DIR / "micro.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "system", "measurement", "median_ms", "p25_ms",
                    "p75_ms", "mean_ms", "n", "per_feature_ms"])
        for cfg, cdata in results["configs"].items():
            for sys_name, sdata in cdata["systems"].items():
                for m, v in sdata.items():
                    if "error" in v:
                        continue
                    w.writerow([cfg, sys_name, m, v["median_ms"], v["p25_ms"],
                                v["p75_ms"], v["mean_ms"], v["n"],
                                v.get("per_feature_ms", "")])
    print(f"[02] Fertig -> {out}")


if __name__ == "__main__":
    main()
