"""
Stage 01: Geo-Zwillings-DBs bauen + validieren.

1. build_geo.build_geo_db fuer config1 + config2 (eine Geometrie-Lesung,
   inkl. Pflicht-Alignment-Assert gegen die jeweilige H3-DB).
2. RTREE-EXPLAIN-Check: nutzt der Planner den RTREE-Index-Scan fuer das
   Einzelpfad-Pattern (konstante Geometrie)?
3. Groessenvergleich H3-DB vs Geo-DB.

Output: results/build.json
"""

from __future__ import annotations

import json
import time

import bench_config as C

C.add_paths()

import duckdb  # noqa: E402
from build_geo import build_geo_db, read_source_geometries  # noqa: E402


def rtree_explain_check(db_path) -> dict:
    """Prueft per EXPLAIN, ob RTREE-Index-Scans greifen."""
    conn = duckdb.connect(str(db_path), read_only=True)
    conn.execute("INSTALL spatial; LOAD spatial;")
    hex_buf = conn.execute(
        "SELECT ST_AsHEXWKB(geom_buffered) FROM features_geo "
        "ORDER BY feature_id LIMIT 1"
    ).fetchone()[0]
    checks = {}
    plans = {
        "single_intersects_const": f"""
            SELECT count(*) FROM features_geo f
            WHERE ST_Intersects(f.geom_buffered, ST_GeomFromHEXWKB('{hex_buf}'))
        """,
        "batch_spatial_join": """
            WITH src AS (
                SELECT feature_id AS src_id, geom_buffered AS g
                FROM features_geo WHERE feature_id IN (0, 1, 2)
            )
            SELECT count(*) FROM src s
            JOIN features_geo f ON ST_Intersects(s.g, f.geom_buffered)
        """,
    }
    for name, sql in plans.items():
        plan = "\n".join(str(r) for r in conn.execute("EXPLAIN " + sql).fetchall())
        checks[name] = {
            "rtree_index_scan": "RTREE" in plan.upper(),
            "spatial_join_operator": "SPATIAL_JOIN" in plan.upper().replace(" ", "_"),
        }
    conn.close()
    return checks


def main() -> None:
    results = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "configs": {}}

    print("[01] Lese Quell-Geometrien (einmal fuer beide Configs)...")
    t0 = time.perf_counter()
    geom_df = read_source_geometries()
    read_s = time.perf_counter() - t0
    print(f"  {len(geom_df)} Features in {read_s:.0f}s gelesen")

    for cfg in C.CONFIGS:
        print(f"[01] Baue geo_{cfg} ...")
        stats = build_geo_db(
            h3_db_path=C.h3_db(cfg),
            output_path=C.geo_db(cfg),
            geom_df=geom_df,
        )
        stats["read_s"] = read_s  # geteilte Lesezeit (einmalig)
        stats["h3_db_size_mb"] = round(C.h3_db(cfg).stat().st_size / 1e6, 1)
        print(f"[01] EXPLAIN-Checks geo_{cfg} ...")
        stats["explain_checks"] = rtree_explain_check(C.geo_db(cfg))
        results["configs"][cfg] = stats
        print(json.dumps(stats["explain_checks"], indent=2))

    out = C.RESULTS_DIR / "build.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[01] Fertig -> {out}")


if __name__ == "__main__":
    main()
