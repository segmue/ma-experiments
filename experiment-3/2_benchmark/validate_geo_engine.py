"""
Validierung der GeometricEngine gegen eine unabhaengige shapely-Nachrechnung.

Fuer 20 zufaellige Quell-Features (stratifiziert: Punkte/Linien/Polygone)
wird find_intersecting_features (buffered-Modus, ungefiltert) gegen eine
Python-seitige Referenz verglichen: Kandidaten im grosszuegigen Bounding-
Box-Umkreis werden mit ihren ROHGEOMETRIEN + Radien geladen, in shapely
unabhaengig gepuffert und exakt geschnitten. Gefordert: identische
feature_id-Mengen (Engine vs Referenz).
"""

from __future__ import annotations

import duckdb
import shapely
from shapely import wkb as swkb

import bench_config as C

C.add_paths()
from engine_geo import GeometricEngine  # noqa: E402

N_PER_TYPE = 7


def validate(cfg: str) -> int:
    db = C.geo_db(cfg)
    eng = GeometricEngine(db, mode="buffered")
    raw = duckdb.connect(str(db), read_only=True)
    raw.execute("INSTALL spatial; LOAD spatial;")

    # Marge MUSS den groessten Kandidaten-Puffer abdecken: auch weit entfernte
    # Features koennen mit grossem eigenem Puffer in die Quelle hineinreichen.
    bbox_margin = raw.execute(
        "SELECT MAX(buffer_radius) FROM features_geo"
    ).fetchone()[0] + 1.0

    srcs = []
    for pat in ("%PKT%", "%LIN%", "%PLY%"):
        rows = raw.execute(f"""
            SELECT feature_id FROM features_geo
            WHERE source LIKE '{pat}' AND NAME IS NOT NULL
            ORDER BY hash(feature_id + 42) LIMIT {N_PER_TYPE}
        """).fetchall()
        srcs += [r[0] for r in rows]

    mismatches = 0
    for fid in srcs:
        g_hex, radius = raw.execute(
            "SELECT ST_AsHEXWKB(geom), buffer_radius FROM features_geo "
            "WHERE feature_id = ?", [fid],
        ).fetchone()
        src_geom = shapely.buffer(
            shapely.make_valid(swkb.loads(bytes.fromhex(g_hex))), radius
        )
        minx, miny, maxx, maxy = src_geom.bounds

        # Referenz-Kandidaten: alles im Bounding-Box-Umkreis (eigener Code-Pfad)
        cand = raw.execute(f"""
            SELECT feature_id, ST_AsHEXWKB(geom), buffer_radius
            FROM features_geo
            WHERE NAME IS NOT NULL
              AND ST_XMax(geom) >= {minx - bbox_margin}
              AND ST_XMin(geom) <= {maxx + bbox_margin}
              AND ST_YMax(geom) >= {miny - bbox_margin}
              AND ST_YMin(geom) <= {maxy + bbox_margin}
        """).fetchall()
        expected = set()
        for cid, c_hex, c_rad in cand:
            if cid == fid:
                continue
            c_geom = shapely.buffer(
                shapely.make_valid(swkb.loads(bytes.fromhex(c_hex))), c_rad
            )
            if src_geom.intersects(c_geom):
                expected.add(cid)

        got = {
            r[0]
            for r in eng.find_intersecting_features(
                fid, exclude_id=fid
            ).fetchall()
        }
        if got != expected:
            mismatches += 1
            print(f"MISMATCH {cfg} fid={fid}:")
            print(f"  nur Engine:   {sorted(got - expected)[:10]}")
            print(f"  nur Referenz: {sorted(expected - got)[:10]}")

    raw.close()
    eng.close()
    print(f"[{cfg}] {len(srcs)} Quell-Features geprueft, Mismatches: {mismatches}")
    return mismatches


if __name__ == "__main__":
    import sys
    total = sum(validate(cfg) for cfg in C.CONFIGS)
    sys.exit(0 if total == 0 else 1)
