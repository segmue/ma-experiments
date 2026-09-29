"""
Build der geometrischen Zwillings-Datenbanken (Experiment 3).

Erzeugt pro Config (config1/config2) eine DuckDB-spatial-Datenbank
`spatial_geo.duckdb` mit Tabelle:

    features_geo(feature_id PK, UUID, NAME, OBJEKTART, source, h3_resolution,
                 buffer_radius, geom GEOMETRY, geom_buffered GEOMETRY)
    + RTREE-Indexe auf geom_buffered und geom, ART-Index auf UUID.

Design-Garantien (Vergleichbarkeit mit der H3-DB):
  - **Feature-Alignment**: Die Metadaten (feature_id, UUID, NAME, OBJEKTART,
    source, h3_resolution) werden per ATTACH 1:1 aus der bestehenden H3-DB
    uebernommen -> identische feature_ids in beiden Welten. Die Geometrien
    werden mit DENSELBEN Lese-Funktionen wie der H3-Build gelesen
    (geoparser_h3_resolver.pipeline.build) und per Pflicht-Assert zeilenweise
    gegen die H3-DB-Sequenz geprueft (UUID, NAME, OBJEKTART, source).
  - **Footprint-Aequivalenz**: buffer_radius = mittlere H3-Zellkantenlaenge
    der pro Feature in der H3-DB zugewiesenen Aufloesung (= Umkreisradius des
    Hexagons). geom_buffered = ST_Buffer(make_valid(geom), r). Der Zwilling
    erhaelt damit exakt die raeumliche Toleranz, die die H3-Diskretisierung
    implizit erzeugt (75% der Features sind Punkte!).
  - Geometrien bleiben in LV95/EPSG:2056 (planar, Meter) — schnellste und
    genaueste Variante fuer die Schweiz, fair zugunsten des Zwillings.

Aufruf (aus 2_benchmark/01_build_geo_dbs.py oder direkt):
    python build_geo.py config1 [config2 ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import h3
import numpy as np
import pandas as pd
import shapely

from geoparser_h3_resolver.pipeline.build import (
    _discover_sources,
    _get_geoparser_db_path,
    _read_all_features,
)

GAZETTEER = "swissnames3d"


def _edge_length_m(resolution: int) -> float:
    """Mittlere H3-Zellkantenlaenge (= Hexagon-Umkreisradius) in Metern."""
    try:
        return float(h3.average_hexagon_edge_length(int(resolution), unit="m"))
    except AttributeError:  # aeltere h3-Versionen
        return float(h3.edge_length(int(resolution), unit="m"))


def read_source_geometries() -> pd.DataFrame:
    """Liest alle Quell-Geometrien EXAKT wie der H3-Build (gleiche Reihenfolge)."""
    geoparser_db = _get_geoparser_db_path()
    sources = _discover_sources(geoparser_db, GAZETTEER)
    print(f"  Quellen: {[s for s, _ in sources]}")
    df = _read_all_features(geoparser_db, sources)
    df = df.reset_index(drop=True)
    df["feature_id"] = range(len(df))
    return df


def _assert_alignment(geom_df: pd.DataFrame, meta_df: pd.DataFrame) -> None:
    """Pflicht-Check: Geometrie-Lesereihenfolge == H3-DB-Sequenz."""
    if len(geom_df) != len(meta_df):
        raise AssertionError(
            f"Feature-Anzahl differiert: Quellen {len(geom_df)} vs H3-DB {len(meta_df)}"
        )
    for col in ("UUID", "NAME", "OBJEKTART", "source"):
        a = geom_df[col].fillna("\x00").to_numpy()
        b = meta_df[col].fillna("\x00").to_numpy()
        neq = a != b
        if neq.any():
            i = int(np.argmax(neq))
            raise AssertionError(
                f"Alignment-Bruch in Spalte {col} bei feature_id {i}: "
                f"Quelle={a[i]!r} vs H3-DB={b[i]!r}"
            )
    print(f"  Alignment-Assert OK ({len(geom_df)} Features, 4 Spalten)")


def build_geo_db(
    h3_db_path: Path,
    output_path: Path,
    geom_df: pd.DataFrame | None = None,
) -> dict:
    """Baut eine Geo-Zwillings-DB zur gegebenen H3-DB. Liefert Timing/Stats."""
    stats: dict = {"h3_db": str(h3_db_path), "output": str(output_path)}
    t_total = time.perf_counter()

    if geom_df is None:
        print("  Lese Quell-Geometrien...")
        t0 = time.perf_counter()
        geom_df = read_source_geometries()
        stats["read_s"] = time.perf_counter() - t0

    # Metadaten + h3_resolution aus der H3-DB uebernehmen (Alignment-Quelle)
    h3_conn = duckdb.connect(str(h3_db_path), read_only=True)
    meta_df = h3_conn.execute(
        "SELECT feature_id, UUID, NAME, OBJEKTART, source, h3_resolution "
        "FROM features ORDER BY feature_id"
    ).df()
    h3_conn.close()
    _assert_alignment(geom_df, meta_df)

    # Pufferradius pro Feature aus der zugewiesenen H3-Aufloesung
    res_values = sorted(meta_df["h3_resolution"].unique())
    radius_by_res = {int(r): _edge_length_m(int(r)) for r in res_values}
    radii = meta_df["h3_resolution"].map(lambda r: radius_by_res[int(r)]).to_numpy()
    stats["radius_by_res_m"] = radius_by_res

    # Geometrien validieren + puffern (vektorisiert, GEOS via shapely 2)
    print("  Puffere Geometrien (make_valid + buffer)...")
    t0 = time.perf_counter()
    geoms = np.asarray(geom_df["geometry"].to_numpy(), dtype=object)
    geoms = shapely.force_2d(geoms)  # Quellen sind 3D (Z); Analyse ist planar 2D
    try:
        geoms_valid = shapely.make_valid(geoms)
    except Exception:
        geoms_valid = geoms  # Buffer repariert die meisten Invaliditaeten selbst
    buffered = shapely.buffer(geoms_valid, radii)
    geom_hex = shapely.to_wkb(geoms, hex=True)
    buf_hex = shapely.to_wkb(buffered, hex=True)
    stats["buffer_s"] = time.perf_counter() - t0

    insert_df = pd.DataFrame({
        "feature_id": meta_df["feature_id"].to_numpy(),
        "UUID": meta_df["UUID"].to_numpy(),
        "NAME": meta_df["NAME"].to_numpy(),
        "OBJEKTART": meta_df["OBJEKTART"].to_numpy(),
        "source": meta_df["source"].to_numpy(),
        "h3_resolution": meta_df["h3_resolution"].to_numpy(),
        "buffer_radius": radii,
        "geom_hex": geom_hex,
        "geom_buf_hex": buf_hex,
    })

    print("  Erstelle DuckDB (CTAS + RTREE-Indexe)...")
    t0 = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    conn = duckdb.connect(str(output_path))
    conn.execute("INSTALL spatial; LOAD spatial;")
    conn.register("insert_df", insert_df)
    conn.execute("""
        CREATE TABLE features_geo AS
        SELECT
            CAST(feature_id AS INTEGER) AS feature_id,
            UUID, NAME, OBJEKTART, source,
            CAST(h3_resolution AS TINYINT) AS h3_resolution,
            CAST(buffer_radius AS DOUBLE) AS buffer_radius,
            ST_GeomFromHEXWKB(geom_hex) AS geom,
            ST_GeomFromHEXWKB(geom_buf_hex) AS geom_buffered
        FROM insert_df
        ORDER BY feature_id
    """)
    conn.unregister("insert_df")
    # View 'features': der SpatialSentenceResolver macht seine UUID-Lookups auf
    # einer Tabelle dieses Namens — so funktioniert er unveraendert auf der Geo-DB.
    conn.execute("""
        CREATE VIEW features AS
        SELECT feature_id, UUID, NAME, OBJEKTART, source FROM features_geo
    """)
    conn.execute("CREATE UNIQUE INDEX idx_geo_fid ON features_geo(feature_id)")
    conn.execute("CREATE INDEX idx_geo_uuid ON features_geo(UUID)")
    conn.execute("CREATE INDEX idx_geo_rtree_buf ON features_geo USING RTREE (geom_buffered)")
    conn.execute("CREATE INDEX idx_geo_rtree_raw ON features_geo USING RTREE (geom)")
    n = conn.execute("SELECT COUNT(*) FROM features_geo").fetchone()[0]
    conn.close()
    stats["db_create_s"] = time.perf_counter() - t0
    stats["n_features"] = int(n)
    stats["total_s"] = time.perf_counter() - t_total
    stats["db_size_mb"] = round(output_path.stat().st_size / 1e6, 1)
    print(
        f"  Geo-DB erstellt: {n} Features, {stats['db_size_mb']} MB, "
        f"{stats['total_s']:.0f}s total"
    )
    return stats


if __name__ == "__main__":
    HERE = Path(__file__).resolve().parent               # experiment-3/1_geometric_resolver
    OUTPUT = HERE.parent.parent / "output"               # ma-experiments/output

    configs = sys.argv[1:] or ["config1", "config2"]
    geom_df = None
    all_stats = {}
    for cfg in configs:
        print(f"[build_geo] {cfg}")
        if geom_df is None:
            geom_df = read_source_geometries()
        all_stats[cfg] = build_geo_db(
            h3_db_path=OUTPUT / cfg / "spatial_h3.duckdb",
            output_path=OUTPUT / f"geo_{cfg}" / "spatial_geo.duckdb",
            geom_df=geom_df,
        )
    import json
    print(json.dumps(all_stats, indent=2, default=str))
