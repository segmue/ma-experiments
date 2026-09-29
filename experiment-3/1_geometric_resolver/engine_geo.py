"""
GeometricEngine — geometrischer Zwilling der H3Engine (Experiment 3).

Identische Schnittstelle wie h3_multi_resolution_engine.engine.H3Engine
(find_intersecting_features / find_overlapping_features + _batch-Varianten,
gleiche Rueckgabespalten, gleiche deterministische Sortierung), aber statt
H3-Zell-Lookups arbeiten die Queries auf einer DuckDB-spatial-Datenbank
(features_geo mit GEOMETRY-Spalten + RTREE-Indexen, LV95/EPSG:2056, Meter).

Damit laufen CandidateSentenceGenerator / BatchSentenceGenerator UNVERAENDERT
auf dieser Engine (duck-typed) — es variiert ausschliesslich die raeumliche
Abfragetechnik. Das ist die Isolations-Anforderung von Experiment 3.

Zwei Modi:
  - "buffered" (Hauptvariante): ST_Intersects auf vorgepufferten Geometrien
    (geom_buffered; Pufferradius = H3-Zellkantenlaenge der pro Feature
    zugewiesenen Aufloesung -> gleiche raeumliche Toleranz, die die
    H3-Diskretisierung implizit erzeugt). Overlap-Mass = Schnittflaeche (m²).
  - "dwithin" (Sensitivitaets-Variante, nur fuer Benchmarks): ST_DWithin auf
    den ROHGEOMETRIEN mit Distanz = Summe beider Pufferradien. Overlap-Mass =
    ST_Distance (aufsteigend gerankt). Wird NICHT fuer Beschreibungs-/
    Paritaets-Laeufe verwendet.

Performance-Detail (bewusst, fuer faire Vergleichbarkeit): Im Einzelpfad wird
die Quellgeometrie zuerst per feature_id geholt und als KONSTANTE
(ST_GeomFromHEXWKB) ins SQL inlined — nur mit konstanter Geometrie nutzt
DuckDBs Planner den RTREE-Index-Scan. Die Batch-Varianten formulieren einen
Spatial Join; ob der Planner dort einen RTree nutzt, wird im Benchmark per
EXPLAIN dokumentiert.

Die Spalte `overlap_cells` heisst aus Duck-Typing-Gruenden wie bei der
H3Engine, enthaelt hier aber die Schnittflaeche in m² (bzw. Distanz in m im
dwithin-Modus). Der Satzgenerator liest sie nicht — nur das Ranking zaehlt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import duckdb


class GeometricEngine:
    """Geometrischer Zwilling der H3Engine (DuckDB spatial + RTREE)."""

    def __init__(self, db_path: Union[str, Path], mode: str = "buffered"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Geo-Datenbank nicht gefunden: {db_path}")
        if mode not in ("buffered", "dwithin"):
            raise ValueError(f"Unbekannter Modus: {mode}")
        self.mode = mode
        self.conn = duckdb.connect(str(self.db_path), read_only=True)
        self.conn.execute("INSTALL spatial; LOAD spatial;")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "GeometricEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- intern ---------------------------------------------------------------

    def _has_geo_table(self) -> bool:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = 'features_geo'"
        ).fetchone()
        return bool(row and row[0])

    def _source_geom(self, feature_id: int) -> tuple[str, str, float] | None:
        """(hexwkb_buffered, hexwkb_raw, buffer_radius) des Quell-Features."""
        return self.conn.execute(
            "SELECT ST_AsHEXWKB(geom_buffered), ST_AsHEXWKB(geom), buffer_radius "
            "FROM features_geo WHERE feature_id = ?",
            [int(feature_id)],
        ).fetchone()

    # -- Einzelpfad -----------------------------------------------------------

    def find_intersecting_features(
        self,
        feature_id: int,
        objektart_list: list[str] | None = None,
        dataset: str | None = None,
        exclude_id: int | None = None,
        exclude_ids: list[int] | None = None,
        order_by_size: bool = False,
        max_results: int | None = None,
    ) -> duckdb.DuckDBPyRelation:
        """Wie H3Engine.find_intersecting_features, geometrisch.

        Returns:
            DuckDBPyRelation mit Spalten: feature_id, NAME, OBJEKTART, source, UUID
        """
        if not self._has_geo_table():
            raise RuntimeError("features_geo Tabelle nicht gefunden (build_geo.py).")

        src = self._source_geom(feature_id)
        if src is None:
            raise ValueError(f"Quell-Feature {feature_id} nicht gefunden.")
        hex_buf, hex_raw, src_radius = src

        where_parts = ["f.NAME IS NOT NULL"]
        if objektart_list:
            quoted = ", ".join(f"'{o}'" for o in objektart_list)
            where_parts.append(f"f.OBJEKTART IN ({quoted})")
        if dataset is not None:
            where_parts.append(f"f.source = '{dataset}'")
        if exclude_id is not None:
            where_parts.append(f"f.feature_id != {int(exclude_id)}")
        if exclude_ids:
            ids_str = ", ".join(str(int(i)) for i in exclude_ids)
            where_parts.append(f"f.feature_id NOT IN ({ids_str})")

        if self.mode == "buffered":
            predicate = (
                f"ST_Intersects(f.geom_buffered, ST_GeomFromHEXWKB('{hex_buf}'))"
            )
        else:
            predicate = (
                f"ST_DWithin(f.geom, ST_GeomFromHEXWKB('{hex_raw}'), "
                f"f.buffer_radius + {float(src_radius)})"
            )
        where_clause = " AND ".join([predicate] + where_parts)

        if order_by_size:
            order_clause = "ORDER BY ST_Area(f.geom_buffered) ASC, f.feature_id"
        else:
            order_clause = "ORDER BY f.feature_id"
        limit_clause = f"LIMIT {int(max_results)}" if max_results else ""

        sql = f"""
            SELECT f.feature_id, f.NAME, f.OBJEKTART, f.source, f.UUID
            FROM features_geo f
            WHERE {where_clause}
            {order_clause}
            {limit_clause}
        """
        return self.conn.sql(sql)

    def find_overlapping_features(
        self,
        feature_id: int,
        objektart: str | None = None,
        dataset: str | None = None,
        max_results: int = 5,
    ) -> duckdb.DuckDBPyRelation:
        """Wie H3Engine.find_overlapping_features, geometrisch.

        Ranking: Schnittflaeche der gepufferten Geometrien absteigend
        (Analog zu overlap_cells), Tie-Break feature_id. Im dwithin-Modus:
        Distanz aufsteigend.

        Returns:
            DuckDBPyRelation mit Spalten: feature_id, NAME, OBJEKTART, overlap_cells
        """
        if not self._has_geo_table():
            raise RuntimeError("features_geo Tabelle nicht gefunden (build_geo.py).")

        src = self._source_geom(feature_id)
        if src is None:
            raise ValueError(f"Quell-Feature {feature_id} nicht gefunden.")
        hex_buf, hex_raw, src_radius = src

        where_parts = ["f.NAME IS NOT NULL"]
        if objektart is not None:
            where_parts.append(f"f.OBJEKTART = '{objektart}'")
        if dataset is not None:
            where_parts.append(f"f.source = '{dataset}'")

        if self.mode == "buffered":
            predicate = (
                f"ST_Intersects(f.geom_buffered, ST_GeomFromHEXWKB('{hex_buf}'))"
            )
            measure = (
                f"ST_Area(ST_Intersection(f.geom_buffered, "
                f"ST_GeomFromHEXWKB('{hex_buf}')))"
            )
            order_clause = "ORDER BY overlap_cells DESC, f.feature_id"
        else:
            predicate = (
                f"ST_DWithin(f.geom, ST_GeomFromHEXWKB('{hex_raw}'), "
                f"f.buffer_radius + {float(src_radius)})"
            )
            measure = f"ST_Distance(f.geom, ST_GeomFromHEXWKB('{hex_raw}'))"
            order_clause = "ORDER BY overlap_cells ASC, f.feature_id"

        where_clause = " AND ".join([predicate] + where_parts)
        sql = f"""
            SELECT f.feature_id, f.NAME, f.OBJEKTART,
                   {measure} AS overlap_cells
            FROM features_geo f
            WHERE {where_clause}
            {order_clause}
            LIMIT {int(max_results)}
        """
        return self.conn.sql(sql)

    # -- Batch-Varianten --------------------------------------------------------

    def find_intersecting_features_batch(
        self,
        feature_ids: list[int],
        objektart_list: list[str] | None = None,
        exclude_self: bool = True,
    ) -> duckdb.DuckDBPyRelation:
        """Batch-Variante (Spatial Join ueber alle Quell-Features).

        Returns:
            DuckDBPyRelation mit Spalten: src_id, feature_id, NAME, OBJEKTART,
            source, UUID — sortiert nach (src_id, feature_id).
        """
        if not self._has_geo_table():
            raise RuntimeError("features_geo Tabelle nicht gefunden (build_geo.py).")

        ids_str = ", ".join(str(int(i)) for i in feature_ids)
        where_parts = ["f.NAME IS NOT NULL"]
        if objektart_list:
            quoted = ", ".join(f"'{o}'" for o in objektart_list)
            where_parts.append(f"f.OBJEKTART IN ({quoted})")
        if exclude_self:
            where_parts.append("f.feature_id != s.src_id")
        where_clause = " AND ".join(where_parts)

        if self.mode == "buffered":
            join_pred = "ST_Intersects(s.g, f.geom_buffered)"
            src_cols = "feature_id AS src_id, geom_buffered AS g"
        else:
            join_pred = "ST_DWithin(s.g, f.geom, f.buffer_radius + s.r)"
            src_cols = "feature_id AS src_id, geom AS g, buffer_radius AS r"

        sql = f"""
            WITH src AS (
                SELECT {src_cols}
                FROM features_geo
                WHERE feature_id IN ({ids_str})
            )
            SELECT s.src_id, f.feature_id, f.NAME, f.OBJEKTART, f.source, f.UUID
            FROM src s
            JOIN features_geo f ON {join_pred}
            WHERE {where_clause}
            ORDER BY s.src_id, f.feature_id
        """
        return self.conn.sql(sql)

    def find_overlapping_features_batch(
        self,
        feature_ids: list[int],
        objektart: str | None = None,
        max_results: int = 5,
    ) -> duckdb.DuckDBPyRelation:
        """Batch-Variante mit Top-N pro Quell-Feature (ROW_NUMBER).

        Returns:
            DuckDBPyRelation mit Spalten: src_id, feature_id, NAME, OBJEKTART,
            overlap_cells — sortiert nach (src_id, Rang).
        """
        if not self._has_geo_table():
            raise RuntimeError("features_geo Tabelle nicht gefunden (build_geo.py).")

        ids_str = ", ".join(str(int(i)) for i in feature_ids)
        where_parts = ["f.NAME IS NOT NULL"]
        if objektart is not None:
            where_parts.append(f"f.OBJEKTART = '{objektart}'")
        where_clause = " AND ".join(where_parts)

        if self.mode == "buffered":
            join_pred = "ST_Intersects(s.g, f.geom_buffered)"
            src_cols = "feature_id AS src_id, geom_buffered AS g"
            measure = "ST_Area(ST_Intersection(s.g, f.geom_buffered))"
            rank_order = "overlap_cells DESC, feature_id"
        else:
            join_pred = "ST_DWithin(s.g, f.geom, f.buffer_radius + s.r)"
            src_cols = "feature_id AS src_id, geom AS g, buffer_radius AS r"
            measure = "ST_Distance(s.g, f.geom)"
            rank_order = "overlap_cells ASC, feature_id"

        sql = f"""
            WITH src AS (
                SELECT {src_cols}
                FROM features_geo
                WHERE feature_id IN ({ids_str})
            ),
            matches AS (
                SELECT s.src_id, f.feature_id, f.NAME, f.OBJEKTART,
                       {measure} AS overlap_cells
                FROM src s
                JOIN features_geo f ON {join_pred}
                WHERE {where_clause}
            ),
            ranked AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY src_id
                           ORDER BY {rank_order}
                       ) AS rn
                FROM matches
            )
            SELECT src_id, feature_id, NAME, OBJEKTART, overlap_cells
            FROM ranked
            WHERE rn <= {int(max_results)}
            ORDER BY src_id, rn
        """
        return self.conn.sql(sql)
