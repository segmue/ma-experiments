"""
StrtreeGeometricEngine — hauptspeicherresidenter geometrischer Zwilling
(Experiment 3, drittes System "geo_strtree").

WOZU
----
`engine_geo.GeometricEngine` beantwortet dieselben Fragen ueber DuckDB-spatial.
Der EXPLAIN-Nachweis in `results/build.json` zeigt, dass DuckDBs Planner den
angelegten RTREE-Index im *Batch*-Muster (Spatial Join) nicht waehlt; die dort
gemessenen Zeiten sind deshalb Kosten eines bestimmten Ausfuehrungsplans und
nicht Kosten geometrischer Praedikate. Diese Engine schliesst die Luecke: Sie
beantwortet exakt dieselben vier Methoden ueber einen EINMAL gebauten
Shapely-2-`STRtree` (GEOS-Paketbaum) im Arbeitsspeicher, also ueber eine
Referenz, die ein Gutachter als kompetent implementiert akzeptiert.

PARITAET (das Wichtigste)
-------------------------
Die Puffer-Semantik ist BITGLEICH zur bestehenden Referenz: Diese Engine
puffert NICHT selbst, sondern liest die Spalte `geom_buffered` aus derselben
`spatial_geo.duckdb`, die `build_geo.py` erzeugt hat. Der Pufferradius ist dort
die mittlere H3-Zellkantenlaenge der je Objekt zugewiesenen Aufloesung
(`build_geo.py:113-131`). Es wird also dieselbe Geometriemenge mit demselben
Praedikat (`intersects`, exakt via GEOS) abgefragt — nur die Indexstruktur und
die Ausfuehrungsschicht sind andere. Die erzeugten Beschreibungssaetze muessen
deshalb zeichengleich zu denen des Systems `geo` sein; genau das prueft
`verify_geo_strtree.py`.

FAIRNESS — was bewusst NICHT optimiert wurde
--------------------------------------------
Die Engine soll den Zellindex nicht kuenstlich gut aussehen lassen, aber auch
nicht gegen ihn frisiert sein. Deshalb sind drei naheliegende Optimierungen
bewusst ausgelassen; jede wuerde diese Referenz NOCH schneller machen, die
gemessenen Zeiten sind also eine obere Schranke:
  1. Keine `shapely.prepare()`-Vorbereitung der Baumgeometrien (spart Speicher,
     kostet Zeit im exakten Praedikat).
  2. Keine nach OBJEKTART vorpartitionierten Teilbaeume. Der Attributfilter
     wirkt als NumPy-Maske NACH der Baumabfrage — genau wie in `engine_geo.py`
     der Filter erst in der WHERE-Klausel steht.
  3. Der Baum enthaelt ALLE Objekte, auch die mit NAME IS NULL; der
     NAME-Filter wird wie im SQL erst nachtraeglich angewandt.
Der Aufbau (WKB lesen, deserialisieren, Baum bauen) wird gemessen und
protokolliert (`build_stats`) und gehoert in die Berichterstattung — er ist das
Gegenstueck zur Bauzeit des Zellindex und zur Pufferzeit der DuckDB-Referenz.

SCHNITTSTELLE
-------------
Duck-typed identisch zu `engine_geo.GeometricEngine` / `H3Engine`:
`find_intersecting_features`, `find_overlapping_features` und die beiden
`_batch`-Varianten, gleiche Argumentnamen, gleiche Rueckgabespalten, gleiche
deterministische Sortierung. Rueckgabe ist kein `DuckDBPyRelation`, sondern ein
`_Result` mit `.df()` und `.fetchall()` — mehr benutzen weder der
Satzgenerator (`.df()`) noch die Benchmark-Skripte (`.df()`) noch
`validate_geo_engine.py` (`.fetchall()`).
`self.conn` bleibt eine Read-only-DuckDB-Verbindung auf dieselbe Datei; der
`SpatialSentenceResolver` macht darueber seine UUID->feature_id-Lookups
(`resolver.py:146,190`), exakt wie bei der bestehenden Geo-Engine.

Nur Modus "buffered". Die Sensitivitaetsvariante "dwithin" bleibt bei der
DuckDB-Referenz: Sie braucht pro Kandidat eine eigene Distanzschwelle
(`f.buffer_radius + s.r`), ist nicht Teil des Beschreibungs-/Paritaetspfads und
wuerde nur eine zweite, nicht vergleichbare Messreihe erzeugen.

Aufruf:
    from geo_strtree import StrtreeGeometricEngine
    with StrtreeGeometricEngine(geo_db_path) as eng:
        df = eng.find_intersecting_features_batch(ids, objektart_list=[...]).df()
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Union

import duckdb
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

INTERSECT_COLS = ["feature_id", "NAME", "OBJEKTART", "source", "UUID"]
OVERLAP_COLS = ["feature_id", "NAME", "OBJEKTART", "overlap_cells"]


class _Result:
    """Minimaler Ersatz fuer DuckDBPyRelation: nur .df() und .fetchall()."""

    __slots__ = ("_df",)

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def df(self) -> pd.DataFrame:
        return self._df

    # Aliase, die DuckDBPyRelation ebenfalls anbietet
    fetchdf = df
    to_df = df

    def fetchall(self) -> list[tuple]:
        return list(self._df.itertuples(index=False, name=None))

    def __len__(self) -> int:
        return len(self._df)


class StrtreeGeometricEngine:
    """Geometrischer Zwilling der H3Engine auf Basis eines Shapely-STRtree."""

    # Als Methode/Attribut ausgelagert, damit der Lesepfad in Tests ohne die
    # DuckDB-spatial-Erweiterung ersetzt werden kann.
    _WKB_EXPR = "ST_AsWKB(geom_buffered)"

    def __init__(
        self,
        db_path: Union[str, Path],
        mode: str = "buffered",
        chunk_size: int = 50_000,
        verbose: bool = True,
    ):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Geo-Datenbank nicht gefunden: {db_path}")
        if mode != "buffered":
            raise ValueError(
                "StrtreeGeometricEngine kennt nur mode='buffered'. Die "
                "dwithin-Sensitivitaetsvariante bleibt bei engine_geo."
            )
        self.mode = mode
        self.verbose = verbose
        self.conn = duckdb.connect(str(self.db_path), read_only=True)
        self._load_spatial_extension()
        self.stats = {"n_queries": 0, "n_batch_queries": 0, "n_tree_hits": 0}
        self._load(chunk_size)

    # -- Aufbau ---------------------------------------------------------------

    def _load_spatial_extension(self) -> None:
        self.conn.execute("INSTALL spatial; LOAD spatial;")

    def _wkb_sql(self) -> str:
        return (
            f"SELECT {self._WKB_EXPR} FROM features_geo ORDER BY feature_id"
        )

    def _load(self, chunk_size: int) -> None:
        """Metadaten + gepufferte Geometrien einmal laden, STRtree einmal bauen."""
        t_total = time.perf_counter()
        row = self.conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = 'features_geo'"
        ).fetchone()
        if not (row and row[0]):
            raise RuntimeError("features_geo Tabelle nicht gefunden (build_geo.py).")

        t0 = time.perf_counter()
        meta = self.conn.execute(
            "SELECT feature_id, NAME, OBJEKTART, source, UUID "
            "FROM features_geo ORDER BY feature_id"
        ).df()
        # feature_id ist die Sortierschluesselspalte -> Positionen im Array sind
        # aufsteigend nach feature_id; das macht ORDER BY f.feature_id zu einem
        # blossen np.sort ueber Positionen.
        self.feature_id = meta["feature_id"].to_numpy(dtype=np.int64)
        self.name = meta["NAME"].to_numpy(dtype=object)
        self.objektart = meta["OBJEKTART"].to_numpy(dtype=object)
        self.source = meta["source"].to_numpy(dtype=object)
        self.uuid = meta["UUID"].to_numpy(dtype=object)
        # NAME IS NOT NULL — im SQL Teil jeder der vier Abfragen.
        self.has_name = np.array(
            [n is not None and n == n for n in self.name], dtype=bool
        )
        meta_s = time.perf_counter() - t0

        n = len(self.feature_id)
        t0 = time.perf_counter()
        geoms = np.empty(n, dtype=object)
        self.conn.execute(self._wkb_sql())
        pos = 0
        while True:
            rows = self.conn.fetchmany(chunk_size)
            if not rows:
                break
            blobs = np.array([r[0] for r in rows], dtype=object)
            geoms[pos:pos + len(rows)] = shapely.from_wkb(blobs)
            pos += len(rows)
        if pos != n:
            raise RuntimeError(f"Geometrien unvollstaendig gelesen: {pos} von {n}")
        wkb_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        self._geoms = geoms
        self._tree = STRtree(geoms)
        tree_s = time.perf_counter() - t0

        self.build_stats = {
            "n_features": int(n),
            "meta_s": round(meta_s, 2),
            "wkb_read_and_parse_s": round(wkb_s, 2),
            "strtree_build_s": round(tree_s, 2),
            "total_s": round(time.perf_counter() - t_total, 2),
            "rss_mb": _rss_mb(),
            "shapely": shapely.__version__,
            "geos": shapely.geos_version_string,
        }
        if self.verbose:
            b = self.build_stats
            print(
                f"    [geo_strtree] Index gebaut: {b['n_features']} Geometrien, "
                f"Metadaten {b['meta_s']}s, WKB+from_wkb {b['wkb_read_and_parse_s']}s, "
                f"STRtree {b['strtree_build_s']}s, total {b['total_s']}s, "
                f"RSS {b['rss_mb']} MB, shapely {b['shapely']}/GEOS {b['geos']}",
                flush=True,
            )

    def close(self) -> None:
        self.conn.close()
        self._tree = None
        self._geoms = None

    def __enter__(self) -> "StrtreeGeometricEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- intern ---------------------------------------------------------------

    def _pos_of(self, feature_id: int) -> int:
        """Position eines feature_id im (nach feature_id sortierten) Array."""
        i = int(np.searchsorted(self.feature_id, int(feature_id)))
        if i >= len(self.feature_id) or self.feature_id[i] != int(feature_id):
            raise ValueError(f"Quell-Feature {feature_id} nicht gefunden.")
        return i

    def _attr_mask(
        self,
        hits: np.ndarray,
        objektart_list=None,
        objektart=None,
        dataset=None,
    ) -> np.ndarray:
        """Die Attributpraedikate der WHERE-Klausel als NumPy-Maske."""
        m = self.has_name[hits]
        if objektart_list:
            m &= np.isin(self.objektart[hits], np.array(list(objektart_list), dtype=object))
        if objektart is not None:
            m &= self.objektart[hits] == objektart
        if dataset is not None:
            m &= self.source[hits] == dataset
        return m

    def _intersect_df(self, hits: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({
            "feature_id": self.feature_id[hits],
            "NAME": self.name[hits],
            "OBJEKTART": self.objektart[hits],
            "source": self.source[hits],
            "UUID": self.uuid[hits],
        }, columns=INTERSECT_COLS)

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
    ) -> _Result:
        """Wie GeometricEngine.find_intersecting_features (buffered).

        Returns:
            _Result mit Spalten: feature_id, NAME, OBJEKTART, source, UUID
        """
        src = self._geoms[self._pos_of(feature_id)]
        # Baumabfrage mit exaktem GEOS-Praedikat -> identische Treffermenge wie
        # ST_Intersects(f.geom_buffered, <konstante Geometrie>).
        hits = np.asarray(self._tree.query(src, predicate="intersects"), dtype=np.int64)
        self.stats["n_queries"] += 1
        self.stats["n_tree_hits"] += int(hits.size)

        m = self._attr_mask(hits, objektart_list=objektart_list, dataset=dataset)
        if exclude_id is not None:
            m &= self.feature_id[hits] != int(exclude_id)
        if exclude_ids:
            m &= ~np.isin(self.feature_id[hits],
                          np.array([int(i) for i in exclude_ids], dtype=np.int64))
        hits = hits[m]

        if order_by_size:
            # ORDER BY ST_Area(f.geom_buffered) ASC, f.feature_id
            # Flaeche erst auf der (kleinen) Treffermenge, wie im SQL.
            areas = shapely.area(self._geoms[hits])
            hits = hits[np.lexsort((self.feature_id[hits], areas))]
        else:
            hits = np.sort(hits)  # Positionen aufsteigend == feature_id aufsteigend
        if max_results:
            hits = hits[:int(max_results)]
        return _Result(self._intersect_df(hits))

    def find_overlapping_features(
        self,
        feature_id: int,
        objektart: str | None = None,
        dataset: str | None = None,
        max_results: int = 5,
    ) -> _Result:
        """Wie GeometricEngine.find_overlapping_features (buffered).

        Ranking: Schnittflaeche der gepufferten Geometrien absteigend,
        Tie-Break feature_id aufsteigend.

        Returns:
            _Result mit Spalten: feature_id, NAME, OBJEKTART, overlap_cells
        """
        src = self._geoms[self._pos_of(feature_id)]
        hits = np.asarray(self._tree.query(src, predicate="intersects"), dtype=np.int64)
        self.stats["n_queries"] += 1
        self.stats["n_tree_hits"] += int(hits.size)

        hits = hits[self._attr_mask(hits, objektart=objektart, dataset=dataset)]
        measure = _intersection_area(self._geoms[hits], src)
        order = np.lexsort((self.feature_id[hits], -measure))[:int(max_results)]
        hits, measure = hits[order], measure[order]
        return _Result(pd.DataFrame({
            "feature_id": self.feature_id[hits],
            "NAME": self.name[hits],
            "OBJEKTART": self.objektart[hits],
            "overlap_cells": measure,
        }, columns=OVERLAP_COLS))

    # -- Batch-Varianten ------------------------------------------------------

    def _src_positions(self, feature_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """(Positionen, src_ids) der Quell-Features; wie `WHERE feature_id IN (...)`
        dedupliziert und aufsteigend (das SQL sortiert am Ende nach src_id)."""
        want = np.unique(np.array([int(i) for i in feature_ids], dtype=np.int64))
        pos = np.searchsorted(self.feature_id, want)
        pos = np.clip(pos, 0, len(self.feature_id) - 1)
        ok = self.feature_id[pos] == want
        return pos[ok], want[ok]

    def find_intersecting_features_batch(
        self,
        feature_ids: list[int],
        objektart_list: list[str] | None = None,
        exclude_self: bool = True,
    ) -> _Result:
        """Batch-Variante: EINE vektorisierte Baumabfrage ueber alle Quellen.

        Returns:
            _Result mit Spalten: src_id, feature_id, NAME, OBJEKTART, source,
            UUID — sortiert nach (src_id, feature_id).
        """
        src_pos, src_ids = self._src_positions(feature_ids)
        pairs = self._tree.query(self._geoms[src_pos], predicate="intersects")
        si = np.asarray(pairs[0], dtype=np.int64)   # Index in src_pos
        ti = np.asarray(pairs[1], dtype=np.int64)   # Position im Baum
        self.stats["n_batch_queries"] += 1
        self.stats["n_tree_hits"] += int(ti.size)

        src_id_arr = src_ids[si]
        m = self._attr_mask(ti, objektart_list=objektart_list)
        if exclude_self:
            m &= self.feature_id[ti] != src_id_arr
        ti, src_id_arr = ti[m], src_id_arr[m]

        order = np.lexsort((self.feature_id[ti], src_id_arr))  # src_id, dann feature_id
        ti, src_id_arr = ti[order], src_id_arr[order]
        df = self._intersect_df(ti)
        df.insert(0, "src_id", src_id_arr)
        return _Result(df)

    def find_overlapping_features_batch(
        self,
        feature_ids: list[int],
        objektart: str | None = None,
        max_results: int = 5,
    ) -> _Result:
        """Batch-Variante mit Top-N je Quell-Feature.

        Returns:
            _Result mit Spalten: src_id, feature_id, NAME, OBJEKTART,
            overlap_cells — sortiert nach (src_id, Rang).
        """
        src_pos, src_ids = self._src_positions(feature_ids)
        pairs = self._tree.query(self._geoms[src_pos], predicate="intersects")
        si = np.asarray(pairs[0], dtype=np.int64)
        ti = np.asarray(pairs[1], dtype=np.int64)
        self.stats["n_batch_queries"] += 1
        self.stats["n_tree_hits"] += int(ti.size)

        m = self._attr_mask(ti, objektart=objektart)
        si, ti = si[m], ti[m]
        src_id_arr = src_ids[si]
        # Mass erst nach dem Attributfilter — semantisch identisch zum SQL,
        # in dem ST_Area(ST_Intersection(...)) nur fuer Treffer der WHERE-Klausel
        # anfaellt.
        measure = _intersection_area(self._geoms[ti], self._geoms[src_pos[si]])

        order = np.lexsort((self.feature_id[ti], -measure, src_id_arr))
        ti, src_id_arr, measure = ti[order], src_id_arr[order], measure[order]
        # ROW_NUMBER() OVER (PARTITION BY src_id ORDER BY ...) <= max_results
        if src_id_arr.size:
            _, first, counts = np.unique(src_id_arr, return_index=True, return_counts=True)
            rank = np.arange(src_id_arr.size) - np.repeat(first, counts)
            keep = rank < int(max_results)
            ti, src_id_arr, measure = ti[keep], src_id_arr[keep], measure[keep]

        return _Result(pd.DataFrame({
            "src_id": src_id_arr,
            "feature_id": self.feature_id[ti],
            "NAME": self.name[ti],
            "OBJEKTART": self.objektart[ti],
            "overlap_cells": measure,
        }, columns=["src_id"] + OVERLAP_COLS))


def _intersection_area(geoms: np.ndarray, other) -> np.ndarray:
    """ST_Area(ST_Intersection(a, b)), vektorisiert; NULL/NaN sortiert ans Ende."""
    if geoms.size == 0:
        return np.zeros(0, dtype=float)
    area = shapely.area(shapely.intersection(geoms, other))
    return np.nan_to_num(np.asarray(area, dtype=float), nan=-1.0)


def _rss_mb() -> float | None:
    try:
        import psutil  # optional
        return round(psutil.Process().memory_info().rss / 1e6, 1)
    except Exception:
        return None
