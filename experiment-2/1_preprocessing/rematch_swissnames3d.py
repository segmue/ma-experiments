"""
Rematcht corpus_swissnames.gpkg (sn25-basiert, LV03) gegen SwissNames3D (2024, LV95).

Matching-Strategie:
  1. Exakter Namensabgleich + räumliche Nähe (< 10 km Cutoff)
  2. Falls kein exakter Match: Fuzzy-Namensabgleich im 500m-Radius

Distanzmetriken je Geometrie-Typ:
  Punkt:   Euklidische Distanz (Punkt↔Punkt)
  Polygon: 0 wenn Punkt im Polygon, sonst Distanz zur Polygon-Grenze
           (bei mehreren enthaltenden Polygonen: kleinstes bevorzugt)
  Linie:   Kürzeste Distanz Punkt→Linie (nearest point on line)

Cutoff:  >10 km → nicht gematcht (falscher Ort trotz gleichem Namen)
5%-Tol.: Falls zweitbester Kandidat ≤5% weiter weg → Ambiguität geloggt

Input:  ma-experiments/data/corpus_swissnames.gpkg, ma-experiments/data/swissnames3d/
Output: ma-experiments/data/corpus_swissnames3d.gpkg  (Layer: corpus_toponyms, CRS: EPSG:2056)
Log:    ma-experiments/data/rematch_swissnames3d.log (enthaelt Korpusinhalt, nicht veroeffentlichen)
"""

import logging
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# Laufzeitdaten liegen ausserhalb des Code-Ordners in ma-experiments/data/ (gitignoriert).
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
INPUT_GPKG = DATA_DIR / "corpus_swissnames.gpkg"
INPUT_LAYER = "corpus_toponyms"

SN3D_DIR = DATA_DIR / "swissnames3d"          # swissNAMES3D 2024, LV95
SN3D_PKT = SN3D_DIR / "swissNAMES3D_PKT.shp"
SN3D_LIN = SN3D_DIR / "swissNAMES3D_LIN.shp"
SN3D_PLY = SN3D_DIR / "swissNAMES3D_PLY.shp"

OUTPUT_GPKG = DATA_DIR / "corpus_swissnames3d.gpkg"
OUTPUT_LAYER = "corpus_toponyms"

AMBIGUITY_TOLERANCE = 0.05  # 5%
DISTANCE_CUTOFF = 10000     # 10 km – darüber = kein Match
FUZZY_RADIUS = 500          # 500 m Suchradius für Fuzzy-Matching
FUZZY_THRESHOLD = 0.80      # Minimum SequenceMatcher-Ratio

TARGET_CRS = "EPSG:2056"

LOG_FILE = DATA_DIR / "rematch_swissnames3d.log"


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("rematch")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def load_swissnames3d(
    logger: logging.Logger,
) -> tuple[dict[str, list[dict]], gpd.GeoDataFrame]:
    """Lädt SwissNames3D und gibt Name-Index + kombiniertes GeoDataFrame zurück."""
    name_index: dict[str, list[dict]] = defaultdict(list)
    all_gdfs = []

    for path, geom_type in [
        (SN3D_PKT, "point"),
        (SN3D_LIN, "line"),
        (SN3D_PLY, "polygon"),
    ]:
        gdf = gpd.read_file(path)
        gdf["_geom_type"] = geom_type
        logger.info(f"  {geom_type}: {len(gdf)} Features aus {path.name}")

        for _, row in gdf.iterrows():
            name = row["NAME"]
            if pd.isna(name) or not name.strip():
                continue
            candidate = {
                "uuid": row["UUID"],
                "name": name,
                "objektart": row.get("OBJEKTART", ""),
                "objektklas": row.get("OBJEKTKLAS", ""),
                "geometry": row.geometry,
                "geom_type": geom_type,
            }
            if geom_type == "polygon":
                candidate["area"] = row.geometry.area
            name_index[name].append(candidate)

        all_gdfs.append(gdf[["UUID", "NAME", "OBJEKTART", "OBJEKTKLAS", "_geom_type", "geometry"]])

    # Kombiniertes GDF für Spatial-Index (Fuzzy-Matching)
    combined_gdf = gpd.GeoDataFrame(
        pd.concat(all_gdfs, ignore_index=True), geometry="geometry", crs=TARGET_CRS
    )

    logger.info(f"  Name-Index: {len(name_index)} unique Namen")
    logger.info(f"  Kombiniertes GDF: {len(combined_gdf)} Features (für Spatial-Index)")
    return dict(name_index), combined_gdf


def load_corpus(logger: logging.Logger) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    gdf = gpd.read_file(INPUT_GPKG, layer=INPUT_LAYER)
    logger.info(f"  Korpus geladen: {len(gdf)} Einträge (CRS: {gdf.crs})")

    has_geom = gdf.geometry.notna() & ~gdf.geometry.is_empty
    matched = gdf[has_geom].copy()
    not_matched = gdf[~has_geom].copy()

    logger.info(f"  Mit Geometrie (für Rematch): {len(matched)}")
    logger.info(f"  Ohne Geometrie (durchgereicht): {len(not_matched)}")

    matched = matched.to_crs(TARGET_CRS)
    # Auch leeres GDF muss korrektes CRS haben (sonst Concat-Fehler)
    not_matched = not_matched.set_crs(TARGET_CRS, allow_override=True)

    return matched, not_matched


def deduplicate_for_matching(
    gdf: gpd.GeoDataFrame, logger: logging.Logger
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    gdf = gdf.copy()
    gdf["_dedup_key"] = gdf["NAME"].astype(str) + "||" + gdf["stid"].astype(str)

    unique = gdf.drop_duplicates(subset="_dedup_key").copy()
    mapping = gdf[["_dedup_key"]].copy()

    logger.info(
        f"  Dedupliziert: {len(gdf)} → {len(unique)} unique (NAME, stid)-Paare"
    )
    return unique, mapping


def _compute_distance(corpus_point, candidate: dict) -> float:
    """Distanz Korpus-Punkt → SwissNames3D-Feature (geometrie-typ-spezifisch)."""
    geom = candidate["geometry"]
    # shapely distance() gibt 0 zurück wenn Punkt im Polygon liegt
    return corpus_point.distance(geom)


def _score_candidates(corpus_point, candidates: list[dict]) -> list[tuple[float, float, dict]]:
    """Bewertet Kandidaten nach Distanz. Returns sorted list of (dist, area, candidate)."""
    scored = []
    for c in candidates:
        dist = _compute_distance(corpus_point, c)
        area = c.get("area", float("inf"))
        scored.append((dist, area, c))
    scored.sort(key=lambda x: (x[0], x[1]))
    return scored


def _check_ambiguity(
    scored: list[tuple[float, float, dict]],
    corpus_name: str, corpus_stid: str,
    logger: logging.Logger,
) -> bool:
    """Prüft 5%-Ambiguität zwischen bestem und zweitbestem Kandidat."""
    if len(scored) < 2:
        return False

    best_dist, best_area, best = scored[0]
    second_dist, second_area, second = scored[1]

    if best_dist == 0 and second_dist == 0:
        if best["geom_type"] == "polygon" and second["geom_type"] == "polygon":
            if abs(best_area - second_area) < 1.0:
                logger.warning(
                    f"AMBIG_EQUAL_POLY: '{corpus_name}' stid={corpus_stid} | "
                    f"UUID1={best['uuid']} (area={best_area:.0f}m²), "
                    f"UUID2={second['uuid']} (area={second_area:.0f}m²)"
                )
                return True
        else:
            logger.warning(
                f"AMBIG_ZERO_DIST: '{corpus_name}' stid={corpus_stid} | "
                f"best={best['geom_type']}({best['uuid']}), "
                f"second={second['geom_type']}({second['uuid']})"
            )
            return True
    elif best_dist > 0 and second_dist <= best_dist * (1 + AMBIGUITY_TOLERANCE):
        logger.warning(
            f"AMBIG_5PCT: '{corpus_name}' stid={corpus_stid} | "
            f"best={best_dist:.1f}m ({best['uuid']}), "
            f"second={second_dist:.1f}m ({second['uuid']})"
        )
        return True

    return False


def _no_match_result(status: str = "no_name_match") -> dict:
    return {
        "sn3d_uuid": None,
        "sn3d_objektart": None,
        "sn3d_objektklas": None,
        "sn3d_geom_type": None,
        "match_distance": None,
        "is_ambiguous": False,
        "rematch_status": status,
        "match_method": None,
    }


def find_best_match(
    corpus_point, corpus_name: str, corpus_stid: str,
    name_index: dict[str, list[dict]],
    combined_gdf: gpd.GeoDataFrame,
    logger: logging.Logger,
) -> dict:
    # --- Schritt 1: Exakter Namensabgleich ---
    candidates = name_index.get(corpus_name, [])
    if candidates:
        scored = _score_candidates(corpus_point, candidates)
        best_dist, best_area, best = scored[0]

        if best_dist <= DISTANCE_CUTOFF:
            is_ambiguous = _check_ambiguity(scored, corpus_name, corpus_stid, logger)

            logger.debug(
                f"EXACT_MATCH: '{corpus_name}' stid={corpus_stid} → "
                f"UUID={best['uuid']} dist={best_dist:.1f}m "
                f"type={best['geom_type']} ambig={is_ambiguous}"
            )

            return {
                "sn3d_uuid": best["uuid"],
                "sn3d_objektart": best["objektart"],
                "sn3d_objektklas": best["objektklas"],
                "sn3d_geom_type": best["geom_type"],
                "match_distance": best_dist,
                "is_ambiguous": is_ambiguous,
                "rematch_status": "matched",
                "match_method": "exact_name",
            }
        else:
            logger.info(
                f"EXACT_OVER_CUTOFF: '{corpus_name}' stid={corpus_stid} | "
                f"best dist={best_dist:.0f}m > {DISTANCE_CUTOFF}m cutoff "
                f"(UUID={best['uuid']}, {best['geom_type']})"
            )

    # --- Schritt 2: Fuzzy-Matching im 500m-Radius ---
    buffer = corpus_point.buffer(FUZZY_RADIUS)
    nearby_idx = combined_gdf.sindex.query(buffer, predicate="intersects")

    if len(nearby_idx) == 0:
        if not candidates:
            logger.info(f"NO_MATCH: '{corpus_name}' stid={corpus_stid} | kein Namens-Match, keine Features in {FUZZY_RADIUS}m")
        else:
            logger.info(f"NO_MATCH: '{corpus_name}' stid={corpus_stid} | exakt >{DISTANCE_CUTOFF / 1000:.0f}km, keine Features in {FUZZY_RADIUS}m")
        return _no_match_result("no_match")

    nearby = combined_gdf.iloc[nearby_idx]
    corpus_name_lower = corpus_name.lower()

    fuzzy_candidates = []
    for _, row in nearby.iterrows():
        sn3d_name = row["NAME"]
        if pd.isna(sn3d_name) or not sn3d_name.strip():
            continue

        ratio = SequenceMatcher(None, corpus_name_lower, sn3d_name.lower()).ratio()
        if ratio >= FUZZY_THRESHOLD:
            dist = corpus_point.distance(row.geometry)
            geom_type = row["_geom_type"]
            area = row.geometry.area if geom_type == "polygon" else float("inf")
            fuzzy_candidates.append((ratio, dist, area, {
                "uuid": row["UUID"],
                "name": sn3d_name,
                "objektart": row.get("OBJEKTART", ""),
                "objektklas": row.get("OBJEKTKLAS", ""),
                "geometry": row.geometry,
                "geom_type": geom_type,
            }))

    if not fuzzy_candidates:
        if not candidates:
            logger.info(f"NO_MATCH: '{corpus_name}' stid={corpus_stid} | kein Fuzzy-Match in {FUZZY_RADIUS}m")
        else:
            logger.info(f"NO_MATCH: '{corpus_name}' stid={corpus_stid} | exakt >{DISTANCE_CUTOFF / 1000:.0f}km, kein Fuzzy in {FUZZY_RADIUS}m")
        return _no_match_result("no_match")

    # Sort: highest ratio first, then lowest distance, then smallest area
    fuzzy_candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    best_ratio, best_dist, best_area, best = fuzzy_candidates[0]

    logger.info(
        f"FUZZY_MATCH: '{corpus_name}' stid={corpus_stid} → "
        f"'{best['name']}' UUID={best['uuid']} "
        f"ratio={best_ratio:.2f} dist={best_dist:.1f}m type={best['geom_type']}"
    )

    return {
        "sn3d_uuid": best["uuid"],
        "sn3d_objektart": best["objektart"],
        "sn3d_objektklas": best["objektklas"],
        "sn3d_geom_type": best["geom_type"],
        "match_distance": best_dist,
        "is_ambiguous": False,
        "rematch_status": "matched",
        "match_method": f"fuzzy({best_ratio:.2f})",
    }


def match_all(
    unique_gdf: gpd.GeoDataFrame,
    name_index: dict[str, list[dict]],
    combined_gdf: gpd.GeoDataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    results = []
    total = len(unique_gdf)

    for i, (_, row) in enumerate(unique_gdf.iterrows()):
        if (i + 1) % 500 == 0:
            print(f"     ... {i + 1}/{total}")

        point = row.geometry
        name = row["NAME"]
        stid = row["stid"]

        if pd.isna(name) or not str(name).strip():
            result = _no_match_result("no_name")
        elif point is None or point.is_empty:
            logger.warning(f"EMPTY_GEOM: '{name}' stid={stid}")
            result = _no_match_result("no_geometry")
        else:
            result = find_best_match(
                point, str(name), str(stid), name_index, combined_gdf, logger
            )

        result["_dedup_key"] = row["_dedup_key"]
        results.append(result)

    print(f"     ... {total}/{total} fertig.")
    return pd.DataFrame(results)


def build_output(
    matched_gdf: gpd.GeoDataFrame,
    not_matched_gdf: gpd.GeoDataFrame,
    match_results: pd.DataFrame,
    mapping: pd.DataFrame,
) -> gpd.GeoDataFrame:
    matched_gdf = matched_gdf.copy()
    matched_gdf["_dedup_key"] = (
        matched_gdf["NAME"].astype(str) + "||" + matched_gdf["stid"].astype(str)
    )

    result_cols = [
        "_dedup_key", "sn3d_uuid", "sn3d_objektart", "sn3d_objektklas",
        "sn3d_geom_type", "match_distance", "is_ambiguous", "rematch_status",
        "match_method",
    ]
    merged = matched_gdf.merge(
        match_results[result_cols], on="_dedup_key", how="left"
    )

    merged["x_lv95"] = merged.geometry.x
    merged["y_lv95"] = merged.geometry.y

    # Nicht-gematchte Einträge durchreichen
    not_matched_gdf = not_matched_gdf.copy()
    for col in ["sn3d_uuid", "sn3d_objektart", "sn3d_objektklas",
                 "sn3d_geom_type", "match_distance", "match_method"]:
        not_matched_gdf[col] = None
    not_matched_gdf["is_ambiguous"] = False
    not_matched_gdf["rematch_status"] = "not_attempted"
    not_matched_gdf["x_lv95"] = not_matched_gdf.geometry.apply(
        lambda g: g.x if g is not None and not g.is_empty else None
    )
    not_matched_gdf["y_lv95"] = not_matched_gdf.geometry.apply(
        lambda g: g.y if g is not None and not g.is_empty else None
    )

    out_cols = [
        "toponym", "stid", "span", "source", "geo_type",
        "OBJECTID", "NAME", "OBJECTVAL", "GEMNAME", "KANTON", "ALTITUDE",
        "x_lv03", "y_lv03", "x_lv95", "y_lv95",
        "sn3d_uuid", "sn3d_objektart", "sn3d_objektklas", "sn3d_geom_type",
        "match_distance", "is_ambiguous", "rematch_status", "match_method",
        "geometry",
    ]

    merged_cols = [c for c in out_cols if c in merged.columns]
    not_matched_cols = [c for c in out_cols if c in not_matched_gdf.columns]

    if len(not_matched_gdf) > 0:
        gdf_out = gpd.GeoDataFrame(
            pd.concat([merged[merged_cols], not_matched_gdf[not_matched_cols]],
                      ignore_index=True),
            geometry="geometry", crs=TARGET_CRS,
        )
    else:
        gdf_out = gpd.GeoDataFrame(merged[merged_cols], geometry="geometry", crs=TARGET_CRS)

    if "_dedup_key" in gdf_out.columns:
        gdf_out.drop(columns=["_dedup_key"], inplace=True)

    return gdf_out


def print_summary(gdf_out: gpd.GeoDataFrame, logger: logging.Logger) -> None:
    total = len(gdf_out)
    attempted = gdf_out[gdf_out["rematch_status"] != "not_attempted"]
    n_attempted = len(attempted)

    n_matched = (attempted["rematch_status"] == "matched").sum()
    n_no_match = (attempted["rematch_status"] == "no_match").sum()
    n_no_name = (attempted["rematch_status"] == "no_name_match").sum()
    n_no_name2 = (attempted["rematch_status"] == "no_name").sum()
    n_not_attempted = (gdf_out["rematch_status"] == "not_attempted").sum()

    matched_sub = attempted[attempted["rematch_status"] == "matched"]
    n_ambiguous = int(matched_sub["is_ambiguous"].sum()) if len(matched_sub) > 0 else 0

    # Match-Methode
    n_exact = (matched_sub["match_method"] == "exact_name").sum() if len(matched_sub) > 0 else 0
    n_fuzzy = (matched_sub["match_method"].str.startswith("fuzzy", na=False)).sum() if len(matched_sub) > 0 else 0

    dist_stats = matched_sub["match_distance"].describe() if len(matched_sub) > 0 else None
    geom_type_counts = matched_sub["sn3d_geom_type"].value_counts() if len(matched_sub) > 0 else pd.Series(dtype=int)

    s = []
    s.append("")
    s.append("=" * 65)
    s.append("  REMATCH-ÜBERSICHT  (SwissNames25 → SwissNames3D)")
    s.append("=" * 65)
    s.append(f"  Einträge gesamt:                    {total:>8}")
    s.append(f"  Rematch versucht:                   {n_attempted:>8}")
    s.append(f"  Nicht versucht (ohne Geometrie):     {n_not_attempted:>8}")
    s.append("")
    s.append(f"  REMATCH-ERGEBNISSE (unique Paare):        ")
    s.append(f"  ├─ matched:                         {n_matched:>8}")
    s.append(f"  │   ├─ via exakter Name:            {n_exact:>8}")
    s.append(f"  │   ├─ via Fuzzy-Match:             {n_fuzzy:>8}")
    s.append(f"  │   ├─ davon ambig (5%-Tol.):       {n_ambiguous:>8}")
    s.append(f"  │   ├─ → point:                     {geom_type_counts.get('point', 0):>8}")
    s.append(f"  │   ├─ → polygon:                   {geom_type_counts.get('polygon', 0):>8}")
    s.append(f"  │   └─ → line:                      {geom_type_counts.get('line', 0):>8}")
    s.append(f"  ├─ no_match (exakt>10km, kein Fuzzy):{n_no_match:>7}")
    s.append(f"  ├─ no_name_match:                   {n_no_name:>8}")
    s.append(f"  └─ no_name (leerer Name):           {n_no_name2:>8}")

    if dist_stats is not None:
        s.append("")
        s.append(f"  DISTANZ-STATISTIK (nur matched):")
        s.append(f"  ├─ Mean:                            {dist_stats['mean']:>8.1f} m")
        s.append(f"  ├─ Median:                          {dist_stats['50%']:>8.1f} m")
        s.append(f"  ├─ Max:                             {dist_stats['max']:>8.1f} m")
        s.append(f"  └─ Std:                             {dist_stats['std']:>8.1f} m")

    # Top 15 nicht gematchte
    all_no = attempted[attempted["rematch_status"].isin(["no_match", "no_name_match"])]
    if len(all_no) > 0:
        s.append("")
        s.append(f"  TOP 15 NICHT GEMATCHTE NAMEN:")
        s.append(f"  " + "-" * 50)
        for name, cnt in all_no["NAME"].value_counts().head(15).items():
            s.append(f"    {str(name):<40} {cnt:>5}x")

    # Top 10 Fuzzy-Matches
    if n_fuzzy > 0:
        s.append("")
        s.append(f"  FUZZY-MATCHES (Beispiele):")
        s.append(f"  " + "-" * 50)
        fuzzy_rows = matched_sub[matched_sub["match_method"].str.startswith("fuzzy", na=False)]
        for _, row in fuzzy_rows.head(10).iterrows():
            s.append(
                f"    '{row['NAME']}' → sn3d UUID={row['sn3d_uuid'][:8]}... "
                f"dist={row['match_distance']:.0f}m ({row['match_method']})"
            )

    s.append("=" * 65)
    s.append("")

    text = "\n".join(s)
    print(text)
    logger.info(text)


def main():
    logger = setup_logging()

    print("1/5  SwissNames3D laden ...")
    name_index, combined_gdf = load_swissnames3d(logger)

    print("2/5  Korpus laden und reprojizieren ...")
    matched_gdf, not_matched_gdf = load_corpus(logger)

    print("3/5  Deduplizieren für effizientes Matching ...")
    unique_gdf, mapping = deduplicate_for_matching(matched_gdf, logger)

    print("4/5  Matching gegen SwissNames3D ...")
    match_results = match_all(unique_gdf, name_index, combined_gdf, logger)

    print("5/5  Output zusammenbauen und speichern ...")
    gdf_out = build_output(matched_gdf, not_matched_gdf, match_results, mapping)
    print_summary(gdf_out, logger)

    gdf_out.to_file(OUTPUT_GPKG, layer=OUTPUT_LAYER, driver="GPKG")
    print(f"     Fertig: {OUTPUT_GPKG}")
    print(f"     Layer '{OUTPUT_LAYER}' mit {len(gdf_out)} Features.")
    print(f"     Log: {LOG_FILE}")


if __name__ == "__main__":
    main()
