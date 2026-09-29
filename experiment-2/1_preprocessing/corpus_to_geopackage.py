"""
Verknüpft Text+Berg Korpus-Geodaten mit dem Swissnames-Shapefile (sn25.shp).

Das NER-Tagging im Korpus enthält zwei Sektionen:
  <geo>     – Geo-Entitäten: mountain, city, valley, glacier, lake, mountain cabin
  <persons> – Personen (firstname, lastname, etc.) → nicht im GeoPackage

Für den GeoPackage-Join:
  - Nur s…-IDs (swisstopo), Left Join → non-matches erhalten Null-Geometrie
  - Faulty-ID s23 = Fallback für mehrdeutige Namen (keine eindeutige Koordinate)
  - GeoNames (g…, cg…) und unlinked (0) nicht im GeoPackage

match_status Werte:
  matched   – erfolgreich mit Swissnames verknüpft (hat Geometrie, LV03)
  faulty    – stid=s23 (Fallback-ID, keine Geometrie)
  no_match  – s…-ID nicht im sn25-Datensatz (keine Geometrie)

Input:  ma-experiments/data/text_berg/ (Text+Berg-Rohkorpus, SWISSUbase)
        ma-experiments/data/swissnames25/sn25.shp (SwissNames 2008, Korpus-IDs)
Output: ma-experiments/data/corpus_swissnames.gpkg  (Layer: corpus_toponyms, CRS: EPSG:21781)
"""

import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import pandas as pd

# Laufzeitdaten liegen ausserhalb des Code-Ordners in ma-experiments/data/ (gitignoriert).
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CORPUS_DIR = DATA_DIR / "text_berg"
SHAPEFILE = DATA_DIR / "swissnames25" / "sn25.shp"
OUTPUT_GPKG = DATA_DIR / "corpus_swissnames.gpkg"
OUTPUT_LAYER = "corpus_toponyms"

FAULTY_ID = "s23"  # Fallback-ID für mehrdeutige Bergnamen

SN_COLS = ["OBJECTID", "NAME", "OBJECTVAL", "GEMNAME", "KANTON", "ALTITUDE", "geometry"]


def classify_stid(stid: str) -> str:
    """
    ID-Typen laut Text+Berg-Dokumentation:
      s<int>   – SwissTopo SwissNames25-ID (Berge, Gletscher, Hütten, Seen, Täler)
      <int>    – Schweizer Postleitzahl (PLZ) für Ortschaften/Städte
      23       – Ambige Fallback-ID für Ortschaften (mehrfach in PLZ-Liste)
      s23      – Ambige Fallback-ID für Berge/Gletscher etc. (mehrfach in SwissTopo)
      g<int>   – GeoNames-ID (internationale Bergnamen von geonames.org)
      cg<int>  – GeoNames composite/collective ID
      0        – Nicht in keiner Liste gefunden (unlinked)
    """
    if stid == "0":
        return "unlinked"
    if stid == FAULTY_ID:
        return "faulty_swisstopo"  # s23: ambige SwissTopo-ID für Berge/etc.
    if stid.startswith("s"):
        try:
            int(stid[1:])
            return "swisstopo"
        except ValueError:
            return "other"
    if stid.startswith("cg"):
        return "geonames_composite"
    if stid.startswith("g"):
        return "geonames"
    # Reine Ganzzahl = Schweizer Postleitzahl (PLZ) für city-Einträge
    try:
        int(stid)
        if stid == "23":
            return "faulty_plz"  # 23: ambige Fallback-ID für Ortschaften
        return "plz"
    except ValueError:
        return "other"


def extract_corpus(corpus_dir: Path) -> tuple[list[dict], dict, int, int]:
    """
    Extrahiert alle Geo-Annotationen und zählt Personen getrennt.

    Returns:
        geo_records    – Liste aller <g>-Einträge (alle geo_types)
        person_counts  – {'total': n, 'with_name': n} aus <persons>
        n_docs         – Anzahl erfolgreich verarbeiteter Dokumente
        n_skipped      – Anzahl übersprungener Dokumente
    """
    geo_records = []
    person_counts = defaultdict(int)
    n_docs = 0
    n_skipped = 0

    for ner_file in sorted(corpus_dir.rglob("*-ner.xml")):
        text_file = ner_file.parent / ner_file.name.replace("-ner.xml", ".xml")
        if not text_file.exists():
            print(f"  [warn] kein Text-XML für {ner_file.name}")
            n_skipped += 1
            continue

        try:
            words = {
                w.get("id"): (w.text or "")
                for w in ET.parse(text_file).iter("w")
            }
            ner_root = ET.parse(ner_file).getroot()
        except ET.ParseError as e:
            print(f"  [warn] Parse-Fehler in {ner_file.name}: {e}")
            n_skipped += 1
            continue

        n_docs += 1
        source = ner_file.stem.replace("-ner", "")

        # --- Geo-Entitäten ---
        geo_section = ner_root.find("geo")
        if geo_section is not None:
            for g in geo_section:
                stid = g.get("stid", "0")
                span = g.get("span", "")
                toponym = " ".join(
                    words.get(wid, f"[{wid}?]") for wid in span.split()
                ).strip()
                geo_records.append({
                    "toponym": toponym,
                    "stid": stid,
                    "stid_type": classify_stid(stid),
                    "span": span,
                    "source": source,
                    "geo_type": g.get("type", ""),
                })

        # --- Personen (nur zählen, nicht ins GeoPackage) ---
        persons_section = ner_root.find("persons")
        if persons_section is not None:
            for person in persons_section:
                person_counts["total"] += 1
                fn = person.findtext("firstname") or ""
                ln = person.findtext("lastname") or ""
                if fn.strip() or ln.strip():
                    person_counts["with_name"] += 1

    return geo_records, dict(person_counts), n_docs, n_skipped


def print_overview(geo_records: list[dict], person_counts: dict,
                   n_docs: int, n_skipped: int,
                   n_matched: int, n_faulty: int, n_no_match: int) -> None:
    df = pd.DataFrame(geo_records)
    total_geo = len(df)
    stid_counts = df["stid_type"].value_counts()
    type_counts = df["geo_type"].value_counts()
    n_swisstopo = stid_counts.get("swisstopo", 0) + n_faulty  # n_faulty = faulty_swisstopo (s23)

    print()
    print("=" * 60)
    print("  DATENSATZ-ÜBERSICHT")
    print("=" * 60)
    print(f"  Dokumente verarbeitet:            {n_docs:>8}")
    print(f"  Dokumente übersprungen:           {n_skipped:>8}")
    print()
    print(f"  GEO-ANNOTATIONEN gesamt:          {total_geo:>8}")
    print(f"  ├─ mountain:                      {type_counts.get('mountain', 0):>8}")
    print(f"  ├─ city:                          {type_counts.get('city', 0):>8}")
    print(f"  ├─ valley:                        {type_counts.get('valley', 0):>8}")
    print(f"  ├─ mountain cabin:                {type_counts.get('mountain cabin', 0):>8}")
    print(f"  ├─ glacier:                       {type_counts.get('glacier', 0):>8}")
    print(f"  └─ lake:                          {type_counts.get('lake', 0):>8}")
    print()
    print(f"  PERSONEN-ANNOTATIONEN:            {person_counts.get('total', 0):>8}  (nicht im GeoPackage)")
    print(f"  └─ davon mit Name:                {person_counts.get('with_name', 0):>8}")
    print()
    n_plz = stid_counts.get("plz", 0) + stid_counts.get("faulty_plz", 0)
    print(f"  STID-TYPEN (Geo-Annotationen):")
    print(f"  ├─ SwissTopo (s…):                {n_swisstopo:>8}  → SwissNames25-ID")
    print(f"  │   ├─ gematcht (mit Geometrie):  {n_matched:>8}")
    print(f"  │   ├─ faulty s23 (ohne Geom.):   {n_faulty:>8}  mehrdeutig in SwissTopo")
    print(f"  │   └─ kein Match sn25.shp:       {n_no_match:>8}")
    print(f"  ├─ PLZ (Postleitzahl, city):       {n_plz:>8}  → Schweizer PLZ-Verzeichnis")
    print(f"  │   └─ davon faulty (PLZ=23):     {stid_counts.get('faulty_plz', 0):>8}  mehrdeutig in PLZ-Liste")
    print(f"  ├─ GeoNames (g…):                 {stid_counts.get('geonames', 0):>8}  → geonames.org (internat. Berge)")
    print(f"  ├─ GeoNames composite (cg…):      {stid_counts.get('geonames_composite', 0):>8}  → geonames.org")
    print(f"  ├─ unlinked (stid=0):             {stid_counts.get('unlinked', 0):>8}  → in keiner Liste gefunden")
    print(f"  └─ andere:                        {stid_counts.get('other', 0):>8}")
    print()
    print(f"  TOP 15 FAULTY (s23) NAMEN")
    print(f"  " + "-" * 45)
    faulty_df = df[df["stid"] == FAULTY_ID]
    for name, cnt in faulty_df["toponym"].value_counts().head(15).items():
        print(f"  {name:<35} {cnt:>5}×")
    print("=" * 60)
    print()


def main():
    print(f"1/4  Korpus einlesen aus {CORPUS_DIR} ...")
    geo_records, person_counts, n_docs, n_skipped = extract_corpus(CORPUS_DIR)
    print(f"     {len(geo_records)} Geo-Annotationen, {person_counts.get('total', 0)} Personen ({n_docs} Dokumente).")

    # Nur s…-IDs für den Join
    sw_records = [r for r in geo_records if r["stid_type"] in ("swisstopo", "faulty_swisstopo")]
    df_sw = pd.DataFrame(sw_records)
    df_sw["objectid_int"] = df_sw["stid"].apply(
        lambda s: int(s[1:]) if s != FAULTY_ID else None
    )

    print(f"2/4  Shapefile laden: {SHAPEFILE} ...")
    gdf_sn = gpd.read_file(SHAPEFILE)[SN_COLS]
    print(f"     {len(gdf_sn)} Swissnames-Features (CRS: {gdf_sn.crs}).")

    print("3/4  Join Geo-Annotationen ↔ Swissnames (Left Join) ...")
    merged = df_sw.merge(
        gdf_sn,
        left_on="objectid_int",
        right_on="OBJECTID",
        how="left",
    )

    def get_status(row):
        if row["stid"] == FAULTY_ID:
            return "faulty_s23"
        if pd.isna(row.get("OBJECTID")):
            return "no_match"
        return "matched"

    merged["match_status"] = merged.apply(get_status, axis=1)
    n_matched  = (merged["match_status"] == "matched").sum()
    n_faulty   = (merged["match_status"] == "faulty_s23").sum()
    n_no_match = (merged["match_status"] == "no_match").sum()

    gdf_out = gpd.GeoDataFrame(merged, geometry="geometry", crs=gdf_sn.crs)
    gdf_out["x_lv03"] = gdf_out.geometry.apply(
        lambda geom: geom.x if geom is not None and not geom.is_empty else None
    )
    gdf_out["y_lv03"] = gdf_out.geometry.apply(
        lambda geom: geom.y if geom is not None and not geom.is_empty else None
    )

    out_cols = [
        "toponym", "stid", "match_status", "span", "source", "geo_type",
        "OBJECTID", "NAME", "OBJECTVAL", "GEMNAME", "KANTON", "ALTITUDE",
        "x_lv03", "y_lv03", "geometry",
    ]
    gdf_out = gdf_out[out_cols]
    print(f"     {n_matched} gematcht | {n_faulty} faulty (s23) | {n_no_match} kein Match")

    print_overview(geo_records, person_counts, n_docs, n_skipped,
                   n_matched, n_faulty, n_no_match)

    print(f"4/4  GeoPackage schreiben: {OUTPUT_GPKG} ...")
    gdf_out.to_file(OUTPUT_GPKG, layer=OUTPUT_LAYER, driver="GPKG")
    print(f"     Fertig. Layer '{OUTPUT_LAYER}' mit {len(gdf_out)} Features.")
    print(f"     ({n_matched} mit Geometrie, {n_faulty + n_no_match} mit Null-Geometrie)")


if __name__ == "__main__":
    main()
