"""
Step 3: Aufloesung mit trainierten Modellen — Vergleich beider Configs.

Zeigt fuer denselben Text, wie jedes Modell Toponyme aufloest und welche
raeumlichen Beschreibungen es generiert. Gut geeignet zum Vergleich der
Auswirkungen von H3-Parametern (overlap/max_res=13 vs center/max_res=10).

Voraussetzungen:
    - models/model_config1/ und models/model_config2/ (von 02_train.py erzeugt)
    - output/config1/ und output/config2/ (von 01_build.py erzeugt)

Verwendung:
    poetry run python 03_resolve_example.py
    poetry run python 03_resolve_example.py --text "Eigener Text mit Ortsnamen."
"""

import argparse
from pathlib import Path

from geoparser import Geoparser, SpacyRecognizer

from spatial_h3_resolver import SpatialSentenceResolver

BASE = Path(__file__).parent
CONFIGS = ["config1", "config2"]
DEFAULT_TEXT = "Das Matterhorn liegt in den Walliser Alpen nahe Zermatt."

parser = argparse.ArgumentParser()
parser.add_argument("--text", default=DEFAULT_TEXT)
args = parser.parse_args()

for config_name in CONFIGS:
    model_path = BASE / "models" / f"model_{config_name}"
    duckdb_path = BASE / "output" / config_name / "spatial_h3.duckdb"

    if not model_path.exists():
        print(f"\n[{config_name}] Kein Modell gefunden ({model_path}). Erst 02_train.py ausfuehren.")
        continue
    if not duckdb_path.exists():
        print(f"\n[{config_name}] Keine DuckDB gefunden ({duckdb_path}). Erst 01_build.py ausfuehren.")
        continue

    print(f"\n{'=' * 60}")
    print(f"Resolver: {config_name}")
    print(f"{'=' * 60}")

    resolver = SpatialSentenceResolver(
        model_name=str(model_path),
        gazetteer_name="swissnames3d",
        config_path=BASE / "configs" / f"{config_name}.yaml",
        duckdb_path=duckdb_path,
    )

    gp = Geoparser(
        recognizer=SpacyRecognizer(model_name="de_core_news_sm"),
        resolver=resolver,
    )

    docs = gp.parse(args.text)
    for doc in docs:
        for toponym in doc.toponyms:
            print(f"  {toponym.text!r:20} -> {toponym.location}")
