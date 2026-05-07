"""
Step 2: SentenceTransformer-Modelle trainieren.

Laedt alle Annotator-JSONs aus data/annotations/ und trainiert fuer jede
Config ein eigenes Modell. Die H3-basierten Saetze aus _generate_description()
dienen als Trainingsziel — das Modell lernt, Toponyme mit ihrem raeumlichen
Kontext zu matchen.

Voraussetzungen:
    - data/annotations/*.json  (JSON-Exporte aus dem Geoparser Annotator)
    - output/config1/ und output/config2/ (von 01_build.py erzeugt)

Output:
    models/model_config1/   (HuggingFace SentenceTransformer Format)
    models/model_config2/

Verwendung:
    poetry run python 02_train.py
"""

from pathlib import Path

from geoparser import Project

from geoparser_h3_resolver import SpatialSentenceResolver

BASE = Path(__file__).parent
#BASE = Path('~/Projekte/UZH_HS24/MA/anwendung').expanduser()
ANNOTATIONS_DIR = BASE / "data" / "annotations"
BASE_MODEL = "sentence-transformers/distiluse-base-multilingual-cased-v1"
CONFIGS = ["config1", "config2"]

# Alle Annotator-JSONs laden (mehrere Sessions akkumulieren)
annotation_files = sorted(ANNOTATIONS_DIR.glob("*.json"))
if not annotation_files:
    raise FileNotFoundError(
        f"Keine Annotations-JSONs in {ANNOTATIONS_DIR} gefunden.\n"
        "JSON-Exporte aus dem Geoparser Annotator dort ablegen."
    )

print(f"Lade {len(annotation_files)} Annotations-Datei(en)...")
project = Project(name="first_test_06-05-2026")
for json_file in annotation_files:
    print(f"  {json_file.name}")
    project.load_annotations(str(json_file), tag="train", create_documents=True)

# Pro Config ein Modell trainieren
for config_name in CONFIGS:
    print(f"\n{'=' * 60}")
    print(f"Training: {config_name}")
    print(f"{'=' * 60}")

    duckdb_path = BASE / "output" / config_name / "spatial_h3.duckdb"
    if not duckdb_path.exists():
        print(f"  FEHLER: {duckdb_path} nicht gefunden. Erst 01_build.py ausfuehren.")
        continue

    resolver = SpatialSentenceResolver(
        model_name=BASE_MODEL,
        gazetteer_name="swissnames3d",
        config_path=BASE / "configs" / f"{config_name}.yaml",
        duckdb_path=duckdb_path,
    )

    output_path = BASE / "models" / f"model_{config_name}"
    project.train_resolver(
        resolver,
        tag="train",
        output_path=output_path,
        epochs=5,
        batch_size=8,
    )
    print(f"  -> Modell gespeichert: {output_path}")

print("\nTraining abgeschlossen.")
