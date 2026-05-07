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

import faulthandler
faulthandler.enable()

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from geoparser import Project

from geoparser_h3_resolver import SpatialSentenceResolver

BASE = Path(__file__).parent
#BASE = Path('~/Projekte/UZH_HS24/MA/anwendung').expanduser()
ANNOTATIONS_DIR = BASE / "data" / "annotations"
MERGED_PATH = BASE / "data" / "annotations_merged.json"
BASE_MODEL = "sentence-transformers/distiluse-base-multilingual-cased-v1"
CONFIGS = ["config1", "config2"]
PROJECT_NAME = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def merge_annotation_files(annotation_dir: Path, output_path: Path) -> Path:
    """Merge multiple annotator JSON exports into a single file.

    The geoparser's load_annotations() overwrites the tag context on each call,
    so multiple calls with the same tag would lose earlier annotations.
    This function combines all documents into one JSON to avoid that.
    """
    files = sorted(annotation_dir.glob("*.json"))
    # Skip previously generated merged file
    files = [f for f in files if f != output_path]

    if not files:
        raise FileNotFoundError(
            f"Keine Annotations-JSONs in {annotation_dir} gefunden.\n"
            "JSON-Exporte aus dem Geoparser Annotator dort ablegen."
        )

    print(f"Merge {len(files)} Annotations-Datei(en):")
    merged = None
    for json_file in files:
        with open(json_file, "r") as f:
            data = json.load(f)
        n_docs = len(data["documents"])
        print(f"  {json_file.name}  ({n_docs} Dokumente)")
        if merged is None:
            merged = data
        else:
            merged["documents"].extend(data["documents"])

    print(f"  -> {len(merged['documents'])} Dokumente total")

    with open(output_path, "w") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"  -> Gespeichert: {output_path.name}")

    return output_path, merged


# --- Annotations zusammenfuehren und laden ---
merged_path, merged = merge_annotation_files(ANNOTATIONS_DIR, MERGED_PATH)

project = Project(name=PROJECT_NAME)
print(f"Project erstellt: {project.name} (id={project.id})")

project.load_annotations(str(merged_path), tag="train", create_documents=True)
print("Annotations geladen.\n")

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
