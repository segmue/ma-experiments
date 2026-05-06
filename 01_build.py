"""
Step 1: DuckDBs + B1-Assoziationsmatrizen bauen.

Liest Geometrien aus Geoparsers SpatiaLite-DB, konvertiert zu H3,
und berechnet Spatial Associations — einmal pro Config.

Output:
    output/config1/spatial_h3.duckdb
    output/config1/b1_matrix.csv
    output/config2/spatial_h3.duckdb
    output/config2/b1_matrix.csv

Verwendung:
    poetry run python 01_build.py
"""

from pathlib import Path

import appdirs

from geoparser_h3_resolver.pipeline import build

BASE = Path(__file__).parent
GEOPARSER_DB = Path(appdirs.user_data_dir("geoparser")) / "geoparser.db"

def get_list_of_yaml_files_in_directory(directory: Path) -> list[Path]:
    """Gibt eine Liste aller YAML-Dateien in einem Verzeichnis zurück, ohne die .yaml-Erweiterung."""
    yaml_files = []
    for file in directory.iterdir():
        if file.is_file() and file.suffix == ".yaml":
            yaml_files.append(file)
    return yaml_files


configs = get_list_of_yaml_files_in_directory(BASE / "configs")

for config_name in configs:
    print(f"\n{'=' * 60}")
    print(f"Building {config_name}...")
    print(f"{'=' * 60}")
    build(
        config=BASE / "configs" / f"{config_name}.yaml",
        gazetteer_db_path=GEOPARSER_DB,
        output_path=BASE / "output" / config_name / "spatial_h3.duckdb",
    )

print(f"\nFertig. Output in: {BASE / 'output'}")

