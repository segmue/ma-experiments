"""
Step 4: H3-DuckDB + B1-Assoziationsmatrix je Config bauen.

Liest Geometrien aus Geoparsers SpatiaLite-DB, konvertiert zu H3
und berechnet Spatial Associations (B1) — einmal pro Config.
Entspricht je Config dem Plugin-Aufruf:
    spatial-h3-build --config ../configs/configN.yaml --output ../output/configN/spatial_h3.duckdb

Output:
    ../output/config1/spatial_h3.duckdb, ../output/config1/b1_matrix.csv
    ../output/config2/spatial_h3.duckdb, ../output/config2/b1_matrix.csv

Verwendung:
    python 04_build.py                  # config1 + config2 (Experiment 1)
    python 04_build.py config3 config4  # beliebige Configs aus ../configs/
"""

import sys
from pathlib import Path

from geoparser_h3_resolver.pipeline import build

ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "configs"
OUTPUT_DIR = ROOT / "output"

config_names = sys.argv[1:] or ["config1", "config2"]

for config_name in config_names:
    print(f"\n{'=' * 60}")
    print(f"Building {config_name}...")
    print(f"{'=' * 60}")
    build(
        config=CONFIGS_DIR / f"{config_name}.yaml",
        output_path=OUTPUT_DIR / config_name / "spatial_h3.duckdb",
    )

print(f"\nFertig. Output in: {OUTPUT_DIR}")
