"""
GeoSentenceResolver — SpatialSentenceResolver mit geometrischer Engine.

Duenne Subclass: injiziert eine GeometricEngine in den unveraenderten
SpatialSentenceResolver (engine=-Parameter). Satzgenerator, B1-Matrix,
Slot-Logik, V1/V3-Overrides und Modelle bleiben identisch zur H3-Variante —
es variiert ausschliesslich die raeumliche Abfragetechnik (Experiment-3-
Isolation). Die B1-Matrix ist bewusst DIESELBE Datei wie beim H3-System
der jeweiligen Config.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union

from geoparser_h3_resolver import SpatialSentenceResolver
from geoparser_h3_resolver.pipeline.build_config import BuildConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine_geo import GeometricEngine  # noqa: E402
from geo_strtree import StrtreeGeometricEngine  # noqa: E402


class GeoSentenceResolver(SpatialSentenceResolver):
    """Geometrischer Zwilling des SpatialSentenceResolver."""

    def __init__(
        self,
        model_name: str = "dguzh/geo-all-MiniLM-L6-v2",
        gazetteer_name: str = "swissnames3d",
        min_similarity: float = 0.6,
        max_tiers: int = 3,
        attribute_map: dict = None,
        # Geo-spezifisch:
        geo_duckdb_path: Union[str, Path] = None,
        config_path: Union[str, Path] = None,
        b1_matrix_path: Optional[Union[str, Path]] = None,
        mode: str = "buffered",
        backend: str = "duckdb",
    ):
        """
        Args:
            geo_duckdb_path: Pfad zur spatial_geo.duckdb (build_geo.py).
            config_path: config.yaml der zugehoerigen H3-Config (config1/config2)
                         — liefert die Sentence-Generator-Parameter.
            b1_matrix_path: Pfad zur b1_matrix.csv. Default: b1_matrix.csv im
                            Ordner der zugehoerigen H3-DB (../config{N}/).
            mode: "buffered" (Hauptvariante) oder "dwithin" (Sensitivitaet).
            backend: "duckdb" (GeometricEngine, DuckDB spatial) oder "strtree"
                     (StrtreeGeometricEngine, hauptspeicherresidenter STRtree).
        """
        geo_duckdb_path = Path(geo_duckdb_path)
        build_config = BuildConfig.from_yaml(config_path)
        if b1_matrix_path is None:
            # geo_configN liegt neben configN: output/{geo_configN,configN}
            cfg_name = geo_duckdb_path.parent.name.removeprefix("geo_")
            b1_matrix_path = (
                geo_duckdb_path.parent.parent / cfg_name / "b1_matrix.csv"
            )
        sentence_config = build_config.to_sentence_generator_config(
            Path(b1_matrix_path)
        )

        super().__init__(
            model_name=model_name,
            gazetteer_name=gazetteer_name,
            min_similarity=min_similarity,
            max_tiers=max_tiers,
            attribute_map=attribute_map,
            sentence_config=sentence_config,
            engine=(StrtreeGeometricEngine(geo_duckdb_path)
                    if backend == "strtree"
                    else GeometricEngine(geo_duckdb_path, mode=mode)),
        )
