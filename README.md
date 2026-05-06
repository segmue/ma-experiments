# anwendung — Experiment-Umgebung

Drei Scripts, drei Schritte: DuckDB bauen → Modell trainieren → Auflösen und vergleichen.

## Setup

```bash
cd anwendung
poetry env use python3.12
poetry install
```

## Workflow

### Schritt 0: Annotationsdaten ablegen

JSON-Exporte aus dem Geoparser Annotator nach `data/annotations/` kopieren.
Mehrere Session-Dateien werden automatisch alle geladen.

```
data/annotations/
    session_1.json
    session_2.json
    ...
```

### Schritt 1: DuckDBs bauen

```bash
poetry run python 01_build.py
```

Erzeugt:
- `output/config1/spatial_h3.duckdb` + `b1_matrix.csv`  (overlap, max_res=13)
- `output/config2/spatial_h3.duckdb` + `b1_matrix.csv`  (center, max_res=10)

### Schritt 2: Modelle trainieren

```bash
poetry run python 02_train.py
```

Erzeugt:
- `models/model_config1/`  (trainiert auf H3-Saetzen aus config1)
- `models/model_config2/`  (trainiert auf H3-Saetzen aus config2)

### Schritt 3: Auflösung vergleichen

```bash
poetry run python 03_resolve_example.py

# Eigener Text:
poetry run python 03_resolve_example.py --text "Der Rhein fliesst durch Basel."
```

## Configs

| | config1 (Baseline) | config2 (Variante) |
|---|---|---|
| `containment_mode` | `overlap` | `center` |
| `max_resolution` | 13 (~43 m²/Zelle) | 10 (~15 km²/Zelle) |
| Statischer Kontext | Gemeinde, Kanton, Bezirk | Gemeinde, Kanton, Bezirk |
| `assoc_threshold` | 0.001 | 0.001 |

Der Vergleich testet: feine räumliche Granularität mit voller Überlappungsabdeckung
(config1) vs. grobe Granularität mit konservativem Center-Modus (config2).
