# experiments — Experiment-Umgebung

Reproduzierbare ML-Pipeline fuer raeumliche Toponym-Aufloesung mit Sentence Transformers.

Systematischer Vergleich: **5 Modelle × 3 Eval-Resolver = 15 Evaluationen** mit K-Fold Cross-Validation.

## Setup

```bash
cd experiments
poetry env use python3.12
poetry install
```

## Workflow-Uebersicht

```
01_build.py          Configs lesen, DuckDBs + Assoziationsmatrizen bauen
       ↓
02_preprocess.py     Annotationen mergen, bereinigen, Frequenzreduktion
       ↓
02_train.py          Phase 1: HP-Suche → Phase 2: Cross-Evaluation → Phase 3: Finale Modelle
       ↓
03_resolve_example.py   Trainierte Modelle auf Beispieltext testen
```

## Experiment-Design

### 5 Modelle

| ID | Sprachmodell | Training | Training-Resolver |
|----|-------------|----------|-------------------|
| **M1** | `dguzh/geo-all-MiniLM-L6-v2` | Keins (Autoren-Modell, engl.) | — |
| **M2** | `distiluse-base-multilingual-cased-v1` | Keins (Base-Modell) | — |
| **M3** | distiluse fine-tuned | K-Fold CV + Final | `SentenceTransformerResolver` (Default) |
| **M4** | distiluse fine-tuned | K-Fold CV + Final | `SpatialSentenceResolver` config1 |
| **M5** | distiluse fine-tuned | K-Fold CV + Final | `SpatialSentenceResolver` config2 |

### 3 Eval-Resolver

| ID | Resolver | Description-Stil |
|----|----------|-----------------|
| **E_default** | `SentenceTransformerResolver` | `"Matterhorn (Alpiner Gipfel) in Zermatt, Wallis, Wallis"` |
| **E_spatial_config1** | `SpatialSentenceResolver` c1 | H3 overlap/res13, voller raeumlicher Kontext |
| **E_spatial_config2** | `SpatialSentenceResolver` c2 | H3 center/res10, groeber |

### Evaluationsmatrix (5 × 3 = 15)

| | E_default | E_spatial_config1 | E_spatial_config2 |
|---|---|---|---|
| **M1** dguzh (as-is) | Autoren-Baseline | Spatial auf fremdem Modell | Spatial auf fremdem Modell |
| **M2** distiluse (as-is) | Rohes Modell | Rohes + Spatial | Rohes + Spatial |
| **M3** default-finetuned | **Faire Baseline** | Spatial-Eval auf Default-Training | Spatial-Eval auf Default-Training |
| **M4** spatial-c1-finetuned | Default-Eval auf Spatial | **Volles System (c1)** | Cross-Config |
| **M5** spatial-c2-finetuned | Default-Eval auf Spatial | Cross-Config | **Volles System (c2)** |

### Schluessel-Vergleiche

1. **M1/E_default vs M3/E_default** — Bringt Fine-Tuning auf Deutsch etwas?
2. **M2/E_default vs M3/E_default** — Bringt Fine-Tuning ueberhaupt etwas?
3. **M3/E_default vs M4/E_spatial_config1** — **Kernfrage:** Sind H3-Spatial-Descriptions besser?
4. **M3/E_spatial_config1 vs M4/E_spatial_config1** — Muss man auch mit Spatial trainieren?
5. **M4/E_spatial_config1 vs M5/E_spatial_config2** — Welche H3-Config ist besser?

## Schritt 0: Annotationsdaten ablegen

JSON-Exporte aus dem Geoparser Annotator nach `data/annotations/` kopieren.

## Schritt 1: DuckDBs bauen

```bash
poetry run python 01_build.py
```

Erzeugt pro Config eine H3-DuckDB und eine B1-Assoziationsmatrix:
- `output/config1/spatial_h3.duckdb` + `b1_matrix.csv`  (overlap, max_res=13)
- `output/config2/spatial_h3.duckdb` + `b1_matrix.csv`  (center, max_res=10)

## Schritt 2a: Annotations vorverarbeiten

```bash
poetry run python 02_preprocess.py
poetry run python 02_preprocess.py --max-count 15     # anderer Schwellwert
poetry run python 02_preprocess.py --no-reduce         # ohne Frequenzreduktion
```

Fuehrt vier Teilschritte aus und speichert das Ergebnis als `data/preprocessed.json`:

1. **Merge:** Alle Annotation-JSONs zusammenfuehren.
2. **Clean:** Toponyme entfernen, bei denen spaCy keine Satzgrenze erkennt.
3. **Frequenzreduktion:** Toponyme die mehr als 10x vorkommen auf 10 reduzieren.
4. **Extraktion:** Dokumente im `fit()`-Format extrahieren, nur Toponyme mit `loc_id`.

## Schritt 2b: Modelle trainieren + evaluieren

```bash
# Vollstaendige Pipeline (Phase 1 + 2 + 3)
poetry run python 02_train.py

# Einzelne Phasen
poetry run python 02_train.py --phase 1             # Nur HP-Suche
poetry run python 02_train.py --phase 2             # Nur Cross-Evaluation
poetry run python 02_train.py --phase 3             # Nur finale Modelle

# Shortcuts
poetry run python 02_train.py --skip-hp-search      # Phase 2+3 mit gecachten HPs
poetry run python 02_train.py --n-folds 3           # Weniger Folds (schneller)
```

### Phase 1: Hyperparameter-Suche

Grid Search ueber `learning_rate × epochs × batch_size` (18 Kombinationen) mit K-Fold CV auf M4 (spatial config1). Beste HPs werden fuer alle Trainings (M3, M4, M5) verwendet.

| Parameter | Werte |
|-----------|-------|
| `learning_rate` | 1e-5, 2e-5, 5e-5 |
| `epochs` | 2, 3, 5 |
| `batch_size` | 8, 16 |
| `warmup_ratio` | 0.1 (fix) |

**Run-Budget:** 18 HP × 5 Folds = 90 Trainer-Runs (mit Caching: 5× `_prepare_training_data()`)

**Output:** `results/hp_search/hp_search_results.json`

### Phase 2: Cross-Evaluation

Pro Fold:

1. **Train** M3 (Default-Resolver), M4 (Spatial c1), M5 (Spatial c2) mit besten HPs
2. **Prepare eval data** fuer alle 3 Eval-Resolver (einmal pro Fold, wiederverwendet fuer alle 5 Modelle)
3. **Evaluate** alle 5 Modelle × 3 Eval-Resolver = 15 Evaluationen

**Run-Budget:** 3 Trainings × 5 Folds = 15 Trainer-Runs + 15 Eval × 5 Folds = 75 Evaluationen

**Output:**
- `results/experiments/{model_id}/{eval_id}/results.json`
- `results/summary.csv` — 15 Zeilen (model × eval_resolver)

### Phase 3: Finale Modelle

3 Modelle (M3, M4, M5) auf Gesamtdaten trainieren. M1/M2 brauchen kein Training.

**Run-Budget:** 3 Trainer-Runs

**Output:** `models/M3_default_finetuned/`, `models/M4_spatial_config1/`, `models/M5_spatial_config2/`

### Evaluation

Da `fit()` keine interne Evaluation unterstuetzt, wird extern evaluiert:

1. Kontext extrahieren (`_extract_context`)
2. Gazetteer-Kandidaten suchen
3. Beschreibungen generieren (`_generate_description` — abhaengig vom Eval-Resolver)
4. Embeddings + Cosine Similarity berechnen
5. Ranking gegen Gold-`loc_id` pruefen

**Metriken:**
- **Accuracy@1** — Ist der Top-Kandidat korrekt?
- **Accuracy@3** — Ist der korrekte Kandidat in den Top 3?
- **MRR** (Mean Reciprocal Rank) — Mittlerer Kehrwert des Rangs

### Performance-Optimierung

Die Pipeline umgeht `fit()` und repliziert dessen Trainer-Setup direkt:

| Ebene | Was | Wirkung |
|-------|-----|---------|
| **A: Feature-Cache** | `feature_id → GeneratedSentence` im Generator | Jede Description 1× generiert |
| **B: Training-Daten** | `_prepare_training_data()` gecacht pro Fold | Phase 1: 90→5 Aufrufe |
| **C: Eval-Daten** | Contexts + Descriptions pro Eval-Resolver gecacht | 75→15 Eval-Vorbereitungen |

### Logging

- Console: INFO-Level (Fortschritt und Ergebnisse)
- Datei: DEBUG-Level in `results/experiment.log`
- Training Loss aus HuggingFace `trainer_state.json`

## Schritt 3: Aufloesung testen

```bash
poetry run python 03_resolve_example.py
poetry run python 03_resolve_example.py --text "Der Rhein fliesst durch Basel."
```

## DB-Configs

| | config1 (Baseline) | config2 (Variante) |
|---|---|---|
| `containment_mode` | `overlap` | `center` |
| `max_resolution` | 13 (~43 m²/Zelle) | 10 (~15 km²/Zelle) |
| Statischer Kontext | Gemeinde, Kanton, Bezirk | Gemeinde, Kanton, Bezirk |

## Verzeichnisstruktur

```
experiments/
  01_build.py
  02_preprocess.py
  02_train.py
  03_resolve_example.py
  configs/
    config1.yaml                  # overlap, max_res=13
    config2.yaml                  # center, max_res=10
  data/
    annotations/                  # Rohe Annotation-JSONs
    preprocessed.json             # Vorverarbeitete Daten
  output/
    config1/                      # DuckDB + Matrix
    config2/
  results/
    experiment.log
    hp_search/
      hp_search_results.json
    experiments/
      M1_dguzh/
        E_default/results.json
        E_spatial_config1/results.json
        E_spatial_config2/results.json
      M2_distiluse_base/
        ...
      M3_default_finetuned/
        E_default/results.json    # ← Faire Baseline
        ...
      M4_spatial_config1/
        E_spatial_config1/results.json  # ← Volles System (c1)
        ...
      M5_spatial_config2/
        ...
    summary.csv                   # 15 Zeilen: model × eval_resolver
  models/
    M3_default_finetuned/
    M4_spatial_config1/
    M5_spatial_config2/
```

## Run-Budget

| Phase | Beschreibung | Trainer-Runs | Evaluationen |
|-------|-------------|-------------|-------------|
| Phase 1 | HP Grid Search (18 × 5 Folds) | 90 | 90 |
| Phase 2 | Cross-Evaluation (3 Train × 5 Folds + 15 Eval × 5 Folds) | 15 | 75 |
| Phase 3 | Finale Modelle (3 × 1) | 3 | — |
| **Total** | | **108** | **165** |
