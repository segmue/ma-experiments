# experiments — Experiment-Umgebung

Reproduzierbare ML-Pipeline fuer raeumliche Toponym-Aufloesung mit Sentence Transformers.

Vier Scripts, vier Schritte: DuckDB bauen → Daten vorverarbeiten → Modelle trainieren (mit HP-Tuning + K-Fold CV) → Aufloesung testen.

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
02_train.py          Phase 1: HP-Suche → Phase 2: Experiment-Matrix → Phase 3: Finale Modelle
       ↓
03_resolve_example.py   Trainierte Modelle auf Beispieltext testen
```

## Schritt 0: Annotationsdaten ablegen

JSON-Exporte aus dem Geoparser Annotator nach `data/annotations/` kopieren.
Mehrere Session-Dateien werden automatisch zusammengefuehrt.

```
data/annotations/
    session_1.json
    session_2.json
    ...
```

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

1. **Merge:** Alle Annotation-JSONs aus `data/annotations/` zusammenfuehren.
2. **Clean:** Toponyme entfernen, bei denen spaCy keine Satzgrenze erkennt (wuerden `_extract_context()` crashen lassen).
3. **Frequenzreduktion:** Toponyme die mehr als 10x vorkommen (z.B. "Zuerichsee" 46x) werden auf 10 reduziert. Strategie: Innerhalb der Dokumente mit den meisten Vorkommen kuerzen, um Dokument-Diversitaet zu maximieren. So lernt das Modell nicht ueberproportional haeufige Toponyme.
4. **Extraktion:** Dokumente mit `references` und `referents` im `fit()`-Format extrahieren. Nur Toponyme mit `loc_id` werden behalten.

Output-Format ist Dokument-Level-Struktur (nicht flache Listen), damit K-Fold CV auf Dokument-Ebene splitten kann ohne Data Leakage.

## Schritt 2b: Modelle trainieren

```bash
# Vollstaendige Pipeline (Phase 1 + 2 + 3)
poetry run python 02_train.py

# Einzelne Phasen
poetry run python 02_train.py --phase 1             # Nur HP-Suche
poetry run python 02_train.py --phase 2             # Nur Experiment-Matrix
poetry run python 02_train.py --phase 3             # Nur finale Modelle

# Shortcuts
poetry run python 02_train.py --skip-hp-search      # Phase 2+3 mit gecachten HPs
poetry run python 02_train.py --n-folds 3           # Weniger Folds (schneller)
```

### Phase 1: Hyperparameter-Suche

Grid Search ueber `learning_rate × epochs × batch_size` (18 Kombinationen) mit K-Fold CV auf der Referenz-Konfiguration `config1/default`.

Warum nur auf einer Config? Die Hyperparameter (Gradientengroesse, Overfitting-Risiko) sind weitgehend unabhaengig von der SentenceGenerator-Konfiguration. Eine HP-Suche pro Variante waere 10x teurer ohne erwarteten Mehrwert.

| Parameter | Werte | Begruendung |
|-----------|-------|-------------|
| `learning_rate` | 1e-5, 2e-5, 5e-5 | Standard-Fine-Tuning-Bereich fuer Transformer |
| `epochs` | 2, 3, 5 | Max 5 wegen Overfitting-Risiko bei ~400 Samples |
| `batch_size` | 8, 16 | 8=Default, 16=stabilere Gradienten |
| `warmup_ratio` | 0.1 (fix) | Standard fuer Sentence Transformers |

Beste Kombination wird nach Mean Accuracy@1 gewaehlt (MRR als Tiebreaker).

**Run-Budget:** 18 HP-Combos × 5 Folds = 90 `fit()`-Aufrufe

**Output:** `results/hp_search/hp_search_results.json`

### Phase 2: Experiment-Matrix

Wendet die besten HPs aus Phase 1 auf alle **10 Konfigurationen** an (2 DB-Configs × 5 SentenceGenerator-Varianten), jeweils mit K-Fold CV.

Die 5 SentenceGenerator-Varianten testen verschiedene Hypothesen:

| Variante | Override | Forschungsfrage |
|----------|----------|-----------------|
| `default` | — | Baseline: aktuelle YAML-Einstellungen |
| `no_dynamic` | `max_slots=0` | Reicht die Admin-Hierarchie allein? |
| `unlimited` | `max_slots=1000` | Helfen mehr dynamische Kontext-Features? |
| `no_static` | `static_slots=[]` | Wie wichtig ist die feste Admin-Hierarchie? |
| `with_filler` | `max_filler_slots=5` | Helfen Filler-Features als Kontext? |

Die Varianten werden **programmatisch** erstellt (keine neuen YAML-Dateien). Die DuckDB bleibt pro config1/config2 dieselbe — nur die `SentenceGeneratorConfig` wird zur Laufzeit via `sentence_config=` Parameter ueberschrieben.

**Run-Budget:** 10 Varianten × 5 Folds = 50 `fit()`-Aufrufe

**Output:**
- `results/experiments/{config}_{variant}/results.json` — Detail-Ergebnisse pro Variante
- `results/summary.csv` — Aggregierte Uebersicht (eine Zeile pro Variante)

### Phase 3: Finale Modelle

Pro Variante ein finales Modell auf **allen** Daten (kein Val-Split) mit besten HPs trainieren. K-Fold CV diente der Modell-Selektion; das Produktionsmodell profitiert von maximaler Datenmenge.

Alle 10 Modelle werden gespeichert, nicht nur das beste — fuer die spaetere Evaluation mit neuen Annotationsdaten sollen mehrere Modelle (inkl. Baselines) verglichen werden.

**Run-Budget:** 10 `fit()`-Aufrufe

**Output:** `models/{config}_{variant}/`

### Evaluation

Da `fit()` keine interne Evaluation unterstuetzt (Black-Box), wird nach jedem Training extern evaluiert:

1. Kontext extrahieren (`_extract_context`)
2. Gazetteer-Kandidaten suchen
3. H3-basierte Beschreibungen generieren (`_generate_description`)
4. Embeddings + Cosine Similarity berechnen
5. Ranking gegen Gold-`loc_id` pruefen

**Metriken:**
- **Accuracy@1** — Ist der Top-Kandidat korrekt?
- **Accuracy@3** — Ist der korrekte Kandidat in den Top 3?
- **MRR** (Mean Reciprocal Rank) — Mittlerer Kehrwert des Rangs

### Performance-Optimierung

Die Pipeline umgeht `fit()` und repliziert dessen Trainer-Setup direkt.
So koennen teure Vorbereitungsschritte gecacht werden:

| Ebene | Was | Wo | Wirkung |
|-------|-----|-----|---------|
| **A: Feature-Cache** | `feature_id → GeneratedSentence` Cache | `CandidateSentenceGenerator` | Jede Description wird nur 1× generiert (5 DuckDB-Queries gespart pro Duplikat) |
| **B: Training-Daten** | `_prepare_training_data()` 1× pro Fold, gecacht fuer alle 18 HP-Combos | `run_hp_search()` | Phase 1: 90→5 Aufrufe |
| **C: Eval-Daten** | Contexts + Descriptions 1× pro Fold vorberechnet | `prepare_eval_data()` | Phase 1: 90→5 Eval-Vorbereitungen |

Ohne Optimierung: ~15-20h. Mit: ~2-4h.

### Logging

Alles wird mit dem Python `logging`-Modul geloggt (nicht `print`):
- Console: INFO-Level (Fortschritt und Ergebnisse)
- Datei: DEBUG-Level in `results/experiment.log` (vollstaendige Nachvollziehbarkeit)
- Training Loss wird aus HuggingFace `trainer_state.json` extrahiert

## Schritt 3: Aufloesung testen

```bash
poetry run python 03_resolve_example.py

# Eigener Text:
poetry run python 03_resolve_example.py --text "Der Rhein fliesst durch Basel."
```

## DB-Configs

| | config1 (Baseline) | config2 (Variante) |
|---|---|---|
| `containment_mode` | `overlap` | `center` |
| `max_resolution` | 13 (~43 m²/Zelle) | 10 (~15 km²/Zelle) |
| Statischer Kontext | Gemeinde, Kanton, Bezirk | Gemeinde, Kanton, Bezirk |
| `assoc_threshold` | 0.001 | 0.001 |

config1 testet feine raeumliche Granularitaet mit voller Ueberlappungsabdeckung,
config2 testet grobe Granularitaet mit konservativem Center-Modus.

## Verzeichnisstruktur

```
experiments/
  01_build.py                     # DuckDBs bauen
  02_preprocess.py                # Annotations vorverarbeiten
  02_train.py                     # Training-Pipeline (3 Phasen)
  03_resolve_example.py           # Aufloesung testen
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
    experiment.log                # Vollstaendiges Log
    hp_search/
      hp_search_results.json     # Grid Search Ergebnisse
    experiments/
      config1_default/           # Pro Variante
        results.json
      config1_no_dynamic/
      ...
    summary.csv                  # Aggregierte Uebersicht
  models/
    config1_default/             # Finale Modelle
    config1_no_dynamic/
    ...
```

## Run-Budget

| Phase | Beschreibung | fit()-Aufrufe | Geschaetzte Dauer |
|-------|-------------|---------------|-------------------|
| Phase 1 | HP Grid Search (18 × 5 Folds) | 90 | ~3–4.5h |
| Phase 2 | Experiment-Matrix (10 × 5 Folds) | 50 | ~1.5–2.5h |
| Phase 3 | Finale Modelle (10 × 1) | 10 | ~20–30min |
| **Total** | | **150** | **~5–8h (GPU)** |
