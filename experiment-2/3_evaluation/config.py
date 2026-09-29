"""
Geteilte Konstanten + Pfade fuer die Text+Berg-Evaluations-Pipeline (Experiment 2).

Alle Skripte in 3_evaluation importieren von hier. Keine Logik, nur Konfiguration.
Alle Pfade sind relativ zum Repo-Ordner ma-experiments/ abgeleitet:
  data/        Laufzeitdaten (Text+Berg-Rohkorpus, swissNAMES3D, eval_dataset.json)
  configs/     configN.yaml (Plugin-Format)
  output/      configN/spatial_h3.duckdb + b1_matrix.csv / d1_matrix.csv (Plugin-Build)
  matrices/    versionierte Assoziationsmatrizen der Arbeit
Die Modelle M1..M5 werden als Hugging-Face-IDs geladen.
"""

from __future__ import annotations

from pathlib import Path

# ── Pfade ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent                 # .../experiment-2/3_evaluation
EXP2 = HERE.parent                                      # .../experiment-2
REPO = EXP2.parent                                      # .../ma-experiments

DATA_DIR = REPO / "data"                                # Laufzeitdaten (gitignoriert)
EVAL_JSON = DATA_DIR / "eval_dataset.json"              # Evaluationsdatensatz (Annotator-JSON)
CORPUS_DIR = DATA_DIR / "text_berg"                     # Text+Berg-Rohkorpus (XML)
SWISSNAMES3D_DIR = DATA_DIR / "swissnames3d"            # swissNAMES3D_{PKT,LIN,PLY}.shp/.dbf
CORPUS_GPKG = DATA_DIR / "corpus_swissnames3d.gpkg"     # Rematching-Ergebnis (1_preprocessing)
CONFIGS_DIR = REPO / "configs"                          # config1..5.yaml
OUTPUT_DIR = REPO / "output"                            # spatial DuckDBs + Matrizen
MATRICES_DIR = REPO / "matrices"                        # versionierte Matrizen der Arbeit

CACHE_DIR = HERE / "cache"
LOGS_DIR = HERE / "logs"
RESULTS_DIR = EXP2 / "results"
MATRIX_DIR = RESULTS_DIR / "matrix"
PER_ITEM_DIR = CACHE_DIR / "per_item"      # per-Item-Dumps aus Stage 02 (nicht veroeffentlichen)
TABLES_DIR = RESULTS_DIR / "tables"
PLOTS_DIR = HERE / "plots"

GAZETTEER = "swissnames3d"
SEED = 42

# ── Modelle (M1..M5) ───────────────────────────────────────────────────────────
DGUZH_MODEL = "dguzh/geo-all-MiniLM-L6-v2"
BASE_MODEL = "sentence-transformers/distiluse-base-multilingual-cased-v1"

# weights: HF-Hub-ID oder lokaler Pfad. local=True → Pfad muss existieren.
MODELS = {
    "M1_dguzh": {"weights": DGUZH_MODEL, "local": False},
    "M2_distiluse_base": {"weights": BASE_MODEL, "local": False},
    "M3_default_finetuned": {"weights": "segmue/geo-distiluse-swissnames3d", "local": False},
    "M4_spatial_config1": {"weights": "segmue/geo-distiluse-swissnames3d-config1", "local": False},
    "M5_spatial_config2": {"weights": "segmue/geo-distiluse-swissnames3d-config2", "local": False},
}

# ── Eval-Resolver ───────────────────────────────────────────────────────────────
# Optionale Schluessel je Spatial-Resolver:
#   measure    Assoziationsmass ('b1' Default, 'd1' Adjazenzmass). Gelesen wird
#              output/<config>/<measure>_matrix.csv neben der DuckDB.
#   overrides  Felder der SentenceGeneratorConfig, die gegenueber der YAML
#              geaendert werden (Schwellwert- und Kappungsachse).
# Die Beschreibungen entstehen fuer alle Resolver im selben Pfad
# (SpatialSentenceResolver -> CandidateSentenceGenerator/BatchSentenceGenerator).
EVAL_RESOLVERS = {
    "E_default": {"kind": "default", "config": None},
    "E_c1": {"kind": "spatial", "config": "config1"},
    "E_c2": {"kind": "spatial", "config": "config2"},
    "E_c3": {"kind": "spatial", "config": "config3"},  # Config-Ablation: overlap, max_res 11
    "E_c4": {"kind": "spatial", "config": "config4"},  # Config-Ablation: overlap, max_res 12
    "E_c5": {"kind": "spatial", "config": "config5"},  # Config-Ablation: center, max_res 13
    # E_d1: config1 mit Adjazenzmass D1 ('spatial-h3-assoc --measure d1'),
    # plus Schwellwert- (s01, s1) und Kappungsachse (c6, c20) fuer D1 und B1.
    "E_d1": {"kind": "spatial", "config": "config1", "measure": "d1"},
    "E_d1_s01": {"kind": "spatial", "config": "config1", "measure": "d1",
                 "overrides": {"assoc_threshold": 0.01}},
    "E_d1_s1": {"kind": "spatial", "config": "config1", "measure": "d1",
                "overrides": {"assoc_threshold": 0.1}},
    "E_d1_c6": {"kind": "spatial", "config": "config1", "measure": "d1",
                "overrides": {"max_categories": 6}},
    "E_d1_c20": {"kind": "spatial", "config": "config1", "measure": "d1",
                 "overrides": {"max_categories": 20}},
    "E_b1_c6": {"kind": "spatial", "config": "config1", "measure": "b1",
                "overrides": {"max_categories": 6}},
    "E_b1_c20": {"kind": "spatial", "config": "config1", "measure": "b1",
                 "overrides": {"max_categories": 20}},
}

# Die 5×3-Hauptmatrix laeuft nur ueber diese drei Resolver. E_c3–E_c5 (Config-
# Ablation) und die E_d1-/E_b1-Arme werden explizit per --resolver(s) angefordert.
MATRIX_RESOLVERS = ["E_default", "E_c1", "E_c2"]
CONFIG_ABLATION_RESOLVERS = ["E_c3", "E_c4", "E_c5"]
D1_RESOLVERS = ["E_d1", "E_d1_s01", "E_d1_s1", "E_d1_c6", "E_d1_c20", "E_b1_c6", "E_b1_c20"]

# Kurzlabels fuer Abbildungen (Tabellen behalten die Langnamen).
SHORT_MODEL = {"M1_dguzh": "M1", "M2_distiluse_base": "M2", "M3_default_finetuned": "M3",
               "M4_spatial_config1": "M4", "M5_spatial_config2": "M5"}

# Haupt-/Diagonal-Systeme (Modell mit dem es trainiert wurde) — fuer Document-Unit
# und Full-Pipeline. (M1/M2 sind untrainierte Baselines mit E_default.)
MAIN_SYSTEMS = [
    ("M1_dguzh", "E_default"),
    ("M2_distiluse_base", "E_default"),
    ("M3_default_finetuned", "E_default"),
    ("M4_spatial_config1", "E_c1"),
    ("M5_spatial_config2", "E_c2"),
    ("M5_spatial_config2", "E_c1"),   # NEU (12.09.): bestes System, fuer Full-Pipeline Kap. 6.2.9
]

# ── Recognizer (Full-Pipeline): spaCy-Modell pro Korpus-Sprache ────────────────
# Sprache aus dem source-Suffix (_de/_fr/_en/_mul). _mul (gemischt) -> Deutsch.
LANG_MODELS = {
    "de": "de_core_news_lg",
    "fr": "fr_core_news_lg",
    "en": "en_core_web_lg",
    "mul": "de_core_news_lg",
}
RECOGNIZER_ENTITY_TYPES = ["FAC", "GPE", "LOC"]

# ── Ablationen (Konzept Abschnitt 5): fixes Encoder-Modell, Resolver-Config variieren ─
ABLATION_MODEL = "M5_spatial_config2"
ABLATION_BASE_CONFIG = "config1"           # Basis-Config fuer die Inferenz-Varianten
ASSOC_THRESHOLDS = [0.001, 0.01, 0.1]      # B: Assoziations-Schwellwert
MAX_SLOTS_SWEEP = [10, 5, 3]               # C: Anzahl dynamischer Slots
# A1: max_slots=0 (nur statische Slots), A2: static_slots=[] (kein Admin-Kontext)

BATCH_SIZE = 256  # RTX 3070, 8GB: 64 war unnoetig konservativ fuer MiniLM/distiluse

# ── Pfad-Helfer ────────────────────────────────────────────────────────────────
def config_yaml(name: str) -> Path:
    return CONFIGS_DIR / f"{name}.yaml"


def duckdb_path(name: str) -> Path:
    return OUTPUT_DIR / name / "spatial_h3.duckdb"


def matrix_path(name: str, measure: str = "b1") -> Path:
    """Assoziationsmatrix neben der DuckDB (so liest sie auch das Plugin)."""
    return OUTPUT_DIR / name / f"{measure}_matrix.csv"


def ensure_dirs() -> None:
    for d in (CACHE_DIR, LOGS_DIR, RESULTS_DIR, MATRIX_DIR, PER_ITEM_DIR, TABLES_DIR, PLOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def language_of(filename: str) -> str:
    """Leitet die Sprache aus dem Dokument-filename ab ('{source}#a{N}')."""
    source = filename.split("#", 1)[0]
    suffix = source.rsplit("_", 1)[-1].lower()
    return suffix if suffix in LANG_MODELS else "mul"


def check_prerequisites(require_spatial: bool = True, require_local_models: bool = True,
                        resolvers: list[str] | None = None) -> list[str]:
    """Gibt eine Liste fehlender Prerequisites zurueck (leer = alles vorhanden).

    resolvers: Eval-Resolver, deren DuckDB/Matrix geprueft wird
    (Default: die Hauptmatrix E_c1/E_c2).
    """
    missing = []
    if not EVAL_JSON.exists():
        missing.append(f"Evaluationsdatensatz fehlt: {EVAL_JSON} (erst 2_dataset/build_eval_dataset.py)")
    if require_local_models:
        for mid, m in MODELS.items():
            if m["local"] and not Path(m["weights"]).exists():
                missing.append(f"Modell fehlt: {m['weights']}")
    if require_spatial:
        needed = []
        for eid in (resolvers or MATRIX_RESOLVERS):
            r = EVAL_RESOLVERS[eid]
            if r["kind"] == "spatial":
                item = (r["config"], r.get("measure", "b1"))
                if item not in needed:
                    needed.append(item)
        for cfg, measure in needed:
            if not duckdb_path(cfg).exists():
                missing.append(f"Spatial-DuckDB fehlt: {duckdb_path(cfg)} "
                               f"(erst 'spatial-h3-build --config configs/{cfg}.yaml')")
            if not matrix_path(cfg, measure).exists():
                befehl = ("spatial-h3-assoc --measure d1" if measure == "d1"
                          else "spatial-h3-build")
                missing.append(f"{measure.upper()}-Matrix fehlt: {matrix_path(cfg, measure)} "
                               f"(erst '{befehl}')")
    return missing
