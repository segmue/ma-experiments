"""
Geteilte Konstanten + Pfade fuer die Experiment-3-Benchmark-Pipeline.

Alle Skripte (01_..06_) importieren von hier. Konfiguration plus drei kleine
Robustheits-Helfer (tee_log / save_json_atomic / load_json), damit lange
Laeufe Logs auf Platte haben und blockweise resume-faehig sind.
Wiederverwendung aus Experiment 2: EVAL3_* Pfade zeigen auf 3_evaluation
(eval_core, Caches, Dataset) -- Import via sys.path in den Skripten.

BENCH_SMOKE=1 (Umgebungsvariable): Mini-Parameter + eigene *_smoke-Ordner,
um die ganze Kette inkl. Resume in Minuten zu testen.
"""

from __future__ import annotations

import io
import json
import os
import random
import sys
import time
from pathlib import Path

# ── Pfade ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent                  # .../experiment-3/2_benchmark
EXP3 = HERE.parent                                       # .../experiment-3
ROOT = EXP3.parent                                       # .../ma-experiments
RESOLVER_DIR = EXP3 / "1_geometric_resolver"

OUTPUT_DIR = ROOT / "output"                             # H3-DBs (configN/) + Geo-DBs (geo_configN/)
CONFIGS_DIR = ROOT / "configs"                           # config1.yaml / config2.yaml
MATRICES_DIR = ROOT / "matrices"                         # configN/b1_matrix.csv

EVAL3_DIR = ROOT / "experiment-2" / "3_evaluation"      # eval_core, config, cache/items_*.pkl

SMOKE = os.environ.get("BENCH_SMOKE") == "1"
_SUFFIX = "_smoke" if SMOKE else ""

# BENCH_VARIANT=<suffix> schreibt Resultate/Logs/Plots/Cache in eigene Ordner,
# damit ein Nachlauf (z.B. das dritte System geo_strtree) die bestehenden
# Artefakte nicht ueberschreibt.
#
# CACHE_DIR folgt der Variante: ein Nachlauf findet keine alten Checkpoints
# und misst ALLE Systeme frisch (so entstand der Lauf vom 16.09.2026, damals
# mit BENCH_VARIANT=_strtree; keine Mischung von Code-Staenden/Messsitzungen).
VARIANT = os.environ.get("BENCH_VARIANT", "")

RESULTS_DIR = EXP3 / f"results{_SUFFIX}{VARIANT}"
TABLES_DIR = RESULTS_DIR / "tables"
PLOTS_DIR = EXP3 / f"plots{_SUFFIX}{VARIANT}"
LOGS_DIR = EXP3 / f"logs{_SUFFIX}{VARIANT}"
CACHE_DIR = EXP3 / f"cache{_SUFFIX}{VARIANT}"

GAZETTEER = "swissnames3d"
SEED = 42

# ── Benchmark-Parameter ────────────────────────────────────────────────────────
N_REPEATS = 20            # Wiederholungen pro Mikro-Messpunkt (Median/IQR)
N_WARMUP = 3              # Warm-up-Laeufe (nicht gewertet)
STRATA_N = 50             # Stichprobe pro Geometrietyp (PKT/LIN/PLY)
BATCH_SIZE_BENCH = 300    # Batch-Groesse der Batch-Messpunkte
N_DOCS_THROUGHPUT = 100   # Dokumente pro Durchsatz-Set (2 disjunkte Sets)
N_DOCS_BENCH = 100        # Doc-Sample fuer Stage 03/05 (NIE der volle Korpus!)
THROUGHPUT_MODEL = "M5_spatial_config2"  # ein festes Modell fuer alle Systeme

# Adaptive Stichprobe: Zeitbudget pro Messpunkt. Die Geo-Systeme haben Mess-
# punkte mit >100 s pro Einzelquery (dwithin/LIN) -- volle 50 Samples x 20
# Repeats wuerden Stage 02 allein auf 6-12 h treiben. Sobald MIN_POINT_SAMPLES
# erreicht sind UND das Budget ueberschritten ist, wird der Punkt beendet;
# das tatsaechliche n steht im Resultat ("truncated": true markiert Kuerzung).
POINT_TIME_BUDGET_S = 120
MIN_POINT_SAMPLES = 5     # Einzelpfad-Punkte (Stage 02)
MIN_BATCH_REPEATS = 3     # Batch-Punkte (Stage 02)
MIN_SINGLE_SAMPLES = 20   # Einzelpfad-Stichprobe (Stage 03)

if SMOKE:
    N_REPEATS = 3
    STRATA_N = 5
    BATCH_SIZE_BENCH = 20
    N_DOCS_THROUGHPUT = 3
    N_DOCS_BENCH = 5
    POINT_TIME_BUDGET_S = 15
    MIN_POINT_SAMPLES = 2
    MIN_BATCH_REPEATS = 2
    MIN_SINGLE_SAMPLES = 3

CONFIGS = ("config1", "config2")


def h3_db(cfg: str) -> Path:
    return OUTPUT_DIR / cfg / "spatial_h3.duckdb"


def geo_db(cfg: str) -> Path:
    return OUTPUT_DIR / f"geo_{cfg}" / "spatial_geo.duckdb"


def config_yaml(cfg: str) -> Path:
    return CONFIGS_DIR / f"{cfg}.yaml"


def b1_matrix(cfg: str) -> Path:
    return MATRICES_DIR / cfg / "b1_matrix.csv"


def add_paths() -> None:
    """sys.path fuer 1_geometric_resolver + 3_evaluation (eval_core etc.)."""
    for p in (str(RESOLVER_DIR), str(EVAL3_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)


def ensure_dirs() -> None:
    for d in (RESULTS_DIR, TABLES_DIR, PLOTS_DIR, LOGS_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ── Doc-Sample (Stage 03/05) ───────────────────────────────────────────────────
def sample_doc_ids(all_doc_ids) -> set[str]:
    """Deterministisches N_DOCS_BENCH-Sample ueber sortierte Doc-IDs (Seed SEED).

    Stage 03 (Exp2-Items) und Stage 05 (eval_dataset) ziehen ueber dieselbe
    ID-Menge dasselbe Sample -- gemeinsamer Workload, nie der volle Korpus.
    """
    ids = sorted(set(all_doc_ids))
    if len(ids) <= N_DOCS_BENCH:
        return set(ids)
    return set(random.Random(SEED).sample(ids, N_DOCS_BENCH))


# ── Checkpoint-Helfer ──────────────────────────────────────────────────────────
def save_json_atomic(path: Path, obj) -> None:
    """JSON via tmp+replace schreiben (kein halbgeschriebener Checkpoint)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def load_json(path: Path):
    """Checkpoint laden; None wenn nicht vorhanden oder unlesbar."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ── File-Logging (tee) ─────────────────────────────────────────────────────────
class _Tee(io.TextIOBase):
    """Spiegelt einen Stream zeilenweise (mit Zeitstempel) in eine Log-Datei."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh
        self._at_line_start = True

    def write(self, s: str) -> int:
        self._stream.write(s)
        for line in s.splitlines(keepends=True):
            if self._at_line_start and line.strip():
                self._fh.write(time.strftime("[%H:%M:%S] "))
            self._fh.write(line)
            self._at_line_start = line.endswith(("\n", "\r"))
        self.flush()
        return len(s)

    def flush(self) -> None:
        self._stream.flush()
        self._fh.flush()


def tee_log(stage: str) -> None:
    """stdout+stderr zusaetzlich in logs/<stage>.log schreiben (UTF-8, append).

    Damit existiert IMMER ein vollstaendiges Log auf Platte, egal wie das
    Skript gestartet wurde.
    """
    fh = open(LOGS_DIR / f"{stage}.log", "a", encoding="utf-8")
    fh.write(time.strftime(f"\n===== Start {stage}: %Y-%m-%d %H:%M:%S =====\n"))
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)


ensure_dirs()
