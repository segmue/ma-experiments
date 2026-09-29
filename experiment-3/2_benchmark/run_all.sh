#!/usr/bin/env bash
# =============================================================================
# Experiment 3 (system performance): full benchmark chain.
#
#   bash 2_benchmark/run_all.sh                  full run  -> experiment-3/results/
#   BENCH_SMOKE=1 bash 2_benchmark/run_all.sh    mini run  -> experiment-3/results_smoke/
#   BENCH_VARIANT=_rerun bash 2_benchmark/run_all.sh
#                                                fresh run -> experiment-3/results_rerun/
#
# Stage order (as in the measured runs): 01 build geo DBs (only if missing)
# -> verify_geo_strtree (gate, must pass) -> 02 micro -> 03 descriptions/parity
# -> 04 document throughput (up to 3 attempts, checkpointed) -> 05 accuracy
# delta -> 06 report.
# Restart is safe: a stage whose result JSON exists is skipped, finished
# blocks are resumed from checkpoints in experiment-3/cache*/.
# Prerequisites: output/configN/spatial_h3.duckdb and the items cache from
# experiment-2/3_evaluation (cache/items_E_c1.pkl, items_E_c2.pkl).
# validate_geo_engine.py (GeometricEngine vs. shapely) is an optional manual
# check after 01. Use PYTHON=... to select the interpreter (default: python).
# =============================================================================
set -uo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1
export BENCH_SMOKE="${BENCH_SMOKE:-}" BENCH_VARIANT="${BENCH_VARIANT:-}"

SUFFIX=""
[ "$BENCH_SMOKE" = "1" ] && SUFFIX="_smoke"
RESDIR="../results${SUFFIX}${BENCH_VARIANT}"
LOGDIR="../logs${SUFFIX}${BENCH_VARIANT}"
OUTDIR="../../output"
mkdir -p "$LOGDIR"
RUNLOG="$LOGDIR/runner.log"

log() { echo "$*"; echo "$*" >> "$RUNLOG"; }
now() { date '+%Y-%m-%d %H:%M:%S'; }

# stage <script> [result-file]: skip if result exists, abort on error
stage() {
    local script="$1" result="${2:-}"
    if [ -n "$result" ] && [ -e "$result" ]; then
        log "[skip]  $script -- $result exists"
        return 0
    fi
    log "[start] $script   $(now)"
    if ! "$PYTHON" -u "$script"; then
        log "[ERROR] $script -- exit code != 0, aborting (see $LOGDIR/)"
        exit 1
    fi
    log "[ok]    $script   $(now)"
}

log "===== run_all start $(now) (BENCH_SMOKE=$BENCH_SMOKE BENCH_VARIANT=$BENCH_VARIANT) ====="

# 01: geometric twin DBs; not part of the timed runs, built once.
if [ -e "$OUTDIR/geo_config1/spatial_geo.duckdb" ] && [ -e "$OUTDIR/geo_config2/spatial_geo.duckdb" ]; then
    log "[skip]  01_build_geo_dbs.py -- geo DBs exist"
else
    stage 01_build_geo_dbs.py
fi

# Gate: geo_strtree must return identical results and sentences as geo.
stage verify_geo_strtree.py "$RESDIR/geo_strtree_verification.json"

stage 02_micro_benchmarks.py  "$RESDIR/micro.json"
stage 03_descriptions_parity.py "$RESDIR/parity.json"

# 04: up to three attempts; every finished (system, set) is checkpointed.
[ -e "$RESDIR/throughput.json" ] && log "[skip]  04_doc_throughput.py -- $RESDIR/throughput.json exists"
for attempt in 1 2 3; do
    [ -e "$RESDIR/throughput.json" ] && break
    log "[start] 04_doc_throughput.py -- attempt $attempt   $(now)"
    if "$PYTHON" -u 04_doc_throughput.py; then
        log "[ok]    04_doc_throughput.py   $(now)"
    else
        log "[warn]  attempt $attempt aborted -- progress is checkpointed"
    fi
done
if [ ! -e "$RESDIR/throughput.json" ]; then
    log "[ERROR] 04_doc_throughput.py not finished after three attempts"
    exit 1
fi

stage 05_acc_geo.py "$RESDIR/acc_geo.json"
stage 06_report.py

log "===== run_all done $(now) ====="
echo "Results in $RESDIR/, logs in $LOGDIR/."
