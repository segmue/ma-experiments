#!/usr/bin/env bash
# Orchestriert die Text+Berg-Evaluation (Stage 01–15).
#
# Verwendung:
#   ./run_pipeline.sh                 # voller Lauf (rechenintensiv!)
#   ./run_pipeline.sh --max-docs 50   # Smoke-Test (an 01–04 durchgereicht)
#   ./run_pipeline.sh --skip-full     # ohne Full-Pipeline (Stage 03)
#
# Voraussetzungen (siehe README):
#   - ma-experiments/data/eval_dataset.json      (1_preprocessing -> 2_dataset)
#   - ma-experiments/output/config1..5/spatial_h3.duckdb + b1_matrix.csv  (spatial-h3-build)
#   - ma-experiments/output/config1/d1_matrix.csv                          (spatial-h3-assoc --measure d1)
#   - Umgebung aus dem Root-pyproject (geoparser, geoparser_h3_resolver, torch, spaCy-Modelle)
set -euo pipefail
cd "$(dirname "$0")"

PASS=()          # an 01–04 durchgereichte Flags (z.B. --max-docs)
RUN_FULL=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-docs) PASS+=("--max-docs" "$2"); shift 2 ;;
    --skip-full) RUN_FULL=0; shift ;;
    *) echo "Unbekanntes Flag: $1"; exit 2 ;;
  esac
done

PY="${PYTHON:-python}"
M345=(M3_default_finetuned M4_spatial_config1 M5_spatial_config2)
D1_ARMS=(E_d1 E_d1_s01 E_d1_s1 E_d1_c6 E_d1_c20 E_b1_c6 E_b1_c20)

echo "== Prerequisite-Check =="
$PY - <<'PYCHECK'
import sys, config as C
missing = C.check_prerequisites(
    require_spatial=True, require_local_models=True,
    resolvers=C.MATRIX_RESOLVERS + C.CONFIG_ABLATION_RESOLVERS + C.D1_RESOLVERS)
if missing:
    print("FEHLENDE PREREQUISITES:")
    for m in missing:
        print("  -", m)
    sys.exit(1)
print("  OK")
PYCHECK

echo "== Stage 01: prepare_items (Hauptmatrix, Config-Ablation, E_d1-Arme) =="
$PY 01_prepare_items.py "${PASS[@]}"
$PY 01_prepare_items.py --resolver E_c3 E_c4 E_c5 "${PASS[@]}"
$PY 01_prepare_items.py --resolver "${D1_ARMS[@]}" "${PASS[@]}"

echo "== Stage 02: evaluate_matrix =="
$PY 02_evaluate_matrix.py "${PASS[@]}"                                   # 5×3 -> summary.csv
$PY 02_evaluate_matrix.py --models "${M345[@]}" --resolvers E_c3 E_c4 E_c5 \
    --summary config_ablation/summary_e8.csv "${PASS[@]}"
# E_d1: 5 Hauptzellen + Achsen (nur M4/M5); Zwischen-Summaries in logs/,
# results/summary_E_d1.csv entsteht aus den Zell-JSONs (4_analysis/merge_summary_E_d1.py).
$PY 02_evaluate_matrix.py --resolvers E_d1 --summary "$PWD/logs/summary_E_d1_main.csv" "${PASS[@]}"
$PY 02_evaluate_matrix.py --models M4_spatial_config1 M5_spatial_config2 \
    --resolvers E_d1_s01 E_d1_s1 E_d1_c6 E_d1_c20 E_b1_c6 E_b1_c20 \
    --summary "$PWD/logs/summary_E_d1_axes.csv" "${PASS[@]}"
$PY ../4_analysis/merge_summary_E_d1.py

if [[ $RUN_FULL -eq 1 ]]; then
  echo "== Stage 03: full_pipeline =="
  $PY 03_full_pipeline.py "${PASS[@]}"
  $PY run_fp_chunked.py --resolver E_d1 --out full_pipeline_E_d1.json
fi

echo "== Stage 04: ablations (M5, M4) =="
$PY 04_ablations.py "${PASS[@]}"
$PY 04_ablations.py --model M4_spatial_config1 --out ablations_M4.json "${PASS[@]}"

echo "== Stage 05–15: Auswertung =="
$PY 05_stratified_objektart.py
$PY 06_no_candidate_examples.py
if [[ $RUN_FULL -eq 1 ]]; then $PY 07_fp_sample_analysis.py; fi
$PY 08_plot_ambiguity.py
$PY 09_mcnemar_main.py
$PY 10_mcnemar_config_ablation.py
$PY 11_class_profile.py
$PY 12_stratified_delta.py
$PY 13_plot_config_ablation.py
$PY 14_random_baseline.py
$PY 15_obergrenze_ambig.py

echo "== Fertig. Ergebnisse in ../results/, Plots in plots/ =="
