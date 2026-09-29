# Experiment 2 — Toponym resolution on Text+Berg

Evaluates the five encoder models (M1–M5) with the default and the H3-based sentence generators (B1/D1 association, config ablation) on the Text+Berg corpus and produces the result tables and figures of the thesis.

```bash
# run from ma-experiments/experiment-2; inputs in ../data/ (text_berg/, swissnames25/, swissnames3d/)
python 1_preprocessing/corpus_to_geopackage.py
python 1_preprocessing/rematch_swissnames3d.py
python 2_dataset/build_eval_dataset.py                      # -> ../data/eval_dataset.json
for c in 1 2 3 4 5; do spatial-h3-build --config ../configs/config$c.yaml --output ../output/config$c/spatial_h3.duckdb; done
spatial-h3-assoc --measure d1 --config ../configs/config1.yaml --db ../output/config1/spatial_h3.duckdb
bash 3_evaluation/run_pipeline.sh                           # stages 01–16 -> results/ (16 = uncapped ablations all_uniform/all_b1/all_d1 + context counts)
python 4_analysis/<script>.py                               # follow-up analyses (E_d1, register numbers)
python 5_figures/abb_3_01_daten_sammeln.py && python 5_figures/abb_3_02_abbildung.py
for f in 5_figures/abb_{4,5,6,7,8,9,10}_*.py; do python "$f"; done   # -> 5_figures/output/
```

The Text+Berg reference annotation (rematching to swissNAMES3D) is not published; the corpus is available via SWISSUbase (doi:10.48656/akt5-q239).
