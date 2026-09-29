# Experiment 3: System performance

Compares the H3 cell-based index with three geometric reference query layers (DuckDB Spatial Join, DuckDB ST_DWithin, Shapely STRtree) on description parity, runtime and accuracy delta.

```bash
bash 2_benchmark/run_all.sh                 # full run -> results/
BENCH_SMOKE=1 bash 2_benchmark/run_all.sh   # smoke run -> results_smoke/
```

Requires the items cache from experiment-2 (`experiment-2/3_evaluation/cache/items_E_c1.pkl`, `items_E_c2.pkl`) and the H3 databases in `output/configN/`.

`results/build.json` (11 June 2026) and `results/acc_geo.json` / `tables/acc_geo_delta.csv` (12 June 2026) come from the June runs; all other files in `results/` come from the run of 16 September 2026.
