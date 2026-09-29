# ma-experiments

Experiments of the master's thesis *Describing Places by Their Surroundings: Enriching Candidate Descriptions with Spatial Context for Toponym Resolution* (University of Zurich, 2026).

- `experiment-1/` cross-validation in the training domain (Swissdox)
- `experiment-2/` evaluation on Text+Berg, ablations
- `experiment-3/` system performance (H3 vs. geometric query layers)
- `configs/` config1–5; `matrices/` association matrices (B1, D1)

```bash
poetry install
python -m spacy download de_core_news_lg
```

Code: [ma-geoparser-h3-resolver](https://github.com/segmue/ma-geoparser-h3-resolver), [ma-h3-multi-resolution-engine](https://github.com/segmue/ma-h3-multi-resolution-engine). Models: [huggingface.co/segmue](https://huggingface.co/segmue).
