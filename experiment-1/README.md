# Experiment 1: Cross-validation in the training domain

Fine-tunes and cross-evaluates five models (M1–M5) against three description variants (E_default, E_spatial_config1, E_spatial_config2) on annotated Swissdox articles.

```bash
python 01_clean_swissdox.py   # ../data/swissdox/*.xz -> ../data/swissdox_preprocessed/
python 02_rank_and_batch.py   # spaCy toponym ranking, batches of 100
python 03_export_txt.py       # text chunks for annotation (<=1200 chars)
# annotate in the geoparser annotator, export JSONs to ../data/annotations/
python 04_build.py            # H3 index + B1 matrix for config1, config2
python 05_preprocess.py       # -> ../data/preprocessed.json
python 06_train.py            # HP search, 5-fold CV, final models -> results/
```

The Swissdox annotations are not included for licensing reasons; the trained models are available at https://huggingface.co/segmue.
