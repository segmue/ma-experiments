"""
Step 2b: Training-Pipeline mit K-Fold CV, Grid Search und Experiment-Matrix.

Dreiphasige Pipeline:
    Phase 1: Hyperparameter-Suche (Grid Search + 5-Fold CV auf config1/default)
    Phase 2: Experiment-Matrix (10 Konfigurationen mit besten HPs, 5-Fold CV)
    Phase 3: Finale Modelle (Training auf Gesamtdaten)

Voraussetzungen:
    - data/preprocessed.json   (von 02_preprocess.py erzeugt)
    - output/config1/ und output/config2/ (von 01_build.py erzeugt)

Output:
    results/hp_search/hp_search_results.json
    results/experiments/{config}_{variant}/results.json
    results/summary.csv
    models/{config}_{variant}/

Verwendung:
    poetry run python 02_train.py                  # Phase 1 + 2 + 3
    poetry run python 02_train.py --phase 1        # Nur HP-Suche
    poetry run python 02_train.py --phase 2        # Nur Experiment-Matrix
    poetry run python 02_train.py --phase 3        # Nur finale Modelle
    poetry run python 02_train.py --skip-hp-search # Phase 2 + 3 mit gecachten HPs
    poetry run python 02_train.py --n-folds 3      # Weniger Folds
"""

import argparse
import csv
import gc
import itertools
import json
import logging
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold

from geoparser_h3_resolver import SpatialSentenceResolver
from geoparser_h3_resolver.pipeline.build_config import BuildConfig
from geoparser_h3_resolver.sentence_generator.config import (
    SentenceGeneratorConfig,
    StaticSlotConfig,
)

# ── Constants ────────────────────────────────────────────────────────────────

BASE = Path(__file__).parent
PREPROCESSED_PATH = BASE / "data" / "preprocessed.json"
RESULTS_DIR = BASE / "results"
MODELS_DIR = BASE / "models"

SEED = 42
DB_CONFIGS = ["config1", "config2"]

# SentenceGenerator-Varianten: Overrides gegenueber der YAML-Config.
# Leeres dict = YAML-Defaults beibehalten (sentence_config=None).
VARIANTS = {
    "default":     {},
    "no_dynamic":  {"max_slots": 0},
    "unlimited":   {"max_slots": 1000},
    "no_static":   {"static_slots": []},
    "with_filler": {"max_filler_slots": 5},
}

# Hyperparameter-Grid fuer Phase 1
HP_GRID = {
    "learning_rate": [1e-5, 2e-5, 5e-5],
    "epochs":        [2, 3, 5],
    "batch_size":    [8, 16],
}

logger = logging.getLogger("experiments")


# ── Setup ────────────────────────────────────────────────────────────────────

def setup_logging():
    """Logging mit Console (INFO) und Datei (DEBUG)."""
    logger.setLevel(logging.DEBUG)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))

    logfile = logging.FileHandler(RESULTS_DIR / "experiment.log", encoding="utf-8")
    logfile.setLevel(logging.DEBUG)
    logfile.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    logger.addHandler(console)
    logger.addHandler(logfile)

    # HuggingFace Trainer Noise reduzieren
    import transformers
    transformers.logging.set_verbosity_warning()


def set_seeds(seed: int = SEED):
    """Reproduzierbare Ergebnisse."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_preprocessed() -> Tuple[Dict, List[Dict]]:
    """Laedt preprocessed.json und gibt (metadata, documents) zurueck."""
    with open(PREPROCESSED_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = data["documents"]
    logger.info(
        "Preprocessed data geladen: %d Dokumente, %d Toponyme",
        len(documents),
        sum(len(doc["references"]) for doc in documents),
    )
    return data, documents


def docs_to_training_format(
    documents: List[Dict],
) -> Tuple[List[str], List[List[Tuple[int, int]]], List[List[Tuple[str, str]]]]:
    """Konvertiert Dokument-Dicts in das Format fuer resolver.fit()."""
    texts = []
    references = []
    referents = []
    for doc in documents:
        texts.append(doc["text"])
        references.append([tuple(r) for r in doc["references"]])
        referents.append([tuple(r) for r in doc["referents"]])
    return texts, references, referents


# ── K-Fold CV ────────────────────────────────────────────────────────────────

def create_document_folds(
    n_documents: int, n_folds: int = 5, seed: int = SEED,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Erstellt K-Fold Splits auf Dokument-Ebene."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(kf.split(range(n_documents)))


# ── SentenceGeneratorConfig ─────────────────────────────────────────────────

def make_sentence_config(
    config_path: Path, duckdb_path: Path, overrides: Dict,
) -> Optional[SentenceGeneratorConfig]:
    """Erstellt SentenceGeneratorConfig mit Varianten-Overrides.

    Returns None fuer leere overrides (= YAML-Defaults verwenden).
    """
    if not overrides:
        return None

    build_config = BuildConfig.from_yaml(config_path)
    matrix_path = duckdb_path.parent / "b1_matrix.csv"
    base = build_config.to_sentence_generator_config(matrix_path)

    # Neue Config mit Overrides
    kwargs = {
        "assoc_threshold": base.assoc_threshold,
        "max_slots": base.max_slots,
        "max_slots_per_category": base.max_slots_per_category,
        "max_categories": base.max_categories,
        "max_filler_slots": base.max_filler_slots,
        "static_slots": list(base.static_slots),
        "matrix_path": base.matrix_path,
    }
    kwargs.update(overrides)
    return SentenceGeneratorConfig(**kwargs)


def sentence_config_to_dict(config: Optional[SentenceGeneratorConfig]) -> Dict:
    """Serialisiert SentenceGeneratorConfig fuer JSON-Output."""
    if config is None:
        return {"_source": "yaml_defaults"}
    d = {
        "assoc_threshold": config.assoc_threshold,
        "max_slots": config.max_slots,
        "max_slots_per_category": config.max_slots_per_category,
        "max_categories": config.max_categories,
        "max_filler_slots": config.max_filler_slots,
        "static_slots": [
            {"objektart": s.objektart, "label": s.label, "slots": s.slots}
            for s in config.static_slots
        ],
    }
    return d


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_fold(resolver: SpatialSentenceResolver, val_documents: List[Dict]) -> Dict:
    """Evaluiert einen trainierten Resolver auf Validation-Dokumenten.

    Berechnet Accuracy@1, Accuracy@3 und MRR indem fuer jedes Toponym
    alle Gazetteer-Kandidaten gerankt und mit der Gold-loc_id verglichen werden.
    """
    correct_at_1 = 0
    correct_at_3 = 0
    reciprocal_ranks = []
    total = 0
    no_candidates = 0

    for doc in val_documents:
        text = doc["text"]
        for ref, referent in zip(doc["references"], doc["referents"]):
            start, end = ref[0], ref[1]
            gold_loc_id = referent[1]
            toponym_text = text[start:end]
            total += 1

            # Kandidaten suchen
            candidates = resolver.gazetteer.search(toponym_text)
            if not candidates:
                reciprocal_ranks.append(0.0)
                no_candidates += 1
                continue

            # Kontext extrahieren und Embeddings berechnen
            context = resolver._extract_context(text, start, end)
            context_emb = resolver.transformer.encode(
                [context], convert_to_tensor=True, show_progress_bar=False,
            )[0]

            # Beschreibungen generieren und Embeddings berechnen
            descriptions = [resolver._generate_description(c) for c in candidates]
            cand_embs = resolver.transformer.encode(
                descriptions, convert_to_tensor=True, show_progress_bar=False,
            )

            # Cosine Similarity
            similarities = torch.nn.functional.cosine_similarity(
                context_emb.unsqueeze(0), cand_embs, dim=1,
            )

            # Ranking (absteigend nach Similarity)
            ranked_indices = similarities.argsort(descending=True).tolist()

            # Gold-Kandidat finden
            rank = None
            for pos, idx in enumerate(ranked_indices):
                if candidates[idx].location_id_value == gold_loc_id:
                    rank = pos + 1
                    break

            if rank == 1:
                correct_at_1 += 1
            if rank is not None and rank <= 3:
                correct_at_3 += 1
            reciprocal_ranks.append(1.0 / rank if rank else 0.0)

    return {
        "accuracy_at_1": correct_at_1 / total if total else 0.0,
        "accuracy_at_3": correct_at_3 / total if total else 0.0,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "total": total,
        "no_candidates": no_candidates,
    }


# ── Training Helpers ─────────────────────────────────────────────────────────

def extract_training_loss(model_dir: Path) -> List[float]:
    """Extrahiert per-epoch Training Loss aus HuggingFace trainer_state.json."""
    state_file = model_dir / "trainer_state.json"
    if not state_file.exists():
        return []

    with open(state_file) as f:
        state = json.load(f)

    log_history = state.get("log_history", [])
    # Sammle Loss-Werte pro Epoch (entries mit 'loss' key)
    losses = [entry["loss"] for entry in log_history if "loss" in entry]
    return losses


def cleanup_model_dir(model_dir: Path):
    """Entfernt Checkpoint-Ordner und temporaere Dateien nach Evaluation."""
    if not model_dir.exists():
        return
    for item in model_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            shutil.rmtree(item)


def free_resources():
    """Speicher freigeben nach Training/Evaluation."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_and_evaluate_fold(
    documents: List[Dict],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    fold_idx: int,
    model_dir: Path,
    base_model: str,
    config_path: Path,
    duckdb_path: Path,
    sentence_config: Optional[SentenceGeneratorConfig],
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> Dict:
    """Trainiert einen Fold und evaluiert. Gibt Metriken zurueck."""
    train_docs = [documents[i] for i in train_idx]
    val_docs = [documents[i] for i in val_idx]

    texts, refs, referents = docs_to_training_format(train_docs)

    train_toponyms = sum(len(r) for r in refs)
    val_toponyms = sum(len(doc["references"]) for doc in val_docs)

    logger.debug(
        "  Fold %d: %d train docs (%d toponyms), %d val docs (%d toponyms)",
        fold_idx, len(train_docs), train_toponyms, len(val_docs), val_toponyms,
    )

    # Training
    model_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    resolver = SpatialSentenceResolver(
        model_name=base_model,
        gazetteer_name="swissnames3d",
        config_path=config_path,
        duckdb_path=duckdb_path,
        sentence_config=sentence_config,
    )

    resolver.fit(
        texts=texts,
        references=refs,
        referents=referents,
        output_path=model_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        save_strategy="no",
    )

    training_loss = extract_training_loss(model_dir)
    train_duration = time.time() - t0

    del resolver
    free_resources()

    # Evaluation mit trainiertem Modell
    t1 = time.time()
    eval_resolver = SpatialSentenceResolver(
        model_name=str(model_dir),
        gazetteer_name="swissnames3d",
        config_path=config_path,
        duckdb_path=duckdb_path,
        sentence_config=sentence_config,
    )

    metrics = evaluate_fold(eval_resolver, val_docs)
    eval_duration = time.time() - t1

    del eval_resolver
    free_resources()

    cleanup_model_dir(model_dir)

    result = {
        "fold": fold_idx,
        "train_documents": len(train_docs),
        "val_documents": len(val_docs),
        "train_toponyms": train_toponyms,
        "val_toponyms": val_toponyms,
        **metrics,
        "training_loss": training_loss,
        "train_duration_seconds": round(train_duration, 1),
        "eval_duration_seconds": round(eval_duration, 1),
    }

    logger.info(
        "  Fold %d: Acc@1=%.3f, Acc@3=%.3f, MRR=%.3f (%d toponyms, %.0fs)",
        fold_idx, metrics["accuracy_at_1"], metrics["accuracy_at_3"],
        metrics["mrr"], metrics["total"], train_duration + eval_duration,
    )

    return result


def aggregate_fold_metrics(fold_results: List[Dict]) -> Dict:
    """Berechnet Mean und Std ueber Fold-Metriken."""
    metrics = {}
    for key in ["accuracy_at_1", "accuracy_at_3", "mrr"]:
        values = [r[key] for r in fold_results]
        metrics[f"mean_{key}"] = float(np.mean(values))
        metrics[f"std_{key}"] = float(np.std(values))
    return metrics


# ── Phase 1: Hyperparameter Search ───────────────────────────────────────────

def run_hp_search(
    documents: List[Dict],
    base_model: str,
    n_folds: int = 5,
) -> Dict:
    """Phase 1: Grid Search ueber Hyperparameter mit K-Fold CV.

    Sucht auf config1/default nach den besten Hyperparametern.
    """
    config_path = BASE / "configs" / "config1.yaml"
    duckdb_path = BASE / "output" / "config1" / "spatial_h3.duckdb"

    if not duckdb_path.exists():
        raise FileNotFoundError(
            f"{duckdb_path} nicht gefunden. Erst 01_build.py ausfuehren."
        )

    hp_dir = RESULTS_DIR / "hp_search"
    hp_dir.mkdir(parents=True, exist_ok=True)

    folds = create_document_folds(len(documents), n_folds=n_folds)

    hp_combos = list(itertools.product(
        HP_GRID["learning_rate"],
        HP_GRID["epochs"],
        HP_GRID["batch_size"],
    ))

    logger.info("=" * 60)
    logger.info("PHASE 1: Hyperparameter-Suche")
    logger.info("  Grid: %d Kombinationen x %d Folds = %d Runs",
                len(hp_combos), n_folds, len(hp_combos) * n_folds)
    logger.info("  Referenz-Config: config1/default")
    logger.info("=" * 60)

    all_results = []

    for hp_idx, (lr, epochs, bs) in enumerate(hp_combos):
        hp_key = f"lr={lr}_ep={epochs}_bs={bs}"
        logger.info(
            "\n[Phase 1] HP %d/%d: %s", hp_idx + 1, len(hp_combos), hp_key,
        )

        fold_results = []
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            model_dir = hp_dir / hp_key / f"fold{fold_idx}"

            result = train_and_evaluate_fold(
                documents=documents,
                train_idx=train_idx,
                val_idx=val_idx,
                fold_idx=fold_idx,
                model_dir=model_dir,
                base_model=base_model,
                config_path=config_path,
                duckdb_path=duckdb_path,
                sentence_config=None,  # default variant
                epochs=epochs,
                batch_size=bs,
                learning_rate=lr,
            )
            fold_results.append(result)

        agg = aggregate_fold_metrics(fold_results)
        logger.info(
            "[Phase 1] HP %d/%d complete: Acc@1=%.3f±%.3f, MRR=%.3f±%.3f",
            hp_idx + 1, len(hp_combos),
            agg["mean_accuracy_at_1"], agg["std_accuracy_at_1"],
            agg["mean_mrr"], agg["std_mrr"],
        )

        all_results.append({
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": bs,
            **agg,
            "fold_metrics": fold_results,
        })

    # Beste Kombination: hoechstes mean_accuracy_at_1, Tiebreaker mean_mrr
    best = max(all_results, key=lambda r: (r["mean_accuracy_at_1"], r["mean_mrr"]))

    output = {
        "grid": HP_GRID,
        "reference_config": "config1",
        "reference_variant": "default",
        "n_folds": n_folds,
        "seed": SEED,
        "results": all_results,
        "best": {
            "learning_rate": best["learning_rate"],
            "epochs": best["epochs"],
            "batch_size": best["batch_size"],
            "mean_accuracy_at_1": best["mean_accuracy_at_1"],
            "mean_mrr": best["mean_mrr"],
        },
    }

    output_path = hp_dir / "hp_search_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1 ABGESCHLOSSEN")
    logger.info("  Beste HPs: lr=%s, epochs=%d, bs=%d",
                best["learning_rate"], best["epochs"], best["batch_size"])
    logger.info("  Acc@1=%.3f±%.3f, MRR=%.3f±%.3f",
                best["mean_accuracy_at_1"], best["std_accuracy_at_1"],
                best["mean_mrr"], best["std_mrr"])
    logger.info("  Gespeichert: %s", output_path)
    logger.info("=" * 60)

    return output


# ── Phase 2: Experiment Matrix ───────────────────────────────────────────────

def run_experiments(
    documents: List[Dict],
    best_hps: Dict,
    base_model: str,
    n_folds: int = 5,
) -> List[Dict]:
    """Phase 2: Alle 10 Konfigurationen mit besten HPs, K-Fold CV."""
    folds = create_document_folds(len(documents), n_folds=n_folds)

    lr = best_hps["learning_rate"]
    epochs = best_hps["epochs"]
    bs = best_hps["batch_size"]

    exp_dir = RESULTS_DIR / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    total_experiments = len(DB_CONFIGS) * len(VARIANTS)
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Experiment-Matrix")
    logger.info("  %d DB-Configs x %d Varianten = %d Experimente",
                len(DB_CONFIGS), len(VARIANTS), total_experiments)
    logger.info("  HPs: lr=%s, epochs=%d, bs=%d", lr, epochs, bs)
    logger.info("  %d Folds pro Experiment = %d Runs total",
                n_folds, total_experiments * n_folds)
    logger.info("=" * 60)

    all_experiment_results = []
    exp_idx = 0

    for config_name in DB_CONFIGS:
        config_path = BASE / "configs" / f"{config_name}.yaml"
        duckdb_path = BASE / "output" / config_name / "spatial_h3.duckdb"

        if not duckdb_path.exists():
            logger.warning("  SKIP %s: %s nicht gefunden", config_name, duckdb_path)
            continue

        for variant_name, overrides in VARIANTS.items():
            exp_idx += 1
            exp_name = f"{config_name}_{variant_name}"

            sentence_config = make_sentence_config(config_path, duckdb_path, overrides)

            logger.info(
                "\n[Phase 2] Experiment %d/%d: %s",
                exp_idx, total_experiments, exp_name,
            )
            if overrides:
                logger.info("  Overrides: %s", overrides)

            fold_results = []
            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                model_dir = exp_dir / exp_name / f"fold{fold_idx}"

                result = train_and_evaluate_fold(
                    documents=documents,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    fold_idx=fold_idx,
                    model_dir=model_dir,
                    base_model=base_model,
                    config_path=config_path,
                    duckdb_path=duckdb_path,
                    sentence_config=sentence_config,
                    epochs=epochs,
                    batch_size=bs,
                    learning_rate=lr,
                )
                fold_results.append(result)

            agg = aggregate_fold_metrics(fold_results)

            experiment_result = {
                "config": config_name,
                "variant": variant_name,
                "hyperparameters": {
                    "learning_rate": lr,
                    "epochs": epochs,
                    "batch_size": bs,
                    "warmup_ratio": 0.1,
                },
                "sentence_config_overrides": overrides,
                "effective_sentence_config": sentence_config_to_dict(sentence_config),
                "n_folds": n_folds,
                "seed": SEED,
                "folds": fold_results,
                "aggregate": agg,
            }

            # Pro Variante speichern
            result_path = exp_dir / exp_name / "results.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(experiment_result, f, ensure_ascii=False, indent=2)

            all_experiment_results.append(experiment_result)

            logger.info(
                "[Phase 2] %s: Acc@1=%.3f±%.3f, Acc@3=%.3f±%.3f, MRR=%.3f±%.3f",
                exp_name,
                agg["mean_accuracy_at_1"], agg["std_accuracy_at_1"],
                agg["mean_accuracy_at_3"], agg["std_accuracy_at_3"],
                agg["mean_mrr"], agg["std_mrr"],
            )

    # Summary CSV
    generate_summary_csv(all_experiment_results)

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2 ABGESCHLOSSEN")
    logger.info("  %d Experimente evaluiert", len(all_experiment_results))
    logger.info("  Ergebnisse: %s", exp_dir)
    logger.info("=" * 60)

    return all_experiment_results


def generate_summary_csv(experiment_results: List[Dict]):
    """Generiert results/summary.csv mit einer Zeile pro Variante."""
    csv_path = RESULTS_DIR / "summary.csv"
    fieldnames = [
        "config", "variant",
        "mean_acc1", "std_acc1",
        "mean_acc3", "std_acc3",
        "mean_mrr", "std_mrr",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for exp in experiment_results:
            agg = exp["aggregate"]
            writer.writerow({
                "config": exp["config"],
                "variant": exp["variant"],
                "mean_acc1": f"{agg['mean_accuracy_at_1']:.4f}",
                "std_acc1": f"{agg['std_accuracy_at_1']:.4f}",
                "mean_acc3": f"{agg['mean_accuracy_at_3']:.4f}",
                "std_acc3": f"{agg['std_accuracy_at_3']:.4f}",
                "mean_mrr": f"{agg['mean_mrr']:.4f}",
                "std_mrr": f"{agg['std_mrr']:.4f}",
            })

    logger.info("  Summary CSV: %s", csv_path)


# ── Phase 3: Final Models ───────────────────────────────────────────────────

def train_final_models(
    documents: List[Dict],
    best_hps: Dict,
    base_model: str,
):
    """Phase 3: Finale Modelle auf Gesamtdaten trainieren."""
    lr = best_hps["learning_rate"]
    epochs = best_hps["epochs"]
    bs = best_hps["batch_size"]

    texts, refs, referents = docs_to_training_format(documents)

    total_models = len(DB_CONFIGS) * len(VARIANTS)
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: Finale Modelle")
    logger.info("  %d Modelle auf Gesamtdaten (%d Dokumente, %d Toponyme)",
                total_models, len(documents),
                sum(len(doc["references"]) for doc in documents))
    logger.info("  HPs: lr=%s, epochs=%d, bs=%d", lr, epochs, bs)
    logger.info("=" * 60)

    model_idx = 0
    for config_name in DB_CONFIGS:
        config_path = BASE / "configs" / f"{config_name}.yaml"
        duckdb_path = BASE / "output" / config_name / "spatial_h3.duckdb"

        if not duckdb_path.exists():
            logger.warning("  SKIP %s: %s nicht gefunden", config_name, duckdb_path)
            continue

        for variant_name, overrides in VARIANTS.items():
            model_idx += 1
            exp_name = f"{config_name}_{variant_name}"
            model_dir = MODELS_DIR / exp_name

            sentence_config = make_sentence_config(config_path, duckdb_path, overrides)

            logger.info(
                "\n[Phase 3] Modell %d/%d: %s", model_idx, total_models, exp_name,
            )

            model_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()

            resolver = SpatialSentenceResolver(
                model_name=base_model,
                gazetteer_name="swissnames3d",
                config_path=config_path,
                duckdb_path=duckdb_path,
                sentence_config=sentence_config,
            )

            resolver.fit(
                texts=texts,
                references=refs,
                referents=referents,
                output_path=model_dir,
                epochs=epochs,
                batch_size=bs,
                learning_rate=lr,
            )

            duration = time.time() - t0
            cleanup_model_dir(model_dir)

            del resolver
            free_resources()

            logger.info(
                "  -> Gespeichert: %s (%.0fs)", model_dir, duration,
            )

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3 ABGESCHLOSSEN")
    logger.info("  %d Modelle gespeichert in %s", model_idx, MODELS_DIR)
    logger.info("=" * 60)


# ── Main ─────────────────────────────────────────────────────────────────────

def load_best_hps() -> Dict:
    """Laedt gecachte HP-Search-Ergebnisse."""
    hp_path = RESULTS_DIR / "hp_search" / "hp_search_results.json"
    if not hp_path.exists():
        raise FileNotFoundError(
            f"{hp_path} nicht gefunden. Erst Phase 1 ausfuehren "
            "(poetry run python 02_train.py --phase 1)"
        )
    with open(hp_path) as f:
        return json.load(f)["best"]


def main():
    parser = argparse.ArgumentParser(description="Training pipeline")
    parser.add_argument(
        "--phase", type=int, choices=[1, 2, 3], default=None,
        help="Nur eine Phase ausfuehren (1=HP-Suche, 2=Experimente, 3=Finale Modelle)",
    )
    parser.add_argument(
        "--skip-hp-search", action="store_true",
        help="Phase 2+3 mit gecachten HPs (ueberspringt Phase 1)",
    )
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Anzahl Folds fuer Cross-Validation (default: 5)",
    )
    args = parser.parse_args()

    setup_logging()
    set_seeds()

    # Daten laden
    data, documents = load_preprocessed()
    base_model = data.get("base_model", "sentence-transformers/distiluse-base-multilingual-cased-v1")

    # Phasen bestimmen
    if args.phase == 1:
        phases = [1]
    elif args.phase == 2:
        phases = [2]
    elif args.phase == 3:
        phases = [3]
    elif args.skip_hp_search:
        phases = [2, 3]
    else:
        phases = [1, 2, 3]

    best_hps = None

    # Phase 1
    if 1 in phases:
        hp_results = run_hp_search(documents, base_model, n_folds=args.n_folds)
        best_hps = hp_results["best"]

    # Phase 2
    if 2 in phases:
        if best_hps is None:
            best_hps = load_best_hps()
            logger.info("Gecachte HPs geladen: %s", best_hps)
        run_experiments(documents, best_hps, base_model, n_folds=args.n_folds)

    # Phase 3
    if 3 in phases:
        if best_hps is None:
            best_hps = load_best_hps()
            logger.info("Gecachte HPs geladen: %s", best_hps)
        train_final_models(documents, best_hps, base_model)

    logger.info("\nPipeline abgeschlossen.")


if __name__ == "__main__":
    main()
