"""
Step 2b: Training-Pipeline mit K-Fold CV, Grid Search und Cross-Evaluation.

Systematischer Vergleich: 5 Modelle x 3 Eval-Resolver = 15 Evaluationen.

Modelle:
    M1: dguzh/geo-all-MiniLM-L6-v2         (Autoren-Modell, kein Training)
    M2: distiluse-multilingual               (Base, kein Training)
    M3: distiluse fine-tuned + Default       (SentenceTransformerResolver)
    M4: distiluse fine-tuned + Spatial c1    (SpatialSentenceResolver config1)
    M5: distiluse fine-tuned + Spatial c2    (SpatialSentenceResolver config2)

Eval-Resolver:
    E_default:         SentenceTransformerResolver (einfache Admin-Hierarchie)
    E_spatial_config1: SpatialSentenceResolver config1 (H3 overlap, res13)
    E_spatial_config2: SpatialSentenceResolver config2 (H3 center, res10)

Pipeline:
    Phase 1: HP-Suche (Grid Search + K-Fold CV auf M4/config1)
    Phase 2: K-Fold CV Training (M3, M4, M5) + Cross-Evaluation (alle 15 Combos)
    Phase 3: Finale Modelle (M3, M4, M5 auf Gesamtdaten)

Performance-Optimierung:
    - fit() wird umgangen: _prepare_training_data() gecacht, Trainer direkt (Ebene B)
    - Eval-Daten (Contexts + Descriptions) pro Eval-Resolver gecacht (Ebene C)
    - Feature-Level Description Cache im CandidateSentenceGenerator (Ebene A)

Voraussetzungen:
    - data/preprocessed.json   (von 02_preprocess.py erzeugt)
    - output/config1/ und output/config2/ (von 01_build.py erzeugt)

Verwendung:
    poetry run python 02_train.py                  # Phase 1 + 2 + 3
    poetry run python 02_train.py --phase 1        # Nur HP-Suche
    poetry run python 02_train.py --phase 2        # Nur Cross-Evaluation
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
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset
from sentence_transformers.losses import ContrastiveLoss
from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sklearn.model_selection import KFold

from geoparser.modules.resolvers.sentencetransformer import SentenceTransformerResolver
from geoparser_h3_resolver import SpatialSentenceResolver

# ── Constants ────────────────────────────────────────────────────────────────

BASE = Path(__file__).parent
PREPROCESSED_PATH = BASE / "data" / "preprocessed.json"
RESULTS_DIR = BASE / "results"
MODELS_DIR = BASE / "models"

SEED = 42

DGUZH_MODEL = "dguzh/geo-all-MiniLM-L6-v2"
BASE_MODEL = "sentence-transformers/distiluse-base-multilingual-cased-v1"

# 5 Modelle: ID -> Konfiguration
MODELS = {
    "M1_dguzh": {
        "base_model": DGUZH_MODEL,
        "train": False,
    },
    "M2_distiluse_base": {
        "base_model": BASE_MODEL,
        "train": False,
    },
    "M3_default_finetuned": {
        "base_model": BASE_MODEL,
        "train": True,
        "train_resolver": "default",
    },
    "M4_spatial_config1": {
        "base_model": BASE_MODEL,
        "train": True,
        "train_resolver": "spatial",
        "config": "config1",
    },
    "M5_spatial_config2": {
        "base_model": BASE_MODEL,
        "train": True,
        "train_resolver": "spatial",
        "config": "config2",
    },
}

# 3 Eval-Resolver
EVAL_RESOLVERS = {
    "E_default": {"type": "default"},
    "E_spatial_config1": {"type": "spatial", "config": "config1"},
    "E_spatial_config2": {"type": "spatial", "config": "config2"},
}

# Trainierbare Modelle (fuer Schleifen)
TRAINABLE_MODELS = {k: v for k, v in MODELS.items() if v["train"]}

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
    """Konvertiert Dokument-Dicts in das Format fuer _prepare_training_data()."""
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
    logger.info("Document-Folds erstellt: %d Folds, %d Dokumente", n_folds, n_documents)
    return list(kf.split(range(n_documents)))


# ── Resolver Factory ─────────────────────────────────────────────────────────

def create_resolver(
    model_name: str,
    resolver_type: str,
    config_name: Optional[str] = None,
) -> Union[SentenceTransformerResolver, SpatialSentenceResolver]:
    """Erstellt einen Resolver basierend auf Typ.

    Args:
        model_name: HuggingFace Modellname oder lokaler Pfad
        resolver_type: "default" oder "spatial"
        config_name: "config1" oder "config2" (nur fuer spatial)
    """
    if resolver_type == "default":
        return SentenceTransformerResolver(
            model_name=model_name,
            gazetteer_name="swissnames3d",
        )
    else:
        config_path = BASE / "configs" / f"{config_name}.yaml"
        duckdb_path = BASE / "output" / config_name / "spatial_h3.duckdb"
        return SpatialSentenceResolver(
            model_name=model_name,
            gazetteer_name="swissnames3d",
            config_path=config_path,
            duckdb_path=duckdb_path,
        )


# ── Training (Ebene B: fit() umgehen) ───────────────────────────────────────

def run_training(
    resolver: Union[SentenceTransformerResolver, SpatialSentenceResolver],
    training_data: Dict[str, list],
    output_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    warmup_ratio: float = 0.1,
    save_strategy: str = "no",
):
    """Trainiert das Modell mit vorbereiteten Training-Daten.

    Repliziert die Trainer-Logik aus SentenceTransformerResolver.fit()
    (geoparser/.../sentencetransformer.py:630-667), aber ohne den
    _prepare_training_data()-Aufruf.
    """
    train_dataset = Dataset.from_dict(training_data)
    train_loss = ContrastiveLoss(resolver.transformer)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        save_strategy=save_strategy,
        logging_strategy="steps",
        logging_steps=max(1, len(training_data["sentence1"]) // (batch_size * 10)),
        eval_strategy="no",
        save_total_limit=2,
        load_best_model_at_end=False,
    )

    trainer = SentenceTransformerTrainer(
        model=resolver.transformer,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
    )

    trainer.train()
    resolver.transformer.save_pretrained(str(output_path))


# ── Evaluation ───────────────────────────────────────────────────────────────

def prepare_eval_data(
    resolver: Union[SentenceTransformerResolver, SpatialSentenceResolver],
    val_documents: List[Dict],
) -> List[Dict]:
    """Vorberechnet Contexts + Descriptions + Gold-IDs fuer alle Val-Toponyme.

    Diese Daten haengen nur vom Eval-Resolver und Fold-Split ab,
    nicht von den Modellgewichten. Kann fuer alle 5 Modelle wiederverwendet werden.
    """
    eval_items = []
    for doc in val_documents:
        text = doc["text"]
        for ref, referent in zip(doc["references"], doc["referents"]):
            start, end = ref[0], ref[1]
            toponym_text = text[start:end]

            candidates = resolver.gazetteer.search(toponym_text)
            if not candidates:
                eval_items.append({
                    "context": None,
                    "descriptions": [],
                    "candidate_ids": [],
                    "gold_id": referent[1],
                })
                continue

            context = resolver._extract_context(text, start, end)
            descriptions = [resolver._generate_description(c) for c in candidates]
            candidate_ids = [c.location_id_value for c in candidates]

            eval_items.append({
                "context": context,
                "descriptions": descriptions,
                "candidate_ids": candidate_ids,
                "gold_id": referent[1],
            })

    return eval_items


def evaluate_with_cached_data(
    resolver: Union[SentenceTransformerResolver, SpatialSentenceResolver],
    eval_items: List[Dict],
) -> Dict:
    """Evaluiert mit vorbereiteten Eval-Daten — nur Encoding + Ranking."""
    correct_at_1 = 0
    correct_at_3 = 0
    reciprocal_ranks = []
    total = 0
    no_candidates = 0

    for item in eval_items:
        total += 1

        if not item["descriptions"]:
            reciprocal_ranks.append(0.0)
            no_candidates += 1
            continue

        context_emb = resolver.transformer.encode(
            [item["context"]], convert_to_tensor=True, show_progress_bar=False,
        )[0]

        cand_embs = resolver.transformer.encode(
            item["descriptions"], convert_to_tensor=True, show_progress_bar=False,
        )

        similarities = torch.nn.functional.cosine_similarity(
            context_emb.unsqueeze(0), cand_embs, dim=1,
        )

        ranked_indices = similarities.argsort(descending=True).tolist()

        rank = None
        for pos, idx in enumerate(ranked_indices):
            if item["candidate_ids"][idx] == item["gold_id"]:
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
    n_folds: int = 5,
) -> Dict:
    """Phase 1: Grid Search ueber Hyperparameter mit K-Fold CV.

    Sucht auf M4 (spatial config1) nach den besten Hyperparametern.
    Folds aussen, HP-Combos innen: Training-Daten und Eval-Daten
    werden pro Fold einmal vorbereitet und fuer alle HP-Combos wiederverwendet.
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
    logger.info("  Referenz: M4_spatial_config1")
    logger.info("  Eval: E_spatial_config1")
    logger.info("=" * 60)

    hp_fold_results: Dict[str, List[Dict]] = {
        f"lr={lr}_ep={ep}_bs={bs}": []
        for lr, ep, bs in hp_combos
    }

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_docs = [documents[i] for i in train_idx]
        val_docs = [documents[i] for i in val_idx]

        texts, refs, referents = docs_to_training_format(train_docs)
        train_toponyms = sum(len(r) for r in refs)
        val_toponyms = sum(len(doc["references"]) for doc in val_docs)

        logger.info(
            "\n[Phase 1] Fold %d/%d: %d train (%d toponyms), %d val (%d toponyms)",
            fold_idx + 1, n_folds, len(train_docs), train_toponyms,
            len(val_docs), val_toponyms,
        )

        # Training-Daten einmal pro Fold vorbereiten (Ebene B)
        logger.info("  Preparing training data...")
        t_prep = time.time()
        prep_resolver = create_resolver(BASE_MODEL, "spatial", "config1")
        training_data = prep_resolver._prepare_training_data(texts, refs, referents)
        logger.info(
            "  Training data: %d Beispiele (%.1fs)",
            len(training_data["sentence1"]), time.time() - t_prep,
        )

        # Eval-Daten einmal pro Fold vorbereiten (Ebene C)
        logger.info("  Preparing eval data...")
        t_eval_prep = time.time()
        eval_items = prepare_eval_data(prep_resolver, val_docs)
        logger.info(
            "  Eval data: %d Toponyme (%.1fs)",
            len(eval_items), time.time() - t_eval_prep,
        )

        del prep_resolver
        free_resources()

        for hp_idx, (lr, epochs, bs) in enumerate(hp_combos):
            hp_key = f"lr={lr}_ep={epochs}_bs={bs}"
            model_dir = hp_dir / hp_key / f"fold{fold_idx}"
            model_dir.mkdir(parents=True, exist_ok=True)

            logger.info(
                "  [Fold %d] HP %d/%d: %s",
                fold_idx + 1, hp_idx + 1, len(hp_combos), hp_key,
            )

            t0 = time.time()
            resolver = create_resolver(BASE_MODEL, "spatial", "config1")
            run_training(
                resolver=resolver,
                training_data=training_data,
                output_path=model_dir,
                epochs=epochs,
                batch_size=bs,
                learning_rate=lr,
            )
            training_loss = extract_training_loss(model_dir)
            train_duration = time.time() - t0

            del resolver
            free_resources()

            # Evaluation mit gecachten Eval-Daten
            t1 = time.time()
            eval_resolver = create_resolver(str(model_dir), "spatial", "config1")
            metrics = evaluate_with_cached_data(eval_resolver, eval_items)
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
            hp_fold_results[hp_key].append(result)

            logger.info(
                "    Acc@1=%.3f, Acc@3=%.3f, MRR=%.3f (%.0fs train, %.0fs eval)",
                metrics["accuracy_at_1"], metrics["accuracy_at_3"],
                metrics["mrr"], train_duration, eval_duration,
            )

    # Aggregieren
    all_results = []
    for lr, epochs, bs in hp_combos:
        hp_key = f"lr={lr}_ep={epochs}_bs={bs}"
        fold_results = hp_fold_results[hp_key]
        agg = aggregate_fold_metrics(fold_results)

        logger.info(
            "[Phase 1] %s: Acc@1=%.3f+-%.3f, MRR=%.3f+-%.3f",
            hp_key,
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

    best = max(all_results, key=lambda r: (r["mean_accuracy_at_1"], r["mean_mrr"]))

    output = {
        "grid": HP_GRID,
        "reference_model": "M4_spatial_config1",
        "reference_eval": "E_spatial_config1",
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
    logger.info("  Acc@1=%.3f+-%.3f, MRR=%.3f+-%.3f",
                best["mean_accuracy_at_1"], best["std_accuracy_at_1"],
                best["mean_mrr"], best["std_mrr"])
    logger.info("  Gespeichert: %s", output_path)
    logger.info("=" * 60)

    return output


# ── Phase 2: Cross-Evaluation ────────────────────────────────────────────────

def run_experiments(
    documents: List[Dict],
    best_hps: Dict,
    n_folds: int = 5,
) -> List[Dict]:
    """Phase 2: K-Fold CV Training (M3-M5) + Cross-Evaluation (5x3=15).

    Pro Fold:
    1. Train M3 (default resolver), M4 (spatial c1), M5 (spatial c2)
    2. Prepare eval data fuer alle 3 Eval-Resolver (einmal pro Fold)
    3. Evaluate alle 5 Modelle x 3 Eval-Resolver = 15 Evaluationen
    """
    folds = create_document_folds(len(documents), n_folds=n_folds)

    lr = best_hps["learning_rate"]
    epochs = best_hps["epochs"]
    bs = best_hps["batch_size"]

    exp_dir = RESULTS_DIR / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    n_trainable = len(TRAINABLE_MODELS)
    n_evals = len(MODELS) * len(EVAL_RESOLVERS)

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Cross-Evaluation")
    logger.info("  %d Modelle x %d Eval-Resolver = %d Evaluationen pro Fold",
                len(MODELS), len(EVAL_RESOLVERS), n_evals)
    logger.info("  %d trainierbare Modelle x %d Folds = %d Trainings",
                n_trainable, n_folds, n_trainable * n_folds)
    logger.info("  HPs: lr=%s, epochs=%d, bs=%d", lr, epochs, bs)
    logger.info("=" * 60)

    # Ergebnisse: model_id -> eval_id -> [fold_results]
    all_results: Dict[str, Dict[str, List[Dict]]] = {
        model_id: {eval_id: [] for eval_id in EVAL_RESOLVERS}
        for model_id in MODELS
    }
    # Training-Infos: model_id -> [fold_training_info]
    training_infos: Dict[str, List[Dict]] = {
        model_id: [] for model_id in TRAINABLE_MODELS
    }

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_docs = [documents[i] for i in train_idx]
        val_docs = [documents[i] for i in val_idx]

        texts, refs, referents = docs_to_training_format(train_docs)
        train_toponyms = sum(len(r) for r in refs)
        val_toponyms = sum(len(doc["references"]) for doc in val_docs)

        logger.info(
            "\n[Phase 2] Fold %d/%d: %d train (%d toponyms), %d val (%d toponyms)",
            fold_idx + 1, n_folds, len(train_docs), train_toponyms,
            len(val_docs), val_toponyms,
        )

        # ── Step 1: Train M3, M4, M5 ────────────────────────────────────
        trained_model_dirs: Dict[str, Path] = {}

        for model_id, model_cfg in TRAINABLE_MODELS.items():
            model_dir = exp_dir / model_id / f"fold{fold_idx}"
            model_dir.mkdir(parents=True, exist_ok=True)

            resolver_type = model_cfg["train_resolver"]
            config_name = model_cfg.get("config")

            logger.info("  Training %s (resolver=%s)...", model_id, resolver_type)
            t0 = time.time()

            # Training-Daten vorbereiten
            resolver = create_resolver(model_cfg["base_model"], resolver_type, config_name)
            training_data = resolver._prepare_training_data(texts, refs, referents)

            logger.debug(
                "    %d training examples prepared", len(training_data["sentence1"]),
            )

            # Training
            run_training(
                resolver=resolver,
                training_data=training_data,
                output_path=model_dir,
                epochs=epochs,
                batch_size=bs,
                learning_rate=lr,
            )
            training_loss = extract_training_loss(model_dir)
            train_duration = time.time() - t0

            trained_model_dirs[model_id] = model_dir

            training_infos[model_id].append({
                "fold": fold_idx,
                "train_documents": len(train_docs),
                "train_toponyms": train_toponyms,
                "training_loss": training_loss,
                "train_duration_seconds": round(train_duration, 1),
            })

            del resolver
            free_resources()

            logger.info(
                "    Trained %s in %.0fs (%d examples)",
                model_id, train_duration, len(training_data["sentence1"]),
            )

        # ── Step 2: Prepare eval data fuer alle 3 Resolver (Ebene C) ────
        cached_eval_data: Dict[str, List[Dict]] = {}

        for eval_id, eval_cfg in EVAL_RESOLVERS.items():
            logger.info("  Preparing eval data for %s...", eval_id)
            t_prep = time.time()

            # Fuer Eval-Daten brauchen wir einen Resolver mit irgendeinem Modell
            # (nur fuer _generate_description, nicht fuer Encoding)
            eval_prep_resolver = create_resolver(
                BASE_MODEL, eval_cfg["type"], eval_cfg.get("config"),
            )
            cached_eval_data[eval_id] = prepare_eval_data(eval_prep_resolver, val_docs)

            del eval_prep_resolver
            free_resources()

            logger.info(
                "    %s: %d items (%.1fs)",
                eval_id, len(cached_eval_data[eval_id]), time.time() - t_prep,
            )

        # ── Step 3: Cross-Evaluate alle 5 Modelle x 3 Eval-Resolver ─────
        for model_id, model_cfg in MODELS.items():
            # Modell-Weights bestimmen
            if model_cfg["train"]:
                model_weights = str(trained_model_dirs[model_id])
            else:
                model_weights = model_cfg["base_model"]

            for eval_id, eval_cfg in EVAL_RESOLVERS.items():
                t_eval = time.time()

                # Resolver mit Modell-Weights + Eval-Resolver-Typ
                eval_resolver = create_resolver(
                    model_weights, eval_cfg["type"], eval_cfg.get("config"),
                )
                metrics = evaluate_with_cached_data(
                    eval_resolver, cached_eval_data[eval_id],
                )
                eval_duration = time.time() - t_eval

                del eval_resolver
                free_resources()

                result = {
                    "fold": fold_idx,
                    "val_documents": len(val_docs),
                    "val_toponyms": val_toponyms,
                    **metrics,
                    "eval_duration_seconds": round(eval_duration, 1),
                }
                all_results[model_id][eval_id].append(result)

                logger.info(
                    "    %s + %s: Acc@1=%.3f, MRR=%.3f (%.1fs)",
                    model_id, eval_id,
                    metrics["accuracy_at_1"], metrics["mrr"], eval_duration,
                )

        # Aufraumen: Checkpoints der trainierten Modelle
        for model_dir in trained_model_dirs.values():
            cleanup_model_dir(model_dir)

    # ── Ergebnisse speichern ─────────────────────────────────────────────
    all_experiment_results = []

    for model_id in MODELS:
        for eval_id in EVAL_RESOLVERS:
            fold_results = all_results[model_id][eval_id]
            agg = aggregate_fold_metrics(fold_results)

            experiment_result = {
                "model": model_id,
                "eval_resolver": eval_id,
                "model_config": MODELS[model_id],
                "eval_config": EVAL_RESOLVERS[eval_id],
                "hyperparameters": {
                    "learning_rate": lr,
                    "epochs": epochs,
                    "batch_size": bs,
                    "warmup_ratio": 0.1,
                } if MODELS[model_id]["train"] else None,
                "n_folds": n_folds,
                "seed": SEED,
                "folds": fold_results,
                "training_info": training_infos.get(model_id),
                "aggregate": agg,
            }

            # Pro Kombination speichern
            result_path = exp_dir / model_id / eval_id / "results.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(experiment_result, f, ensure_ascii=False, indent=2)

            all_experiment_results.append(experiment_result)

            logger.info(
                "[Phase 2] %s + %s: Acc@1=%.3f+-%.3f, MRR=%.3f+-%.3f",
                model_id, eval_id,
                agg["mean_accuracy_at_1"], agg["std_accuracy_at_1"],
                agg["mean_mrr"], agg["std_mrr"],
            )

    generate_summary_csv(all_experiment_results)

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2 ABGESCHLOSSEN")
    logger.info("  %d Evaluationen (%d Modelle x %d Resolver)",
                len(all_experiment_results), len(MODELS), len(EVAL_RESOLVERS))
    logger.info("  Ergebnisse: %s", exp_dir)
    logger.info("=" * 60)

    return all_experiment_results


def generate_summary_csv(experiment_results: List[Dict]):
    """Generiert results/summary.csv mit einer Zeile pro Model x Eval-Resolver."""
    csv_path = RESULTS_DIR / "summary.csv"
    fieldnames = [
        "model", "eval_resolver",
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
                "model": exp["model"],
                "eval_resolver": exp["eval_resolver"],
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
):
    """Phase 3: Finale Modelle (M3, M4, M5) auf Gesamtdaten trainieren."""
    lr = best_hps["learning_rate"]
    epochs = best_hps["epochs"]
    bs = best_hps["batch_size"]

    texts, refs, referents = docs_to_training_format(documents)

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: Finale Modelle")
    logger.info("  %d Modelle auf Gesamtdaten (%d Dokumente, %d Toponyme)",
                len(TRAINABLE_MODELS), len(documents),
                sum(len(doc["references"]) for doc in documents))
    logger.info("  HPs: lr=%s, epochs=%d, bs=%d", lr, epochs, bs)
    logger.info("=" * 60)

    for model_idx, (model_id, model_cfg) in enumerate(TRAINABLE_MODELS.items()):
        model_dir = MODELS_DIR / model_id

        resolver_type = model_cfg["train_resolver"]
        config_name = model_cfg.get("config")

        logger.info(
            "\n[Phase 3] Modell %d/%d: %s",
            model_idx + 1, len(TRAINABLE_MODELS), model_id,
        )

        model_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        resolver = create_resolver(model_cfg["base_model"], resolver_type, config_name)
        training_data = resolver._prepare_training_data(texts, refs, referents)
        run_training(
            resolver=resolver,
            training_data=training_data,
            output_path=model_dir,
            epochs=epochs,
            batch_size=bs,
            learning_rate=lr,
        )

        duration = time.time() - t0
        cleanup_model_dir(model_dir)

        del resolver
        free_resources()

        logger.info("  -> Gespeichert: %s (%.0fs)", model_dir, duration)

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3 ABGESCHLOSSEN")
    logger.info("  %d Modelle gespeichert in %s", len(TRAINABLE_MODELS), MODELS_DIR)
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
        help="Nur eine Phase ausfuehren (1=HP-Suche, 2=Cross-Evaluation, 3=Finale Modelle)",
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

    data, documents = load_preprocessed()

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

    if 1 in phases:
        hp_results = run_hp_search(documents, n_folds=args.n_folds)
        best_hps = hp_results["best"]

    if 2 in phases:
        if best_hps is None:
            best_hps = load_best_hps()
            logger.info("Gecachte HPs geladen: %s", best_hps)
        run_experiments(documents, best_hps, n_folds=args.n_folds)

    if 3 in phases:
        if best_hps is None:
            best_hps = load_best_hps()
            logger.info("Gecachte HPs geladen: %s", best_hps)
        train_final_models(documents, best_hps)

    logger.info("\nPipeline abgeschlossen.")


if __name__ == "__main__":
    main()
