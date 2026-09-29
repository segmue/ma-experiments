"""
02_rank_and_batch.py
--------------------
Loads the preprocessed Parquet batches, runs spaCy NER (de_core_news_lg) on each
article, computes the toponym ratio (toponyms / tokens), sorts by ratio DESC, and
saves 100-article annotation batches per dataset.

Toponym labels (identical to Geoparser's SpacyRecognizer defaults): FAC, GPE, LOC

Output:
  ma-experiments/data/annotation_batches/{dataset_name}/batch_NNNN.parquet
  ma-experiments/data/annotation_batches/{dataset_name}/summary.csv
"""

import sys
from pathlib import Path

import pandas as pd
import spacy
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PREPROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "swissdox_preprocessed"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "annotation_batches"

SPACY_MODEL = "de_core_news_lg"
TOPONYM_LABELS = {"FAC", "GPE", "LOC"}  # identical to SpacyRecognizer defaults
NER_BATCH_SIZE = 64                      # texts per spaCy pipe batch
ANNOTATION_BATCH_SIZE = 100             # articles per output batch file

# ---------------------------------------------------------------------------
# spaCy setup
# ---------------------------------------------------------------------------

def load_nlp() -> spacy.language.Language:
    print(f"Loading spaCy model '{SPACY_MODEL}' …", flush=True)
    nlp = spacy.load(SPACY_MODEL, exclude=["tagger", "morphologizer", "parser", "senter"])
    print("  Model loaded.", flush=True)
    return nlp


# ---------------------------------------------------------------------------
# Ratio computation
# ---------------------------------------------------------------------------

def compute_ratios(nlp: spacy.language.Language, texts: list[str]) -> list[dict]:
    """Run spaCy NER on texts, return list of {toponym_count, token_count, toponym_ratio}."""
    results = []
    pipe = nlp.pipe(texts, batch_size=NER_BATCH_SIZE)
    for doc in tqdm(pipe, total=len(texts), desc="      articles", leave=False):
        token_count = len(doc)
        toponym_count = sum(1 for ent in doc.ents if ent.label_ in TOPONYM_LABELS)
        ratio = toponym_count / token_count if token_count > 0 else 0.0
        results.append({
            "toponym_count": toponym_count,
            "token_count": token_count,
            "toponym_ratio": ratio,
        })
    return results


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------

def process_dataset(nlp: spacy.language.Language, dataset_name: str) -> None:
    in_dir = PREPROCESSED_DIR / dataset_name
    out_dir = OUT_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_files = sorted(in_dir.glob("batch_*.parquet"))
    if not batch_files:
        print(f"  No preprocessed batches found in {in_dir}, skipping.", file=sys.stderr)
        return

    print(f"\n=== {dataset_name} ===", flush=True)
    print(f"  Found {len(batch_files)} preprocessed batch(es)", flush=True)

    # Accumulate all rows with ratio data
    all_chunks = []

    for batch_file in tqdm(batch_files, desc="  Running NER"):
        print(f"    {batch_file.name} …", flush=True)
        df = pd.read_parquet(batch_file)
        texts = df["content_clean"].fillna("").tolist()
        ratios = compute_ratios(nlp, texts)

        ratio_df = pd.DataFrame(ratios, index=df.index)
        df = pd.concat([df, ratio_df], axis=1)
        all_chunks.append(df)

    full_df = pd.concat(all_chunks, ignore_index=True)
    print(f"  Total articles: {len(full_df):,}", flush=True)
    print(f"  Toponym ratio — mean: {full_df['toponym_ratio'].mean():.4f}, "
          f"max: {full_df['toponym_ratio'].max():.4f}", flush=True)

    # Sort by toponym_ratio descending
    full_df = full_df.sort_values("toponym_ratio", ascending=False).reset_index(drop=True)

    # Save annotation batches
    n_batches = (len(full_df) + ANNOTATION_BATCH_SIZE - 1) // ANNOTATION_BATCH_SIZE
    print(f"  Saving {n_batches} annotation batch(es) of up to {ANNOTATION_BATCH_SIZE} articles …",
          flush=True)

    full_df["batch_nr"] = full_df.index // ANNOTATION_BATCH_SIZE

    for i in tqdm(range(n_batches), desc="  Writing batches"):
        chunk = full_df[full_df["batch_nr"] == i].drop(columns=["batch_nr"])
        out_path = out_dir / f"batch_{i:04d}.parquet"
        chunk.to_parquet(out_path, index=False)

    # Summary CSV
    summary = full_df[["id", "toponym_ratio", "toponym_count", "token_count", "batch_nr"]].copy()
    summary.to_csv(out_dir / "summary.csv", index=False)

    print(f"  Done → {out_dir}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dataset_names = [d.name for d in sorted(PREPROCESSED_DIR.iterdir()) if d.is_dir()]

    if not dataset_names:
        print(f"No preprocessed datasets found in {PREPROCESSED_DIR}.", file=sys.stderr)
        print("Run 01_clean_swissdox.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Datasets to process: {dataset_names}", flush=True)

    nlp = load_nlp()

    for name in dataset_names:
        process_dataset(nlp, name)

    print("\nAll datasets ranked and batched.", flush=True)


if __name__ == "__main__":
    main()
