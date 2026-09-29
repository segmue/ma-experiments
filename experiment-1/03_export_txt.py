"""
03_export_txt.py
----------------
Converts annotation batch Parquet files to plain text.
Articles are deduplicated by content_clean and grouped into chunks of at most
MAX_CHARS characters, with at least one article per chunk. A single article
exceeding MAX_CHARS gets its own file.

Each article is rendered as:

  === <head> ===

  <content_clean>

Articles within a chunk are separated by a --- divider line.

Output:
  ma-experiments/data/annotation_txt/{dataset_name}/batch_NNNN_KK.txt
  (KK = chunk index within the batch, zero-padded)
"""

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

BATCHES_DIR = Path(__file__).resolve().parent.parent / "data" / "annotation_batches"
OUT_DIR     = Path(__file__).resolve().parent.parent / "data" / "annotation_txt"
SEPARATOR   = "\n\n---\n\n"
MAX_CHARS   = 1200


def article_to_text(head: str, content: str) -> str:
    return f"=== {head} ===\n\n{content}"


def export_dataset(dataset_name: str) -> None:
    in_dir  = BATCHES_DIR / dataset_name
    out_dir = OUT_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_files = sorted(in_dir.glob("batch_*.parquet"))
    if not batch_files:
        print(f"  No batches found in {in_dir}, skipping.", file=sys.stderr)
        return

    print(f"\n=== {dataset_name} ===", flush=True)

    for batch_file in tqdm(batch_files, desc="  Exporting"):
        batch_stem = batch_file.stem  # e.g. "batch_0001"
        df = pd.read_parquet(batch_file)
        df["content_clean"] = df["content_clean"].fillna("")
        df["head"] = df["head"].fillna("")

        # Deduplicate by content_clean (keep first occurrence)
        before = len(df)
        df = df.drop_duplicates(subset=["content_clean"])
        after = len(df)
        if before != after:
            print(f"    {batch_stem}: removed {before - after} duplicate(s)", flush=True)

        articles = [article_to_text(h, c) for h, c in zip(df["head"], df["content_clean"])]

        chunk_texts: list[str] = []
        chunk_chars = 0
        chunk_idx = 1

        def flush_chunk(texts: list[str], idx: int) -> None:
            out_path = out_dir / f"{batch_stem}_{idx:02d}.txt"
            out_path.write_text(SEPARATOR.join(texts), encoding="utf-8")

        for article in articles:
            cc = len(article)
            if chunk_texts and chunk_chars + len(SEPARATOR) + cc > MAX_CHARS:
                flush_chunk(chunk_texts, chunk_idx)
                chunk_idx += 1
                chunk_texts = []
                chunk_chars = 0
            chunk_texts.append(article)
            chunk_chars += (len(SEPARATOR) if chunk_chars else 0) + cc

        if chunk_texts:
            flush_chunk(chunk_texts, chunk_idx)

        print(f"    {batch_stem}: {after} articles → {chunk_idx} file(s)", flush=True)

    print(f"  Done → {out_dir}", flush=True)


def main():
    dataset_names = [d.name for d in sorted(BATCHES_DIR.iterdir()) if d.is_dir()]

    if not dataset_names:
        print(f"No annotation batches found in {BATCHES_DIR}.", file=sys.stderr)
        print("Run 02_rank_and_batch.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Datasets to export: {dataset_names}", flush=True)

    for name in dataset_names:
        export_dataset(name)

    print("\nAll batches exported.", flush=True)


if __name__ == "__main__":
    main()
