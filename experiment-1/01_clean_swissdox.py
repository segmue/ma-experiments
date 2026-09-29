"""
01_clean_swissdox.py
--------------------
Reads the 3 Swissdox XZ-compressed TSV datasets directly (no prior decompression),
strips XML tags from the content column, and saves the result as Parquet batches.

Input:  ma-experiments/data/swissdox/*.xz
Output: ma-experiments/data/swissdox_preprocessed/{dataset_name}/batch_NNNN.parquet
"""

import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DATASETS = [
    DATA_DIR / "swissdox" / "topografische_features.xz",
    DATA_DIR / "swissdox" / "Tourismus_aktivitaeten.xz",
    DATA_DIR / "swissdox" / "gewaesserverbauungen_lokalNews.xz",
]

OUT_DIR = DATA_DIR / "swissdox_preprocessed"

BATCH_SIZE = 5_000  # rows per Parquet file

TSV_COLUMNS = [
    "id", "pubtime", "medium_code", "medium_name", "rubric", "regional",
    "doctype", "doctype_description", "language", "char_count", "dateline",
    "head", "subhead", "article_link", "content_id", "content",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_XML_TAG_RE = re.compile(r"<[^>]+>")


def strip_xml(text: str) -> str:
    """Remove all XML/HTML tags and normalise whitespace."""
    if not isinstance(text, str):
        return ""
    text = _XML_TAG_RE.sub(" ", text)
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def preprocess_dataset(xz_path: Path) -> None:
    dataset_name = xz_path.stem  # filename without extension
    out_dir = OUT_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {dataset_name} ===")
    print(f"Reading {xz_path} …")

    df = pd.read_csv(
        xz_path,
        sep="\t",
        compression="xz",
        names=TSV_COLUMNS,
        header=0,         # first row is header
        dtype=str,        # keep everything as strings for now
        on_bad_lines="warn",
    )

    print(f"  Loaded {len(df):,} rows")

    # Strip XML from content (and head/subhead for safety)
    for col in ("content", "head", "subhead"):
        if col in df.columns:
            df[col] = df[col].apply(strip_xml)

    # Rename content → content_clean to make it explicit
    df.rename(columns={"content": "content_clean"}, inplace=True)

    # Save in batches
    n_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  Saving {n_batches} batch(es) of up to {BATCH_SIZE:,} rows …")

    for i in tqdm(range(n_batches), desc="  Writing batches"):
        chunk = df.iloc[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        out_path = out_dir / f"batch_{i:04d}.parquet"
        chunk.to_parquet(out_path, index=False)

    print(f"  Done → {out_dir}")


def main():
    for xz_path in DATASETS:
        if not xz_path.exists():
            print(f"WARNING: {xz_path} not found, skipping.", file=sys.stderr)
            continue
        preprocess_dataset(xz_path)

    print("\nAll datasets preprocessed.")


if __name__ == "__main__":
    main()
