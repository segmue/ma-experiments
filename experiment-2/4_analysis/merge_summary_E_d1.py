"""Stellt results/summary_E_d1.csv aus den Zell-JSONs in results/matrix/ zusammen.

Die E_d1-Zellen entstehen in 3_evaluation/02_evaluate_matrix.py in zwei Aufrufen
(5 Hauptzellen M1..M5 x E_d1, Achsen nur M4/M5). Dieses Skript schreibt alle 17
Zeilen in eine Datei, im selben Spaltenformat wie results/summary.csv; es rechnet
nichts neu, sondern liest nur die aggregierten Zell-JSONs.

Aufruf: python merge_summary_E_d1.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
MATRIX = RESULTS / "matrix"
OUT = RESULTS / "summary_E_d1.csv"

MAIN = [(m, "E_d1") for m in ("M1_dguzh", "M2_distiluse_base", "M3_default_finetuned",
                               "M4_spatial_config1", "M5_spatial_config2")]
AXES = [(m, a) for m in ("M5_spatial_config2", "M4_spatial_config1")
        for a in ("E_d1_s01", "E_d1_s1", "E_d1_c6", "E_d1_c20", "E_b1_c6", "E_b1_c20")]

HEADER = ["model", "eval_resolver", "total", "no_candidates",
          "acc1", "acc3", "mrr", "acc1_with_cand", "acc3_with_cand",
          "doc_accuracy", "n_documents",
          "correct_unambiguous", "correct_ambiguous",
          "unambiguous_total", "ambiguous_total", "acc1_ambiguous_only"]


def row(cell: dict) -> list:
    """Wortgleich zur Zeilenformatierung in 02_evaluate_matrix.py."""
    loc, doc, err = cell["location"], cell["document"], cell["errors"]
    return [cell["model"], cell["eval_resolver"], loc["total"], loc["no_candidates"],
            f"{loc['accuracy_at_1']:.4f}", f"{loc['accuracy_at_3']:.4f}",
            f"{loc['mrr']:.4f}", f"{loc['accuracy_at_1_with_candidates']:.4f}",
            f"{loc['accuracy_at_3_with_candidates']:.4f}",
            f"{doc['document_accuracy']:.4f}", doc["n_documents"],
            err["correct_unambiguous"], err["correct_ambiguous"],
            err["unambiguous_total"], err["ambiguous_total"],
            f"{err['acc1_ambiguous_only']:.4f}"]


def main() -> None:
    rows, fehlt = [], []
    for model, arm in MAIN + AXES:
        p = MATRIX / f"{model}_{arm}.json"
        if not p.exists():
            fehlt.append(p.name)
            continue
        rows.append(row(json.loads(p.read_text(encoding="utf-8"))))
    if fehlt:
        raise SystemExit(f"Zell-JSONs fehlen: {fehlt}")
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"{OUT}: {len(rows)} Zeilen")


if __name__ == "__main__":
    main()
