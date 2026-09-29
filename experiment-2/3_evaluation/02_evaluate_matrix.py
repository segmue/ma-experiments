"""
Stage 02 — 5×3-Evaluationsmatrix (Location- + Document-Unit).

Fuer jedes Modell M1..M5 wird der Encoder geladen, die Kontexte einmal extrahiert
+ encodiert (identisch ueber die 3 Resolver), und je Eval-Resolver werden die in
Stage 01 gecachten Kandidaten-Beschreibungen encodiert und gerankt.

Metriken pro Zelle: Acc@1, Acc@3, MRR (Location) + makro pro-Dokument-Accuracy (Document).
Output: results/matrix/{M}_{E}.json und results/summary.csv.

Verwendung:
    python 02_evaluate_matrix.py                       # alle Modelle × Resolver
    python 02_evaluate_matrix.py --models M3_default_finetuned M4_spatial_config1
    python 02_evaluate_matrix.py --max-docs 50
    # Config-Ablation / E_d1-Arme in eigene Summary-Dateien (summary.csv bleibt die 5×3-Matrix):
    python 02_evaluate_matrix.py --models M3_default_finetuned M4_spatial_config1 M5_spatial_config2 \
        --resolvers E_c3 E_c4 E_c5 --summary config_ablation/summary_e8.csv
    python 02_evaluate_matrix.py --resolvers E_d1 --summary "$PWD/logs/summary_E_d1_main.csv"
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import json
import pickle
import time
from pathlib import Path

import torch

import config as C
import eval_core as E


def load_items(eid: str):
    path = C.CACHE_DIR / f"items_{eid}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)["items"]


def error_counts(per_item):
    """Fehlerklassen + Ambiguity-Split.

    "Nicht ambig" = genau 1 Gazetteer-Kandidat (n_candidates == 1): der Resolver
    kann dort nichts falsch ranken. acc1_ambiguous_only misst Acc@1 nur auf den
    Items mit >1 Kandidaten — der eigentliche Disambiguierungs-Hebel.
    """
    correct_unambig = sum(1 for p in per_item if p["rank"] == 1 and p["n_candidates"] == 1)
    correct_ambig = sum(1 for p in per_item if p["rank"] == 1 and p["n_candidates"] > 1)
    ambig_total = sum(1 for p in per_item if p["n_candidates"] > 1)
    return {
        "correct": correct_unambig + correct_ambig,
        "correct_unambiguous": correct_unambig,
        "correct_ambiguous": correct_ambig,
        "unambiguous_total": sum(1 for p in per_item if p["n_candidates"] == 1),
        "ambiguous_total": ambig_total,
        "acc1_ambiguous_only": correct_ambig / ambig_total if ambig_total else 0.0,
        "wrong_rank": sum(1 for p in per_item if p["rank"] is not None and p["rank"] > 1),
        "gold_not_in_candidates": sum(1 for p in per_item if p["n_candidates"] > 0 and p["rank"] is None),
        "no_candidate": sum(1 for p in per_item if p["n_candidates"] == 0),
    }


def main():
    ap = argparse.ArgumentParser(description="5×3 Evaluationsmatrix")
    ap.add_argument("--models", nargs="*", default=list(C.MODELS))
    ap.add_argument("--resolvers", nargs="*", default=list(C.MATRIX_RESOLVERS))
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--summary", default="summary.csv",
                    help="Summary-Datei relativ zu results/ oder absolut (Default: summary.csv)")
    args = ap.parse_args()

    C.ensure_dirs()
    documents = E.load_documents(args.max_docs)
    text_by_doc = E.text_index(documents)

    # Gecachte Items pro Resolver laden.
    items_by_resolver = {eid: load_items(eid) for eid in args.resolvers}
    missing = [eid for eid, it in items_by_resolver.items() if it is None]
    if missing:
        print(f"FEHLT: Cache fuer {missing}. Erst 01_prepare_items.py ausfuehren.")
        raise SystemExit(1)

    summary_rows = []
    for mid in args.models:
        mcfg = C.MODELS[mid]
        if mcfg["local"] and not Path(mcfg["weights"]).exists():
            print(f"[skip] {mid}: Modell nicht gefunden ({mcfg['weights']})")
            continue

        print(f"\n=== Modell {mid} ===")
        t_model = time.time()
        # Default-Resolver liefert Tokenizer + Encoder + spaCy fuer Kontext-Extraktion.
        try:
            ctx_resolver = E.build_resolver(mcfg["weights"], "default")
        except RuntimeError as err:  # NaN-Guard: Modell ueberspringen statt Lauf killen
            print(f"[skip] {mid}: {err}")
            continue

        # Kontexte einmal pro Modell (Spans sind ueber alle Resolver identisch).
        any_items = next(iter(items_by_resolver.values()))
        ctx_by_key = E.extract_contexts(ctx_resolver, any_items, text_by_doc)
        ctx_emb = E.encode_strings(ctx_resolver.transformer, list(ctx_by_key.values()), C.BATCH_SIZE)

        for eid in args.resolvers:
            items = items_by_resolver[eid]
            t0 = time.time()
            desc = [d for it in items for d in it["descriptions"]]
            emb = {**ctx_emb, **E.encode_strings(ctx_resolver.transformer, desc, C.BATCH_SIZE)}
            per_item = E.rank_items(items, ctx_by_key, emb)

            loc = E.location_metrics(per_item)
            doc = E.document_metrics(per_item)
            errs = error_counts(per_item)
            cell = {
                "model": mid, "eval_resolver": eid,
                "n_documents_eval": len(documents),
                "location": loc, "document": doc, "errors": errs,
                "eval_seconds": round(time.time() - t0, 1),
            }
            out = C.MATRIX_DIR / f"{mid}_{eid}.json"
            out.write_text(json.dumps(cell, ensure_ascii=False, indent=2), encoding="utf-8")
            # per-Item-Dump fuer post-hoc-Analysen (08 Stratifizierung, 09 No-Candidate)
            with gzip.open(C.PER_ITEM_DIR / f"{mid}_{eid}.pkl.gz", "wb") as f:
                pickle.dump(per_item, f)
            summary_rows.append(cell)
            print(f"  {eid}: Acc@1={loc['accuracy_at_1']:.3f} Acc@3={loc['accuracy_at_3']:.3f} "
                  f"MRR={loc['mrr']:.3f} DocAcc={doc['document_accuracy']:.3f} "
                  f"Acc@1(ambig)={errs['acc1_ambiguous_only']:.3f} "
                  f"(no-cand={loc['no_candidates']}, {cell['eval_seconds']}s)")

        del ctx_resolver, ctx_emb
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  -> {mid} fertig ({time.time()-t_model:.0f}s)")

    # summary.csv
    csv_path = C.RESULTS_DIR / args.summary
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "eval_resolver", "total", "no_candidates",
                    "acc1", "acc3", "mrr", "acc1_with_cand", "acc3_with_cand",
                    "doc_accuracy", "n_documents",
                    "correct_unambiguous", "correct_ambiguous",
                    "unambiguous_total", "ambiguous_total", "acc1_ambiguous_only"])
        for c in summary_rows:
            loc, doc, err = c["location"], c["document"], c["errors"]
            w.writerow([c["model"], c["eval_resolver"], loc["total"], loc["no_candidates"],
                        f"{loc['accuracy_at_1']:.4f}", f"{loc['accuracy_at_3']:.4f}",
                        f"{loc['mrr']:.4f}", f"{loc['accuracy_at_1_with_candidates']:.4f}",
                        f"{loc['accuracy_at_3_with_candidates']:.4f}",
                        f"{doc['document_accuracy']:.4f}", doc["n_documents"],
                        err["correct_unambiguous"], err["correct_ambiguous"],
                        err["unambiguous_total"], err["ambiguous_total"],
                        f"{err['acc1_ambiguous_only']:.4f}"])
    print(f"\nSummary: {csv_path} ({len(summary_rows)} Zellen)")


if __name__ == "__main__":
    main()
