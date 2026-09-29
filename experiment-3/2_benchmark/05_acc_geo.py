"""
Stage 05: Inhaltliche Paritaet -- Acc@1/Acc@3/MRR mit Zwillings-Beschreibungen.

Laeuft auf dem gemeinsamen N_DOCS_BENCH-Doc-Sample (NIE der volle Korpus) und
wiederverwendet die Experiment-2-Maschinerie 1:1:
  1. prepare_items mit dem GeoSentenceResolver (Gold-Spans, Doc-Sample,
     persistenter Description-Cache E_c1_geo/E_c2_geo) -> Items wie Stage 01.
  2. Die Exp2-Items (items_E_c1/E_c2.pkl) werden auf DIESELBEN Docs gefiltert
     -> H3-Referenzzellen E_c1_h3/E_c2_h3 auf identischer Teilmenge (fairer
     Vergleich; die Full-Corpus-Aggregate aus Exp2 waeren Aepfel/Birnen).
  3. Stage-02-Logik (extract_contexts/encode_strings/rank_items/Metriken)
     fuer ALLE 5 Modelle x 4 Zellen; Checkpoint pro Modell (resume-faehig).
  4. Delta-Tabelle geo minus h3 pro Modell/Config aus den paarweisen Zellen.

Damit wird die Thesis-Aussage quantifizierbar: gleiche Resolutionsqualitaet
(Delta-Acc@1), aber X-facher Kostenunterschied (Stages 02-04).

Output: results/acc_geo.json, results/tables/acc_geo_delta.csv
"""

from __future__ import annotations

import csv
import gc
import pickle
import time

import torch

import bench_config as B

B.add_paths()

import config as EC  # noqa: E402  (3_evaluation/config.py)
import eval_core as E  # noqa: E402
from resolver_geo import GeoSentenceResolver  # noqa: E402

GEO_RESOLVERS = {"E_c1_geo": "config1", "E_c2_geo": "config2"}
H3_ITEMS = {"E_c1_h3": "items_E_c1.pkl", "E_c2_h3": "items_E_c2.pkl"}


def load_h3_items(eid: str, doc_ids: set[str]) -> list:
    """Exp2-Items (voller Korpus, nur lesen) aufs Doc-Sample filtern."""
    with open(B.EVAL3_DIR / "cache" / H3_ITEMS[eid], "rb") as f:
        items = pickle.load(f)["items"]
    return [it for it in items if it["doc_id"] in doc_ids]


def build_items(eid: str, cfg: str, documents) -> list:
    """Items mit dem Geo-Resolver vorbereiten (gecacht in cache/items_{eid}.pkl)."""
    cache_path = B.CACHE_DIR / f"items_{eid}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        if payload.get("n_docs") == len(documents):
            print(f"  [cache] {cache_path.name}")
            return payload["items"]

    resolver = GeoSentenceResolver(
        model_name=EC.MODELS["M1_dguzh"]["weights"],  # Modell fuer Items irrelevant
        gazetteer_name=B.GAZETTEER,
        geo_duckdb_path=B.geo_db(cfg),
        config_path=B.config_yaml(cfg),
        b1_matrix_path=B.b1_matrix(cfg),
    )
    t0 = time.perf_counter()
    items = E.prepare_items(resolver, documents, cache_key=eid)
    print(f"  Items {eid}: {len(items)} in {time.perf_counter()-t0:.0f}s")
    with open(cache_path, "wb") as f:
        pickle.dump({"resolver": eid, "n_docs": len(documents), "items": items}, f)
    del resolver
    gc.collect()
    return items


def main() -> None:
    B.tee_log("05_acc_geo")
    all_docs = E.load_documents()
    doc_ids = B.sample_doc_ids(d["filename"] for d in all_docs)
    documents = [d for d in all_docs if d["filename"] in doc_ids]
    print(f"[05] Doc-Sample: {len(documents)} von {len(all_docs)} Dokumenten "
          f"(seed {B.SEED})")
    text_by_doc = E.text_index(documents)

    items_by_eid = {}
    for eid, cfg in GEO_RESOLVERS.items():
        print(f"[05] Items fuer {eid} ({cfg}) ...")
        items_by_eid[eid] = build_items(eid, cfg, documents)
    for eid in H3_ITEMS:
        items_by_eid[eid] = load_h3_items(eid, doc_ids)
        print(f"[05] Items fuer {eid}: {len(items_by_eid[eid])} (Exp2, gefiltert)")

    results: dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_documents": len(documents),
        "doc_sample_seed": B.SEED,
        "cells": [],
    }

    for mid, mcfg in EC.MODELS.items():
        ckpt = B.CACHE_DIR / f"acc_geo_{mid}.json"
        cached = B.load_json(ckpt)
        if cached is not None and cached.get("n_documents") == len(documents):
            print(f"\n=== Modell {mid} === [skip] Checkpoint {ckpt.name}")
            results["cells"] += cached["cells"]
            continue

        print(f"\n=== Modell {mid} ===")
        try:
            ctx_resolver = E.build_resolver(mcfg["weights"], "default")
        except RuntimeError as err:
            print(f"[skip] {mid}: {err}")
            continue

        any_items = next(iter(items_by_eid.values()))
        ctx_by_key = E.extract_contexts(ctx_resolver, any_items, text_by_doc)
        ctx_emb = E.encode_strings(
            ctx_resolver.transformer, list(ctx_by_key.values()), EC.BATCH_SIZE
        )

        cells = []
        for eid, items in items_by_eid.items():
            desc = [d for it in items for d in it["descriptions"]]
            emb = {**ctx_emb, **E.encode_strings(
                ctx_resolver.transformer, desc, EC.BATCH_SIZE)}
            per_item = E.rank_items(items, ctx_by_key, emb)
            loc = E.location_metrics(per_item)
            doc = E.document_metrics(per_item)
            cells.append({
                "model": mid, "eval_resolver": eid,
                "location": loc, "document": doc,
            })
            print(f"  {eid}: Acc@1={loc['accuracy_at_1']:.3f} "
                  f"Acc@3={loc['accuracy_at_3']:.3f} MRR={loc['mrr']:.3f}")

        B.save_json_atomic(ckpt, {"n_documents": len(documents), "cells": cells})
        results["cells"] += cells

        del ctx_resolver, ctx_emb
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    B.save_json_atomic(B.RESULTS_DIR / "acc_geo.json", results)

    # Delta-Tabelle: geo minus h3, beide auf demselben Doc-Sample gerechnet
    by_cell = {(c["model"], c["eval_resolver"]): c for c in results["cells"]}
    rows = []
    for mid in EC.MODELS:
        for base in ("E_c1", "E_c2"):
            g = by_cell.get((mid, f"{base}_geo"))
            h = by_cell.get((mid, f"{base}_h3"))
            if not g or not h:
                continue
            rows.append({
                "model": mid, "geo": f"{base}_geo", "h3": f"{base}_h3",
                "acc1_geo": g["location"]["accuracy_at_1"],
                "acc1_h3": h["location"]["accuracy_at_1"],
                "delta_acc1": round(g["location"]["accuracy_at_1"]
                                    - h["location"]["accuracy_at_1"], 4),
                "mrr_geo": g["location"]["mrr"],
                "mrr_h3": h["location"]["mrr"],
                "delta_mrr": round(g["location"]["mrr"]
                                   - h["location"]["mrr"], 4),
            })
    if rows:
        with open(B.TABLES_DIR / "acc_geo_delta.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    else:
        print("[05] WARNUNG: keine geo/h3-Zellenpaare -> kein acc_geo_delta.csv")
    print(f"[05] Fertig -> results/acc_geo.json + tables/acc_geo_delta.csv")


if __name__ == "__main__":
    main()
