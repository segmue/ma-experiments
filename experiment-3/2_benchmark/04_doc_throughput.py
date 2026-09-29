"""
Stage 04: Dokument-Durchsatz end-to-end (die zentrale Zeitmessung).

2 disjunkte Sets von je N_DOCS_THROUGHPUT Dokumenten (deterministisch aus dem
Text+Berg-Eval-Datensatz gesampelt, Gold-Spans als references) laufen durch
das komplette predict() von fuenf Systemen:

    default      : SentenceTransformerResolver (non-spatial Admin-Hierarchie)
    h3_config1/2 : SpatialSentenceResolver (H3, inkl. Auto-Batching-Overrides)
    geo_config1/2: GeoSentenceResolver (geometrischer Zwilling, buffered)

Ein festes Modell (THROUGHPUT_MODEL) fuer alle Systeme -- der Encoder kuerzt
sich im Faktor. Pro Lauf eine FRISCHE Resolver-Instanz (kalte Prozess-Caches);
Modell-Ladezeit ist NICHT in der Messung (nur predict()).

Output: results/throughput.json -- s/Set, s/Dokument, Faktor vs default.
"""

from __future__ import annotations

import gc
import json
import random
import time

import torch

import bench_config as B

B.add_paths()

import config as EC  # noqa: E402  (3_evaluation/config.py)
import eval_core as E  # noqa: E402
from resolver_geo import GeoSentenceResolver  # noqa: E402


def make_system(name: str, weights: str):
    if name == "default":
        return E.build_resolver(weights, "default")
    if name.startswith("h3_"):
        cfg = name.removeprefix("h3_")
        return E.build_resolver(weights, "spatial", cfg)
    if name.startswith("geostr_"):
        cfg = name.removeprefix("geostr_")
        return GeoSentenceResolver(
            model_name=weights,
            gazetteer_name=B.GAZETTEER,
            geo_duckdb_path=B.geo_db(cfg),
            config_path=B.config_yaml(cfg),
            b1_matrix_path=B.b1_matrix(cfg),
            backend="strtree",
        )
    if name.startswith("geo_"):
        cfg = name.removeprefix("geo_")
        return GeoSentenceResolver(
            model_name=weights,
            gazetteer_name=B.GAZETTEER,
            geo_duckdb_path=B.geo_db(cfg),
            config_path=B.config_yaml(cfg),
            b1_matrix_path=B.b1_matrix(cfg),
        )
    raise ValueError(name)


def main() -> None:
    B.tee_log("04_throughput")
    weights = EC.MODELS[B.THROUGHPUT_MODEL]["weights"]
    docs = E.load_documents()
    rng = random.Random(B.SEED)
    idx = rng.sample(range(len(docs)), 2 * B.N_DOCS_THROUGHPUT)
    sets = {
        "set_a": [docs[i] for i in idx[: B.N_DOCS_THROUGHPUT]],
        "set_b": [docs[i] for i in idx[B.N_DOCS_THROUGHPUT:]],
    }

    ckpt = B.CACHE_DIR / "throughput_partial.json"
    partial = B.load_json(ckpt) or {}
    if partial.get("n_docs_per_set") != B.N_DOCS_THROUGHPUT:
        partial = {"n_docs_per_set": B.N_DOCS_THROUGHPUT, "systems": {}}

    results: dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": B.THROUGHPUT_MODEL,
        "n_docs_per_set": B.N_DOCS_THROUGHPUT,
        "systems": partial["systems"],
    }

    systems = ["default", "h3_config1", "h3_config2", "geo_config1", "geo_config2",
               "geostr_config1", "geostr_config2"]
    for name in systems:
        results["systems"].setdefault(name, {})
        for set_name, set_docs in sets.items():
            if set_name in results["systems"][name]:
                print(f"[04] {name}/{set_name}: [skip] Checkpoint")
                continue
            texts = [d["text"] for d in set_docs]
            refs = [[(t["start"], t["end"]) for t in d["toponyms"]] for d in set_docs]
            n_refs = sum(len(r) for r in refs)

            resolver = make_system(name, weights)  # Modell-Load ausserhalb der Messung
            t0 = time.perf_counter()
            resolver.predict(texts, refs)
            dt = time.perf_counter() - t0

            results["systems"][name][set_name] = {
                "seconds": round(dt, 1),
                "n_refs": n_refs,
                "s_per_doc": round(dt / len(set_docs), 3),
                "ms_per_ref": round(1000 * dt / n_refs, 1),
            }
            print(f"[04] {name}/{set_name}: {dt:7.1f}s "
                  f"({dt/len(set_docs):.2f} s/Doc, {n_refs} Refs)")
            partial["systems"] = results["systems"]
            B.save_json_atomic(ckpt, partial)

            # Engine explizit schliessen, BEVOR der Resolver faellt: der
            # StrtreeGeometricEngine haelt 442'001 Shapely-Geometrien (~2.2 GB)
            # im Prozess. close() gibt sie nachweislich vollstaendig frei
            # (gemessen 2213 -> 127 MB), das blosse del/gc nicht zuverlaessig.
            # Reines Teardown nach der gemessenen Sektion -- die Zeitmessung
            # oben ist davon unberuehrt.
            eng = getattr(resolver, "_engine", None)
            if eng is not None and hasattr(eng, "close"):
                try:
                    eng.close()
                except Exception:
                    pass
            del resolver
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Faktoren vs default
    for name in systems:
        for set_name in sets:
            base = results["systems"]["default"][set_name]["seconds"]
            cur = results["systems"][name][set_name]["seconds"]
            results["systems"][name][set_name]["factor_vs_default"] = round(cur / base, 2)

    out = B.RESULTS_DIR / "throughput.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[04] Fertig -> {out}")


if __name__ == "__main__":
    main()
