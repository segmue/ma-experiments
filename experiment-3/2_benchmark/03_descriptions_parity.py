"""
Stage 03: Beschreibungs-Generierung -- Zeit + Output-Paritaet.

Workload: die realen, eindeutigen Gold-Kandidaten-Features aus Experiment 2
(cache/items_E_c1.pkl bzw. items_E_c2.pkl, UUIDs -> Features der jeweiligen DB),
begrenzt auf das gemeinsame N_DOCS_BENCH-Doc-Sample (bench_config.sample_doc_ids).

Pro Config und System (h3, geo -- geo_dwithin nur in Stage 02, s. main()):
  - Batch-Generierung ALLER Workload-Features (BatchSentenceGenerator), Zeit.
  - Einzelpfad-Stichprobe (SINGLE_N Features, frischer Generator), Zeit/Feature.

Paritaet (nur h3 vs geo-buffered, pro Config):
  - identische-Saetze-Rate (string-gleich)
  - Jaccard der Kontext-Namensmengen (static + dynamic) pro Feature, Mittel/Median
  - Jaccard der dynamischen Kategorien-Mengen

Output: results/descriptions.json, results/parity.json
"""

from __future__ import annotations

import json
import pickle
import statistics
import time

import duckdb

import bench_config as C

C.add_paths()

from engine_geo import GeometricEngine  # noqa: E402
from geo_strtree import StrtreeGeometricEngine  # noqa: E402
from h3_multi_resolution_engine.engine import H3Engine  # noqa: E402
from geoparser_h3_resolver.pipeline.build_config import BuildConfig  # noqa: E402
from geoparser_h3_resolver.sentence_generator import (  # noqa: E402
    CandidateSentenceGenerator,
    FeatureInput,
)
from geoparser_h3_resolver.sentence_generator.batch_generator import (  # noqa: E402
    BatchSentenceGenerator,
)

SINGLE_N = 200
ITEMS_BY_CFG = {"config1": "items_E_c1.pkl", "config2": "items_E_c2.pkl"}


def workload_features(cfg: str) -> list[FeatureInput]:
    """Gold-Kandidaten-UUIDs des 100-Doc-Samples aus Experiment 2 -> FeatureInputs.

    Bewusst NICHT der volle Korpus: Items werden vor dem UUID-Ziehen auf das
    gemeinsame N_DOCS_BENCH-Sample gefiltert (gleiches Sample wie Stage 05).
    """
    with open(C.EVAL3_DIR / "cache" / ITEMS_BY_CFG[cfg], "rb") as f:
        items = pickle.load(f)["items"]
    doc_ids = C.sample_doc_ids(it["doc_id"] for it in items)
    items = [it for it in items if it["doc_id"] in doc_ids]
    uuids = sorted({u for it in items for u in it["candidate_ids"]})

    conn = duckdb.connect(str(C.h3_db(cfg)), read_only=True)
    feats = []
    for i in range(0, len(uuids), 1000):
        chunk = uuids[i:i + 1000]
        ph = ", ".join("?" for _ in chunk)
        rows = conn.execute(f"""
            SELECT UUID, feature_id, NAME, OBJEKTART FROM features
            WHERE UUID IN ({ph})
            QUALIFY ROW_NUMBER() OVER (PARTITION BY UUID ORDER BY feature_id) = 1
        """, chunk).fetchall()
        feats += [FeatureInput(feature_id=r[1], name=r[2], objektart=r[3])
                  for r in rows]
    conn.close()
    feats.sort(key=lambda f: f.feature_id)
    return feats


def sentence_cfg(cfg: str):
    matrix = C.b1_matrix(cfg)
    return BuildConfig.from_yaml(C.config_yaml(cfg)).to_sentence_generator_config(matrix)


def make_engine(system: str, cfg: str):
    if system == "h3":
        return H3Engine(C.h3_db(cfg))
    if system == "geo_strtree":
        return StrtreeGeometricEngine(C.geo_db(cfg))
    mode = "dwithin" if system == "geo_dwithin" else "buffered"
    return GeometricEngine(C.geo_db(cfg), mode=mode)


def run_system(system: str, cfg: str, feats: list[FeatureInput]) -> tuple[dict, dict]:
    """Liefert (timing-dict, {feature_id: GeneratedSentence}). Checkpointed."""
    ckpt = C.CACHE_DIR / f"desc_{cfg}_{system}.pkl"
    if ckpt.exists():
        with open(ckpt, "rb") as f:
            payload = pickle.load(f)
        if payload.get("n_features") == len(feats):
            print(f"    [skip] Checkpoint {ckpt.name}")
            return payload["timing"], payload["sentences"]

    scfg = sentence_cfg(cfg)
    timing: dict = {}

    # Batch: alle Features, frischer Generator
    with make_engine(system, cfg) as engine:
        gen = CandidateSentenceGenerator(engine, scfg)
        print(f"    Batch-Generierung {len(feats)} Features ...", flush=True)
        t0 = time.perf_counter()
        BatchSentenceGenerator(gen).precompute(feats)
        dt = time.perf_counter() - t0
        timing["batch_total_s"] = round(dt, 2)
        timing["batch_per_feature_ms"] = round(1000 * dt / len(feats), 3)
        sentences = dict(gen._cache)

    # Einzelpfad-Stichprobe: frischer Generator (keine Cache-Hits).
    # Zeitbudget: bei langsamen Systemen (geo: >10 s/Feature moeglich) nach
    # MIN_SINGLE_SAMPLES + Budget abbrechen; single_n dokumentiert das echte n.
    sample = feats[:: max(1, len(feats) // SINGLE_N)][:SINGLE_N]
    with make_engine(system, cfg) as engine:
        gen = CandidateSentenceGenerator(engine, scfg)
        per = []
        for f in sample[: C.N_WARMUP]:
            gen.generate(f)
        gen._cache.clear()
        t_block = time.perf_counter()
        for f in sample:
            if (len(per) >= C.MIN_SINGLE_SAMPLES
                    and time.perf_counter() - t_block > C.POINT_TIME_BUDGET_S):
                timing["single_truncated"] = True
                break
            t0 = time.perf_counter()
            gen.generate(f)
            per.append(time.perf_counter() - t0)
        timing["single_n"] = len(per)
        timing["single_median_ms"] = round(1000 * statistics.median(per), 2)
        timing["single_mean_ms"] = round(1000 * statistics.fmean(per), 2)

    tmp = ckpt.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"n_features": len(feats), "timing": timing,
                     "sentences": sentences}, f)
    tmp.replace(ckpt)
    return timing, sentences


def parity(h3_s: dict, geo_s: dict) -> dict:
    ids = sorted(set(h3_s) & set(geo_s))
    identical = 0
    jacc_names, jacc_cats = [], []
    for fid in ids:
        a, b = h3_s[fid], geo_s[fid]
        if a.sentence == b.sentence:
            identical += 1

        def names(s):
            out = set()
            for v in s.static_context.values():
                out.update(v)
            for v in s.context_by_category.values():
                out.update(v)
            return out

        na, nb = names(a), names(b)
        jacc_names.append(len(na & nb) / len(na | nb) if (na | nb) else 1.0)
        ca = set(a.context_by_category)
        cb = set(b.context_by_category)
        jacc_cats.append(len(ca & cb) / len(ca | cb) if (ca | cb) else 1.0)

    return {
        "n_features": len(ids),
        "identical_sentence_rate": round(identical / len(ids), 4),
        "jaccard_context_names_mean": round(statistics.fmean(jacc_names), 4),
        "jaccard_context_names_median": round(statistics.median(jacc_names), 4),
        "jaccard_dynamic_categories_mean": round(statistics.fmean(jacc_cats), 4),
    }


def main() -> None:
    C.tee_log("03_descriptions")
    desc_res: dict = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "configs": {}}
    par_res: dict = {"timestamp": desc_res["timestamp"], "configs": {}}

    for cfg in C.CONFIGS:
        print(f"[03] {cfg}: Workload laden...")
        feats = workload_features(cfg)
        print(f"  {len(feats)} eindeutige Gold-Kandidaten-Features")
        desc_res["configs"][cfg] = {"n_features": len(feats), "systems": {}}

        sent_by_system = {}
        # geo_dwithin bewusst NICHT dabei: laut engine_geo-Doku keine
        # Beschreibungs-/Paritaets-Variante, und sein Batch-Join ist unbudgetiert
        # pathologisch langsam (>70 min fuer 50 Features im Smoke-Test).
        # dwithin-Latenzen sind durch die budgetierten Mikro-Messungen (02) belegt.
        for system in ("h3", "geo", "geo_strtree"):
            print(f"  System {system} ...")
            timing, sentences = run_system(system, cfg, feats)
            desc_res["configs"][cfg]["systems"][system] = timing
            sent_by_system[system] = sentences
            print(f"    {timing}")

        par = parity(sent_by_system["h3"], sent_by_system["geo"])
        par_res["configs"][cfg] = par
        print(f"  Paritaet h3 vs geo: {par}")

        if "geo_strtree" in sent_by_system:
            pairs = {
                "h3_vs_geo": par,
                "h3_vs_geo_strtree": parity(
                    sent_by_system["h3"], sent_by_system["geo_strtree"]),
                # MUSS 1.0 sein: identische Puffergeometrien, identisches Praedikat,
                # identische Sortierung -- nur die Indexstruktur ist anders.
                "geo_vs_geo_strtree": parity(
                    sent_by_system["geo"], sent_by_system["geo_strtree"]),
            }
            par_res.setdefault("pairs", {})[cfg] = pairs
            for k, v in pairs.items():
                print(f"  Paritaet {k}: {v}")

        # Saetze fuer qualitative Inspektion ablegen
        with open(C.CACHE_DIR / f"sentences_{cfg}{C.VARIANT}.pkl", "wb") as f:
            pickle.dump({
                s: {fid: g.sentence for fid, g in d.items()}
                for s, d in sent_by_system.items()
            }, f)

    with open(C.RESULTS_DIR / "descriptions.json", "w", encoding="utf-8") as f:
        json.dump(desc_res, f, indent=2)
    with open(C.RESULTS_DIR / "parity.json", "w", encoding="utf-8") as f:
        json.dump(par_res, f, indent=2)
    print(f"[03] Fertig -> {C.RESULTS_DIR}/descriptions.json, parity.json")


if __name__ == "__main__":
    main()
