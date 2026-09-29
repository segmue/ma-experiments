"""
End-to-End-Validierung der V1/V3-Overrides im SpatialSentenceResolver.

Simuliert eine End-Anwendung: geoparser-`predict()` ueber einige Text+Berg-
Dokumente, einmal mit den neuen Overrides (Auto-Batching `_embed_candidates`,
Such-Memoization `_gather_candidates`) und einmal mit den Original-Methoden des
Parents (per types.MethodType zurueckgepatcht = Verhalten vor dem Umbau).

Gefordert: identische Resolutionsergebnisse. Zusaetzlich wird die Wall-Clock
beider Laeufe gemessen (Achtung: die Baseline profitiert bereits vom
V2-Equi-Join-Fix in der Engine; der Vergleich isoliert also nur V1+V3).
"""

from __future__ import annotations

import time
import types

import config as C
import winfix_sqlite_pool

winfix_sqlite_pool.apply()

from geoparser.modules.resolvers.sentencetransformer import (  # noqa: E402
    SentenceTransformerResolver,
)
from geoparser_h3_resolver import SpatialSentenceResolver  # noqa: E402
import eval_core as E  # noqa: E402

N_DOCS = 3
MODEL = C.DGUZH_MODEL  # kleines, schnelles Modell; fuer den Vergleich egal


def make_resolver(patched_back: bool) -> SpatialSentenceResolver:
    r = SpatialSentenceResolver(
        model_name=MODEL,
        gazetteer_name=C.GAZETTEER,
        config_path=C.config_yaml("config1"),
        duckdb_path=C.duckdb_path("config1"),
    )
    if patched_back:
        # Original-Parent-Verhalten (vor V1/V3) wiederherstellen
        r._gather_candidates = types.MethodType(
            SentenceTransformerResolver._gather_candidates, r
        )
        r._embed_candidates = types.MethodType(
            SentenceTransformerResolver._embed_candidates, r
        )
    return r


def main() -> int:
    docs = E.load_documents(max_docs=N_DOCS)
    texts = [d["text"] for d in docs]
    references = [[(t["start"], t["end"]) for t in d["toponyms"]] for d in docs]
    n_refs = sum(len(r) for r in references)
    print(f"{len(docs)} Dokumente, {n_refs} Toponym-Referenzen")

    runs = {}
    for label, patched_back in (("baseline_alt", True), ("mit_overrides", False)):
        resolver = make_resolver(patched_back)
        t0 = time.perf_counter()
        runs[label] = resolver.predict(texts, references)
        dt = time.perf_counter() - t0
        print(f"{label:14s}: {dt:7.1f} s")
        del resolver

    mismatches = 0
    for doc_idx, (a, b) in enumerate(zip(runs["baseline_alt"], runs["mit_overrides"])):
        for ref_idx, (ra, rb) in enumerate(zip(a, b)):
            if ra != rb:
                mismatches += 1
                print(f"MISMATCH doc={doc_idx} ref={ref_idx}: alt={ra} neu={rb}")

    print(f"\nVerglichen: {n_refs} Referenzen, Mismatches: {mismatches}")
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
