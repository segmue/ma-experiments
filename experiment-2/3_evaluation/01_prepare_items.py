"""
Stage 01 — Modell-unabhaengige Eval-Items pro Eval-Resolver cachen.

Fuer jeden Eval-Resolver (E_default / E_c1 / E_c2, per --resolver auch die
Config-Ablation E_c3–E_c5 und die E_d1-Arme) wird einmal die teure,
modell-unabhaengige Arbeit erledigt: Gazetteer-Suche + Kandidaten-Beschreibungen
(bei den Spatial-Resolvern via H3-CandidateSentenceGenerator) fuer jedes Gold-Toponym.
Ergebnis pro Resolver → cache/items_{E}.pkl. Wird in Stage 02 von allen 5 Modellen
wiederverwendet (Encoding/Ranking ist dort der modell-abhaengige Teil).

Verwendung:
    python 01_prepare_items.py                 # alle 3 Resolver, ganzer Korpus
    python 01_prepare_items.py --max-docs 50   # Smoke-Test
    python 01_prepare_items.py --resolver E_c1 # nur ein Resolver
    python 01_prepare_items.py --resolver E_c3 E_c4 E_c5     # Config-Ablation
    python 01_prepare_items.py --resolver E_d1 E_d1_c6 ...   # Adjazenzmass D1
"""

from __future__ import annotations

import argparse
import pickle
import time

import config as C
import eval_core as E


def main():
    ap = argparse.ArgumentParser(description="Eval-Items cachen (modell-unabhaengig)")
    ap.add_argument("--max-docs", type=int, default=None, help="nur die ersten N Dokumente")
    ap.add_argument("--resolver", nargs="+", choices=list(C.EVAL_RESOLVERS), default=None,
                    help="nur diese Eval-Resolver (default: E_default, E_c1, E_c2)")
    args = ap.parse_args()

    C.ensure_dirs()
    resolvers = args.resolver or list(C.MATRIX_RESOLVERS)
    # Fuer die Spatial-Resolver brauchen wir die DuckDBs + Matrizen; Modelle hier noch nicht.
    missing = C.check_prerequisites(require_spatial=True, require_local_models=False,
                                    resolvers=resolvers)
    if missing:
        print("FEHLENDE PREREQUISITES:")
        for m in missing:
            print("  -", m)
        raise SystemExit(1)

    documents = E.load_documents(args.max_docs)
    n_topo = sum(len(d["toponyms"]) for d in documents)
    print(f"Dokumente: {len(documents)} | Gold-Toponyme: {n_topo}")

    for eid in resolvers:
        cfg = C.EVAL_RESOLVERS[eid]
        print(f"\n=== {eid} (kind={cfg['kind']}, config={cfg['config']}) ===")
        t0 = time.time()
        # Basismodell genuegt — prepare_items nutzt nur Gazetteer + _generate_description.
        resolver = E.build_eval_resolver(eid)
        items = E.prepare_items(resolver, documents, cache_key=eid)
        out = C.CACHE_DIR / f"items_{eid}.pkl"
        with open(out, "wb") as f:
            pickle.dump({"resolver": eid, "n_docs": len(documents), "items": items}, f)
        no_cand = sum(1 for it in items if not it["candidate_ids"])
        print(f"  {len(items)} Items ({no_cand} ohne Kandidat) -> {out} ({time.time()-t0:.0f}s)")
        del resolver


if __name__ == "__main__":
    main()
