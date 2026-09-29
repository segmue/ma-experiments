"""
Stage 04 — Config-Ablationen (Konzept Abschnitt 5), fixes Modell (Default M5, --model).

Zur Inferenzzeit wird die SentenceGeneratorConfig des Spatial-Resolvers variiert
(ohne Retraining), das Encoder-Modell bleibt fix (M4). Varianten:
  - baseline_c1            (config1 unveraendert = M4/E_c1)
  - no_dynamic             max_slots=0  (nur statische Slots)
  - no_static              static_slots=[]  (kein Admin-Kontext)
  - assoc={0.001,0.01,0.1} Assoziations-Schwellwert
  - max_slots={10,5,3}     Anzahl dynamischer Slots
  - uniform_b1             B1-Matrix ohne Gewichtung (alle Werte 1.0) — misst den
                           Effekt der raeumlichen Assoziations-GEWICHTE isoliert
                           (Betreuer-Feedback 12.6.)

Pro Variante: Beschreibungen neu generieren (H3, modell-unabhaengig) → mit M4
encodieren/ranken. Output: results/ablations.json (Acc@1/MRR + Delta zur Baseline);
bereits vorhandene Varianten im JSON bleiben erhalten (Merge statt Ueberschreiben).

Verwendung:
    python 04_ablations.py                        # alle Varianten, ganzer Korpus
    python 04_ablations.py --variants uniform_b1  # nur einzelne Varianten rechnen
    python 04_ablations.py --max-docs 200         # Stichprobe
    python 04_ablations.py --model M4_spatial_config1 --out ablations_M4.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from geoparser_h3_resolver.pipeline.build_config import BuildConfig

import config as C
import eval_core as E


def uniform_b1_matrix() -> Path:
    """Erzeugt eine ungewichtete B1-Matrix (alle Werte 1.0, gleiche Kategorien
    wie config1). Effekt: alle Kategorien gleich assoziiert; get_associated_categories
    liefert Ties in deterministischer Spaltenreihenfolge, max_categories cappt."""
    import pandas as pd

    src = C.matrix_path(C.ABLATION_BASE_CONFIG)
    dst = C.CACHE_DIR / "uniform_b1_matrix.csv"
    df = pd.read_csv(src, sep=";", index_col=0)
    df.loc[:, :] = 1.0
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, sep=";")
    return dst


def build_variants():
    base_build = BuildConfig.from_yaml(C.config_yaml(C.ABLATION_BASE_CONFIG))
    base_sg = base_build.to_sentence_generator_config(C.matrix_path(C.ABLATION_BASE_CONFIG))
    variants = {
        "baseline_c1": base_sg,
        "no_dynamic": replace(base_sg, max_slots=0),
        "no_static": replace(base_sg, static_slots=[]),
        "uniform_b1": replace(base_sg, matrix_path=uniform_b1_matrix()),
    }
    for t in C.ASSOC_THRESHOLDS:
        variants[f"assoc_{t}"] = replace(base_sg, assoc_threshold=t)
    for s in C.MAX_SLOTS_SWEEP:
        variants[f"max_slots_{s}"] = replace(base_sg, max_slots=s)
    return variants


def main():
    ap = argparse.ArgumentParser(description="Config-Ablationen auf M4")
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--variants", nargs="*", default=None,
                    help="nur diese Varianten rechnen (Default: alle); "
                         "bestehende Eintraege in ablations.json bleiben erhalten")
    ap.add_argument("--model", default=C.ABLATION_MODEL, choices=list(C.MODELS),
                    help=f"Encoder-Modell (Default {C.ABLATION_MODEL})")
    ap.add_argument("--out", default="ablations.json",
                    help="Ausgabedatei relativ zu results/ (Default: ablations.json)")
    args = ap.parse_args()

    C.ensure_dirs()
    mcfg = C.MODELS[args.model]
    if mcfg["local"] and not Path(mcfg["weights"]).exists():
        print(f"FEHLT: Ablationsmodell {args.model} ({mcfg['weights']})")
        raise SystemExit(1)

    documents = E.load_documents(args.max_docs)
    text_by_doc = E.text_index(documents)
    variants = build_variants()
    if args.variants is not None:
        unknown = [v for v in args.variants if v not in variants]
        if unknown:
            print(f"Unbekannte Varianten: {unknown} (verfuegbar: {list(variants)})")
            raise SystemExit(1)
        variants = {v: sg for v, sg in variants.items() if v in args.variants}
    print(f"Ablationsmodell: {args.model} | {len(variants)} Varianten | "
          f"{len(documents)} Docs")

    # Encoder (M4) + Kontexte einmal (Spans/Tokenizer ueber Varianten identisch).
    ctx_resolver = E.build_resolver(mcfg["weights"], "default")

    results = {}
    ctx_emb = None
    ctx_by_key = None
    for vid, sg in variants.items():
        print(f"\n=== {vid} ===")
        t0 = time.time()
        # Beschreibungen mit dieser Config generieren (Basismodell, H3 = modell-unabhaengig).
        gen_resolver = E.build_resolver(C.BASE_MODEL, "spatial", C.ABLATION_BASE_CONFIG,
                                        sentence_config=sg)
        # baseline_c1 == config1 unveraendert -> teilt den E_c1-Cache aus Stage 01/03.
        cache_key = "E_c1" if vid == "baseline_c1" else f"abl_{vid}"
        items = E.prepare_items(gen_resolver, documents, cache_key=cache_key)
        del gen_resolver

        if ctx_by_key is None:  # einmalig fuer alle Varianten
            ctx_by_key = E.extract_contexts(ctx_resolver, items, text_by_doc)
            ctx_emb = E.encode_strings(ctx_resolver.transformer, list(ctx_by_key.values()), C.BATCH_SIZE)

        desc = [d for it in items for d in it["descriptions"]]
        emb = {**ctx_emb, **E.encode_strings(ctx_resolver.transformer, desc, C.BATCH_SIZE)}
        per_item = E.rank_items(items, ctx_by_key, emb)
        loc = E.location_metrics(per_item)
        results[vid] = loc
        print(f"  Acc@1={loc['accuracy_at_1']:.3f} MRR={loc['mrr']:.3f} "
              f"(no-cand={loc['no_candidates']}, {time.time()-t0:.0f}s)")

    # Merge mit bestehenden Resultaten (gleiche Korpusgroesse vorausgesetzt),
    # damit einzelne Varianten nachgerechnet werden koennen (--variants).
    # Bei anderer Korpusgroesse (Smoke-Test) wird in eine separate Datei
    # geschrieben, damit das Voll-Korpus-JSON nicht ueberschrieben wird.
    path = C.RESULTS_DIR / args.out
    merged = dict(results)
    if path.exists():
        prev = json.loads(path.read_text(encoding="utf-8"))
        if prev.get("n_documents") == len(documents):
            for vid, loc in prev.get("variants", {}).items():
                merged.setdefault(vid, {k: v for k, v in loc.items()
                                        if not k.startswith("delta_")})
        else:
            path = C.RESULTS_DIR / f"{Path(args.out).stem}_maxdocs{len(documents)}.json"
            print(f"Hinweis: Korpusgroesse weicht vom bestehenden {args.out} ab "
                  f"— schreibe nach {path.name} (kein Merge, kein Ueberschreiben).")

    # Deltas zur Baseline (aus diesem Lauf oder aus dem Merge)
    base = merged.get("baseline_c1", {})
    out = {"model": args.model, "base_config": C.ABLATION_BASE_CONFIG,
           "n_documents": len(documents), "variants": {}}
    for vid, loc in merged.items():
        out["variants"][vid] = {
            **loc,
            "delta_acc1": loc["accuracy_at_1"] - base.get("accuracy_at_1", 0.0),
            "delta_mrr": loc["mrr"] - base.get("mrr", 0.0),
        }
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGespeichert: {path} ({len(out['variants'])} Varianten)")


if __name__ == "__main__":
    main()
