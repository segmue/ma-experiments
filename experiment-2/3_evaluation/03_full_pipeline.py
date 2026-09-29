"""
Stage 03 — Full-Pipeline-Evaluation (Recognizer + Resolver).

Im Gegensatz zur Resolver-Eval (Stage 02, auf Gold-Mentions) laeuft hier der
spaCy-Recognizer ueber die Volltexte und erzeugt eigene Toponym-Spans (inkl.
False Positives). Damit werden echte Document-Precision≠Recall, Full-Pipeline-
Precision und Recognizer-Detection-Metriken messbar.

Ablauf:
  1. Recognizer pro Korpus-Sprache (aus filename-Suffix) einmal ueber alle Docs
     → predicted Spans (cache/recognizer_spans.pkl). Detection P/R/F1 vs Gold-Spans.
  2. Pro Haupt-System (config.MAIN_SYSTEMS): predicted Spans aufloesen (Top-1 loc_id),
     gegen Gold matchen (Span-Overlap). Korrekt = Span trifft Gold UND loc_id == Gold-loc_id.
     → Full-Pipeline-Precision (mikro) + Document-P/R/F1 (makro).

Output: results/full_pipeline.json

Verwendung:
    python 03_full_pipeline.py                 # alle Haupt-Systeme, ganzer Korpus
    python 03_full_pipeline.py --max-docs 50   # Smoke-Test
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import config as C
import eval_core as E


# ── Recognizer ───────────────────────────────────────────────────────────────
def run_recognizer(documents, cache_path: Path, force: bool = False) -> dict:
    """Predicted Spans pro Dokument (sprachspezifisches spaCy-Modell). Gecacht."""
    if cache_path.exists() and not force:
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if set(cached["spans"]) >= {d["filename"] for d in documents}:
            print(f"  Recognizer-Spans aus Cache: {cache_path}")
            return cached["spans"]

    from geoparser.modules.recognizers.spacy import SpacyRecognizer

    by_lang = defaultdict(list)
    for d in documents:
        by_lang[C.language_of(d["filename"])].append(d)

    spans: dict[str, list] = {}
    for lang, docs in by_lang.items():
        model = C.LANG_MODELS.get(lang, C.LANG_MODELS["mul"])
        print(f"  Sprache {lang}: {len(docs)} Docs, spaCy={model}")
        rec = SpacyRecognizer(model_name=model, entity_types=C.RECOGNIZER_ENTITY_TYPES)
        preds = rec.predict([d["text"] for d in docs])
        for d, p in zip(docs, preds):
            spans[d["filename"]] = list(p or [])
        del rec
    with open(cache_path, "wb") as f:
        pickle.dump({"spans": spans}, f)
    return spans


# ── Span-Matching ───────────────────────────────────────────────────────────
def _overlap(a, b) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def detection_metrics(documents, pred_by_doc) -> dict:
    """Recognizer-Detection: Span-Overlap-Matching gegen Gold (mikro P/R/F1)."""
    tp = fp = fn = 0
    for d in documents:
        gold = [(t["start"], t["end"]) for t in d["toponyms"]]
        preds = pred_by_doc.get(d["filename"], [])
        detected_gold = set()
        for ps in preds:
            hit = [gi for gi, g in enumerate(gold) if _overlap(ps, g) > 0]
            if hit:
                tp += 1
                detected_gold.update(hit)
            else:
                fp += 1
        fn += len(gold) - len(detected_gold)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def pipeline_metrics(documents, pred_by_doc, predid_by_span) -> dict:
    """Aufloesungs-Korrektheit + Full-Pipeline-Precision + Document-P/R/F1.

    predid_by_span: {(doc_id,start,end): predicted_loc_id}
    Korrekt = pred-Span ueberlappt ein Gold-Toponym UND pred-loc_id == dessen loc_id.
    """
    total_pred = correct = 0
    doc_P, doc_R = [], []
    for d in documents:
        gold = [(t["start"], t["end"], t["loc_id"]) for t in d["toponyms"]]
        preds = pred_by_doc.get(d["filename"], [])
        n_correct_doc = 0
        for ps in preds:
            total_pred += 1
            # bestes ueberlappendes Gold-Toponym
            best = max(gold, key=lambda g: _overlap(ps, g), default=None)
            if best is None or _overlap(ps, best) == 0:
                continue
            pred_id = predid_by_span.get((d["filename"], ps[0], ps[1]))
            if pred_id is not None and pred_id == best[2]:
                correct += 1
                n_correct_doc += 1
        n_pred, n_gold = len(preds), len(gold)
        doc_P.append(n_correct_doc / n_pred if n_pred else 0.0)
        doc_R.append(n_correct_doc / n_gold if n_gold else 0.0)
    macro_p = sum(doc_P) / len(doc_P) if doc_P else 0.0
    macro_r = sum(doc_R) / len(doc_R) if doc_R else 0.0
    f1 = 2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) else 0.0
    return {
        "full_pipeline_precision": correct / total_pred if total_pred else 0.0,
        "total_predicted": total_pred, "correct_resolutions": correct,
        "document_precision": macro_p, "document_recall": macro_r, "document_f1": f1,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Full-Pipeline-Evaluation")
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--force-recognizer", action="store_true", help="Recognizer-Cache neu bauen")
    args = ap.parse_args()

    C.ensure_dirs()
    documents = E.load_documents(args.max_docs)
    text_by_doc = E.text_index(documents)

    # 1) Recognizer
    print("1) Recognizer ueber alle Dokumente ...")
    t0 = time.time()
    pred_by_doc = run_recognizer(documents, C.CACHE_DIR / "recognizer_spans.pkl",
                                 force=args.force_recognizer)
    det = detection_metrics(documents, pred_by_doc)
    print(f"   Detection: P={det['precision']:.3f} R={det['recall']:.3f} F1={det['f1']:.3f} "
          f"(TP={det['tp']} FP={det['fp']} FN={det['fn']}, {time.time()-t0:.0f}s)")

    # Dokumente mit predicted Spans (gold_id=None) als Span-Getter fuer prepare_items.
    def pred_span_getter(doc):
        return [(s, e, None) for (s, e) in pred_by_doc.get(doc["filename"], [])]

    # 2) Pro benoetigtem Resolver: Items aus predicted Spans (modell-unabhaengig).
    #    Persistent gecacht (items_pred_{eid}.pkl), damit ein spaeterer Lauf
    #    (z.B. nach Modell-Nachlieferung) direkt bei Schritt 3 weitermacht.
    needed = {eid for _, eid in C.MAIN_SYSTEMS}
    pred_items = {}
    for eid in needed:
        cache_path = C.CACHE_DIR / f"items_pred_{eid}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("n_docs") == len(documents):
                print(f"2) predicted Items fuer {eid} aus Cache: {cache_path}")
                pred_items[eid] = cached["items"]
                continue
        print(f"2) prepare_items (predicted Spans) fuer {eid} ...")
        r = E.build_eval_resolver(eid)
        # Gleicher Description-Cache wie Stage 01: Beschreibungen haengen nur vom
        # Feature ab, nicht davon, ob der Span gold oder predicted ist.
        pred_items[eid] = E.prepare_items(r, documents, span_getter=pred_span_getter,
                                          cache_key=eid)
        with open(cache_path, "wb") as f:
            pickle.dump({"n_docs": len(documents), "items": pred_items[eid]}, f)
        del r

    # 3) Pro Haupt-System aufloesen + Metriken.
    results = {"detection": det, "systems": {}}
    for mid, eid in C.MAIN_SYSTEMS:
        mcfg = C.MODELS[mid]
        if mcfg["local"] and not Path(mcfg["weights"]).exists():
            print(f"[skip] {mid}: Modell nicht gefunden")
            continue
        print(f"3) {mid} + {eid} aufloesen ...")
        t = time.time()
        try:
            ctx_resolver = E.build_resolver(mcfg["weights"], "default")
        except RuntimeError as err:  # NaN-Guard: System ueberspringen statt Lauf killen
            print(f"[skip] {mid}: {err}")
            continue
        per_item = E.encode_and_rank(ctx_resolver, pred_items[eid], text_by_doc)
        predid_by_span = {(p["doc_id"], p["start"], p["end"]): p["pred_id"] for p in per_item}
        m = pipeline_metrics(documents, pred_by_doc, predid_by_span)
        results["systems"][f"{mid}+{eid}"] = m
        print(f"   FullPipelineP={m['full_pipeline_precision']:.3f} "
              f"DocP={m['document_precision']:.3f} DocR={m['document_recall']:.3f} "
              f"DocF1={m['document_f1']:.3f} ({time.time()-t:.0f}s)")
        del ctx_resolver

    out = C.RESULTS_DIR / "full_pipeline.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGespeichert: {out}")


if __name__ == "__main__":
    main()
