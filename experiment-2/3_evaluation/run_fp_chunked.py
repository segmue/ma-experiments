"""Full-Pipeline fuer EIN System (Default M5/E_c1), in Dokument-Chunks (speicherschonend).

Warum: 03_full_pipeline.py encodiert alle ~209k eindeutigen Strings eines Systems
in einem Rutsch und wurde deshalb zweimal vom Speicher-Watchdog gekillt (auch mit
nur einem System). Ranking ist pro Item unabhaengig und pipeline_metrics rechnet
erst am Schluss ueber alle Dokumente — die Zerlegung in Chunks ist daher exakt,
nicht approximativ: identische predid_by_span-Map, identische Metrik.

Wiederverwendet unveraendert: eval_core.encode_and_rank, 03_full_pipeline.
detection_metrics und 03_full_pipeline.pipeline_metrics. Voraussetzung ist der
Recognizer-Cache aus 03_full_pipeline.py (cache/recognizer_spans.pkl). Fehlen die
predicted Items des Resolvers (cache/items_pred_<E>.pkl, z.B. fuer E_d1), werden
sie wie in 03_full_pipeline.py erzeugt.

Verwendung:
    python run_fp_chunked.py                                   # M5/E_c1 -> full_pipeline_nur_M5Ec1.json
    python run_fp_chunked.py --resolver E_d1 --out full_pipeline_E_d1.json
"""
import argparse, gc, importlib.util, json, pickle, sys, time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--model", default="M5_spatial_config2")
ap.add_argument("--resolver", default="E_c1")
ap.add_argument("--out", default=None, help="Ausgabedatei relativ zu results/")
ap.add_argument("--chunk-docs", type=int, default=400)
A = ap.parse_args()

import torch
import config as C
import eval_core as E

MODEL, RESOLVER = A.model, A.resolver
CHUNK_DOCS = A.chunk_docs

spec = importlib.util.spec_from_file_location("fp03", HERE / "03_full_pipeline.py")
fp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp)

t0 = time.time()
documents = E.load_documents(None)
text_by_doc = E.text_index(documents)
print(f"{len(documents)} Dokumente", flush=True)

# 1) Recognizer-Spans direkt aus dem Cache (kein spaCy-Lauf, kein Neubau).
cache_path = C.CACHE_DIR / "recognizer_spans.pkl"
with open(cache_path, "rb") as f:
    pred_by_doc = pickle.load(f)["spans"]
assert set(pred_by_doc) >= {d["filename"] for d in documents}, "Recognizer-Cache unvollstaendig"
print(f"  Recognizer-Spans aus Cache: {cache_path}", flush=True)
det = fp.detection_metrics(documents, pred_by_doc)
print(f"   Detection: P={det['precision']:.3f} R={det['recall']:.3f} F1={det['f1']:.3f} "
      f"(TP={det['tp']} FP={det['fp']} FN={det['fn']})", flush=True)

# 2) predicted Items aus dem Cache, nach Dokument gruppiert.
items_path = C.CACHE_DIR / f"items_pred_{RESOLVER}.pkl"
if not items_path.exists():
    print(f"  {items_path.name} fehlt -> prepare_items (predicted Spans) fuer {RESOLVER}", flush=True)
    r = E.build_eval_resolver(RESOLVER)
    pred_items = E.prepare_items(
        r, documents, cache_key=RESOLVER,
        span_getter=lambda d: [(s, e, None) for (s, e) in pred_by_doc.get(d["filename"], [])])
    with open(items_path, "wb") as f:
        pickle.dump({"n_docs": len(documents), "items": pred_items}, f)
    del r, pred_items
with open(items_path, "rb") as f:
    cached = pickle.load(f)
assert cached["n_docs"] == len(documents), f"{cached['n_docs']} != {len(documents)}"
items_by_doc = defaultdict(list)
for it in cached["items"]:
    items_by_doc[it["doc_id"]].append(it)
n_items = sum(len(v) for v in items_by_doc.values())
del cached
gc.collect()
print(f"  predicted Items aus Cache: {items_path} ({n_items} Spans)", flush=True)

# 3) Chunkweise encodieren + ranken.
print(f"3) {MODEL} + {RESOLVER} aufloesen, Chunks von {CHUNK_DOCS} Dokumenten ...", flush=True)
resolver = E.build_resolver(C.MODELS[MODEL]["weights"], "default")
filenames = [d["filename"] for d in documents]
predid_by_span: dict[tuple, str] = {}
for i in range(0, len(filenames), CHUNK_DOCS):
    part = filenames[i:i + CHUNK_DOCS]
    chunk_items = [it for fn in part for it in items_by_doc[fn]]
    per_item = E.encode_and_rank(resolver, chunk_items, text_by_doc)
    for p in per_item:
        predid_by_span[(p["doc_id"], p["start"], p["end"])] = p["pred_id"]
    del per_item, chunk_items
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"   Docs {i + len(part)}/{len(filenames)} | {len(predid_by_span)} Spans "
          f"| {time.time() - t0:.0f}s", flush=True)

m = fp.pipeline_metrics(documents, pred_by_doc, predid_by_span)
print(f"   FullPipelineP={m['full_pipeline_precision']:.3f} "
      f"DocP={m['document_precision']:.3f} DocR={m['document_recall']:.3f} "
      f"DocF1={m['document_f1']:.3f} ({time.time() - t0:.0f}s)", flush=True)

out = C.RESULTS_DIR / (A.out or f"full_pipeline_nur_{MODEL.split('_')[0]}{RESOLVER.replace('_', '')}.json")
out.write_text(json.dumps({"detection": det, "systems": {f"{MODEL}+{RESOLVER}": m}},
                          ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nGespeichert: {out}", flush=True)
