"""
Ende-zu-Ende-Lauf fuer M5/E_d1 — der Treiber.

Im Repo ist der regulaere Weg 3_evaluation/run_fp_chunked.py --resolver E_d1
(Beschreibungen ueber config.EVAL_RESOLVERS["E_d1"], Mass D1 aus dem Plugin).
Dieser Treiber bleibt als Nachweis des Laufs vom 17.09.2026: er liest eine
fertige Beschreibungsdatei (Default 3_evaluation/cache/descriptions_E_d1.pkl)
und erzeugt selbst keine Beschreibung. Er holt die eine Groesse nach, die sich nicht aus vorhandenen Abzuegen rechnen laesst: die
Zeile M5 + E_d1 in Tabelle 6.9 (PipelineP, DocP, DocR, DocF1).

Warum ein eigener Treiber und nicht 03_full_pipeline.py:

  * 03_full_pipeline.py laeuft ueber C.MAIN_SYSTEMS und C.EVAL_RESOLVERS. Beide
    kennen E_d1 nicht. Dieser Treiber nimmt Modell und Arm als Argument und
    ruehrt config.py nicht an.
  * 03_full_pipeline.py encodiert alle rund 209'000 eindeutigen Strings eines
    Systems in einem Zug und wurde dabei am 12.09.2026 zweimal vom
    Speicherwaechter abgeraeumt. Dieser Treiber encodiert in Dokumentchunks,
    wie run_fp_chunked.py es danach getan hat. Die Zerlegung ist exakt und
    nicht naeherungsweise: das Ranking ist je Item unabhaengig, und
    pipeline_metrics rechnet erst am Schluss ueber alle Dokumente.
  * 03_full_pipeline.py ruft prepare_items mit cache_key und wuerde fehlende
    Beschreibungen NACHERZEUGEN. Fuer E_d1 waere das der schlimmste Fall: der
    Resolver im Projektbaum traegt die B1-Matrix, nicht npmi_dist_matrix_D1.csv,
    und der Lauf mischte still zwei Masse. Dieser Treiber erzeugt keine einzige
    Beschreibung. Er prueft die Abdeckung und bricht ab, wenn auch nur eine
    fehlt (Torwaechter G4). Die Beschreibungsdatei wird ausschliesslich gelesen.

Unveraendert wiederverwendet: eval_core.build_resolver, eval_core.load_documents,
eval_core.text_index, eval_core.encode_and_rank sowie detection_metrics und
pipeline_metrics aus 03_full_pipeline.py.

Es schreibt NICHTS Bestehendes um. Alle Ausgaben tragen neue Namen:
  cache/items_pred_E_d1.pkl            (neu)
  results/full_pipeline_E_d1.json      (neu)
full_pipeline.json, summary.csv, die Beschreibungs- und Item-Caches des Laufs
vom 17.09. und results/matrix/ bleiben unberuehrt.
"""
from __future__ import annotations

import argparse, gc, hashlib, importlib.util, json, pickle, sys, time
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
_EVAL_DEFAULT = Path(__file__).resolve().parents[1] / "3_evaluation"
ap.add_argument("--eval-dir", default=str(_EVAL_DEFAULT), help="Ordner 3_evaluation (Default: ../3_evaluation)")
ap.add_argument("--arm", default="E_d1")
ap.add_argument("--model", default="M5_spatial_config2")
ap.add_argument("--weights", default=None,
                help="Modellgewichte (Ordner oder HF-ID; Default: config.MODELS[--model])")
ap.add_argument("--descriptions", default=None,
                help="descriptions_<arm>.pkl, nur lesend (Default: 3_evaluation/cache/)")
ap.add_argument("--desc-sha", default=None,
                help="erwarteter SHA256 der Beschreibungsdatei (optional)")
ap.add_argument("--config", default="config1")
ap.add_argument("--chunk-docs", type=int, default=400)
ap.add_argument("--nur-items", action="store_true", help="bei Schritt 2 anhalten")
A = ap.parse_args()

EVAL = Path(A.eval_dir).resolve()
sys.path.insert(0, str(EVAL))
sys.argv = ["fp_E-d1.py"]

import config as C          # noqa: E402
import eval_core as E       # noqa: E402

if A.weights is None:
    A.weights = C.MODELS[A.model]["weights"]
if A.descriptions is None:
    A.descriptions = str(C.CACHE_DIR / f"descriptions_{A.arm}.pkl")

spec = importlib.util.spec_from_file_location("fp03", EVAL / "03_full_pipeline.py")
fp = importlib.util.module_from_spec(spec); spec.loader.exec_module(fp)

def sagen(s=""):
    print(s, flush=True)

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def abbruch(s: str):
    sagen(f"\nABBRUCH: {s}")
    raise SystemExit(1)

T0 = time.time()
ITEMS_PRED = C.CACHE_DIR / f"items_pred_{A.arm}.pkl"
OUT = C.RESULTS_DIR / f"full_pipeline_{A.arm}.json"

sagen("=" * 74)
sagen(f"Ende zu Ende — {A.model} + {A.arm}")
sagen("=" * 74)

# ── G1  Beschreibungsdatei ist die des Laufs vom 17.09. ──────────────────────
desc_path = Path(A.descriptions)
if not desc_path.exists():
    abbruch(f"Beschreibungsdatei fehlt: {desc_path}")
ist = sha256(desc_path)
if A.desc_sha and ist != A.desc_sha:
    abbruch(f"Die Beschreibungsdatei hat nicht den erwarteten Hash.\n"
            f"  erwartet: {A.desc_sha}\n  erhalten: {ist}\n"
            f"  Das ist nicht die Datei, auf der die 22 Zellen vom 17.09. beruhen.")
with open(desc_path, "rb") as fh:
    DESC = pickle.load(fh)
sagen(f"G1  Beschreibungen bestaetigt: {len(DESC):,} Schluessel, SHA {ist[:16]}…".replace(",", "'"))

# ── G2  Nichts Bestehendes wird ueberschrieben ───────────────────────────────
for p in (OUT,):
    if p.exists():
        abbruch(f"{p} liegt schon. Benenn sie um (nicht loeschen) und starte neu.")
sagen("G2  Keine Ausgabedatei liegt im Weg.")

# ── Dokumente und Recognizer-Spannen ─────────────────────────────────────────
documents = E.load_documents(None)
text_by_doc = E.text_index(documents)
sagen(f"    {len(documents):,} Dokumente geladen".replace(",", "'"))

rec_cache = C.CACHE_DIR / "recognizer_spans.pkl"
if not rec_cache.exists():
    abbruch(f"Recognizer-Cache fehlt: {rec_cache}\n"
            "  Ohne ihn liefe spaCy ueber alle Volltexte und die Spannenmenge waere\n"
            "  eine andere als die der vier bestehenden Zeilen von Tabelle 6.9.")
with open(rec_cache, "rb") as fh:
    pred_by_doc = pickle.load(fh)["spans"]
fehlend = {d["filename"] for d in documents} - set(pred_by_doc)
if fehlend:
    abbruch(f"G3  Recognizer-Cache unvollstaendig: {len(fehlend)} Dokumente fehlen.")
det = fp.detection_metrics(documents, pred_by_doc)
n_spans = sum(len(v) for v in pred_by_doc.values())
sagen(f"G3  Recognizer-Spannen aus dem Cache: {n_spans:,} ".replace(",", "'")
      + f"— Detection P={det['precision']:.4f} R={det['recall']:.4f} F1={det['f1']:.4f}")
if det["tp"] != 54890 or det["fp"] != 673535 or det["fn"] != 20678:
    abbruch("Der Detection-Block weicht vom Bestand ab (Soll 54'890 / 673'535 / 20'678).\n"
            "  Er ist modellunabhaengig und muss ueber alle Laeufe bitgleich sein.")
sagen("    Detection-Block zeichengleich mit dem Bestand (A 3.3).")

# ── Schritt 1 von 3 — die Items aus den vorhergesagten Spannen ───────────────
sagen("\nSchritt 1 von 3 — Items aus den vorhergesagten Spannen")
if ITEMS_PRED.exists():
    with open(ITEMS_PRED, "rb") as fh:
        cached = pickle.load(fh)
    if cached.get("n_docs") != len(documents):
        abbruch(f"{ITEMS_PRED} passt nicht zur Dokumentzahl "
                f"({cached.get('n_docs')} statt {len(documents)}).")
    items = cached["items"]
    sagen(f"    aus dem Cache: {ITEMS_PRED} ({len(items):,} Spannen)".replace(",", "'"))
    del cached
else:
    sagen(f"    Gazetteer-Suche ueber {n_spans:,} vorhergesagte Spannen …".replace(",", "'"))
    t = time.time()
    res = E.build_resolver(A.weights, "spatial", A.config)
    memo: dict[str, list] = {}
    spans = []
    for d in documents:
        text, doc_id = d["text"], d["filename"]
        for (s, e) in pred_by_doc.get(doc_id, []):
            tt = text[s:e]
            if tt not in memo:
                memo[tt] = res.gazetteer.search(tt)
            spans.append((doc_id, s, e, tt))
    sagen(f"    {len(memo):,} eindeutige Toponymtexte, {time.time()-t:.0f} s".replace(",", "'"))

    # G4 — die Abdeckungspruefung. Hier haengt die Gueltigkeit des Laufs.
    noetig = {c.location_id_value for cs in memo.values() for c in cs}
    ohne = noetig - set(DESC)
    if ohne:
        abbruch(f"G4  {len(ohne)} Kandidaten haben keine D1-Beschreibung.\n"
                "  Dieser Treiber erzeugt grundsaetzlich keine Beschreibungen: der Resolver\n"
                "  im Projektbaum traegt die B1-Matrix, und ein nacherzeugter Eintrag waere\n"
                "  still ein anderes Mass. Wer die Zeile trotzdem will, baut die fehlenden\n"
                "  Beschreibungen mit 05b_build_caches_E-d1.py und npmi_dist_matrix_D1.csv\n"
                "  nach und startet erneut.")
    sagen(f"G4  Abdeckung vollstaendig: alle {len(noetig):,} benoetigten Kandidaten ".replace(",", "'")
          + "haben eine D1-Beschreibung. Es wird keine erzeugt.")

    items = [{"doc_id": doc_id, "start": s, "end": e, "gold_id": None,
              "candidate_ids": [c.location_id_value for c in memo[tt]],
              "descriptions": [DESC[c.location_id_value] for c in memo[tt]]}
             for (doc_id, s, e, tt) in spans]
    tmp = ITEMS_PRED.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump({"n_docs": len(documents), "items": items, "arm": A.arm,
                     "descriptions_sha256": ist}, fh)
    tmp.replace(ITEMS_PRED)
    sagen(f"    geschrieben: {ITEMS_PRED} ({len(items):,} Spannen, ".replace(",", "'")
          + f"{time.time()-t:.0f} s)")
    del res, memo, spans
    gc.collect()

if A.nur_items:
    sagen("\n--nur-items gesetzt. Halt nach Schritt 1.")
    raise SystemExit(0)

# ── Schritt 2 von 3 — encodieren und ranken, in Dokumentchunks ───────────────
sagen(f"\nSchritt 2 von 3 — encodieren und ranken mit {A.model}, "
      f"Chunks zu {A.chunk_docs} Dokumenten")
items_by_doc: dict[str, list] = defaultdict(list)
for it in items:
    items_by_doc[it["doc_id"]].append(it)
del items
gc.collect()

t = time.time()
ctx = E.build_resolver(A.weights, "default")
namen = [d["filename"] for d in documents]
teile = [namen[i:i + A.chunk_docs] for i in range(0, len(namen), A.chunk_docs)]
predid_by_span: dict[tuple, str] = {}
for i, teil in enumerate(teile, 1):
    chunk = [it for fn in teil for it in items_by_doc[fn]]
    if not chunk:
        continue
    per_item = E.encode_and_rank(ctx, chunk, text_by_doc)
    for p in per_item:
        predid_by_span[(p["doc_id"], p["start"], p["end"])] = p["pred_id"]
    del per_item, chunk
    gc.collect()
    sagen(f"    Chunk {i:>3}/{len(teile)}  {len(predid_by_span):>7,} Spannen  "
          .replace(",", "'") + f"{time.time()-t:.0f} s")
del ctx
gc.collect()
sagen(f"    fertig nach {time.time()-t:.0f} s")

# ── Schritt 3 von 3 — die Metriken ──────────────────────────────────────────
sagen("\nSchritt 3 von 3 — Metriken")
m = fp.pipeline_metrics(documents, pred_by_doc, predid_by_span)
OUT.write_text(json.dumps(
    {"_meta": {"skript": "fp_E-d1.py", "arm": A.arm, "modell": A.model,
               "descriptions_sha256": ist, "chunk_docs": A.chunk_docs,
               "hinweis": "Ergaenzt results/full_pipeline.json um die Zeile "
                          f"{A.model}+{A.arm}; jene Datei ist unberuehrt."},
     "detection": det, "systems": {f"{A.model}+{A.arm}": m}},
    ensure_ascii=False, indent=2), encoding="utf-8")

sagen("")
sagen("=" * 74)
sagen(f"ERGEBNIS  {A.model} + {A.arm}")
sagen("=" * 74)
sagen(f"  PipelineP  {m['full_pipeline_precision']:.4f}")
sagen(f"  DocP       {m['document_precision']:.4f}")
sagen(f"  DocR       {m['document_recall']:.4f}")
sagen(f"  DocF1      {m['document_f1']:.4f}")
sagen(f"  aufgeloest {m['correct_resolutions']:,} von {m['total_predicted']:,} "
      .replace(",", "'") + "vorhergesagten Spannen")
sagen("")
sagen("  Vergleichsarm, Tabelle 6.9, aus results/tables/full_pipeline.csv:")
sagen("    M5+E_c1  PipelineP .0586  DocP .0661  DocR .5126  DocF1 .1171")
sagen("    M3+E_def PipelineP .0573  DocP .0655  DocR .5044  DocF1 .1159")
sagen("")
sagen(f"  Geschrieben: {OUT}")
sagen(f"  Gesamtdauer: {(time.time()-T0)/60:.1f} min")
