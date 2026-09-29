"""
Wiederverwendbare Eval-Logik fuer die Text+Berg-Evaluation.

Adaptiert aus experiment-1/06_train.py (create_resolver, prepare_eval_data,
evaluate_with_cached_data), aber:
  - kein K-Fold (ganzer Korpus = Test-Set, ein Durchlauf)
  - Trennung modell-unabhaengiger Teil (Gazetteer-Suche + H3-Beschreibung) vom
    modell-abhaengigen Teil (Kontext-Extraktion + Encoding)
  - gebatchtes Encoding eindeutiger Strings (statt 1 encode-Call pro Toponym)
  - zusaetzlich Document-Unit-Metriken (pro-Dokument-Accuracy, makro-gemittelt)

Item-Schema (modell-unabhaengig, in Stage 01 gecacht):
    {"doc_id": str, "start": int, "end": int, "gold_id": str|None,
     "candidate_ids": [str, ...], "descriptions": [str, ...]}
"""

from __future__ import annotations

import json
import pickle
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Tuple

import torch

from geoparser.modules.resolvers.sentencetransformer import SentenceTransformerResolver
from geoparser_h3_resolver import SpatialSentenceResolver

import config as C
import winfix_sqlite_pool

winfix_sqlite_pool.apply()

SpanGetter = Callable[[dict], List[Tuple[int, int, Optional[str]]]]
Key = Tuple[str, int, int]


# ── Daten ──────────────────────────────────────────────────────────────────────
def load_documents(max_docs: Optional[int] = None) -> List[dict]:
    """Laedt eval_dataset.json → Liste von {filename, text, toponyms:[...]}."""
    with open(C.EVAL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = data["documents"]
    if max_docs is not None:
        docs = docs[:max_docs]
    return docs


def gold_spans(doc: dict) -> List[Tuple[int, int, Optional[str]]]:
    """Span-Getter fuer Gold-Toponyme (Standard fuer die Resolver-Eval)."""
    return [(t["start"], t["end"], t["loc_id"]) for t in doc["toponyms"]]


def text_index(documents: List[dict]) -> Dict[str, str]:
    return {d["filename"]: d["text"] for d in documents}


# ── Resolver-Factory ────────────────────────────────────────────────────────────
def build_resolver(
    weights: str,
    kind: str,
    config_name: Optional[str] = None,
    sentence_config=None,
):
    """Baut einen Resolver (vgl. create_resolver in experiment-1/06_train.py).

    kind="default" → SentenceTransformerResolver
    kind="spatial" → SpatialSentenceResolver (config1/config2 oder eigener sentence_config)
    """
    if kind == "default":
        resolver = SentenceTransformerResolver(weights, gazetteer_name=C.GAZETTEER)
    else:
        resolver = SpatialSentenceResolver(
            model_name=weights,
            gazetteer_name=C.GAZETTEER,
            config_path=C.config_yaml(config_name) if config_name else None,
            duckdb_path=C.duckdb_path(config_name) if config_name else None,
            sentence_config=sentence_config,
        )
    _assert_model_sane(resolver.transformer, weights)
    return resolver


def eval_sentence_config(eid: str):
    """SentenceGeneratorConfig fuer einen Eval-Resolver mit 'measure'/'overrides'.

    Liefert None, wenn der Resolver die YAML unveraendert nutzt (dann leitet der
    SpatialSentenceResolver die Config selbst aus config_path ab). Sonst: YAML
    laden, Assoziationsmass setzen (Matrix <measure>_matrix.csv neben der
    DuckDB) und die angegebenen Felder ueberschreiben — derselbe Weg wie
    BuildConfig.to_sentence_generator_config im Plugin.
    """
    from geoparser_h3_resolver.pipeline.build_config import BuildConfig

    cfg = C.EVAL_RESOLVERS[eid]
    if cfg["kind"] != "spatial" or not ("measure" in cfg or cfg.get("overrides")):
        return None
    build = BuildConfig.from_yaml(C.config_yaml(cfg["config"]))
    measure = cfg.get("measure", build.association_measure)
    build = replace(build, association_measure=measure, matrix_path=None)
    sg = build.to_sentence_generator_config(C.matrix_path(cfg["config"], measure))
    return replace(sg, **cfg.get("overrides", {}))


def build_eval_resolver(eid: str, weights: str = C.BASE_MODEL):
    """Resolver fuer einen Eintrag aus config.EVAL_RESOLVERS (inkl. E_d1-Arme)."""
    cfg = C.EVAL_RESOLVERS[eid]
    return build_resolver(weights, cfg["kind"], cfg["config"],
                          sentence_config=eval_sentence_config(eid))


def _assert_model_sane(transformer, weights: str) -> None:
    """Probe-Encoding gegen stilles Degenerieren (korrupte Weights -> NaN-Embeddings).

    Hintergrund: Die per Dateitransfer korrumpierten M3/M4/M5 lieferten NaN fuer
    jeden Input; cosine_similarity(NaN)=NaN liess das Ranking zur Kandidaten-
    reihenfolge kollabieren (alle Zellen identisch) statt zu crashen.
    """
    probe = transformer.encode(
        ["Zuerich ist eine Stadt.", "Der Eiger ist ein Berg."],
        convert_to_tensor=True, show_progress_bar=False,
    )
    if torch.isnan(probe).any():
        raise RuntimeError(
            f"Modell '{weights}' liefert NaN-Embeddings (korrupte Weights?). "
            "Weights pruefen/neu beschaffen, siehe NaN-Scan via safetensors."
        )


# ── Stage 1: modell-unabhaengige Items (Gazetteer + Beschreibungen) ─────────────
class DescriptionCache:
    """Persistenter {candidate-UUID: description}-Cache pro Variante.

    Descriptions haengen nur vom Feature + der Generator-Config ab (nicht vom
    Modell, nicht vom Span) — pro Variante (E_default/E_c1/E_c2/Ablation) wird
    eine Pickle-Datei gefuehrt. Inkrementeller, atomarer Flush macht lange
    Laeufe resume-faehig (Windows-Update-Reboots, Crashes).
    """

    def __init__(self, key: str):
        self.path = C.CACHE_DIR / f"descriptions_{key}.pkl"
        self.data: Dict[str, str] = {}
        if self.path.exists():
            with open(self.path, "rb") as f:
                self.data = pickle.load(f)
        self._dirty = 0

    def set(self, uuid: str, description: str) -> None:
        self.data[uuid] = description
        self._dirty += 1

    def flush(self) -> None:
        if self._dirty == 0:
            return
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self.data, f)
        tmp.replace(self.path)
        self._dirty = 0


DESC_CHUNK = 300  # Features pro Batch-/Flush-Chunk in prepare_items


def prepare_items(
    resolver,
    documents: List[dict],
    span_getter: SpanGetter = gold_spans,
    cache_key: Optional[str] = None,
) -> List[dict]:
    """Gazetteer-Suche + Kandidaten-Beschreibungen pro Span. Modell-unabhaengig.

    Drei Phasen statt Doppelschleife ueber alle Spans:
      1. Spans sammeln, Gazetteer-Suche pro eindeutigem Toponym-Text (memoized;
         75k Toponyme im Korpus -> ~3.5k eindeutige Texte).
      2. Beschreibungen nur fuer eindeutige Kandidaten-Features erzeugen —
         chunk-weise, mit persistentem Cache (cache_key) und Batch-Vorberechnung
         (resolver.precompute_descriptions, falls vorhanden).
      3. Items aus Memo + Beschreibungs-Map zusammensetzen.
    """
    # Phase 1: Spans + memoized Kandidaten.
    search_memo: Dict[str, list] = {}
    spans: List[Tuple[str, int, int, Optional[str], str]] = []
    for doc in documents:
        text, doc_id = doc["text"], doc["filename"]
        for start, end, gold in span_getter(doc):
            toponym_text = text[start:end]
            if toponym_text not in search_memo:
                search_memo[toponym_text] = resolver.gazetteer.search(toponym_text)
            spans.append((doc_id, start, end, gold, toponym_text))

    # Phase 2: Beschreibungen fuer eindeutige Kandidaten sicherstellen.
    cand_by_uuid: Dict[str, object] = {}
    for cands in search_memo.values():
        for c in cands:
            cand_by_uuid.setdefault(c.location_id_value, c)

    cache = DescriptionCache(cache_key) if cache_key else None
    desc_by_uuid: Dict[str, str] = dict(cache.data) if cache else {}
    missing = [c for u, c in cand_by_uuid.items() if u not in desc_by_uuid]
    if missing:
        print(f"    {len(cand_by_uuid)} eindeutige Kandidaten, {len(missing)} ohne "
              f"gecachte Beschreibung", flush=True)

    done = 0
    for i in range(0, len(missing), DESC_CHUNK):
        chunk = missing[i:i + DESC_CHUNK]
        if hasattr(resolver, "precompute_descriptions"):
            resolver.precompute_descriptions(chunk)  # Batch-DuckDB-Pfad (spatial)
        for c in chunk:
            u = c.location_id_value
            desc_by_uuid[u] = resolver._generate_description(c)
            if cache:
                cache.set(u, desc_by_uuid[u])
        if cache:
            cache.flush()
        done += len(chunk)
        if len(missing) > DESC_CHUNK:
            print(f"    descriptions: {done}/{len(missing)}", flush=True)

    # Phase 3: Items bauen.
    items = []
    for doc_id, start, end, gold, toponym_text in spans:
        candidates = search_memo[toponym_text]
        items.append({
            "doc_id": doc_id, "start": start, "end": end, "gold_id": gold,
            "candidate_ids": [c.location_id_value for c in candidates],
            "descriptions": [desc_by_uuid[c.location_id_value] for c in candidates],
        })
    return items


# ── Stage 2: Encoding + Ranking (modell-abhaengig) ──────────────────────────────
def extract_contexts(
    resolver,
    items: List[dict],
    text_by_doc: Dict[str, str],
) -> Dict[Key, str]:
    """Extrahiert den Kontext-String je Toponym mit Kandidaten (Modell-Tokenizer)."""
    contexts: Dict[Key, str] = {}
    for it in items:
        if not it["candidate_ids"]:
            continue
        key: Key = (it["doc_id"], it["start"], it["end"])
        if key in contexts:
            continue
        text = text_by_doc[it["doc_id"]]
        contexts[key] = resolver._extract_context(text, it["start"], it["end"])
    return contexts


def encode_strings(transformer, strings: List[str], batch_size: int) -> Dict[str, torch.Tensor]:
    """Encodiert eindeutige Strings gebatcht → {string: embedding (CPU)}."""
    uniq = sorted(set(strings))
    if not uniq:
        return {}
    embs = transformer.encode(
        uniq, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=True,
    )
    embs = embs.cpu()
    return {s: embs[i] for i, s in enumerate(uniq)}


def rank_items(
    items: List[dict],
    ctx_by_key: Dict[Key, str],
    emb: Dict[str, torch.Tensor],
) -> List[dict]:
    """Rankt Kandidaten je Item per Cosine-Similarity; liefert pro-Item-Resultate.

    per_item: {doc_id, gold_id, n_candidates, rank (1-indexiert oder None), rr}
    """
    per_item = []
    for it in items:
        n = len(it["candidate_ids"])
        base = {"doc_id": it["doc_id"], "start": it["start"], "end": it["end"],
                "gold_id": it["gold_id"], "n_candidates": n}
        if n == 0:
            per_item.append({**base, "rank": None, "rr": 0.0, "pred_id": None})
            continue
        key: Key = (it["doc_id"], it["start"], it["end"])
        ctx_emb = emb[ctx_by_key[key]]
        cand_embs = torch.stack([emb[d] for d in it["descriptions"]])
        sims = torch.nn.functional.cosine_similarity(ctx_emb.unsqueeze(0), cand_embs, dim=1)
        ranked = sims.argsort(descending=True).tolist()
        rank = None
        for pos, idx in enumerate(ranked):
            if it["candidate_ids"][idx] == it["gold_id"]:
                rank = pos + 1
                break
        per_item.append({**base, "rank": rank, "rr": (1.0 / rank) if rank else 0.0,
                         "pred_id": it["candidate_ids"][ranked[0]]})
    return per_item


def encode_and_rank(resolver, items, text_by_doc, batch_size=C.BATCH_SIZE,
                    ctx_by_key: Optional[Dict[Key, str]] = None) -> List[dict]:
    """Komfort: Kontext extrahieren (falls nicht uebergeben), encodieren, ranken."""
    if ctx_by_key is None:
        ctx_by_key = extract_contexts(resolver, items, text_by_doc)
    desc = [d for it in items for d in it["descriptions"]]
    emb = encode_strings(resolver.transformer, list(ctx_by_key.values()) + desc, batch_size)
    return rank_items(items, ctx_by_key, emb)


# ── Metriken ─────────────────────────────────────────────────────────────────────
def location_metrics(per_item: List[dict]) -> dict:
    """Acc@1/@3/MRR. Primaer: no-candidate zaehlt als Fehler (Konzept 2.1)."""
    total = len(per_item)
    no_cand = sum(1 for p in per_item if p["n_candidates"] == 0)
    c1 = sum(1 for p in per_item if p["rank"] == 1)
    c3 = sum(1 for p in per_item if p["rank"] is not None and p["rank"] <= 3)
    mrr = sum(p["rr"] for p in per_item) / total if total else 0.0
    with_cand = [p for p in per_item if p["n_candidates"] > 0]
    nc = len(with_cand)
    c1_wc = sum(1 for p in with_cand if p["rank"] == 1)
    c3_wc = sum(1 for p in with_cand if p["rank"] is not None and p["rank"] <= 3)
    return {
        "total": total,
        "no_candidates": no_cand,
        "accuracy_at_1": c1 / total if total else 0.0,
        "accuracy_at_3": c3 / total if total else 0.0,
        "mrr": mrr,
        "accuracy_at_1_with_candidates": c1_wc / nc if nc else 0.0,
        "accuracy_at_3_with_candidates": c3_wc / nc if nc else 0.0,
    }


def document_metrics(per_item: List[dict]) -> dict:
    """Makro-gemittelte pro-Dokument-Accuracy (= P=R=F1 ohne Recognizer)."""
    by_doc: Dict[str, List[dict]] = {}
    for p in per_item:
        by_doc.setdefault(p["doc_id"], []).append(p)
    accs = []
    for items in by_doc.values():
        correct = sum(1 for p in items if p["rank"] == 1)
        accs.append(correct / len(items))
    macro = sum(accs) / len(accs) if accs else 0.0
    return {"document_accuracy": macro, "n_documents": len(by_doc),
            "document_precision": macro, "document_recall": macro, "document_f1": macro}
