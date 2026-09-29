"""
Stage 07 — FP-Analyse des Recognizers: Wie viele "False Positives" sind echte Toponyme?

Kontext (concepts/spacy_finetuning_concept.md): Detection-P=0.075 wird gegen den
STRIKTEN Gold-Standard gemessen (nur exact_name/nicht-ambig/matched). Das
ungefilterte GeoPackage (corpus_swissnames3d.gpkg) enthaelt aber ALLE original
in Text+Berg geflaggten Toponyme (geo_type/stid), auch fuzzy/ambig/unmatched.

Klassifikation jedes Recognizer-FPs (kein Overlap mit striktem Gold):
  A  overlap_original   — ueberlappt eine Original-Annotation, die nur der strikte
                          Filter entfernt hat -> sicher echtes Toponym (nach Status
                          aufgeschluesselt)
  B  name_in_gazetteer  — kein Original-Overlap, aber exakter SwissNames3D-NAME
                          -> plausibel echtes Toponym ODER Ortsname=Alltagswort
  C  rest               — weder noch -> Kandidat fuer "echter FP"

Fuer B und C werden Stichproben mit Kontext fuer die manuelle Klassifikation
ausgegeben. Output: results/fp_sample_analysis.md

Verwendung (nach 03_full_pipeline.py):
    python 07_fp_sample_analysis.py
"""

from __future__ import annotations

import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import geopandas as gpd

import config as C
import eval_core as E

sys.path.insert(0, str(C.EXP2 / "2_dataset"))
import build_eval_dataset as B  # reconstruct_articles, article_of, find_text_xml

GPKG = C.CORPUS_GPKG
SAMPLE_B, SAMPLE_C = 25, 40
CTX = 120


def overlaps(a, b) -> bool:
    return min(a[1], b[1]) > max(a[0], b[0])


def original_spans_by_doc(needed_sources: set) -> dict:
    """{doc_id: [(start, end, status, toponym)]} fuer ALLE Original-Annotationen."""
    print(f"GeoPackage laden (alle Status): {GPKG}")
    gdf = gpd.read_file(GPKG, layer="corpus_toponyms",
                        columns=["toponym", "span", "source", "rematch_status",
                                 "match_method", "is_ambiguous"])
    print(f"  {len(gdf)} Original-Annotationen gesamt.")

    by_source = defaultdict(list)
    for row in gdf.itertuples(index=False):
        if row.source in needed_sources:
            by_source[row.source].append(row)

    text_xml = B.find_text_xml(B.CORPUS_DIR)
    result = defaultdict(list)
    for i, source in enumerate(sorted(by_source)):
        xml = text_xml.get(source)
        if xml is None:
            continue
        articles = B.reconstruct_articles(xml)
        for row in by_source[source]:
            wids = (row.span or "").split()
            if not wids:
                continue
            article = B.article_of(wids[0])
            entry = articles.get(article)
            if entry is None:
                continue
            _, offsets = entry
            if any(w not in offsets for w in wids):
                continue
            start = offsets[wids[0]][0]
            end = offsets[wids[-1]][1]
            status = f"{row.rematch_status}/{row.match_method}" + \
                     ("/ambig" if row.is_ambiguous else "")
            result[f"{source}#{article}"].append((start, end, status, row.toponym))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(by_source)} Quellen rekonstruiert", flush=True)
    return result


def gazetteer_names() -> set:
    import duckdb
    con = duckdb.connect(str(C.duckdb_path("config1")), read_only=True)
    names = {n for (n,) in con.execute("SELECT DISTINCT NAME FROM features").fetchall()}
    con.close()
    return names


def main():
    C.ensure_dirs()
    documents = E.load_documents()
    text_by_doc = E.text_index(documents)
    gold_by_doc = {d["filename"]: [(t["start"], t["end"]) for t in d["toponyms"]]
                   for d in documents}

    with open(C.CACHE_DIR / "recognizer_spans.pkl", "rb") as f:
        pred_by_doc = pickle.load(f)["spans"]

    # FPs: predicted Spans ohne Overlap mit striktem Gold.
    fps = []
    for d in documents:
        doc_id = d["filename"]
        gold = gold_by_doc[doc_id]
        for ps in pred_by_doc.get(doc_id, []):
            if not any(overlaps(ps, g) for g in gold):
                fps.append((doc_id, ps[0], ps[1]))
    print(f"{len(fps)} FP-Spans (kein Overlap mit striktem Gold).")

    # Klasse A: Overlap mit irgendeiner Original-Annotation.
    needed_sources = {d["filename"].split("#", 1)[0] for d in documents}
    orig = original_spans_by_doc(needed_sources)

    status_counts = Counter()
    class_a, rest = [], []
    for doc_id, s, e in fps:
        hit = next((o for o in orig.get(doc_id, []) if overlaps((s, e), (o[0], o[1]))), None)
        if hit is not None:
            class_a.append((doc_id, s, e, hit[2]))
            status_counts[hit[2]] += 1
        else:
            rest.append((doc_id, s, e))

    # Klasse B/C: exakter SwissNames3D-Name?
    names = gazetteer_names()
    class_b, class_c = [], []
    for doc_id, s, e in rest:
        text = text_by_doc[doc_id][s:e]
        (class_b if text in names else class_c).append((doc_id, s, e, text))

    n = len(fps)
    lines = [
        "# FP-Analyse des Recognizers (Stage 07)",
        "",
        f"Basis: {n} Recognizer-Spans ohne Overlap mit dem **strikten** Gold "
        f"(de_core_news_lg, {len(documents)} deutsche Docs).",
        "",
        "## Klassifikation",
        "",
        "| Klasse | Definition | n | Anteil |",
        "|---|---|---|---|",
        f"| A | überlappt Original-Annotation (nur vom strikten Filter entfernt) | {len(class_a)} | {len(class_a)/n:.1%} |",
        f"| B | kein Original-Overlap, aber exakter SwissNames3D-NAME | {len(class_b)} | {len(class_b)/n:.1%} |",
        f"| C | weder noch | {len(class_c)} | {len(class_c)/n:.1%} |",
        "",
        "## Klasse A nach Original-Status",
        "",
        "| Status (rematch/method) | n |",
        "|---|---|",
    ]
    for status, cnt in status_counts.most_common():
        lines.append(f"| {status} | {cnt} |")

    # Top-Oberflaechenformen je Klasse (fuer Muster-Erkennung).
    for label, cls in (("B", class_b), ("C", class_c)):
        forms = Counter(t for *_, t in cls)
        lines += ["", f"## Top-25 Oberflächenformen Klasse {label}", "",
                  "| Form | n |", "|---|---|"]
        for form, cnt in forms.most_common(25):
            lines.append(f"| {form} | {cnt} |")

    # Stichproben mit Kontext (manuelle Klassifikation).
    rng = random.Random(C.SEED)
    for label, cls, k in (("B", class_b, SAMPLE_B), ("C", class_c, SAMPLE_C)):
        sample = rng.sample(cls, min(k, len(cls)))
        lines += ["", f"## Stichprobe Klasse {label} ({len(sample)} von {len(cls)}, Seed {C.SEED})", ""]
        for i, (doc_id, s, e, text) in enumerate(sample, 1):
            doc_text = text_by_doc[doc_id]
            a, b = max(0, s - CTX), min(len(doc_text), e + CTX)
            ctx = " ".join((doc_text[a:s] + "**" + doc_text[s:e] + "**" + doc_text[e:b]).split())
            lines.append(f"{i}. `{doc_id}` „{text}“: …{ctx}…")

    out = C.LOGS_DIR / "fp_sample_analysis.md"   # enthaelt Korpusauszuege, nicht veroeffentlichen
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Gespeichert: {out}")
    print(f"A={len(class_a)} ({len(class_a)/n:.1%})  B={len(class_b)} "
          f"({len(class_b)/n:.1%})  C={len(class_c)} ({len(class_c)/n:.1%})")


if __name__ == "__main__":
    main()
