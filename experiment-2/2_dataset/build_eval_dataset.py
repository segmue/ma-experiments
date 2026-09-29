"""
Baut den einheitlichen Evaluationsdatensatz fuer experiment2 (Text+Berg).

Quelle der Wahrheit:
  - ma-experiments/data/corpus_swissnames3d.gpkg  (gematchte Toponyme + sn3d_uuid)
  - ma-experiments/data/text_berg/**.xml          (Wort-Tokens fuer Volltext + Offsets)

Ziel: eine geoparser-Annotator-JSON (Format des geoparser-Annotators), direkt nutzbar
von geoparser Project.load_annotations() und vom Eval-Loop (3_evaluation/eval_core.py).

Designentscheidungen:
  - Strenger Ground-Truth-Scope: nur rematch_status=matched, match_method=exact_name,
    is_ambiguous=False, gueltige sn3d_uuid.
  - Nur deutschsprachige Artikel: gefiltert ueber article/@lang in KEEP_LANGS. Das ist
    noetig, weil der Korpus mehrsprachig ist (de/fr/en/it/rm) und die `mul`-Dateien
    deutsche und fremdsprachige Artikel mischen -- Dateinamen reichen nicht.
  - Ein "Dokument" = ein Artikel (a{N}) innerhalb einer Quelle (source). Das GeoPackage-
    `source` ist ein ganzes Jahrbuch mit vielen Artikeln; die Wort-ID a{N}-s{S}-w{W}
    liefert die Artikelzugehoerigkeit gratis. Artikel ist die natuerliche "Article Unit".
  - Volltext + Zeichen-Offsets werden aus den Wort-Tokens rekonstruiert; Offsets stammen
    aus exakt dem String, der als `text` gespeichert wird (Invariante hart geprueft).

Output: ma-experiments/data/eval_dataset.json
"""

import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import geopandas as gpd

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE.parents[1] / "data"          # ma-experiments/data (gitignoriert)
INPUT_GPKG = DATA_DIR / "corpus_swissnames3d.gpkg"
INPUT_LAYER = "corpus_toponyms"
CORPUS_DIR = DATA_DIR / "text_berg"
OUTPUT_JSON = DATA_DIR / "eval_dataset.json"

GAZETTEER = "swissnames3d"

# Nur deutschsprachige Artikel werden verarbeitet (siehe Modul-Docstring).
KEEP_LANGS = {"de"}

# Smart-Detokenisierung: kein Space vor schliessender Interpunktion ...
NO_SPACE_BEFORE = set(".,;:!?)]}»…%‰’\"'")
# ... und kein Space nach oeffnender Interpunktion.
NO_SPACE_AFTER = set("([{«„\"'")


def strict_filter(gdf):
    """Strenger Ground-Truth-Scope (siehe Modul-Docstring)."""
    uuid = gdf["sn3d_uuid"].astype("string")
    mask = (
        (gdf["rematch_status"] == "matched")
        & (gdf["match_method"] == "exact_name")
        & (gdf["is_ambiguous"] == False)  # noqa: E712 (pandas-Vektor)
        & uuid.notna()
        & (uuid.str.len() > 0)
    )
    return gdf.loc[mask, ["toponym", "span", "source", "sn3d_uuid"]].copy()


def find_text_xml(corpus_dir):
    """Map source-Stem -> Text-XML-Pfad (ohne -ner / -TIMEX3 Varianten)."""
    mapping = {}
    for xml in corpus_dir.rglob("*.xml"):
        stem = xml.stem
        if stem.endswith("-ner") or stem.endswith("-TIMEX3"):
            continue
        mapping[stem] = xml
    return mapping


def article_of(word_id):
    """'a2-s622-w23' -> 'a2'."""
    return word_id.split("-", 1)[0]


def reconstruct_articles(text_xml):
    """
    Liest ein Text-XML und rekonstruiert pro Artikel den Volltext + Wort-Offsets.

    Returns: (result, article_lang)
      result       -> dict article -> (text, {word_id: (start, end)})
      article_lang -> dict article -> lang (article/@lang, z.B. "de")
    Satzgrenzen kommen aus den <s>-Elementen; innerhalb eines Satzes wird
    smart-detokenisiert. Saetze werden mit einem einzelnen Space getrennt.
    """
    root = ET.parse(text_xml).getroot()

    # Sprache je Artikel aus dem <article>-Element (Pflicht-Attribut lang).
    article_lang = {}
    for art in root.iter("article"):
        aid = art.get("id")
        if aid:
            article_lang[aid] = art.get("lang")

    # Saetze je Artikel in Dokumentreihenfolge sammeln.
    sentences_by_article = defaultdict(list)  # article -> list[ list[(wid, wtext)] ]
    for s in root.iter("s"):
        sid = s.get("id")
        if not sid:
            continue
        article = article_of(sid)
        words = [
            (w.get("id"), (w.text or "").strip())
            for w in s.iter("w")
        ]
        words = [(wid, wt) for wid, wt in words if wid and wt]
        if words:
            sentences_by_article[article].append(words)

    result = {}
    for article, sentences in sentences_by_article.items():
        chunks = []
        pos = 0
        offsets = {}
        prev_tok = ""
        for si, sent in enumerate(sentences):
            if si > 0:  # Satztrenner
                chunks.append(" ")
                pos += 1
                prev_tok = " "
            for wi, (wid, wtext) in enumerate(sent):
                need_space = wi > 0 or si > 0
                if need_space and prev_tok != " ":
                    if wtext[0] in NO_SPACE_BEFORE or (prev_tok and prev_tok[-1] in NO_SPACE_AFTER):
                        need_space = False
                else:
                    # direkt nach Satztrenner-Space: keinen weiteren Space
                    need_space = need_space and prev_tok != " "
                if need_space:
                    chunks.append(" ")
                    pos += 1
                start = pos
                chunks.append(wtext)
                pos += len(wtext)
                offsets[wid] = (start, pos)
                prev_tok = wtext
        result[article] = ("".join(chunks), offsets)
    return result, article_lang


def smart_join(words):
    """Detokenisiert eine Wortliste mit denselben Regeln wie reconstruct_articles."""
    out = ""
    for i, w in enumerate(words):
        if i == 0:
            out = w
            continue
        if w[0] in NO_SPACE_BEFORE or (out and out[-1] in NO_SPACE_AFTER):
            out += w
        else:
            out += " " + w
    return out


def main():
    print(f"1/4  GeoPackage laden: {INPUT_GPKG}")
    gdf = gpd.read_file(INPUT_GPKG, layer=INPUT_LAYER)
    print(f"     {len(gdf)} Zeilen gesamt.")
    df = strict_filter(gdf)
    print(f"     {len(df)} Zeilen nach strengem Filter.")

    print(f"2/4  Text-XML-Dateien indexieren in {CORPUS_DIR}")
    text_xml = find_text_xml(CORPUS_DIR)
    print(f"     {len(text_xml)} Text-XML gefunden.")

    # Toponyme nach source gruppieren.
    by_source = defaultdict(list)
    for row in df.itertuples(index=False):
        by_source[row.source].append((row.span, row.sn3d_uuid, row.toponym))

    print("3/4  Volltext rekonstruieren + Toponyme zuordnen ...")
    documents = []  # (filename, text, [toponyms])
    stats = defaultdict(int)
    missing_sources = []

    for source in sorted(by_source):
        xml_path = text_xml.get(source)
        if xml_path is None:
            missing_sources.append(source)
            stats["dropped_no_xml"] += len(by_source[source])
            continue
        articles, article_lang = reconstruct_articles(xml_path)

        # Toponyme dieser Quelle nach Artikel gruppieren.
        topo_by_article = defaultdict(list)
        for span, uuid, surface in by_source[source]:
            wids = span.split()
            if not wids:
                stats["dropped_empty_span"] += 1
                continue
            topo_by_article[article_of(wids[0])].append((wids, uuid, surface))

        for article, topos in topo_by_article.items():
            if article_lang.get(article) not in KEEP_LANGS:
                stats["dropped_non_german"] += len(topos)
                continue
            entry = articles.get(article)
            if entry is None:
                stats["dropped_no_article"] += len(topos)
                continue
            text, offsets = entry
            toponyms = []
            for wids, uuid, surface in topos:
                if any(w not in offsets for w in wids):
                    stats["dropped_missing_word"] += 1
                    continue
                start = offsets[wids[0]][0]
                end = offsets[wids[-1]][1]
                expected = smart_join([text[offsets[w][0]:offsets[w][1]] for w in wids])
                if text[start:end] != expected:
                    stats["dropped_noncontiguous"] += 1
                    continue
                toponyms.append({
                    "text": text[start:end],
                    "start": start,
                    "end": end,
                    "loc_id": uuid,
                })
                stats["toponyms_kept"] += 1
            if toponyms:
                toponyms.sort(key=lambda t: t["start"])
                documents.append((f"{source}#{article}", text, toponyms))

    print("4/4  JSON schreiben ...")
    out = {
        "gazetteer": GAZETTEER,
        "documents": [
            {"filename": fn, "text": text, "toponyms": tops}
            for fn, text, tops in documents
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 60)
    print("  EVALUATIONSDATENSATZ ERSTELLT")
    print("=" * 60)
    print(f"  Dokumente (source#article):       {len(documents):>8}")
    print(f"  Toponyme behalten:                {stats['toponyms_kept']:>8}")
    print(f"  -- verworfen --")
    print(f"  nicht-deutscher Artikel:          {stats['dropped_non_german']:>8}")
    print(f"  fehlendes Wort-Token:             {stats['dropped_missing_word']:>8}")
    print(f"  nicht-kontiguer Span:             {stats['dropped_noncontiguous']:>8}")
    print(f"  Artikel nicht im XML:             {stats['dropped_no_article']:>8}")
    print(f"  leerer Span:                      {stats['dropped_empty_span']:>8}")
    print(f"  keine Text-XML fuer source:       {stats['dropped_no_xml']:>8}")
    if missing_sources:
        print(f"  Quellen ohne XML: {missing_sources}")
    print(f"  Output: {OUTPUT_JSON}")
    print("=" * 60)


if __name__ == "__main__":
    main()
