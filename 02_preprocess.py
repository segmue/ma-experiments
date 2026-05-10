"""
Step 2a: Annotations vorverarbeiten.

Laedt alle Annotator-JSONs aus data/annotations/, fuehrt sie zusammen,
bereinigt sie (spaCy Sentence-Check), reduziert ueberrepresentierte Toponyme,
und speichert das Ergebnis als data/preprocessed.json.

Die vorverarbeiteten Daten behalten die Dokument-Level-Struktur bei,
damit K-Fold CV auf Dokument-Ebene splitten kann (kein Data Leakage).

Voraussetzungen:
    - data/annotations/*.json  (JSON-Exporte aus dem Geoparser Annotator)

Output:
    data/preprocessed.json

Verwendung:
    poetry run python 02_preprocess.py
    poetry run python 02_preprocess.py --max-count 15
    poetry run python 02_preprocess.py --no-reduce
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger("preprocess")

BASE = Path(__file__).parent
ANNOTATIONS_DIR = BASE / "data" / "annotations"
OUTPUT_PATH = BASE / "data" / "preprocessed.json"
BASE_MODEL = "sentence-transformers/distiluse-base-multilingual-cased-v1"


def merge_annotation_files(annotation_dir: Path) -> Dict:
    """Merge multiple annotator JSON exports into a single dict.

    The geoparser's load_annotations() overwrites the tag context on each call,
    so multiple calls with the same tag would lose earlier annotations.
    This function combines all documents into one structure to avoid that.
    """
    files = sorted(annotation_dir.glob("*.json"))

    if not files:
        raise FileNotFoundError(
            f"Keine Annotations-JSONs in {annotation_dir} gefunden.\n"
            "JSON-Exporte aus dem Geoparser Annotator dort ablegen."
        )

    logger.info("Merge %d Annotations-Datei(en):", len(files))
    merged = None
    for json_file in files:
        with open(json_file, "r") as f:
            data = json.load(f)
        n_docs = len(data["documents"])
        logger.info("  %s  (%d Dokumente)", json_file.name, n_docs)
        if merged is None:
            merged = data
        else:
            merged["documents"].extend(data["documents"])

    logger.info("  -> %d Dokumente total", len(merged["documents"]))
    return merged


def clean_annotations(merged: Dict) -> Dict:
    """Entfernt Toponyme deren Position von spaCy keiner Sentence zugeordnet werden kann.

    _extract_context() im Geoparser crasht wenn spaCy's Sentence Splitter
    keine Sentence fuer eine Toponym-Position findet (z.B. bei Headern wie
    "=== Toggenburg ==="). Diese Funktion prueft das vorab und entfernt
    fehlerhafte Eintraege.
    """
    import spacy

    try:
        nlp = spacy.load("xx_sent_ud_sm")
    except OSError:
        spacy.cli.download("xx_sent_ud_sm")
        nlp = spacy.load("xx_sent_ud_sm")

    total_removed = 0
    for doc in merged["documents"]:
        spacy_doc = nlp(doc["text"])
        sentences = list(spacy_doc.sents)

        clean_toponyms = []
        for toponym in doc["toponyms"]:
            start = toponym["start"]
            found = any(sent.start_char <= start < sent.end_char for sent in sentences)
            if found:
                clean_toponyms.append(toponym)
            else:
                total_removed += 1
                logger.debug(
                    "  Entfernt: '%s' (pos %d-%d) in '%s...'",
                    toponym["text"], start, toponym["end"], doc["text"][:50],
                )

        doc["toponyms"] = clean_toponyms

    total_remaining = sum(len(doc["toponyms"]) for doc in merged["documents"])
    if total_removed:
        logger.info(
            "  -> %d Toponym(e) entfernt (keine spaCy-Sentence), %d uebrig",
            total_removed, total_remaining,
        )
    else:
        logger.info("  -> Alle %d Toponyme OK", total_remaining)

    return merged


def reduce_toponym_frequency(merged: Dict, max_count: int = 10) -> Dict:
    """Reduziert ueberrepresentierte Toponyme auf max_count Vorkommen.

    Strategie: Diversitaet maximieren. Vorkommen werden zuerst aus Dokumenten
    mit den meisten Eintraegen des betreffenden Toponyms entfernt. Innerhalb
    eines Dokuments werden die letzten Vorkommen (nach Textposition) zuerst
    entfernt, da das erste Vorkommen oft die informativste Kontexteinbettung hat.

    Args:
        merged: Zusammengefuehrte Annotations-Daten
        max_count: Maximale Anzahl Vorkommen pro Toponym-String
    """
    # 1. Frequenztabelle bauen
    # toponym_text (lower) -> [(doc_idx, toponym_idx_in_doc), ...]
    freq: Dict[str, List[tuple]] = defaultdict(list)
    for doc_idx, doc in enumerate(merged["documents"]):
        for topo_idx, toponym in enumerate(doc["toponyms"]):
            if toponym.get("loc_id") and toponym["loc_id"] != "":
                key = toponym["text"].lower()
                freq[key].append((doc_idx, topo_idx))

    # 2. Toponyme identifizieren die reduziert werden muessen
    reduction_stats = {}
    # Sammle Indizes die entfernt werden sollen: doc_idx -> set(topo_idx)
    to_remove: Dict[int, set] = defaultdict(set)

    for toponym_text, occurrences in freq.items():
        total = len(occurrences)
        if total <= max_count:
            continue

        logger.info(
            "  Reduziere '%s': %d -> %d Vorkommen", toponym_text, total, max_count,
        )
        reduction_stats[toponym_text] = {"before": total, "after": max_count}

        # Pro Dokument gruppieren: doc_idx -> [(topo_idx, start_pos), ...]
        doc_occurrences: Dict[int, List[tuple]] = defaultdict(list)
        for doc_idx, topo_idx in occurrences:
            start_pos = merged["documents"][doc_idx]["toponyms"][topo_idx]["start"]
            doc_occurrences[doc_idx].append((topo_idx, start_pos))

        # Innerhalb jedes Dokuments nach Position sortieren (aufsteigend)
        for doc_idx in doc_occurrences:
            doc_occurrences[doc_idx].sort(key=lambda x: x[1])

        # Iterativ entfernen: immer 1 vom Dokument mit den meisten Vorkommen
        remaining = total
        while remaining > max_count:
            # Finde Dokument mit den meisten noch nicht entfernten Vorkommen
            max_doc_idx = None
            max_doc_count = 0
            for doc_idx, occ_list in doc_occurrences.items():
                active_count = sum(
                    1 for topo_idx, _ in occ_list if topo_idx not in to_remove[doc_idx]
                )
                if active_count > max_doc_count:
                    max_doc_count = active_count
                    max_doc_idx = doc_idx

            if max_doc_idx is None or max_doc_count == 0:
                break

            # Letztes (nach Position) noch aktives Vorkommen in diesem Dokument entfernen
            for topo_idx, _ in reversed(doc_occurrences[max_doc_idx]):
                if topo_idx not in to_remove[max_doc_idx]:
                    to_remove[max_doc_idx].add(topo_idx)
                    remaining -= 1
                    break

    # 3. Entfernen anwenden
    for doc_idx, remove_indices in to_remove.items():
        doc = merged["documents"][doc_idx]
        doc["toponyms"] = [
            t for i, t in enumerate(doc["toponyms"]) if i not in remove_indices
        ]

    total_removed = sum(len(indices) for indices in to_remove.values())
    total_remaining = sum(len(doc["toponyms"]) for doc in merged["documents"])
    logger.info(
        "  -> %d Vorkommen entfernt, %d Toponyme uebrig",
        total_removed, total_remaining,
    )

    return merged, reduction_stats


def extract_documents(merged: Dict) -> List[Dict]:
    """Extrahiert Dokumente mit references/referents im fit()-Format.

    Filtert Toponyme ohne loc_id und Dokumente ohne Annotationen.
    Behält die Dokument-Level-Struktur bei (fuer K-Fold CV Split).
    """
    gazetteer_name = merged["gazetteer"]
    documents = []

    for doc_idx, doc in enumerate(merged["documents"]):
        refs = []
        referents = []
        for toponym in doc["toponyms"]:
            if toponym.get("loc_id") and toponym["loc_id"] != "":
                refs.append([toponym["start"], toponym["end"]])
                referents.append([gazetteer_name, toponym["loc_id"]])

        if refs:
            documents.append({
                "doc_id": doc_idx,
                "filename": doc.get("filename", f"doc_{doc_idx}"),
                "text": doc["text"],
                "references": refs,
                "referents": referents,
            })

    return documents


def main():
    parser = argparse.ArgumentParser(description="Preprocess annotations for training")
    parser.add_argument(
        "--max-count", type=int, default=10,
        help="Max occurrences per toponym string (default: 10)",
    )
    parser.add_argument(
        "--no-reduce", action="store_true",
        help="Skip toponym frequency reduction",
    )
    args = parser.parse_args()

    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Step 1: Merge
    logger.info("=== Schritt 1: Annotations zusammenfuehren ===")
    merged = merge_annotation_files(ANNOTATIONS_DIR)

    total_before = sum(len(doc["toponyms"]) for doc in merged["documents"])

    # Step 2: Clean
    logger.info("\n=== Schritt 2: Bereinigen (spaCy Sentence-Check) ===")
    merged = clean_annotations(merged)

    total_after_clean = sum(len(doc["toponyms"]) for doc in merged["documents"])

    # Step 3: Reduce frequency
    reduction_stats = {}
    if not args.no_reduce:
        logger.info("\n=== Schritt 3: Toponym-Frequenzreduktion (max %d) ===", args.max_count)
        merged, reduction_stats = reduce_toponym_frequency(merged, args.max_count)
    else:
        logger.info("\n=== Schritt 3: Frequenzreduktion uebersprungen ===")

    total_after_reduce = sum(len(doc["toponyms"]) for doc in merged["documents"])

    # Step 4: Extract documents
    logger.info("\n=== Schritt 4: Dokumente extrahieren ===")
    documents = extract_documents(merged)

    total_toponyms = sum(len(doc["references"]) for doc in documents)
    logger.info(
        "  -> %d Dokumente, %d annotierte Toponyme (mit loc_id)",
        len(documents), total_toponyms,
    )

    # Step 5: Save
    output = {
        "gazetteer": merged["gazetteer"],
        "base_model": BASE_MODEL,
        "preprocessing": {
            "max_count": args.max_count,
            "total_documents": len(documents),
            "total_toponyms_raw": total_before,
            "total_toponyms_after_clean": total_after_clean,
            "total_toponyms_after_reduce": total_after_reduce,
            "total_toponyms_with_loc_id": total_toponyms,
            "reduced_toponyms": reduction_stats,
        },
        "documents": documents,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info("\n=== Gespeichert: %s ===", OUTPUT_PATH)
    logger.info("Preprocessing abgeschlossen.")


if __name__ == "__main__":
    main()
