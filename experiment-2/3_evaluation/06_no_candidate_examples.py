"""
Stage 06 — Fehleranalyse: Gold-Toponyme ohne Gazetteer-Kandidaten.

Betreuer-Feedback 12.6.: Diese Items haben einen Gold-Eintrag im Gazetteer,
aber `gazetteer.search(<Oberflaechenform>)` liefert nichts — warum?
Quelle: cache/items_E_default.pkl (die Kandidatensuche ist resolver-unabhaengig,
alle Eval-Resolver nutzen denselben Gazetteer).

Output: results/no_candidate_examples.md mit
  1. Top-20 der haeufigsten No-Candidate-Oberflaechenformen (+ Gold-NAME daneben,
     um Muster wie Flexion/Getrennt-/Zusammenschreibung direkt zu sehen), und
  2. 10 diversen Beispielen: Kontextfenster mit **markiertem** Toponym,
     Offsets, doc_id und dem eigentlich korrekten Gazetteer-Eintrag.

Verwendung (nach 01_prepare_items.py):
    python 06_no_candidate_examples.py
    python 06_no_candidate_examples.py --n-examples 10 --context 150
"""

from __future__ import annotations

import argparse
import pickle
import random
from collections import Counter, defaultdict

import config as C
import eval_core as E


def load_items():
    path = C.CACHE_DIR / "items_E_default.pkl"
    if not path.exists():
        print(f"FEHLT: {path}. Erst 01_prepare_items.py ausfuehren.")
        raise SystemExit(1)
    with open(path, "rb") as f:
        return pickle.load(f)["items"]


def gold_lookup(gold_ids: set) -> dict:
    """UUID -> (NAME, OBJEKTART) aus der DuckDB features-Tabelle (config1)."""
    import duckdb

    con = duckdb.connect(str(C.duckdb_path("config1")), read_only=True)
    rows = con.execute("SELECT UUID, NAME, OBJEKTART FROM features").fetchall()
    con.close()
    return {u: (n, o) for u, n, o in rows if u in gold_ids}


def main():
    ap = argparse.ArgumentParser(description="No-Candidate-Fehleranalyse")
    ap.add_argument("--n-examples", type=int, default=10)
    ap.add_argument("--context", type=int, default=150, help="Kontextfenster in Zeichen")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    C.ensure_dirs()
    items = load_items()
    documents = E.load_documents()
    text_by_doc = E.text_index(documents)

    no_cand = [it for it in items if not it["candidate_ids"]]
    print(f"{len(no_cand)}/{len(items)} Items ohne Gazetteer-Kandidat "
          f"({len(no_cand) / len(items):.1%})")

    # Oberflaechenform + Gold-Info pro Item.
    surface = lambda it: text_by_doc[it["doc_id"]][it["start"]:it["end"]]
    gold_info = gold_lookup({it["gold_id"] for it in no_cand if it["gold_id"]})

    counts = Counter(surface(it) for it in no_cand)
    by_surface = defaultdict(list)
    for it in no_cand:
        by_surface[surface(it)].append(it)

    lines = [
        "# No-Candidate-Fehleranalyse",
        "",
        f"Gold-Toponyme, fuer die `gazetteer.search()` keine Kandidaten liefert: "
        f"**{len(no_cand)}** von {len(items)} ({len(no_cand) / len(items):.1%}); "
        f"{len(counts)} eindeutige Oberflaechenformen.",
        "",
        f"## Top-{args.top} Oberflaechenformen",
        "",
        "Gegenueberstellung Oberflaechenform ↔ Gazetteer-NAME des Gold-Eintrags "
        "(haeufigster Gold-Eintrag der Form), um Nicht-Match-Muster zu erkennen.",
        "",
        "| # | Oberflaechenform | n | Gold-NAME | OBJEKTART |",
        "|---|---|---|---|---|",
    ]
    for i, (text, n) in enumerate(counts.most_common(args.top), 1):
        gid = Counter(it["gold_id"] for it in by_surface[text]).most_common(1)[0][0]
        name, art = gold_info.get(gid, ("(nicht in features-Tabelle)", "?"))
        lines.append(f"| {i} | {text} | {n} | {name} | {art} |")

    # Diverse Beispiele: eindeutige Oberflaechenformen, moeglichst verschiedene Docs.
    rng = random.Random(C.SEED)
    pool = list(by_surface)
    rng.shuffle(pool)
    picked, used_docs = [], set()
    for text in pool:
        cands = [it for it in by_surface[text] if it["doc_id"] not in used_docs]
        it = cands[0] if cands else by_surface[text][0]
        picked.append(it)
        used_docs.add(it["doc_id"])
        if len(picked) >= args.n_examples:
            break

    lines += ["", f"## {len(picked)} Beispiele (Seed {C.SEED})", ""]
    for i, it in enumerate(picked, 1):
        doc_text = text_by_doc[it["doc_id"]]
        s, e = it["start"], it["end"]
        a, b = max(0, s - args.context), min(len(doc_text), e + args.context)
        ctx = (doc_text[a:s] + "**" + doc_text[s:e] + "**" + doc_text[e:b])
        ctx = " ".join(ctx.split())  # Zeilenumbrueche/Mehrfach-Spaces glaetten
        name, art = gold_info.get(it["gold_id"], ("(nicht in features-Tabelle)", "?"))
        lines += [
            f"### Beispiel {i}: „{doc_text[s:e]}“",
            "",
            f"- **Dokument**: `{it['doc_id']}`, Offsets {s}–{e}",
            f"- **Gold-Eintrag**: NAME=„{name}“, OBJEKTART={art}, UUID=`{it['gold_id']}`",
            f"- **Kontext**: …{ctx}…",
            "",
        ]

    out = C.LOGS_DIR / "no_candidate_examples.md"   # enthaelt Korpusauszuege, nicht veroeffentlichen
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Gespeichert: {out}")


if __name__ == "__main__":
    main()
