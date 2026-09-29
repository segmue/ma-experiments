"""
Stage 09 — Gepaarte McNemar-Tests fuer die Hauptvergleiche (Nachtrag 2026-09-04).

07_mcnemar_ec1.py deckt nur M3/M4/M5 auf E_c1 ab. Der Vergleich, der die
Kernaussage des Kapitels traegt — faire Baseline M3/E_default gegen das beste
System M5/E_c1 — war ungetestet, ebenso alle Tests auf der ambigen Teilmenge
(der Leitmetrik). Dieses Skript holt beides nach.

Grundlage: results/matrix/per_item/{MODEL}_{RESOLVER}.pkl.gz (alle 15 Zellen).
Kein Modell-Forward noetig, reine Auswertung.

Test: exakter McNemar (zweiseitiger Binomialtest auf den diskordanten Paaren,
p = 0.5). Ohne scipy implementiert, Ergebnis identisch zu binomtest(...).

Output: results/mcnemar_main.json
"""
from __future__ import annotations

import gzip
import json
import math
import pickle
import config as C

PER_ITEM = C.PER_ITEM_DIR
OUT = C.RESULTS_DIR / "mcnemar_main.json"

PAIRS = [
    # (Modell A, Resolver A, Modell B, Resolver B, Bezeichnung)
    ("M5_spatial_config2", "E_c1", "M3_default_finetuned", "E_default", "bestes System gegen faire Baseline"),
    ("M4_spatial_config1", "E_c1", "M3_default_finetuned", "E_default", "Diagonale config1 gegen faire Baseline"),
    ("M5_spatial_config2", "E_c2", "M3_default_finetuned", "E_default", "Diagonale config2 gegen faire Baseline"),
    ("M3_default_finetuned", "E_c1", "M3_default_finetuned", "E_default", "Beschreibungseffekt bei fixem Modell"),
    ("M5_spatial_config2", "E_c1", "M4_spatial_config1", "E_c1", "bestes System gegen Diagonale config1"),
    ("M5_spatial_config2", "E_c1", "M3_default_finetuned", "E_c1", "bestes System gegen M3 auf E_c1"),
    ("M4_spatial_config1", "E_c1", "M3_default_finetuned", "E_c1", "M4 gegen M3 auf E_c1 (Reproduktion 07)"),
    ("M5_spatial_config2", "E_c1", "M5_spatial_config2", "E_c2", "Kreuzkombination gegen eigene Diagonale"),
]


def load(model: str, resolver: str) -> list[dict]:
    with gzip.open(PER_ITEM / f"{model}_{resolver}.pkl.gz", "rb") as fh:
        return pickle.load(fh)


def key(it: dict) -> tuple:
    return (it["doc_id"], it["start"], it["end"], it["gold_id"])


def correct(it: dict) -> bool:
    return it.get("rank") == 1


def exact_mcnemar_p(b: int, c: int) -> float:
    """Zweiseitiger exakter Binomialtest auf den diskordanten Paaren (p0 = 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # Log-Raum, weil die Binomialkoeffizienten bei n in der Groessenordnung 10^3
    # den float-Bereich sprengen.
    logs = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        + n * math.log(0.5)
        for i in range(k + 1)
    ]
    m = max(logs)
    tail = math.exp(m) * sum(math.exp(x - m) for x in logs)
    return min(1.0, 2.0 * tail)


def compare(a_items: list[dict], b_items: list[dict], subset: str) -> dict:
    a_map = {key(it): it for it in a_items}
    b_map = {key(it): it for it in b_items}
    shared = a_map.keys() & b_map.keys()
    only_a = only_b = both = neither = 0
    for k in shared:
        ia, ib = a_map[k], b_map[k]
        if subset == "ambiguous" and (ia.get("n_candidates") or 0) <= 1:
            continue
        ca, cb = correct(ia), correct(ib)
        if ca and cb:
            both += 1
        elif ca:
            only_a += 1
        elif cb:
            only_b += 1
        else:
            neither += 1
    n = both + only_a + only_b + neither
    return {
        "subset": subset,
        "n_items": n,
        "acc_a": round((both + only_a) / n, 4) if n else None,
        "acc_b": round((both + only_b) / n, 4) if n else None,
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "discordant": only_a + only_b,
        "mcnemar_p": exact_mcnemar_p(only_a, only_b),
    }


def main() -> None:
    cache: dict[tuple[str, str], list[dict]] = {}

    def get(model: str, resolver: str) -> list[dict]:
        if (model, resolver) not in cache:
            cache[(model, resolver)] = load(model, resolver)
        return cache[(model, resolver)]

    out = {
        "_meta": {
            "date": "2026-09-04",
            "script": "09_mcnemar_main.py",
            "method": ("Exakter McNemar-Test (zweiseitiger Binomialtest auf den diskordanten "
                       "Paaren). Gepaart ueber (doc_id, start, end, gold_id); identische Items "
                       "und Kandidaten in allen Zellen."),
            "subsets": {
                "all": "alle Gold-Toponyme, Items ohne Kandidat zaehlen als Fehler",
                "ambiguous": "nur Items mit mehr als einem Gazetteer-Kandidaten (Leitmetrik)",
            },
        },
        "pairs": [],
    }

    for ma, ra, mb, rb, label in PAIRS:
        a_items, b_items = get(ma, ra), get(mb, rb)
        entry = {
            "label": label,
            "system_a": f"{ma}/{ra}",
            "system_b": f"{mb}/{rb}",
            "all": compare(a_items, b_items, "all"),
            "ambiguous": compare(a_items, b_items, "ambiguous"),
        }
        out["pairs"].append(entry)
        for sub in ("all", "ambiguous"):
            r = entry[sub]
            print(f"{label:45s} [{sub:9s}] "
                  f"{r['acc_a']:.4f} vs {r['acc_b']:.4f}  "
                  f"{r['only_a_correct']:5d}/{r['only_b_correct']:5d} "
                  f"von {r['discordant']:5d}  p = {r['mcnemar_p']:.3g}")

    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
