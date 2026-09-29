"""
Regressions-Smoke nach den Library-Optimierungen (V1/V2/V3/V6 vom 11.6. Abend).

Berechnet die Stage-01-Items fuer 1 Dokument auf E_c1 FRISCH (ohne persistenten
Description-Cache, d.h. der komplette Such- + Beschreibungs-Pfad laeuft durch die
geaenderte Engine) und vergleicht sie feldweise mit den Items des vollen Laufs
(cache/items_E_c1.pkl). Gefordert: identisch.
"""

from __future__ import annotations

import pickle

import config as C
import eval_core as E

DOC_IDX = 0

with open(C.CACHE_DIR / "items_E_c1.pkl", "rb") as f:
    full = pickle.load(f)

docs = E.load_documents()
doc = docs[DOC_IDX]
doc_id = doc["filename"]
ref_items = [it for it in full["items"] if it["doc_id"] == doc_id]
print(f"Dokument: {doc_id}, {len(ref_items)} Referenz-Items aus dem Voll-Lauf")

resolver = E.build_resolver(C.MODELS["M1_dguzh"]["weights"], "spatial", "config1")
new_items = E.prepare_items(resolver, [doc])  # bewusst OHNE cache_key -> frisch

assert len(new_items) == len(ref_items), (
    f"{len(new_items)} neue vs {len(ref_items)} Referenz-Items"
)

mismatches = 0
for a, b in zip(new_items, ref_items):
    if a != b:
        mismatches += 1
        for k in a:
            if a[k] != b[k]:
                print(f"MISMATCH {a['doc_id']}@{a['start']} Feld '{k}':")
                print(f"  neu: {a[k]}")
                print(f"  ref: {b[k]}")

print(f"\nVerglichen: {len(ref_items)} Items, Mismatches: {mismatches}")
raise SystemExit(0 if mismatches == 0 else 1)
