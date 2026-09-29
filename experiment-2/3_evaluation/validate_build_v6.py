"""
Validierung des V6-Bulk-Inserts in build._create_duckdb.

Baut eine Mini-DuckDB aus einem synthetischen Feature-DataFrame (inkl.
Hex-String- und Int-Zellen, None-Werten, leerer Zellliste) und prueft den
Tabelleninhalt Zeile fuer Zeile gegen die Erwartung — also exakt das, was der
alte Row-wise-Insert geschrieben haette.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import duckdb
import pandas as pd

from geoparser_h3_resolver.pipeline.build import _create_duckdb

df = pd.DataFrame([
    {"UUID": "u1", "NAME": "Alpha", "OBJEKTART": "Gipfel", "source": "PKT",
     "h3_cells": ["8a1fa6d6da97fff", "8a1fa6d6da87fff"], "h3_resolution": 10,
     "h3_cell_count": 2},
    {"UUID": "u2", "NAME": "Beta", "OBJEKTART": "Gemeindegebiet", "source": "TLM",
     "h3_cells": [622204039496499199], "h3_resolution": 8, "h3_cell_count": 1},
    {"UUID": None, "NAME": None, "OBJEKTART": "Strasse", "source": "TLM",
     "h3_cells": [], "h3_resolution": 13, "h3_cell_count": 0},
])

expected = [
    (0, "u1", "Alpha", "Gipfel", "PKT",
     [int("8a1fa6d6da97fff", 16), int("8a1fa6d6da87fff", 16)], 10, 2),
    (1, "u2", "Beta", "Gemeindegebiet", "TLM", [622204039496499199], 8, 1),
    (2, None, None, "Strasse", "TLM", [], 13, 0),
]

with tempfile.TemporaryDirectory() as tmp:
    db = Path(tmp) / "test.duckdb"
    _create_duckdb(db, df.copy())

    conn = duckdb.connect(str(db), read_only=True)
    rows = conn.execute(
        "SELECT feature_id, UUID, NAME, OBJEKTART, source, h3_cells, "
        "h3_resolution, h3_cell_count FROM features ORDER BY feature_id"
    ).fetchall()
    lookup = conn.execute(
        "SELECT feature_id, cell, cell_res FROM h3_lookup ORDER BY feature_id, cell"
    ).fetchall()
    conn.close()

assert len(rows) == len(expected), f"{len(rows)} statt {len(expected)} Zeilen"
for got, exp in zip(rows, expected):
    assert tuple(got) == exp, f"Zeile weicht ab:\n  got: {got}\n  exp: {exp}"

exp_lookup = sorted(
    [(0, int("8a1fa6d6da97fff", 16), 10), (0, int("8a1fa6d6da87fff", 16), 10),
     (1, 622204039496499199, 8)]
)
assert sorted(lookup) == exp_lookup, f"h3_lookup weicht ab: {lookup}"

print("V6 Bulk-Insert OK: features + h3_lookup inhaltlich identisch zur Erwartung")
