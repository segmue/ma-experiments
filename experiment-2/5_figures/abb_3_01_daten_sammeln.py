"""Abbildung 3 der Arbeit (Abb. 4.3, Assoziationsmasse) — Schritt 1: Daten zusammensuchen.

Sammelt fuer das Leitbeispiel Saentis (Alpiner Gipfel) aus echten Daten:

  1. die beiden erzeugten Beschreibungssaetze
       B1-Satz = Variante E_c1 (Ueberlagerungsmass, config1)
       D1-Satz = Variante E_d1 (Adjazenzmass)
     und daraus die genannten Kontextobjekte (dynamische Slots, Block "bei ...").
     Die festen Slots (Block "in ...", Verwaltungshierarchie) werden bewusst NICHT
     uebernommen, weil sie unabhaengig vom Mass in jedem Satz stehen.
  2. fuer jedes Kontextobjekt seine Zellen in nativer Aufloesung (Feld a) und seine
     Repraesentantenzellen auf der Ankerstufe r* = 10 (Feld b), exakt nach der Regel
     der D1-Rechnung (Plugin: geoparser_h3_resolver/association/d1.py):
         Stufe > r*  -> Elternzelle,  Stufe = r* -> Zelle selbst,
         Stufe < r*  -> ein Mittelpunktskind je Ursprungszelle.
  3. die Ringsuche von der Repraesentantenzelle des Saentis aus (d = 0..D), mit
     Rangbildung wie in der D1-Rechnung (dense rank ueber die
     kleinste Ringdistanz je Zielobjekt, Quellobjekt ausgeschlossen, Rang <= k),
     ueber ALLE Objekte des Gazetteers, damit die Raenge stimmen.
  4. ein Gegenbeispiel fuer Feld a: ein benachbarter Gipfel ohne gemeinsame Zelle.
  5. die Matrixwerte B1 und D1 fuer die beteiligten Kategorienpaare (beide Richtungen).

Wichtig fuer die Lesart: Auch im D1-Satz werden die Instanzen ueber UEBERLAGERUNG
gefunden (Generator-Modus "overlap", wie bei E_c1). D1 entscheidet
nur, welche Kategorien genannt werden. Feld b zeigt, wie dieselben Objekte bei der
SCHAETZUNG von D1 vertreten sind.

Eingabe: ma-experiments/output/config1/spatial_h3.duckdb (+ b1_matrix.csv, d1_matrix.csv daneben),
         3_evaluation/cache/descriptions_E_c1.pkl und descriptions_E_d1.pkl (01_prepare_items.py)
Ausgabe: output/abb_3_daten/abb_4_3_daten.json, kontextobjekte.csv, ringe.csv
Aufruf:  python3 abb_3_01_daten_sammeln.py        (Pfade ueber Umgebungsvariablen ueberschreibbar)
Abhaengigkeiten: duckdb, h3 (4.x), pandas
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path

import duckdb
import h3
import pandas as pd

# --------------------------------------------------------------------------- Pfade
HIER = Path(__file__).resolve().parent
EXP2 = HIER.parent                                       # .../experiment-2
REPO = EXP2.parent                                       # .../ma-experiments
DB = Path(os.environ.get("MA_GAZETTEER_DB", REPO / "output/config1/spatial_h3.duckdb"))
B1_MATRIX = Path(os.environ.get("MA_B1_MATRIX", DB.parent / "b1_matrix.csv"))
D1_MATRIX = Path(os.environ.get("MA_D1_MATRIX", DB.parent / "d1_matrix.csv"))
SATZ_B1 = EXP2 / "3_evaluation/cache/descriptions_E_c1.pkl"
SATZ_D1 = EXP2 / "3_evaluation/cache/descriptions_E_d1.pkl"
OUT = HIER / "output" / "abb_3_daten"

# --------------------------------------------------------------------------- Parameter
QUELLE = ("Säntis", "Alpiner Gipfel")
GEGENBEISPIEL = ("Girenspitz", "Gipfel")   # Nachbargipfel ohne gemeinsame Zelle
R_STERN = 10          # Ankerauflösung D1
D_MAX = 12            # Suchradius D1 in Ringschritten
K = 10                # zugelassene Ränge D1
AUSSCHNITT_KM = 2.6   # Radius, innerhalb dessen Zellen für die Karten exportiert werden


# --------------------------------------------------------------------------- Hilfen
def s(c: int) -> str:
    return h3.int_to_str(int(c))


def i(c: str) -> int:
    return h3.str_to_int(c)


def repraesentant(cell: int) -> int:
    """Regel aus 05_sample_index.py."""
    r = h3.get_resolution(s(cell))
    if r > R_STERN:
        return i(h3.cell_to_parent(s(cell), R_STERN))
    if r < R_STERN:
        return i(h3.cell_to_center_child(s(cell), R_STERN))
    return int(cell)


def ueberlagert(zellen: list[int], res: int, quellzelle: int) -> bool:
    """Überlagerung wie in der Engine: Vergleich auf der gröberen Stufe."""
    rq = h3.get_resolution(s(quellzelle))
    if res <= rq:
        return i(h3.cell_to_parent(s(quellzelle), res)) in set(zellen)
    return any(i(h3.cell_to_parent(s(z), rq)) == quellzelle for z in zellen)


def kontext_aus_satz(satz: str, kategorien: set[str]) -> list[tuple[str, str]]:
    """Liest den Block 'bei ...' (dynamische Slots) als Liste (Name, Kategorie)."""
    m = re.search(r", bei (.*?), in ", satz)
    if not m:
        return []
    token = m.group(1).split(", ")
    paare, namen = [], []
    for t in token:
        if t in kategorien:
            for n in namen:
                paare.append((n, t))
            namen = []
        else:
            namen.extend(x.strip() for x in t.split(" und "))
    if namen:
        raise ValueError(f"Namen ohne Kategorie am Blockende: {namen}")
    return paare


# --------------------------------------------------------------------------- Hauptteil
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB), read_only=True)
    kategorien = {r[0] for r in con.execute("SELECT DISTINCT OBJEKTART FROM features").fetchall()}

    def objekt(fid: int) -> dict:
        r = con.execute("SELECT feature_id, UUID, NAME, OBJEKTART, h3_cells, h3_resolution, "
                        "h3_cell_count FROM features WHERE feature_id = ?", [fid]).fetchone()
        return dict(feature_id=r[0], uuid=r[1], name=r[2], kategorie=r[3],
                    zellen=[int(x) for x in r[4]], stufe=int(r[5]), zellzahl=int(r[6]))

    # 1. Quelle ----------------------------------------------------------------------
    fids = con.execute("SELECT feature_id FROM features WHERE NAME=? AND OBJEKTART=?",
                       list(QUELLE)).fetchall()
    assert len(fids) == 1, f"Quelle nicht eindeutig: {fids}"
    quelle = objekt(fids[0][0])
    assert len(quelle["zellen"]) == 1, "Quelle ist kein Punkt"
    q13 = quelle["zellen"][0]
    q10 = repraesentant(q13)
    lat0, lng0 = h3.cell_to_latlng(s(q10))

    satz = {"B1": pickle.load(open(SATZ_B1, "rb"))[quelle["uuid"]],
            "D1": pickle.load(open(SATZ_D1, "rb"))[quelle["uuid"]]}

    # 2. Kontextobjekte je Satz, per Name+Kategorie+Überlagerung aufgelöst -------------
    kontext = {}
    for mass, text in satz.items():
        liste = []
        for name, kat in kontext_aus_satz(text, kategorien):
            kand = [objekt(r[0]) for r in con.execute(
                "SELECT feature_id FROM features WHERE NAME=? AND OBJEKTART=?", [name, kat]).fetchall()]
            treffer = [o for o in kand if ueberlagert(o["zellen"], o["stufe"], q13)]
            if len(treffer) != 1:
                print(f"  WARNUNG {mass}: '{name}' ({kat}) -> {len(treffer)} überlagernde Objekte")
            for o in treffer:
                liste.append(o["feature_id"])
        kontext[mass] = liste
        print(f"[{mass}] {text}\n      -> {len(liste)} Kontextobjekte")

    # 3. Ringsuche D1 über den ganzen Index --------------------------------------------
    dist = {q10: 0}
    for d in range(1, D_MAX + 1):
        for c in h3.grid_ring(s(q10), d):
            dist[i(c)] = d
    con.execute("CREATE TEMP TABLE disk(cell UBIGINT, d INTEGER)")
    con.executemany("INSERT INTO disk VALUES (?, ?)", list(dist.items()))
    maske = 0xF << 52
    eltern10 = (f"((l.cell & ~{maske}::UBIGINT) | ({R_STERN << 52}::UBIGINT) "
                f"| ({(1 << (3 * (15 - R_STERN))) - 1}::UBIGINT))")
    treffer = con.execute(f"""
        SELECT l.feature_id, min(dk.d) FROM h3_lookup l JOIN disk dk ON {eltern10} = dk.cell
        WHERE l.cell_res > {R_STERN} GROUP BY 1
        UNION ALL
        SELECT l.feature_id, min(dk.d) FROM h3_lookup l JOIN disk dk ON l.cell = dk.cell
        WHERE l.cell_res = {R_STERN} GROUP BY 1""").fetchall()
    for r in range(0, R_STERN):
        kand = {i(h3.cell_to_parent(s(c), r)) for c in dist}
        if not kand:
            continue
        for fid, cell in con.execute(
                f"SELECT feature_id, cell FROM h3_lookup WHERE cell_res={r} AND cell IN "
                f"({','.join(map(str, kand))})").fetchall():
            cc = i(h3.cell_to_center_child(s(cell), R_STERN))
            if cc in dist:
                treffer.append((fid, dist[cc]))
    naechste = {}
    for fid, d in treffer:
        naechste[fid] = min(d, naechste.get(fid, 99))
    naechste.pop(quelle["feature_id"], None)
    besetzt = sorted(set(naechste.values()))
    rang_von_d = {d: n + 1 for n, d in enumerate(besetzt)}
    m_r = defaultdict(int)
    for d in naechste.values():
        m_r[d] += 1

    # 4. Gegenbeispiel -----------------------------------------------------------------
    # gleichnamige Objekte gibt es mehrfach in der Schweiz -> das der Quelle nächste nehmen
    gkand = [objekt(r[0]) for r in con.execute(
        "SELECT feature_id FROM features WHERE NAME=? AND OBJEKTART=?", list(GEGENBEISPIEL)).fetchall()]
    gegen = min(gkand, key=lambda o: h3.great_circle_distance(
        (lat0, lng0), h3.cell_to_latlng(s(o["zellen"][0])), "km"))
    assert not ueberlagert(gegen["zellen"], gegen["stufe"], q13), "Gegenbeispiel überlagert doch"

    # 5. Matrixwerte --------------------------------------------------------------------
    b1 = pd.read_csv(B1_MATRIX, sep=";", index_col=0)
    d1 = pd.read_csv(D1_MATRIX, sep=";", index_col=0)

    # --- Export ------------------------------------------------------------------------
    def nahe(c: int) -> bool:
        return h3.great_circle_distance((lat0, lng0), h3.cell_to_latlng(s(c)), "km") <= AUSSCHNITT_KM

    alle_fids = sorted(set(kontext["B1"]) | set(kontext["D1"]))
    objekte = {}
    zeilen = []
    for fid in alle_fids + [gegen["feature_id"]]:
        o = objekt(fid) if fid != gegen["feature_id"] else gegen
        reps = sorted({repraesentant(c) for c in o["zellen"]})
        d = naechste.get(fid)
        eintrag = dict(
            name=o["name"], kategorie=o["kategorie"], stufe=o["stufe"], zellzahl=o["zellzahl"],
            im_satz_B1=fid in kontext["B1"], im_satz_D1=fid in kontext["D1"],
            gegenbeispiel=fid == gegen["feature_id"],
            zellen=[s(c) for c in o["zellen"] if nahe(c)],
            repraesentanten=[s(c) for c in reps if nahe(c)],
            n_repraesentanten=len(reps),
            ringdistanz=d, rang=rang_von_d.get(d) if d is not None else None,
            gewicht=(1 / m_r[d]) if d is not None and rang_von_d[d] <= K else 0.0,
            B1_quelle_nach_obj=float(b1.loc[QUELLE[1], o["kategorie"]]),
            B1_obj_nach_quelle=float(b1.loc[o["kategorie"], QUELLE[1]]),
            D1_quelle_nach_obj=float(d1.loc[QUELLE[1], o["kategorie"]]),
            D1_obj_nach_quelle=float(d1.loc[o["kategorie"], QUELLE[1]]),
        )
        objekte[str(fid)] = eintrag
        zeilen.append({k: v for k, v in eintrag.items() if k not in ("zellen", "repraesentanten")}
                      | {"feature_id": fid})

    ringe = [dict(d=d, besetzt=d in rang_von_d, rang=rang_von_d.get(d),
                  mitglieder=m_r.get(d, 0), in_k=(d in rang_von_d and rang_von_d[d] <= K))
             for d in range(D_MAX + 1)]

    json.dump(dict(
        quelle=dict(name=QUELLE[0], kategorie=QUELLE[1], zelle13=s(q13), zelle10=s(q10)),
        parameter=dict(r_stern=R_STERN, d_max=D_MAX, k=K, ausschnitt_km=AUSSCHNITT_KM,
                       db=str(DB), satz_b1=str(SATZ_B1), satz_d1=str(SATZ_D1)),
        saetze=satz, kontext={k: [str(x) for x in v] for k, v in kontext.items()},
        objekte=objekte, ringe=ringe,
        ringzellen={s(c): d for c, d in dist.items()},
    ), open(OUT / "abb_4_3_daten.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    with open(OUT / "kontextobjekte.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(zeilen[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(zeilen)
    with open(OUT / "ringe.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ringe[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(ringe)

    print("\nKontextobjekte:")
    for z in zeilen:
        print(f"  {z['name']:22s} {z['kategorie']:16s} Stufe {z['stufe']:2d}  "
              f"B1-Satz {int(z['im_satz_B1'])}  D1-Satz {int(z['im_satz_D1'])}  "
              f"Rep. {z['n_repraesentanten']:4d}  d={z['ringdistanz']}  Rang={z['rang']}  "
              f"w={z['gewicht']:.3f}")
    print("Ringe:", [(r["d"], r["rang"], r["mitglieder"]) for r in ringe])
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
