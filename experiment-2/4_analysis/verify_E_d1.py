"""
Verifikation des Laufs E_d1 — die vierte Beschreibungsvariante auf der korrigierten
Adjazenzmatrix D1, samt Schwellwert- und Kappungsachse. Nach dem Vorbild von
verify_arm_M.py (M-Arm) und verify_kontrollarm.py (Kontrollarm M2/K2), mit vier
Aenderungen gegenueber jenen:

  * Der Matrixpfad ist ein Parameter, kein fester Name. Die korrigierte Matrix entsteht
    unter einem neuen Dateinamen (T-141), und das Skript weigert sich, die alte
    npmi_dist_matrix_B.csv als "neue" Matrix zu akzeptieren.
  * Die Schrankenpruefung: Zeilensumme der Gewichtsmatrix <= 10 mal Objektzahl der
    Quellkategorie. Das ist der Test, der den behobenen Defekt nachweist.
  * Die Abdeckungspruefung: der Neulauf materialisiert die Quellzellen einmal als
    sortierte Liste, und die Teillaeufe nehmen zusammenhaengende, disjunkte Bereiche
    daraus. M4 prueft, dass diese Bereiche die Liste luecken- und ueberschneidungsfrei
    kacheln und dass alle Teile dieselbe Liste benutzt haben.
  * "Nichts ueberschrieben" entscheidet nach **Hash gegen ein vorher geschriebenes
    Manifest**, nicht nach Aenderungsdatum. Die Vorgaengerfassung hat wegen des Datums
    einen Fehlalarm ausgeloest (Kontrollarm, ERGEBNIS.md Abschnitt 8 Punkt 2).

Benennung: **B1** ist das Ueberlagerungsmass, **D1** das Adjazenzmass. Die vom
Generator genannten Partner heissen **Kontextobjekte**.

Vier Stufen, in dieser Reihenfolge auszufuehren:

  --stufe manifest   VOR jedem Lauf. Schreibt die Hashes aller bestehenden Ergebnis-
                     dateien nach ERGEBNIS_E_d1_manifest.json. Ohne diese Datei kann
                     Z6 spaeter nichts beweisen.
  --stufe matrix     Nach dem Neulauf der Matrix, VOR dem Beschreibungsbau.
                     M1 Name und Hash, M2 Form, M3 Schranke, M4 Abdeckung,
                     M5 Eichungskennzahlen, M6 Gegenprobe gegen die alte Matrix.
  --stufe caches     Nach dem Beschreibungs- und Item-Bau (Mac), VOR Stage 02.
                     C0 bis C6, darunter C2: b1_c6 muss zeichengleich zu Arm K2 vom
                     16.09.2026 sein — die Gleichheitskontrolle zwischen Mac und
                     Windows ueber alle 69'165 Kandidaten.
  --stufe zellen     Nach Stage 02 (Windows). Z1 bis Z8.

Repo-Layout (ma-experiments/experiment-2): --ma ist der Ordner experiment-2
(Default: der Elternordner dieses Skripts). Beschreibungs- und Item-Caches liegen in
3_evaluation/cache/ (erzeugt von 01_prepare_items.py), Zell-JSONs in results/matrix/,
per-Item-Dumps in 3_evaluation/cache/per_item/. Die Zwischenprodukte der D1-Rechnung
(pool.parquet, W_cellwise*.npy, cellwise_meta*.json) legt das Plugin nicht ab; Stufe
`matrix` braucht sie ueber --d1-dir bzw. --pool/--w/--meta.

Aufrufe (Beispiele):

    python verify_E_d1.py --stufe manifest
    python verify_E_d1.py --stufe matrix   --ma ... --matrix .../npmi_dist_matrix_B_nopart.csv
    python verify_E_d1.py --stufe caches   --ma ... --matrix ... --paket .../E-d1_Lauf
    python verify_E_d1.py --stufe zellen   --ma ... --paket .../E-d1_Lauf

Stufe `matrix` braucht pandas und numpy und liest pool.parquet ueber pyarrow oder
duckdb. Alle anderen Stufen laufen mit der Standardbibliothek; scipy wird benutzt,
falls vorhanden, sonst rechnet das Skript den McNemar-p-Wert exakt selbst.
Exit-Code 0 nur, wenn alle Pruefungen der gewaehlten Stufe bestehen.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import pickle
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ── Sollwerte aus dem Referenzlauf (Windows, 08.07.2026 und 12.09.2026) ────────────
# Invarianten: in allen 15 Zellen von results/matrix/ identisch nachgeprueft.
SOLL = {
    "total": 75741,
    "no_candidates": 8159,
    "ambiguous_total": 23848,
    "unambiguous_total": 43734,
    "gold_not_in_candidates": 555,
    "correct_unambiguous": 43544,
    "n_descriptions": 69165,
    "n_used_uuids": 4610,
}

# Referenzzellen M{1..5} x E_c1, volle Stellenzahl aus results/matrix/*.json.
# Sie sind die Reproduktionskontrolle: wird E_c1 im selben Lauf mitgerechnet, muss
# jede dieser Zahlen zeichengleich wiederkommen.
REF_E_C1 = {
    "M1_dguzh": {
        "accuracy_at_1": 0.7595225835412789, "accuracy_at_3": 0.8497775313238537,
        "mrr": 0.8084674425185816, "document_accuracy": 0.7111271102504036,
        "acc1_ambiguous_only": 0.5863384770211338},
    "M2_distiluse_base": {
        "accuracy_at_1": 0.80148136412247, "accuracy_at_3": 0.8616205225703384,
        "mrr": 0.8334899407996672, "document_accuracy": 0.7744813313622869,
        "acc1_ambiguous_only": 0.7195991278094599},
    "M3_default_finetuned": {
        "accuracy_at_1": 0.8367198743084987, "accuracy_at_3": 0.8712718342773399,
        "mrr": 0.8550467123111407, "document_accuracy": 0.815560552552162,
        "acc1_ambiguous_only": 0.8315162697081516},
    "M4_spatial_config1": {
        "accuracy_at_1": 0.8368783089740035, "accuracy_at_3": 0.8691329662930249,
        "mrr": 0.8538524894024887, "document_accuracy": 0.8190476132446055,
        "acc1_ambiguous_only": 0.832019456558202},
    "M5_spatial_config2": {
        "accuracy_at_1": 0.840297857171149, "accuracy_at_3": 0.8709681678351223,
        "mrr": 0.8568043009158341, "document_accuracy": 0.8217381947649542,
        "acc1_ambiguous_only": 0.842879906071788},
}

SHA_DESC_E_C1 = "0e2de7554faea41e86087f2d4c256a481ba38d42ea0b4f18bdc0d957a207f8d0"
SHA_ITEMS_E_C1 = "9d0fb7020daee3aa16c8e4a9dfde2b5f340eff25a03e61e70d64b68d189588ba"
SHA_MATRIX_ALT = "398e8337657eef98f165f02d28e9c1b11a36001aa06bcc503549a0ea84b50353"
# Das Ueberlagerungsmass B1, unveraendert seit dem Bauzeitlauf vom 07.05.2026.
SHA_B1_MATRIX = "eb302d35015cf403598daee259fb441dd99f9c4f964e9ea69d1efac2912445e3"
NAME_MATRIX_ALT = "npmi_dist_matrix_B.csv"
# Die korrigierte Matrix vom 17.09.2026. Kein Zwang — wer sie neu rechnet, bekommt
# einen anderen Hash —, aber M1 meldet, ob es diese ist.
SHA_MATRIX_D1_17_09 = "627b94e7762d21af9bb20a555eaa90f74b9fe1c61627e78b00a59de1e811ad29"

# Kennzahlen des Stichprobenindex der D1-Rechnung (Lauf vom 17.09.2026).
# Quelle: README des Experiments, Abschnitt "Daten und Ergebnisse"; am 17.09.2026
# gegen sample_index.parquet nachgerechnet.
SAMPLE_INDEX_ZEILEN = 3994376
SAMPLE_INDEX_ZELLEN = 2099115
N_OBJEKTE_POOL = 442001
K_RAENGE = 10          # 06_neighbours_cellwise.py, --k, Default 10

# SHA256 ueber die aufsteigend sortierte Liste der verschiedenen Quellzellen, als
# uint64-Bytes (little endian). Am 17.09.2026 aus 1_data/sample_index.parquet
# gerechnet. Der Neulauf muss genau diese Liste materialisieren.
SHA_SRC_CELLS = "c7f5da828dd04985e4b6d00eced91f2273d69774101d54e7097e155ef02bda17"

# Das Gitter dieses Laufs: Mass x Kappung, dazu die Schwellwertachse auf D1 x 10.
# Schluessel = Armname in 05b_build_caches_E-d1.py.
ARME = {
    "d1":     dict(resolver="E_d1",     mass="D1", schwelle=0.001, kappung=10),
    "d1_s01": dict(resolver="E_d1_s01", mass="D1", schwelle=0.01,  kappung=10),
    "d1_s1":  dict(resolver="E_d1_s1",  mass="D1", schwelle=0.1,   kappung=10),
    "d1_c6":  dict(resolver="E_d1_c6",  mass="D1", schwelle=0.001, kappung=6),
    "d1_c20": dict(resolver="E_d1_c20", mass="D1", schwelle=0.001, kappung=20),
    "b1_c6":  dict(resolver="E_b1_c6",  mass="B1", schwelle=0.001, kappung=6),
    "b1_c20": dict(resolver="E_b1_c20", mass="B1", schwelle=0.001, kappung=20),
}

# Auf welchen Modellen welcher Arm gerechnet wird. Die Hauptspalte D1 x 10 laeuft ueber
# alle fuenf Modelle, die beiden Achsen ueber M5 und M4 — dieselbe Modellauswahl wie die
# bestehende B1-Reihe (results/ablations.json fuehrt M5, ablations_M4_backup.json M4).
ALLE_MODELLE = ["M1_dguzh", "M2_distiluse_base", "M3_default_finetuned",
                "M4_spatial_config1", "M5_spatial_config2"]
ACHSEN_MODELLE = ["M5_spatial_config2", "M4_spatial_config1"]
MODELLE_JE_ARM = {a: (ALLE_MODELLE if a == "d1" else ACHSEN_MODELLE) for a in ARME}

# Arm K2 vom 16.09.2026 ist B1 mit max_categories = 6 und damit dieselbe Belegung wie
# der neue Arm b1_c6 (kontrollarm_rueckgabe/meta_E_K2.json, params.max_categories 6).
# B1 ist von der Matrixkorrektur nicht beruehrt, also muessen die beiden
# Beschreibungscaches zeichengleich sein — quer ueber Mac und Windows.
SHA_DESCRIPTIONS_E_K2 = "a5b15d630f04080f12eeb1c1b3795f6b8b888c35561c2213ddb364194c5c3088"
REF_K2 = {  # M5_spatial_config2 x E_K2, volle Stellenzahl
    "accuracy_at_1": 0.834831861211233, "accuracy_at_3": 0.8701363858412221,
    "mrr": 0.8536370891120918, "document_accuracy": 0.8185503667231394,
    "acc1_ambiguous_only": 0.825519959745052,
}

BESTEHENDE_ARME = ["E_c1", "E_c2", "E_default", "E_K", "E_Z", "E_P", "E_D",
                   "E_prop", "E_c3", "E_c4", "E_c5", "E_M", "E_M2", "E_K2"]


# ── Helfer ──────────────────────────────────────────────────────────────────────
def find_ma(explizit) -> Path:
    if explizit:
        return Path(explizit).resolve()
    base = HERE.parent                      # experiment-2
    if (base / "3_evaluation").is_dir():
        return base
    raise SystemExit(
        "Ordner experiment-2 nicht gefunden. Bitte --ma angeben (der Ordner, der "
        "3_evaluation und results enthaelt).")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pruefung(report: dict, key: str, ok: bool, **details) -> bool:
    report[key] = {"ok": bool(ok), **details}
    text = json.dumps(details, ensure_ascii=False, default=str)
    print(f"  {'ok  ' if ok else 'FAIL'} {key}" + ("" if not details else "  " + text[:400]))
    return bool(ok)


def bericht(report: dict, key: str, **details) -> None:
    """Kennzahl ohne Bestehen/Durchfallen — sie gehoert in den Bericht, nicht ins Urteil."""
    report[key] = {"nur_bericht": True, **details}
    print(f"  ..   {key}  " + json.dumps(details, ensure_ascii=False, default=str)[:400])


def binom_p_zweiseitig(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest
        return float(binomtest(k, n, 0.5).pvalue)
    except Exception:
        pass
    from math import comb
    m = min(k, n - k)
    s = sum(comb(n, i) for i in range(0, m + 1))
    return min(1.0, 2.0 * s / (2.0 ** n))


def lade_pickle(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)


def desc_pfad(ma: Path, arm: str) -> Path | None:
    """descriptions_E_<arm>.pkl im Eval-Cache (01_prepare_items.py, DescriptionCache)."""
    for p in (ma / "3_evaluation" / "cache" / f"descriptions_E_{arm}.pkl",):
        if p.exists():
            return p
    return None


def meta_pfad(ma: Path, arm: str) -> Path:
    return ma / "3_evaluation" / "cache" / f"meta_E_{arm}.json"


def benutzte_uuids(ma: Path) -> set:
    """Die 4'610 Kandidaten, die in der Gold-Auswertung tatsaechlich vorkommen."""
    p = ma / "3_evaluation" / "cache" / "items_E_c1.pkl"
    if not p.exists():
        return set()
    out = set()
    for it in lade_pickle(p)["items"]:
        out.update(it["candidate_ids"])
    return out


# ══════════════════════════════════════════════════════════════════════════════════
# Stufe manifest — Hashes vor dem Lauf
# ══════════════════════════════════════════════════════════════════════════════════
def manifest_dateien(ma: Path):
    ev = ma / "3_evaluation"
    ziele = []
    ziele += sorted((ma / "results" / "matrix").glob("*.json"))
    ziele += sorted((ev / "cache" / "per_item").glob("*.pkl.gz"))
    ziele += sorted((ma / "results").glob("summary.csv*"))
    ziele += sorted((ma / "results").glob("ablations*.json"))
    for name in ("items_E_c1.pkl", "descriptions_E_c1.pkl"):
        p = ev / "cache" / name
        if p.exists():
            ziele.append(p)
    return [z for z in ziele if z.is_file()]


def stufe_manifest(ma: Path, out: Path) -> bool:
    eintraege = {}
    for p in manifest_dateien(ma):
        rel = str(p.relative_to(ma)).replace("\\", "/")
        eintraege[rel] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    daten = {
        "zeitpunkt": time.strftime("%Y-%m-%d %H:%M:%S"),
        "projektwurzel": str(ma),
        "n_dateien": len(eintraege),
        "dateien": eintraege,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(daten, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  {len(eintraege)} Dateien gehasht -> {out}")
    fehlt = [n for n in ("results/summary.csv",
                         "3_evaluation/cache/items_E_c1.pkl")
             if n not in eintraege]
    if fehlt:
        print(f"  WARNUNG: erwartete Datei nicht im Manifest: {fehlt}")
    return not fehlt


# ══════════════════════════════════════════════════════════════════════════════════
# Stufe matrix — M1 bis M6
# ══════════════════════════════════════════════════════════════════════════════════
def m1_name_und_hash(matrix: Path, report: dict) -> bool:
    if not matrix.exists():
        return pruefung(report, "M1_matrix_datei", False, fehler=f"fehlt: {matrix}")
    h = sha256(matrix)
    neuer_name = matrix.name != NAME_MATRIX_ALT
    neuer_inhalt = h != SHA_MATRIX_ALT
    ok = neuer_name and neuer_inhalt
    return pruefung(
        report, "M1_matrix_datei", ok,
        pfad=str(matrix), name=matrix.name, sha256=h, bytes=matrix.stat().st_size,
        name_verschieden_von_alt=neuer_name, inhalt_verschieden_von_alt=neuer_inhalt,
        ist_die_matrix_vom_17_09_2026=(h == SHA_MATRIX_D1_17_09),
        hinweis=("Die korrigierte Matrix muss einen neuen Dateinamen UND einen neuen "
                 "Inhalt haben. Gleicher Name heisst, dass die Metadateien der Arme "
                 "spaeter nicht mehr unterscheidbar sind, welche Matrix gelaufen ist."))


def m2_form(matrix: Path, alt_matrix: Path | None, report: dict) -> bool:
    try:
        import numpy as np
        import pandas as pd
    except ImportError as e:
        return pruefung(report, "M2_form", False, fehler=f"pandas/numpy fehlt: {e}")
    df = pd.read_csv(matrix, sep=";", index_col=0)
    v = df.to_numpy(dtype=float)
    quadratisch = df.shape[0] == df.shape[1] == 110
    idx_gleich_spalten = list(df.index) == list(df.columns)
    endlich = bool(np.isfinite(v).all())
    im_bereich = bool(v.min() >= -1.0 - 1e-12 and v.max() <= 1.0 + 1e-12)
    gleiche_kategorien = True
    if alt_matrix and alt_matrix.exists():
        alt = pd.read_csv(alt_matrix, sep=";", index_col=0)
        gleiche_kategorien = list(alt.index) == list(df.index)
    ok = quadratisch and idx_gleich_spalten and endlich and im_bereich and gleiche_kategorien
    return pruefung(
        report, "M2_form", ok, shape=list(df.shape), quadratisch_110=quadratisch,
        index_gleich_spalten=idx_gleich_spalten, alle_werte_endlich=endlich,
        bereich=[float(v.min()), float(v.max())], bereich_in_minus1_bis_1=im_bereich,
        kategorienfolge_wie_alte_matrix=gleiche_kategorien,
        diagonale_median=float(np.median(np.diag(v))))


def _lade_pool(pool: Path):
    """OBJEKTART -> Objektzahl, aus pool.parquet. pyarrow, sonst duckdb, sonst pandas."""
    try:
        import pyarrow.parquet as pq
        t = pq.read_table(pool, columns=["OBJEKTART"])
        aus = {}
        for v in t.column("OBJEKTART").to_pylist():
            aus[v] = aus.get(v, 0) + 1
        return aus, "pyarrow"
    except Exception:
        pass
    try:
        import duckdb
        rows = duckdb.connect().execute(
            f"SELECT OBJEKTART, count(*) AS n FROM read_parquet('{pool.as_posix()}') "
            "GROUP BY 1").fetchall()
        return {a: int(n) for a, n in rows}, "duckdb"
    except Exception:
        pass
    import pandas as pd
    s = pd.read_parquet(pool, columns=["OBJEKTART"])["OBJEKTART"].value_counts()
    return {k: int(v) for k, v in s.items()}, "pandas"


def m3_schranke(w_dateien, cats_meta: Path, pool: Path, report: dict) -> bool:
    """Die harte Schranke. Jedes Quellobjekt gibt hoechstens Gewicht k=10 ab:

    06_neighbours_cellwise.py Zeile 79 verteilt je (Quellobjekt, Quellzelle, Rang) genau
    Gewicht 1 auf die Mitglieder des Rangs (1/count ueber der Partition), Zeile 78 teilt
    das durch die Zellzahl m des Quellobjekts, und Zeile 74 laesst hoechstens k Raenge zu.
    Summiert ueber die m Zellen eines Objekts sind das hoechstens k*m/m = k = 10.
    Auf Kategorieebene: Zeilensumme von W <= 10 mal Objektzahl der Quellkategorie.

    Genau das verletzten am 17.09.2026 40 der 110 Kategorien im Lauf vom 04.09.2026,
    angefuehrt von "Zollamt eingeschraenkt" mit Faktor 1.667.
    """
    try:
        import numpy as np
    except ImportError as e:
        return pruefung(report, "M3_schranke", False, fehler=f"numpy fehlt: {e}")
    if not w_dateien:
        return pruefung(report, "M3_schranke", False,
                        fehler="keine W-Datei angegeben (--w oder --w-glob)")
    if not cats_meta.exists():
        return pruefung(report, "M3_schranke", False, fehler=f"fehlt: {cats_meta}")
    if not pool.exists():
        return pruefung(report, "M3_schranke", False, fehler=f"fehlt: {pool}")

    cats = json.loads(cats_meta.read_text(encoding="utf-8"))["cats"]
    W = None
    leer = []
    for p in w_dateien:
        a = np.load(p)
        if a.sum() == 0:
            leer.append(Path(p).name)
        W = a if W is None else W + a
    if W.shape != (len(cats), len(cats)):
        return pruefung(report, "M3_schranke", False,
                        fehler=f"W hat Form {W.shape}, erwartet {(len(cats), len(cats))}")

    zahl_je_kat, quelle = _lade_pool(pool)
    fehlende = [c for c in cats if c not in zahl_je_kat]
    if fehlende:
        return pruefung(report, "M3_schranke", False,
                        fehler=f"Kategorien nicht im Pool: {fehlende[:5]}")
    n = np.array([zahl_je_kat[c] for c in cats], dtype=float)
    zeilensumme = W.sum(axis=1)
    schranke = K_RAENGE * n
    verhaeltnis = zeilensumme / np.maximum(schranke, 1e-12)
    verletzt = verhaeltnis > 1.0 + 1e-9
    schlimmste = sorted(
        ({"kategorie": cats[i], "zeilensumme": float(zeilensumme[i]),
          "objekte": int(n[i]), "schranke": float(schranke[i]),
          "verhaeltnis": round(float(verhaeltnis[i]), 4)}
         for i in range(len(cats))),
        key=lambda d: -d["verhaeltnis"])[:10]
    ok = int(verletzt.sum()) == 0 and not leer
    return pruefung(
        report, "M3_schranke", ok,
        n_w_dateien=len(w_dateien), leere_w_dateien=leer,
        pool_gelesen_mit=quelle, objekte_gesamt=int(n.sum()),
        W_total=float(W.sum()), W_total_obergrenze=float(K_RAENGE * n.sum()),
        n_verletzungen=int(verletzt.sum()), n_kategorien=len(cats),
        max_verhaeltnis=round(float(verhaeltnis.max()), 4),
        median_verhaeltnis=round(float(np.median(verhaeltnis)), 4),
        zehn_hoechste=schlimmste,
        hinweis=("Null Verletzungen ist Pflicht. Der Test faengt Doppelzaehlung, nicht "
                 "Auslassung — siehe M4 fuer die Abdeckung."))


def _bereich(meta: dict):
    """Liest den Quellzellenbereich eines Teillaufs aus der Metadatei.

    Der korrigierte 06_neighbours_cellwise.py materialisiert die Quellzellen EINMAL als
    aufsteigend sortierte Liste und gibt jedem Teillauf einen zusammenhaengenden
    Bereich [von, bis) daraus. Die Metadatei muss diesen Bereich nennen, sonst ist die
    Disjunktheit nicht pruefbar. Akzeptierte Schreibweisen, in dieser Reihenfolge:

        "src_range": [von, bis]          bevorzugt
        "src_from" / "src_to"
        "src_von"  / "src_bis"
    """
    r = meta.get("src_range")
    if isinstance(r, (list, tuple)) and len(r) == 2:
        return int(r[0]), int(r[1])
    for a, b in (("src_from", "src_to"), ("src_von", "src_bis")):
        if a in meta and b in meta:
            return int(meta[a]), int(meta[b])
    return None


def _src_total(meta: dict):
    for k in ("src_total", "src_cells_total", "n_src_cells_total"):
        if k in meta:
            return int(meta[k])
    return None


def _src_sha(meta: dict):
    for k in ("src_sha256", "src_cells_sha256", "src_list_sha256"):
        if k in meta:
            return str(meta[k])
    return None


def m4_abdeckung(metas, report: dict, src_cells=None) -> bool:
    """Abdeckung der Quellzellen — der Gegentest zur Schranke.

    M3 faengt Doppelzaehlung, M4 faengt Auslassung. Der fehlerhafte Lauf vom
    04.09.2026 zeigt, warum M4 scharf sein muss: dort summierten sich die acht Felder
    n_src_cells auf exakt 2'099'115, obwohl sich die Teile ueberlappten UND zugleich
    Zellen auslassen. Die Summe allein beweist nichts.

    Beweiskraeftig ist erst dreierlei zusammen:
      1. jeder Teillauf nennt dieselbe Gesamtlaenge der materialisierten Liste,
      2. jeder Teillauf nennt denselben Hash dieser Liste,
      3. die Bereiche [von, bis) kacheln [0, Gesamtlaenge) luecken- und
         ueberschneidungsfrei.
    """
    if not metas:
        return pruefung(report, "M4_abdeckung", False,
                        fehler="keine cellwise_meta-Datei angegeben (--meta oder --meta-glob)")
    teile, raenge = [], [0] * 11
    for q in metas:
        d = json.loads(Path(q).read_text(encoding="utf-8"))
        for i, v in enumerate(d.get("rank_fill_hist", [])):
            if i < len(raenge):
                raenge[i] += int(v)
        teile.append({
            "datei": Path(q).name,
            "n_src_cells": int(d.get("n_src_cells", -1)),
            "bereich": _bereich(d), "src_total": _src_total(d), "src_sha256": _src_sha(d),
            "W_total": d.get("W_total"), "seconds": d.get("seconds"),
            "k": d.get("k"), "d_max": d.get("d_max"),
        })
    paare = sum(raenge)
    fehler = []

    # 1 und 2: alle Teile beziehen sich auf dieselbe Liste
    totals = {t["src_total"] for t in teile}
    shas = {t["src_sha256"] for t in teile}
    if totals == {None}:
        fehler.append("kein Teillauf nennt src_total — der korrigierte "
                      "06_neighbours_cellwise.py muss die Laenge der materialisierten "
                      "Zellliste in jede Metadatei schreiben")
    elif len(totals) > 1:
        fehler.append(f"die Teillaeufe nennen verschiedene src_total: {sorted(totals)}")
    elif totals != {SAMPLE_INDEX_ZELLEN}:
        fehler.append(f"src_total ist {totals.pop()} statt {SAMPLE_INDEX_ZELLEN}")

    if shas == {None}:
        fehler.append("kein Teillauf nennt src_sha256 — ohne ihn ist nicht gezeigt, dass "
                      "alle Teile dieselbe materialisierte Liste benutzt haben")
    elif len(shas) > 1:
        fehler.append(f"die Teillaeufe nennen verschiedene src_sha256: {sorted(shas)}")
    elif shas != {SHA_SRC_CELLS}:
        fehler.append(f"src_sha256 ist {shas.pop()} statt {SHA_SRC_CELLS} — die "
                      "materialisierte Liste ist nicht die Menge der verschiedenen "
                      "Zellen aus sample_index.parquet")

    # 3: die Bereiche kacheln [0, N)
    bereiche = [t["bereich"] for t in teile if t["bereich"] is not None]
    kachelung = None
    if len(bereiche) != len(teile):
        fehler.append("mindestens ein Teillauf nennt keinen Bereich (src_range) — die "
                      "Disjunktheit ist damit nicht pruefbar")
    else:
        b = sorted(bereiche)
        luecken, ueberlappungen = [], []
        if b[0][0] != 0:
            luecken.append([0, b[0][0]])
        for (v1, s1), (v2, s2) in zip(b, b[1:]):
            if s1 < v2:
                luecken.append([s1, v2])
            elif s1 > v2:
                ueberlappungen.append([v2, s1])
        ziel = SAMPLE_INDEX_ZELLEN if totals == {None} else (sorted(totals)[-1] or 0)
        if b[-1][1] != ziel:
            luecken.append([b[-1][1], ziel])
        abgedeckt = sum(bis - von for von, bis in b)
        kachelung = {"bereiche": b, "abgedeckt": abgedeckt,
                     "luecken": luecken, "ueberlappungen": ueberlappungen}
        if luecken:
            fehler.append(f"Luecken in der Abdeckung: {luecken[:5]}")
        if ueberlappungen:
            fehler.append(f"Ueberlappungen zwischen Teillaeufen: {ueberlappungen[:5]}")
        for t in teile:
            von, bis = t["bereich"]
            if t["n_src_cells"] not in (-1, bis - von):
                fehler.append(f"{t['datei']}: n_src_cells {t['n_src_cells']} passt nicht "
                              f"zum Bereich [{von}, {bis})")

    # Gegenprobe auf der Datei selbst, falls sie mitgegeben wurde
    liste = None
    if src_cells:
        liste = _pruefe_src_cells(Path(src_cells))
        if liste.get("fehler"):
            fehler.append("materialisierte Zellliste: " + liste["fehler"])

    if paare > SAMPLE_INDEX_ZEILEN:
        fehler.append(f"mehr Quellpaare mit Treffer ({paare}) als Zeilen in "
                      f"sample_index.parquet ({SAMPLE_INDEX_ZEILEN}) — Doppelzaehlung")

    return pruefung(
        report, "M4_abdeckung", not fehler, n_metas=len(metas),
        summe_n_src_cells=sum(t["n_src_cells"] for t in teile if t["n_src_cells"] >= 0),
        soll_n_src_cells=SAMPLE_INDEX_ZELLEN, kachelung=kachelung,
        summe_quell_paare_mit_treffer=paare, obergrenze_quell_paare=SAMPLE_INDEX_ZEILEN,
        anteil_volle_zehn_raenge=(round(raenge[10] / paare, 4) if paare else None),
        materialisierte_liste=liste, einzelteile=teile, fehler=fehler,
        hinweis=("Die blosse Summe der n_src_cells beweist nichts: im fehlerhaften Lauf "
                 "vom 04.09.2026 stimmte sie exakt. Beweiskraeftig sind gleicher "
                 "src_sha256 in allen Teilen und eine luecken- und "
                 "ueberschneidungsfreie Kachelung."))


def _pruefe_src_cells(pfad: Path) -> dict:
    """Prueft die materialisierte Zellliste direkt, falls der Neulauf sie ablegt.

    Akzeptiert .npy (uint64) und .parquet/.csv mit einer Spalte `cell`.
    """
    if not pfad.exists():
        return {"fehler": f"nicht gefunden: {pfad}"}
    try:
        import numpy as np
        if pfad.suffix == ".npy":
            a = np.load(pfad).astype(np.uint64)
        else:
            import duckdb
            a = (duckdb.connect()
                 .execute(f"SELECT cell FROM read_parquet('{pfad.as_posix()}')"
                          if pfad.suffix == ".parquet" else
                          f"SELECT cell FROM read_csv_auto('{pfad.as_posix()}')")
                 .fetch_arrow_table().column("cell").to_numpy().astype(np.uint64))
    except Exception as e:  # pragma: no cover
        return {"fehler": f"nicht lesbar: {e}"}
    aufsteigend = bool((np.diff(a) > 0).all()) if a.size > 1 else True
    h = hashlib.sha256(a.tobytes()).hexdigest()
    aus = {"datei": str(pfad), "n": int(a.size), "streng_aufsteigend": aufsteigend,
           "sha256": h, "sha256_soll": SHA_SRC_CELLS}
    if a.size != SAMPLE_INDEX_ZELLEN:
        aus["fehler"] = f"{a.size} Zellen statt {SAMPLE_INDEX_ZELLEN}"
    elif not aufsteigend:
        aus["fehler"] = "nicht streng aufsteigend sortiert — dann ist die Zerlegung in "\
                        "Bereiche nicht reproduzierbar"
    elif h != SHA_SRC_CELLS:
        aus["fehler"] = f"sha256 {h} statt {SHA_SRC_CELLS}"
    return aus


def _top_liste(df, kategorie, schwelle, max_kat=10):
    """Was get_associated_categories liefert: Diagonale weg, >= Schwelle, absteigend
    sortiert, auf max_categories gekappt (association_loader.py Zeilen 89-100)."""
    r = df.loc[kategorie].drop(kategorie).astype(float)
    s = r[r >= schwelle].sort_values(ascending=False)
    return list(s.index[:max_kat]), int(len(s))


def m5_eichung(matrix: Path, b1: Path | None, report: dict,
               schwellen=(0.001, 0.01, 0.1), kappungen=(6, 10, 20)) -> None:
    """Die Eichungskennzahlen, aus der Matrix allein — ohne einen einzigen Lauf.

    Sie sind der Kern der ERWARTUNG: welche Zeilen ihre geordnete Zehnerliste beim
    Schwellwertwechsel ueberhaupt aendern koennen. Aendert eine Zeile ihre Liste nicht,
    aendert sich keine Beschreibung eines Kandidaten dieser Kategorie.
    """
    try:
        import pandas as pd  # noqa: F401
    except ImportError as e:
        bericht(report, "M5_eichung", fehler=f"pandas fehlt: {e}")
        return
    import pandas as pd
    aus = {}
    quellen = {"D1_neu": matrix}
    if b1 and Path(b1).exists():
        quellen["B1"] = Path(b1)
    for name, pfad in quellen.items():
        df = pd.read_csv(pfad, sep=";", index_col=0)
        basis = {i: _top_liste(df, i, schwellen[0])[0] for i in df.index}
        block = {"datei": Path(pfad).name}
        for t in schwellen:
            listen, zahlen = {}, []
            for i in df.index:
                l, nz = _top_liste(df, i, t)
                listen[i] = l
                zahlen.append(nz)
            geaendert = [i for i in df.index if listen[i] != basis[i]]
            kurz = [i for i in df.index if len(listen[i]) < 10]
            leer = [i for i in df.index if len(listen[i]) == 0]
            block[str(t)] = {
                "kategorien_je_zeile_mittel": round(sum(zahlen) / len(zahlen), 2),
                "kategorien_je_zeile_median": float(sorted(zahlen)[len(zahlen) // 2]),
                "zeilen_mit_geaenderter_zehnerliste": len(geaendert),
                "zeilen_unter_zehn_kategorien": len(kurz),
                "zeilen_ohne_kategorie": len(leer),
                "geaenderte_zeilen": geaendert[:20],
                "Flurname_swisstopo_gewaehlt": listen.get("Flurname swisstopo", []),
            }
        # Kappungsachse, bei fester Schwelle 0.001. Massgeblich ist, wie viele Zeilen
        # ihre geordnete Liste gegenueber der Kappung 10 aendern — nur Kandidaten
        # solcher Quellkategorien koennen eine andere Beschreibung bekommen.
        basis10 = {i: _top_liste(df, i, schwellen[0], 10)[0] for i in df.index}
        kblock = {}
        for c in kappungen:
            listen = {i: _top_liste(df, i, schwellen[0], c)[0] for i in df.index}
            laengen = [len(listen[i]) for i in df.index]
            geaendert = [i for i in df.index if listen[i] != basis10[i]]
            kblock[str(c)] = {
                "mittlere_listenlaenge": round(sum(laengen) / len(laengen), 2),
                "zeilen_gekappt": sum(1 for i in df.index
                                      if _top_liste(df, i, schwellen[0], 10**6)[1] > c),
                "zeilen_mit_geaenderter_liste_gegen_kappung_10": len(geaendert),
                "Flurname_swisstopo_n_kategorien": len(listen.get("Flurname swisstopo", [])),
            }
        block["kappungsachse_bei_schwelle_0.001"] = kblock
        aus[name] = block
    # Jaccard der gewaehlten Zehnermengen zwischen den beiden Massen bei 0.001
    if "B1" in quellen:
        d = pd.read_csv(matrix, sep=";", index_col=0)
        b = pd.read_csv(quellen["B1"], sep=";", index_col=0)
        js, wechsel = [], 0
        for i in d.index:
            A = set(_top_liste(b, i, schwellen[0])[0])
            B = set(_top_liste(d, i, schwellen[0])[0])
            js.append(len(A & B) / len(A | B) if (A | B) else 1.0)
            a1 = _top_liste(b, i, schwellen[0])[0][:1]
            b1_ = _top_liste(d, i, schwellen[0])[0][:1]
            wechsel += int(a1 != b1_)
        aus["B1_gegen_D1_neu"] = {
            "jaccard_der_zehnermengen_mittel": round(sum(js) / len(js), 3),
            "staerkster_partner_weicht_ab_in_zeilen": wechsel,
        }
    bericht(report, "M5_eichung", **aus)


def m6_alt_gegen_neu(matrix: Path, alt: Path | None, report: dict) -> None:
    if not alt or not Path(alt).exists():
        bericht(report, "M6_alt_gegen_neu", hinweis="alte Matrix nicht angegeben")
        return
    try:
        import numpy as np
        import pandas as pd
    except ImportError as e:
        bericht(report, "M6_alt_gegen_neu", fehler=f"pandas/numpy fehlt: {e}")
        return
    a = pd.read_csv(alt, sep=";", index_col=0)
    n = pd.read_csv(matrix, sep=";", index_col=0)
    if list(a.index) != list(n.index):
        bericht(report, "M6_alt_gegen_neu", fehler="Kategorienfolge verschieden")
        return
    va, vn = a.to_numpy(float), n.to_numpy(float)
    off = ~np.eye(len(a), dtype=bool)
    js, wechsel, gleich = [], 0, 0
    for i in a.index:
        A = set(_top_liste(a, i, 0.001)[0])
        B = set(_top_liste(n, i, 0.001)[0])
        js.append(len(A & B) / len(A | B) if (A | B) else 1.0)
        ta = _top_liste(a, i, 0.001)[0]
        tn = _top_liste(n, i, 0.001)[0]
        wechsel += int(ta[:1] != tn[:1])
        gleich += int(ta == tn)
    bericht(report, "M6_alt_gegen_neu",
            mittlere_absolute_differenz=round(float(np.abs(va[off] - vn[off]).mean()), 6),
            groesste_absolute_differenz=round(float(np.abs(va - vn).max()), 6),
            jaccard_der_zehnermengen_mittel=round(sum(js) / len(js), 3),
            zeilen_mit_unveraenderter_zehnerliste=gleich,
            staerkster_partner_wechselt_in_zeilen=wechsel,
            hinweis=("Rein beschreibend. Eine kleine Differenz heisst, dass der Defekt "
                     "die Matrix nur wenig bewegt hat — nicht, dass er unwichtig war."))


# ══════════════════════════════════════════════════════════════════════════════════
# Stufe caches — C1 bis C6
# ══════════════════════════════════════════════════════════════════════════════════
def c0_armspec(paket: Path, ma: Path, report: dict) -> bool:
    """Statische Pruefung des Skripts: die drei Arme muessen reuse=True, mode=overlap
    und alloc=greedy tragen. reuse=True ist der Schalter, der ueber build_arm_K in den
    produktiven BatchSentenceGenerator fuehrt — nur dort gilt die feature_id-Ordnung des
    Batch-Query, also derselbe Tie-Break wie Arm K und wie descriptions_E_c1.pkl.
    Mit reuse=False traegt der Arm den Tie-Break nach Objektflaeche (descgen.py Zeilen
    44-46) und misst nicht mehr den Matrixtausch allein."""
    kandidaten = [paket / "05b_build_caches_E-d1.py"]
    pfad = next((p for p in kandidaten if p.exists()), None)
    if pfad is None:
        return pruefung(report, "C0_armspec", False, fehler=f"nicht gefunden: {kandidaten}")
    baum = ast.parse(pfad.read_text(encoding="utf-8"))
    spec = None
    for knoten in baum.body:
        if isinstance(knoten, ast.Assign) and \
           any(getattr(t, "id", None) == "ARM_SPEC" for t in knoten.targets):
            spec = {}
            for k, v in zip(knoten.value.keys, knoten.value.values):
                spec[k.value] = {kw.arg: ast.unparse(kw.value) for kw in v.keywords}
    if spec is None:
        return pruefung(report, "C0_armspec", False, fehler=f"ARM_SPEC fehlt in {pfad}")
    ist, fehler = {}, []
    for arm, d in ARME.items():
        e = spec.get(arm)
        ist[arm] = e
        if e is None:
            fehler.append(f"{arm} fehlt")
            continue
        if e.get("reuse") != "True":
            fehler.append(f"{arm}: reuse != True")
        if e.get("mode") != "'overlap'":
            fehler.append(f"{arm}: mode != overlap")
        if e.get("alloc") != "'greedy'":
            fehler.append(f"{arm}: alloc != greedy")
        # D1-Arme duerfen keinen festen Matrixnamen tragen, B1-Arme muessen einen tragen.
        if d["mass"] == "D1" and e.get("matrix") != "None":
            fehler.append(f"{arm}: matrix ist fest verdrahtet ({e.get('matrix')}) — "
                          "die korrigierte D1-Matrix kommt aus --matrix")
        if d["mass"] == "B1" and e.get("matrix") != "C.B1_MATRIX":
            fehler.append(f"{arm}: matrix {e.get('matrix')} statt C.B1_MATRIX")
        if e.get("assoc_threshold") != repr(d["schwelle"]):
            fehler.append(f"{arm}: assoc_threshold {e.get('assoc_threshold')} "
                          f"statt {d['schwelle']}")
        soll_kappung = None if d["kappung"] == 10 else repr(d["kappung"])
        if e.get("max_categories") != soll_kappung:
            fehler.append(f"{arm}: max_categories {e.get('max_categories')} statt "
                          f"{soll_kappung} (10 kommt aus config1.yaml Zeile 30)")
    k_unveraendert = spec.get("K") == {
        "matrix": "C.B1_MATRIX", "mode": "'overlap'", "alloc": "'greedy'", "reuse": "True"}
    if not k_unveraendert:
        fehler.append("Arm K ist nicht mehr der unveraenderte Referenzarm")
    return pruefung(report, "C0_armspec", not fehler, skript=str(pfad),
                    sha256=sha256(pfad), arme=ist, arm_K=spec.get("K"), fehler=fehler)


def c1_c2_caches(ma: Path, matrix: Path, report: dict) -> bool:
    ev = ma / "3_evaluation"
    ref_p = ev / "cache" / "descriptions_E_c1.pkl"
    if not ref_p.exists():
        return pruefung(report, "C1_beschreibungscaches", False, fehler=f"fehlt: {ref_p}")
    ref = lade_pickle(ref_p)
    matrix_sha = sha256(matrix) if matrix.exists() else None
    ok_alles, details = True, {}
    for arm, dd in ARME.items():
        th, kappung, mass = dd["schwelle"], dd["kappung"], dd["mass"]
        p = desc_pfad(ma, arm)
        m = meta_pfad(ma, arm)
        if p is None or not m.exists():
            ok_alles = False
            details[arm] = {"fehler": f"descriptions oder meta fehlt (desc={p}, meta={m})"}
            continue
        got = lade_pickle(p)
        meta = json.loads(m.read_text(encoding="utf-8"))
        par = meta.get("params", {})
        stats = meta.get("stats_A3", {})
        fehler = []
        if len(got) != SOLL["n_descriptions"]:
            fehler.append(f"{len(got)} Schluessel statt {SOLL['n_descriptions']}")
        if set(got) != set(ref):
            fehler.append("Schluesselmenge weicht von descriptions_E_c1.pkl ab")
        if meta.get("mode") != "overlap":
            fehler.append(f"mode={meta.get('mode')}")
        if meta.get("alloc") != "greedy":
            fehler.append(f"alloc={meta.get('alloc')}")
        if meta.get("n_uuids") != SOLL["n_descriptions"]:
            fehler.append(f"n_uuids={meta.get('n_uuids')}")
        if stats.get("stop_stage_hist") != {"0": SOLL["n_descriptions"]}:
            fehler.append(f"stop_stage_hist={stats.get('stop_stage_hist')}")
        if abs(float(par.get("assoc_threshold", -1)) - th) > 1e-12:
            fehler.append(f"assoc_threshold={par.get('assoc_threshold')} statt {th}")
        for feld, soll in (("max_slots", 10), ("max_slots_per_category", 5),
                           ("max_categories", kappung), ("max_filler_slots", 0)):
            if par.get(feld) != soll:
                fehler.append(f"{feld}={par.get(feld)} statt {soll}")
        if mass == "D1":
            if meta.get("matrix") == NAME_MATRIX_ALT:
                fehler.append("meta fuehrt die ALTE Matrix npmi_dist_matrix_B.csv")
            if matrix_sha and meta.get("matrix_sha256") not in (None, matrix_sha):
                fehler.append(f"matrix_sha256 in meta ({str(meta.get('matrix_sha256'))[:12]}…) "
                              f"passt nicht zur uebergebenen D1-Matrix ({matrix_sha[:12]}…)")
        else:
            if meta.get("matrix") != "b1_matrix.csv":
                fehler.append(f"B1-Arm, aber meta fuehrt {meta.get('matrix')}")
            if meta.get("matrix_sha256") not in (None, SHA_B1_MATRIX):
                fehler.append(f"b1_matrix.csv hat den Hash {meta.get('matrix_sha256')} "
                              f"statt {SHA_B1_MATRIX}")
        if meta.get("matrix_sha256") is None:
            fehler.append("meta fuehrt kein matrix_sha256 — altes 05_build_caches.py benutzt?")
        details[arm] = {
            "mass": mass, "kappung": kappung, "schwelle": th,
            "descriptions": str(p), "sha256_descriptions": sha256(p),
            "n_schluessel": len(got), "meta": meta, "fehler": fehler,
        }
        ok_alles = ok_alles and not fehler
    return pruefung(report, "C1_beschreibungscaches", ok_alles,
                    sha256_descriptions_E_c1=sha256(ref_p),
                    sha256_descriptions_E_c1_erwartet=SHA_DESC_E_C1, arme=details)


def c2_mac_gegen_windows(ma: Path, paket: Path, report: dict) -> bool:
    """Die staerkste verfuegbare Gleichheitskontrolle zwischen den beiden Maschinen.

    Arm K2 (Windows, 16.09.2026) ist B1 mit max_categories = 6
    (kontrollarm_rueckgabe/meta_E_K2.json, params.max_categories 6). Der neue Arm b1_c6
    ist dieselbe Belegung, gerechnet auf dem Mac. B1 ist von der Matrixkorrektur nicht
    beruehrt, denn der Defekt sitzt in 06_neighbours_cellwise.py und betrifft nur D1.
    Die beiden Beschreibungscaches muessen deshalb **byteweise gleich** sein.

    Besteht dieser Block, ist ueber alle 69'165 Kandidaten gezeigt, dass Mac und
    Windows aus derselben Gazetteer-Datenbank und demselben Generator dieselben
    Beschreibungen erzeugen — eine staerkere Aussage als jeder Dateihash der DuckDB,
    und sie prueft die ganze Kette statt nur ihren Anfang.
    """
    p = desc_pfad(ma, "b1_c6")
    ref = paket / "descriptions_E_K2_ALT_referenz.pkl"
    if p is None:
        return pruefung(report, "C2_mac_gegen_windows", False,
                        fehler="descriptions_E_b1_c6.pkl fehlt")
    h_neu = sha256(p)
    details = {"b1_c6": str(p), "sha256_b1_c6": h_neu,
               "sha256_arm_K2_windows": SHA_DESCRIPTIONS_E_K2}
    if h_neu == SHA_DESCRIPTIONS_E_K2:
        details["befund"] = ("byteweise gleich — Mac und Windows erzeugen fuer B1 x 6 "
                             "denselben Beschreibungscache ueber alle 69'165 Kandidaten")
        return pruefung(report, "C2_mac_gegen_windows", True, **details)
    # Ungleich: sagen, WIE weit auseinander, statt nur zu scheitern.
    if ref.exists():
        a, b = lade_pickle(ref), lade_pickle(p)
        benutzt = benutzte_uuids(ma)
        details["gleiche_schluesselmenge"] = set(a) == set(b)
        details["abweichende_beschreibungen_gesamt"] = sum(1 for u in a if b.get(u) != a[u])
        details["abweichende_beschreibungen_benutzt"] = sum(
            1 for u in benutzt if b.get(u) != a.get(u))
        beispiele = [u for u in list(a)[:5000] if b.get(u) != a[u]][:2]
        details["beispiele"] = [{"uuid": u, "windows": a[u], "mac": b.get(u)}
                                for u in beispiele]
    details["deutung"] = (
        "Mac und Windows erzeugen fuer dieselbe Belegung verschiedene Beschreibungen. "
        "Moegliche Ursachen in dieser Reihenfolge: verschiedene Gazetteer-Datenbank "
        "(fingerprint_gazetteer.py vergleichen), verschiedene b1_matrix.csv, "
        "verschiedene Fassung des Generators oder der Engine. Bevor irgendetwas "
        "berichtet wird, ist das zu klaeren — es beruehrt auch alle bestehenden Zahlen.")
    return pruefung(report, "C2_mac_gegen_windows", False, **details)


def c3_neue_matrix_kam_an(ma: Path, paket: Path, report: dict) -> bool:
    """Der entscheidende Test des ganzen Pakets: ist die KORRIGIERTE Matrix in den
    Beschreibungen angekommen, oder steht dort noch der Stand von Arm M?"""
    ev = ma / "3_evaluation"
    ref_p = ev / "cache" / "descriptions_E_c1.pkl"
    alt_p = paket / "descriptions_E_M_ALT_referenz.pkl"
    d1_p = desc_pfad(ma, "d1")
    if d1_p is None or not ref_p.exists():
        return pruefung(report, "C3_neue_matrix_kam_an", False,
                        fehler=f"fehlt: {d1_p if d1_p is None else ref_p}")
    ref, d1 = lade_pickle(ref_p), lade_pickle(d1_p)
    benutzt = benutzte_uuids(ma)
    gegen_c1 = sum(1 for u in benutzt if d1.get(u) != ref.get(u))
    gegen_c1_gesamt = sum(1 for u in ref if d1.get(u) != ref[u])
    details = {
        "n_benutzte_uuids": len(benutzt),
        "geaendert_gegen_E_c1_benutzt": gegen_c1,
        "geaendert_gegen_E_c1_gesamt": gegen_c1_gesamt,
    }
    ok = gegen_c1 > 0
    if alt_p.exists():
        alt = lade_pickle(alt_p)
        gegen_m = sum(1 for u in benutzt if d1.get(u) != alt.get(u))
        gegen_m_gesamt = sum(1 for u in ref if d1.get(u) != alt.get(u))
        details["geaendert_gegen_arm_M_alt_benutzt"] = gegen_m
        details["geaendert_gegen_arm_M_alt_gesamt"] = gegen_m_gesamt
        details["hinweis"] = (
            "geaendert_gegen_arm_M_alt muss GROESSER NULL sein. Arm M vom 16.09.2026 lief "
            "auf der alten, fehlerhaften Matrix. Null Abweichung hiesse, dass der "
            "Beschreibungsbau wieder die alte Matrix gelesen hat. Erwartet werden "
            "MEHRERE TAUSEND der 4'610 benutzten Kandidaten: 106 der 110 Matrixzeilen "
            "aendern zwischen alter und neuer Matrix ihre geordnete Zehnerliste, und "
            "95.4 Prozent der benutzten Kandidaten haben eine solche Quellkategorie "
            "(ERWARTUNG.md Abschnitt 11.6). Eine zweistellige Zahl waere ein "
            "berichtenswerter Nebenbefund, keine Bestaetigung.")
        ok = ok and gegen_m > 0
    else:
        details["hinweis"] = ("descriptions_E_M_ALT_referenz.pkl fehlt im Paketordner — "
                             "der Vergleich gegen den Stand von Arm M entfaellt.")
    return pruefung(report, "C3_neue_matrix_kam_an", ok, **details)


def c4_achsen(ma: Path, report: dict) -> bool:
    """Beide Achsen sind angekommen: Schwellwert und Kappung.

    Eine Beschreibung aendert sich gegenueber der Basis D1 x 10 genau dann, wenn die
    geordnete Kontextkategorienliste ihrer Quellkategorie eine andere wird
    (association_loader.py Zeilen 93, 98 und 100: der Schwellwert filtert, die
    Sortierung bleibt, die Kappung schneidet vorne ab). Null Abweichung heisst deshalb,
    dass der Parameter nicht angekommen ist — ausser die Matrix gibt es her, und dann
    steht das in M5_eichung.
    """
    d1_p = desc_pfad(ma, "d1")
    if d1_p is None:
        return pruefung(report, "C4_achsen", False, fehler="descriptions_E_d1 fehlt")
    d1 = lade_pickle(d1_p)
    benutzt = benutzte_uuids(ma)
    aus, ok = {}, True
    for arm in ("d1_s01", "d1_s1", "d1_c6", "d1_c20", "b1_c6", "b1_c20"):
        q = desc_pfad(ma, arm)
        if q is None:
            aus[arm] = {"fehler": "fehlt"}
            ok = False
            continue
        g = lade_pickle(q)
        # B1-Arme gegen E_c1 vergleichen, D1-Arme gegen E_d1 — sonst misst der
        # Vergleich das Mass mit und nicht die Achse.
        if ARME[arm]["mass"] == "B1":
            basis_p = ma / "3_evaluation" / "cache" / "descriptions_E_c1.pkl"
            basis, basis_name = lade_pickle(basis_p), "E_c1 (B1 x 10)"
        else:
            basis, basis_name = d1, "E_d1 (D1 x 10)"
        n_b = sum(1 for u in benutzt if g.get(u) != basis.get(u))
        n_g = sum(1 for u in basis if g.get(u) != basis[u])
        aus[arm] = {"basis": basis_name, "achse": ARME[arm],
                    "geaendert_gegen_basis_benutzt": n_b,
                    "geaendert_gegen_basis_gesamt": n_g,
                    "anteil_benutzt": round(n_b / len(benutzt), 4) if benutzt else None}
        if n_b == 0:
            aus[arm]["fehler"] = ("null Abweichung — der Parameter ist nicht angekommen "
                                  "oder er aendert an dieser Matrix nichts. Gegen "
                                  "M5_eichung pruefen, bevor berichtet wird.")
            ok = False
    aus["hinweis"] = ("Die erwarteten Groessenordnungen stehen in ERWARTUNG.md "
                      "Abschnitt 3 (Schwellwertachse) und Abschnitt 4 (Kappungsachse) "
                      "und folgen aus M5_eichung.")
    return pruefung(report, "C4_achsen", ok, **aus)


def c5_itemcaches(ma: Path, report: dict) -> bool:
    cache = ma / "3_evaluation" / "cache"
    a = cache / "items_E_c1.pkl"
    if not a.exists():
        return pruefung(report, "C5_itemcaches", False, fehler=f"fehlt: {a}")
    ia = lade_pickle(a)["items"]
    aus, ok = {"sha256_items_E_c1": sha256(a),
               "sha256_items_E_c1_erwartet": SHA_ITEMS_E_C1}, True
    for arm in ARME:
        b = cache / f"items_E_{arm}.pkl"
        if not b.exists():
            aus[arm] = {"fehler": f"fehlt: {b}"}
            ok = False
            continue
        blob = lade_pickle(b)
        ib = blob["items"]
        gleich_lang = len(ia) == len(ib) == SOLL["total"]
        abw_kand = abw_span = 0
        if gleich_lang:
            for x, y in zip(ia, ib):
                if x["candidate_ids"] != y["candidate_ids"]:
                    abw_kand += 1
                if (x["doc_id"], x["start"], x["end"]) != (y["doc_id"], y["start"], y["end"]):
                    abw_span += 1
        dp = desc_pfad(ma, arm)
        desc = lade_pickle(dp) if dp else {}
        falsch = 0
        if desc and gleich_lang:
            for y in ib:
                for u, d in zip(y["candidate_ids"], y["descriptions"]):
                    if desc.get(u) != d:
                        falsch += 1
        # Wie viele Beschreibungen weichen von E_c1 ab? Das geht ohne den
        # Beschreibungscache, direkt aus den beiden Item-Caches — und ist damit auch
        # auf einer Maschine pruefbar, auf die nur die Item-Caches transferiert wurden.
        anders_gegen_c1 = 0
        if gleich_lang:
            for x, y in zip(ia, ib):
                anders_gegen_c1 += sum(1 for dx, dy in zip(x["descriptions"],
                                                           y["descriptions"]) if dx != dy)
        aus[arm] = {"n_items": len(ib), "resolver_feld": blob.get("resolver"),
                    "items_mit_abweichender_kandidatenliste": abw_kand,
                    "items_mit_abweichendem_span": abw_span,
                    "beschreibungen_abweichend_gegen_E_c1": anders_gegen_c1,
                    "beschreibungscache_vorhanden": dp is not None,
                    "beschreibungen_nicht_aus_dem_cache": (falsch if dp else
                                                           "nicht geprueft (Cache nicht "
                                                           "auf dieser Maschine)"),
                    "sha256": sha256(b)}
        ok = ok and gleich_lang and abw_kand == 0 and abw_span == 0 and falsch == 0
        if ARME[arm]["mass"] == "D1" and anders_gegen_c1 == 0:
            aus[arm]["fehler"] = ("null abweichende Beschreibungen gegen E_c1 — der "
                                  "Matrixtausch ist im Item-Cache nicht angekommen")
            ok = False
    return pruefung(report, _c5_name[0], ok, **aus)


_c5_name = ["C5_itemcaches"]


def c6_torwaechter(ma: Path, report: dict) -> bool:
    cache = ma / "3_evaluation" / "cache"
    ref_p, k_p = cache / "descriptions_E_c1.pkl", cache / "descriptions_E_K.pkl"
    quelle = k_p if k_p.exists() else None
    if quelle is None:
        return pruefung(report, "C6_torwaechter", True,
                        hinweis=("descriptions_E_K.pkl nicht vorhanden — die Paritaet ist "
                                 "dann allein durch 01_arm_K_parity.py belegt; dessen "
                                 "Ausgabe (verglichen/abweichend) gehoert in den Bericht."))
    gleich = sha256(ref_p) == sha256(quelle)
    return pruefung(report, "C6_torwaechter", gleich, quelle=str(quelle),
                    sha256_descriptions_E_c1=sha256(ref_p),
                    sha256_descriptions_E_K=sha256(quelle))


# ══════════════════════════════════════════════════════════════════════════════════
# Stufe zellen — Z1 bis Z8
# ══════════════════════════════════════════════════════════════════════════════════
def lade_zelle(matrix_dir: Path, model: str, resolver: str):
    p = matrix_dir / f"{model}_{resolver}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def z1_z2_zellen(matrix_dir: Path, modelle, report: dict) -> bool:
    ok, aus, fehlend = True, {}, []
    for arm, d in ARME.items():
        eid = d["resolver"]
        for mid in MODELLE_JE_ARM[arm]:
            if mid not in modelle:
                continue
            z = lade_zelle(matrix_dir, mid, eid)
            if z is None:
                fehlend.append(f"{mid} x {eid}")
                continue
            loc, err = z["location"], z["errors"]
            ist = {"total": loc["total"], "no_candidates": loc["no_candidates"],
                   "ambiguous_total": err["ambiguous_total"],
                   "unambiguous_total": err["unambiguous_total"],
                   "gold_not_in_candidates": err["gold_not_in_candidates"],
                   "correct_unambiguous": err["correct_unambiguous"]}
            soll = {k: SOLL[k] for k in ist}
            aus[f"{mid} x {eid}"] = {"ist": ist, "gleich_soll": ist == soll}
            ok = ok and ist == soll
    if not aus:
        return pruefung(report, "Z1_grundgesamtheit", False,
                        fehler="keine einzige Zelle dieses Laufs gefunden")
    ok = ok and not fehlend
    return pruefung(report, "Z1_grundgesamtheit", ok, soll={k: SOLL[k] for k in
                    ("total", "no_candidates", "ambiguous_total", "unambiguous_total",
                     "gold_not_in_candidates", "correct_unambiguous")},
                    n_zellen=len(aus), fehlende_zellen=fehlend, zellen=aus)


def z3_referenzreproduktion(matrix_dir: Path, manifest: dict | None, report: dict) -> bool:
    """Wurde E_c1 im selben Lauf mitgerechnet, muss jede Zahl zeichengleich wiederkommen.
    Das ist die einzige Kontrolle, die Umgebung, Modellgewichte und Kandidatenmenge in
    einem Zug prueft — und sie kostet rund fuenf Sekunden Ranking je Modell."""
    aus, ok, geprueft = {}, True, 0
    for mid, ref in REF_E_C1.items():
        z = lade_zelle(matrix_dir, mid, "E_c1")
        if z is None:
            continue
        ist = {"accuracy_at_1": z["location"]["accuracy_at_1"],
               "accuracy_at_3": z["location"]["accuracy_at_3"],
               "mrr": z["location"]["mrr"],
               "document_accuracy": z["document"]["document_accuracy"],
               "acc1_ambiguous_only": z["errors"]["acc1_ambiguous_only"]}
        gleich = all(ist[k] == ref[k] for k in ref)
        aus[mid] = {"ist": ist, "referenz": ref, "zeichengleich": gleich,
                    "groesste_abweichung": max(abs(ist[k] - ref[k]) for k in ref)}
        ok = ok and gleich
        geprueft += 1
    # Zweite, unabhaengige Reproduktion: Arm K2 vom 16.09.2026 ist B1 mit Kappung 6,
    # also dieselbe Belegung wie der neue Arm b1_c6. B1 ist von der Matrixkorrektur
    # nicht beruehrt, die Zelle muss also zeichengleich wiederkommen — quer ueber
    # Maschine UND ueber drei Monate.
    k2 = lade_zelle(matrix_dir, "M5_spatial_config2", "E_b1_c6")
    if k2 is not None:
        ist = {"accuracy_at_1": k2["location"]["accuracy_at_1"],
               "accuracy_at_3": k2["location"]["accuracy_at_3"],
               "mrr": k2["location"]["mrr"],
               "document_accuracy": k2["document"]["document_accuracy"],
               "acc1_ambiguous_only": k2["errors"]["acc1_ambiguous_only"]}
        gleich = all(ist[k] == REF_K2[k] for k in REF_K2)
        aus["M5_spatial_config2 x E_b1_c6 gegen Arm K2"] = {
            "ist": ist, "referenz_arm_K2": REF_K2, "zeichengleich": gleich,
            "groesste_abweichung": max(abs(ist[k] - REF_K2[k]) for k in REF_K2),
            "hinweis": ("Arm K2 ist B1 mit max_categories = 6, gerechnet am 16.09.2026 "
                        "auf Windows. b1_c6 ist dieselbe Belegung. Eine Abweichung "
                        "heisst, dass Gazetteer, Matrix oder Generator nicht mehr "
                        "dieselben sind.")}
        ok = ok and gleich
        geprueft += 1
    if geprueft == 0:
        return pruefung(report, "Z3_referenzreproduktion", False,
                        fehler=("E_c1 wurde in diesem Lauf nicht mitgerechnet. Ohne diese "
                                "Kontrolle ist nicht gezeigt, dass Umgebung und Gewichte "
                                "dieselben sind wie im Referenzlauf. Schritt mit "
                                "--resolvers E_c1 ... wiederholen."))
    return pruefung(report, "Z3_referenzreproduktion", ok, n_geprueft=geprueft, modelle=aus)


def z5_mcnemar(per_item_dir: Path, modelle, report: dict) -> bool:
    """Gepaarte Tests, rein lesend. Vergleiche:
       je Modell E_d1 gegen E_c1, und auf den Modellen der Reihe E_d1_s01/E_d1_s1
       gegen E_d1."""
    def lade(p: Path):
        with gzip.open(p, "rb") as f:
            return pickle.load(f)

    def paar(a_p: Path, b_p: Path):
        if not (a_p.exists() and b_p.exists()):
            return {"fehler": f"fehlt: {a_p.name if not a_p.exists() else b_p.name}"}
        A, B = lade(a_p), lade(b_p)
        if len(A) != len(B):
            return {"fehler": "ungleiche Itemzahl"}
        for x, y in zip(A, B):
            if (x["doc_id"], x["start"], x["end"]) != (y["doc_id"], y["start"], y["end"]):
                return {"fehler": "Itemreihenfolge weicht ab"}
        erg = {}
        for name, nur_ambig in (("alle", False), ("ambig", True)):
            idx = [i for i in range(len(A)) if (not nur_ambig) or A[i]["n_candidates"] > 1]
            ka = [A[i]["rank"] == 1 for i in idx]
            kb = [B[i]["rank"] == 1 for i in idx]
            nur_a = sum(1 for x, y in zip(ka, kb) if x and not y)
            nur_b = sum(1 for x, y in zip(ka, kb) if y and not x)
            n = nur_a + nur_b
            erg[name] = {"n": len(idx),
                         "acc1_a": sum(ka) / len(idx) if idx else None,
                         "acc1_b": sum(kb) / len(idx) if idx else None,
                         "nur_a_richtig": nur_a, "nur_b_richtig": nur_b,
                         "diskordant": n, "p": binom_p_zweiseitig(nur_a, n)}
        erg["alle_diskordanten_items_sind_ambig"] = (
            erg["alle"]["diskordant"] == erg["ambig"]["diskordant"])
        return erg

    aus = {}
    for mid in modelle:
        d1 = per_item_dir / f"{mid}_E_d1.pkl.gz"
        c1 = per_item_dir / f"{mid}_E_c1.pkl.gz"
        if d1.exists() and c1.exists():
            aus[f"{mid}: B1 x 10 (E_c1) gegen D1 x 10 (E_d1)"] = paar(c1, d1)
        # Achsen: jede Stufe gegen ihre eigene Basis, damit der Vergleich die Achse
        # misst und nicht das Mass.
        for arm in ("d1_s01", "d1_s1", "d1_c6", "d1_c20"):
            q = per_item_dir / f"{mid}_E_{arm}.pkl.gz"
            if q.exists() and d1.exists():
                aus[f"{mid}: D1 x 10 gegen {ARME[arm]['resolver']}"] = paar(d1, q)
        for arm in ("b1_c6", "b1_c20"):
            q = per_item_dir / f"{mid}_E_{arm}.pkl.gz"
            if q.exists() and c1.exists():
                aus[f"{mid}: B1 x 10 gegen {ARME[arm]['resolver']}"] = paar(c1, q)
        # Gitterdiagonale: gleiche Kappung, verschiedenes Mass.
        for b, d in (("b1_c6", "d1_c6"), ("b1_c20", "d1_c20")):
            qb = per_item_dir / f"{mid}_E_{b}.pkl.gz"
            qd = per_item_dir / f"{mid}_E_{d}.pkl.gz"
            if qb.exists() and qd.exists():
                k = ARME[b]["kappung"]
                aus[f"{mid}: B1 x {k} gegen D1 x {k}"] = paar(qb, qd)
    if not aus:
        return pruefung(report, "Z5_mcnemar", False, fehler="keine Per-Item-Dumps von E_d1")
    return pruefung(report, "Z5_mcnemar", True, vergleiche=aus,
                    hinweis=("Erwartet wird, dass alle diskordanten Items ambig sind — "
                             "eindeutige Items koennen nicht falsch gerankt werden. "
                             "Dieser Block ist Bericht, nicht Bestehenskriterium."))


def z6_unberuehrt(ma: Path, manifest_p: Path, report: dict) -> bool:
    """Nach Hash, nicht nach Aenderungsdatum. Die Vorgaengerfassung hat jedes am Lauftag
    geschriebene Matrix-JSON als ueberschrieben gewertet und deshalb bei einem zweiten
    Lauf am selben Tag einen Fehlalarm ausgeloest (Kontrollarm, ERGEBNIS.md, Abschnitt 8
    Punkt 2)."""
    if not manifest_p.exists():
        return pruefung(report, "Z6_unberuehrt", False,
                        fehler=(f"Manifest fehlt: {manifest_p}. Es wird VOR dem Lauf mit "
                                "--stufe manifest geschrieben; ohne es ist nicht "
                                "beweisbar, dass nichts ueberschrieben wurde."))
    alt = json.loads(manifest_p.read_text(encoding="utf-8"))["dateien"]
    neue_arme = {d["resolver"] for d in ARME.values()}
    # E_c1 wird in diesem Lauf absichtlich noch einmal gerechnet (Z3, die
    # Reproduktionskontrolle). Seine Dateien duerfen sich deshalb aendern; ihr Inhalt
    # wird nicht hier, sondern in Z3 auf Zeichengleichheit geprueft. Das Feld
    # eval_seconds im JSON aendert sich ohnehin bei jedem Lauf.
    erlaubt_neu_geschrieben = {"E_c1"} | neue_arme

    def gehoert_zu(rel: str, arme) -> bool:
        name = Path(rel).name
        for eid in arme:
            if name.endswith(f"_{eid}.json") or name.endswith(f"_{eid}.pkl.gz"):
                return True
        return False

    geaendert, verschwunden, erwartet_geschrieben = [], [], []
    for rel, e in alt.items():
        q = ma / rel
        if not q.exists():
            verschwunden.append(rel)
            continue
        if Path(rel).name.startswith("summary.csv"):
            continue  # gesondert in Z7
        h = sha256(q)
        if h == e["sha256"]:
            continue
        eintrag = {"datei": rel, "sha256_vorher": e["sha256"], "sha256_jetzt": h}
        if gehoert_zu(rel, erlaubt_neu_geschrieben):
            erwartet_geschrieben.append(eintrag)
        else:
            geaendert.append(eintrag)
    # Dateien, die dieser Lauf neu anlegen darf
    erlaubt_neu = sorted(
        str(q.relative_to(ma)).replace("\\", "/")
        for q in manifest_dateien(ma)
        if str(q.relative_to(ma)).replace("\\", "/") not in alt)
    unerwartet_neu = [r for r in erlaubt_neu if not gehoert_zu(r, neue_arme)]
    ok = not geaendert and not verschwunden and not unerwartet_neu
    return pruefung(report, "Z6_unberuehrt", ok,
                    manifest=str(manifest_p), n_im_manifest=len(alt),
                    bestehende_dateien_mit_neuem_hash=geaendert,
                    bestehende_dateien_verschwunden=verschwunden,
                    erwartet_neu_geschrieben_E_c1=erwartet_geschrieben,
                    neu_hinzugekommen=erlaubt_neu,
                    neu_und_nicht_diesem_lauf_zuzuordnen=unerwartet_neu,
                    hinweis=("Die Dateien unter erwartet_neu_geschrieben_E_c1 sind die "
                             "Reproduktionskontrolle: E_c1 wird absichtlich noch einmal "
                             "gerechnet. Ihr Inhalt wird in Z3 geprueft, nicht hier."))


def z7_summary(ma: Path, report: dict) -> bool:
    res = ma / "results"
    summary = res / "summary.csv"
    sicherungen = sorted(res.glob("summary.csv.bak_vor_E_d1*"))
    if not summary.exists() or not sicherungen:
        return pruefung(report, "Z7_summary", False,
                        fehler=("summary.csv oder die Sicherung summary.csv.bak_vor_E_d1 "
                                "fehlt. 02_evaluate_matrix.py schreibt summary.csv "
                                "vollstaendig neu und nur mit den Zellen des aktuellen "
                                "Laufs (Zeilen 138 bis 156)."))
    b = sicherungen[0]
    gleich = sha256(summary) == sha256(b)
    return pruefung(report, "Z7_summary", gleich, sicherung=b.name,
                    sha256_summary_csv=sha256(summary), sha256_sicherung=sha256(b))


def acc1_ambig_aus_acc1(acc1: float) -> float:
    """Geschlossene Form fuer acc1_ambiguous_only aus acc1.

    In allen fuenfzehn Referenzzellen ist correct_unambiguous = 43'544 und
    unambiguous_total = 43'734 bei total = 75'741 — der eindeutige Sockel ist konstant,
    weil eindeutige Items nicht falsch gerankt werden koennen. Also gilt exakt

        acc1_ambiguous_only = (acc1 * 75'741 - 43'544) / 23'848

    Am 17.09.2026 an fuenf Zellen gegengeprueft; groesste Abweichung 3.3e-16, also reine
    Gleitkommarundung. Damit ist die Leitmetrik auch fuer results/ablations.json
    verfuegbar, das sie selbst nicht fuehrt (04_ablations.py verdichtet per_item
    sofort zu location_metrics, Zeilen 119 bis 121).

    **Gueltig nur**, solange die Kandidatenmenge unveraendert ist — genau das pruefen
    Z1 und Z2.
    """
    return (acc1 * SOLL["total"] - SOLL["correct_unambiguous"]) / SOLL["ambiguous_total"]


def z8_ergebnis(matrix_dir: Path, modelle, report: dict) -> None:
    """Die Ergebnistabelle, als Gitter Mass x Kappung und als Schwellwertachse."""
    tab, gitter = {}, {}
    for arm, d in ARME.items():
        eid = d["resolver"]
        for mid in MODELLE_JE_ARM[arm]:
            if mid not in modelle:
                continue
            z = lade_zelle(matrix_dir, mid, eid)
            if z is None:
                continue
            ist = {"accuracy_at_1": z["location"]["accuracy_at_1"],
                   "accuracy_at_3": z["location"]["accuracy_at_3"],
                   "mrr": z["location"]["mrr"],
                   "document_accuracy": z["document"]["document_accuracy"],
                   "acc1_ambiguous_only": z["errors"]["acc1_ambiguous_only"],
                   "wrong_rank": z["errors"]["wrong_rank"]}
            eintrag = {"mass": d["mass"], "kappung": d["kappung"],
                       "schwelle": d["schwelle"], "werte": ist}
            basis = REF_E_C1.get(mid)
            if basis:
                eintrag["delta_gegen_B1_x_10"] = {k: ist[k] - basis[k] for k in basis}
            tab[f"{mid} x {eid}"] = eintrag
            if d["schwelle"] == 0.001:
                gitter.setdefault(mid, {}).setdefault(d["mass"], {})[d["kappung"]] = {
                    "acc1": ist["accuracy_at_1"],
                    "acc1_ambig": ist["acc1_ambiguous_only"]}
            print(f"  {mid:<22} {eid:<12} {d['mass']} x{d['kappung']:<3} "
                  f"thr {d['schwelle']:<6} Acc@1 {ist['accuracy_at_1']:.6f}  "
                  f"Acc@1(ambig) {ist['acc1_ambiguous_only']:.6f}")

    # Die bestehende Spalte B1 x 10 gehoert ins Gitter, sie ist gerechnet.
    for mid, ref in REF_E_C1.items():
        if mid in modelle:
            gitter.setdefault(mid, {}).setdefault("B1", {})[10] = {
                "acc1": ref["accuracy_at_1"], "acc1_ambig": ref["acc1_ambiguous_only"],
                "quelle": "Referenzlauf, results/matrix/*_E_c1.json"}

    print("\n  Gitter Mass x Kappung (Acc@1 ambig), Schwellwert 0.001:")
    for mid in sorted(gitter):
        for mass in ("B1", "D1"):
            zeile = gitter[mid].get(mass, {})
            if zeile:
                teile = "  ".join(f"x{k}: {zeile[k]['acc1_ambig']:.4f}"
                                  for k in sorted(zeile))
                print(f"    {mid:<22} {mass}   {teile}")
    bericht(report, "Z8_ergebnis", zellen=tab, gitter_acc1_ambig=gitter)


# ══════════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stufe", required=True,
                    choices=["manifest", "matrix", "caches", "zellen", "alles"])
    ap.add_argument("--ma", default=None, help="Projektwurzel (sonst gesucht)")
    ap.add_argument("--paket", default=str(HERE), help="Paketordner E-d1_Lauf")
    ap.add_argument("--matrix", default=None, help="die korrigierte Distanzmatrix (CSV)")
    ap.add_argument("--alt-matrix", default=None,
                    help="die alte npmi_dist_matrix_B.csv (Default: im Paketordner)")
    ap.add_argument("--b1", default=None, help="b1_matrix.csv fuer den Massvergleich")
    ap.add_argument("--w", nargs="*", default=None, help="W_cellwise*.npy des Neulaufs")
    ap.add_argument("--w-glob", default=None,
                    help="Muster fuer die W-Dateien, z.B. 'results/W_cellwise_neu*.npy'")
    ap.add_argument("--meta", nargs="*", default=None, help="cellwise_meta*.json des Neulaufs")
    ap.add_argument("--meta-glob", default=None, help="Muster fuer die cellwise_meta-Dateien")
    ap.add_argument("--pool", default=None, help="pool.parquet (Default: <--d1-dir>/1_data/pool.parquet)")
    ap.add_argument("--d1-dir", default=None,
                    help="Ordner mit den Zwischenprodukten der D1-Rechnung (1_data/, results/); "
                         "Default: ma-experiments/data/d1")
    ap.add_argument("--src-cells", default=None,
                    help="die materialisierte, sortierte Quellzellliste des Neulaufs "
                         "(.npy mit uint64 oder .parquet/.csv mit Spalte cell). Wird sie "
                         "angegeben, prueft M4 sie unmittelbar statt nur die Metadaten.")
    ap.add_argument("--models", nargs="*", default=list(ALLE_MODELLE))
    ap.add_argument("--out", default=None)
    ap.add_argument("--manifest", default=None,
                    help="Pfad des Hash-Manifests (Default: "
                         "3_evaluation/logs/E_d1/ERGEBNIS_E_d1_manifest.json)")
    a = ap.parse_args()

    ma = find_ma(a.ma)
    paket = Path(a.paket).resolve()
    ev = ma / "3_evaluation"
    exp5 = Path(a.d1_dir) if a.d1_dir else ma.parent / "data" / "d1"
    matrix_dir = ma / "results" / "matrix"
    # Pruefberichte enthalten lokale Pfade -> nach 3_evaluation/logs/, nicht nach results/.
    ziel = Path(a.out) if a.out else (ev / "logs" / "E_d1" / f"ERGEBNIS_E_d1_{a.stufe}.json")
    manifest_p = Path(a.manifest) if a.manifest else \
        (ev / "logs" / "E_d1" / "ERGEBNIS_E_d1_manifest.json")

    print(f"Projektwurzel: {ma}")
    print(f"Paketordner:   {paket}")
    print(f"Stufe:         {a.stufe}")
    print(f"Gitter:        {len(ARME)} Arme, "
          + ", ".join(f"{d['mass']}x{d['kappung']}@{d['schwelle']}"
                      for d in ARME.values()))

    if a.stufe == "manifest":
        ok = stufe_manifest(ma, manifest_p)
        print(f"\n[verify] {'BESTANDEN' if ok else 'FEHLGESCHLAGEN'} -> {manifest_p}")
        raise SystemExit(0 if ok else 1)

    report = {"zeitpunkt": time.strftime("%Y-%m-%d %H:%M:%S"), "stufe": a.stufe,
              "projektwurzel": str(ma), "paket": str(paket), "pruefungen": {}}
    p = report["pruefungen"]
    alles_ok = True

    alt_matrix = Path(a.alt_matrix) if a.alt_matrix else (paket / "npmi_dist_matrix_B_ALT_referenz.csv")
    matrix = Path(a.matrix) if a.matrix else None

    if a.stufe in ("matrix", "alles"):
        if matrix is None:
            raise SystemExit("--matrix ist fuer diese Stufe Pflicht.")
        w = [Path(x) for x in (a.w or [])]
        if a.w_glob:
            w += sorted(exp5.glob(a.w_glob)) if not Path(a.w_glob).is_absolute() \
                else sorted(Path(a.w_glob).parent.glob(Path(a.w_glob).name))
        metas = [Path(x) for x in (a.meta or [])]
        if a.meta_glob:
            metas += sorted(exp5.glob(a.meta_glob)) if not Path(a.meta_glob).is_absolute() \
                else sorted(Path(a.meta_glob).parent.glob(Path(a.meta_glob).name))
        pool = Path(a.pool) if a.pool else exp5 / "1_data" / "pool.parquet"
        cats_meta = metas[0] if metas else exp5 / "results" / "cellwise_meta_p0.json"

        print("\n[M1] Name und Hash der korrigierten Matrix")
        alles_ok &= m1_name_und_hash(matrix, p)
        print("\n[M2] Form der Matrix")
        alles_ok &= m2_form(matrix, alt_matrix, p)
        print("\n[M3] Schranke: Zeilensumme <= 10 mal Objektzahl")
        alles_ok &= m3_schranke([str(x) for x in w], cats_meta, pool, p)
        print("\n[M4] Abdeckung der Quellzellen")
        alles_ok &= m4_abdeckung([str(x) for x in metas], p, a.src_cells)
        print("\n[M5] Eichungskennzahlen (Bericht)")
        m5_eichung(matrix, a.b1, p)
        print("\n[M6] Alte gegen neue Matrix (Bericht)")
        m6_alt_gegen_neu(matrix, alt_matrix, p)

    if a.stufe in ("caches", "alles"):
        print("\n[C0] Verdrahtung der drei Arme (statisch)")
        alles_ok &= c0_armspec(paket, ma, p)
        print("\n[C1] Beschreibungscaches und Metadateien")
        alles_ok &= c1_c2_caches(ma, matrix or Path("/nicht/angegeben"), p)
        print("\n[C2] Mac gegen Windows: b1_c6 zeichengleich zu Arm K2?")
        alles_ok &= c2_mac_gegen_windows(ma, paket, p)
        print("\n[C3] Ist die korrigierte D1-Matrix angekommen?")
        alles_ok &= c3_neue_matrix_kam_an(ma, paket, p)
        print("\n[C4] Schwellwert- und Kappungsachse")
        alles_ok &= c4_achsen(ma, p)
        print("\n[C5] Item-Caches")
        _c5_name[0] = "C5_itemcaches"
        alles_ok &= c5_itemcaches(ma, p)
        print("\n[C6] Torwaechter Arm K")
        alles_ok &= c6_torwaechter(ma, p)

    if a.stufe in ("zellen", "alles"):
        print("\n[Z1/Z2] Grundgesamtheit und Kandidatenmenge")
        alles_ok &= z1_z2_zellen(matrix_dir, a.models, p)
        print("\n[Z3] Reproduktion der Referenz (E_c1 und Arm K2)")
        alles_ok &= z3_referenzreproduktion(matrix_dir, None, p)
        print("\n[Z4] Item-Caches (derselbe Block wie C5, hier gegen den Transfer)")
        _c5_name[0] = "Z4_itemcaches"
        alles_ok &= c5_itemcaches(ma, p)
        _c5_name[0] = "C5_itemcaches"
        print("\n[Z5] Gepaarte Tests (Bericht)")
        alles_ok &= z5_mcnemar(ev / "cache" / "per_item", a.models, p)
        print("\n[Z6] Nichts ueberschrieben (Hash gegen Manifest)")
        alles_ok &= z6_unberuehrt(ma, manifest_p, p)
        print("\n[Z7] summary.csv zurueckgesetzt")
        alles_ok &= z7_summary(ma, p)
        print("\n[Z8] Ergebnistabelle")
        z8_ergebnis(matrix_dir, a.models, p)

    report["all_ok"] = bool(alles_ok)
    ziel.parent.mkdir(parents=True, exist_ok=True)
    ziel.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
    print(f"\n[verify] {'BESTANDEN' if alles_ok else 'FEHLGESCHLAGEN'} -> {ziel}")
    raise SystemExit(0 if alles_ok else 1)


if __name__ == "__main__":
    main()
