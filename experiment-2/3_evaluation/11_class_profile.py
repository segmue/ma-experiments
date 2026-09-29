"""
Stage 11 — Profil je Objektklasse (Nachtrag 2026-09-04).

Fuehrt zusammen, was bisher auf drei Dateien verteilt war, und ergaenzt zwei
Bezugsgroessen, die im Ergebniskapitel fehlen:

  * Featurezahl der Klasse im gesamten Gazetteer (aus den swissNAMES3D-DBFs)
  * vorherrschender Geometrietyp der gematchten Features (aus dem Korpus-GPKG)
  * Acc@1 der fairen Baseline M3/E_default neben dem besten System M5/E_c1,
    plus Delta — ohne das laesst sich nicht ablesen, WO das System etwas leistet

Output:
  results/tables/class_profile.csv        (vollstaendig, alle Klassen)
  results/tables/class_profile.tex        (n >= 150, Druckfassung)
  results/tables/corpus_class_distribution.csv   (fuer Kapitel 5.2.3)
"""
from __future__ import annotations

import collections
import csv
import sqlite3
import struct
from pathlib import Path

import config as C

TABLES = C.TABLES_DIR
CORPUS_GPKG = C.CORPUS_GPKG
DBF_DIR = C.SWISSNAMES3D_DIR
DBFS = ["swissNAMES3D_PKT.dbf", "swissNAMES3D_LIN.dbf", "swissNAMES3D_PLY.dbf"]

BASELINE = ("M3_default_finetuned", "E_default")
DIAG_C1 = ("M4_spatial_config1", "E_c1")
BEST = ("M5_spatial_config2", "E_c1")

GEOM_LABEL = {"point": "Punkt", "polygon": "Flaeche", "line": "Linie"}
PRINT_MIN_N = 150


def dbf_objektart_counts(path: Path) -> collections.Counter:
    """Minimaler dBASE-III-Leser; nur das Feld OBJEKTART wird gezaehlt."""
    with open(path, "rb") as fh:
        head = fh.read(32)
        n_rec = struct.unpack("<I", head[4:8])[0]
        hdr_len = struct.unpack("<H", head[8:10])[0]
        rec_len = struct.unpack("<H", head[10:12])[0]
        fields, off = [], 1
        while True:
            fd = fh.read(32)
            if fd[0:1] == b"\r":
                break
            name = fd[:11].split(b"\x00")[0].decode("latin1")
            fields.append((name, off, fd[16]))
            off += fd[16]
        pos, length = next((o, l) for n, o, l in fields if n == "OBJEKTART")
        fh.seek(hdr_len)
        counts = collections.Counter()
        for _ in range(n_rec):
            rec = fh.read(rec_len)
            if len(rec) < rec_len:
                break
            counts[rec[pos:pos + length].decode("latin1").strip()] += 1
    return counts


def gazetteer_counts() -> collections.Counter:
    total = collections.Counter()
    for name in DBFS:
        total.update(dbf_objektart_counts(DBF_DIR / name))
    return total


def geometry_by_class() -> dict[str, str]:
    con = sqlite3.connect(CORPUS_GPKG)
    out: dict[str, collections.Counter] = {}
    q = ("select sn3d_objektart, sn3d_geom_type, count(*) from corpus_toponyms "
         "where sn3d_objektart is not null group by 1, 2")
    for art, geom, n in con.execute(q):
        out.setdefault(art, collections.Counter())[geom or "unbekannt"] = n
    con.close()
    return {a: c.most_common(1)[0][0] for a, c in out.items()}


def read_stratified() -> dict[tuple[str, str], dict[str, dict]]:
    rows = list(csv.DictReader(open(TABLES / "stratified_objektart.csv", encoding="utf-8")))
    out: dict[tuple[str, str], dict[str, dict]] = {}
    for r in rows:
        out.setdefault((r["model"], r["eval_resolver"]), {})[r["objektart"]] = r
    return out


def main() -> None:
    strat = read_stratified()
    retrieval = {r["objektart"]: r for r in
                 csv.DictReader(open(TABLES / "stratified_retrieval.csv", encoding="utf-8"))}
    gaz = gazetteer_counts()
    geom = geometry_by_class()

    base, diag, best = strat[BASELINE], strat[DIAG_C1], strat[BEST]
    total_n = sum(int(r["n"]) for r in base.values())

    header = ["objektart", "geometrie", "n_korpus", "anteil_korpus", "n_gazetteer",
              "acc1_M3_E_default", "acc1_M4_E_c1", "acc1_M5_E_c1",
              "delta_best_vs_baseline", "mrr_M5_E_c1", "wrong_rank_M5_E_c1",
              "no_candidate_rate", "gold_not_in_candidates_rate"]
    rows = []
    for art in sorted(base, key=lambda a: -int(base[a]["n"])):
        n = int(base[art]["n"])
        ret = retrieval.get(art, {})
        rows.append({
            "objektart": art,
            "geometrie": GEOM_LABEL.get(geom.get(art, ""), "—"),
            "n_korpus": n,
            "anteil_korpus": round(n / total_n, 5),
            "n_gazetteer": gaz.get(art, ""),
            "acc1_M3_E_default": round(float(base[art]["acc1"]), 4),
            "acc1_M4_E_c1": round(float(diag[art]["acc1"]), 4),
            "acc1_M5_E_c1": round(float(best[art]["acc1"]), 4),
            "delta_best_vs_baseline": round(float(best[art]["acc1"]) - float(base[art]["acc1"]), 4),
            "mrr_M5_E_c1": round(float(best[art]["mrr"]), 4),
            "wrong_rank_M5_E_c1": round(float(best[art]["wrong_rank_rate"]), 4),
            "no_candidate_rate": round(float(ret["no_candidate_rate"]), 4) if ret else "",
            "gold_not_in_candidates_rate": (
                round(float(ret["gold_not_in_candidates_rate"]), 4) if ret else ""),
        })

    out_csv = TABLES / "class_profile.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

    # ── Druckfassung ─────────────────────────────────────────────────────────
    def num(x: float) -> str:
        return f"{x:.3f}".lstrip("0") if x >= 0 else "$-$" + f"{abs(x):.3f}".lstrip("0")

    def thousands(x) -> str:
        return f"{int(x):,}".replace(",", "\\,") if x != "" else "---"

    printed = [r for r in rows if r["n_korpus"] >= PRINT_MIN_N]
    covered = sum(r["n_korpus"] for r in printed) / total_n
    lines = [
        "% erzeugt von 11_class_profile.py — nicht von Hand editieren",
        "% fett = bester Wert der Zeile (nicht der Spalte)",
        "\\begin{tabular}{llrrrrrrr}",
        "\\toprule",
        (" & & & & \\multicolumn{3}{c}{Accuracy@1} & & \\\\"),
        "\\cmidrule(lr){5-7}",
        ("Objektart & Geom. & $n$ & Gaz. & M3/E\\textsubscript{def} & "
         "M4/E\\textsubscript{c1} & M5/E\\textsubscript{c1} & $\\Delta$ & o.\\,Kand. \\\\"),
        "\\midrule",
    ]
    sep = " & "
    eol = " " + chr(92) * 2
    bold_open = chr(92) + "textbf{"
    bs = chr(92)
    # swisstopo schreibt die Objektarten ohne Umlaute; fuer den Druck zurueck.
    umlaut = {
        "Huegelzug": "H" + bs + '"' + "ugelzug",
        "Haupthuegel": "Haupth" + bs + '"' + "ugel",
        "Uebrige Bahnen": bs + '"' + "Ubrige Bahnen",
        "Flaeche": "Fl" + bs + '"' + "ache",
        "Abwasserreinigungsareal": "Abwasserreinigungsareal",
    }
    for r in printed:
        accs = [r["acc1_M3_E_default"], r["acc1_M4_E_c1"], r["acc1_M5_E_c1"]]
        top = max(accs)
        # Fett markiert den Zeilenbesten, nicht die Spalte — sonst liest man
        # eine Spaltenhervorhebung als Ueberlegenheit, die in einzelnen
        # Klassen nicht besteht (z. B. Huegelzug, wo M4/E_c1 vorn liegt).
        acc_cells = [(bold_open + num(a) + "}") if a == top else num(a) for a in accs]
        nocand = num(r["no_candidate_rate"]) if r["no_candidate_rate"] != "" else "---"
        cells = [umlaut.get(r["objektart"], r["objektart"]),
                 umlaut.get(r["geometrie"], r["geometrie"]),
                 thousands(r["n_korpus"]), thousands(r["n_gazetteer"]),
                 *acc_cells, num(r["delta_best_vs_baseline"]), nocand]
        lines.append(sep.join(cells) + eol)
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "class_profile.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ── Korpusverteilung fuer Kapitel 5.2.3 ──────────────────────────────────
    geom_agg = collections.Counter()
    with open(TABLES / "corpus_class_distribution.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["objektart", "geometrie", "n_gold", "anteil_gold", "n_gazetteer"])
        for r in rows:
            w.writerow([r["objektart"], r["geometrie"], r["n_korpus"],
                        r["anteil_korpus"], r["n_gazetteer"]])
            geom_agg[r["geometrie"]] += r["n_korpus"]
        w.writerow([])
        w.writerow(["# Geometrieverteilung des Goldstandards"])
        for g, n in geom_agg.most_common():
            w.writerow([g, "", n, round(n / total_n, 5), ""])

    print(f"{len(rows)} Klassen, n gesamt = {total_n:,}")
    print(f"Druckfassung: {len(printed)} Zeilen (n >= {PRINT_MIN_N}), "
          f"Abdeckung {covered:.1%}")
    print("Geometrieverteilung Goldstandard:",
          {g: f"{n} ({n/total_n:.1%})" for g, n in geom_agg.most_common()})
    print("->", out_csv)


if __name__ == "__main__":
    main()
