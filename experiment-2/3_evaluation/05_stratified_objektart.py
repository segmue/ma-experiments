"""
Stage 05 — Stratifizierte Evaluation nach Objektklasse (OBJEKTART).

Betreuer-Feedback 12.6.: Werden z.B. Berge schlechter aufgeloest als Fluesse?
Rein post-hoc auf den per-Item-Dumps aus Stage 02 (cache/per_item/),
kein Encoding. Die OBJEKTART des Gold-Eintrags wird per UUID aus der DuckDB
features-Tabelle (config1) gejoint; Fallback: corpus_swissnames3d.gpkg.

Zwei Sichten:
  1. Retrieval pro Klasse (modell-unabhaengig, Kandidaten sind fuer alle Systeme
     identisch): no_candidate- und gold_not_in_candidates-Raten.
  2. Resolution pro Klasse und System: Acc@1/Acc@3/MRR (alle 15 Zellen).

Output:
  results/tables/stratified_objektart.csv     (System x Klasse, lang)
  results/tables/stratified_retrieval.csv     (Klasse, Retrieval-Raten)
  plots/stratified_objektart_acc1.png         (Heatmap Klasse x System)
  plots/stratified_retrieval.png              (Retrieval-Fehlerraten pro Klasse)

Verwendung (nach 02_evaluate_matrix.py):
    python 05_stratified_objektart.py
    python 05_stratified_objektart.py --min-n 100
"""

from __future__ import annotations

import argparse
import csv
import gzip
import pickle
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config as C

OTHER_LABEL = "Other"
UNKNOWN_LABEL = "UNBEKANNT"


def load_per_item_dumps() -> dict:
    """Laedt alle per-Item-Dumps aus Stage 02: {(model, resolver): per_item}."""
    dumps = {}
    for mid in C.MODELS:
        for eid in C.MATRIX_RESOLVERS:
            path = C.PER_ITEM_DIR / f"{mid}_{eid}.pkl.gz"
            if path.exists():
                with gzip.open(path, "rb") as f:
                    dumps[(mid, eid)] = pickle.load(f)
    return dumps


def _gpkg_mapping(gpkg, wanted: set) -> dict:
    """UUID -> OBJEKTART aus dem GeoPackage. Erst geopandas, sonst sqlite3
    (ein GeoPackage ist eine SQLite-Datei; kein GDAL-Stack noetig)."""
    out = {}
    try:
        import geopandas as gpd
        gdf = gpd.read_file(gpkg, columns=["sn3d_uuid", "sn3d_objektart"],
                            ignore_geometry=True)
        pairs = zip(gdf["sn3d_uuid"], gdf["sn3d_objektart"])
    except Exception:
        import sqlite3
        con = sqlite3.connect(str(gpkg))
        pairs = con.execute(
            "SELECT sn3d_uuid, sn3d_objektart FROM corpus_toponyms").fetchall()
        con.close()
    for u, o in pairs:
        if u in wanted and isinstance(o, str) and o:
            out[u] = o
    return out


def objektart_by_uuid(gold_ids: set) -> dict:
    """UUID -> OBJEKTART. Primaer DuckDB features (config1), Fallback gpkg
    (geopandas oder sqlite3). Ohne DuckDB traegt der gpkg-Pfad allein."""
    gpkg = C.CORPUS_GPKG
    mapping = {}
    try:
        import duckdb

        con = duckdb.connect(str(C.duckdb_path("config1")), read_only=True)
        rows = con.execute("SELECT UUID, OBJEKTART FROM features").fetchall()
        con.close()
        mapping = {u: o for u, o in rows if u in gold_ids}
    except Exception as err:
        print(f"  DuckDB nicht verfuegbar ({err}) — nutze {gpkg.name}")

    missing = gold_ids - mapping.keys()
    if missing:
        try:
            mapping.update(_gpkg_mapping(gpkg, missing))
        except Exception as err:
            print(f"  gpkg-Fallback nicht verfuegbar ({err}) — "
                  f"{len(missing)} UUIDs bleiben {UNKNOWN_LABEL}")
    return mapping


def bundle_classes(class_counts: dict, min_n: int) -> dict:
    """Klasse -> Anzeige-Klasse; Klassen unter min_n werden zu OTHER gebuendelt."""
    return {cls: (cls if n >= min_n else OTHER_LABEL)
            for cls, n in class_counts.items()}


def main():
    ap = argparse.ArgumentParser(description="Stratifizierung nach OBJEKTART")
    ap.add_argument("--min-n", type=int, default=50,
                    help="Klassen mit weniger Gold-Items werden zu 'Other' gebuendelt")
    ap.add_argument("--plot-only", action="store_true",
                    help="nur Abbildungen aus den vorhandenen Tabellen zeichnen (kein Join, keine Neuberechnung)")
    args = ap.parse_args()

    C.ensure_dirs()
    if args.plot_only:
        make_plots(pd.read_csv(C.TABLES_DIR / "stratified_objektart.csv"),
                   pd.read_csv(C.TABLES_DIR / "stratified_retrieval.csv"), args.min_n)
        return
    dumps = load_per_item_dumps()
    if not dumps:
        print(f"FEHLT: keine per-Item-Dumps in {C.PER_ITEM_DIR}. "
              f"Erst 02_evaluate_matrix.py (aktuelle Version) ausfuehren.")
        raise SystemExit(1)
    print(f"{len(dumps)} Zellen mit per-Item-Dumps geladen.")

    # OBJEKTART-Join (Items/Spans sind ueber alle Zellen identisch -> Referenzdump).
    ref = next(iter(dumps.values()))
    gold_ids = {p["gold_id"] for p in ref if p["gold_id"]}
    mapping = objektart_by_uuid(gold_ids)
    covered = sum(1 for g in gold_ids if g in mapping)
    print(f"OBJEKTART-Lookup: {covered}/{len(gold_ids)} eindeutige Gold-UUIDs "
          f"({covered / len(gold_ids):.1%})")

    def cls_of(p) -> str:
        return mapping.get(p["gold_id"], UNKNOWN_LABEL)

    class_counts = defaultdict(int)
    for p in ref:
        class_counts[cls_of(p)] += 1
    display = bundle_classes(class_counts, args.min_n)

    # ── 1) Retrieval pro Klasse (modell-unabhaengig) ────────────────────────────
    retr = defaultdict(lambda: {"n": 0, "no_candidate": 0, "gold_not_in_candidates": 0})
    for p in ref:
        r = retr[display[cls_of(p)]]
        r["n"] += 1
        if p["n_candidates"] == 0:
            r["no_candidate"] += 1
        elif p["rank"] is None:
            r["gold_not_in_candidates"] += 1

    retr_rows = [{"objektart": cls, "n": v["n"],
                  "no_candidate": v["no_candidate"],
                  "no_candidate_rate": v["no_candidate"] / v["n"],
                  "gold_not_in_candidates": v["gold_not_in_candidates"],
                  "gold_not_in_candidates_rate": v["gold_not_in_candidates"] / v["n"]}
                 for cls, v in sorted(retr.items(), key=lambda kv: -kv[1]["n"])]
    retr_df = pd.DataFrame(retr_rows)
    retr_df.to_csv(C.TABLES_DIR / "stratified_retrieval.csv", index=False)

    # ── 2) Resolution pro Klasse und System ─────────────────────────────────────
    rows = []
    for (mid, eid), per_item in sorted(dumps.items()):
        agg = defaultdict(lambda: {"n": 0, "c1": 0, "c3": 0, "rr": 0.0, "wrong_rank": 0})
        for p in per_item:
            a = agg[display[cls_of(p)]]
            a["n"] += 1
            a["rr"] += p["rr"]
            if p["rank"] == 1:
                a["c1"] += 1
            if p["rank"] is not None and p["rank"] <= 3:
                a["c3"] += 1
            if p["rank"] is not None and p["rank"] > 1:
                a["wrong_rank"] += 1
        for cls, a in agg.items():
            rows.append({"model": mid, "eval_resolver": eid, "objektart": cls,
                         "n": a["n"],
                         "acc1": a["c1"] / a["n"], "acc3": a["c3"] / a["n"],
                         "mrr": a["rr"] / a["n"],
                         "wrong_rank_rate": a["wrong_rank"] / a["n"]})
    df = pd.DataFrame(rows)
    df.to_csv(C.TABLES_DIR / "stratified_objektart.csv", index=False)
    print(f"Tabellen: {C.TABLES_DIR / 'stratified_objektart.csv'}, "
          f"{C.TABLES_DIR / 'stratified_retrieval.csv'}")

    make_plots(df, retr_df, args.min_n)


def make_plots(df: pd.DataFrame, retr_df: pd.DataFrame, min_n: int) -> None:
    """Zeichnet beide Abbildungen allein aus den beiden Tabellen (auch per --plot-only)."""
    # Klassen nach Gesamt-n absteigend, Other/UNBEKANNT ans Ende.
    retr_df = retr_df.sort_values("n", ascending=False)
    order = [c for c in retr_df["objektart"] if c not in (OTHER_LABEL, UNKNOWN_LABEL)]
    order += [c for c in (OTHER_LABEL, UNKNOWN_LABEL) if c in retr_df["objektart"].values]

    df = df.copy()
    df["system"] = df["model"].map(C.SHORT_MODEL) + "+" + df["eval_resolver"]
    present = set(zip(df["model"], df["eval_resolver"]))
    sys_order = [f"{C.SHORT_MODEL[m]}+{e}" for m in C.MODELS for e in C.MATRIX_RESOLVERS
                 if (m, e) in present]
    pivot = df.pivot_table(index="objektart", columns="system", values="acc1")
    pivot = pivot.reindex(index=order, columns=sys_order)

    n_by_cls = retr_df.set_index("objektart")["n"]
    # swisstopo fuehrt die Objektarten ohne Umlaute; fuer die Abbildung zurueck.
    uml = {"Huegelzug": "Hügelzug", "Haupthuegel": "Haupthügel", "Uebrige Bahnen": "Übrige Bahnen"}
    ylabels = [f"{uml.get(cls, cls)} (n={n_by_cls[cls]:,})".replace(",", " ") for cls in pivot.index]
    plt.figure(figsize=(max(8.0, 4 + 0.75 * len(pivot.columns)), 1.5 + 0.42 * len(pivot)))
    sns.heatmap(pivot.astype(float), annot=True, fmt=".2f", cmap="viridis",
                vmin=0, vmax=1, cbar_kws={"label": "Accuracy@1"},
                yticklabels=ylabels)
    plt.title(f"Accuracy@1 by object class (OBJEKTART) × system (classes with n≥{min_n})")
    plt.ylabel("Object class (OBJEKTART, gold)"); plt.xlabel("System (model+description variant)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(C.PLOTS_DIR / "stratified_objektart_acc1.png", dpi=150)
    plt.close()

    plot_df = retr_df.set_index("objektart").reindex(order)
    ax = plot_df[["no_candidate_rate", "gold_not_in_candidates_rate"]].plot(
        kind="barh", stacked=True, figsize=(9, 1.5 + 0.35 * len(plot_df)),
        color=["#d62728", "#ff9896"])
    ax.invert_yaxis()
    ax.set_yticklabels([uml.get(c, c) for c in plot_df.index])
    ax.set_xlabel("Share of gold toponyms")
    ax.set_ylabel("Object class (OBJEKTART, gold)")
    ax.set_title("Retrieval failures per object class (model-independent)")
    ax.legend(["no candidate", "gold not among candidates"], loc="upper right")
    plt.tight_layout()
    plt.savefig(C.PLOTS_DIR / "stratified_retrieval.png", dpi=150)
    plt.close()
    print(f"Plots: {C.PLOTS_DIR / 'stratified_objektart_acc1.png'}, "
          f"{C.PLOTS_DIR / 'stratified_retrieval.png'}")


if __name__ == "__main__":
    main()
