"""
Stage 06: Report -- Tabellen + Plots aus den Benchmark-Resultaten.

Liest results/{build,micro,descriptions,parity,throughput,acc_geo}.json
(nur vorhandene Dateien) und erzeugt:
  plots/bench_build_size.png      Build-Zeiten (Geo-Phasen) + DB-Groessen H3 vs Geo
  plots/bench_micro.png           Mikro-Latenzen (Median, log-Skala), Facet pro Config
  plots/bench_descriptions.png    Beschreibungs-Generierung: Batch- + Einzelpfad-Zeiten
  plots/bench_throughput.png      s/100 Dokumente pro System, Faktor vs default annotiert
  plots/bench_acc_delta.png       Delta-Acc@1 geo vs h3 pro Modell/Config
  results/tables/*.csv            flache Tabellen zu jedem Plot
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

import bench_config as C  # noqa: E402

sns.set_theme(style="whitegrid")
DPI = 150


def _load(name: str):
    p = C.RESULTS_DIR / f"{name}.json"
    if not p.exists():
        print(f"  [skip] {name}.json fehlt")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def plot_build() -> None:
    data = _load("build")
    if not data:
        return
    rows = []
    for cfg, s in data["configs"].items():
        rows.append({"config": cfg, "db": "H3 (spatial_h3.duckdb)",
                     "size_mb": s["h3_db_size_mb"]})
        rows.append({"config": cfg, "db": "Geo (spatial_geo.duckdb)",
                     "size_mb": s["db_size_mb"]})
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.barplot(df, x="config", y="size_mb", hue="db", ax=axes[0])
    axes[0].set_title("Database size")
    axes[0].set_ylabel("MB")

    rows = []
    for cfg, s in data["configs"].items():
        for phase in ("read_s", "buffer_s", "db_create_s"):
            rows.append({"config": cfg, "phase": phase, "s": s.get(phase, 0)})
    sns.barplot(pd.DataFrame(rows), x="config", y="s", hue="phase", ax=axes[1])
    axes[1].set_title("Geometric build phases (H3 build reported separately)")
    axes[1].set_ylabel("Seconds")
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "bench_build_size.png", dpi=DPI)
    plt.close(fig)
    df.to_csv(C.TABLES_DIR / "bench_db_sizes.csv", index=False)
    print("  bench_build_size.png")


def plot_micro() -> None:
    p = C.TABLES_DIR / "micro.csv"
    if not p.exists():
        print("  [skip] micro.csv fehlt")
        return
    df = pd.read_csv(p)
    # Messungsnamen fuer die Abbildung ins Englische (Tabelle bleibt unveraendert).
    df["measurement"] = df["measurement"].str.replace("gemeinde", "municipality", regex=False)
    single = df[~df["measurement"].str.startswith("batch_")]
    fig, axes = plt.subplots(len(C.CONFIGS), 1, figsize=(13, 5 * len(C.CONFIGS)),
                             sharex=True)
    for ax, cfg in zip(axes, C.CONFIGS):
        sub = single[single["config"] == cfg]
        sns.barplot(sub, x="measurement", y="median_ms", hue="system", ax=ax)
        ax.set_yscale("log")
        ax.set_title(f"Single-query latencies {cfg} (median, log)")
        ax.set_ylabel("ms")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "bench_micro.png", dpi=DPI)
    plt.close(fig)

    batch = df[df["measurement"].str.startswith("batch_")].copy()
    if not batch.empty:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        batch["label"] = batch["config"] + " " + batch["measurement"]
        sns.barplot(batch, x="label", y="per_feature_ms", hue="system", ax=ax)
        ax.set_yscale("log")
        ax.set_title("Batch queries: ms per feature (median, log)")
        ax.set_ylabel("ms per feature"); ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(C.PLOTS_DIR / "bench_micro_batch.png", dpi=DPI)
        plt.close(fig)
    print("  bench_micro.png / bench_micro_batch.png")


def plot_descriptions() -> None:
    data = _load("descriptions")
    if not data:
        return
    rows = []
    for cfg, c in data["configs"].items():
        for system, t in c["systems"].items():
            rows.append({"config": cfg, "system": system, "pfad": "batch",
                         "ms_pro_feature": t["batch_per_feature_ms"]})
            rows.append({"config": cfg, "system": system, "pfad": "einzel",
                         "ms_pro_feature": t["single_median_ms"]})
    df = pd.DataFrame(rows)
    plot_df = df.rename(columns={"pfad": "path", "ms_pro_feature": "ms per feature"})
    plot_df["path"] = plot_df["path"].replace({"einzel": "single"})
    g = sns.catplot(plot_df, x="config", y="ms per feature", hue="system",
                    col="path", kind="bar", height=4.5, aspect=1.1)
    g.set(yscale="log")
    g.figure.suptitle("Description generation: ms per feature (median, log)", y=1.03)
    g.savefig(C.PLOTS_DIR / "bench_descriptions.png", dpi=DPI)
    plt.close(g.figure)
    df.to_csv(C.TABLES_DIR / "bench_descriptions.csv", index=False)
    print("  bench_descriptions.png")


def plot_throughput() -> None:
    data = _load("throughput")
    if not data:
        return
    rows = []
    for system, sets in data["systems"].items():
        for set_name, v in sets.items():
            rows.append({"system": system, "set": set_name,
                         "seconds": v["seconds"],
                         "factor": v["factor_vs_default"]})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(df, x="system", y="seconds", hue="set", ax=ax)
    ax.set_title(
        f"End-to-end predict(): seconds per {data['n_docs_per_set']} documents "
        f"(model {data['model']})"
    )
    # Faktor-Annotation (Mittel beider Sets)
    means = df.groupby("system", sort=False).agg(
        s=("seconds", "mean"), f=("factor", "mean"))
    for i, (system, row) in enumerate(means.iterrows()):
        ax.annotate(f"{row.f:.1f}x", (i, row.s), ha="center",
                    va="bottom", fontweight="bold")
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "bench_throughput.png", dpi=DPI)
    plt.close(fig)
    df.to_csv(C.TABLES_DIR / "bench_throughput.csv", index=False)
    print("  bench_throughput.png")


def plot_acc_delta() -> None:
    p = C.TABLES_DIR / "acc_geo_delta.csv"
    if not p.exists():
        print("  [skip] acc_geo_delta.csv fehlt")
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.barplot(df, x="model", y="delta_acc1", hue="geo", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Δ Acc@1: geometric twin minus H3 (same models and items)")
    ax.set_ylabel("Δ Acc@1 (geo − h3)")
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "bench_acc_delta.png", dpi=DPI)
    plt.close(fig)
    print("  bench_acc_delta.png")


def main() -> None:
    print("[06] Report ...")
    plot_build()
    plot_micro()
    plot_descriptions()
    plot_throughput()
    plot_acc_delta()
    par = _load("parity")
    if par:
        pd.DataFrame(par["configs"]).T.to_csv(C.TABLES_DIR / "bench_parity.csv")
        print("  tables/bench_parity.csv")
    print("[06] Fertig.")


if __name__ == "__main__":
    main()
