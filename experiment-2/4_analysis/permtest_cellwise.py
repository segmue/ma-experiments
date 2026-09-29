#!/usr/bin/env python3
"""Permutationstest fuer die zellweise Assoziationsmatrix (Variante B).

Testet jede Zelle der gerichteten nPMI-Matrix gegen ein Nullmodell, das die
Kategorieetiketten der Gazetteer-Objekte bei vollstaendig fixer Nachbarschafts-
und Gewichtsstruktur vertauscht (Random Labelling nach Leslie/Kronenfeld 2011).

Das Verfahren in vier Schritten:

  1. Zwischenspeicher: Die Paarlisten (Spalten sf, tf, wgt) werden ueber alle
     Teillaeufe hinweg auf eindeutige Objektpaare aggregiert. Das ist exakt,
     weil W[A,B] linear in den Gewichten ist, und verkuerzt die Paarzahl um
     rund den Faktor sechs.
  2. Beobachtete Matrix: W aus den wahren Etiketten, daraus nPMI nach Bouma
     (zeichengleich zu 2_analysis/common.py:137-153).
  3. B Permutationen: je Durchgang ein neuer Etikettenvektor, daraus W, daraus
     nPMI; gezaehlt wird, wie oft der permutierte Wert mindestens so extrem ist
     wie der beobachtete.
  4. p-Werte als (Treffer + 1) / (B + 1), q-Werte als Benjamini-Hochberg ueber
     die Ausserdiagonalzellen.

Der Test schreibt ausschliesslich neue Dateien. Die Matrix wird weder gefiltert
noch ueberschrieben.

Alle Eingabepfade sind Parameter; nichts ist fest verdrahtet. Der Lauf ist
unterbrechbar und fortsetzbar (Pruefpunkt-Datei) und bei gleichem Startwert
reproduzierbar, auch ueber eine Unterbrechung hinweg.

Aufruf (Kurzform):
  python3 permtest_cellwise.py \
      --pairs '<ordner>/pairs_cellwise_p*.parquet' \
      --pool  '<ordner>/1_data/pool.parquet' \
      --out   '<ausgabeordner>' --B 1000 --seed 7
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Das Mass
# ---------------------------------------------------------------------------
def npmi_from_W(W: np.ndarray) -> np.ndarray:
    """nPMI nach Bouma (2009) auf der Verbundverteilung der Nachbarschaft.

    Zeichengleich zur kanonischen Fassung in 2_analysis/common.py:137-153.
    Zeile = Quellkategorie, Spalte = Zielkategorie; Zellen ohne Gewicht
    bekommen den Boden -1.
    """
    p = W / W.sum()
    pa = p.sum(1, keepdims=True)
    pb = p.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(
            p > 0,
            np.log2(p / (pa * pb)) / (-np.log2(np.where(p > 0, p, 1e-300))),
            -1.0,
        )
    return np.clip(np.nan_to_num(out, nan=-1.0), -1.0, 1.0)


# ---------------------------------------------------------------------------
# Zwischenspeicher: Paarlisten -> drei numpy-Dateien
# ---------------------------------------------------------------------------
def expand_globs(patterns: list[str]) -> list[str]:
    files: list[str] = []
    for pat in patterns:
        hits = sorted(globmod.glob(os.path.expanduser(pat)))
        if not hits:
            raise SystemExit(f"Kein Treffer fuer Muster: {pat}")
        files.extend(hits)
    if len(set(files)) != len(files):
        raise SystemExit("Doppelte Paarlisten in der Eingabe.")
    return files


def input_fingerprint(files: list[str]) -> dict:
    """Beschreibt die Eingabedateien so, dass ein Wechsel auffaellt."""
    return {
        "files": [
            {"path": f, "bytes": os.path.getsize(f), "mtime": round(os.path.getmtime(f), 3)}
            for f in files
        ]
    }


def build_cache(files: list[str], cache_dir: Path, mem_limit: str, threads: int,
                tmp_dir: Path, buckets: int, deadline: float, log) -> dict:
    """Aggregiert alle Paarlisten auf eindeutige (sf, tf) und legt sie als
    numpy-Dateien ab: sf (int32), tf (int32), w (float64).

    Die Aggregation ist exakt: W[A,B] ist eine Summe ueber Paare, und das
    Zusammenfassen gleicher Paare vor dem Etikettieren aendert die Summe nicht.
    Summiert wird in doppelter Genauigkeit, die Paarlisten halten wgt als
    FLOAT (einfache Genauigkeit).
    """
    import duckdb

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    con = duckdb.connect(":memory:")
    con.execute(f"SET memory_limit='{mem_limit}'; SET threads={threads};")
    con.execute(f"SET temp_directory='{tmp_dir}';")
    con.execute("SET preserve_insertion_order=false;")

    lst = ",".join(f"'{f}'" for f in files)
    n_raw, w_raw = con.execute(
        f"SELECT count(*), sum(wgt::DOUBLE) FROM read_parquet([{lst}])"
    ).fetchone()
    log(f"[cache] {len(files)} Paarlisten · {n_raw:,} Rohzeilen · Gewichtssumme {w_raw:,.4f}")

    # Die Aggregation laeuft in Eimern ueber sf % nbuckets. Ein einziger
    # GROUP BY ueber alle 213 Mio. Zeilen laesst DuckDB mehrere Gigabyte auf
    # Platte auslagern; eimerweise bleibt der Bedarf klein. Jeder fertige Eimer
    # wird sofort abgelegt, damit auch der Aufbau des Zwischenspeichers
    # fortsetzbar ist, wenn der Lauf unterbrochen wird.
    for b in range(buckets):
        bf = cache_dir / f"bucket_{b:02d}.npz"
        if bf.exists():
            log(f"[cache] Eimer {b + 1}/{buckets}: schon vorhanden")
            continue
        tb = con.execute(
            f"""SELECT sf::INTEGER AS sf, tf::INTEGER AS tf, sum(wgt::DOUBLE) AS w
                FROM read_parquet([{lst}]) WHERE sf % {buckets} = {b} GROUP BY 1, 2"""
        ).fetch_arrow_table()
        bsf = np.array(tb.column("sf").to_numpy(zero_copy_only=False), np.int32)
        btf = np.array(tb.column("tf").to_numpy(zero_copy_only=False), np.int32)
        bw = np.array(tb.column("w").to_numpy(zero_copy_only=False), np.float64)
        del tb
        tmpf = cache_dir / f"bucket_{b:02d}.tmp.npz"
        np.savez(tmpf, sf=bsf, tf=btf, w=bw)
        os.replace(tmpf, bf)
        log(f"[cache] Eimer {b + 1}/{buckets}: {bsf.size:,} eindeutige Paare abgelegt")
        del bsf, btf, bw
        if deadline and time.time() > deadline:
            con.close()
            raise SystemExit(
                "[halt] Zeitgrenze beim Aufbau des Zwischenspeichers erreicht. "
                "Derselbe Aufruf setzt beim naechsten Eimer fort."
            )
    con.close()

    parts = [np.load(cache_dir / f"bucket_{b:02d}.npz") for b in range(buckets)]
    sf = np.concatenate([z["sf"] for z in parts])
    tf = np.concatenate([z["tf"] for z in parts])
    w = np.concatenate([z["w"] for z in parts])
    for z in parts:
        z.close()
    del parts

    np.save(cache_dir / "sf.npy", sf)
    np.save(cache_dir / "tf.npy", tf)
    np.save(cache_dir / "w.npy", w)
    meta = {
        "n_raw_rows": int(n_raw),
        "w_sum_raw": float(w_raw),
        "n_pairs": int(sf.size),
        "w_sum_agg": float(w.sum()),
        "max_feature_id": int(max(sf.max(), tf.max())),
        "n_self_pairs": int((sf == tf).sum()),
        "seconds": round(time.time() - t0, 1),
        "input": input_fingerprint(files),
    }
    (cache_dir / "cache_meta.json").write_text(json.dumps(meta, indent=2))
    for b in range(buckets):
        (cache_dir / f"bucket_{b:02d}.npz").unlink(missing_ok=True)
    log(f"[cache] {meta['n_pairs']:,} eindeutige Objektpaare "
        f"(Faktor {n_raw / max(sf.size, 1):.2f}) · Gewichtssumme {meta['w_sum_agg']:,.4f} "
        f"· {meta['seconds']}s")
    return meta


def load_cache(cache_dir: Path, files: list[str], log):
    meta = json.loads((cache_dir / "cache_meta.json").read_text())
    if meta["input"] != input_fingerprint(files):
        raise SystemExit(
            "Der Zwischenspeicher stammt aus anderen Paarlisten (Pfad, Groesse oder "
            "Zeitstempel weichen ab). Mit --rebuild-cache neu bauen."
        )
    sf = np.load(cache_dir / "sf.npy", mmap_mode="r")
    tf = np.load(cache_dir / "tf.npy", mmap_mode="r")
    w = np.load(cache_dir / "w.npy", mmap_mode="r")
    log(f"[cache] geladen: {sf.size:,} Objektpaare aus {cache_dir}")
    return sf, tf, w, meta


# ---------------------------------------------------------------------------
# Etiketten
# ---------------------------------------------------------------------------
def read_labels(pool_path: str, max_fid: int, log):
    """Liest pool.parquet und gibt (Kategorienliste, Etikettenvektor) zurueck.

    Die Kategorienordnung ist die aufsteigende Sortierung der OBJEKTART-Werte.
    Sie ist damit dieselbe wie in 06_neighbours_cellwise.py:51/53-54
    (SELECT DISTINCT ... ORDER BY 1 beziehungsweise dense_rank() OVER (ORDER BY OBJEKTART)).
    """
    import duckdb

    con = duckdb.connect(":memory:")
    tab = con.execute(
        f"SELECT feature_id, OBJEKTART FROM read_parquet('{pool_path}')"
    ).fetch_arrow_table()
    fid = tab.column("feature_id").to_numpy().astype(np.int64)
    oa = np.asarray(tab.column("OBJEKTART").to_pylist(), dtype=object)
    con.close()
    cats = sorted(set(oa.tolist()))
    ci = {c: i for i, c in enumerate(cats)}
    size = int(max(fid.max(), max_fid)) + 1
    lab = np.full(size, -1, np.int16)
    lab[fid] = np.asarray([ci[x] for x in oa], np.int16)
    n_missing = int((lab < 0).sum())
    log(f"[etiketten] {fid.size:,} Objekte · {len(cats)} Kategorien "
        f"· Vektorlaenge {size:,} · ohne Etikett {n_missing:,}")
    return cats, lab, fid


def permute_labels(lab: np.ndarray, present: np.ndarray, strata: np.ndarray | None,
                   rng: np.random.Generator) -> np.ndarray:
    """Vertauscht die Etiketten der vorhandenen Objekte.

    Freies Nullmodell: eine Permutation ueber alle Objekte.
    Geschichtetes Nullmodell: eine Permutation je Schicht, also nur zwischen
    Objekten vergleichbarer Zellzahl.
    """
    out = lab.copy()
    if strata is None:
        out[present] = lab[present][rng.permutation(present.size)]
        return out
    for _, idx in strata:
        out[idx] = lab[idx][rng.permutation(idx.size)]
    return out


def make_strata(present: np.ndarray, m_a: np.ndarray, edges: list[int], log):
    """Schichtet die Objekte nach ihrer Zellzahl M_a.

    Die Grenzen sind Zweierpotenzen und keine Quantile. Quantile scheitern hier,
    weil 76.4 Prozent der Objekte genau eine Repraesentantenzelle haben: Die
    unteren Quantile fallen alle auf den Wert eins zusammen, und es bleiben drei
    statt zehn Schichten uebrig. Geometrische Grenzen trennen dagegen sauber
    zwischen Punkt, kleinem Flaechenobjekt und ausgedehntem Objekt.
    """
    v = m_a[present]
    b = np.digitize(v, np.asarray(edges, np.int64), right=True)
    strata = []
    for k in np.unique(b):
        idx = present[b == k]
        if idx.size >= 2:
            strata.append((int(k), idx))
        elif idx.size == 1:
            log(f"[schichten] Schicht {k} hat ein einziges Objekt und bleibt fix.")
    lo = 0
    for k, idx in strata:
        hi = edges[k] if k < len(edges) else -1
        log(f"[schichten] M_a in ({lo}, {hi if hi > 0 else 'oo'}]: {idx.size:,} Objekte")
        lo = hi
    return strata


# ---------------------------------------------------------------------------
# W bauen
# ---------------------------------------------------------------------------
def build_W(lab: np.ndarray, sf, tf, w, n: int, chunk: int) -> np.ndarray:
    """Baut die gerichtete Gewichtsmatrix aus Paarliste und Etikettenvektor.

    Blockweise, damit die Zwischenarrays klein bleiben: je Block entstehen nur
    Block x 8 Byte an Hilfsspeicher, unabhaengig von der Gesamtzahl der Paare.
    """
    flat = np.zeros(n * n, np.float64)
    total = sf.shape[0]
    for a in range(0, total, chunk):
        b = min(a + chunk, total)
        s = lab[sf[a:b]].astype(np.int32)
        t = lab[tf[a:b]].astype(np.int32)
        code = s * n + t
        flat += np.bincount(code, weights=np.asarray(w[a:b]), minlength=n * n)
    return flat.reshape(n, n)


# ---------------------------------------------------------------------------
# Benjamini-Hochberg
# ---------------------------------------------------------------------------
def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH-Korrektur ueber einen eindimensionalen p-Wert-Vektor."""
    m = p.size
    order = np.argsort(p, kind="stable")
    ranked = p[order] * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m, np.float64)
    q[order] = np.clip(q_sorted, 0.0, 1.0)
    return q


# ---------------------------------------------------------------------------
# Hauptlauf
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="Paarlisten als Pfad oder Glob-Muster (Spalten sf, tf, wgt).")
    ap.add_argument("--pool", required=True,
                    help="pool.parquet mit feature_id und OBJEKTART.")
    ap.add_argument("--out", required=True, help="Ausgabeordner (wird angelegt).")
    ap.add_argument("--B", type=int, default=1000, help="Zahl der Permutationen.")
    ap.add_argument("--seed", type=int, default=7, help="Zufallsstartwert.")
    ap.add_argument("--null", choices=["frei", "geschichtet"], default="frei",
                    help="Nullmodell: freie Etikettenpermutation oder nach Zellzahl geschichtet.")
    ap.add_argument("--sample-index", default=None,
                    help="sample_index.parquet; nur fuer --null geschichtet noetig.")
    ap.add_argument("--strata-edges", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128],
                    help="Obere Grenzen der Zellzahl-Schichten (geschichtetes Nullmodell).")
    ap.add_argument("--cache-dir", default=None,
                    help="Ordner fuer den Paar-Zwischenspeicher (Default: <out>/cache).")
    ap.add_argument("--placebo", type=int, default=-1,
                    help="Eichprobe: statt der wahren Etiketten eine eigene Zufallspermutation "
                         "als 'beobachtet' einsetzen. Die p-Werte muessen dann annaehernd "
                         "gleichverteilt sein. Wert >= 0 waehlt die Permutation.")
    ap.add_argument("--reset-checkpoint", action="store_true",
                    help="Vorhandenen Pruefpunkt verwerfen und von vorn zaehlen.")
    ap.add_argument("--rebuild-cache", action="store_true",
                    help="Zwischenspeicher neu bauen, auch wenn er vorhanden ist.")
    ap.add_argument("--chunk", type=int, default=4_000_000,
                    help="Blockgroesse in Paaren beim Matrixbau.")
    ap.add_argument("--checkpoint-every", type=int, default=25,
                    help="Pruefpunkt nach je so vielen Permutationen.")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="Nach so vielen Sekunden geordnet anhalten (0 = kein Limit).")
    ap.add_argument("--duckdb-memory", default="1500MB", help="Speichergrenze fuer DuckDB.")
    ap.add_argument("--duckdb-threads", type=int, default=4, help="Threads fuer DuckDB.")
    ap.add_argument("--tmp-dir", default=None, help="Auslagerungsordner fuer DuckDB.")
    ap.add_argument("--cache-buckets", type=int, default=8,
                    help="Zahl der Eimer beim Aggregieren der Paarlisten (mehr = weniger Speicher).")
    ap.add_argument("--check-w", nargs="*", default=None,
                    help="W_cellwise_p*.npy zur Gegenprobe der beobachteten Matrix.")
    ap.add_argument("--check-matrix", default=None,
                    help="npmi_dist_matrix_B.csv zur Gegenprobe der beobachteten nPMI-Matrix.")
    ap.add_argument("--tag", default="B", help="Namenszusatz der Ausgabedateien.")
    a = ap.parse_args()

    out = Path(os.path.expanduser(a.out))
    out.mkdir(parents=True, exist_ok=True)
    logfile = out / f"permtest_{a.tag}_log.txt"

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        with logfile.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    t_start = time.time()
    log("=" * 78)
    log(f"[start] {' '.join(sys.argv)}")

    files = expand_globs(a.pairs)
    cache_dir = Path(os.path.expanduser(a.cache_dir)) if a.cache_dir else out / "cache"
    tmp_dir = Path(os.path.expanduser(a.tmp_dir)) if a.tmp_dir else cache_dir / "tmp"

    if a.rebuild_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)
    if not (cache_dir / "cache_meta.json").exists():
        deadline = (t_start + a.max_seconds) if a.max_seconds else 0.0
        build_cache(files, cache_dir, a.duckdb_memory, a.duckdb_threads, tmp_dir,
                    a.cache_buckets, deadline, log)
    sf, tf, w, cmeta = load_cache(cache_dir, files, log)

    cats, lab, fid = read_labels(a.pool, cmeta["max_feature_id"], log)
    n = len(cats)
    present = np.sort(fid).astype(np.int64)

    if int((lab < 0).sum()):
        # Nur pruefen, wenn es ueberhaupt Luecken gibt: Ein Objekt, das in der
        # Paarliste vorkommt, aber nicht im Pool steht, wuerde sonst still ein
        # falsches Etikett bekommen.
        fehl = int((lab[np.unique(np.asarray(sf))] < 0).sum()
                   + (lab[np.unique(np.asarray(tf))] < 0).sum())
        if fehl:
            raise SystemExit(
                f"{fehl} in den Paarlisten referenzierte Objekte fehlen im Pool. "
                f"Paarlisten und pool.parquet gehoeren nicht zusammen."
            )

    strata = None
    if a.null == "geschichtet":
        if not a.sample_index:
            raise SystemExit("--null geschichtet verlangt --sample-index.")
        import duckdb
        con = duckdb.connect(":memory:")
        tb = con.execute(
            f"SELECT feature_id, count(*) AS m FROM read_parquet('{a.sample_index}') GROUP BY 1"
        ).fetch_arrow_table()
        con.close()
        m_a = np.zeros(lab.size, np.int64)
        m_a[tb.column("feature_id").to_numpy().astype(np.int64)] = tb.column("m").to_numpy()
        strata = make_strata(present, m_a, list(a.strata_edges), log)

    # --- beobachtete Matrix -------------------------------------------------
    if a.placebo >= 0:
        # Eichprobe: Die "beobachtete" Matrix stammt selbst aus einer Permutation.
        # Die Nullhypothese ist dann wahr, und die p-Werte muessen gleichverteilt
        # sein. Der Zufallsstrom liegt weit ausserhalb der Teststroeme.
        rng0 = np.random.default_rng([a.seed, 10 ** 9 + a.placebo])
        lab = permute_labels(lab, present, strata, rng0)
        log(f"[eichprobe] Etiketten vor dem Test einmal permutiert (Placebo {a.placebo}).")
    t0 = time.time()
    W_obs = build_W(lab, sf, tf, w, n, a.chunk)
    M_obs = npmi_from_W(W_obs)
    log(f"[beobachtet] W-Summe {W_obs.sum():,.4f} · nPMI-Bereich "
        f"[{M_obs.min():+.4f}, {M_obs.max():+.4f}] · Matrixbau {time.time() - t0:.2f}s")

    checks: dict = {}
    if a.check_w:
        ref = sum(np.load(p) for p in expand_globs(list(a.check_w)))
        d = float(np.abs(W_obs - ref).max())
        rel = float(abs(W_obs.sum() - ref.sum()) / ref.sum())
        checks["W_vs_npy"] = {"max_abs_diff": d, "rel_diff_total": rel,
                              "ref_total": float(ref.sum())}
        log(f"[gegenprobe] W gegen abgelegte Teilmatrizen: groesste Abweichung {d:.6g}, "
            f"relative Abweichung der Summe {rel:.3e}")
    if a.check_matrix:
        import pandas as pd
        ref = pd.read_csv(a.check_matrix, sep=";", index_col=0)
        same_cats = list(ref.index) == cats
        d = float(np.abs(M_obs - ref.to_numpy()).max())
        checks["npmi_vs_csv"] = {"max_abs_diff": d, "kategorien_identisch": bool(same_cats)}
        log(f"[gegenprobe] nPMI gegen {Path(a.check_matrix).name}: groesste Abweichung "
            f"{d:.6g} · Kategorienordnung identisch: {same_cats}")

    # --- Zaehler und Pruefpunkt --------------------------------------------
    ck = out / f"permtest_{a.tag}_checkpoint.npz"
    cfg = json.dumps({"B_seed": a.seed, "null": a.null, "placebo": a.placebo, "edges": list(a.strata_edges),
                      "pairs": cmeta["input"], "pool": a.pool, "n": n}, sort_keys=True)
    ge_abs = np.zeros((n, n), np.int64)   # |nPMI_perm| >= |nPMI_beob|
    ge_hi = np.zeros((n, n), np.int64)    # nPMI_perm  >= nPMI_beob
    le_lo = np.zeros((n, n), np.int64)    # nPMI_perm  <= nPMI_beob
    psum = np.zeros((n, n), np.float64)
    psum2 = np.zeros((n, n), np.float64)
    done = 0
    if ck.exists() and a.reset_checkpoint:
        log("[fortsetzung] --reset-checkpoint: vorhandener Pruefpunkt wird verworfen.")
    elif ck.exists():
        z = np.load(ck, allow_pickle=False)
        if str(z["cfg"]) != cfg:
            raise SystemExit(
                f"Der Pruefpunkt {ck} gehoert zu einer anderen Konfiguration. "
                f"Entweder dieselbe Konfiguration verwenden oder die Datei wegnehmen."
            )
        ge_abs, ge_hi, le_lo = z["ge_abs"], z["ge_hi"], z["le_lo"]
        psum, psum2 = z["psum"], z["psum2"]
        done = int(z["done"])
        log(f"[fortsetzung] Pruefpunkt gefunden: {done} von {a.B} Permutationen erledigt")

    dev = np.abs(M_obs)

    def save_checkpoint(k: int) -> None:
        tmp = ck.with_name(ck.stem + ".tmp.npz")
        np.savez(tmp, ge_abs=ge_abs, ge_hi=ge_hi, le_lo=le_lo, psum=psum, psum2=psum2,
                 done=np.int64(k), cfg=np.str_(cfg))
        os.replace(tmp, ck)

    # --- Permutationen ------------------------------------------------------
    t_loop = time.time()
    per_perm: list[float] = []
    stopped_early = False
    for i in range(done, a.B):
        ti = time.time()
        rng = np.random.default_rng([a.seed, i])   # je Index eigener Strom -> fortsetzbar
        lp = permute_labels(lab, present, strata, rng)
        Mp = npmi_from_W(build_W(lp, sf, tf, w, n, a.chunk))
        ge_abs += (np.abs(Mp) >= dev)
        ge_hi += (Mp >= M_obs)
        le_lo += (Mp <= M_obs)
        psum += Mp
        psum2 += Mp * Mp
        per_perm.append(time.time() - ti)
        k = i + 1
        if k % a.checkpoint_every == 0 or k == a.B:
            save_checkpoint(k)
            sp = float(np.mean(per_perm[-a.checkpoint_every:]))
            rest = (a.B - k) * sp
            log(f"  [perm] {k}/{a.B} · {sp:.3f}s je Permutation · "
                f"verbleibend rund {rest / 60:.1f} min · Pruefpunkt geschrieben")
        if a.max_seconds and (time.time() - t_start) > a.max_seconds:
            save_checkpoint(k)
            log(f"[halt] Zeitgrenze erreicht nach {k} Permutationen. "
                f"Derselbe Aufruf setzt fort.")
            stopped_early = True
            break

    if stopped_early:
        log(f"[ende] vorzeitig angehalten nach {time.time() - t_start:.0f}s")
        return

    done_total = a.B
    sp_mean = float(np.mean(per_perm)) if per_perm else float("nan")

    # --- p- und q-Werte -----------------------------------------------------
    B = done_total
    p_abs = (ge_abs + 1.0) / (B + 1.0)
    p_hi = (ge_hi + 1.0) / (B + 1.0)
    p_lo = (le_lo + 1.0) / (B + 1.0)
    p_tails = np.clip(2.0 * np.minimum(p_hi, p_lo), 0.0, 1.0)

    off = ~np.eye(n, dtype=bool)
    q_abs = np.full((n, n), np.nan)
    q_abs[off] = benjamini_hochberg(p_abs[off])
    q_tails = np.full((n, n), np.nan)
    q_tails[off] = benjamini_hochberg(p_tails[off])

    perm_mean = psum / B
    perm_sd = np.sqrt(np.maximum(psum2 / B - perm_mean ** 2, 0.0))

    import pandas as pd

    def wr(mat, name):
        pd.DataFrame(mat, index=cats, columns=cats).to_csv(out / name, sep=";")

    sfx = f"_{a.tag}"
    wr(p_abs, f"npmi_dist_pvalues{sfx}.csv")
    wr(q_abs, f"npmi_dist_qvalues{sfx}.csv")
    wr(p_tails, f"npmi_dist_pvalues_zweiseitig{sfx}.csv")
    wr(q_tails, f"npmi_dist_qvalues_zweiseitig{sfx}.csv")
    wr(perm_mean, f"npmi_dist_nullmittel{sfx}.csv")
    wr(perm_sd, f"npmi_dist_nullstreuung{sfx}.csv")
    wr(M_obs, f"npmi_dist_matrix_nachgerechnet{sfx}.csv")

    meta = {
        "tag": a.tag,
        "nullmodell": a.null,
        "placebo": a.placebo if a.placebo >= 0 else None,
        "strata_edges": list(a.strata_edges) if a.null == "geschichtet" else None,
        "n_permutationen": B,
        "seed": a.seed,
        "n_kategorien": n,
        "pairs": files,
        "pool": a.pool,
        "cache": cmeta,
        "W_total": float(W_obs.sum()),
        "npmi_min": float(M_obs.min()),
        "npmi_max": float(M_obs.max()),
        "bodenanteil_ausserdiagonal": float(np.mean(M_obs[off] <= -0.999)),
        "median_diagonale": float(np.median(np.diag(M_obs))),
        "p_minimum_moeglich": 1.0 / (B + 1.0),
        "n_signifikant_q05_betrag": int((q_abs[off] <= 0.05).sum()),
        "n_signifikant_q05_zweiseitig": int((q_tails[off] <= 0.05).sum()),
        "n_ausserdiagonalzellen": int(off.sum()),
        "nullmittel_median": float(np.median(perm_mean[off])),
        "nullmittel_max_betrag": float(np.abs(perm_mean[off]).max()),
        "sekunden_je_permutation": round(sp_mean, 4),
        "sekunden_gesamt": round(time.time() - t_start, 1),
        "gegenproben": checks,
    }
    (out / f"permtest_meta{sfx}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    log(f"[fertig] B={B} · {sp_mean:.3f}s je Permutation · "
        f"signifikant (q<=0.05, Betrag) {meta['n_signifikant_q05_betrag']:,} von "
        f"{meta['n_ausserdiagonalzellen']:,} · Gesamtzeit {time.time() - t_start:.0f}s")
    log(f"[fertig] geschrieben nach {out}")


if __name__ == "__main__":
    main()
