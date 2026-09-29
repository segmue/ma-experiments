"""
Nachrechnung C — die Nachlaufgroessen zur vierten Beschreibungsvariante E_d1.

Rechnet allein auf den Per-Item-Abzuegen zweier Laeufe; kein Encoder-Forward,
kein Resolver-Durchlauf. Laufzeit rund 2 Minuten.

Quellen (beide Seiten per-Item, identische Itemmenge und Kandidatenmenge):
  A) Hauptmatrix (E_default/E_c1/E_c2) und B) E_d1-Arme, beide aus
     3_evaluation/02_evaluate_matrix.py: 3_evaluation/cache/per_item/<M>_<E>.pkl.gz,
     Zell-JSONs in results/matrix/.

Zugehoerigkeitsbeleg: die fuenf E_c1-Zellen des Windows-Laufs sind mit dem
Mac-Bestand in jedem inhaltlichen Feld zeichengleich (AUSWERTUNG_E_d1.md 1.2).
Dieses Skript prueft das an den Aggregaten nochmals selbst (Torwaechter T2)
und reproduziert zusaetzlich einen bekannten McNemar-Vergleich (T3).

Der Arm Z, M und M2 beruhen auf einer defekten Matrix und werden hier
nirgends angefasst.

Ausgabe: ERGEBNIS_C_nachlauf.json und ERGEBNIS_C_nachlauf.txt im Ausgabeordner
(Default results/). Aufruf: python nachrechnung_C_nachlauf.py [AUSGABEORDNER]
"""
from __future__ import annotations

import gzip, json, math, pickle, sqlite3, sys
from collections import defaultdict
from pathlib import Path

# Pfade des Repos: 3_evaluation/config.py (nur pathlib, keine Rechenabhaengigkeiten).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3_evaluation"))
import config as C  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else C.RESULTS_DIR
OUT.mkdir(parents=True, exist_ok=True)

PER_ALT = C.PER_ITEM_DIR          # per-Item-Dumps (Hauptmatrix und E_d1-Arme)
NEU = C.PER_ITEM_DIR
MATRIX = C.MATRIX_DIR             # Zell-JSONs
GPKG = C.CORPUS_GPKG

MODELS = ["M1_dguzh", "M2_distiluse_base", "M3_default_finetuned",
          "M4_spatial_config1", "M5_spatial_config2"]

log_lines: list[str] = []
def log(s=""):
    print(s)
    log_lines.append(str(s))


# ── Laden ────────────────────────────────────────────────────────────────────
_cache: dict[tuple[str, str], list[dict]] = {}

def per_item(model: str, arm: str) -> list[dict]:
    if (model, arm) in _cache:
        return _cache[(model, arm)]
    if arm in ("E_default", "E_c1", "E_c2"):
        p = PER_ALT / f"{model}_{arm}.pkl.gz"
    else:
        p = NEU / f"{model}_{arm}.pkl.gz"
    if not p.exists():
        raise FileNotFoundError(p)
    with gzip.open(p, "rb") as fh:
        d = pickle.load(fh)
    _cache[(model, arm)] = d
    return d

def zelle_json(model: str, arm: str) -> dict:
    p = MATRIX / f"{model}_{arm}.json"
    return json.loads(p.read_text(encoding="utf-8"))


# ── Metriken, wortgleich zu eval_core.py / 02_evaluate_matrix.py ─────────────
def metriken(pi: list[dict]) -> dict:
    total = len(pi)
    c1 = sum(1 for p in pi if p["rank"] == 1)
    c3 = sum(1 for p in pi if p["rank"] is not None and p["rank"] <= 3)
    rr = sum(p["rr"] for p in pi)
    wc = [p for p in pi if p["n_candidates"] > 0]
    nc = len(wc)
    cu = sum(1 for p in pi if p["rank"] == 1 and p["n_candidates"] == 1)
    ca = sum(1 for p in pi if p["rank"] == 1 and p["n_candidates"] > 1)
    at = sum(1 for p in pi if p["n_candidates"] > 1)
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for p in pi:
        by_doc[p["doc_id"]].append(p)
    docacc = sum(sum(1 for q in v if q["rank"] == 1) / len(v) for v in by_doc.values()) / len(by_doc)
    return {
        "total": total, "no_candidate": sum(1 for p in pi if p["n_candidates"] == 0),
        "acc1": c1 / total, "acc3": c3 / total, "mrr": rr / total,
        "acc1_with_cand": sum(1 for p in wc if p["rank"] == 1) / nc,
        "acc3_with_cand": sum(1 for p in wc if p["rank"] is not None and p["rank"] <= 3) / nc,
        "n_with_cand": nc,
        "doc_accuracy": docacc, "n_documents": len(by_doc),
        "correct": cu + ca, "correct_unambiguous": cu, "correct_ambiguous": ca,
        "unambiguous_total": sum(1 for p in pi if p["n_candidates"] == 1),
        "ambiguous_total": at, "acc1_ambiguous_only": ca / at,
        "wrong_rank": sum(1 for p in pi if p["rank"] is not None and p["rank"] > 1),
        "gold_not_in_candidates": sum(1 for p in pi if p["n_candidates"] > 0 and p["rank"] is None),
    }


# ── McNemar, wortgleich zu 12_mcnemar_main.py ────────────────────────────────
def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + n * math.log(0.5)
            for i in range(k + 1)]
    m = max(logs)
    return min(1.0, 2.0 * math.exp(m) * sum(math.exp(x - m) for x in logs))

def schluessel(it: dict) -> tuple:
    return (it["doc_id"], it["start"], it["end"], it["gold_id"])

def vergleich(a: list[dict], b: list[dict], subset: str) -> dict:
    am = {schluessel(i): i for i in a}
    bm = {schluessel(i): i for i in b}
    geteilt = am.keys() & bm.keys()
    only_a = only_b = beide = keins = 0
    disk_ambig = disk_gesamt = 0
    for k in geteilt:
        ia, ib = am[k], bm[k]
        if subset == "ambiguous" and (ia.get("n_candidates") or 0) <= 1:
            continue
        ca, cb = ia.get("rank") == 1, ib.get("rank") == 1
        if ca and cb: beide += 1
        elif ca:
            only_a += 1
            disk_gesamt += 1
            disk_ambig += (ia.get("n_candidates") or 0) > 1
        elif cb:
            only_b += 1
            disk_gesamt += 1
            disk_ambig += (ia.get("n_candidates") or 0) > 1
        else: keins += 1
    n = beide + only_a + only_b + keins
    return {"subset": subset, "n_items": n, "n_paare": len(geteilt),
            "acc_a": (beide + only_a) / n, "acc_b": (beide + only_b) / n,
            "only_a_correct": only_a, "only_b_correct": only_b,
            "discordant": only_a + only_b, "mcnemar_p": exact_mcnemar_p(only_a, only_b),
            "alle_diskordanten_items_sind_ambig": disk_ambig == disk_gesamt}

def paar(ma, ra, mb, rb, label) -> dict:
    a, b = per_item(ma, ra), per_item(mb, rb)
    e = {"label": label, "system_a": f"{ma}/{ra}", "system_b": f"{mb}/{rb}",
         "all": vergleich(a, b, "all"), "ambiguous": vergleich(a, b, "ambiguous")}
    for s in ("all", "ambiguous"):
        r = e[s]
        log(f"  {label:44s} [{s:9s}] {r['acc_a']:.4f} / {r['acc_b']:.4f}  "
            f"{r['only_a_correct']:5d}/{r['only_b_correct']:5d}  d={r['discordant']:5d}  "
            f"p={r['mcnemar_p']:.4g}")
    return e


ergebnis: dict = {"_meta": {
    "skript": "nachrechnung_C_nachlauf.py", "datum": "2026-09-17",
    "grundlage": "Per-Item-Abzuege Mac-Lauf 08.07.2026 und Windows-Lauf 17.09.2026",
    "verfahren": "Exakter McNemar (zweiseitiger Binomialtest auf den diskordanten Paaren), "
                 "gepaart ueber (doc_id, start, end, gold_id); Metriken wortgleich zu "
                 "eval_core.py und 02_evaluate_matrix.py"}}

# ── Torwaechter ──────────────────────────────────────────────────────────────
log("=" * 78); log("TORWAECHTER"); log("=" * 78)

log("\nT1  Grundgesamtheit je Zelle")
soll = {"total": 75741, "no_candidate": 8159, "ambiguous_total": 23848,
        "unambiguous_total": 43734, "gold_not_in_candidates": 555,
        "correct_unambiguous": 43544, "n_documents": 6323}
t1_fehler = []
ALLE_ZELLEN = [(m, a) for m in MODELS for a in ("E_default", "E_c1", "E_c2", "E_d1")]
mess: dict[str, dict] = {}
for m, a in ALLE_ZELLEN:
    k = metriken(per_item(m, a))
    mess[f"{m}/{a}"] = k
    for f, v in soll.items():
        if k[f] != v:
            t1_fehler.append(f"{m}/{a}: {f} = {k[f]}, erwartet {v}")
if t1_fehler:
    log("  FEHLER:\n   " + "\n   ".join(t1_fehler)); sys.exit(1)
log(f"  ok — alle {len(ALLE_ZELLEN)} Zellen fuehren die sieben Sollwerte")

log("\nT2  Zugehoerigkeit: nachgerechnete Aggregate gegen die Zellendateien")
t2 = []
for m, a in ALLE_ZELLEN:
    j = zelle_json(m, a); k = mess[f"{m}/{a}"]
    for feld, gerechnet in (("accuracy_at_1", k["acc1"]), ("accuracy_at_3", k["acc3"]),
                            ("mrr", k["mrr"]),
                            ("accuracy_at_1_with_candidates", k["acc1_with_cand"])):
        if abs(j["location"][feld] - gerechnet) > 1e-12:
            t2.append(f"{m}/{a}.{feld}: {j['location'][feld]} gegen {gerechnet}")
    if abs(j["errors"]["acc1_ambiguous_only"] - k["acc1_ambiguous_only"]) > 1e-12:
        t2.append(f"{m}/{a}.acc1_ambiguous_only")
    if abs(j["document"]["document_accuracy"] - k["doc_accuracy"]) > 1e-12:
        t2.append(f"{m}/{a}.document_accuracy")
    for feld in ("correct", "wrong_rank", "gold_not_in_candidates"):
        if j["errors"][feld] != k[feld]:
            t2.append(f"{m}/{a}.{feld}: {j['errors'][feld]} gegen {k[feld]}")
if t2:
    log("  FEHLER:\n   " + "\n   ".join(t2)); sys.exit(1)
log(f"  ok — {len(ALLE_ZELLEN)} Zellen zeichengleich mit ihren Zellendateien")

log("\nT3  Reproduktion eines bekannten McNemar-Vergleichs (mcnemar_main.json, Paar 0)")
ref = paar("M5_spatial_config2", "E_c1", "M3_default_finetuned", "E_default",
           "M5/E_c1 gegen M3/E_default (Sollwert)")
if not (ref["all"]["only_a_correct"] == 2417 and ref["all"]["only_b_correct"] == 956
        and abs(ref["all"]["mcnemar_p"] - 4.26e-144) < 1e-146):
    log("  FEHLER: Sollwert 2417/956, p = 4.26e-144 nicht reproduziert"); sys.exit(1)
log("  ok — 2'417 / 956, p = 4.26e-144 zeichengleich reproduziert")
log("\n  Damit ist belegt: die Per-Item-Abzuege beider Laeufe sind paarbar und tragen\n"
    "  dieselbe Itemmenge. Der E_d1-Arm darf gegen den Altbestand gestellt werden.")


# ── 1 Fehlerzerlegung E_d1 ───────────────────────────────────────────────────
log("\n" + "=" * 78); log("1  FEHLERZERLEGUNG E_d1, alle fuenf Modelle"); log("=" * 78)
log(f"\n{'System':28s} {'korrekt':>8s} {'(eind.':>7s} {'/ambig)':>8s} {'falscher Rang':>14s}"
    f" {'Gold n.i.K.':>12s} {'kein Kand.':>11s}")
fz = {}
for m in MODELS:
    k = mess[f"{m}/E_d1"]
    fz[m] = {x: k[x] for x in ("correct", "correct_unambiguous", "correct_ambiguous",
                               "wrong_rank", "gold_not_in_candidates", "no_candidate")}
    fz[m]["wrong_rank_rate"] = k["wrong_rank"] / k["total"]
    log(f"{m + '/E_d1':28s} {k['correct']:8d} {k['correct_unambiguous']:7d} "
        f"{k['correct_ambiguous']:8d} {k['wrong_rank']:8d} ({k['wrong_rank']/k['total']:.1%})"
        f" {k['gold_not_in_candidates']:8d} {k['no_candidate']:11d}")
ergebnis["fehlerzerlegung_E_d1"] = fz


# ── 2 McNemar M5/E_d1 gegen M3/E_default ─────────────────────────────────────
log("\n" + "=" * 78); log("2  GEPAARTER McNEMAR — das neue beste System gegen die faire Baseline")
log("=" * 78 + "\n")
paare = [
    ("M5_spatial_config2", "E_d1", "M3_default_finetuned", "E_default",
     "M5/E_d1 gegen M3/E_default (Hauptvergleich neu)"),
    ("M5_spatial_config2", "E_d1", "M4_spatial_config1", "E_c1",
     "M5/E_d1 gegen M4/E_c1 (Diagonale config1)"),
    ("M5_spatial_config2", "E_d1", "M3_default_finetuned", "E_c1",
     "M5/E_d1 gegen M3/E_c1"),
]
ergebnis["mcnemar"] = {"referenz_T3": ref, "neu": [paar(*p) for p in paare]}


# ── 3 Differenz E_d1 minus E_default ─────────────────────────────────────────
log("\n" + "=" * 78); log("3  DIFFERENZ E_d1 MINUS E_default, alle fuenf Modelle")
log("=" * 78)
log(f"\n{'Modell':22s} {'acc1 E_def':>10s} {'acc1 E_d1':>10s} {'Δ acc1':>9s}"
    f" {'ambig E_def':>11s} {'ambig E_d1':>10s} {'Δ ambig':>9s}")
diffs = {}
for m in MODELS:
    d, n = mess[f"{m}/E_default"], mess[f"{m}/E_d1"]
    diffs[m] = {"acc1_E_default": d["acc1"], "acc1_E_d1": n["acc1"],
                "delta_acc1": n["acc1"] - d["acc1"],
                "acc1_ambiguous_only_E_default": d["acc1_ambiguous_only"],
                "acc1_ambiguous_only_E_d1": n["acc1_ambiguous_only"],
                "delta_acc1_ambiguous_only": n["acc1_ambiguous_only"] - d["acc1_ambiguous_only"]}
    log(f"{m:22s} {d['acc1']:10.4f} {n['acc1']:10.4f} {n['acc1']-d['acc1']:+9.4f}"
        f" {d['acc1_ambiguous_only']:11.4f} {n['acc1_ambiguous_only']:10.4f}"
        f" {n['acc1_ambiguous_only']-d['acc1_ambiguous_only']:+9.4f}")
ergebnis["delta_E_d1_minus_E_default"] = diffs

log("\n  Spaltensprung gegen E_default, alle drei angereicherten Varianten, Gesamtmetrik:")
spruenge = []
for m in MODELS:
    for arm in ("E_c1", "E_c2", "E_d1"):
        spruenge.append((mess[f"{m}/{arm}"]["acc1"] - mess[f"{m}/E_default"]["acc1"], m, arm,
                         mess[f"{m}/{arm}"]["acc1_ambiguous_only"]
                         - mess[f"{m}/E_default"]["acc1_ambiguous_only"]))
for d, m, arm, da in sorted(spruenge, reverse=True)[:4]:
    log(f"    {m}/{arm:10s} {d:+.4f} gesamt   {da:+.4f} ambig")
log(f"  groesster Spaltensprung gesamt: {max(spruenge)[1]}/{max(spruenge)[2]} "
    f"{max(spruenge)[0]:+.4f}")
ergebnis["spaltenspruenge_gegen_E_default"] = [
    {"modell": m, "arm": a, "delta_acc1": d, "delta_acc1_ambiguous_only": da}
    for d, m, a, da in sorted(spruenge, reverse=True)]


# ── 4 Klassenprofil E_d1 ─────────────────────────────────────────────────────
log("\n" + "=" * 78); log("4  KLASSENPROFIL E_d1 — Acc@1 je Objektklasse"); log("=" * 78)
ref_items = per_item("M5_spatial_config2", "E_c1")
gold_ids = {p["gold_id"] for p in ref_items if p["gold_id"]}
mapping: dict[str, str] = {}
con = sqlite3.connect(str(GPKG))
for u, o in con.execute("SELECT sn3d_uuid, sn3d_objektart FROM corpus_toponyms"):
    if u in gold_ids and isinstance(o, str) and o:
        mapping[u] = o
con.close()
log(f"\n  OBJEKTART-Lookup aus corpus_swissnames3d.gpkg: "
    f"{len(mapping)}/{len(gold_ids)} eindeutige Gold-UUIDs ({len(mapping)/len(gold_ids):.1%})")

MIN_N = 150
cls_of = lambda p: mapping.get(p["gold_id"], "UNBEKANNT")
counts: dict[str, int] = defaultdict(int)
for p in ref_items:
    counts[cls_of(p)] += 1
anzeige = {c: (c if n >= MIN_N else "Other") for c, n in counts.items()}

def je_klasse(pi: list[dict]) -> dict:
    agg: dict[str, dict] = defaultdict(lambda: {"n": 0, "c1": 0, "c3": 0, "rr": 0.0, "wr": 0})
    for p in pi:
        a = agg[anzeige[cls_of(p)]]
        a["n"] += 1; a["rr"] += p["rr"]
        if p["rank"] == 1: a["c1"] += 1
        if p["rank"] is not None and p["rank"] <= 3: a["c3"] += 1
        if p["rank"] is not None and p["rank"] > 1: a["wr"] += 1
    return {c: {"n": a["n"], "acc1": a["c1"] / a["n"], "acc3": a["c3"] / a["n"],
                "mrr": a["rr"] / a["n"], "wrong_rank_rate": a["wr"] / a["n"]}
            for c, a in agg.items()}

kp_basis = je_klasse(per_item("M3_default_finetuned", "E_default"))
kp_ec1 = je_klasse(per_item("M5_spatial_config2", "E_c1"))
kp_ed1 = {m: je_klasse(per_item(m, "E_d1")) for m in MODELS}
kp5 = kp_ed1["M5_spatial_config2"]

log(f"\n{'Objektklasse':28s} {'n':>6s} {'M3/E_def':>9s} {'M5/E_c1':>8s} {'M5/E_d1':>8s}"
    f" {'Δ z. Basis':>10s} {'Δ z. E_c1':>10s}")
klassen = sorted((c for c in kp_basis if c != "UNBEKANNT"), key=lambda c: -kp_basis[c]["n"])
tab = []
for c in klassen:
    b, e1, d1 = kp_basis[c]["acc1"], kp_ec1[c]["acc1"], kp5[c]["acc1"]
    tab.append({"objektart": c, "n": kp_basis[c]["n"], "acc1_M3_E_default": b,
                "acc1_M5_E_c1": e1, "acc1_M5_E_d1": d1,
                "delta_E_d1_vs_baseline": d1 - b, "delta_E_d1_vs_E_c1": d1 - e1,
                "mrr_M5_E_d1": kp5[c]["mrr"], "wrong_rank_M5_E_d1": kp5[c]["wrong_rank_rate"]})
    log(f"{c:28s} {kp_basis[c]['n']:6d} {b:9.4f} {e1:8.4f} {d1:8.4f} {d1-b:+10.4f} {d1-e1:+10.4f}")
ergebnis["klassenprofil_E_d1"] = {"min_n": MIN_N, "lookup_abdeckung": len(mapping) / len(gold_ids),
                                  "zeilen": tab,
                                  "alle_modelle_E_d1": {m: kp_ed1[m] for m in MODELS}}
gew = sorted(tab, key=lambda r: -r["delta_E_d1_vs_E_c1"])
log(f"\n  groesster Klassengewinn E_d1 gegen E_c1: {gew[0]['objektart']} "
    f"{gew[0]['delta_E_d1_vs_E_c1']:+.4f} (n = {gew[0]['n']})")
log(f"  groesster Klassenverlust:                {gew[-1]['objektart']} "
    f"{gew[-1]['delta_E_d1_vs_E_c1']:+.4f} (n = {gew[-1]['n']})")


# ── 5 + 6 Volltabellen aller zwanzig Zellen ──────────────────────────────────
log("\n" + "=" * 78); log("5/6  ALLE ZWANZIG ZELLEN — MRR, Acc@3, DocAcc, nur-mit-Kandidaten")
log("=" * 78)
for feld, titel in (("acc3", "Acc@3 ueber 75'741"), ("mrr", "MRR ueber 75'741"),
                    ("doc_accuracy", "Dokumentgenauigkeit ueber 6'323 Artikel"),
                    ("acc1_with_cand", "Acc@1 nur mit Kandidaten (67'582)"),
                    ("acc3_with_cand", "Acc@3 nur mit Kandidaten (67'582)")):
    log(f"\n  {titel}")
    log(f"  {'Modell':22s} {'E_default':>10s} {'E_c1':>10s} {'E_c2':>10s} {'E_d1':>10s}")
    for m in MODELS:
        log(f"  {m:22s}" + "".join(f" {mess[f'{m}/{a}'][feld]:10.4f}"
                                   for a in ("E_default", "E_c1", "E_c2", "E_d1")))
ergebnis["alle_zellen"] = mess

(OUT / "ERGEBNIS_C_nachlauf.json").write_text(
    json.dumps(ergebnis, indent=2, ensure_ascii=False), encoding="utf-8")
(OUT / "ERGEBNIS_C_nachlauf.txt").write_text("\n".join(log_lines), encoding="utf-8")
print(f"\nGeschrieben: {OUT / 'ERGEBNIS_C_nachlauf.json'}")
