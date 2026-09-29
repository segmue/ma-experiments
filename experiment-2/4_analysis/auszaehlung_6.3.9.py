#!/usr/bin/env python3
"""
Auszählung für den Umbau von Abschnitt 6.3.9 (Recognizer und Gesamtsystem).

Zweck
    Rechnet aus vorhandenen Daten alle Zahlen, die für den Umbau von 6.3.9 und die
    abhängigen Stellen (5.2, 5.3, 7.1, 7.7, Kapitel 8) neu ins Zahlenregister A gehen:
      1. Nachzählung der Erkennung (tp, fp, fn, getroffene Gold-Toponyme, Recall .7270).
      2. Randeffekte der Spannenzählung in pipeline_metrics (03_full_pipeline.py):
         Gold-Toponyme mit mehr als einer überlappenden Vorhersage, Vorhersagen mit
         mehr als einem überlappten Gold-Toponym, und daraus die harten Schranken der
         Näherung correct_resolutions / 75'741 gegenüber einer goldseitigen Zählung.
      3. Mikro-Kette je System: correct_resolutions / 75'741 (Näherung).
      4. Differenzen zwischen Systemen (Mikro-Kette, Dokument-Recall, Acc@1 auf
         Gold-Spannen) und Rangfolgen auf den drei Ebenen.
      5. Zerlegung 72.7 % -> Mikro-Kette -> Dokument-Recall (Auflösungsverlust und
         Mittelungseffekt) sowie die Probe .7270 x .7787.
      6. Anteile für 7.7 / Kapitel 8: vom Recognizer verpasste und kandidatenlose
         Gold-Toponyme, deren Überschneidung, und was davon ein System auf
         Gold-Spannen richtig auflöst (ergänzende Einordnung).
      7. Nachtrag 25.09.2026 nach Entscheid des Nutzers: Spannweite der drei räumlichen
         Systeme (M4/E_c1, M5/E_c1, M5/E_d1) in der Kette gegen die Schranken, und die
         Zerlegung des Abstands des Dokument-Recalls zu eins (Erkennung, Auflösung,
         Mittelung).
    Es läuft kein Encoder und kein Resolver. Das Skript liest nur und schreibt nur
    seine Ausgabedatei.

Eingaben (Pfade aus 3_evaluation/config.py)
    ma-experiments/data/eval_dataset.json                            (= config.EVAL_JSON)
    3_evaluation/cache/recognizer_spans.pkl                          (03_full_pipeline.py)
    results/full_pipeline.json, results/full_pipeline_E_d1.json      (03 / run_fp_chunked.py)
    3_evaluation/cache/per_item/
        M3_default_finetuned_E_default.pkl.gz, M4_spatial_config1_E_c1.pkl.gz,
        M5_spatial_config2_E_c1.pkl.gz, M5_spatial_config2_E_c2.pkl.gz,
        M5_spatial_config2_E_d1.pkl.gz

Ausgabe
    results/auszaehlung_6.3.9_ausgabe.txt

Datum      25.09.2026
Bezug      04 Writing/revision/fixes/Auftrag_Umbau_6.3.9_Ende-zu-Ende_2026-09-25.md,
           Abschnitte 2.3, 2.4, 3.5, 3.8 und 4.
Aufruf     python3 -B auszaehlung_6.3.9.py
"""
import gzip
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

# Pfade des Repos: 3_evaluation/config.py (nur pathlib, keine Rechenabhaengigkeiten).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3_evaluation"))
import config as C  # noqa: E402

EVAL_JSON = C.EVAL_JSON
SPANS = C.CACHE_DIR / "recognizer_spans.pkl"
FP_MAIN = C.RESULTS_DIR / "full_pipeline.json"
FP_D1 = C.RESULTS_DIR / "full_pipeline_E_d1.json"
PER_ITEM = C.PER_ITEM_DIR
PER_ITEM_D1 = C.PER_ITEM_DIR / "M5_spatial_config2_E_d1.pkl.gz"
OUT = C.RESULTS_DIR / "auszaehlung_6.3.9_ausgabe.txt"

lines = []
def out(s=""):
    lines.append(s)
    print(s)

def ov(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

for p in (EVAL_JSON, SPANS, FP_MAIN, FP_D1, PER_ITEM_D1):
    if not p.exists():
        sys.exit(f"Eingabe fehlt: {p}")

out("Auszählung Umbau 6.3.9 — Ausgabe von auszaehlung_6.3.9.py (25.09.2026)")
out(f"Repo: {C.EXP2}")
out("")

docs = json.load(open(EVAL_JSON, encoding="utf-8"))["documents"]
pred_by_doc = pickle.load(open(SPANS, "rb"))["spans"]
N_DOCS = len(docs)
N_GOLD = sum(len(d["toponyms"]) for d in docs)

# ---------------------------------------------------------------- 1 Erkennung
tp = fp = 0
gold_hit = 0
gold_multi = 0          # Gold-Toponyme, die >1 Vorhersage überlappt
gold_multi_extra = 0    # Summe (k-1) über diese
pred_multi = 0          # Vorhersagen, die >1 Gold-Toponym überlappen
assigned = Counter()    # Gold-Toponym -> Zahl der Vorhersagen, die es als "bestes" wählen (Logik pipeline_metrics)
docs_no_gold = 0
undetected = set()      # (doc, start, end) nicht gefundener Gold-Toponyme
for d in docs:
    fn_ = d["filename"]
    gold = [(t["start"], t["end"]) for t in d["toponyms"]]
    if not gold:
        docs_no_gold += 1
    preds = pred_by_doc.get(fn_, [])
    hits_per_gold = Counter()
    for ps in preds:
        hit = [gi for gi, g in enumerate(gold) if ov(ps, g) > 0]
        if hit:
            tp += 1
            for gi in hit:
                hits_per_gold[gi] += 1
            if len(hit) > 1:
                pred_multi += 1
            # wie pipeline_metrics: max() liefert bei Gleichstand das erste Maximum
            best = max(range(len(gold)), key=lambda gi: ov(ps, gold[gi]))
            assigned[(fn_, best)] += 1
        else:
            fp += 1
    gold_hit += len(hits_per_gold)
    for gi, k in hits_per_gold.items():
        if k > 1:
            gold_multi += 1
            gold_multi_extra += k - 1
    for gi, g in enumerate(gold):
        if gi not in hits_per_gold:
            undetected.add((fn_, g[0], g[1]))

fn = N_GOLD - gold_hit
distinct_assigned = len(assigned)
over = tp - distinct_assigned          # Vorhersagen, die ein bereits zugewiesenes Gold-Toponym erneut treffen
under = gold_hit - distinct_assigned   # getroffene Gold-Toponyme, die keiner Vorhersage als "bestes" zugewiesen sind

out("1 Erkennung (Nachzählung aus recognizer_spans.pkl und eval_dataset.json)")
out(f"  Dokumente: {N_DOCS}; ohne Gold-Toponym: {docs_no_gold}")
out(f"  Gold-Toponyme: {N_GOLD}")
out(f"  Vorhersagen: {tp + fp}  (tp {tp}, fp {fp}; fp-Anteil {fp/(tp+fp):.4f})")
out(f"  Gold-Toponyme mit Überlappung: {gold_hit}; ohne: {fn}")
out(f"  Recall goldseitig: {gold_hit}/{N_GOLD} = {gold_hit/N_GOLD:.6f}")
out("")
out("2 Randeffekte der Spannenzählung in pipeline_metrics")
out(f"  Gold-Toponyme, die mehr als eine Vorhersage überlappt: {gold_multi} (überzählige Vorhersagen: {gold_multi_extra})")
out(f"  Vorhersagen, die mehr als ein Gold-Toponym überlappen: {pred_multi}")
out(f"  Verschiedene Gold-Toponyme, die als 'bestes' einer Vorhersage gewählt werden: {distinct_assigned}")
out(f"  Schranke Überzählung  (tp - zugewiesene Gold): {over}")
out(f"  Schranke Unterzählung (getroffene - zugewiesene Gold): {under}")
out("  Für jede goldseitige Zählung G gilt: correct - Überzählung <= G <= correct + Unterzählung.")
out(f"  Breite in Punkten der 75'741: -{100*over/N_GOLD:.3f} / +{100*under/N_GOLD:.3f}")
out("")

# ---------------------------------------------------------------- 3 Mikro-Kette
fpm = json.load(open(FP_MAIN, encoding="utf-8"))["systems"]
fpd = json.load(open(FP_D1, encoding="utf-8"))["systems"]
SYS = [
    ("M3/E_default", fpm["M3_default_finetuned+E_default"], PER_ITEM / "M3_default_finetuned_E_default.pkl.gz"),
    ("M4/E_c1",      fpm["M4_spatial_config1+E_c1"],        PER_ITEM / "M4_spatial_config1_E_c1.pkl.gz"),
    ("M5/E_c2",      fpm["M5_spatial_config2+E_c2"],        PER_ITEM / "M5_spatial_config2_E_c2.pkl.gz"),
    ("M5/E_c1",      fpm["M5_spatial_config2+E_c1"],        PER_ITEM / "M5_spatial_config2_E_c1.pkl.gz"),
    ("M5/E_d1",      fpd["M5_spatial_config2+E_d1"],        PER_ITEM_D1),
]
per_item = {}
acc1 = {}
for name, _, path in SYS:
    items = pickle.load(gzip.open(path, "rb"))
    assert len(items) == N_GOLD, (name, len(items))
    per_item[name] = {(it["doc_id"], it["start"], it["end"]): it for it in items}
    acc1[name] = sum(1 for it in items if it["pred_id"] is not None and it["pred_id"] == it["gold_id"])

out("3 Mikro-Kette: correct_resolutions / 75'741 (Näherung), mit Schranken")
out("  System         correct   Mikro   [untere, obere Schranke]   DocR     Acc@1 Gold (korrekt)")
rows = {}
for name, m, _ in SYS:
    c = m["correct_resolutions"]
    rows[name] = dict(c=c, micro=c / N_GOLD, docr=m["document_recall"], acc=acc1[name] / N_GOLD, acc_n=acc1[name])
    out(f"  {name:13s}  {c:6d}   {c/N_GOLD:.4f}  [{(c-over)/N_GOLD:.4f}, {(c+under)/N_GOLD:.4f}]"
        f"       {m['document_recall']:.4f}   {acc1[name]/N_GOLD:.4f} ({acc1[name]})")
out("")

# ---------------------------------------------------------------- 4 Differenzen und Rangfolgen
def diff(a, b):
    ra, rb = rows[a], rows[b]
    out(f"  {a} gegen {b}: Spannen {ra['c']-rb['c']:+d} ({100*(ra['c']-rb['c'])/N_GOLD:+.3f} Pkt der 75'741); "
        f"DocR {ra['docr']-rb['docr']:+.4f} ({100*(ra['docr']-rb['docr']):+.2f} Pkt); "
        f"Acc@1 Gold {ra['acc']-rb['acc']:+.4f} ({100*(ra['acc']-rb['acc']):+.2f} Pkt)")
    return ra, rb
out("4 Differenzen")
ra, rb = diff("M5/E_d1", "M3/E_default")
out(f"    Verhältnis Mikro-Kette / Acc@1: {(ra['micro']-rb['micro'])/(ra['acc']-rb['acc']):.3f}; "
    f"DocR / Acc@1: {(ra['docr']-rb['docr'])/(ra['acc']-rb['acc']):.3f}")
diff("M5/E_d1", "M5/E_c1")
diff("M5/E_c1", "M4/E_c1")
diff("M4/E_c1", "M3/E_default")
out(f"  Breite der Näherungsunsicherheit in Spannen: {over} (Überzählung) / {under} (Unterzählung)")
out("")
out("  Rangfolgen (absteigend)")
for key, lab in (("acc", "Acc@1 Gold"), ("c", "Mikro-Kette"), ("docr", "Dokument-Recall")):
    order = sorted(rows, key=lambda n: -rows[n][key])
    out(f"    {lab:16s}: " + " > ".join(order))
out("")

# ---------------------------------------------------------------- 5 Zerlegung
best = rows["M5/E_d1"]
rec = gold_hit / N_GOLD
res_acc_tp = best["c"] / tp
out("5 Zerlegung Recall -> Mikro-Kette -> Dokument-Recall (M5/E_d1)")
out(f"  Recall {rec:.4f} -> Mikro {best['micro']:.4f} -> DocR {best['docr']:.4f}")
out(f"  Auflösungsverlust (Recall - Mikro): {100*(rec-best['micro']):.2f} Pkt")
out(f"  Mittelungseffekt  (Mikro - DocR):   {100*(best['micro']-best['docr']):.2f} Pkt")
out(f"  Gesamt (Recall - DocR):             {100*(rec-best['docr']):.2f} Pkt")
out(f"  Auflösungsgenauigkeit auf tp-Spannen: {best['c']}/{tp} = {res_acc_tp:.4f}")
out(f"  Probe Recall x Genauigkeit auf tp: {rec:.4f} x {res_acc_tp:.4f} = {rec*res_acc_tp:.4f} (gegen Mikro {best['micro']:.4f})")
for n in rows:
    out(f"  {n:13s}: Mittelungseffekt Mikro - DocR = {100*(rows[n]['micro']-rows[n]['docr']):.2f} Pkt")
out("")

# ---------------------------------------------------------------- 6 Anteile für 7.7 / Kapitel 8
ref = per_item["M3/E_default"]
nocand = {k for k, it in ref.items() if it["n_candidates"] == 0}
both = undetected & nocand
out("6 Anteile für 7.7 und Kapitel 8")
out(f"  Vom Recognizer nicht gefunden: {len(undetected)} = {100*len(undetected)/N_GOLD:.2f} % der {N_GOLD}")
out(f"  Ohne Gazetteer-Kandidat (Gold-Spannen): {len(nocand)} = {100*len(nocand)/N_GOLD:.2f} %")
out(f"  Überschneidung (nicht gefunden UND ohne Kandidat): {len(both)} = {100*len(both)/N_GOLD:.2f} %")
out(f"  Nicht gefunden, aber mit Kandidat: {len(undetected-nocand)} = {100*len(undetected-nocand)/N_GOLD:.2f} %")
d1 = per_item["M5/E_d1"]
und_ok = sum(1 for k in undetected if d1[k]["pred_id"] is not None and d1[k]["pred_id"] == d1[k]["gold_id"])
out(f"  Davon löst M5/E_d1 auf Gold-Spannen richtig: {und_ok} = {100*und_ok/N_GOLD:.2f} % der {N_GOLD}")
out("")
# ---------------------------------------------------------------- 7 Nachtrag
spatial = ["M4/E_c1", "M5/E_c1", "M5/E_d1"]
cs = [rows[n]["c"] for n in spatial]
ds = [rows[n]["docr"] for n in spatial]
out("7 Nachtrag 25.09.2026")
out(f"  Räumliche Systeme {', '.join(spatial)}: correct {min(cs)} bis {max(cs)}, Spannweite {max(cs)-min(cs)} Spannen "
    f"({100*(max(cs)-min(cs))/N_GOLD:.3f} Pkt); Unterzählungsschranke je System {under}, Überzählung {over}")
out(f"    Spannweite < Schranke Unterzählung: {max(cs)-min(cs) < under}")
out(f"  Dokument-Recall der räumlichen Systeme: {min(ds):.4f} bis {max(ds):.4f}, Spannweite {max(ds)-min(ds):.4f}")
gap = 1 - best["docr"]
out(f"  Abstand des Dokument-Recalls zu eins (M5/E_d1): {100*gap:.2f} Pkt = Erkennung {100*(1-rec):.2f} "
    f"+ Auflösung {100*(rec-best['micro']):.2f} + Mittelung {100*(best['micro']-best['docr']):.2f}")
out("")
out("Ende.")

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
