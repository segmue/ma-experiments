"""Stage 10 — gepaarter McNemar fuer die H3-Config-Ablation (E_c1..E_c5), read-only
aus den Per-Item-Dumps von Stage 02 (cache/per_item/*.pkl.gz).

Kontraste:
  A) Aufloesungsreihe bei overlap:  E_c3 (11) -> E_c4 (12) -> E_c1 (13)
  B) Modus-Kontrast bei res 13:     E_c1 (overlap) vs E_c5 (center)
  C) Kontrolle: Aufloesung innerhalb center: E_c2 (10) vs E_c5 (13)

Aufruf: python 10_mcnemar_config_ablation.py [ziel.json]
        (Default: results/config_ablation/ZUSATZ_mcnemar_e8.json)
"""
import gzip, pickle, json, pathlib, sys
from scipy.stats import binomtest

import config as C

PI = C.PER_ITEM_DIR
MODELS = ["M3_default_finetuned", "M4_spatial_config1", "M5_spatial_config2"]

# (A-Arm, B-Arm, Familie) — Delta wird immer als B minus A gerechnet.
VERGLEICHE = [
    ("E_c3", "E_c4", "A_aufloesung"),   # 11 -> 12
    ("E_c4", "E_c1", "A_aufloesung"),   # 12 -> 13
    ("E_c3", "E_c1", "A_aufloesung"),   # 11 -> 13 (Gesamtspanne)
    ("E_c1", "E_c5", "B_modus"),        # overlap -> center, beide res 13
    ("E_c2", "E_c5", "C_kontrolle"),    # center 10 -> center 13
]


def load(model, arm):
    with gzip.open(PI / f"{model}_{arm}.pkl.gz", "rb") as f:
        return pickle.load(f)


def cmp(model, a_name, b_name, ambig_only):
    A, B = load(model, a_name), load(model, b_name)
    assert len(A) == len(B), f"{model}: Laengen {len(A)} != {len(B)}"
    for x, y in zip(A, B):   # Paarung ueber (doc_id, start, end) verifizieren
        assert (x["doc_id"], x["start"], x["end"]) == (y["doc_id"], y["start"], y["end"])
    idx = [i for i in range(len(A)) if (not ambig_only) or A[i]["n_candidates"] > 1]
    ca = [A[i]["rank"] == 1 for i in idx]
    cb = [B[i]["rank"] == 1 for i in idx]
    oa = sum(1 for x, y in zip(ca, cb) if x and not y)
    ob = sum(1 for x, y in zip(ca, cb) if y and not x)
    n = oa + ob
    p = binomtest(oa, n, 0.5).pvalue if n else 1.0
    return dict(model=model, a=a_name, b=b_name,
                subset="ambig" if ambig_only else "all", n=len(idx),
                acc1_a=sum(ca) / len(idx), acc1_b=sum(cb) / len(idx),
                only_a=oa, only_b=ob, discordant=n,
                delta=sum(cb) / len(idx) - sum(ca) / len(idx), p=p)


def holm(werte):
    """Holm-Bonferroni ueber eine Liste von p-Werten; gibt korrigierte p zurueck."""
    m = len(werte)
    ordnung = sorted(range(m), key=lambda i: werte[i])
    korr = [0.0] * m
    lauf = 0.0
    for rang, i in enumerate(ordnung):
        lauf = max(lauf, (m - rang) * werte[i])
        korr[i] = min(1.0, lauf)
    return korr


out = []
hdr = (f"{'Modell':<22}{'Vergleich':<18}{'Familie':<14}{'Subset':<8}"
       f"{'Acc@1 A':>9}{'Acc@1 B':>9}{'Delta':>9}{'nur A':>7}{'nur B':>7}"
       f"{'diskord':>9}{'McNemar p':>13}")
print(hdr)
print("-" * len(hdr))

for model in MODELS:
    for a, b, familie in VERGLEICHE:
        if not (PI / f"{model}_{a}.pkl.gz").exists() or not (PI / f"{model}_{b}.pkl.gz").exists():
            print(f"{model:<22}{a+' vs '+b:<18}{familie:<14}FEHLT — uebersprungen")
            continue
        for amb in (False, True):
            r = cmp(model, a, b, amb)
            r["familie"] = familie
            out.append(r)
            print(f"{r['model']:<22}{a+' vs '+b:<18}{familie:<14}{r['subset']:<8}"
                  f"{r['acc1_a']:>9.4f}{r['acc1_b']:>9.4f}{r['delta']:>+9.4f}"
                  f"{r['only_a']:>7}{r['only_b']:>7}{r['discordant']:>9}{r['p']:>13.3e}")

# Holm-Korrektur ueber die ambigen Tests (das ist die berichtete Familie).
ambig = [r for r in out if r["subset"] == "ambig"]
for r, pk in zip(ambig, holm([r["p"] for r in ambig])):
    r["p_holm"] = pk

print(f"\nHolm-korrigiert ueber die {len(ambig)} ambigen Tests:")
print(f"{'Modell':<22}{'Vergleich':<18}{'Delta':>9}{'p roh':>13}{'p Holm':>13}  Signifikanz")
for r in ambig:
    stern = "***" if r["p_holm"] < 0.001 else "**" if r["p_holm"] < 0.01 else "*" if r["p_holm"] < 0.05 else "n.s."
    print(f"{r['model']:<22}{r['a']+' vs '+r['b']:<18}{r['delta']:>+9.4f}"
          f"{r['p']:>13.3e}{r['p_holm']:>13.3e}  {stern}")

ziel = sys.argv[1] if len(sys.argv) > 1 else str(C.RESULTS_DIR / "config_ablation" / "ZUSATZ_mcnemar_e8.json")
pathlib.Path(ziel).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(ziel).write_text(json.dumps(out, indent=2), encoding="utf-8")
print("\nGespeichert:", ziel)
