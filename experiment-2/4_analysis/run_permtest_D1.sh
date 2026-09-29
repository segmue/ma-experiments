#!/usr/bin/env bash
# =============================================================================
# Permutationstest fuer die Adjazenzmatrix D1
#
# Prueft jede Zelle der gerichteten nPMI-Matrix gegen ein Nullmodell, das die
# Kategorieetiketten der 442'001 Gazetteerobjekte bei unveraenderter
# Nachbarschafts- und Gewichtsstruktur vertauscht - das Random Labelling, mit
# dem Leslie und Kronenfeld (2011) den Colocation Quotient testen.
#
# Warum das noetig ist: Fuer die aeltere objektweise Variante gibt es einen
# solchen Test, fuer die tatsaechlich verwendete zellweise Fassung nicht. Wenn
# das Adjazenzmass in der Arbeit gleichberechtigt neben dem Ueberlagerungsmass
# steht, wird nach Signifikanz gefragt.
#
# WICHTIG: Der Test braucht die Paarliste (Spalten sf, tf, wgt) und den
# Objektpool, aus denen die D1-Matrix gerechnet wurde. 'spatial-h3-assoc
# --measure d1' legt diese Zwischenprodukte nicht ab; sie muessen per PAIRS=
# und POOL= (und optional W= fuer die Gegenprobe) angegeben werden.
# Matrix und Paarliste muessen aus demselben Lauf stammen.
#
# Zwei Laeufe, beide noetig:
#   1. Der Test selbst, B = 1'000
#   2. Die Eichprobe: die Etiketten werden vorab einmal permutiert, die
#      Nullhypothese gilt also. Sie belegt, dass der Test sein Niveau haelt,
#      und gehoert in den Anhang. Im Probelauf lag sie bei 2.16 statt nominal
#      5 Prozent, der Test ist also konservativ.
#
# Es ueberschreibt nichts. Die Matrix wird weder gefiltert noch veraendert.
#
# Aufruf:
#     PAIRS=/pfad/pairs_cellwise.parquet POOL=/pfad/pool.parquet bash run_permtest_D1.sh
#
# Andere Umgebung:  PY=/pfad/zu/python3 bash run_permtest_D1.sh
#
# Dauer: rund fuenfzehn bis dreissig Minuten fuer beide Laeufe zusammen, plus
# einmalig ein paar Minuten fuer den Zwischenspeicher (rund 520 MB, liegt
# ausserhalb des Projektordners). Der Lauf ist unterbrechbar und fortsetzbar:
# Derselbe Aufruf nimmt einen Pruefpunkt auf und rechnet weiter, bitgleich.
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"          # experiment-2/4_analysis
EXP2="$(dirname "$HERE")"                       # experiment-2
REPO="$(dirname "$EXP2")"                       # ma-experiments
PY="${PY:-python3}"
FIX="$HERE"                                     # Ort von permtest_cellwise.py
OUTDIR="${OUTDIR:-$EXP2/results}"               # Ergebnisordner permtest_D1/ und permtest_D1_eichprobe/
CACHE="${CACHE:-$HOME/permtest_cache_D1}"
B="${B:-1000}"

PAIRS="${PAIRS:?PAIRS= Pfad zur Paarliste (parquet, Spalten sf, tf, wgt) setzen}"
POOL="${POOL:?POOL= Pfad zum Objektpool (parquet) setzen}"
W="${W:-}"                                      # optional: Gewichtsmatrix (.npy) fuer --check-w
MATRIX="${MATRIX:-$REPO/matrices/config1/d1_matrix.csv}"

mkdir -p "$EXP2/3_evaluation/logs"
LOG="$EXP2/3_evaluation/logs/permtest_D1_$(date +%Y-%m-%d_%H%M).log"
sagen() { printf '\n\033[1m%s\033[0m\n' "$*" | tee -a "$LOG"; }
zeile() { printf '%s\n' "$*" | tee -a "$LOG"; }
abbruch() { zeile ""; zeile "ABBRUCH: $*"; zeile "Protokoll: $LOG"; exit 1; }

sagen "Permutationstest fuer die Adjazenzmatrix D1"
zeile "Paarliste:      $PAIRS"
zeile "Matrix:         $MATRIX"
zeile "Permutationen:  $B"
zeile "Zwischenspeicher: $CACHE"
zeile "Protokoll:      $LOG"
zeile "Beginn:         $(date '+%Y-%m-%d %H:%M:%S')"

# --------------------------------------------------------------------------
sagen "Schritt 1 von 4 — Voraussetzungen"
# --------------------------------------------------------------------------
[ -x "$PY" ] || abbruch "Python nicht gefunden: $PY"
for f in "$PAIRS" "$POOL" ${W:+"$W"} "$MATRIX" "$FIX/permtest_cellwise.py"; do
  [ -e "$f" ] || abbruch "fehlt: $f"
done
"$PY" -c "import numpy, duckdb, pyarrow; print(f'numpy {numpy.__version__} · duckdb {duckdb.__version__} · pyarrow {pyarrow.__version__}')" 2>&1 | tee -a "$LOG" \
  || abbruch "Bibliotheken fehlen. PY= auf die richtige Umgebung zeigen lassen."
zeile "Paarliste: $(du -h "$PAIRS" | cut -f1)"
zeile "Eingaben vollstaendig."

# --------------------------------------------------------------------------
sagen "Schritt 2 von 4 — Der Test (B = $B)"
# --------------------------------------------------------------------------
zeile "Das Skript prueft dabei selbst, ob die aus der Paarliste rekonstruierte"
zeile "Matrix mit der abgelegten uebereinstimmt (--check-w, --check-matrix)."
zeile "Weicht sie ab, gehoeren Paarliste und Matrix nicht zusammen, und der"
zeile "Befund ist wichtiger als der Test."
zeile ""
T0=$(date +%s)
"$PY" "$FIX/permtest_cellwise.py" \
  --pairs "$PAIRS" --pool "$POOL" \
  --out "$OUTDIR/permtest_D1" --cache-dir "$CACHE" \
  --B "$B" --seed 7 --null frei \
  --checkpoint-every 50 \
  ${W:+--check-w "$W"} --check-matrix "$MATRIX" \
  --tag D1 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
T1=$(date +%s)
zeile ""
zeile "Laufzeit: $(( (T1-T0)/60 )) min $(( (T1-T0)%60 )) s"
[ "$RC" -eq 0 ] || abbruch "Der Test ist fehlgeschlagen (Exit $RC).
  Derselbe Aufruf setzt am Pruefpunkt fort; bei einer echten Fehlermeldung
  das Protokoll schicken."

# --------------------------------------------------------------------------
sagen "Schritt 3 von 4 — Die Eichprobe"
# --------------------------------------------------------------------------
zeile "Die Etiketten werden vorab einmal permutiert, die Nullhypothese gilt also."
zeile "Erwartet sind hoechstens rund fuenf Prozent signifikante Zellen; im"
zeile "Probelauf waren es 2.16 Prozent. Deutlich mehr hiesse, der Test haelt sein"
zeile "Niveau nicht, und dann ist auch der Hauptlauf nicht zu gebrauchen."
zeile ""
T2=$(date +%s)
"$PY" "$FIX/permtest_cellwise.py" \
  --pairs "$PAIRS" --pool "$POOL" \
  --out "$OUTDIR/permtest_D1_eichprobe" --cache-dir "$CACHE" \
  --B "$B" --seed 7 --placebo 0 --tag P 2>&1 | tee -a "$LOG"
RC2=${PIPESTATUS[0]}
T3=$(date +%s)
zeile ""
zeile "Laufzeit: $(( (T3-T2)/60 )) min $(( (T3-T2)%60 )) s"
[ "$RC2" -eq 0 ] || zeile "WARNUNG: Die Eichprobe ist fehlgeschlagen (Exit $RC2). Der Hauptlauf steht."

# --------------------------------------------------------------------------
sagen "Schritt 4 von 4 — Zusammenzug"
# --------------------------------------------------------------------------
"$PY" - "$OUTDIR/permtest_D1" "$OUTDIR/permtest_D1_eichprobe" <<'PYSUM' 2>&1 | tee -a "$LOG"
import json, sys
from pathlib import Path
import numpy as np

def lies(ordner, was):
    p = Path(ordner)
    if not p.exists():
        print(f"  {was}: Ordner fehlt"); return None
    metas = sorted(p.glob("permtest_meta*.json"))
    if not metas:
        print(f"  {was}: keine Metadatei"); return None
    m = json.loads(metas[0].read_text())
    print(f"  {was}: {metas[0].name}")
    for k in ("n_permutations", "seed", "null", "n_signifikant_q005",
              "n_signifikant_positiv", "n_signifikant_negativ",
              "anteil_signifikant", "seconds"):
        if k in m: print(f"    {k:26s} {m[k]}")
    return m

print("Hauptlauf:")
h = lies(sys.argv[1], "Ergebnis")
print("\nEichprobe:")
e = lies(sys.argv[2], "Ergebnis")
print("""
Zu berichten ist die Zahl signifikanter Zellen als UNTERGRENZE — die Eichprobe
zeigt, dass der Test konservativ ist. Und falls eine Betragsstatistik ausgegeben
wird: Sie setzt Zellen mit starkem positivem nPMI faelschlich auf p = 1.0, weil
der Betrag von -1 der groesstmoegliche ist. Berichtet wird die randbasierte
zweiseitige Fassung.""")
PYSUM

sagen "Fertig"
zeile "Ende: $(date '+%Y-%m-%d %H:%M:%S')"
zeile ""
zeile "Ergebnisse:"
zeile "  $OUTDIR/permtest_D1"
zeile "  $OUTDIR/permtest_D1_eichprobe"
zeile "Protokoll: $LOG"
