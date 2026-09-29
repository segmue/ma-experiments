# -*- coding: utf-8 -*-
"""
stil_kap6 — gemeinsame Bildsprache der Abbildungen von Kapitel 6.

Erzeugt keine Abbildung. Wird von den Skripten `abb_*.py` im selben Ordner
eingebunden und haelt die Typografie, die Farbfolge und die Speicherroutine
an einer Stelle zusammen.

Herkunft: Hausstil der Druckabbildungen der Evaluation (Funktionen `style`,
`save`, `dump`, rcParams-Block und die Farbkonstanten).

Haelt ausserdem die Pfade des Repos (ma-experiments/experiment-2/5_figures),
damit die Abbildungsskripte keine festen Pfade tragen.

Farbfolge: helligkeitsgestaffelt, damit die Reihenfolge in Graustufen lesbar
bleibt. Wo Farbe allein die Unterscheidung traegt, kommt zusaetzlich eine
Strich- oder Markerform dazu.

Datum: 2026-09-17
"""
from __future__ import annotations

import csv
import datetime as _dt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Pfade des Repos ──────────────────────────────────────────────────────────
HIER = Path(__file__).resolve().parent            # experiment-2/5_figures
EXP2 = HIER.parent                                # experiment-2
REPO = EXP2.parent                                # ma-experiments
RESULTS = EXP2 / "results"                        # veroeffentlichte Kennzahlen
MATRICES = REPO / "matrices"                      # b1_matrix.csv / d1_matrix.csv je Config
DATA = REPO / "data"                              # Laufzeitdaten (gitignoriert)
SWISSNAMES3D = DATA / "swissnames3d"              # swissNAMES3D_{PKT,LIN,PLY}.dbf
AUSGABE = HIER / "output"                         # Abbildungen und __daten.csv

# ── Farben, helligkeitsgestaffelt ─────────────────────────────────────────────
C_BASE = "#c8ccd4"   # hellste Stufe   — faire Baseline / E_default
C_MITT = "#8fa8bd"   # mittlere Stufe  — E_c2
C_DIAG = "#6a8caf"   # dunklere Stufe  — E_c1 / Diagonale config1
C_BEST = "#1f3b57"   # dunkelste Stufe — bestes System / E_d1
C_WARN = "#a6432f"   # Hervorhebung, Gegenbefund
C_GRID = "#dcdee3"
C_LINE = "#333333"

# Die beiden Masse. In Graustufen traegt zusaetzlich die Strichform.
FARBE_MASS = {"B1": C_WARN, "D1": C_BEST}
STRICH_MASS = {"B1": (0, (4, 2)), "D1": "solid"}
MARKER_MASS = {"B1": "o", "D1": "s"}
FUELL_MASS = {"B1": "none", "D1": "full"}

RC = {
    "font.family": "sans-serif",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#4a4a4a",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
}


def setze_stil() -> None:
    plt.rcParams.update(RC)


def style(ax, grid_axis: str = "x") -> None:
    """Rahmen weg, feines Gitter hinter die Daten."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=C_GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def tausender(n) -> str:
    """Schweizer Tausendertrennzeichen: 75741 -> 75'741."""
    return f"{int(n):,}".replace(",", "’")


def zahl(x, stellen: int = 4) -> str:
    """.8403 statt 0.8403; negative Werte mit Minuszeichen."""
    s = f"{x:.{stellen}f}"
    return s.replace("0.", ".", 1) if s.startswith("0.") else s.replace("-0.", "−.", 1)


def speichern(fig, ziel_ordner: Path, name: str, datum: str | None = None) -> Path:
    """Legt <name>_<datum>.png und .pdf ab. Ueberschreibt nie eine Datei."""
    datum = datum or _dt.date.today().isoformat()
    ziel_ordner = Path(ziel_ordner)
    ziel_ordner.mkdir(parents=True, exist_ok=True)
    stamm = f"{name}_{datum}"
    png = ziel_ordner / f"{stamm}.png"
    n = 1
    while png.exists():
        n += 1
        stamm = f"{name}_{datum}_v{n}"
        png = ziel_ordner / f"{stamm}.png"
    fig.savefig(png)
    fig.savefig(ziel_ordner / f"{stamm}.pdf")
    plt.close(fig)
    print("  ->", png)
    print("  ->", ziel_ordner / f"{stamm}.pdf")
    return png


def daten_ablegen(ziel_ordner: Path, name: str, kopf, zeilen,
                  datum: str | None = None) -> Path:
    """Schreibt exakt die gezeichneten Werte als CSV neben die Abbildung."""
    datum = datum or _dt.date.today().isoformat()
    ziel_ordner = Path(ziel_ordner)
    ziel_ordner.mkdir(parents=True, exist_ok=True)
    q = ziel_ordner / f"{name}_{datum}__daten.csv"
    with open(q, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(kopf)
        w.writerows(zeilen)
    print("  ->", q)
    return q


# ── Sequentielle Farbkarte fuer Felder mit Farbskala (18.09.2026) ────────────
# Neu hinzugekommen. Kein bestehender Export ist dabei angeruehrt worden —
# weder die Palette noch die Typografie noch die Gitter- und Achseneinstellungen;
# die fuenf Abbildungen von Kapitel 6 zeichnen unveraendert weiter.
#
# Anlass sind zwei Befunde an der Abbildung mit Farbskala:
#   1. Ein Verlauf, der bei C_BEST endet, verliert am oberen Ende Kontrast:
#      C_BEST liegt bei L* 24 und damit weit von Schwarz entfernt. Der Verlauf
#      bekommt deshalb mit C_ENDE einen nahezu schwarzen Endton (L* 6.4).
#   2. C_GRID als Grundton der Zellen unter der Schwelle liegt nur 6.5 L* ueber
#      dem hellsten Skalenton C_BASE; schwache Zellen versinken darin. Der
#      Grundton ist deshalb C_UNTERSCHWELLE, ein heller, leicht warmer
#      Neutralton bei L* 94.5 — 12.6 L* ueber C_BASE und dadurch auch im
#      Graustufendruck vom Skalenanfang zu trennen. Er ist zugleich anders
#      gesaettigt als die durchweg kuehle Skala.
#
# Die Stuetzstellen sitzen nicht in gleichen Abstaenden, sondern dort, wo ihre
# Helligkeit es verlangt. Ueber t = 0 bis 1 faellt L* dadurch in gleichmaessigen
# Schritten von 3.5 bis 4.1 von 81.96 auf 6.42; die Ordnung der Werte bleibt
# damit auch ohne Farbe lesbar (Graustufe 204 bis 20 von 255).
from matplotlib.colors import LinearSegmentedColormap as _LinearSegmentedColormap

C_ENDE = "#0a1521"            # nahezu schwarzer Endton der Farbskala
C_UNTERSCHWELLE = "#f2efe9"   # Grundton der Zellen unter der Schwelle

# (Position im Verlauf, Ton) — die Position folgt der Helligkeit des Tons.
STUETZEN_SEQ = ((0.000, C_BASE), (0.190, C_MITT), (0.331, C_DIAG),
                (0.767, C_BEST), (1.000, C_ENDE))

KARTE_SEQ = _LinearSegmentedColormap.from_list("stil_kap6_seq", STUETZEN_SEQ)
