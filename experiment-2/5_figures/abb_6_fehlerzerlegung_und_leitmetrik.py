#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERZEUGNIS  Abbildung 6, Fassung vom 26.09.2026: ohne die Fusszeile (ihr Inhalt
           steht in Text und Unterschrift), Feld (b) breiter, sodass die
           Zellen annaehernd quadratisch sind. Zahlen unveraendert.
           Vorlage: abb_6-2_fehlerzerlegung_und_leitmetrik_mit_E_d1.py.
           Links die Fehlerzerlegung von jetzt SECHS Systemen, rechts die
           Leitmetrik `acc1_ambiguous_only` ueber alle zwanzig Zellen.
           Geschrieben werden in den Zielordner (Vorgabe `5_figures/output/`):
             abb_6-2_fehlerzerlegung_und_leitmetrik_en_mit_E-d1_<Datum>.png
             abb_6-2_fehlerzerlegung_und_leitmetrik_en_mit_E-d1_<Datum>.pdf
             abb_6-2_fehlerzerlegung_und_leitmetrik_en_mit_E-d1_<Datum>__daten.csv
           Die bestehende Fassung `..._en_2026-09-17.{png,pdf,csv}` und das
           bestehende Skript `abb_6-2_fehlerzerlegung_und_leitmetrik.py`
           bleiben unberuehrt; diese Fassung tritt daneben, nicht an ihre
           Stelle.

WAS NEU IST
           1. Feld (a) fuehrt als sechste Zeile `M5/E_d1` — das beste je
              gemessene System, mit 3'113 falsch gerankten Toponymen dem
              kleinsten Wert aller zwanzig Zellen.
           2. Die Anmerkung «No error decomposition is available for the four
              E_d1 cells …» ist gestrichen. Sie war doppelt ueberholt: es sind
              fuenf E_d1-Zellen, nicht vier, und die Fehlerzerlegung aller
              fuenf liegt seit dem Nachlauf vom 17.09.2026 im Register.
              An ihrer Stelle steht, was jetzt noch zutrifft: welches System
              welches Assoziationsmass benutzt.
           3. Die Obergrenze ist vierstellig beschriftet (.8850 statt .885),
              wie es REGELN Abbildungen und Agentenarbeit.md, Teil 1, fuer
              Genauigkeitswerte verlangt.
           4. Die Farbleiste von Feld (b) heisst «Acc@1 (ambiguous toponyms)»
              und benennt damit ausdruecklich die Teilmenge; `acc1` und
              `acc1_ambiguous_only` sind verschiedene Groessen.

EINGABE    Keine Datei. Alle Zahlen stehen als Literal im Abschnitt ZAHLEN und
           stammen ausschliesslich aus dem Zahlenregister unter
           `04 Writing/revision/fixes/Register/`:
             * `Zahlenregister_A_Aufloesungsergebnisse.md`, Abschnitt 3.1
               — Fehlerzerlegung der fuenf B1-Systeme, Sockel 43'544,
                 555 Gold ausserhalb der Kandidatenliste, 8'159 ohne
                 Kandidat, harte Obergrenze .8850;
             * `Zahlenregister_A_Aufloesungsergebnisse.md`, Abschnitt 2
               — `acc1_ambiguous_only` der fuenfzehn B1-Zellen;
             * `Zahlenregister_A_Aufloesungsergebnisse.md`, Abschnitt 7.1
               — `acc1_ambiguous_only` der fuenf E_d1-Zellen;
             * `Zahlenregister_C_Nachlauf_E-d1.md`, Abschnitt 1, Zeilen 81
               bis 98 — Fehlerzerlegung der fuenf E_d1-Zellen; gezeichnet
               ist davon M5/E_d1 (43'544 / 20'370 / 3'113 / 555 / 8'159).
           Kein Wert kommt aus einer Matrixdatei, einer Auswertungsnotiz oder
           einem Ergebnisabzug.

BILDSPRACHE
           Wie die bestehende Fassung ueber `stil_kap6.py`, damit die neue
           Abbildung Zug um Zug mit der alten verglichen werden kann. Die
           offene Entscheidung aus REGELN, Teil 1 (Bildsprache des Bestands
           oder `stil_kap6.py`), wird hier NICHT getroffen und nicht
           vorweggenommen; umgestellt ist nichts.

AUFRUF     python3 abb_6_fehlerzerlegung_und_leitmetrik.py
                   [--out ZIELORDNER] [--datum JJJJ-MM-TT] [--pruefen]
           Ohne --out schreibt das Skript nach 5_figures/output/. Feste
           Pfade stehen keine im Skript. `--pruefen` zeichnet zusaetzlich und
           misst Zeilenabstand und Ueberschneidungen der Beschriftungen.

VERAENDERT keine bestehende Datei. `stil_kap6.speichern` ueberschreibt nie;
           laeuft das Skript zweimal am selben Tag, entsteht `_v2`.

Datum      2026-09-18
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True          # kein neues __pycache__ im Projektordner

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stil_kap6 as S                                   # noqa: E402

DATUM = "2026-09-26"
NAME = "abb_6_fehlerzerlegung_und_leitmetrik_en"

# ── ZAHLEN ───────────────────────────────────────────────────────────────────
# Durchgehende Bezugsgroessen, in allen zwanzig Zellen gleich.
# Register A 3.1 und Register C 1 (Zeilen 93 bis 95) stimmen darin ueberein.
N_ITEMS, N_AMBIG = 75741, 23848
SOCKEL = 43544          # correct_unambiguous
GOLD_FEHLT = 555        # gold_not_in_candidates
OHNE_KANDIDAT = 8159    # no_candidate
OBERGRENZE = .8850      # harte Obergrenze acc1 (67'027 von 75'741);
                        # gilt nach Register C 1, Z. 98 auch fuer die E_d1-Zellen

# Feld (a): korrekt ambig (`correct` minus Sockel) und falsch gerankt
# (`wrong_rank`) je System.
#   Zeilen 1 bis 5 — Register A, Abschnitt 3.1.
#   Zeile 6        — Register C, Abschnitt 1 (43'544 / 20'370, wrong_rank 3'113).
# Je Eintrag: Maschinenname (fuer die CSV), Bildbeschriftung, korrekt ambig,
# falsch gerankt.
ZERLEGUNG = [
    ("M1/E_default", "M1/E$_\\mathrm{default}$", 3165, 20318),
    ("M2/E_default", "M2/E$_\\mathrm{default}$", 10351, 13132),   # ambiguity_breakdown.csv, Z. 5 (26.09.2026)
    ("M3/E_default", "M3/E$_\\mathrm{default}$", 18640, 4843),
    ("M4/E_c1", "M4/E$_\\mathrm{c1}$", 19842, 3641),
    ("M5/E_c1", "M5/E$_\\mathrm{c1}$", 20101, 3382),
    ("M5/E_d1", "M5/E$_\\mathrm{d1}$", 20370, 3113),
]
TEILE = [("correct, single candidate", "#8fa8bd"),
         ("correct, ambiguous", S.C_BEST),
         ("wrong rank", S.C_WARN),
         ("gold not among candidates", "#d8a48f"),
         ("no candidate", "#e8e8e8")]

# Feld (b): Accuracy@1 auf den 23'848 ambigen Toponymen — `acc1_ambiguous_only`,
# NICHT die Gesamtmetrik `acc1`.
#   Spalten E_default, E_c1, E_c2 — Register A, Abschnitt 2.
#   Spalte  E_d1                  — Register A, Abschnitt 7.1.
MODELLE = ["M1", "M2", "M3", "M4", "M5"]
VARIANTEN = ["E_default", "E_c1", "E_c2", "E_d1"]
AMBIG = {
    "M1": {"E_default": .1327, "E_c1": .5863, "E_c2": .3374, "E_d1": .6620},
    "M2": {"E_default": .4340, "E_c1": .7196, "E_c2": .5340, "E_d1": .7470},
    "M3": {"E_default": .7816, "E_c1": .8315, "E_c2": .7268, "E_d1": .8100},
    "M4": {"E_default": .5165, "E_c1": .8320, "E_c2": .6268, "E_d1": .8287},
    "M5": {"E_default": .6606, "E_c1": .8429, "E_c2": .7556, "E_d1": .8542},
}
SPALTE = {"E_default": "E$_\\mathrm{default}$", "E_c1": "E$_\\mathrm{c1}$",
          "E_c2": "E$_\\mathrm{c2}$", "E_d1": "E$_\\mathrm{d1}$"}

CMAP = LinearSegmentedColormap.from_list(
    "kap6", ["#f4f5f7", S.C_BASE, S.C_DIAG, S.C_BEST, "#0d1c2b"])

# Die Anmerkung unter dem Bild. An die Stelle der gestrichenen Luecken-Anmerkung
# tritt der Satz, welches der drei in Feld (a) vorkommenden Systeme welches
# Assoziationsmass benutzt — Feld (a) mischt seit der sechsten Zeile beide
# Masse. Belegt in Register A, Abschnitt 7 (Z. 523 f.: «b1_matrix.csv (Arm
# E_c1, Mass B₁) gegen npmi_dist_matrix_D1.csv (Arm E_d1, Mass D₁)») und
# Abschnitt 1 (Z. 99: E_default ist der Arm ohne raeumliche Beschreibung).
# Ueber E_c2 sagt der Satz nichts: welches Mass dahintersteht, fuehrt das
# Register nicht, und E_c2 kommt in Feld (a) nicht vor.
ANMERKUNG = (
    f"Panel (a): the base of {S.tausender(SOCKEL)} unambiguous toponyms, "
    f"{S.tausender(GOLD_FEHLT)} gold outside the candidate set\n"
    f"and {S.tausender(OHNE_KANDIDAT)} without any candidate is identical in "
    "every cell.\n"
    "E$_\\mathrm{c1}$ uses B$_1$ — footprint overlap, E$_\\mathrm{d1}$ uses "
    "D$_1$ — ring adjacency;\n"
    "E$_\\mathrm{default}$ has no spatial description. "
    "Red outline in (b): best cell."
)


def zeichne(out: Path, datum: str, pruefen: bool = False) -> None:
    S.setze_stil()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 3.5),
                                   gridspec_kw={"width_ratios": [1.0, 1.22]})

    # ── links: Fehlerzerlegung, sechs Systeme ────────────────────────────────
    y = list(range(len(ZERLEGUNG)))
    werte = [[SOCKEL, ka, fr, GOLD_FEHLT, OHNE_KANDIDAT]
             for _, _, ka, fr in ZERLEGUNG]
    for w, (n, _, _, _) in zip(werte, ZERLEGUNG):
        if sum(w) != N_ITEMS:                       # Torwaechter gegen Tippfehler
            raise SystemExit(f"Zerlegung von {n} summiert auf {sum(w)}, "
                             f"nicht auf {N_ITEMS}.")
    links = [0.0] * len(ZERLEGUNG)
    for k, (lab, col) in enumerate(TEILE):
        anteil = [w[k] / N_ITEMS for w in werte]
        ax1.barh(y, anteil, left=links, color=col, label=lab, height=0.62,
                 zorder=3, linewidth=0.3, edgecolor="white")
        links = [l + a for l, a in zip(links, anteil)]
    ax1.set_yticks(y)
    ax1.set_yticklabels([lab for _, lab, _, _ in ZERLEGUNG])
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1)
    ax1.set_xticks([0, .25, .5, .75, 1.0])
    ax1.set_xticklabels([".00", ".25", ".50", ".75", "1.00"])
    ax1.set_xlabel(f"Share of the {S.tausender(N_ITEMS)} gold toponyms")
    ax1.set_title("(a) Error decomposition, six systems", loc="left", fontsize=9.2)
    ax1.axvline(OBERGRENZE, color=S.C_LINE, linewidth=0.9,
                linestyle=(0, (4, 2)), zorder=5)
    ax1.text(OBERGRENZE + 0.013, 0.02, f"upper bound {S.zahl(OBERGRENZE)}",
             transform=ax1.get_xaxis_transform(), rotation=90,
             fontsize=6.9, ha="left", va="bottom", color=S.C_LINE)
    leg = ax1.legend(loc="upper center", bbox_to_anchor=(0.48, -0.20), ncol=2,
                     columnspacing=1.2, handlelength=1.1, fontsize=7.2)
    S.style(ax1, "x")

    # ── rechts: Leitmetrik ueber alle zwanzig Zellen ─────────────────────────
    M = np.array([[AMBIG[m][v] for v in VARIANTEN] for m in MODELLE])
    bild = ax2.imshow(M, cmap=CMAP, vmin=0.10, vmax=0.90, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            w = M[i, j]
            ax2.text(j, i, S.zahl(w), ha="center", va="center", fontsize=7.4,
                     color="#ffffff" if w > 0.66 else "#1a1a1a",
                     fontweight="bold" if w == M.max() else "normal")
    bi, bj = divmod(int(M.argmax()), M.shape[1])
    ax2.add_patch(plt.Rectangle((bj - .5, bi - .5), 1, 1, fill=False,
                                edgecolor=S.C_WARN, linewidth=1.5, zorder=5))
    ax2.axvline(2.5, color="#ffffff", linewidth=2.2, zorder=4)
    ax2.set_xticks(range(len(VARIANTEN)))
    ax2.set_xticklabels([SPALTE[v] for v in VARIANTEN], fontsize=8)
    ax2.set_yticks(range(len(MODELLE)))
    ax2.set_yticklabels(MODELLE, fontsize=8)
    ax2.tick_params(length=0)
    for s in ax2.spines.values():
        s.set_visible(False)
    ax2.set_title(f"(b) Lead metric, {S.tausender(N_AMBIG)} ambiguous toponyms",
                  loc="left", fontsize=9.2)
    cb = fig.colorbar(bild, ax=ax2, fraction=0.035, pad=0.025)
    cb.set_label("Acc@1 (ambiguous toponyms)", fontsize=7.6)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.4)

    fig.subplots_adjust(wspace=0.30)
    anm = None   # Fusszeile entfernt (26.09.2026)

    if pruefen and anm is not None:
        _messen(fig, ax1, leg, anm)

    # ── die gezeichneten Werte daneben ───────────────────────────────────────
    zeilen = [["a", n, f, w, f"{w / N_ITEMS:.4f}"]
              for (n, _, _, _), werte_n in zip(ZERLEGUNG, werte)
              for f, w in zip([t for t, _ in TEILE], werte_n)]
    zeilen += [["a", "alle sechs Systeme", "obergrenze_acc1",
                f"{OBERGRENZE:.4f}", ""]]
    zeilen += [["b", f"{m}/{v}", "acc1_ambiguous_only", f"{AMBIG[m][v]:.4f}", ""]
               for m in MODELLE for v in VARIANTEN]
    S.daten_ablegen(out, NAME,
                    ["feld", "system", "kennzahl", "wert", "anteil_an_75741"],
                    zeilen, datum)
    S.speichern(fig, out, NAME, datum)


def _messen(fig, ax1, leg, anm) -> None:
    """Misst Zeilenabstand und Ueberschneidungen — Druckregel «lesbar in
    Spaltenbreite, keine ueberlappenden Beschriftungen»."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    dpi = fig.dpi
    h = ax1.get_window_extent(r).height / dpi
    print(f"  Feld (a): Achsenhoehe {h:.3f} in, {len(ZERLEGUNG)} Zeilen, "
          f"Zeilenabstand {h / len(ZERLEGUNG) * 72:.1f} pt, "
          f"Balkendicke {h / len(ZERLEGUNG) * 0.62 * 72:.1f} pt")
    ab = anm.get_window_extent(r)
    print(f"  Anmerkung: {ab.width / dpi:.2f} in breit, "
          f"{ab.height / dpi:.2f} in hoch")
    bb = fig.get_tightbbox(r)
    print(f"  Tafel gesamt (tight): {bb.width:.2f} x {bb.height:.2f} in")
    kaesten = {"Legende": leg.get_window_extent(r),
               "Anmerkung": anm.get_window_extent(r),
               "x-Titel": ax1.xaxis.label.get_window_extent(r)}
    for t in ax1.get_yticklabels():
        kaesten.setdefault("y-Beschriftung", t.get_window_extent(r))
    namen = list(kaesten)
    for i in range(len(namen)):
        for j in range(i + 1, len(namen)):
            a, b = kaesten[namen[i]], kaesten[namen[j]]
            if a.overlaps(b):
                print(f"  ACHTUNG Ueberschneidung: {namen[i]} / {namen[j]}")
            else:
                print(f"  frei: {namen[i]} / {namen[j]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(S.AUSGABE))
    p.add_argument("--datum", default=DATUM)
    p.add_argument("--pruefen", action="store_true")
    a = p.parse_args()
    zeichne(Path(a.out), a.datum, a.pruefen)


if __name__ == "__main__":
    main()
