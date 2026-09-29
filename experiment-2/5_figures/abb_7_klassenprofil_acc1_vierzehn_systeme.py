#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Erzeugt: Abbildung 7 — Acc@1 je Objektklasse (OBJEKTART) und System, Fassung
         vom 26.09.2026 mit VIERZEHN Spalten: M1/E_default, M2/E_default und
         M3, M4, M5 mit allen vier Sentence Generators. Ohne Titel und
         Fusszeile; ohne rote Umrandung der E_d1-Spalten. Die Figurgroesse ist
         auf die Satzbreite von 16 cm ausgelegt, damit die Schrift im Druck
         dieselbe Groesse hat wie in den uebrigen Abbildungen.
         Klassen mit n >= 150; kleinere Klassen sind in `Other` gebuendelt.
         Zahlen, Eingabedateien und Proben wie in
         abb_6-3_klassenprofil_acc1_zwanzig_systeme.py (unveraendert).

         --karte waehlt den Farbverlauf: `sandnavy` (Vorgabe, Fassung der Arbeit);
         `warm` ist ein mehrfarbiger,
         helligkeitsmonotoner Verlauf von Creme ueber Sand und Salbei zum
         Marineblau der Palette und nahe Schwarz; `cividis` die gleichnamige
         Matplotlib-Karte zum Vergleich; `haus` die bisherige S.KARTE_SEQ.

Aufruf:  python3 abb_7_klassenprofil_acc1_vierzehn_systeme.py
                [--wurzel <experiment-2>] [--out <Zielordner>] [--datum JJJJ-MM-TT]
                [--karte sandnavy|warm|...]

Datum:   2026-09-26
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import stil_kap6 as S

DATUM = "2026-09-26"
NAME = "abb_7_klassenprofil_acc1_vierzehn_systeme_en"

MODELLE = ["M1", "M2", "M3", "M4", "M5"]
VARIANTEN = ["E_default", "E_c1", "E_c2", "E_d1"]
# Die vierzehn gezeichneten Spalten (Nutzerentscheid 26.09.2026).
SPALTEN = [("M1", "E_default"), ("M2", "E_default")] + \
          [(m, v) for m in ("M3", "M4", "M5") for v in VARIANTEN]

# Farbverlaeufe. `warm` ist helligkeitsmonoton (L* 94.8, 81.7, 71.6, 60.2,
# 35.1, 24.0, 6.4), die Ordnung bleibt also auch in Graustufen lesbar.
from matplotlib.colors import LinearSegmentedColormap as _LSC
KARTE_WARM = _LSC.from_list("kap6_warm", [
    (0.00, "#f6f0dc"), (0.15, "#e2c98a"), (0.27, "#a9b39a"), (0.40, "#6f96b0"),
    (0.68, "#2e5677"), (0.80, S.C_BEST), (1.00, S.C_ENDE)])


def _entsaettigt(name: str, faktor: float, n: int = 256):
    """Matplotlib-Karte mit reduzierter Buntheit (Chroma in CIELAB skaliert,
    Helligkeit L* unveraendert). Reihenfolge und Farbtonwechsel bleiben."""
    import numpy as np
    rgb = matplotlib.colormaps[name](np.linspace(0, 1, n))[:, :3]
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[:, 1] - 16; a = 500 * (f[:, 0] - f[:, 1]) * faktor; b = 200 * (f[:, 1] - f[:, 2]) * faktor
    fy = (L + 16) / 116; fx = a / 500 + fy; fz = fy - b / 200
    def inv(t): return np.where(t ** 3 > 0.008856, t ** 3, (t - 16 / 116) / 7.787)
    xyz2 = np.stack([inv(fx), inv(fy), inv(fz)], 1) * np.array([0.95047, 1.0, 1.08883])
    lin2 = xyz2 @ np.linalg.inv(M).T
    rgb2 = np.where(lin2 <= 0.0031308, 12.92 * lin2, 1.055 * np.clip(lin2, 0, None) ** (1 / 2.4) - 0.055)
    return _LSC.from_list(f"{name}_c{int(faktor * 100)}", np.clip(rgb2, 0, 1))


# YlGnBu-Verlauf (Gelb -> Gruen -> Blau) mit gedaempfter Buntheit, damit er
# neben der uebrigen Palette (Marineblau, Ziegelrot, Grau) nicht grell wirkt.
KARTE_YLGNBU_60 = _entsaettigt("YlGnBu", 0.60)
KARTE_YLGNBU_45 = _entsaettigt("YlGnBu", 0.45)



def _L(hexfarbe: str) -> float:
    import numpy as np
    rgb = np.array([int(hexfarbe[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return 116 * y ** (1 / 3) - 16 if y > 0.008856 else 903.3 * y


def _karte_nach_helligkeit(name: str, toene):
    """Stuetzstellen nach ihrer Helligkeit L* verteilt: gleiche L*-Schritte je
    Einheit des Werts, damit der Verlauf wahrnehmungsgleich faellt."""
    L = [_L(t) for t in toene]
    assert all(a > b for a, b in zip(L, L[1:])), ("nicht helligkeitsmonoton", list(zip(toene, L)))
    pos = [(L[0] - l) / (L[0] - L[-1]) for l in L]
    return _LSC.from_list(name, list(zip(pos, toene)))


# Eigene Karte in der Farbfolge von YlGnBu (Gelb -> Gruen -> Blau), aber aus
# der Palette der Arbeit: sie endet bei S.C_BEST, dem dunkelsten Blau der
# uebrigen Abbildungen, und beginnt mit einem gedeckten, satten Gelb.
KARTE_HAUS_YGB = _karte_nach_helligkeit("kap6_gelb_gruen_blau", [
    "#e3cf5c",   # gedecktes Gelb
    "#c6c96a",   # Gelbgruen
    "#9dbd82",   # Gruen
    "#74ac98",   # Gruen-Tuerkis
    "#5c98a8",   # Tuerkis-Blau
    "#4d80a3",   # Blau
    "#385f83",   # dunkles Blau
    S.C_BEST,    # Marineblau der Palette
])

# Vom Nutzer vorgegebener Verlauf (26.09.2026): Gelb, Gruen genau in der Mitte,
# mittleres Blau als Uebergang, dunkles Blau am Ende.
KARTE_NUTZER = _LSC.from_list("gelb_gruen_blau", [
    (0.00, "#E8CE4E"), (0.50, "#7CB58A"), (0.75, "#4A7A9A"), (1.00, "#2C3E5C")])

# Endgueltiger Verlauf nach Vorgabe des Nutzers (26.09.2026): Creme, Sand,
# Uebergang, Blau, Navy.
KARTE_SAND_NAVY = _LSC.from_list("sand_navy", [
    (0.00, "#f4eed9"), (0.15, "#f0e5c7"), (0.40, "#dcc999"), (0.50, "#a3ac95"),
    (0.60, "#668ba5"), (0.70, "#4f738e"), (0.85, "#284764"), (1.00, "#161c25")])

KARTEN = {"warm": KARTE_WARM, "hausygb": KARTE_HAUS_YGB, "nutzer": KARTE_NUTZER,
          "sandnavy": KARTE_SAND_NAVY,
          "ylgnbu60": KARTE_YLGNBU_60, "ylgnbu45": KARTE_YLGNBU_45, "cividis": "cividis", "haus": S.KARTE_SEQ,
          "ylgnbu": "YlGnBu", "viridis": "viridis_r", "pubugn": "PuBuGn", "gnbu": "GnBu"}

# Modellschluessel der Ergebnisdateien -> Kurzname der Abbildung.
MODELLSCHLUESSEL = {
    "M1_dguzh": "M1",
    "M2_distiluse_base": "M2",
    "M3_default_finetuned": "M3",
    "M4_spatial_config1": "M4",
    "M5_spatial_config2": "M5",
}
# Das JSON schreibt zwei Klassennamen ohne Umlaut; die Abbildung nicht.
UMLAUT = {"Huegelzug": "Hügelzug", "Haupthuegel": "Haupthügel"}


def klassenname(roh: str) -> str:
    """Ein Klassenname, drei Schreibungen. Das JSON schreibt `Huegelzug`, die
    CSV der Auswertung schreibt den Umlaut zerlegt (macOS, NFD), das Register
    zusammengesetzt (NFC). Hier wird alles auf NFC mit Umlaut gebracht."""
    n = unicodedata.normalize("NFC", roh.strip())
    return UMLAUT.get(n, n)

MIN_N = 150


# ── Einlesen ─────────────────────────────────────────────────────────────────
def lies_bestand(wurzel: Path) -> tuple[dict, dict]:
    """Die fuenfzehn bestehenden Systeme aus `stratified_objektart.csv`."""
    q = wurzel / "results" / "tables" / "stratified_objektart.csv"
    if not q.exists():
        sys.exit(f"Eingabedatei fehlt: {q}")
    acc, n_klasse = {}, {}
    with open(q, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            m = MODELLSCHLUESSEL.get(r["model"])
            if m is None:
                continue
            klasse = klassenname(r["objektart"])
            acc[(m, r["eval_resolver"], klasse)] = float(r["acc1"])
            n_klasse[klasse] = int(r["n"])
    return acc, n_klasse


def lies_e_d1(wurzel: Path) -> tuple[dict, list]:
    """Die fuenf neuen Zellen auf `E_d1` aus `ERGEBNIS_C_nachlauf.json`."""
    q = wurzel / "results" / "ERGEBNIS_C_nachlauf.json"
    if not q.exists():
        sys.exit(f"Eingabedatei fehlt: {q}")
    with open(q, encoding="utf-8") as fh:
        k = json.load(fh)["klassenprofil_E_d1"]
    if int(k["min_n"]) != MIN_N:
        sys.exit(f"min_n im JSON ist {k['min_n']}, erwartet {MIN_N}")
    acc = {}
    for schluessel, je_klasse in k["alle_modelle_E_d1"].items():
        m = MODELLSCHLUESSEL[schluessel]
        for klasse, w in je_klasse.items():
            acc[(m, "E_d1", klassenname(klasse))] = float(w["acc1"])
    return acc, k["zeilen"]


def lies_register(wurzel: Path) -> dict:
    """Die zwanzig tabellierten M5/`E_d1`-Werte aus Register C, Abschnitt 4."""
    # Das Zahlenregister der Arbeit liegt nicht im Repo; die Probe (a) entfaellt dann.
    q = wurzel / "results" / "Zahlenregister_C_Nachlauf_E-d1.md"
    if not q.exists():
        print(f"  Hinweis: Registerdatei fehlt ({q.name}) — Probe (a) entfaellt.")
        return {}
    text = q.read_text(encoding="utf-8")
    abschnitt = text.split("## 4 Klassenprofil")[1].split("## 5 ")[0]
    werte = {}
    for zeile in abschnitt.splitlines():
        m = re.match(r"\|\s*\**Acc@1 ([^|*]+?)\**\s*\|\s*\**\.?(\d{4})\**\s*\|", zeile)
        if m:
            klasse = klassenname(m.group(1).replace("(gebündelt)", ""))
            werte[klasse] = float("0." + m.group(2))
    return werte


# ── Proben ───────────────────────────────────────────────────────────────────
def pruefe(acc_alt: dict, acc_neu: dict, zeilen: list, register: dict) -> None:
    fehler = []

    # a) Register Abschnitt 4 gegen die M5-Eintraege des JSON.
    if register and len(register) != 20:
        fehler.append(f"Register Abschnitt 4: {len(register)} Zeilen statt 20")
    for klasse, wert in register.items():
        hat = acc_neu.get(("M5", "E_d1", klasse))
        if hat is None:
            fehler.append(f"Register fuehrt {klasse!r}, das JSON nicht")
        elif round(hat, 4) != wert:
            fehler.append(f"{klasse}: Register {wert:.4f} gegen JSON {hat:.4f}")

    # b) Gueltigkeitsbeleg: das JSON reproduziert zwei bestehende Spalten.
    paare = (("acc1_M3_E_default", ("M3", "E_default")),
             ("acc1_M5_E_c1", ("M5", "E_c1")))
    for z in zeilen:
        klasse = klassenname(z["objektart"])
        if klasse == "Other":
            continue
        for schluessel, (m, v) in paare:
            alt = acc_alt.get((m, v, klasse))
            if alt is None:
                fehler.append(f"{klasse}/{m}+{v} fehlt im Bestand")
            elif round(float(z[schluessel]), 4) != round(alt, 4):
                fehler.append(f"{klasse}/{m}+{v}: JSON {z[schluessel]:.4f} "
                              f"gegen Bestand {alt:.4f}")

    # c) alle_modelle_E_d1[M5] gegen die tabellierte Spalte acc1_M5_E_d1.
    for z in zeilen:
        klasse = klassenname(z["objektart"])
        hat = acc_neu.get(("M5", "E_d1", klasse))
        if hat is None or round(hat, 6) != round(float(z["acc1_M5_E_d1"]), 6):
            fehler.append(f"{klasse}: alle_modelle[M5] weicht von acc1_M5_E_d1 ab")

    if fehler:
        print("Die Proben sind nicht aufgegangen:", file=sys.stderr)
        for f in fehler:
            print("  -", f, file=sys.stderr)
        sys.exit(1)
    print(f"  Proben aufgegangen: {len(register)} Registerzeilen, "
          f"{len(zeilen) - 1} x 2 reproduzierte Spalten, {len(zeilen)} M5-Zellen.")


# ── Zeichnen ─────────────────────────────────────────────────────────────────
def zeichne(wurzel: Path, out: Path, datum: str, karte: str = "warm") -> None:
    acc_alt, n_klasse = lies_bestand(wurzel)
    acc_neu, zeilen = lies_e_d1(wurzel)
    pruefe(acc_alt, acc_neu, zeilen, lies_register(wurzel))
    acc = {**acc_alt, **acc_neu}

    # Klassen nach Korpushaeufigkeit, `Other` immer zuletzt.
    klassen = sorted((k for k in n_klasse if k != "Other"),
                     key=lambda k: -n_klasse[k])
    if "Other" in n_klasse:
        klassen.append("Other")
    spalten = SPALTEN

    M = np.array([[acc[(m, v, k)] for m, v in spalten] for k in klassen])

    S.setze_stil()
    fig, ax = plt.subplots(figsize=(6.5, 6.6))
    bild = ax.imshow(M, cmap=KARTEN[karte], vmin=0.0, vmax=1.0, aspect="auto")

    # Werte in die Felder. Zweistellig, weil vierstellig in einem Feld von
    # zwanzig mal zwanzig ueberlappte; die vierstelligen Werte stehen in der
    # `__daten.csv` daneben (REGELN, Teil 1: keine ueberlappenden Beschriftungen).
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            w = M[i, j]
            ax.text(j, i, f"{w:.2f}".lstrip("0"), ha="center", va="center",
                    fontsize=5.6, color=_textfarbe(bild.cmap(w)))

    # Modellgrenzen: weisse Trennlinie, wo das Modell wechselt; die E_d1-Spalten
    # (bestes Mass) tragen wie in der alten Fassung einen feinen roten Rahmen.
    for j in range(1, len(spalten)):
        if spalten[j][0] != spalten[j - 1][0]:
            ax.axvline(j - 0.5, color="#ffffff", linewidth=2.2, zorder=4)
    for j, (m, v) in enumerate(spalten):
        if v == "E_d1":
            ax.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, M.shape[0], fill=False,
                                       edgecolor=S.C_WARN, linewidth=1.0, zorder=5))

    ax.set_xticks(range(len(spalten)))
    ax.set_xticklabels([f"{m}/E$_\\mathrm{{{v[2:]}}}$" for m, v in spalten], rotation=60,
                       ha="right", fontsize=6.6)
    ax.set_yticks(range(len(klassen)))
    ax.set_yticklabels([f"{k}  (n = {S.tausender(n_klasse[k])})" for k in klassen],
                       fontsize=6.8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("System (model / sentence generator)", fontsize=8, labelpad=5)
    ax.set_ylabel("Object class (OBJEKTART, gold)", fontsize=8)

    cb = fig.colorbar(bild, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label("Acc@1 (share of items in the class)", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    cb.outline.set_linewidth(0.4)


    S.daten_ablegen(out, NAME + ("" if karte == "warm" else f"_{karte}"),
                    ["objektart", "n", "modell", "variante", "acc1"],
                    [[k, n_klasse[k], m, v, f"{acc[(m, v, k)]:.4f}"]
                     for k in klassen for m, v in spalten], datum)
    S.speichern(fig, out, NAME + ("" if karte == "warm" else f"_{karte}"), datum)


def _textfarbe(rgba) -> str:
    """Weiss auf dunklen, dunkelgrau auf hellen Zellen (relative Leuchtdichte)."""
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgba[:3]]
    y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return "#ffffff" if y < 0.18 else "#1a1a1a"


def wurzel_vorgabe() -> Path:
    """Ordner experiment-2 (enthaelt results/)."""
    return S.EXP2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wurzel", default=str(wurzel_vorgabe()),
                   help="Ordner experiment-2 (Vorgabe: aus dem Skriptort abgeleitet)")
    p.add_argument("--out", default=str(S.AUSGABE),
                   help="Zielordner der Bilddateien (Vorgabe: 5_figures/output/)")
    p.add_argument("--datum", default=DATUM, help="Datum im Dateinamen")
    p.add_argument("--karte", default="sandnavy", choices=sorted(KARTEN))
    a = p.parse_args()
    zeichne(Path(a.wurzel), Path(a.out), a.datum, a.karte)


if __name__ == "__main__":
    main()
