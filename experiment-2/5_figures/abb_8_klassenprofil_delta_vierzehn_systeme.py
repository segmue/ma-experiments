#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Erzeugt: Abbildung 8 — Delta Acc@1 je Objektklasse gegen die faire Baseline
         M3/E_default, Fassung vom 26.09.2026 mit VIERZEHN Spalten:
         M1/E_default, M2/E_default und M3, M4, M5 mit allen vier Sentence
         Generators. Ohne Titel und Fusszeile, ohne rote Umrandung der
         E_d1-Spalten; Schraffur (Verlust) und Fettdruck (gekappt) sind als
         Legende im Bild erklaert. Figurgroesse auf 16 cm Satzbreite ausgelegt.
         Zahlen, Eingabedateien und Proben wie in
         abb_6-4_klassenprofil_delta_zwanzig_systeme.py (unveraendert).

Aufruf:  python3 abb_8_klassenprofil_delta_vierzehn_systeme.py
                [--wurzel <experiment-2>] [--out <Zielordner>] [--datum JJJJ-MM-TT]

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
from matplotlib.colors import LinearSegmentedColormap

import stil_kap6 as S

DATUM = "2026-09-26"
NAME = "abb_8_klassenprofil_delta_vierzehn_systeme_en"

MODELLE = ["M1", "M2", "M3", "M4", "M5"]
VARIANTEN = ["E_default", "E_c1", "E_c2", "E_d1"]
# Die vierzehn gezeichneten Spalten (Nutzerentscheid 26.09.2026).
SPALTEN = [("M1", "E_default"), ("M2", "E_default")] + \
          [(m, v) for m in ("M3", "M4", "M5") for v in VARIANTEN]
BASIS = ("M3", "E_default")
KAPPUNG = 0.50          # Farbskala gekappt; die Werte selbst bleiben unberuehrt
MIN_N = 150

MODELLSCHLUESSEL = {
    "M1_dguzh": "M1", "M2_distiluse_base": "M2", "M3_default_finetuned": "M3",
    "M4_spatial_config1": "M4", "M5_spatial_config2": "M5",
}
UMLAUT = {"Huegelzug": "Hügelzug", "Haupthuegel": "Haupthügel"}

# Divergierende Karte aus den Toenen von `stil_kap6`; das Modul bleibt unberuehrt.
KARTE_DIV = LinearSegmentedColormap.from_list(
    "kap6_div", ((0.00, S.C_WARN), (0.28, "#d8a99c"), (0.50, S.C_UNTERSCHWELLE),
                 (0.72, S.C_DIAG), (0.90, S.C_BEST), (1.00, S.C_ENDE)))


def klassenname(roh: str) -> str:
    """Ein Klassenname, drei Schreibungen: `Huegelzug` im JSON, der Umlaut in
    den CSV zerlegt (macOS, NFD), im Register zusammengesetzt (NFC)."""
    n = unicodedata.normalize("NFC", roh.strip())
    return UMLAUT.get(n, n)


def tabellen(wurzel: Path) -> Path:
    return wurzel / "results" / "tables"


def datei(p: Path) -> Path:
    if not p.exists():
        sys.exit(f"Eingabedatei fehlt: {p}")
    return p


# ── Einlesen ─────────────────────────────────────────────────────────────────
def lies_bestand(wurzel: Path) -> tuple[dict, dict]:
    acc, n_klasse = {}, {}
    with open(datei(tabellen(wurzel) / "stratified_objektart.csv"),
              encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            m = MODELLSCHLUESSEL.get(r["model"])
            if m is None:
                continue
            k = klassenname(r["objektart"])
            acc[(m, r["eval_resolver"], k)] = float(r["acc1"])
            n_klasse[k] = int(r["n"])
    return acc, n_klasse


def lies_e_d1(wurzel: Path) -> tuple[dict, list]:
    q = datei(wurzel / "results" / "ERGEBNIS_C_nachlauf.json")
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
    # Das Zahlenregister der Arbeit liegt nicht im Repo; die Probe (a) entfaellt dann.
    q = wurzel / "results" / "Zahlenregister_C_Nachlauf_E-d1.md"
    if not q.exists():
        print(f"  Hinweis: Registerdatei fehlt ({q.name}) — Probe (a) entfaellt.")
        return {}
    abschnitt = q.read_text(encoding="utf-8").split(
        "## 4 Klassenprofil")[1].split("## 5 ")[0]
    werte = {}
    for zeile in abschnitt.splitlines():
        m = re.match(r"\|\s*\**Acc@1 ([^|*]+?)\**\s*\|\s*\**\.?(\d{4})\**\s*\|", zeile)
        if m:
            werte[klassenname(m.group(1).replace("(gebündelt)", ""))] = \
                float("0." + m.group(2))
    return werte


def lies_gazetteer(wurzel: Path) -> dict:
    gaz = {}
    with open(datei(tabellen(wurzel) / "class_profile.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                gaz[klassenname(r["objektart"])] = int(r["n_gazetteer"])
            except (KeyError, ValueError):
                pass
    return gaz


def lies_delta_bestand(wurzel: Path) -> dict:
    """Die abgelegten Differenzen der fuenfzehn Systeme — nur fuer die Probe."""
    d = {}
    with open(datei(tabellen(wurzel) / "stratified_objektart_delta.csv"),
              encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            roh = next(iter(r.values()))
            k = klassenname(re.sub(r"\s*\(.*\)\s*$", "", roh))
            for spalte, wert in r.items():
                if spalte and "+" in spalte and wert not in ("", None):
                    m, v = spalte.split("+", 1)
                    d[(m, v, k)] = float(wert)
    return d


# ── Proben ───────────────────────────────────────────────────────────────────
def pruefe(delta: dict, acc_neu: dict, zeilen: list, register: dict,
           delta_alt: dict, klassen: list, spalten: list) -> None:
    fehler = []

    if register and len(register) != 20:
        fehler.append(f"Register Abschnitt 4: {len(register)} Zeilen statt 20")
    for klasse, wert in register.items():
        hat = acc_neu.get(("M5", "E_d1", klasse))
        if hat is None:
            fehler.append(f"Register fuehrt {klasse!r}, das JSON nicht")
        elif round(hat, 4) != wert:
            fehler.append(f"{klasse}: Register {wert:.4f} gegen JSON {hat:.4f}")

    for z in zeilen:
        k = klassenname(z["objektart"])
        hier = delta.get(("M5", "E_d1", k))
        dort = float(z["delta_E_d1_vs_baseline"])
        if hier is None or round(hier, 4) != round(dort, 4):
            fehler.append(f"{k}: Differenz hier {hier} gegen JSON {dort:.4f}")

    n_probe = 0
    for k in klassen:
        for m, v in spalten:
            if v == "E_d1":
                continue
            dort = delta_alt.get((m, v, k))
            if dort is None:
                fehler.append(f"{k}/{m}+{v} fehlt in stratified_objektart_delta.csv")
            elif round(delta[(m, v, k)], 4) != round(dort, 4):
                fehler.append(f"{k}/{m}+{v}: hier {delta[(m, v, k)]:.4f} "
                              f"gegen Bestand {dort:.4f}")
            else:
                n_probe += 1

    for k in klassen:
        if delta[(*BASIS, k)] != 0.0:
            fehler.append(f"{k}: Baselinespalte ist nicht null")

    if fehler:
        print("Die Proben sind nicht aufgegangen:", file=sys.stderr)
        for f in fehler:
            print("  -", f, file=sys.stderr)
        sys.exit(1)
    print(f"  Proben aufgegangen: {len(register)} Registerzeilen, "
          f"{len(zeilen)} JSON-Differenzen, {n_probe} Bestandsdifferenzen, "
          f"{len(klassen)} Nullfelder der Baseline.")


# ── Zeichnen ─────────────────────────────────────────────────────────────────
def zeichne(wurzel: Path, out: Path, datum: str) -> None:
    acc_alt, n_klasse = lies_bestand(wurzel)
    acc_neu, zeilen = lies_e_d1(wurzel)
    acc = {**acc_alt, **acc_neu}
    gaz = lies_gazetteer(wurzel)

    klassen = sorted((k for k in n_klasse if k != "Other"), key=lambda k: -n_klasse[k])
    if "Other" in n_klasse:
        klassen.append("Other")
    spalten = SPALTEN

    delta = {(m, v, k): acc[(m, v, k)] - acc[(*BASIS, k)]
             for k in klassen for m, v in spalten}
    pruefe(delta, acc_neu, zeilen, lies_register(wurzel),
           lies_delta_bestand(wurzel), klassen, spalten)

    M = np.array([[delta[(m, v, k)] for m, v in spalten] for k in klassen])
    gekappt = np.abs(M) > KAPPUNG

    S.setze_stil()
    fig, ax = plt.subplots(figsize=(6.5, 6.6))
    bild = ax.imshow(M, cmap=KARTE_DIV, vmin=-KAPPUNG, vmax=KAPPUNG, aspect="auto")

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            w = M[i, j]
            hell = abs(w) < 0.30
            zahl = f"{abs(w):.2f}"[1:]
            ax.text(j, i, ("±" if zahl == ".00" else "+" if w > 0 else "−") + zahl,
                    ha="center", va="center", fontsize=5.6, zorder=3,
                    color="#1a1a1a" if hell else "#ffffff")

    for j in range(1, len(spalten)):
        if spalten[j][0] != spalten[j - 1][0]:
            ax.axvline(j - 0.5, color="#ffffff", linewidth=2.2, zorder=4)
    for j, (m, v) in enumerate(spalten):
        if v == "E_d1":
            ax.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, M.shape[0], fill=False,
                                       edgecolor=S.C_WARN, linewidth=1.0, zorder=5))
    j0 = spalten.index(BASIS)
    ax.add_patch(plt.Rectangle((j0 - 0.5, -0.5), 1, M.shape[0], fill=False,
                               edgecolor=S.C_LINE, linewidth=1.4,
                               linestyle=(0, (3, 2)), zorder=5))

    ax.set_xticks(range(len(spalten)))
    ax.set_xticklabels([f"{m}/E$_\\mathrm{{{v[2:]}}}$" for m, v in spalten], rotation=60,
                       ha="right", fontsize=6.6)
    ax.set_yticks(range(len(klassen)))
    ax.set_yticklabels(
        [f"{k}  (n = {S.tausender(n_klasse[k])}"
         + (f" · gaz. {S.tausender(gaz[k])})" if k in gaz else ")")
         for k in klassen], fontsize=6.8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("System (model / sentence generator)", fontsize=8, labelpad=5)
    ax.set_ylabel("Object class (OBJEKTART, gold · gazetteer entries)", fontsize=8)

    cb = fig.colorbar(bild, ax=ax, fraction=0.030, pad=0.02, extend="both")
    cb.set_label(f"Delta Acc@1 against M3/E_default, clipped at ±{KAPPUNG:.2f}",
                 fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    cb.outline.set_linewidth(0.4)

    # Legende im Bild: Schraffur = Verlust, Fettdruck = Farbskala gekappt,
    # gestrichelter Rahmen = Baseline-Spalte.
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    zeichen = [
        Line2D([0], [0], color=S.C_WARN, linewidth=1.2, label="red frame: E$_\mathrm{d1}$ columns"),
        Line2D([0], [0], color=S.C_LINE, linewidth=1.2, linestyle=(0, (3, 2)),
               label="dashed frame: baseline column"),
    ]
    ax.legend(handles=zeichen, loc="upper left", bbox_to_anchor=(-0.02, -0.19),
              ncol=2, frameon=False, fontsize=6.6, handlelength=1.6,
              columnspacing=1.6, handletextpad=0.6)

    S.daten_ablegen(out, NAME,
                    ["objektart", "n", "n_gazetteer", "modell", "variante",
                     "acc1", "delta_gegen_M3_E_default", "farbskala_gekappt"],
                    [[k, n_klasse[k], gaz.get(k, ""), m, v,
                      f"{acc[(m, v, k)]:.4f}", f"{delta[(m, v, k)]:+.4f}",
                      "ja" if abs(delta[(m, v, k)]) > KAPPUNG else "nein"]
                     for k in klassen for m, v in spalten], datum)
    S.speichern(fig, out, NAME, datum)


def wurzel_vorgabe() -> Path:
    """Ordner experiment-2 (enthaelt results/)."""
    return S.EXP2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wurzel", default=str(wurzel_vorgabe()))
    p.add_argument("--out", default=str(S.AUSGABE))
    p.add_argument("--datum", default=DATUM)
    a = p.parse_args()
    zeichne(Path(a.wurzel), Path(a.out), a.datum)


if __name__ == "__main__":
    main()
