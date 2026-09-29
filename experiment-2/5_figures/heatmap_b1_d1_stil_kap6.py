# -*- coding: utf-8 -*-
"""
Hilfsmodul fuer Abbildung 4 (abb_4_matrizen_untereinander.py): Einlesen der
beiden Assoziationsmatrizen, Objektklassen aus den swissNAMES3D-DBF-Dateien,
Ordnung nach Objektklasse, Maske der Partner, die der Resolver liest (Wert >=
0.001, je Zeile die zehn staerksten), und das Zeichnen eines Feldes.

Herkunft: die Funktionen der Heatmap-Fassung vom 18.09.2026 (zwei Felder
nebeneinander); deren eigener Hauptteil, der eine inzwischen ueberholte
Abbildung schrieb, ist entfernt. Kein Wert und keine Funktion ist geaendert.

EINGABE (ueber abb_4)  ma-experiments/matrices/config1/b1_matrix.csv  (B1)
                       ma-experiments/matrices/config1/d1_matrix.csv  (D1)
                       ma-experiments/data/swissnames3d/swissNAMES3D_{PKT,LIN,PLY}.dbf
                       (Attribut OBJEKTKLASSE, im DBF-Feldnamen auf OBJEKTKLAS gekuerzt)
Module kategorie_anzeigenamen.py (Umlaute) und stil_kap6.py (Bildsprache).
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as _dt
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patheffects as pfx
import matplotlib.transforms as mtrans
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from kategorie_anzeigenamen import anzeige

import stil_kap6 as S

S.setze_stil()

NAME = "heatmap_assoziationsmatrizen_B1_D1_en_stil_kap6_2026-09-18"
ORDNUNGSTABELLE = "Kategorie_Objektklasse_Gruppe_stil_kap6_2026-09-18.csv"
LEGENDENTEXT = "Abbildung_Heatmap_Legende_stil_kap6_2026-09-18.md"

SCHWELLE = 0.001             # assoc_threshold des Resolvers
KAPPUNG = 10                 # max_categories des Resolvers
SKALA_UNTEN, SKALA_OBEN = 0.001, 0.5
# Die Toene kommen aus stil_kap6.py. FARBE_REST ist der neue, deutlich hellere
# Neutralton des Moduls; die Skala endet neu in einem nahezu schwarzen Ton.
FARBE_REST = S.C_UNTERSCHWELLE   # unbunt: alles, was der Resolver nicht liest
FARBE_DIAGONALE = "#ffffff"
FARBE_GITTER = "#4a4a4a"         # axes.edgecolor aus stil_kap6.RC
FARBE_EIGENGRUPPE = S.C_WARN     # Hervorhebungsfarbe des Moduls
FARBKARTE = S.KARTE_SEQ

# Die fuenf Kategorien aus swissBOUNDARIES3D, die in den Shapefiles nicht vorkommen.
VERWALTUNG = ("Gemeindegebiet", "Kommunanz", "Bezirk", "Kantonsgebiet", "Kanton")
KLASSE_VERWALTUNG = "swissBOUNDARIES3D"

# Die vierzehn Bloecke: Anzeigename und die Objektklassen, die darin aufgehen.
BLOECKE: list[tuple[str, tuple[str, ...]]] = [
    ("Summits and passes",           ("TLM_NAME_PKT",)),
    ("Terrain names",                ("TLM_GELAENDENAME",)),
    ("Area names",                   ("TLM_GEBIETSNAME",)),
    ("Field names",                  ("TLM_FLURNAME",)),
    ("Single objects and minor landforms", ("TLM_EINZELOBJEKT", "TLM_MORPH_KLEINFORM_PKT")),
    ("Water bodies and water structures", ("TLM_FLIESSGEWAESSER", "TLM_STEHENDES_GEWAESSER",
                                      "TLM_STAUBAUTE")),
    ("Settlement names",             ("TLM_SIEDLUNGSNAME",)),
    ("Administrative units",         (KLASSE_VERWALTUNG,)),
    ("Buildings",                    ("TLM_GEBAEUDE",)),
    ("Land-use areas",               ("TLM_NUTZUNGSAREAL",)),
    ("Leisure and sports facilities", ("TLM_FREIZEITAREAL", "TLM_SPORTBAUTE_LIN")),
    ("Transport areas",              ("TLM_VERKEHRSAREAL",)),
    ("Road network",                 ("TLM_STRASSE", "TLM_STRASSENINFO", "TLM_AUS_EINFAHRT")),
    ("Public transport",             ("TLM_EISENBAHN", "TLM_UEBRIGE_BAHN",
                                      "TLM_HALTESTELLE", "TLM_SCHIFFFAHRT")),
]


def _tsd(n) -> str:
    """Schweizer Tausendertrennzeichen: 1070 -> 1’070 (REGELN, Teil 1)."""
    return f"{int(n):,}".replace(",", "’")


# --- DBF ------------------------------------------------------------------
def dbf_lesen(pfad: Path, felder: list[str]):
    """Minimaler Leser fuer dBase-III-Dateien; das Paket dbfread wird nicht gebraucht."""
    with pfad.open("rb") as f:
        kopf = f.read(32)
        n_datensaetze, kopflaenge, satzlaenge = struct.unpack("<I H H", kopf[4:12])
        beschreibung, versatz = [], 1
        while True:
            d = f.read(32)
            if d[0:1] == b"\r":
                break
            name = d[0:11].split(b"\x00")[0].decode("latin-1")
            laenge = d[16]
            beschreibung.append((name, versatz, laenge))
            versatz += laenge
        register = {name: (v, l) for name, v, l in beschreibung}
        fehlend = [x for x in felder if x not in register]
        if fehlend:
            raise SystemExit(f"{pfad.name}: Feld fehlt: {', '.join(fehlend)}")
        auswahl = [register[x] for x in felder]
        f.seek(kopflaenge)
        for _ in range(n_datensaetze):
            satz = f.read(satzlaenge)
            if not satz or satz[0:1] == b"*":
                continue
            yield [satz[v:v + l].decode("utf-8", "replace").strip() for v, l in auswahl]


def klassenzuordnung(shp: Path) -> tuple[dict[str, str], collections.Counter]:
    """Liest OBJEKTART -> OBJEKTKLASSE und die Objektzahl je Objektart."""
    zaehler = collections.Counter()
    klassen = collections.defaultdict(collections.Counter)
    for teil in ("PKT", "LIN", "PLY"):
        for art, klasse in dbf_lesen(shp / f"swissNAMES3D_{teil}.dbf",
                                     ["OBJEKTART", "OBJEKTKLAS"]):
            zaehler[art] += 1
            klassen[art][klasse] += 1
    return {art: m.most_common(1)[0][0] for art, m in klassen.items()}, zaehler


# --- Matrizen -------------------------------------------------------------
def matrix_lesen(pfad: Path) -> pd.DataFrame:
    m = pd.read_csv(pfad, sep=";", index_col=0, encoding="utf-8")
    m.index = [str(i) for i in m.index]
    m.columns = [str(c) for c in m.columns]
    if list(m.index) != list(m.columns):
        raise SystemExit(f"{pfad.name}: Zeilen- und Spaltennamen stimmen nicht ueberein")
    return m


def ordnung_bilden(kategorien: list[str], art2klasse: dict[str, str]):
    """Gibt die Kategorienreihenfolge und die Blockgrenzen zurueck."""
    klasse_von = {}
    for k in kategorien:
        klasse_von[k] = KLASSE_VERWALTUNG if k in VERWALTUNG else art2klasse.get(k)
    ohne = [k for k, v in klasse_von.items() if v is None]
    if ohne:
        raise SystemExit("Ohne Objektklasse: " + ", ".join(ohne))

    block_von_klasse = {}
    for name, klassen in BLOECKE:
        for kl in klassen:
            block_von_klasse[kl] = name
    unbekannt = sorted({v for v in klasse_von.values() if v not in block_von_klasse})
    if unbekannt:
        raise SystemExit("Diese Objektklassen sind keinem Block zugeordnet: "
                         + ", ".join(unbekannt))

    reihenfolge, grenzen, beschriftung = [], [], []
    for nummer, (name, _) in enumerate(BLOECKE, start=1):
        mitglieder = sorted((k for k in kategorien if block_von_klasse[klasse_von[k]] == name),
                            key=lambda k: anzeige(k).lower())
        if not mitglieder:
            raise SystemExit(f"Block ohne Mitglieder: {name}")
        beschriftung.append((f"{nummer} {name}", len(reihenfolge), len(mitglieder)))
        reihenfolge.extend(mitglieder)
        grenzen.append(len(reihenfolge))
    if len(reihenfolge) != len(kategorien):
        raise SystemExit("Die Ordnung deckt nicht alle Kategorien ab")
    return reihenfolge, grenzen[:-1], beschriftung, klasse_von


# --- Zeichnen -------------------------------------------------------------
def auseinanderziehen(mitten: list[float], mindestabstand: float,
                      untere: float, obere: float) -> list[float]:
    """Schiebt zu dicht liegende Beschriftungen auseinander.

    Die Bloecke sind verschieden gross; die Mitten der kleinsten liegen so dicht
    beieinander, dass sich ihre Beschriftungen ueberdecken wuerden. Ein Vorwaerts-
    und ein Rueckwaertsdurchgang erzwingen einen Mindestabstand, ohne die
    Reihenfolge zu aendern. Die wahre Blockmitte bleibt erhalten und wird mit
    einer Fuehrungslinie angezeigt.
    """
    p = list(mitten)
    for i in range(1, len(p)):
        p[i] = max(p[i], p[i - 1] + mindestabstand)
    ueberhang = p[-1] - obere
    if ueberhang > 0:
        p = [x - ueberhang for x in p]
    for i in range(len(p) - 2, -1, -1):
        p[i] = min(p[i], p[i + 1] - mindestabstand)
    p[0] = max(p[0], untere)
    for i in range(1, len(p)):
        p[i] = max(p[i], p[i - 1] + mindestabstand)
    return p


def resolversicht(werte: np.ndarray) -> np.ndarray:
    """Maske der Partner, die der Resolver wirklich liest.

    Genau die Auswahlregel des Resolvers und die von kap6_basis.partner():
    Wert >= SCHWELLE, davon je Zeile die KAPPUNG staerksten, Diagonale nie.
    """
    n = werte.shape[0]
    w = werte.copy()
    np.fill_diagonal(w, -np.inf)
    maske = np.zeros((n, n), dtype=bool)
    for i in range(n):
        kandidaten = np.where(w[i] >= SCHWELLE)[0]
        if kandidaten.size:
            maske[i, kandidaten[np.argsort(-w[i][kandidaten])][:KAPPUNG]] = True
    return maske


def feld_zeichnen(ax, werte: np.ndarray, maske: np.ndarray, norm, grenzen,
                  spannen, titel: str):
    n = werte.shape[0]
    maske_diag = np.eye(n, dtype=bool)

    ax.set_facecolor(FARBE_REST)
    darstellbar = np.ma.masked_where(~maske, werte)
    bild = ax.imshow(darstellbar, cmap=FARBKARTE, norm=norm, origin="upper",
                     interpolation="nearest", aspect="equal")
    nur_diag = np.ma.masked_where(~maske_diag, np.zeros_like(werte))
    ax.imshow(nur_diag, cmap=matplotlib.colors.ListedColormap([FARBE_DIAGONALE]),
              origin="upper", interpolation="nearest", aspect="equal", zorder=3)

    for g in grenzen:
        ax.axhline(g - 0.5, color=FARBE_GITTER, linewidth=0.45, zorder=4)
        ax.axvline(g - 0.5, color=FARBE_GITTER, linewidth=0.45, zorder=4)
    # Das Quadrat auf der Blockdiagonale ist der Bereich der eigenen Gruppe.
    # Ohne diese Umrandung ist die Haeufung in der eigenen Gruppe im Druck
    # nicht als Struktur zu erkennen, sondern nur als Wolke.
    for start, laenge in spannen:
        ax.add_patch(plt.Rectangle((start - 0.5, start - 0.5), laenge, laenge,
                                   fill=False, edgecolor=FARBE_EIGENGRUPPE,
                                   linewidth=0.85, zorder=5))
    for rand in ("top", "bottom", "left", "right"):
        ax.spines[rand].set_color("#555555")
        ax.spines[rand].set_linewidth(0.8)
    ax.set_title(titel, fontsize=10, pad=8)
    return bild
