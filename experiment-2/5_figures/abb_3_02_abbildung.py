"""Abbildung 3 der Arbeit (Abb. 4.3, Assoziationsmasse) — Schritt 2: Zeichnen.

Liest output/abb_3_daten/abb_4_3_daten.json (aus abb_3_01_daten_sammeln.py) und zeichnet zwei Karten
desselben Ausschnitts um den Säntis:

  (a) Überlagerungsmass B1: die Kontextobjekte des B1-Satzes in ihrer nativen
      Zellauflösung und eine Lupe auf die Stufe-13-Zelle des Säntis.
  (b) Adjazenzmass D1: die Ringe d = 0..D um die Repräsentantenzelle des Säntis auf
      Ankerstufe 10, eingefärbt nach Rang, und die Repräsentantenzellen der
      Kontextobjekte des D1-Satzes.

Beschriftung englisch (Toponyme und Objektarten im Original). Jedes Feld hat eine eigene Legende. Jedes in einem Feld verwendete Symbol steht in
dessen Legende, und die Legende enthält nichts, was im Feld nicht vorkommt.

Projektion: lokale abstandstreue Zylinderprojektion um die Säntis-Zelle (km). Bei
einem Ausschnitt von wenigen Kilometern ist die Verzerrung vernachlässigbar.

Ausgabe: output/abb_4_3_assoziationsmasse.{pdf,png,svg}
Aufruf:  python3 abb_3_02_abbildung.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import h3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, RegularPolygon
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

HIER = Path(__file__).resolve().parent
AUSGABE = HIER / "output"
J = json.load(open(AUSGABE / "abb_3_daten" / "abb_4_3_daten.json", encoding="utf-8"))

# ------------------------------------------------------------------ Stil
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 7.5, "legend.fontsize": 6.8,
    "axes.edgecolor": "#4a4a4a", "axes.linewidth": 0.6,
    "figure.dpi": 300, "savefig.dpi": 300, "hatch.linewidth": 0.4,
})
TXT = "#1f1f1f"
# kategoriale Farben: die ersten drei Slots der validierten Referenzpalette
# (alle Paare CVD-tauglich); alles Weitere über Linienstil/Markerform + Beschriftung
C_ALPSTEIN = "#2a78d6"
C_FLIS = "#eb6834"
C_NORDWAND = "#1baf7a"
C_GRAU = "#3a3a3a"
C_LINIE = "#6e6e6e"          # Umrisse der Landschaftsnamen: zurückhaltender als Text
C_SCHRAFFUR = "#b4b4b4"
C_ALPSTEIN_HELL = "#d3e3f7"   # Flächenton von C_ALPSTEIN (Stufe 9 als Hintergrund)
AUSSCHNITT = 1.55          # halbe Kartenbreite in km

# ------------------------------------------------------------------ Geometrie
Q10 = J["quelle"]["zelle10"]
LAT0, LNG0 = h3.cell_to_latlng(Q10)
KX = 111.32 * math.cos(math.radians(LAT0))
KY = 110.57


def xy(lat, lng):
    return ((lng - LNG0) * KX, (lat - LAT0) * KY)


def hexa(c):
    return [xy(a, b) for a, b in h3.cell_to_boundary(c)]


def mitte(c):
    return xy(*h3.cell_to_latlng(c))


OBJ = J["objekte"]


def finde(name, kat):
    for o in OBJ.values():
        if o["name"] == name and o["kategorie"] == kat:
            return o
    raise KeyError(name)


def zellen(ax, cells, **kw):
    ax.add_collection(PolyCollection([hexa(c) for c in cells], **kw))


def karte(ax):
    ax.set_xlim(-AUSSCHNITT, AUSSCHNITT)
    ax.set_ylim(-AUSSCHNITT, AUSSCHNITT)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    massstab(ax, -AUSSCHNITT + 0.12, -AUSSCHNITT + 0.12, 0.5, "500 m", AUSSCHNITT)


def massstab(ax, x0, y0, laenge, text, halbbreite, fs=6.5, dicke=0.0225):
    """Massstabsbalken; Dicke und Abstand relativ zur Kartenbreite."""
    h = dicke * halbbreite
    ax.add_patch(Rectangle((x0, y0), laenge, h, fc=TXT, ec="white", lw=0.4, zorder=20))
    ax.text(x0 + laenge / 2, y0 + 2.2 * h, text, ha="center", va="bottom", fontsize=fs,
            color=TXT, zorder=20, bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.85))


alpstein = finde("Alpstein", "Massiv")
flis = finde("Flis", "Gebiet")
nordwand = finde("Säntis-Nordwand", "Gebiet")
toggenburg = finde("Toggenburg", "Landschaftsname")
obertoggenburg = finde("Obertoggenburg", "Landschaftsname")
gegen = [o for o in OBJ.values() if o["gegenbeispiel"]][0]
q13 = J["quelle"]["zelle13"]
QX, QY = mitte(q13)

fig = plt.figure(figsize=(6.5, 5.3))
axa = fig.add_axes([0.01, 0.385, 0.485, 0.565])
axb = fig.add_axes([0.505, 0.385, 0.485, 0.565])

# ================================================================== Feld (a) B1
def zeichne_b1(ax, lupe=False):
    lw = 0.9 if lupe else 0.35
    zellen(ax, alpstein["zellen"], facecolors=C_ALPSTEIN_HELL, edgecolors="white",
           linewidths=lw * 1.6, zorder=1)
    zellen(ax, nordwand["zellen"], facecolors=C_NORDWAND, edgecolors="white",
           linewidths=lw * 0.6, zorder=2)
    zellen(ax, flis["zellen"], facecolors="none", edgecolors=C_FLIS,
           linewidths=1.2 if lupe else 0.4, zorder=3)
    if not lupe:
        zellen(ax, obertoggenburg["zellen"], facecolors="none", edgecolors=C_LINIE,
               linewidths=0.6, linestyles="-", zorder=4)
        zellen(ax, toggenburg["zellen"], facecolors="none", edgecolors=C_LINIE,
               linewidths=0.9, linestyles=(0, (4, 2)), zorder=4)


zeichne_b1(axa)
axa.plot(QX, QY, marker="*", ms=9, color=TXT, mec="white", mew=0.5, ls="", zorder=10)


def schild(ax, x, y, t, **kw):
    ax.text(x, y, t, fontsize=6.6, color=TXT, zorder=12,
            bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.85), **kw)


schild(axa, QX + 0.08, QY - 0.12, "Säntis", ha="left", va="top")
karte(axa)

# Lupe auf die Säntis-Zelle
ins = inset_axes(axa, width="34%", height="34%", loc="upper right", borderpad=0.5)
zeichne_b1(ins, lupe=True)
ins.plot(QX, QY, marker="*", ms=11, color=TXT, mec="white", mew=0.6, ls="", zorder=10)
L = 0.075
ins.set_xlim(QX - L, QX + L)
ins.set_ylim(QY - L, QY + L)
ins.set_aspect("equal")
massstab(ins, QX - L + 0.012, QY - L + 0.012, 0.05, "50 m", L, fs=6.0, dicke=0.045)
ins.set_xticks([])
ins.set_yticks([])
for sp in ins.spines.values():
    sp.set_edgecolor(TXT)
    sp.set_linewidth(0.7)
axa.add_patch(Rectangle((QX - L, QY - L), 2 * L, 2 * L, fc="none", ec=TXT, lw=0.8, zorder=15))

axa.set_title("(a) Overlap measure B$_1$", loc="left", fontsize=8.2, color=TXT, pad=4)

hex_a = dict(marker="h", ls="", ms=7.5)
leg_a = [
    Line2D([], [], marker="*", ls="", ms=8, color=TXT, mec="white", mew=0.5,
           label="Säntis (Alpiner Gipfel), res. 13"),
    Patch(fc=C_ALPSTEIN_HELL, ec="white", label="Alpstein (Massiv), res. 9"),
    Patch(fc="none", ec=C_FLIS, lw=0.8, label="Flis (Gebiet), res. 10"),
    Patch(fc=C_NORDWAND, ec="white", label="Säntis-Nordwand (Gebiet), res. 11"),
    Line2D([], [], color=C_LINIE, lw=0.6, label="Obertoggenburg (Landschaftsname), res. 8"),
    Line2D([], [], color=C_LINIE, lw=0.9, ls=(0, (4, 2)),
           label="Toggenburg (Landschaftsname), res. 7"),
]
fig.legend(handles=leg_a, loc="upper left", bbox_to_anchor=(0.01, 0.37), ncol=1,
           frameon=False, handlelength=1.8, borderaxespad=0, labelspacing=0.45)

# ================================================================== Feld (b) D1
ringe = {r["d"]: r for r in J["ringe"]}
K = J["parameter"]["k"]
grau = plt.get_cmap("Greys")


def rangfarbe(rang):              # Rang 1 dunkel -> Rang k hell
    return grau(0.50 - 0.43 * (rang - 1) / (K - 1))


ohne_neue, jenseits_k, rang_zellen = [], [], {}
for c, d in J["ringzellen"].items():
    r = ringe[d]
    if r["in_k"]:
        rang_zellen.setdefault(r["rang"], []).append(c)
    elif r["besetzt"]:          # neue Objekte, aber Rang > k
        jenseits_k.append(c)
    else:                       # alle Objekte dieses Rings liegen näher schon vor
        ohne_neue.append(c)
for rang, cs in rang_zellen.items():
    zellen(axb, cs, facecolors=rangfarbe(rang), edgecolors="white", linewidths=0.25, zorder=1)
zellen(axb, ohne_neue, facecolors="white", edgecolors=C_SCHRAFFUR, linewidths=0.25,
       hatch="//////", zorder=1)
zellen(axb, jenseits_k, facecolors="white", edgecolors=C_SCHRAFFUR, linewidths=0.25,
       hatch="......", zorder=1)

# Repräsentantenzellen; gleiche Zelle mehrerer Objekte -> Marker leicht versetzt
V = 0.024
d1_objekte = [
    (alpstein, dict(marker="o", ms=2.6, mfc=C_ALPSTEIN, mec=C_ALPSTEIN, mew=0), (0, 0)),
    (flis, dict(marker="s", ms=2.2, mfc=C_FLIS, mec=C_FLIS, mew=0), (-V, -V * 0.6)),
    (nordwand, dict(marker="^", ms=2.6, mfc=C_NORDWAND, mec=C_NORDWAND, mew=0), (V, -V * 0.6)),
]
gross = [o for o in OBJ.values() if o["kategorie"] == "Grossregion" and o["im_satz_D1"]]
for o, stil, (dx, dy) in d1_objekte:
    pts = [mitte(c) for c in o["repraesentanten"]]
    axb.plot([p[0] + dx for p in pts], [p[1] + dy for p in pts], ls="", zorder=5, **stil)
# Grossregionen: je drei Sprachfassungen mit identischer Geometrie -> ein Symbol je Region
gr_stil = {"Mittelland": dict(marker="D", mfc="white"), "Glarner Alpen": dict(marker="D", mfc=C_GRAU)}
gr_reps = {}
for o in gross:
    for key in gr_stil:
        if o["name"] == key:
            gr_reps[key] = o["repraesentanten"]
for key, reps in gr_reps.items():
    pts = [mitte(c) for c in reps]
    axb.plot([p[0] for p in pts], [p[1] + V * 1.2 for p in pts], ls="", ms=3.6, mec=C_GRAU,
             mew=0.7, zorder=6, **gr_stil[key])
axb.plot(0, 0, marker="*", ms=9, color=TXT, mec="white", mew=0.5, ls="", zorder=10)
schild(axb, 0.08, -0.12, "Säntis", ha="left", va="top")
karte(axb)
axb.set_title("(b) Adjacency measure D$_1$", loc="left", fontsize=8.2, color=TXT, pad=4)


def rang_text(o):
    return f"rank {o['rang']}" if o["rang"] is not None else "no rank"


leg_b = [
    Line2D([], [], marker="*", ls="", ms=8, color=TXT, mec="white", mew=0.5,
           label="Säntis, source cell (res. 10)"),
    Line2D([], [], marker="o", ls="", ms=3.2, mfc=C_ALPSTEIN, mec=C_ALPSTEIN, mew=0,
           label=f"Alpstein (Massiv), res. 9; {rang_text(alpstein)}"),
    Line2D([], [], marker="s", ls="", ms=2.8, mfc=C_FLIS, mec=C_FLIS, mew=0,
           label=f"Flis (Gebiet), res. 10; {rang_text(flis)}"),
    Line2D([], [], marker="^", ls="", ms=3.2, mfc=C_NORDWAND, mec=C_NORDWAND, mew=0,
           label=f"Säntis-Nordwand (Gebiet), res. 11; {rang_text(nordwand)}"),
]
for key in ("Glarner Alpen", "Mittelland"):
    o = finde(key, "Grossregion")
    leg_b.append(Line2D([], [], ls="", ms=3.8, mec=C_GRAU, mew=0.7,
                        label=f"{key} (Grossregion), res. {o['stufe']}; {rang_text(o)}",
                        **gr_stil[key]))
fig.legend(handles=leg_b, loc="upper left", bbox_to_anchor=(0.505, 0.37), ncol=1,
           frameon=False, handlelength=1.8, borderaxespad=0, labelspacing=0.45)

# Rangskala als eigene kleine Legende unter Feld (b)
sk = fig.add_axes([0.512, 0.085, 0.44, 0.125])
for r in range(1, K + 1):
    sk.add_patch(Rectangle((r - 1, 0), 1, 1, fc=rangfarbe(r), ec="white", lw=0.6))
    sk.text(r - 0.5, -0.25, str(r), ha="center", va="top", fontsize=6.3, color=TXT)
sk.add_patch(Rectangle((0, -2.7), 1, 1, fc="white", ec=C_SCHRAFFUR, lw=0.4, hatch="//////"))
sk.text(1.3, -2.2, "no new object (no rank)", ha="left", va="center", fontsize=6.3, color=TXT)
sk.add_patch(Rectangle((0, -4.2), 1, 1, fc="white", ec=C_SCHRAFFUR, lw=0.4, hatch="......"))
sk.text(1.3, -3.7, f"rank > k = {K}, not counted", ha="left", va="center", fontsize=6.3, color=TXT)
sk.text(0, 1.35, "Rank of the ring (D$_1$ search from Säntis)", ha="left", va="bottom", fontsize=6.8, color=TXT)
sk.set_xlim(-0.1, 25)
sk.set_ylim(-4.5, 2.6)
sk.axis("off")

for ext in ("pdf", "png", "svg"):
    fig.savefig(AUSGABE / f"abb_4_3_assoziationsmasse.{ext}", facecolor="white",
                bbox_inches="tight", pad_inches=0.04)
print("->", AUSGABE / "abb_4_3_assoziationsmasse.{pdf,png,svg}")
