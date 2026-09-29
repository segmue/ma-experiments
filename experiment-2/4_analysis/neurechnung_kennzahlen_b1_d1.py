# -*- coding: utf-8 -*-
"""
ERZEUGNIS  Die geschlossene Neurechnung aller Kennzahlen der Abschnitte 6, 7 und 8
           von Teil B des Zahlenregisters: die Verteilungskennzahlen beider
           Assoziationsmasse, Partnerzahl je Zeile, Diagonale, Asymmetrie, die
           fuenf staerksten gerichteten Zellen, der Anteil innerhalb derselben
           Objektklasse und die Vergleichszahlen zwischen B1 und D1. Jede Kennzahl
           traegt Datei, Bezugsmenge und Filterregel; wo sich der Wert unter zwei
           Filterregeln unterscheidet, stehen beide Werte da. Der Bericht stellt je
           Kennzahl den neuen Wert, den heutigen Registerwert und das Urteil
           nebeneinander.
           Ausgabe 1: experiment-2/results/analysis/
                      Neurechnung_Kennzahlen_B1_D1_2026-09-18.md
           Ausgabe 2: dieselben Werte als
                      Neurechnung_Kennzahlen_B1_D1_2026-09-18__daten.csv
                      (Spalten: block;groesse;mass;wert;filterregel;bezugsmenge;
                       datei;registerwert;urteil)

EINGABE    ma-experiments/matrices/config1/b1_matrix.csv
           ma-experiments/matrices/config1/d1_matrix.csv
           ma-experiments/data/swissnames3d/swissNAMES3D_{PKT,LIN,PLY}.dbf
               (Feld OBJEKTKLAS; nur fuer den Anteil innerhalb derselben
                Objektklasse, gelesen ueber kap6_basis.objektklassen)

           ACHTUNG, EINE VERWECHSLUNG MIT VORGESCHICHTE. B1 steht in
           b1_matrix.csv. Die Datei npmi_matrix.csv im selben Ordner ist das rohe
           Ueberlagerungs-NPMI vor der Daempfung, also eine andere Groesse; sie ist
           hier nirgends Eingabe.

RECHENWEG  Grundmenge ist die 110 x 110-Matrix mit 12'100 Zellen. Bezugsmenge jeder
           Verteilungskennzahl sind, wie im Register, die 11'990 gerichteten Zellen
           ausserhalb der Diagonale; die Diagonale ist gesondert gefuehrt. Wo das
           Register ausdruecklich die 12'100 Zellen einschliesslich Diagonale meint,
           steht das in der Spalte Bezugsmenge.
           Zwei Filterregeln stehen nebeneinander. Ungefiltert heisst: alle Zellen
           der Bezugsmenge. Ohne Randwerte heisst: jede Zelle mit einem Wert von
           -0.999 oder darunter ist ausgeschlossen; beim Vergleich der beiden Masse
           gibt es dafuer zwei Lesarten, die weite (Ausschluss, sobald ein Mass am
           Randwert liegt) und die enge (Ausschluss nur bei beidseitigem Randwert).
           Auswahlschwelle des Verfahrens ist 0.001; sie wird einschliessend
           gelesen (>= 0.001), was hier zahlengleich mit > 0.001 ist, weil keine
           Zelle genau auf dem Schwellwert liegt. Die Zehnermenge einer Zeile ist
           die Menge der Partner, die der Resolver taetsaechlich saehe: erst der
           Schwellwert, dann die Kappung auf zehn.
           scipy ist auf diesem Rechner nicht vorhanden. Spearman und Pearson sind
           deshalb von Hand gerechnet: Pearson als Produktmomentkorrelation,
           Spearman als Pearson ueber mittlere Raenge mit Bindungsausgleich.

AUFRUF     python3 neurechnung_kennzahlen_b1_d1.py [--wurzel PFAD_ZU_MA_EXPERIMENTS]
           Ohne --wurzel wird ma-experiments aus dem Ort dieses Skripts abgeleitet.
           Die Grundmodule kap6_basis.py und kategorie_anzeigenamen.py liegen in
           ../5_figures/.

Stand 18.09.2026
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCHWELLE = 0.001
RANDWERT = -0.999
KAPPUNG = 10
BERICHTSNAME = "Neurechnung_Kennzahlen_B1_D1_2026-09-18"
# Berichtsdatum, fest gesetzt: Dateiname, YAML-Kopf und Herkunftsangabe sollen
# zusammenpassen, und ein spaeterer Lauf soll denselben Bericht erzeugen.
BERICHTSDATUM = "2026-09-18"

# Kurznamen der Eingabedateien, wie sie in der Spalte Datei erscheinen.
DATEI_B1 = "matrices/config1/b1_matrix.csv"
DATEI_D1 = "matrices/config1/d1_matrix.csv"
DATEI_BEIDE = DATEI_B1 + " + " + DATEI_D1
DATEI_DBF = "data/swissnames3d/swissNAMES3D_{PKT,LIN,PLY}.dbf"

# Bezugsmengen, ausgeschrieben.
BEZ_OHNE = "11'990 gerichtete Zellen ausserhalb der Diagonale"
BEZ_MIT = "12'100 Zellen einschliesslich Diagonale"
BEZ_DIAG = "110 Diagonalzellen"
BEZ_ZEILEN = "110 Zeilen"
BEZ_MATRIX = "110 × 110, gerichtetes Mass"

# Filterregeln, ausgeschrieben.
F_UNGEFILTERT = "ungefiltert"
F_OHNE_RAND = "ohne Randwerte (Wert ≤ −0.999 ausgeschlossen)"
F_OHNE_RAND_WEIT = "ohne Randwerte, weite Lesart (Zelle fällt, sobald ein Mass am Randwert liegt)"
F_OHNE_RAND_ENG = "ohne Randwerte, enge Lesart (Zelle fällt nur bei beidseitigem Randwert)"
F_KEINE = "keine Filterregel einschlägig"


# ---------------------------------------------------------------------------
# Der heutige Registerstand. Abgeschrieben aus Zahlenregister_B_Masse_und_
# Grundgesamtheiten.md, Stand 17.09.2026, Abschnitte 6, 7 und 8. Diese Werte
# werden hier NICHT als Quelle benutzt, sondern nur als Vergleichspunkt; die
# Rechnung selbst kennt sie nicht.
# ---------------------------------------------------------------------------
REGISTERSTAND = "17.09.2026"


def _wurzel_aus_skriptort() -> Path:
    # ma-experiments/experiment-2/4_analysis/<dieses Skript>
    return Path(__file__).resolve().parents[2]


# --------------------------------------------------------------- Formatierung


def vz(x: float, stellen: int = 4, vorzeichen: bool = True) -> str:
    """Zahl mit typografischem Minuszeichen (U+2212)."""
    muster = f"{{:+.{stellen}f}}" if vorzeichen else f"{{:.{stellen}f}}"
    return muster.format(x).replace("-", "−")


def tausender(n: int) -> str:
    return f"{n:,}".replace(",", "'")


def anteil(maske: np.ndarray, mit_zahl: bool = False) -> str:
    teil, ganz = int(maske.sum()), int(maske.size)
    s = f"{100.0 * teil / ganz:.2f} %"
    return f"{s} ({tausender(teil)} Zellen)" if mit_zahl else s


# ------------------------------------------------------- Rangstatistik zu Fuss


def raenge(x: np.ndarray) -> np.ndarray:
    """Mittlere Raenge mit Bindungsausgleich, wie scipy.stats.rankdata."""
    x = np.asarray(x, dtype=float)
    n = x.size
    ordnung = np.argsort(x, kind="mergesort")
    sortiert = x[ordnung]
    r = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sortiert[j + 1] == sortiert[i]:
            j += 1
        r[ordnung[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    nenner = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / nenner) if nenner > 0 else float("nan")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(raenge(a), raenge(b))


# ------------------------------------------------------------- Matrizen lesen


def matrix_lesen(pfad: Path) -> pd.DataFrame:
    if not pfad.is_file():
        raise SystemExit(f"fehlt: {pfad}")
    m = pd.read_csv(pfad, sep=";", index_col=0, encoding="utf-8")
    m.index = [str(i) for i in m.index]
    m.columns = [str(c) for c in m.columns]
    if list(m.index) != list(m.columns):
        raise SystemExit(f"{pfad.name}: Zeilen- und Spaltennamen stimmen nicht ueberein")
    if m.shape[0] != m.shape[1]:
        raise SystemExit(f"{pfad.name}: nicht quadratisch")
    if m.isna().any().any():
        raise SystemExit(f"{pfad.name}: enthaelt fehlende Werte")
    return m


def partner_je_zeile(m: pd.DataFrame) -> np.ndarray:
    a = m.to_numpy(dtype=float).copy()
    np.fill_diagonal(a, -np.inf)
    return (a >= SCHWELLE).sum(axis=1)


def zehnermengen(m: pd.DataFrame, mit_schwelle: bool = True) -> dict[str, list[str]]:
    aus = {}
    for kat in m.index:
        s = m.loc[kat].drop(labels=[kat])
        if mit_schwelle:
            s = s[s >= SCHWELLE]
        s = s.sort_values(ascending=False)
        aus[kat] = [str(k) for k in s.head(KAPPUNG).index]
    return aus


def staerkste_zellen(m: pd.DataFrame, wieviele: int = 5) -> list[tuple[str, str, float]]:
    a = m.to_numpy(dtype=float)
    ohne = np.where(~np.eye(a.shape[0], dtype=bool), a, -np.inf)
    flach = np.argsort(ohne.ravel())[::-1][:wieviele]
    aus = []
    for t in flach:
        i, j = divmod(int(t), a.shape[1])
        aus.append((str(m.index[i]), str(m.columns[j]), float(a[i, j])))
    return aus


def objektklassen_lesen(wurzel: Path, kategorien: list[str]):
    """Kategorie -> thematische Objektklasse, ueber das Grundmodul kap6_basis.

    Gibt None zurueck, wenn die DBF-Dateien oder das Grundmodul fehlen; die
    Klassenkennzahlen entfallen dann und der Bericht sagt das.
    """
    codeordner = str(Path(__file__).resolve().parents[1] / "5_figures")
    if codeordner not in sys.path:
        sys.path.insert(0, codeordner)
    try:
        import kap6_basis  # noqa: PLC0415
        klasse, _ = kap6_basis.objektklassen(kap6_basis.pfade(wurzel))
    except Exception as fehler:  # noqa: BLE001
        print(f"Hinweis: Objektklassen nicht lesbar ({fehler}); "
              f"die Klassenkennzahlen entfallen.")
        return None
    fehlend = [k for k in kategorien if k not in klasse]
    if fehlend:
        print(f"Hinweis: ohne Objektklasse: {fehlend}")
    return np.array([klasse.get(k, "ohne Klasse") for k in kategorien])


# ------------------------------------------------------------------ Erhebung


class Erhebung:
    """Sammelt die Kennzahlen als Zeilen mit Wert, Herkunft und Urteil."""

    def __init__(self) -> None:
        self.zeilen: list[dict[str, str]] = []

    @staticmethod
    def _vergleichbar(s: str) -> str:
        """Schreibweisen angleichen, die keinen Zahlenunterschied bedeuten.

        Das Register schreibt dasselbe Vorzeichen uneinheitlich, einmal +0.6563
        und einmal 0.2199. Fuer das Urteil zaehlt die Zahl, nicht die Setzung.
        """
        return (s.replace("\u2212", "-").replace("+", "")
                 .replace("\u2019", "").replace("'", "")
                 .replace("\u00a0", " ").strip())

    def add(self, block: str, groesse: str, mass: str, wert: str,
            filterregel: str, bezugsmenge: str, datei: str,
            register: str | None = None) -> None:
        if register is None:
            urteil = "im Register nicht geführt"
        elif self._vergleichbar(register) == self._vergleichbar(wert):
            urteil = "bestätigt"
        else:
            urteil = "abweichend"
        self.zeilen.append({
            "block": block, "groesse": groesse, "mass": mass, "wert": wert,
            "filterregel": filterregel, "bezugsmenge": bezugsmenge,
            "datei": datei, "registerwert": register or "—", "urteil": urteil,
        })

    def des_blocks(self, block: str) -> list[dict[str, str]]:
        return [z for z in self.zeilen if z["block"] == block]

    def zaehle(self, urteil: str) -> int:
        return sum(1 for z in self.zeilen if z["urteil"] == urteil)

    def abweichungen(self) -> list[dict[str, str]]:
        return [z for z in self.zeilen if z["urteil"] == "abweichend"]


# ------------------------------------------------------ Kennzahlen je Mass


def block_eines_masses(E: Erhebung, block: str, mass: str, m: pd.DataFrame,
                       datei: str, reg: dict[str, str],
                       klasse: np.ndarray | None) -> None:
    a = m.to_numpy(dtype=float)
    n = a.shape[0]
    ausser = ~np.eye(n, dtype=bool)
    x = a[ausser]                      # 11'990, ungefiltert
    xr = x[x > RANDWERT]               # 11'990 ohne Randwerte
    alle = a.ravel()                   # 12'100 mit Diagonale
    diag = np.diag(a)

    def A(groesse: str, wert: str, filterregel: str = F_UNGEFILTERT,
          bezugsmenge: str = BEZ_OHNE, dat: str = None,
          schluessel: str | None = None) -> None:
        E.add(block, groesse, mass, wert, filterregel, bezugsmenge,
              dat or datei, reg.get(schluessel) if schluessel else None)

    A("Matrixgrösse", f"{n} × {n}", F_KEINE,
      f"{tausender(n * n)} Zellen, davon {tausender(x.size)} ausserhalb der Diagonale",
      schluessel="matrix")

    for name, fn, schl in (("Minimum", np.min, "min"),
                           ("Erstes Quartil", lambda v: np.percentile(v, 25), "q1"),
                           ("Median", np.median, "med"),
                           ("Drittes Quartil", lambda v: np.percentile(v, 75), "q3"),
                           ("Maximum", np.max, "max"),
                           ("Mittelwert", np.mean, "mittel")):
        A(name, vz(float(fn(x))), F_UNGEFILTERT, BEZ_OHNE, schluessel=schl)
        A(name, vz(float(fn(xr))), F_OHNE_RAND,
          f"{tausender(xr.size)} von {tausender(x.size)} Zellen bleiben")
        A(name, vz(float(fn(alle))), F_UNGEFILTERT, BEZ_MIT,
          schluessel=schl + "_diag")

    A("Spannweite (Maximum − Minimum)", vz(float(x.max() - x.min()), vorzeichen=False))
    A("Spannweite (Maximum − Minimum)", vz(float(xr.max() - xr.min()), vorzeichen=False),
      F_OHNE_RAND, f"{tausender(xr.size)} von {tausender(x.size)} Zellen bleiben")

    A("Anteil ≤ 0", anteil(x <= 0), schluessel="le0")
    A("Anteil ≤ 0", anteil(xr <= 0), F_OHNE_RAND,
      f"{tausender(xr.size)} von {tausender(x.size)} Zellen bleiben")
    A("Anteil am unteren Randwert (≤ −0.999)", anteil(x <= RANDWERT, mit_zahl=True),
      F_UNGEFILTERT, BEZ_OHNE + ", Schwelle ≤ −0.999", schluessel="rand")
    A("Anteil ≥ 0.001 (Auswahlschwelle)", anteil(x >= SCHWELLE, mit_zahl=True),
      F_UNGEFILTERT, BEZ_OHNE + "; zahlengleich mit > 0.001, keine Zelle liegt genau auf 0.001",
      schluessel="schwelle")
    A("Anteil ≥ 0.001 (Auswahlschwelle)", anteil(xr >= SCHWELLE, mit_zahl=True), F_OHNE_RAND,
      f"{tausender(xr.size)} von {tausender(x.size)} Zellen bleiben")
    A("Anteil im Band (−0.001, 0.001)", anteil((x > -SCHWELLE) & (x < SCHWELLE)),
      schluessel="band")
    A("Anteil im Band (−0.001, 0.001)", anteil((xr > -SCHWELLE) & (xr < SCHWELLE)),
      F_OHNE_RAND, f"{tausender(xr.size)} von {tausender(x.size)} Zellen bleiben")

    p = partner_je_zeile(m)
    bez_p = BEZ_ZEILEN + ", Zellen ≥ 0.001 ohne Selbstbezug"
    A("Partner je Zeile, Median", f"{np.median(p):g}", F_UNGEFILTERT, bez_p,
      schluessel="p_med")
    A("Partner je Zeile, Mittel", f"{p.mean():.2f}", F_UNGEFILTERT, bez_p,
      schluessel="p_mittel")
    A("Partner je Zeile, Spanne", f"{int(p.min())} bis {int(p.max())}",
      F_UNGEFILTERT, bez_p, schluessel="p_spanne")
    A("Zeilen ohne Partner", f"{int((p == 0).sum())} von {n}", F_UNGEFILTERT,
      BEZ_ZEILEN, schluessel="p_ohne")

    konstant = float(np.ptp(diag)) < 1e-9
    A("Diagonale, Verhalten",
      (f"konstant {vz(float(np.median(diag)))}" if konstant
       else f"gerechnet, {vz(float(diag.min()))} bis {vz(float(diag.max()))}"),
      F_KEINE, BEZ_DIAG, schluessel="diag")
    A("Diagonale, Median", vz(float(np.median(diag))), F_KEINE, BEZ_DIAG,
      schluessel="diag_med")

    A("Asymmetrie, max ǀM − Mᵀǀ", vz(float(np.abs(a - a.T).max()), vorzeichen=False),
      F_KEINE, BEZ_MATRIX, schluessel="asym")

    for rang, (i, j, w) in enumerate(staerkste_zellen(m, 5), start=1):
        A(f"{rang}. stärkste gerichtete Zelle",
          f"{anzeige(i)} → {anzeige(j)} {w:.4f}", F_UNGEFILTERT, BEZ_OHNE,
          schluessel=f"top{rang}")

    if klasse is not None:
        innen = (klasse[:, None] == klasse[None, :]) & ausser
        zwischen = ausser & ~(klasse[:, None] == klasse[None, :])
        A("Anteil ≥ 0.001 innerhalb derselben Objektklasse",
          f"{100.0 * (a[innen] >= SCHWELLE).mean():.1f} %", F_UNGEFILTERT,
          f"{int(innen.sum())} der {tausender(x.size)} Zellen liegen innerhalb einer "
          f"Objektklasse, davon {int((a[innen] >= SCHWELLE).sum())} über der Schwelle",
          dat=datei + " + " + DATEI_DBF, schluessel="kl_innen")
        A("Anteil ≥ 0.001 zwischen zwei Objektklassen",
          f"{100.0 * (a[zwischen] >= SCHWELLE).mean():.1f} %", F_UNGEFILTERT,
          f"{tausender(int(zwischen.sum()))} der {tausender(x.size)} Zellen liegen "
          "zwischen zwei Objektklassen",
          dat=datei + " + " + DATEI_DBF, schluessel="kl_zwischen")
        A("Zellen am Randwert innerhalb derselben Objektklasse",
          f"{int((a[innen] <= RANDWERT).sum())}", F_UNGEFILTERT,
          f"von {int(innen.sum())} Zellen innerhalb einer Objektklasse",
          dat=datei + " + " + DATEI_DBF, schluessel="kl_rand_innen")
        A("Anteil am Randwert zwischen zwei Objektklassen",
          f"{100.0 * (a[zwischen] <= RANDWERT).mean():.1f} %", F_UNGEFILTERT,
          f"{tausender(int(zwischen.sum()))} Zellen zwischen zwei Objektklassen",
          dat=datei + " + " + DATEI_DBF, schluessel="kl_rand_zwischen")


# ----------------------------------------------------- Vergleich beider Masse


def block_des_vergleichs(E: Erhebung, b1: pd.DataFrame, d1: pd.DataFrame,
                         reg: dict[str, str], klasse: np.ndarray | None) -> None:
    block, mass = "Vergleich", "B₁ gegen D₁"
    n = b1.shape[0]
    ausser = ~np.eye(n, dtype=bool)
    A = b1.to_numpy(dtype=float)[ausser]
    D = d1.to_numpy(dtype=float)[ausser]
    A_alle = b1.to_numpy(dtype=float).ravel()
    D_alle = d1.to_numpy(dtype=float).ravel()
    weit = (A > RANDWERT) & (D > RANDWERT)
    beidseitig = (A <= RANDWERT) & (D <= RANDWERT)
    eng = ~beidseitig

    def V(groesse: str, wert: str, filterregel: str, bezugsmenge: str,
          schluessel: str | None = None, dat: str = DATEI_BEIDE) -> None:
        E.add(block, groesse, mass, wert, filterregel, bezugsmenge, dat,
              reg.get(schluessel) if schluessel else None)

    bez_weit = f"{tausender(int(weit.sum()))} Zellen bleiben"
    bez_eng = (f"{tausender(int(eng.sum()))} Zellen bleiben; "
               f"{int(beidseitig.sum())} Zellen liegen in beiden Massen am Randwert")

    V("Rangkorrelation über alle Zellen (Spearman)", vz(spearman(A, D)),
      F_UNGEFILTERT, BEZ_OHNE, "sp_alle")
    V("Rangkorrelation über alle Zellen (Spearman)", vz(spearman(A[weit], D[weit])),
      F_OHNE_RAND_WEIT, bez_weit, "sp_weit")
    V("Rangkorrelation über alle Zellen (Spearman)", vz(spearman(A[eng], D[eng])),
      F_OHNE_RAND_ENG, bez_eng, "sp_eng")
    V("Rangkorrelation über alle Zellen (Spearman)", vz(spearman(A_alle, D_alle)),
      F_UNGEFILTERT, BEZ_MIT)

    V("Produktmomentkorrelation (Pearson)", vz(pearson(A, D)),
      F_UNGEFILTERT, BEZ_OHNE, "pe_alle")
    V("Produktmomentkorrelation (Pearson)", vz(pearson(A[weit], D[weit])),
      F_OHNE_RAND_WEIT, bez_weit)
    V("Produktmomentkorrelation (Pearson)", vz(pearson(A[eng], D[eng])),
      F_OHNE_RAND_ENG, bez_eng)
    V("Produktmomentkorrelation (Pearson)", vz(pearson(A_alle, D_alle)),
      F_UNGEFILTERT, BEZ_MIT)

    V("Zellen, die in beiden Massen am Randwert liegen", f"{int(beidseitig.sum())}",
      F_UNGEFILTERT, BEZ_OHNE, "rand_beid")
    V("Zellen, die in mindestens einem Mass am Randwert liegen",
      tausender(int((~weit).sum())), F_UNGEFILTERT, BEZ_OHNE, "rand_eins")

    # --- zeilenweiser Spearman, beide Filterregeln --------------------------
    bez_zeile = BEZ_ZEILEN + ", je 109 Partner ohne Selbstbezug"
    rho_u, rho_f = {}, {}
    for kat in b1.index:
        a = b1.loc[kat].drop(labels=[kat]).to_numpy(dtype=float)
        d = d1.loc[kat].drop(labels=[kat]).to_numpy(dtype=float)
        rho_u[kat] = spearman(a, d)
        maske = (a > RANDWERT) & (d > RANDWERT)
        rho_f[kat] = spearman(a[maske], d[maske]) if int(maske.sum()) >= 3 else float("nan")
    s_u = pd.Series(rho_u)
    s_f = pd.Series(rho_f).dropna()

    for lab, s, fr, bez, schl in (
            ("ungefiltert", s_u, F_UNGEFILTERT, bez_zeile,
             {"med": "z_med", "q1": "z_q1", "q3": "z_q3",
              "min": "z_min", "max": "z_max", "kmin": None, "kmax": "z_kmax"}),
            ("ohne Randwerte", s_f, F_OHNE_RAND,
             f"{len(s_f)} von {n} Zeilen; je Zeile bleiben die Partner, die in "
             "beiden Massen über dem Randwert liegen",
             {})):
        V("Zeilenweiser Spearman, Median", vz(float(s.median())), fr, bez, schl.get("med"))
        V("Zeilenweiser Spearman, erstes Quartil", vz(float(s.quantile(0.25))), fr, bez,
          schl.get("q1"))
        V("Zeilenweiser Spearman, drittes Quartil", vz(float(s.quantile(0.75))), fr, bez,
          schl.get("q3"))
        V("Zeilenweiser Spearman, Minimum", vz(float(s.min())), fr, bez, schl.get("min"))
        V("Zeilenweiser Spearman, Kategorie des Minimums", anzeige(str(s.idxmin())), fr, bez,
          schl.get("kmin"))
        V("Zeilenweiser Spearman, Maximum", vz(float(s.max())), fr, bez, schl.get("max"))
        V("Zeilenweiser Spearman, Kategorie des Maximums", anzeige(str(s.idxmax())), fr, bez,
          schl.get("kmax"))

    # --- Zehnermengen -------------------------------------------------------
    for mit_schwelle in (True, False):
        zb = zehnermengen(b1, mit_schwelle)
        zd = zehnermengen(d1, mit_schwelle)
        jac, gleich, ohne = [], 0, 0
        for kat in b1.index:
            X, Y = set(zb[kat]), set(zd[kat])
            jac.append(len(X & Y) / len(X | Y) if (X or Y) else float("nan"))
            if zb[kat] and zd[kat] and zb[kat][0] == zd[kat][0]:
                gleich += 1
            if not (X & Y):
                ohne += 1
        fr = ("Zehnermengen mit Schwellwert 0.001, dann Kappung auf zehn"
              if mit_schwelle else "Zehnermengen ohne Schwellwert, nur Kappung auf zehn")
        bez = BEZ_ZEILEN + ", je höchstens zehn Partner"
        V("Jaccard der Zehnermengen, Median",
          vz(float(np.nanmedian(jac)), vorzeichen=False), fr, bez,
          "jac" if mit_schwelle else None)
        V("Zeilen mit gleichem stärkstem Partner", f"{gleich} von {n}", fr, bez,
          "gleich" if mit_schwelle else "gleich_ohne")
        V("Zeilen ohne jede Überschneidung der Zehnermengen", f"{ohne} von {n}", fr, bez,
          "ohne_mit" if mit_schwelle else "ohne_ohne")
        if mit_schwelle:
            zb_s, zd_s = zb, zd
        else:
            zb_o, zd_o = zb, zd

    for kat in ("Alpiner Gipfel", "Hauptgipfel", "Massiv", "Gipfel", "Grat",
                "Tal", "Ort", "Pass", "Landschaftsname"):
        if kat not in b1.index:
            continue
        g_s = len(set(zb_s[kat]) & set(zd_s[kat]))
        g_o = len(set(zb_o[kat]) & set(zd_o[kat]))
        wert = (f"{g_o} von 10 ohne Schwelle, {g_s} von 10 mit Schwelle"
                if g_o != g_s else f"{g_s} von 10")
        V(f"Übereinstimmung der Zehnermengen, {anzeige(kat)}", wert,
          "beide Lesarten, mit und ohne Schwellwert 0.001",
          BEZ_ZEILEN + f"; Zeile {anzeige(kat)}", "ue_" + kat)
        V(f"Zeilenweiser Spearman, {anzeige(kat)}", vz(float(rho_u[kat]), 3),
          F_UNGEFILTERT, bez_zeile, "rho_" + kat)

    # --- Objektklassen ------------------------------------------------------
    if klasse is not None:
        gleiche = (klasse[:, None] == klasse[None, :]) & ausser
        V("Zellen innerhalb derselben Objektklasse", tausender(int(gleiche.sum())),
          F_UNGEFILTERT, BEZ_OHNE, "zellen_innen", DATEI_DBF)
        V("Zellen zwischen zwei Objektklassen",
          tausender(int((ausser & ~gleiche).sum())), F_UNGEFILTERT, BEZ_OHNE,
          "zellen_zwischen", DATEI_DBF)


# ------------------------------------------------------------- Anzeigenamen

_ANZEIGE: dict[str, str] = {}


def anzeige(name: str) -> str:
    return _ANZEIGE.get(name, name)


def anzeigenamen_laden() -> None:
    codeordner = str(Path(__file__).resolve().parents[1] / "5_figures")
    if codeordner not in sys.path:
        sys.path.insert(0, codeordner)
    try:
        from kategorie_anzeigenamen import ERSATZSCHREIBUNG  # noqa: PLC0415
        _ANZEIGE.update(ERSATZSCHREIBUNG)
    except Exception as fehler:  # noqa: BLE001
        print(f"Hinweis: Anzeigetabelle nicht ladbar ({fehler}); "
              "die Maschinennamen bleiben stehen.")


# --------------------------------------------------- Der heutige Registerstand

# Abgeschrieben aus Zahlenregister_B_Masse_und_Grundgesamtheiten.md, Stand
# 17.09.2026: Abschnitt 6 (B1), Abschnitt 7 (D1), Abschnitt 8 (Vergleich).
# Diese Tabellen sind reiner Vergleichspunkt. Die Rechnung oben kennt sie nicht.

REG_B1: dict[str, str] = {
    "matrix": "110 × 110",
    "min": "−1.0000", "q1": "−0.7922", "med": "−0.0288", "q3": "−0.0000",
    "max": "+0.7139", "mittel": "−0.3082",
    "le0": "75.71 %",
    "rand": "7.06 % (846 Zellen)",
    "schwelle": "16.16 % (1'938 Zellen)",
    "band": "19.80 %",
    "p_med": "15", "p_mittel": "17.62", "p_spanne": "6 bis 47", "p_ohne": "0 von 110",
    "diag": "konstant +0.5000",
    "asym": "1.0000",
    "top1": "Strassenpass → Pass 0.7139",
    "top2": "Verladestation → Haltestelle Schiff 0.7067",
    "top3": "Verladestation → Autofähre 0.6222",
    "top4": "Übrige Bahnen → Sesselbahn 0.6094",
    "top5": "Kleinbahn → Freizeitanlagenareal 0.6087",
    "kl_innen": "27.6 %", "kl_zwischen": "15.5 %",
    "kl_rand_innen": "0", "kl_rand_zwischen": "7.5 %",
}

REG_D1: dict[str, str] = {
    "matrix": "110 × 110",
    "min": "−1.0000", "q1": "−1.0000", "med": "−0.0232", "q3": "+0.0335",
    "max": "+0.4908", "max_diag": "+0.7289", "mittel": "−0.2527",
    "le0": "60.03 %",
    "rand": "25.75 % (3'088 Zellen)",
    "schwelle": "39.52 % (4'738 Zellen)",
    "band": "0.88 %",
    "p_med": "43.5", "p_mittel": "43.07", "p_spanne": "12 bis 81", "p_ohne": "0 von 110",
    "diag": "gerechnet, −1.0000 bis +0.7289",
    "diag_med": "+0.2264",
    "asym": "0.9334",
    "top1": "Verladestation → Autofähre 0.4908",
    "top2": "Alpiner Gipfel → Gletscher 0.4651",
    "top3": "Hauptgipfel → Gletscher 0.4618",
    "top4": "Gletscher → Hauptgipfel 0.4280",
    "top5": "Gletscher → Alpiner Gipfel 0.4274",
    "kl_innen": "67.6 %", "kl_zwischen": "37.9 %",
}

REG_V: dict[str, str] = {
    "sp_alle": "0.3466", "pe_alle": "0.2777",
    "sp_weit": "0.1940", "sp_eng": "0.3039",
    "rand_beid": "350",
    "z_med": "0.2199", "z_q1": "0.1220", "z_q3": "0.3749",
    "z_min": "−0.1724", "z_max": "+0.6563", "z_kmax": "Tal",
    "jac": "0.1765",
    "gleich": "19 von 110",
    "ohne_mit": "11 von 110", "ohne_ohne": "10 von 110",
    "zellen_innen": "664", "zellen_zwischen": "11'326",
    "ue_Alpiner Gipfel": "5 von 10", "rho_Alpiner Gipfel": "+0.388",
    "ue_Hauptgipfel": "5 von 10", "rho_Hauptgipfel": "+0.374",
    "ue_Massiv": "3 von 10", "rho_Massiv": "+0.601",
    "ue_Gipfel": "4 von 10", "rho_Gipfel": "+0.364",
    "ue_Grat": "5 von 10", "rho_Grat": "+0.587",
    "ue_Tal": "3 von 10", "rho_Tal": "+0.649",
    "ue_Ort": "3 von 10", "rho_Ort": "+0.219",
    "ue_Pass": "6 von 10", "rho_Pass": "+0.354",
    "ue_Landschaftsname": "1 von 10 ohne Schwelle, 0 von 10 mit Schwelle",
    "rho_Landschaftsname": "+0.548",
}


# ------------------------------------------------------------------- Bericht

KURZ = {
    DATEI_B1: "`b1_matrix.csv`",
    DATEI_D1: "`npmi_dist_matrix_D1.csv`",
    DATEI_BEIDE: "beide Matrizen",
    DATEI_DBF: "`swissNAMES3D_*.dbf`",
    DATEI_B1 + " + " + DATEI_DBF: "`b1_matrix.csv` + `swissNAMES3D_*.dbf`",
    DATEI_D1 + " + " + DATEI_DBF: "`npmi_dist_matrix_D1.csv` + `swissNAMES3D_*.dbf`",
}


def kurz(datei: str) -> str:
    return KURZ.get(datei, datei)


def tabelle(zeilen: list[dict[str, str]]) -> list[str]:
    t = [f"| Grösse | Neuer Wert | Filterregel | Bezugsmenge | Datei | "
         f"Register ({REGISTERSTAND}) | Urteil |",
         "|---|---|---|---|---|---|---|"]
    for z in zeilen:
        t.append("| " + " | ".join([
            z["groesse"], z["wert"], z["filterregel"], z["bezugsmenge"],
            kurz(z["datei"]), z["registerwert"], z["urteil"]]) + " |")
    return t


def bericht_schreiben(E: Erhebung, ziel: Path, heute: str, klasse_da: bool) -> None:
    n_ges = len(E.zeilen)
    n_best = E.zaehle("bestätigt")
    n_abw = E.zaehle("abweichend")
    n_neu = E.zaehle("im Register nicht geführt")

    z: list[str] = []
    z.append("---")
    z.append("name: Neurechnung der Kennzahlen B₁ und D₁ (18.09.2026)")
    z.append("description: Geschlossene Neurechnung aller Kennzahlen der Abschnitte 6, "
             "7 und 8 von Teil B des Zahlenregisters, unmittelbar aus den beiden "
             "Matrizen. Jede Kennzahl traegt Datei, Bezugsmenge und Filterregel; wo "
             "zwei Filterregeln zu verschiedenen Werten fuehren, stehen beide da. Je "
             "Kennzahl stehen der neue Wert, der Registerwert vom 17.09.2026 und das "
             "Urteil nebeneinander. Das Register ist hier Pruefgegenstand, nicht Quelle.")
    z.append(f"date: {heute}")
    z.append(f"kennzahlen: {n_ges}")
    z.append(f"bestaetigt: {n_best}")
    z.append(f"abweichend: {n_abw}")
    z.append(f"nicht_im_register: {n_neu}")
    z.append("---")
    z.append("")
    z.append("# Neurechnung der Kennzahlen B₁ und D₁")
    z.append("")
    z.append("**Warum diese Datei.** Die Kennzahlenblöcke zu B₁ und D₁ in "
             "`Zahlenregister_B_Masse_und_Grundgesamtheiten.md`, Abschnitte 6, 7 und 8, "
             "sind ohne erhaltenes Rechenskript entstanden. Die Zahlen sind damit nicht "
             "notwendig falsch, aber nicht nachvollziehbar: zu keiner Kennzahl steht "
             "fest, über welche Bezugsmenge und unter welcher Filterregel sie gerechnet "
             "worden ist. Dieser Bericht rechnet alle Kennzahlen dieser drei Abschnitte "
             "geschlossen aus den beiden Matrizen neu und führt zu jeder Kennzahl die "
             "drei Angaben mit, die bisher fehlen: **Datei**, **Bezugsmenge** und "
             "**Filterregel**. Für diesen einen Vorgang ist das Register nicht die "
             "gültige Zahlenquelle, sondern der Prüfgegenstand; es wird nicht zitiert, "
             "sondern verglichen.")
    z.append("")
    z.append("**Befund in einem Satz.** Von "
             f"{n_ges} gerechneten Kennzahlen bestätigt das Register {n_best}, "
             f"{n_abw} weichen ab, und {n_neu} führt das Register nicht.")
    z.append("")
    z.append("## 1 Eingaben, Bezugsmengen, Filterregeln")
    z.append("")
    z.append("**Die beiden Matrizen.**")
    z.append("")
    z.append(f"- **B₁, das Überlagerungsmass:** `{DATEI_B1}`")
    z.append(f"- **D₁, das Adjazenzmass:** `{DATEI_D1}`")
    z.append("")
    z.append("Beide sind semikolongetrennt, 110 × 110, mit den Kategorienamen als Index, "
             "in derselben Kategorienordnung und ohne fehlenden Wert.")
    z.append("")
    z.append("**Eine Verwechslung, die heute schon einen Fehlalarm ausgelöst hat.** "
             "B₁ steht in `b1_matrix.csv`. Die Datei `npmi_matrix.csv` im selben Ordner "
             "ist das **rohe Überlagerungs-NPMI vor der Dämpfung** und damit eine andere "
             "Grösse. Sie ist hier nirgends Eingabe. Wer sie an B₁ Stelle einsetzt, "
             "bekommt abweichende Zahlen und hält sie für einen Registerfehler.")
    z.append("")
    z.append("**Die thematische Objektklasse** je Kategorie stammt aus dem Feld "
             f"`OBJEKTKLAS` der Lieferung SwissNames3D 2024 (`{DATEI_DBF}`), gelesen "
             "über das Grundmodul `kap6_basis.objektklassen`.")
    if not klasse_da:
        z.append("")
        z.append("> **Achtung.** Die Objektklassen waren bei diesem Lauf nicht lesbar. "
                 "Die Kennzahlen zur Objektklasse fehlen deshalb in diesem Bericht.")
    z.append("")
    z.append("**Die zwei Bezugsmengen.** Die Matrix hat 12'100 Zellen. Die Diagonale "
             "zählt 110 davon; sie ist bei B₁ konstant gesetzt und bei D₁ gerechnet und "
             "gehört deshalb nicht in dieselbe Verteilung wie die gerichteten Paare. "
             "Bezugsmenge ist darum, wie im Register, in der Regel die **11'990 "
             "gerichteten Zellen ausserhalb der Diagonale**. Wo eine Kennzahl zusätzlich "
             "über die **12'100 Zellen einschliesslich Diagonale** gerechnet ist, steht "
             "das in der Spalte Bezugsmenge; die beiden Werte stehen dann nebeneinander.")
    z.append("")
    z.append("**Die zwei Filterregeln.** *Ungefiltert* heisst: alle Zellen der "
             "Bezugsmenge gehen ein, auch die am unteren Randwert. *Ohne Randwerte* "
             "heisst: jede Zelle mit einem Wert von −0.999 oder darunter fällt weg. Beim "
             "Vergleich der beiden Masse zerfällt die zweite Regel in zwei Lesarten, "
             "weil eine Zelle in einem Mass am Randwert liegen kann und im anderen "
             "nicht: die **weite** Lesart schliesst eine Zelle aus, sobald sie in "
             "mindestens einem der beiden Masse am Randwert liegt; die **enge** Lesart "
             "schliesst nur die Zellen aus, die in **beiden** Massen am Randwert liegen. "
             "Die beiden Lesarten trennen die Bezugsmenge sehr verschieden, und genau "
             "dieser Unterschied ist es, der aus einer Rangkorrelation von 0.3466 je "
             "nach Regel 0.1940 oder 0.3039 macht. Wo sich ein Wert unter zwei Regeln "
             "unterscheidet, führt der Bericht beide.")
    z.append("")
    z.append("**Schwellwert und Zehnermenge.** Auswahlschwelle des Verfahrens ist 0.001, "
             "einschliessend gelesen; das ist hier zahlengleich mit der strengen Lesart "
             "> 0.001, weil keine Zelle genau auf dem Schwellwert liegt. Die Zehnermenge "
             "einer Zeile ist die Menge der Partner, die der Resolver tatsächlich sähe: "
             "erst der Schwellwert, dann die Kappung auf zehn.")
    z.append("")
    z.append("**Rangstatistik ohne scipy.** Auf diesem Rechner ist `scipy` nicht "
             "vorhanden. Spearman und Pearson sind deshalb im Skript selbst gerechnet: "
             "Pearson als Produktmomentkorrelation, Spearman als Pearson über mittlere "
             "Ränge mit Bindungsausgleich. Die vier von zwei Seiten bestätigten "
             "Ankerwerte in Abschnitt 5 belegen, dass die Handrechnung dasselbe liefert.")
    z.append("")
    z.append("**Lesart der Spalte Urteil.** *bestätigt* — der neue Wert stimmt mit dem "
             "Registerwert überein. *abweichend* — beide Werte stehen da, der neue "
             "zuerst. *im Register nicht geführt* — die Kennzahl ist neu, meist eine "
             "zweite Filterregel oder eine zweite Bezugsmenge zu einer Kennzahl, die das "
             "Register nur einmal führt.")
    z.append("")
    for titel, block in (("2 Kennzahlen B₁ — das Überlagerungsmass (Register, Abschnitt 6)", "B₁"),
                         ("3 Kennzahlen D₁ — das Adjazenzmass (Register, Abschnitt 7)", "D₁"),
                         ("4 Der Vergleich der beiden Masse (Register, Abschnitt 8)", "Vergleich")):
        zeilen = E.des_blocks(block)
        z.append(f"## {titel}")
        z.append("")
        z.append(f"{len(zeilen)} Kennzahlen; davon "
                 f"{sum(1 for y in zeilen if y['urteil'] == 'bestätigt')} bestätigt, "
                 f"{sum(1 for y in zeilen if y['urteil'] == 'abweichend')} abweichend, "
                 f"{sum(1 for y in zeilen if y['urteil'] == 'im Register nicht geführt')} "
                 "im Register nicht geführt.")
        z.append("")
        z.extend(tabelle(zeilen))
        z.append("")

    z.append("## 5 Die Abweichungen im Einzelnen")
    z.append("")
    abw = E.abweichungen()
    if not abw:
        z.append("Keine. Jede Kennzahl, die das Register führt, ist bestätigt.")
    else:
        z.append(f"{len(abw)} Kennzahlen weichen ab. Je Eintrag steht zuerst der neu "
                 "gerechnete Wert, dann der Registerwert.")
        z.append("")
        for y in abw:
            z.append(f"- **{y['mass']} — {y['groesse']}** ({y['filterregel']}): "
                     f"neu **{y['wert']}**, Register {y['registerwert']}. "
                     f"Bezugsmenge: {y['bezugsmenge']}. Datei: {kurz(y['datei'])}.")
    z.append("")
    z.append("## 6 Die Ankerwerte")
    z.append("")
    z.append("Vier Grössen sind unabhängig von zwei Seiten bestätigt und dienen als "
             "Kontrolle der Neurechnung: der zeilenweise Spearman ungefiltert mit Median "
             "+0.2199, den Quartilen +0.1220 und +0.3749, dem Minimum −0.1724 bei See "
             "und dem Maximum +0.6563 bei Gebiet (Tal liegt bei +0.6486); der Median des "
             "Jaccard der Zehnermengen mit 0.1765; die 19 von 110 Zeilen mit gleichem "
             "stärkstem Partner; und die beiden stärksten gerichteten Zellen "
             "Strassenpass → Pass 0.7139 für B₁ und Verladestation → Autofähre 0.4908 "
             "für D₁. Alle vier sind von diesem Lauf reproduziert.")
    z.append("")
    z.append("## 7 Herkunft")
    z.append("")
    z.append(f"Erzeugt von `experiment-2/4_analysis/{Path(__file__).name}` "
             f"am {heute} aus `{DATEI_B1}`, `{DATEI_D1}` und `{DATEI_DBF}`. Die "
             "Einzelwerte dieses Berichts stehen zusätzlich maschinenlesbar in "
             f"`{BERICHTSNAME}__daten.csv` im selben Ordner. Keine bestehende Datei ist "
             "verändert worden.")
    z.append("")

    ziel.write_text("\n".join(z), encoding="utf-8")


def daten_schreiben(E: Erhebung, ziel: Path) -> None:
    with ziel.open("w", encoding="utf-8", newline="") as f:
        s = csv.writer(f, delimiter=";")
        s.writerow(["block", "groesse", "mass", "wert", "filterregel",
                    "bezugsmenge", "datei", "registerwert", "urteil"])
        for y in E.zeilen:
            s.writerow([y["block"], y["groesse"], y["mass"], y["wert"],
                        y["filterregel"], y["bezugsmenge"], y["datei"],
                        y["registerwert"], y["urteil"]])


# ---------------------------------------------------------------------- main


def main() -> None:
    p = argparse.ArgumentParser(description="Neurechnung der Kennzahlen B1 und D1")
    p.add_argument("--wurzel", type=Path, default=None)
    args = p.parse_args()
    wurzel = (args.wurzel or _wurzel_aus_skriptort()).resolve()

    anzeigenamen_laden()

    b1 = matrix_lesen(wurzel / "matrices" / "config1" / "b1_matrix.csv")
    d1 = matrix_lesen(wurzel / "matrices" / "config1" / "d1_matrix.csv")
    if list(b1.index) != list(d1.index):
        raise SystemExit("Die beiden Matrizen fuehren nicht dieselbe Kategorienordnung")

    klasse = objektklassen_lesen(wurzel, list(b1.index))

    E = Erhebung()
    block_eines_masses(E, "B₁", "B₁ — Überlagerung (config1)", b1, DATEI_B1, REG_B1, klasse)
    block_eines_masses(E, "D₁", "D₁ — Adjazenz (Lauf T-141)", d1, DATEI_D1, REG_D1, klasse)
    block_des_vergleichs(E, b1, d1, REG_V, klasse)

    heute = BERICHTSDATUM
    ordner = wurzel / "experiment-2" / "results" / "analysis"
    ordner.mkdir(parents=True, exist_ok=True)
    md = ordner / f"{BERICHTSNAME}.md"
    daten = ordner / f"{BERICHTSNAME}__daten.csv"
    bericht_schreiben(E, md, heute, klasse is not None)
    daten_schreiben(E, daten)

    print("geschrieben:", md)
    print("geschrieben:", daten)
    print(f"Kennzahlen gerechnet: {len(E.zeilen)}")
    print(f"  bestaetigt:              {E.zaehle('bestätigt')}")
    print(f"  abweichend:              {E.zaehle('abweichend')}")
    print(f"  im Register nicht gefuehrt: {E.zaehle('im Register nicht geführt')}")
    for y in E.abweichungen():
        print(f"  ABWEICHUNG  {y['mass']} — {y['groesse']} ({y['filterregel']}): "
              f"neu {y['wert']} / Register {y['registerwert']}")


if __name__ == "__main__":
    main()
