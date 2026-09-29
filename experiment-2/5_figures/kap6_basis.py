# kap6_basis.py
#
# Erzeugt: nichts für sich allein — dies ist das gemeinsame Grundmodul der vier
#          Anhangsskripte anhang_a1_partnerlisten.py, anhang_a2_verteilung.py,
#          anhang_a3_schwellwertspur.py und anhang_a4_zeilenvergleich.py.
#          Als Skript aufgerufen schreibt es zusätzlich das Kategorienregister
#          Material_Anhang/A0_Kategorienregister.md.
# Eingaben: <ROOT>/matrices/config1/b1_matrix.csv   (B1, config1)
#           <ROOT>/matrices/config1/d1_matrix.csv   (D1, config1)
#           <ROOT>/data/swissnames3d/swissNAMES3D_{PKT,LIN,PLY}.dbf
#               (Feld OBJEKTKLAS, thematische Objektklasse)
#           optional <ROOT>/experiment-2/5_figures/output/  (Anzeigetabelle aus
#               kategorie_anzeigenamen.py; wird übernommen, wenn vorhanden, sonst die eigene)
# Aufruf:   python3 kap6_basis.py [--wurzel <ma-experiments>]
#           Ohne --wurzel wird MA_ROOT aus der Umgebung genommen, sonst ma-experiments
#           aus dem Ort dieses Skripts (<ROOT>/experiment-2/5_figures/).
#
# Der Wurzelpfad steht ausschliesslich in ROOT_DEFAULT; feste Pfade gibt es sonst nicht.

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import struct
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------- Wurzelpfade

ROOT_DEFAULT = Path(os.environ.get("MA_ROOT", Path(__file__).resolve().parents[2]))

SCHWELLE = 0.001  # Auswahlschwelle des Resolvers
KAPPUNG = 10  # höchste Zahl von Partnern je Zeile


def pfade(root: Path) -> dict:
    """Alle Eingabe- und Ausgabepfade, aus der Repo-Wurzel ma-experiments abgeleitet."""
    exp2 = root / "experiment-2"
    analyse = exp2 / "results" / "analysis"          # Berichte und Datendateien der Analysen
    return {
        "root": root,
        "vault": analyse,                               # frueher: Notizen-Vault
        "b1": root / "matrices" / "config1" / "b1_matrix.csv",
        "d1": root / "matrices" / "config1" / "d1_matrix.csv",
        "shp": root / "data" / "swissnames3d",
        "caches": exp2 / "3_evaluation" / "cache",      # descriptions_E_*.pkl, items_E_*.pkl
        "rueckgabe": exp2 / "3_evaluation" / "cache" / "per_item",
        "pool": root / "data" / "d1" / "1_data" / "pool.parquet",   # optional, nicht vom Plugin abgelegt
        "fingerprint": root / "data" / "fp_gazetteer_config1.json",  # optional
        "material_k6": exp2 / "5_figures" / "output",
        "material_anhang": analyse,
        "abbildungen": exp2 / "5_figures" / "output",
    }


def argumente() -> Path:
    p = argparse.ArgumentParser()
    p.add_argument("--wurzel", "--root", dest="wurzel", default=str(ROOT_DEFAULT),
                   help="Projektwurzel MA/ (Vorgabe: Umgebungsvariable MA_ROOT, sonst "
                        "ROOT_DEFAULT; --root ist ein gleichwertiger Zweitname)")
    return Path(p.parse_args().wurzel)


# ------------------------------------------------------------------ Matrizen


def lade_matrizen(P: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lädt B1 und D1 wie der Resolver: Semikolon, erste Spalte als Index."""
    b1 = pd.read_csv(P["b1"], sep=";", index_col=0)
    d1 = pd.read_csv(P["d1"], sep=";", index_col=0)
    if list(b1.index) != list(d1.index) or list(b1.columns) != list(d1.columns):
        raise SystemExit("B1 und D1 tragen nicht dieselbe Kategorienordnung.")
    return b1, d1


def partner(matrix: pd.DataFrame, kategorie: str,
            schwelle: float = SCHWELLE, kappung: int = KAPPUNG) -> list[tuple[str, float]]:
    """Die Partner, die der Resolver sieht.

    Nachbildung von AssociationMatrixLoader.get_associated_categories aus
    geoparser_h3_resolver/sentence_generator/association_loader.py: Selbstbezug
    weg, Wert >= Schwelle, absteigend stabil sortieren, vorne kappen. Die
    Stabilität der Sortierung ist erheblich, weil Gleichstände dadurch die
    Spaltenreihenfolge der Matrix behalten — genau wie im Code.
    """
    if kategorie not in matrix.index:
        return []
    zeile = matrix.loc[kategorie]
    kandidaten = [(sp, float(zeile[sp]))for sp in zeile.index
                  if sp != kategorie and float(zeile[sp]) >= schwelle]
    kandidaten.sort(key=lambda x: x[1], reverse=True)
    return kandidaten[:kappung]


# ------------------------------------------------------- DBF ohne Fremdpaket


def _dbf_spalten(pfad: Path, gewuenscht: list[str]) -> list[tuple[str, ...]]:
    """Liest genannte Zeichenfelder einer DBF-Datei. Kein geopandas nötig."""
    with open(pfad, "rb")as f:
        kopf = f.read(32)
        n_rec = struct.unpack("<I", kopf[4:8])[0]
        k_len = struct.unpack("<H", kopf[8:10])[0]
        r_len = struct.unpack("<H", kopf[10:12])[0]
        felder, versatz = {}, 1
        while True:
            d = f.read(32)
            if len(d) < 32 or d[0:1] == b"\r":
                break
            name = d[0:11].split(b"\x00")[0].decode("latin-1")
            felder[name] = (versatz, d[16])
            versatz += d[16]
        fehlend = [g for g in gewuenscht if g not in felder]
        if fehlend:
            raise KeyError(f"{pfad.name}: Feld fehlt: {fehlend}")
        f.seek(k_len)
        aus = []
        for _ in range(n_rec):
            rec = f.read(r_len)
            if len(rec) < r_len:
                break
            if rec[0:1] == b"*":
                continue
            aus.append(tuple(rec[o:o + l].decode("utf-8", "replace").strip()
                             for g in gewuenscht for (o, l) in [felder[g]]))
    return aus


def objektklassen(P: dict) -> tuple[dict, dict]:
    """Kategorie -> Objektklasse und Kategorie -> Objektzahl aus der Lieferung 2024.

    Die fünf Kategorien aus swissBOUNDARIES3D kommen in den SwissNames3D-Dateien
    nicht vor und werden der eigenen Gruppe swissBOUNDARIES3D zugeschlagen.
    """
    klasse, zahl = {}, collections.Counter()
    for teil in ("PKT", "LIN", "PLY"):
        for oa, ok in _dbf_spalten(P["shp"] / f"swissNAMES3D_{teil}.dbf",
                                   ["OBJEKTART", "OBJEKTKLAS"]):
            klasse[oa] = ok
            zahl[oa] += 1
    for verwaltung in ("Gemeindegebiet", "Kommunanz", "Bezirk", "Kantonsgebiet", "Kanton"):
        klasse[verwaltung] = "swissBOUNDARIES3D"
        # Die fünf Verwaltungskategorien stehen nicht in den SwissNames3D-Dateien.
        # Ihre Objektzahl liefert der Fingerabdruck des config1-Gazetteers, der für
        # die 105 übrigen Kategorien Zahl für Zahl mit der DBF-Zählung übereinstimmt.
    if P["fingerprint"].is_file():
        fp = json.loads(P["fingerprint"].read_text(encoding="utf-8"))
        for kat, n in fp.get("kategorien", {}).items():
            zahl.setdefault(kat, n)
    return klasse, dict(zahl)

    # --------------------------------------------- Anzeigenamen mit Umlauten

    # swisstopo schreibt das Attribut OBJEKTART durchgehend in Ersatzschreibung, während
    # das Feld NAME derselben Quelle Umlaute führt. Für den Satz ist die Ersatzschreibung
    # zurückzunehmen. In Schweizer Rechtschreibung betrifft das ausschliesslich die
    # Umlaute; ss bleibt ss und ist keine Ersatzschreibung.
ANZEIGENAMEN = {
    "Autofaehre": "Autofähre",
    "Fliessgewaesser": "Fliessgewässer",
    "Gebaeude": "Gebäude",
    "Grotte, Hoehle": "Grotte, Höhle",
    "Haupthuegel": "Haupthügel",
    "Huegel": "Hügel",
    "Huegelzug": "Hügelzug",
    "Oeffentliches Parkareal": "Öffentliches Parkareal",
    "Oeffentliches Parkplatzareal": "Öffentliches Parkplatzareal",
    "Offenes Gebaeude": "Offenes Gebäude",
    "Personenfaehre mit Seil": "Personenfähre mit Seil",
    "Personenfaehre ohne Seil": "Personenfähre ohne Seil",
    "Sakrales Gebaeude": "Sakrales Gebäude",
    "Truppenuebungsplatz": "Truppenübungsplatz",
    "Uebrige Bahnen": "Übrige Bahnen",
    "Zollamt 24h eingeschraenkt": "Zollamt 24h eingeschränkt",
    "Zollamt eingeschraenkt": "Zollamt eingeschränkt",
}


def anzeigenamen(P: dict) -> tuple[dict, str]:
    """Die Anzeigetabelle, bevorzugt aus Material_Kapitel6, sonst die eigene.

    Ein anderer Arbeitsstrang legt unter Material_Kapitel6/ eine Zuordnungstabelle
    ab. Liegt dort eine CSV oder JSON mit den Spalten Quelle/Anzeige, wird sie
    genommen; sonst gilt die Tabelle ANZEIGENAMEN oben. Der zweite Rückgabewert
    nennt die benutzte Quelle und gehört in die Legende.
    """
    ordner = P["material_k6"]
    if ordner.is_dir():
        for datei in sorted(ordner.iterdir()):
            try:
                if datei.suffix.lower() == ".json":
                    roh = json.loads(datei.read_text(encoding="utf-8"))
                    if isinstance(roh, dict) and roh:
                        return {str(k): str(v)for k, v in roh.items()}, datei.name
                elif datei.suffix.lower() == ".csv":
                    tab = pd.read_csv(datei, sep=None, engine="python")
                    sp = [c for c in tab.columns
                          if re.search(r"quelle|objektart|ersatz|maschinenname", str(c), re.I)]
                    za = [c for c in tab.columns
                          if re.search(r"anzeige|umlaut|korrekt", str(c), re.I)]
                    if sp and za:
                        return dict(zip(tab[sp[0]].astype(str),
                                    tab[za[0]].astype(str))), datei.name
            except Exception:
                continue
    return dict(ANZEIGENAMEN), "eigene Tabelle in kap6_basis.py"


def anzeige(name: str, tabelle: dict) -> str:
    return tabelle.get(name, name)

    # --------------------------------------------------- Gruppen, deutsch benannt


GRUPPEN_DEUTSCH = {
    "TLM_NAME_PKT": "Gipfel und Pässe",
    "TLM_GELAENDENAME": "Geländenamen",
    "TLM_MORPH_KLEINFORM_PKT": "Morphologische Kleinformen",
    "TLM_GEBIETSNAME": "Gebietsnamen",
    "TLM_FLURNAME": "Flurnamen",
    "TLM_SIEDLUNGSNAME": "Siedlungsnamen",
    "swissBOUNDARIES3D": "Verwaltungseinheiten",
    "TLM_FLIESSGEWAESSER": "Fliessgewässer",
    "TLM_STEHENDES_GEWAESSER": "Stehende Gewässer",
    "TLM_STAUBAUTE": "Staubauten",
    "TLM_GEBAEUDE": "Gebäude",
    "TLM_EINZELOBJEKT": "Einzelobjekte",
    "TLM_NUTZUNGSAREAL": "Nutzungsareale",
    "TLM_FREIZEITAREAL": "Freizeitareale",
    "TLM_VERKEHRSAREAL": "Verkehrsareale",
    "TLM_SPORTBAUTE_LIN": "Sportbauten",
    "TLM_STRASSE": "Strassen",
    "TLM_STRASSENINFO": "Strasseninfrastruktur",
    "TLM_AUS_EINFAHRT": "Aus- und Einfahrten",
    "TLM_EISENBAHN": "Eisenbahn",
    "TLM_UEBRIGE_BAHN": "Seil- und übrige Bahnen",
    "TLM_HALTESTELLE": "Haltestellen",
    "TLM_SCHIFFFAHRT": "Schifffahrt",
}

# Druckreihenfolge: zuerst das Gelände, dann Siedlung und Verwaltung, dann Gewässer,
# dann Bauten und Areale, zuletzt die Verkehrsträger.
GRUPPEN_ORDNUNG = [
    "TLM_NAME_PKT", "TLM_GELAENDENAME", "TLM_MORPH_KLEINFORM_PKT", "TLM_GEBIETSNAME",
    "TLM_FLURNAME", "TLM_SIEDLUNGSNAME", "swissBOUNDARIES3D",
    "TLM_FLIESSGEWAESSER", "TLM_STEHENDES_GEWAESSER", "TLM_STAUBAUTE",
    "TLM_GEBAEUDE", "TLM_EINZELOBJEKT", "TLM_NUTZUNGSAREAL", "TLM_FREIZEITAREAL",
    "TLM_SPORTBAUTE_LIN", "TLM_VERKEHRSAREAL",
    "TLM_STRASSE", "TLM_STRASSENINFO", "TLM_AUS_EINFAHRT",
    "TLM_EISENBAHN", "TLM_UEBRIGE_BAHN", "TLM_HALTESTELLE", "TLM_SCHIFFFAHRT",
]


def geordnete_kategorien(kategorien: list[str], klasse: dict,
                         namen: dict) -> list[tuple[str, list[str]]]:
    """Die 110 Kategorien nach Objektklasse gruppiert, innerhalb alphabetisch."""
    nach_gruppe = collections.defaultdict(list)
    for k in kategorien:
        nach_gruppe[klasse.get(k, "ohne Klasse")].append(k)
    aus = []
    for g in GRUPPEN_ORDNUNG:
        if g in nach_gruppe:
            aus.append((g, sorted(nach_gruppe.pop(g), key=lambda k: anzeige(k, namen))))
    for g in sorted(nach_gruppe):
        aus.append((g, sorted(nach_gruppe[g], key=lambda k: anzeige(k, namen))))
    return aus


# ----------------------------------------------------------------- Satzhilfen


def tausender(n) -> str:
    """Tausendertrennzeichen als Apostroph, wie in der Arbeit."""
    return f"{int(n):,}".replace(",", "’")


def kopf(name: str, beschreibung: str, datum: str = "2026-09-17") -> str:
    return (f"---\nname: {name}\ndescription: {beschreibung}\ndate: {datum}\n---\n\n")


# ---------------------------------------------------------------- A0 schreiben


def _schreibe_a0(P: dict) -> None:
    b1, _ = lade_matrizen(P)
    klasse, zahl = objektklassen(P)
    namen, quelle = anzeigenamen(P)
    kategorien = list(b1.index)
    gruppen = geordnete_kategorien(kategorien, klasse, namen)

    z = [kopf(
        "A0 — Kategorienregister der 110 Objektkategorien",
        "Die 110 Objektkategorien der beiden Assoziationsmatrizen mit ihrer "
        "Schreibweise in der Quelle, dem Anzeigenamen für den Satz, der "
        "thematischen Objektklasse von swisstopo und der Objektzahl der "
        "Lieferung 2024. Grundlage der Ordnung aller übrigen Anhangsposten.")]
    z.append("# A0 — Kategorienregister\n")
    z.append(
        "Die Objektkategorien der beiden Assoziationsmatrizen sind die Werte des "
        "Attributs `OBJEKTART` von SwissNames3D, ergänzt um fünf Kategorien aus "
        "swissBOUNDARIES3D. swisstopo schreibt dieses Attribut durchgehend in "
        "Ersatzschreibung, während das Feld `NAME` derselben Quelle Umlaute "
        "führt. Für den Satz ist die Ersatzschreibung zurückzunehmen; die "
        "Spalte «Anzeigename» leistet das. Die Gruppierung stammt aus dem "
        "zweiten Attribut `OBJEKTKLASSE`, das swisstopo selbst als thematische "
        "Obergruppe vergibt und das hier nicht erfunden, sondern gelesen ist.\n")
    z.append(f"Quelle der Anzeigetabelle: {quelle}.\n")

    ersatz = [(k, namen[k])for k in kategorien if k in namen and namen[k] != k]
    z.append(f"\n## Ersatzschreibungen ({len(ersatz)} von {len(kategorien)})\n")
    z.append("| Schreibweise der Quelle | Anzeigename |")
    z.append("|---|---|")
    for q, a in sorted(ersatz, key=lambda x: x[1]):
        z.append(f"| {q} | {a} |")

    z.append("\n## Die 110 Kategorien nach Objektklasse\n")
    z.append("| Gruppe | Kategorie (Anzeigename) | Schreibweise der Quelle | Objekte |")
    z.append("|---|---|---|---:|")
    for g, kats in gruppen:
        for i, k in enumerate(kats):
            gname = GRUPPEN_DEUTSCH.get(g, g)if i == 0 else ""
            n = zahl.get(k)
            z.append(f"| {gname} | {anzeige(k, namen)} | {k} | "
                     f"{tausender(n) if n else '—'} |")

    ziel = P["material_anhang"] / "A0_Kategorienregister.md"
    ziel.parent.mkdir(parents=True, exist_ok=True)
    ziel.write_text("\n".join(z) + "\n", encoding="utf-8")
    print(f"geschrieben: {ziel}")
    print(f"  Gruppen: {len(gruppen)} · Kategorien: {len(kategorien)} · "
          f"Ersatzschreibungen: {len(ersatz)}")


if __name__ == "__main__":
    _schreibe_a0(pfade(argumente()))
