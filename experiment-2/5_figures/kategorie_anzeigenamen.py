# -*- coding: utf-8 -*-
"""
ERZEUGNIS  Anzeigetabelle der Ersatzschreibungen: die vollstaendige Zuordnung vom
           Maschinennamen der Objektkategorie (so, wie swisstopo das Attribut
           OBJEKTART schreibt und wie er in den Matrixdateien steht) auf den
           Anzeigenamen mit korrekten Umlauten.
           Ausgabe 1: 5_figures/output/Anzeigetabelle_Kategorien.md
           Ausgabe 2: dieselbe Zuordnung als Anzeigetabelle_Kategorien.csv
                      (Spalten: maschinenname;anzeigename;geaendert)

EINGABE    ma-experiments/matrices/config1/b1_matrix.csv
           (nur die Kopfzeile; sie liefert die 110 Kategorienamen und dient als
           Vollstaendigkeitspruefung gegen die unten fest hinterlegte Tabelle)

ZWECK      Die Zuordnung wird von jeder Tabelle und jeder Abbildung gebraucht, in
           der Kategorienamen vorkommen. Andere Skripte binden sie als Modul ein:
               from kategorie_anzeigenamen import anzeige, ERSATZSCHREIBUNG

AUFRUF     python3 kategorie_anzeigenamen.py [--wurzel PFAD_ZU_MA_EXPERIMENTS]
           Ohne --wurzel wird ma-experiments aus dem Ort dieses Skripts abgeleitet.

HINWEIS    Zwei Kategorienamen enthalten die Buchstabenfolge ue beziehungsweise
           ae, ohne eine Ersatzschreibung zu sein: Quelle und Staumauer. Wer die
           Umlaute mit einem Suchmuster wiederherstellt statt mit dieser Tabelle,
           zerstoert diese beiden Namen. Das ist der Grund, weshalb die Zuordnung
           von Hand gefuehrt und nicht gerechnet wird.

Stand 17.09.2026
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
from pathlib import Path

# ---------------------------------------------------------------------------
# Die Zuordnung. Links der Maschinenname aus dem Attribut OBJEKTART, rechts der
# Anzeigename fuer Satz und Druck. Nur Namen, die sich unterscheiden, stehen hier.
# ---------------------------------------------------------------------------
ERSATZSCHREIBUNG: dict[str, str] = {
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

# Namen, die ein Suchmuster faelschlich treffen wuerde. Sie bleiben unveraendert.
SCHEINTREFFER: tuple[str, ...] = ("Quelle", "Staumauer")


def anzeige(name: str) -> str:
    """Gibt den Anzeigenamen einer Objektkategorie zurueck."""
    return ERSATZSCHREIBUNG.get(name, name)


def _wurzel_aus_skriptort() -> Path:
    # ma-experiments/experiment-2/5_figures/<dieses Skript>
    return Path(__file__).resolve().parents[2]


def kategorien_lesen(b1_csv: Path) -> list[str]:
    with b1_csv.open(encoding="utf-8") as f:
        kopf = next(csv.reader(f, delimiter=";"))
    return kopf[1:]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wurzel", type=Path, default=None,
                   help="Ordner ma-experiments; ohne Angabe aus dem Skriptort abgeleitet")
    args = p.parse_args()
    wurzel = (args.wurzel or _wurzel_aus_skriptort()).resolve()

    b1 = wurzel / "matrices" / "config1" / "b1_matrix.csv"
    kategorien = kategorien_lesen(b1)

    unbekannt = sorted(set(ERSATZSCHREIBUNG) - set(kategorien))
    if unbekannt:
        raise SystemExit("Diese Maschinennamen stehen nicht in der Matrix: " + ", ".join(unbekannt))

    geaendert = [k for k in kategorien if k in ERSATZSCHREIBUNG]
    ziel = Path(__file__).resolve().parent / "output"
    ziel.mkdir(parents=True, exist_ok=True)
    heute = _dt.date.today().isoformat()

    # --- CSV ---------------------------------------------------------------
    with (ziel / "Anzeigetabelle_Kategorien.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["maschinenname", "anzeigename", "geaendert"])
        for k in kategorien:
            w.writerow([k, anzeige(k), "ja" if k in ERSATZSCHREIBUNG else "nein"])

    # --- Markdown ----------------------------------------------------------
    z = []
    z.append("---")
    z.append("name: Anzeigetabelle der Objektkategorien")
    z.append("description: Die vollstaendige Zuordnung vom Maschinennamen der 110 "
             "Objektkategorien auf den Anzeigenamen mit korrekten Umlauten. "
             "swisstopo schreibt das Attribut OBJEKTART in Ersatzschreibung; die "
             "Tabelle ist vor jeder Ausgabe anzuwenden, in der Kategorienamen "
             "vorkommen.")
    z.append(f"date: {heute}")
    z.append("---")
    z.append("")
    z.append("# Anzeigetabelle der Objektkategorien")
    z.append("")
    z.append("Die Matrixdateien und alle Zwischenergebnisse führen die Kategorienamen so, "
             "wie swisstopo das Attribut `OBJEKTART` schreibt, nämlich in Ersatzschreibung "
             "ohne Umlaute. Das Feld `NAME` derselben Quelle führt daneben Umlaute "
             "(«Hägelen», «Lützelflüh»); die Ersatzschreibung ist also eine Eigenschaft des "
             "Attributs, kein Fehler der Verarbeitung. Für Satz und Druck ist deshalb vor "
             "jeder Ausgabe die folgende Zuordnung anzuwenden.")
    z.append("")
    z.append(f"Von den 110 Kategorienamen tragen **{len(geaendert)}** eine Ersatzschreibung. "
             "Die übrigen 93 bleiben unverändert; sie stehen vollständig in der Nebendatei "
             "`Anzeigetabelle_Kategorien.csv`, damit die Zuordnung maschinell angewandt "
             "werden kann.")
    z.append("")
    z.append(f"## Die {len(geaendert)} Kategorien mit Ersatzschreibung")
    z.append("")
    z.append("| Maschinenname (`OBJEKTART`) | Anzeigename |")
    z.append("|---|---|")
    for k in geaendert:
        z.append(f"| `{k}` | {anzeige(k)} |")
    z.append("")
    z.append("## Zwei Namen, die nicht zu ändern sind")
    z.append("")
    z.append("Wer die Umlaute mit einem Suchmuster über `ae`, `oe` und `ue` wiederherstellt "
             "statt mit dieser Tabelle, zerstört zwei Namen, in denen die Buchstabenfolge "
             "keine Ersatzschreibung ist:")
    z.append("")
    z.append("| Maschinenname | bleibt |")
    z.append("|---|---|")
    for k in SCHEINTREFFER:
        z.append(f"| `{k}` | {k} |")
    z.append("")
    z.append("Das ist der Grund, weshalb die Zuordnung von Hand geführt wird.")
    z.append("")
    z.append("## Ein Sonderfall, der nicht die Schreibweise betrifft")
    z.append("")
    z.append("Die Kategorie `Zollamt 24h 24h` trägt in der Quelle eine offenkundige "
             "Verdoppelung. Sie ist hier bewusst unverändert gelassen, weil ihre Bereinigung "
             "eine inhaltliche Entscheidung über den Quelldatensatz wäre und nicht eine über "
             "die Schreibweise. Wer sie im Satz zu «Zollamt 24h» kürzt, vermerkt das in der "
             "Legende.")
    z.append("")
    z.append("## Herkunft")
    z.append("")
    z.append("Die Namensliste ist der Kopfzeile von "
             "`matrices/config1/b1_matrix.csv` entnommen; sie ist "
             "mit der Kopfzeile von `matrices/config1/d1_matrix.csv` identisch. Erzeugt von "
             "`experiment-2/5_figures/kategorie_anzeigenamen.py` "
             f"am {heute}.")
    z.append("")
    (ziel / "Anzeigetabelle_Kategorien.md").write_text("\n".join(z), encoding="utf-8")

    print(f"Kategorien gelesen: {len(kategorien)}")
    print(f"davon mit Ersatzschreibung: {len(geaendert)}")
    print("geschrieben:", ziel / "Anzeigetabelle_Kategorien.md")
    print("geschrieben:", ziel / "Anzeigetabelle_Kategorien.csv")


if __name__ == "__main__":
    main()
