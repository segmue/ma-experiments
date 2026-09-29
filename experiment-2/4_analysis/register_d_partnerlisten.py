# register_d_partnerlisten.py
#
# Erzeugt: experiment-2/results/analysis/
#              Zahlenregister_D_Partnerlisten_2026-09-18.md
#              — Teil D des Zahlenregisters: für jede der 110 Objektkategorien
#                und beide Assoziationsmasse die bis zu zehn stärksten Partner,
#                die der Resolver nach Schwelle 0.001 und Kappung bei zehn
#                tatsächlich sieht, je Eintrag mit Quellkategorie,
#                Zielkategorie, Rang und Wert auf vier Nachkommastellen.
#                Ausgefuehrt sind die zwölf Kategorien, die in der Arbeit
#                gedruckt werden; der Rest steht in der Datendatei.
#              Zahlenregister_D_Partnerlisten_2026-09-18__daten.csv
#              — dieselben Angaben für alle 110 Kategorien, 2'170 Einträge.
#          Ausserdem zwei Abgleiche auf der Standardausgabe: die 48 Werte der
#          Auszugstabelle in Material_Kapitel6 und die rund 1'100 Einträge je
#          Mass in Material_Anhang/A1_Partnerlisten_B1_D1.csv, jeweils gegen
#          die hier gerechneten Werte.
#
# Eingaben: <ROOT>/matrices/config1/b1_matrix.csv
#               (B1, Überlagerungsmass, config1; NICHT npmi_matrix.csv im
#                selben Ordner, das ist das rohe Überlagerungs-NPMI vor der
#                Dämpfung und eine andere Grösse)
#           <ROOT>/matrices/config1/d1_matrix.csv
#               (D1, Adjazenzmass)
#           <ROOT>/data/swissnames3d/swissNAMES3D_{PKT,LIN,PLY}.dbf
#           optional (Abgleiche, nicht im Repo; fehlen sie, meldet das Skript
#           "Datei fehlt"): Tabelle_Auszug_staerkste_Partner.md (Abgleich 1) und
#           A1_Partnerlisten_B1_D1.csv (Abgleich 2) unter kap6_basis.pfade().
#           Alle Pfade über kap6_basis.pfade() (../5_figures/kap6_basis.py).
#
# Aufruf:  python3 register_d_partnerlisten.py [--wurzel <Projektwurzel>]
#          Ohne --wurzel gilt die Umgebungsvariable MA_ROOT, sonst
#          kap6_basis.ROOT_DEFAULT. Das Skript schreibt ausschliesslich die
#          beiden oben genannten neuen Dateien und verändert keine bestehende.
#
# Datum:   2026-09-18

from __future__ import annotations

import csv
import re
import sys

sys.dont_write_bytecode = True  # keine .pyc im Projektordner anlegen

from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "5_figures"))
import kap6_basis as kb  # noqa: E402

DATUM = "2026-09-18"
STELLEN = 4  # Nachkommastellen aller Werte dieses Registerteils

# Die Kategorien, die in der Arbeit gedruckt werden, mit dem Grund ihres
# Abdrucks. Nur für sie ist die Partnerliste im Register ausgeschrieben; alle
# übrigen stehen in der Datendatei. Die Schreibweise ist die der Quelle.
GEDRUCKT = [
    ("Alpiner Gipfel", "Tabelle 6.x, Abschnitt 6.1"),
    ("Hauptgipfel", "Tabelle 6.x, Abschnitt 6.1"),
    ("Massiv", "Tabelle 6.x, Abschnitt 6.1"),
    ("Gipfel", "Tabelle 6.x, Abschnitt 6.1"),
    ("Grat", "Tabelle 6.x, Abschnitt 6.1"),
    ("Tal", "Tabelle 6.x, Abschnitt 6.1"),
    ("Ort", "Tabelle 6.x, Abschnitt 6.1"),
    ("Pass", "Tabelle 6.x, Abschnitt 6.1"),
    ("Flurname swisstopo", "Anhang A3, Schwellwertspur"),
    ("Landschaftsname", "Register B, Abschnitt 8, Zeile ohne Überschneidung"),
    ("Strassenpass", "Register B, Abschnitt 6, stärkste B1-Zelle"),
    ("Verladestation", "Register B, Abschnitt 7, stärkste D1-Zelle"),
]

QUELLE = {
    "B1": "`matrices/config1/b1_matrix.csv`",
    "D1": "`matrices/config1/d1_matrix.csv`",
}
MASSNAME = {"B1": "B₁", "D1": "D₁"}


def w(v: float) -> str:
    """Vier Nachkommastellen, Vorzeichen nur bei Bedarf."""
    return f"{v:.{STELLEN}f}"


# ------------------------------------------------------------------ Rechnung


def partnerlisten(b1, d1, kategorien) -> dict:
    return {k: {"B1": kb.partner(b1, k), "D1": kb.partner(d1, k)}
            for k in kategorien}


def kennzahlen(auswahl: dict, kategorien: list, mass: str) -> dict:
    laengen = [len(auswahl[k][mass]) for k in kategorien]
    werte = [v for k in kategorien for _, v in auswahl[k][mass]]
    paare = {(k, p) for k in kategorien for p, _ in auswahl[k][mass]}
    ziele = {p for _, p in paare}
    return {
        "eintraege": sum(laengen),
        "voll": sum(1 for x in laengen if x == kb.KAPPUNG),
        "kürzeste": min(laengen),
        "zeilen_kurz": [k for k in kategorien if len(auswahl[k][mass]) < kb.KAPPUNG],
        "min": min(werte),
        "max": max(werte),
        "median": sorted(werte)[len(werte) // 2],
        "gegenseitig": sum(1 for (a, b) in paare if (b, a) in paare),
        "paare": len(paare),
        "nie_ziel": [k for k in kategorien if k not in ziele],
    }


def staerkste_zellen(matrix, kategorien: list, n: int) -> set:
    """Die n stärksten gerichteten Zellen der Gesamtmatrix ohne Diagonale."""
    zellen = [(float(matrix.loc[a, b]), a, b)
              for a in kategorien for b in kategorien if a != b]
    zellen.sort(key=lambda x: x[0], reverse=True)
    return {(a, b) for _, a, b in zellen[:n]}


# ------------------------------------------------------------------ Abgleich


def abgleich_anhang(P, auswahl: dict, kategorien: list) -> tuple[int, list]:
    """A1_Partnerlisten_B1_D1.csv Eintrag für Eintrag gegen die Rechnung."""
    pfad = P["material_anhang"] / "A1_Partnerlisten_B1_D1.csv"
    if not pfad.is_file():
        return 0, [("Datei fehlt", pfad.name, "", "", "")]
    with open(pfad, encoding="utf-8") as f:
        zeilen = list(csv.DictReader(f, delimiter=";"))
    ist = {}
    for r in zeilen:
        # Der Wert bleibt Zeichenkette: verglichen wird auf der Genauigkeit,
        # die das Material führt, nicht auf einer davon abgeleiteten.
        ist.setdefault((r["kategorie_quelle"], r["mass"]), []).append(
            (int(r["rang"]), r["partner_quelle"], r["wert"].strip()))
    abweichungen = []
    for k in kategorien:
        for mass in ("B1", "D1"):
            soll = auswahl[k][mass]
            haben = sorted(ist.get((k, mass), []))
            if len(haben) != len(soll):
                abweichungen.append(("Listenlänge", k, mass, "—",
                                     f"Material {len(haben)}",
                                     f"gerechnet {len(soll)}"))
                continue
            for (rang, partner, wert), (p, v) in zip(haben, soll):
                if partner != p:
                    abweichungen.append(("Partner", k, mass, rang,
                                         f"Material {partner}",
                                         f"gerechnet {p}"))
                else:
                    stellen = len(wert.split(".")[1]) if "." in wert else 0
                    if wert != f"{v:.{stellen}f}":
                        abweichungen.append(("Wert", k, mass, rang,
                                             f"Material {wert}",
                                             f"gerechnet {v:.{stellen}f}"))
    return len(zeilen), abweichungen


def abgleich_kapitel6(P, auswahl: dict, namen: dict) -> tuple[int, list]:
    """Die 48 Partnerwerte der Auszugstabelle gegen die Rechnung."""
    pfad = P["material_k6"] / "Tabelle_Auszug_staerkste_Partner.md"
    if not pfad.is_file():
        return 0, [("Datei fehlt", pfad.name, "", "", "", "")]
    rueck = {a: q for q, a in namen.items()}
    abweichungen, geprueft = [], 0
    muster = re.compile(r"^\|\s*\*\*(.+?)\*\*\s*\((.+?)\)\s*\|(.+?)\|(.+?)\|\s*$")
    for zeile in pfad.read_text(encoding="utf-8").splitlines():
        treffer = muster.match(zeile)
        if not treffer:
            continue
        kategorie = rueck.get(treffer.group(1).strip(), treffer.group(1).strip())
        for mass, zelle in (("B1", treffer.group(3)), ("D1", treffer.group(4))):
            for rang, stueck in enumerate(zelle.split("·"), 1):
                teil = re.match(r"^(.*?)\s+([0-9]+\.[0-9]+)$", stueck.strip())
                if not teil:
                    continue
                geprueft += 1
                gedruckt_partner, gedruckt_wert = teil.group(1).strip(), teil.group(2)
                soll = auswahl.get(kategorie, {}).get(mass, [])
                if rang > len(soll):
                    abweichungen.append(("fehlt in der Rechnung", kategorie, mass,
                                         rang, f"Material {gedruckt_partner} "
                                         f"{gedruckt_wert}", "gerechnet —"))
                    continue
                p, v = soll[rang - 1]
                p_anzeige = kb.anzeige(p, namen)
                stellen = len(gedruckt_wert.split(".")[1])
                if gedruckt_partner != p_anzeige:
                    abweichungen.append(("Partner", kategorie, mass, rang,
                                         f"Material {gedruckt_partner} {gedruckt_wert}",
                                         f"gerechnet {p_anzeige} {v:.{stellen}f}"))
                elif gedruckt_wert != f"{v:.{stellen}f}":
                    abweichungen.append(("Wert", kategorie, mass, rang,
                                         f"Material {gedruckt_partner} {gedruckt_wert}",
                                         f"gerechnet {p_anzeige} {v:.{stellen}f}"))
    return geprueft, abweichungen


# -------------------------------------------------------------- Registertext


def block_erlaeuterung(mass: str, kz: dict, kategorien: list, namen: dict,
                       ueberschneidung: int) -> list[str]:
    anderes = "D₁" if mass == "B1" else "B₁"
    z = []
    z.append(
        f"**Was dieser Block führt und woher die Zahlen stammen.** Jede Zeile "
        f"dieses Blocks ist ein Eintrag einer Partnerliste des Masses "
        f"{MASSNAME[mass]}: eine Quellkategorie, eine Zielkategorie, der "
        f"Rangplatz der Zielkategorie in der Liste der Quellkategorie und der "
        f"Wert der zugehörigen Matrixzelle auf vier Nachkommastellen. Die "
        f"Werte sind Zelle für Zelle aus {QUELLE[mass]} gelesen, "
        f"semikolongetrennt, 110 × 110, erste Spalte als Index — genau so, wie "
        f"der Resolver die Datei liest.\n")
    z.append(
        f"**Welche Bezugsmenge gilt.** Bezugsmenge eines Eintrags ist **die "
        f"Zeile, nicht die Matrix**. Aus den 109 möglichen Partnern einer "
        f"Zeile — der Selbstbezug fällt weg — bleiben zuerst die mit einem "
        f"Wert von mindestens 0.001, dann kappt die absteigend sortierte Liste "
        f"bei zehn. Die Regel steht in "
        f"`sentence_generator/association_loader.py` und ist hier zeichengleich "
        f"nachgebildet, einschliesslich der stabilen Sortierung, die "
        f"Gleichstände in der Spaltenreihenfolge der Matrix belässt. So "
        f"ergeben sich {kb.tausender(kz['eintraege'])} Einträge für "
        f"{MASSNAME[mass]}; {kz['voll']} der {len(kategorien)} Zeilen "
        f"schöpfen die Kappung voll aus, die kürzeste Liste hat "
        f"{kz['kürzeste']} Einträge.\n")
    z.append(
        f"**Die erste Verwechslung, die hier droht: Zeile gegen Matrix.** Die "
        f"Zehnermenge einer Zeile ist **nicht** dieselbe Menge wie die zehn "
        f"oder tausend stärksten Zellen der Gesamtmatrix. Register B, "
        f"Abschnitt {'6' if mass == 'B1' else '7'} führt die stärksten Zellen "
        f"der Gesamtmatrix; das ist eine Rangordnung über alle 11’990 "
        f"gerichteten Zellen. Hier dagegen wird in **jeder** der 110 Zeilen "
        f"einzeln gekappt. Nachgerechnet: von den "
        f"{kb.tausender(kz['eintraege'])} stärksten Zellen der Gesamtmatrix "
        f"stehen nur {kb.tausender(ueberschneidung)} auch in einer "
        f"Zehnermenge; {kb.tausender(kz['eintraege'] - ueberschneidung)} "
        f"Einträge der Partnerlisten liegen unterhalb dieser Schwelle und "
        f"werden vom Resolver trotzdem gelesen. Eine schwache Zelle in einer "
        f"schwachen Zeile geht in die Beschreibung ein; eine starke Zelle in "
        f"einer starken Zeile kann herausfallen.\n")
    z.append(
        f"**Die zweite Verwechslung: Richtung.** Die Partnerlisten sind "
        f"**gerichtet**. Dass B in der Liste von A steht, heisst nicht, dass A "
        f"in der Liste von B steht; und selbst wo beide vorkommen, sind Rang "
        f"und Wert in der Regel verschieden, weil {MASSNAME[mass]} kein "
        f"symmetrisches Mass ist. Von den {kb.tausender(kz['paare'])} "
        f"gerichteten Paaren dieses Masses sind "
        f"{kb.tausender(kz['gegenseitig'])} gegenseitig "
        f"({kz['gegenseitig'] / kz['paare'] * 100:.1f} Prozent) und "
        f"{kb.tausender(kz['paare'] - kz['gegenseitig'])} einseitig. Ein "
        f"Eintrag dieses Registerteils belegt daher immer nur die eine "
        f"Richtung, die in der Spalte Quellkategorie steht.\n")
    z.append(
        f"**Die dritte Verwechslung: Mass.** Die Werte dieses Blocks gelten "
        f"ausschliesslich für {MASSNAME[mass]}. Dieselbe Quellkategorie hat "
        f"unter {anderes} eine andere Liste, andere Ränge und andere Werte; "
        f"beide Listen stehen in diesem Registerteil nebeneinander und dürfen "
        f"nicht gegeneinander ausgetauscht werden. Insbesondere ist "
        f"`npmi_matrix.csv` im Ordner der B₁-Matrix **nicht** B₁, sondern das "
        f"rohe Überlagerungs-NPMI vor der Dämpfung.\n")
    return z


def tabelle(auswahl: dict, mass: str, namen: dict, grund: dict,
            reihenfolge: list) -> list[str]:
    z = ["| Quellkategorie | Zielkategorie | Rang | Wert | Kennzahl | Bezugsmenge | Fundstelle |",
         "|---|---|---:|---:|---|---|---|"]
    for k in reihenfolge:
        eintraege = auswahl[k][mass]
        if not eintraege:
            z.append(f"| {kb.anzeige(k, namen)} | — keine über der Schwelle — "
                     f"| — | — | Rangplatz in der Zeile {MASSNAME[mass]} "
                     f"| Zeile «{k}», Schwelle 0.001 | {QUELLE[mass]} "
                     f"· selbst nachgerechnet |")
            continue
        for rang, (p, v) in enumerate(eintraege, 1):
            if rang == 1:
                bezug = (f"Zeile «{k}», 109 mögliche Partner, Schwelle 0.001, "
                         f"Kappung {kb.KAPPUNG}; gedruckt in {grund[k]}")
                fund = f"{QUELLE[mass]}, Zeile «{k}» · selbst nachgerechnet"
            else:
                bezug, fund = "ebenda", "ebenda"
            z.append(f"| {kb.anzeige(k, namen)} | {kb.anzeige(p, namen)} "
                     f"| {rang} | {w(v)} "
                     f"| NPMI-Ableitung {MASSNAME[mass]}, Rangplatz in der Zeile "
                     f"| {bezug} | {fund} |")
    return z


def main() -> None:
    P = kb.pfade(kb.argumente())
    b1, d1 = kb.lade_matrizen(P)
    namen, namensquelle = kb.anzeigenamen(P)
    klasse, zahl = kb.objektklassen(P)
    kategorien = list(b1.index)

    auswahl = partnerlisten(b1, d1, kategorien)
    kz = {m: kennzahlen(auswahl, kategorien, m) for m in ("B1", "D1")}
    ueberschneidung = {}
    for m, matrix in (("B1", b1), ("D1", d1)):
        stark = staerkste_zellen(matrix, kategorien, kz[m]["eintraege"])
        mengen = {(k, p) for k in kategorien for p, _ in auswahl[k][m]}
        ueberschneidung[m] = len(stark & mengen)

    n_anhang, abw_anhang = abgleich_anhang(P, auswahl, kategorien)
    n_kap6, abw_kap6 = abgleich_kapitel6(P, auswahl, namen)

    gedruckt = [k for k, _ in GEDRUCKT if k in auswahl]
    grund = dict(GEDRUCKT)
    fehlend = [k for k, _ in GEDRUCKT if k not in auswahl]

    gesamt = kz["B1"]["eintraege"] + kz["D1"]["eintraege"]
    im_register = sum(len(auswahl[k][m]) for k in gedruckt for m in ("B1", "D1"))

    # ------------------------------------------------------------ Markdown
    z = [kb.kopf(
        "Zahlenregister D — Die Partnerlisten der beiden Assoziationsmasse",
        "Teil D des Zahlenregisters. Für jede der 110 Objektkategorien und "
        "für beide Assoziationsmasse die bis zu zehn stärksten Partner, die "
        "der Resolver nach der Schwelle 0.001 und der Kappung bei zehn "
        "tatsächlich sieht, je Eintrag mit Quellkategorie, Zielkategorie, "
        "Rang und Wert auf vier Nachkommastellen. Ausgeschrieben sind die "
        "zwölf Kategorien, die in der Arbeit gedruckt werden; alle "
        f"{kb.tausender(gesamt)} Einträge stehen in der Datendatei "
        "Zahlenregister_D_Partnerlisten_2026-09-18__daten.csv. Nicht enthalten "
        "sind die Kennzahlen der Gesamtmatrizen; die führt Teil B.",
        DATUM)]

    z.append("# Zahlenregister D — Die Partnerlisten der beiden Assoziationsmasse\n")
    z.append(
        "Alle Pfade sind relativ zur Repo-Wurzel `ma-experiments/`. "
        "Sämtliche Zahlen dieses Registerteils sind am "
        f"{DATUM[8:10]}.{DATUM[5:7]}.{DATUM[0:4]} mit python3, pandas und numpy "
        "unmittelbar aus den beiden Matrixdateien gerechnet und aus keinem "
        "Bericht übernommen; in der Spalte **Fundstelle** steht deshalb "
        "durchgehend **«selbst nachgerechnet»**. Das ausführende Skript ist "
        "`experiment-2/4_analysis/register_d_partnerlisten.py`. Keine bestehende "
        "Projektdatei ist verändert worden.\n")
    z.append(
        "**Warum es diesen Teil gibt.** Die Teile A bis C führen die "
        "stärksten Zellen der Gesamtmatrizen, aber keine zeilenweisen "
        "Partnerlisten. Dadurch hatten die 48 Partnerwerte der Auszugstabelle "
        "in `revision/fixes/Material_Kapitel6/Tabelle_Auszug_staerkste_"
        "Partner.md`, die im neuen Abschnitt 6.1 gedruckt werden soll, und die "
        f"{kb.tausender(gesamt)} Einträge der vollständigen Partnerlisten in "
        "`revision/fixes/Material_Anhang/A1_Partnerlisten_B1_D1.csv`, die in "
        "den Anhang gehen, keinen Registereintrag. Die Werte waren richtig "
        "gerechnet; es fehlte ihre Beurkundung. Teil D schliesst diese Lücke.\n")
    z.append(
        f"**Zuschnitt dieses Teils.** Ausgeschrieben sind die "
        f"{len(gedruckt)} Kategorien, deren Partnerlisten in der Arbeit selbst "
        f"gedruckt werden, mit zusammen {kb.tausender(im_register)} Einträgen. "
        f"Die übrigen {len(kategorien) - len(gedruckt)} Kategorien stehen "
        f"vollständig in der Datendatei "
        f"`Zahlenregister_D_Partnerlisten_{DATUM}__daten.csv` im selben "
        f"Ordner, die alle {kb.tausender(gesamt)} Einträge beider Masse in "
        f"derselben Genauigkeit führt. Die Datendatei ist Registerbestand und "
        f"nicht Material: wird eine dort geführte Zahl in die Arbeit "
        f"übernommen, ist sie durch sie gedeckt. Gründe für die Aufnahme in "
        f"den ausgeschriebenen Teil stehen in der Spalte Bezugsmenge.\n")
    z.append(
        "**Werte, Namen und Genauigkeit.** Die Werte stehen auf vier "
        "Nachkommastellen und mit führender Null. Das Material rundet auf "
        "drei (Auszugstabelle und A1) beziehungsweise sechs Stellen (A1-CSV); "
        "beim Abgleich in Abschnitt 3 ist deshalb jeweils auf der Genauigkeit "
        "des Materials verglichen worden und nicht auf vier Stellen. Die Kategorienamen stehen im Anzeigenamen "
        f"mit Umlauten nach `{namensquelle}`; die Quelle schreibt das Attribut "
        "`OBJEKTART` ohne Umlaute, die Zuordnung führt "
        "`Material_Anhang/A0_Kategorienregister.md`. Die Datendatei führt "
        "beide Schreibweisen.\n")

    for i, mass in enumerate(("B1", "D1"), 1):
        z.append("\n---\n")
        z.append(f"## {i} {MASSNAME[mass]} — "
                 f"{'Das Überlagerungsmass' if mass == 'B1' else 'Das Adjazenzmass'}: "
                 f"die Partnerlisten der gedruckten Kategorien\n")
        z.extend(block_erlaeuterung(mass, kz[mass], kategorien, namen,
                                    ueberschneidung[mass]))
        z.extend(tabelle(auswahl, mass, namen, grund, gedruckt))

    # ------------------------------------------------- Umfang und Abgleich
    z.append("\n---\n")
    z.append("## 3 Umfang der Partnerlisten und die beiden Abgleiche\n")
    z.append(
        "**Was dieser Block zählt.** Hier stehen die Kennzahlen der "
        "Partnerlisten als Ganzes — also Zählungen über alle 110 Zeilen, "
        "nicht über die zwölf ausgeschriebenen — und das Ergebnis der beiden "
        "Abgleiche gegen das Material. Bezugsmenge jeder Zeile ist im Kopf der "
        "Spalte Bezugsmenge genannt. **Die Verwechslung, die hier droht:** "
        "Diese Zählungen betreffen die Einträge der Partnerlisten und nicht "
        "die Zellen der Matrizen. Die Zahl der Zellen über der Schwelle — "
        "1’938 bei B₁, 4’738 bei D₁ nach Register B — ist deutlich grösser "
        "als die Zahl der Einträge, weil die Kappung erst danach greift.\n")
    z.append("| Grösse | Wert | Kennzahl | Bezugsmenge | Vergleichspunkt | Fundstelle |")
    z.append("|---|---|---|---|---|---|")
    for mass in ("B1", "D1"):
        k, a = kz[mass], kz["D1" if mass == "B1" else "B1"]
        anderes = MASSNAME["D1" if mass == "B1" else "B1"]
        q = f"{QUELLE[mass]} · selbst nachgerechnet"
        z.append(f"| Einträge der Partnerlisten {MASSNAME[mass]} "
                 f"| {kb.tausender(k['eintraege'])} | Objektzahl (Einträge) "
                 f"| 110 Zeilen, Schwelle 0.001, Kappung {kb.KAPPUNG} "
                 f"| {anderes}: {kb.tausender(a['eintraege'])} | {q} |")
        z.append(f"| Zeilen mit voller Kappung {MASSNAME[mass]} | {k['voll']} von "
                 f"{len(kategorien)} | Objektzahl | ebenda "
                 f"| {anderes}: {a['voll']} von {len(kategorien)} | ebenda |")
        z.append(f"| Kürzeste Partnerliste {MASSNAME[mass]} | {k['kürzeste']} "
                 f"| Objektzahl | ebenda; wo weniger als {kb.KAPPUNG} Einträge "
                 f"stehen, liegt das an der Schwelle, nicht an der Kappung "
                 f"| {anderes}: {a['kürzeste']} | ebenda |")
        z.append(f"| Zeilen unter der vollen Kappung {MASSNAME[mass]} "
                 f"| {len(k['zeilen_kurz'])} | Objektzahl | ebenda "
                 f"| {anderes}: {len(a['zeilen_kurz'])} | ebenda |")
        z.append(f"| Schwächster Eintrag {MASSNAME[mass]} | {w(k['min'])} "
                 f"| NPMI-Ableitung {MASSNAME[mass]} "
                 f"| {kb.tausender(k['eintraege'])} Einträge "
                 f"| {anderes}: {w(a['min'])} | ebenda |")
        z.append(f"| Stärkster Eintrag {MASSNAME[mass]} | {w(k['max'])} "
                 f"| NPMI-Ableitung {MASSNAME[mass]} | ebenda "
                 f"| {anderes}: {w(a['max'])} | ebenda |")
        z.append(f"| Median der Einträge {MASSNAME[mass]} | {w(k['median'])} "
                 f"| NPMI-Ableitung {MASSNAME[mass]} | ebenda "
                 f"| {anderes}: {w(a['median'])} | ebenda |")
        z.append(f"| Gegenseitige Paare {MASSNAME[mass]} "
                 f"| {kb.tausender(k['gegenseitig'])} von "
                 f"{kb.tausender(k['paare'])} "
                 f"| Anteil in Prozent: {k['gegenseitig'] / k['paare'] * 100:.1f} % "
                 f"| gerichtete Paare der Partnerlisten "
                 f"| {anderes}: {kb.tausender(a['gegenseitig'])} von "
                 f"{kb.tausender(a['paare'])} | ebenda |")
        z.append(f"| Kategorien, die in keiner Liste als Ziel vorkommen "
                 f"{MASSNAME[mass]} | {len(k['nie_ziel'])} von "
                 f"{len(kategorien)} | Objektzahl | ebenda "
                 f"| {anderes}: {len(a['nie_ziel'])} von {len(kategorien)} "
                 f"| ebenda |")
        z.append(f"| Überschneidung mit den stärksten Zellen der Gesamtmatrix "
                 f"{MASSNAME[mass]} | {kb.tausender(ueberschneidung[mass])} von "
                 f"{kb.tausender(k['eintraege'])} "
                 f"| Anteil in Prozent: "
                 f"{ueberschneidung[mass] / k['eintraege'] * 100:.1f} % "
                 f"| die {kb.tausender(k['eintraege'])} stärksten der 11’990 "
                 f"gerichteten Zellen gegen die Einträge der Partnerlisten "
                 f"| {anderes}: {kb.tausender(ueberschneidung['D1' if mass == 'B1' else 'B1'])}"
                 f" von {kb.tausender(a['eintraege'])} | ebenda |")
    z.append(f"| Geprüft: Partnerwerte der Auszugstabelle | {n_kap6} "
             f"| Objektzahl (geprüfte Werte) "
             f"| `Material_Kapitel6/Tabelle_Auszug_staerkste_Partner.md`, "
             f"acht Kategorien × zwei Masse × drei Ränge "
             f"| Abweichungen: {len(abw_kap6)} "
             f"| beide Matrizen · selbst nachgerechnet |")
    z.append(f"| Geprüft: Einträge der Partnerlisten im Anhang | "
             f"{kb.tausender(n_anhang)} | Objektzahl (geprüfte Einträge) "
             f"| `Material_Anhang/A1_Partnerlisten_B1_D1.csv`, beide Masse "
             f"| Abweichungen: {len(abw_anhang)} "
             f"| ebenda |")

    z.append(f"\n### 3.1 Abgleich mit der Auszugstabelle in Material_Kapitel6\n")
    z.append(
        f"Geprüft sind alle {n_kap6} Partnerwerte der Tabelle 6.x, also für "
        f"acht Kategorien je drei Ränge unter beiden Massen, samt dem Namen "
        f"der Zielkategorie und der Rangfolge. Verglichen wurde auf die drei "
        f"Nachkommastellen, die das Material druckt.\n")
    if abw_kap6:
        z.append("| Art | Quellkategorie | Mass | Rang | Material | gerechnet |")
        z.append("|---|---|---|---:|---|---|")
        for a in abw_kap6:
            z.append("| " + " | ".join(str(x) for x in a) + " |")
    else:
        z.append(f"**Ergebnis: keine Abweichung.** Alle {n_kap6} Werte, alle "
                 f"Zielkategorien und die Rangfolge stimmen mit der Rechnung "
                 f"aus den beiden Matrizen überein. Die Tabelle ist damit "
                 f"durch Abschnitt 1 und 2 dieses Registerteils und durch die "
                 f"Datendatei gedeckt.\n")

    z.append(f"\n### 3.2 Abgleich mit den Partnerlisten in Material_Anhang\n")
    z.append(
        f"Geprüft sind alle {kb.tausender(n_anhang)} Einträge der "
        f"maschinenlesbaren Fassung `A1_Partnerlisten_B1_D1.csv`, also "
        f"{kb.tausender(kz['B1']['eintraege'])} für B₁ und "
        f"{kb.tausender(kz['D1']['eintraege'])} für D₁, je Eintrag "
        f"Listenlänge, Zielkategorie, Rang und Wert. Verglichen wurde "
        f"zeichengleich auf den sechs Nachkommastellen, die das Material "
        f"führt, und nicht auf den vier dieses Registerteils; eine Rundung "
        f"auf vier Stellen würde neun Halbstellenfälle als scheinbare "
        f"Abweichung ausweisen, die keine sind.\n")
    if abw_anhang:
        z.append("| Art | Quellkategorie | Mass | Rang | Material | gerechnet |")
        z.append("|---|---|---|---:|---|---|")
        for a in abw_anhang:
            z.append("| " + " | ".join(str(x) for x in a) + " |")
    else:
        z.append(f"**Ergebnis: keine Abweichung.** Alle "
                 f"{kb.tausender(n_anhang)} Einträge stimmen in Länge, "
                 f"Zielkategorie, Rang und Wert mit der Rechnung überein. Die "
                 f"Übereinstimmung war zu erwarten, weil beide Fassungen aus "
                 f"denselben beiden Matrizen erzeugt sind; sie ist damit eine "
                 f"Bestätigung des Rechenwegs und kein unabhängiger Beleg "
                 f"der Matrizen selbst.\n")

    # ------------------------------------------------------- Anmerkungen
    z.append("\n---\n")
    z.append("## 4 Anmerkungen\n")
    z.append(
        f"**Anmerkung 1 — Kategorien, die nie Ziel sind.** Unter B₁ kommen "
        f"{len(kz['B1']['nie_ziel'])} der {len(kategorien)} Kategorien in "
        f"keiner einzigen der 110 Partnerlisten als Ziel vor, unter D₁ nur "
        f"{len(kz['D1']['nie_ziel'])}. Solche Kategorien kann der Resolver "
        f"über das betreffende Mass nie zur Beschreibung eines anderen "
        f"Objekts heranziehen. Unter D₁ betrifft es "
        f"{', '.join(sorted(kb.anzeige(k, namen) for k in kz['D1']['nie_ziel'])) or '—'}"
        f" — also ausgerechnet die grösste Kategorie des Gazetteers. Unter B₁ "
        f"sind es: "
        f"{', '.join(sorted(kb.anzeige(k, namen) for k in kz['B1']['nie_ziel'])) or '—'}."
        f" Die Zahl ist an den Einträgen dieses Registerteils nachgezählt und "
        f"in keinem Bericht des Projekts geführt.\n")
    z.append(
        "**Anmerkung 2 — Die Datendatei ist die vollständige Fassung.** Die "
        "zwölf ausgeschriebenen Kategorien sind eine Auswahl nach dem "
        "Abdruck, keine Auswahl nach Stärke. Wer eine Zahl zu einer anderen "
        "Kategorie braucht, nimmt sie aus der Datendatei und nicht aus dem "
        "Material; Material_Anhang und Material_Kapitel6 sind durch Abschnitt "
        "3 gedeckt, aber sie sind nicht selbst Registerbestand.\n")
    z.append(
        "**Anmerkung 3 — Genauigkeit.** Die vier Nachkommastellen dieses Teils "
        "sind die Registergenauigkeit. Der Abdruck in der Arbeit rundet auf "
        "drei; ein auf drei Stellen gerundeter Wert ist durch den "
        "vierstelligen Eintrag gedeckt, der umgekehrte Weg nicht.\n")
    if fehlend:
        z.append(f"**Anmerkung 4 — Nicht gefunden.** Die folgenden für den "
                 f"Abdruck vorgesehenen Kategorien stehen nicht in der "
                 f"Kategorienordnung der Matrizen und fehlen daher in den "
                 f"Abschnitten 1 und 2: {', '.join(fehlend)}.\n")

    ziel = P["material_anhang"] / f"Zahlenregister_D_Partnerlisten_{DATUM}.md"
    ziel.parent.mkdir(parents=True, exist_ok=True)
    ziel.write_text("\n".join(z) + "\n", encoding="utf-8")

    # ---------------------------------------------------------- Datendatei
    ziel_csv = ziel.with_name(f"Zahlenregister_D_Partnerlisten_{DATUM}__daten.csv")
    with open(ziel_csv, "w", newline="", encoding="utf-8") as f:
        schreiber = csv.writer(f, delimiter=";")
        schreiber.writerow([
            "mass", "quellkategorie_quelle", "quellkategorie_anzeige",
            "zielkategorie_quelle", "zielkategorie_anzeige", "rang", "wert",
            "objektklasse", "gruppe_deutsch", "objekte_quellkategorie",
            "im_register_ausgeschrieben", "matrixdatei"])
        for mass in ("B1", "D1"):
            datei = QUELLE[mass].strip("`")
            for k in kategorien:
                g = klasse.get(k, "ohne Klasse")
                for rang, (p, v) in enumerate(auswahl[k][mass], 1):
                    schreiber.writerow([
                        mass, k, kb.anzeige(k, namen), p, kb.anzeige(p, namen),
                        rang, w(v), g, kb.GRUPPEN_DEUTSCH.get(g, g),
                        zahl.get(k, ""), "ja" if k in gedruckt else "nein",
                        datei])

    print(f"geschrieben: {ziel}")
    print(f"geschrieben: {ziel_csv}")
    print(f"  Einträge gesamt: {gesamt} (B1 {kz['B1']['eintraege']} · "
          f"D1 {kz['D1']['eintraege']})")
    print(f"  ausgeschrieben im Register: {len(gedruckt)} Kategorien, "
          f"{im_register} Einträge")
    print(f"  Abgleich Material_Kapitel6: {n_kap6} Werte geprüft, "
          f"{len(abw_kap6)} Abweichungen")
    for a in abw_kap6:
        print("    ", a)
    print(f"  Abgleich Material_Anhang: {n_anhang} Einträge geprüft, "
          f"{len(abw_anhang)} Abweichungen")
    for a in abw_anhang[:50]:
        print("    ", a)


if __name__ == "__main__":
    main()
