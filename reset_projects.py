"""
Alle Geoparser-Projekte auflisten und loeschen.

Verwendung:
    poetry run python reset_projects.py
"""

from geoparser import Project
from geoparser.db.db import get_session
from geoparser.db.crud import ProjectRepository


def list_and_delete_all():
    with get_session() as session:
        projects = ProjectRepository.get_all(session)

        if not projects:
            print("Keine Projekte gefunden.")
            return

        print(f"{len(projects)} Projekt(e) gefunden:")
        for p in projects:
            print(f"  - {p.name}  (id={p.id})")

    # Delete via Project API (handles cascade)
    for p in projects:
        proj = Project(name=p.name)
        proj.delete()
        print(f"  -> Geloescht: {p.name}")

    print("Fertig.")


if __name__ == "__main__":
    list_and_delete_all()
