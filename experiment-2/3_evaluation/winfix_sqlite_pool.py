"""
Windows-Fix: geoparser-Engine auf StaticPool umstellen (eine persistente Connection).

geoparser/db/db.py nutzt NullPool — jede Session oeffnet eine neue SQLite-Connection,
und der globale connect-Listener laedt dabei mod_spatialite.dll + spellfix.dll NEU.
Auf Windows leakt jedes Laden der DLLs FLS/TLS-Slots; nach einigen hundert
Connections bricht der Prozess mit abort() ab (silent crash, exit code 3 — vgl.
Session-Notiz 2026-05-09, dort als SpatiaLite/NullPool-Problem diagnostiziert).

Der Patch ersetzt die Modul-Engine durch eine StaticPool-Engine: eine einzige
Connection fuer den ganzen Prozess, Extensions werden genau einmal geladen.
get_session()/get_connection() lesen die Engine zur Laufzeit aus dem Modul
(laut Docstring explizit patch-freundlich), daher reicht der Austausch hier.

Verwendung: `import winfix_sqlite_pool; winfix_sqlite_pool.apply()` vor der
ersten Gazetteer-Nutzung (passiert in eval_core beim Import).
"""

from __future__ import annotations

import sys

from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine

import geoparser.db.db as _gdb

_applied = False


def apply() -> None:
    global _applied
    if _applied or not sys.platform.startswith("win"):
        return
    _gdb.engine.dispose()
    _gdb.engine = create_engine(
        _gdb.DATABASE_URL,
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    _applied = True
