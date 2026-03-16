"""WALStateManager — SQLite WAL position persistence for Story 4.6."""
from __future__ import annotations

import sqlite3
from pathlib import Path


class WALStateManager:
    def __init__(self, db_path: str) -> None:
        self._db = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    size REAL NOT NULL,
                    entry_price REAL NOT NULL
                )
            """)
            conn.commit()

    def save_position(self, symbol: str, size: float, entry_price: float) -> None:
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO positions (symbol, size, entry_price) VALUES (?, ?, ?)",
                (symbol, size, entry_price),
            )
            conn.commit()

    def load_positions(self) -> list[dict]:
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute(
                "SELECT symbol, size, entry_price FROM positions"
            ).fetchall()
        return [{"symbol": r[0], "size": r[1], "entry_price": r[2]} for r in rows]
