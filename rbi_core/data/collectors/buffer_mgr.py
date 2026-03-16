"""rbi_core/data/collectors/buffer_mgr.py — SQLite WAL tick buffer + atomic state writer."""
import sqlite3
import threading
import tempfile
import os
import time
from collections import deque
from typing import Optional


class TickBuffer:
    """
    High-throughput tick data storage using SQLite WAL mode.
    Batches writes from an in-memory ring buffer.
    Thread-safe: multiple readers (combiner, RL agent) never block writers.
    """

    def __init__(self, db_path: str, flush_interval_ms: int = 500, flush_batch_size: int = 100):
        self.db_path = db_path
        self.flush_interval_s = flush_interval_ms / 1000.0
        self.flush_batch_size = flush_batch_size
        self._buffer: deque = deque()
        self._lock = threading.Lock()
        self._running = False
        self._flush_thread: Optional[threading.Thread] = None

        # Initialize DB with WAL mode
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")  # Safe with WAL, faster writes
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                volume REAL NOT NULL,
                bid REAL,
                ask REAL,
                atr REAL,
                raw_json TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ticks_ts ON ticks(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ticks_symbol ON ticks(symbol)")
        conn.commit()
        conn.close()

    def append_tick(self, tick: dict) -> None:
        """Add tick to in-memory ring buffer. Non-blocking."""
        with self._lock:
            self._buffer.append(tick)

    def start(self) -> None:
        """Start the background flush thread."""
        self._running = True
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

    def stop(self) -> None:
        """Stop flush thread and drain remaining buffer."""
        self._running = False
        if self._flush_thread:
            self._flush_thread.join(timeout=5.0)
        self._flush_to_db()  # Final drain

    def _flush_loop(self) -> None:
        while self._running:
            time.sleep(self.flush_interval_s)
            self._flush_to_db()

    def _flush_to_db(self) -> None:
        # Snapshot the buffer under lock, then write outside lock
        with self._lock:
            if not self._buffer:
                return
            batch = list(self._buffer)
            self._buffer.clear()

        conn = sqlite3.connect(self.db_path)
        try:
            conn.executemany(
                "INSERT INTO ticks (timestamp, symbol, price, volume, bid, ask, atr, raw_json) "
                "VALUES (:timestamp, :symbol, :price, :volume, :bid, :ask, :atr, :raw_json)",
                batch
            )
            conn.commit()
        finally:
            conn.close()

    def query_recent(self, symbol: str, limit: int = 500) -> list:
        """Read recent ticks. Safe to call from any thread (WAL readers never block)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM ticks WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
                (symbol, limit)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


class AtomicStateWriter:
    """
    Atomic file writer for small state blobs (orderbook snapshot, active params).
    Uses the .tmp rename pattern. NOT for high-throughput tick data.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._lock = threading.Lock()

    def write(self, content: str) -> None:
        with self._lock:
            fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(self.filepath) or '.')
            try:
                with os.fdopen(fd, 'w') as f:
                    f.write(content)
                os.replace(tmp_path, self.filepath)
            except Exception:
                # Clean up temp file on failure
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise

    def read(self) -> Optional[str]:
        if not os.path.exists(self.filepath):
            return None
        with open(self.filepath, 'r') as f:
            return f.read()
