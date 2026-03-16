"""rbi_core/data/strategy_metrics.py — Persistent per-strategy performance metrics.

SQLite table 'strategy_metrics' tracking regime-tagged performance stats.
Used by StrategyCombiner for auto-weighting and RL agent for fitness scoring.

Thread-safe: written by trade close callback, read by combiner/forge.
"""
import sqlite3
import threading
import time
from typing import Optional


class StrategyMetrics:
    """
    Persistent strategy performance tracking in SQLite.

    Records per-trade results keyed by (strategy, regime) and maintains
    running statistics: Sortino, winrate, max drawdown, trade count.

    Reuses the same database as TickBuffer for simplicity.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS strategy_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL,
                regime TEXT NOT NULL,
                trade_pnl REAL NOT NULL,
                cumulative_pnl REAL NOT NULL DEFAULT 0.0,
                trade_count INTEGER NOT NULL DEFAULT 0,
                win_count INTEGER NOT NULL DEFAULT 0,
                peak_pnl REAL NOT NULL DEFAULT 0.0,
                max_drawdown REAL NOT NULL DEFAULT 0.0,
                last_updated REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS strategy_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL,
                regime TEXT NOT NULL,
                pnl REAL NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sm_strat ON strategy_metrics(strategy)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sm_regime ON strategy_metrics(strategy, regime)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_st_strat ON strategy_trades(strategy)")
        conn.commit()
        conn.close()

    def record_trade(self, strategy: str, regime: str, pnl: float) -> None:
        """
        Record a closed trade result. Updates running stats for (strategy, regime).

        Args:
            strategy: Strategy name.
            regime: Market regime at time of trade ("trending", "ranging", etc.).
            pnl: Realized PnL of the trade.
        """
        now = time.time()
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                # Insert raw trade record
                conn.execute(
                    "INSERT INTO strategy_trades (strategy, regime, pnl, timestamp) "
                    "VALUES (?, ?, ?, ?)",
                    (strategy, regime, pnl, now),
                )

                # Upsert running metrics
                row = conn.execute(
                    "SELECT cumulative_pnl, trade_count, win_count, peak_pnl, max_drawdown "
                    "FROM strategy_metrics WHERE strategy = ? AND regime = ?",
                    (strategy, regime),
                ).fetchone()

                if row:
                    cum_pnl = row[0] + pnl
                    count = row[1] + 1
                    wins = row[2] + (1 if pnl > 0 else 0)
                    peak = max(row[3], cum_pnl)
                    dd = min(row[4], cum_pnl - peak)
                    conn.execute(
                        "UPDATE strategy_metrics SET trade_pnl=?, cumulative_pnl=?, "
                        "trade_count=?, win_count=?, peak_pnl=?, max_drawdown=?, last_updated=? "
                        "WHERE strategy=? AND regime=?",
                        (pnl, cum_pnl, count, wins, peak, dd, now, strategy, regime),
                    )
                else:
                    peak = max(0.0, pnl)
                    dd = min(0.0, pnl - peak)
                    conn.execute(
                        "INSERT INTO strategy_metrics "
                        "(strategy, regime, trade_pnl, cumulative_pnl, trade_count, "
                        "win_count, peak_pnl, max_drawdown, last_updated) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (strategy, regime, pnl, pnl, 1, 1 if pnl > 0 else 0,
                         peak, dd, now),
                    )
                conn.commit()
            finally:
                conn.close()

    def get_metrics(self, strategy: Optional[str] = None) -> list[dict]:
        """
        Get performance metrics.

        Args:
            strategy: Filter by strategy name. None = all strategies.

        Returns:
            List of dicts with keys: strategy, regime, cumulative_pnl, trade_count,
            winrate, max_drawdown, sortino, last_updated.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            if strategy:
                rows = conn.execute(
                    "SELECT * FROM strategy_metrics WHERE strategy = ?",
                    (strategy,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM strategy_metrics").fetchall()

            results = []
            for r in rows:
                r_dict = dict(r)
                count = r_dict.get('trade_count', 0)
                wins = r_dict.get('win_count', 0)
                r_dict['winrate'] = wins / count if count > 0 else 0.0

                # Compute Sortino from trade history
                strat = r_dict['strategy']
                regime = r_dict['regime']
                r_dict['sortino'] = self._compute_sortino(conn, strat, regime)
                results.append(r_dict)

            return results
        finally:
            conn.close()

    def get_top_strategies(self, regime: Optional[str] = None,
                           metric: str = "cumulative_pnl",
                           limit: int = 5) -> list[dict]:
        """Get top N strategies ranked by a metric, optionally filtered by regime."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            if regime:
                rows = conn.execute(
                    f"SELECT * FROM strategy_metrics WHERE regime = ? "
                    f"ORDER BY {metric} DESC LIMIT ?",
                    (regime, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT strategy, SUM(cumulative_pnl) as cumulative_pnl, "
                    f"SUM(trade_count) as trade_count, MIN(max_drawdown) as max_drawdown "
                    f"FROM strategy_metrics GROUP BY strategy "
                    f"ORDER BY cumulative_pnl DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def _compute_sortino(self, conn, strategy: str, regime: str,
                         window: int = 20) -> float:
        """Compute Sortino ratio from recent trade PnLs."""
        rows = conn.execute(
            "SELECT pnl FROM strategy_trades "
            "WHERE strategy = ? AND regime = ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (strategy, regime, window),
        ).fetchall()

        if len(rows) < 2:
            return 0.0

        pnls = [r[0] for r in rows]
        mean_return = sum(pnls) / len(pnls)
        downside = [min(0, p) for p in pnls]
        downside_sq = sum(d * d for d in downside) / len(downside)

        if downside_sq <= 0:
            return mean_return * 10.0 if mean_return > 0 else 0.0

        import math
        downside_dev = math.sqrt(downside_sq)
        return mean_return / downside_dev
