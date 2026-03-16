"""DataIngestor — Parquet ingestion and OHLCV data sanitization.

Fills timestamp gaps (forward-fill), removes NaN values (ffill→bfill),
and writes Zstd-compressed Parquet to ``data/{TICKER}/1m.parquet``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from rbi_core.exceptions import DataSanitizationError

logger = logging.getLogger(__name__)

_OHLCV_COLS = ("open", "high", "low", "close", "volume")


class DataIngestor:
    """Static pipeline for Parquet ingestion and OHLCV sanitation."""

    @staticmethod
    def ingest(
        df: pd.DataFrame,
        ticker: str,
        output_dir: str = "data",
    ) -> Path:
        """Sanitize *df* and persist as Zstd-compressed Parquet.

        Processing pipeline:
        1. Validate the DataFrame is not empty.
        2. Normalize ticker to uppercase.
        3. Reindex to 1-minute frequency, forward-filling timestamp gaps.
        4. Fill remaining NaN values with ``ffill()`` → ``bfill()``.
        5. Write ``{output_dir}/{TICKER}/1m.parquet`` with Zstd compression.

        Args:
            df:         Raw OHLCV DataFrame with a DatetimeIndex.
            ticker:     Asset symbol (e.g. ``"BTCUSDT"``).  Case-insensitive.
            output_dir: Root output directory.  Defaults to ``"data"``.

        Returns:
            ``Path`` of the written Parquet file.

        Raises:
            DataSanitizationError: If the DataFrame is empty after sanitization.
        """
        if df.empty:
            raise DataSanitizationError(
                f"Cannot ingest empty DataFrame for ticker {ticker!r}."
            )

        ticker = ticker.upper()

        # 1 — Ensure DatetimeIndex, then reindex to 1-min grid (fills gaps)
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index(pd.to_datetime(df.index))

        full_idx = pd.date_range(df.index.min(), df.index.max(), freq="1min")
        df = df.reindex(full_idx)

        # 2 — Fill NaN: forward-fill first, then back-fill for leading NaNs
        df = df.ffill().bfill()

        if df.empty:
            raise DataSanitizationError(
                f"DataFrame became empty after sanitization for ticker {ticker!r}."
            )

        # 3 — Write Parquet
        out_dir = Path(output_dir) / ticker
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "1m.parquet"

        df.to_parquet(str(out_path), compression="zstd")

        logger.info(
            "parquet_ingested",
            extra={
                "event": "parquet_ingested",
                "ticker": ticker,
                "rows": len(df),
                "path": str(out_path),
            },
        )
        return out_path

    @staticmethod
    def from_csv(
        csv_path: str,
        ticker: str,
        output_dir: str = "data",
    ) -> Path:
        """Load a CSV file and ingest it as Parquet.

        The CSV must have a parseable datetime column (index or first column).

        Args:
            csv_path:   Path to the source CSV file.
            ticker:     Asset symbol.
            output_dir: Root output directory.

        Returns:
            ``Path`` of the written Parquet file.
        """
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        return DataIngestor.ingest(df, ticker=ticker, output_dir=output_dir)
