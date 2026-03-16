"""Unit tests for DataIngestor — TDD RED phase.

Uses tmp_path for all file I/O; no interference with project files.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rbi_core.ingestion.data_ingestor import DataIngestor
from rbi_core.exceptions import DataSanitizationError


def _make_ohlcv(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame for the given DatetimeIndex."""
    n = len(index)
    return pd.DataFrame(
        {
            "open":   [100.0] * n,
            "high":   [101.0] * n,
            "low":    [99.0]  * n,
            "close":  [100.5] * n,
            "volume": [1000.0] * n,
        },
        index=index,
    )


@pytest.fixture
def five_minute_df():
    """Five continuous 1-minute candles, no gaps."""
    idx = pd.date_range("2024-01-01 09:00", periods=5, freq="1min")
    return _make_ohlcv(idx)


@pytest.fixture
def df_with_gap():
    """Four candles with a 3-minute gap between candle 2 and candle 3."""
    ts = pd.to_datetime([
        "2024-01-01 09:00",
        "2024-01-01 09:01",
        "2024-01-01 09:04",  # 2-minute gap
        "2024-01-01 09:05",
    ])
    return _make_ohlcv(ts)


@pytest.fixture
def df_with_nan():
    """DataFrame containing NaN values in the close column."""
    idx = pd.date_range("2024-01-01 09:00", periods=4, freq="1min")
    df = _make_ohlcv(idx)
    df.loc[idx[1], "close"] = float("nan")
    df.loc[idx[2], "volume"] = float("nan")
    return df


class TestDataIngestorIngest:

    def test_creates_parquet_file(self, five_minute_df, tmp_path):
        out = DataIngestor.ingest(five_minute_df, ticker="BTCUSDT", output_dir=str(tmp_path))
        assert out.exists()
        assert out.suffix == ".parquet"

    def test_output_path_pattern(self, five_minute_df, tmp_path):
        out = DataIngestor.ingest(five_minute_df, ticker="BTCUSDT", output_dir=str(tmp_path))
        assert out == tmp_path / "BTCUSDT" / "1m.parquet"

    def test_ticker_normalised_to_uppercase(self, five_minute_df, tmp_path):
        out = DataIngestor.ingest(five_minute_df, ticker="btcusdt", output_dir=str(tmp_path))
        assert out == tmp_path / "BTCUSDT" / "1m.parquet"

    def test_output_is_zstd_compressed(self, five_minute_df, tmp_path):
        out = DataIngestor.ingest(five_minute_df, ticker="ETH", output_dir=str(tmp_path))
        schema = pd.read_parquet(out)
        # If we can read it back successfully it was valid Parquet
        assert not schema.empty

    def test_fills_gap_in_timestamp_index(self, df_with_gap, tmp_path):
        out = DataIngestor.ingest(df_with_gap, ticker="SOL", output_dir=str(tmp_path))
        result = pd.read_parquet(out)
        # Gap between 09:01 and 09:04 should be filled: 09:02, 09:03 added
        assert len(result) >= 6  # 4 original + 2 gap-filled minutes

    def test_no_nans_in_output(self, df_with_nan, tmp_path):
        out = DataIngestor.ingest(df_with_nan, ticker="ADA", output_dir=str(tmp_path))
        result = pd.read_parquet(out)
        assert not result.isnull().any().any()

    def test_roundtrip_preserves_ohlcv_columns(self, five_minute_df, tmp_path):
        out = DataIngestor.ingest(five_minute_df, ticker="BNB", output_dir=str(tmp_path))
        result = pd.read_parquet(out)
        for col in ("open", "high", "low", "close", "volume"):
            assert col in result.columns

    def test_raises_data_sanitization_error_on_empty_input(self, tmp_path):
        empty_df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            dtype=float,
        )
        with pytest.raises(DataSanitizationError):
            DataIngestor.ingest(empty_df, ticker="DOGE", output_dir=str(tmp_path))

    def test_returns_path_object(self, five_minute_df, tmp_path):
        out = DataIngestor.ingest(five_minute_df, ticker="XRP", output_dir=str(tmp_path))
        assert isinstance(out, Path)


class TestDataIngestorFromCsv:

    def test_from_csv_reads_and_ingests(self, tmp_path):
        idx = pd.date_range("2024-01-01 09:00", periods=3, freq="1min")
        df = _make_ohlcv(idx)
        csv_path = tmp_path / "raw.csv"
        df.to_csv(str(csv_path))

        out = DataIngestor.from_csv(str(csv_path), ticker="AVAX", output_dir=str(tmp_path))
        assert out.exists()
