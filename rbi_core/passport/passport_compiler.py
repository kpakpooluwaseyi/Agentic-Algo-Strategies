"""PassportCompiler — mints Strategy Passports to the Golden Archive.

Writes a JSON file per strategy to the golden directory, never
publishes raw DNA to the stream (URI only).
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class StrategyPassport:
    """Immutable record of a validated strategy's proof-of-edge."""

    passport_id: str
    dna: dict
    ticker: str
    total_return: float
    sharpe: float
    max_drawdown: float
    uri: str
    created_at: datetime


class PassportCompiler:
    """Compiles and archives Strategy Passports to the Golden Archive folder."""

    def __init__(self, golden_dir: str = "golden") -> None:
        self._golden_dir = Path(golden_dir)
        self._golden_dir.mkdir(parents=True, exist_ok=True)

    def mint(
        self,
        dna: dict,
        ticker: str,
        total_return: float,
        sharpe: float,
        max_drawdown: float,
    ) -> StrategyPassport:
        """Mint a new Strategy Passport and persist it to golden_dir.

        Args:
            dna:           Full Strategy DNA dict.
            ticker:        Trading pair symbol.
            total_return:  Backtest total return.
            sharpe:        Sharpe ratio.
            max_drawdown:  Maximum drawdown.

        Returns:
            A populated ``StrategyPassport`` with a unique ID and URI.
        """
        passport_id = str(uuid.uuid4())
        out_path = self._golden_dir / f"{passport_id}.json"
        passport = StrategyPassport(
            passport_id=passport_id,
            dna=dna,
            ticker=ticker,
            total_return=total_return,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            uri=str(out_path),
            created_at=datetime.now(timezone.utc),
        )
        with open(out_path, "w") as fh:
            payload = asdict(passport)
            payload["created_at"] = passport.created_at.isoformat()
            json.dump(payload, fh, indent=2)

        logger.info(
            "passport_minted",
            extra={
                "event": "passport_minted",
                "passport_id": passport_id,
                "ticker": ticker,
                "uri": str(out_path),
            },
        )
        return passport
