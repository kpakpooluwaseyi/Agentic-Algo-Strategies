"""IncubatorFactory — spawns/kills BotWorker instances (Story 5.2)."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BotWorker:
    bot_id: str
    dna: dict
    capital: float


class IncubatorFactory:
    def __init__(self, max_bots: int = 50) -> None:
        self._max = max_bots
        self._bots: dict[str, BotWorker] = {}

    @property
    def active_count(self) -> int:
        return len(self._bots)

    def spawn(self, dna: dict, capital: float) -> BotWorker:
        if self.active_count >= self._max:
            raise RuntimeError(f"Incubator at capacity ({self._max} bots)")
        bot = BotWorker(bot_id=str(uuid.uuid4()), dna=dna, capital=capital)
        self._bots[bot.bot_id] = bot
        return bot

    def kill(self, bot_id: str) -> None:
        self._bots.pop(bot_id, None)
