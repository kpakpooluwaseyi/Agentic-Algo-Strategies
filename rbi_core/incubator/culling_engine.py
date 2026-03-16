"""CullingEngine — performance-based bot culling (Story 5.3)."""
from __future__ import annotations


class CullingEngine:
    def __init__(
        self, min_profit_factor: float = 1.25, max_drawdown: float = 0.15
    ) -> None:
        self._pf_floor = min_profit_factor
        self._dd_ceil = max_drawdown

    def cull(self, bots: list[dict]) -> list[str]:
        """Return IDs of bots below profit_factor threshold or over drawdown."""
        culled = []
        for bot in bots:
            if (
                bot.get("profit_factor", 0) < self._pf_floor
                or bot.get("drawdown", 0) > self._dd_ceil
            ):
                culled.append(bot["bot_id"])
        return culled
