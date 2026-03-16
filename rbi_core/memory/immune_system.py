"""ToxicityTagger — marks failed Strategy DNA as Toxic-DNA in ChromaDB.
ToxicityDecay — chronological 180-day decay of toxicity tags (Story 3.4).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from rbi_core.memory.dna_hasher import DNAHasher

logger = logging.getLogger(__name__)


class ToxicityTagger:
    """Tags failed Strategy DNA as 'toxic' in the ChromaDB vector store."""

    def __init__(self, vector_store) -> None:
        self._store = vector_store

    def tag(self, dna: dict, embedding: list[float]) -> str:
        """Mark *dna* as Toxic-DNA in the vector store.

        Args:
            dna:       Strategy DNA dict.
            embedding: Pre-computed DNA embedding vector.

        Returns:
            SHA-256 hash of the DNA (used as the ChromaDB document ID).
        """
        self._store.store(dna=dna, embedding=embedding, label="toxic")
        dna_hash = DNAHasher.hash(dna)
        logger.warning(
            "toxic_dna_tagged",
            extra={"event": "toxic_dna_tagged", "hash": dna_hash},
        )
        return dna_hash

    def is_blocked(self, embedding: list[float]) -> bool:
        """Return True if the given embedding is similar to a known Toxic-DNA."""
        return self._store.is_toxic(embedding=embedding)


class ToxicityDecay:
    """Determines whether a Toxic-DNA tag has expired."""

    @staticmethod
    def is_expired(tagged_at: datetime, decay_days: int = 180) -> bool:
        """Return True if the toxicity tag is older than *decay_days*.

        Args:
            tagged_at:  UTC datetime when the DNA was tagged toxic.
            decay_days: Days after which the tag expires (default: 180).

        Returns:
            True if ``(now - tagged_at).days >= decay_days``.
        """
        elapsed = datetime.now(timezone.utc) - tagged_at
        return elapsed.days >= decay_days
