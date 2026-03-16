"""DNAHasher — SHA-256 fingerprint for Strategy DNA dicts.
ChromaVectorStore — thin ChromaDB collection wrapper with toxicity query.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class DNAHasher:
    """Deterministic, key-order-independent SHA-256 hash of Strategy DNA."""

    @staticmethod
    def hash(dna: dict) -> str:
        """Return a 64-char hex SHA-256 hash of *dna*.

        Keys are sorted so ``{"a":1,"b":2}`` == ``{"b":2,"a":1}``.
        """
        canonical = json.dumps(dna, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


class ChromaVectorStore:
    """Thin wrapper around a ChromaDB collection for DNA vector operations."""

    def __init__(self, collection) -> None:
        self._col = collection

    def store(
        self,
        dna: dict,
        embedding: list[float],
        label: str = "valid",
    ) -> None:
        """Add a DNA entry to the ChromaDB collection.

        Args:
            dna:       Strategy DNA dict (used to derive the hash ID).
            embedding: Pre-computed L2-normalised float vector.
            label:     ``"valid"`` or ``"toxic"``.
        """
        doc_id = DNAHasher.hash(dna)
        self._col.add(
            embeddings=[embedding],
            documents=[json.dumps(dna, sort_keys=True)],
            metadatas=[{"hash": doc_id, "label": label}],
            ids=[doc_id],
        )
        logger.info(
            "dna_stored",
            extra={"event": "dna_stored", "hash": doc_id, "label": label},
        )

    def query_similar(
        self,
        embedding: list[float],
        n_results: int = 5,
    ) -> dict:
        """Return the *n_results* most similar DNA entries."""
        return self._col.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas"],
        )

    def is_toxic(self, embedding: list[float], n_results: int = 1) -> bool:
        """Return True if the nearest similar entry is labelled 'toxic'."""
        result = self.query_similar(embedding, n_results=n_results)
        metadatas = result.get("metadatas", [[]])
        for meta in (metadatas[0] if metadatas else []):
            if meta.get("label") == "toxic":
                return True
        return False
