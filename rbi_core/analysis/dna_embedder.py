"""DNAEmbedder — encodes Strategy DNA dicts into normalized L2 vectors.

Extracts numeric fields in sorted-key order, builds a raw feature
vector, and normalizes to unit L2 norm for ChromaDB similarity search.
"""
from __future__ import annotations

import numpy as np


class DNAEmbedder:
    """Converts a Strategy DNA dict into a normalized numpy embedding."""

    @staticmethod
    def embed(dna: dict) -> np.ndarray:
        """Embed *dna* into a unit-norm float64 vector.

        Arguments are extracted in sorted-key order for determinism.

        Args:
            dna: Strategy DNA dictionary of numeric fields.

        Returns:
            1-D float64 ndarray of unit L2 norm.

        Raises:
            ValueError: If *dna* is empty or contains no numeric values.
        """
        if not dna:
            raise ValueError("Cannot embed an empty DNA dict.")

        numeric_values = [
            float(v)
            for k, v in sorted(dna.items())
            if isinstance(v, (int, float))
        ]

        if not numeric_values:
            raise ValueError("DNA dict contains no numeric values to embed.")

        vec = np.array(numeric_values, dtype=np.float64)
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            return vec
        return vec / norm
