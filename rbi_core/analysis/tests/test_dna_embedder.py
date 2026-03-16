"""Unit tests for DNAEmbedder — TDD RED phase."""
from __future__ import annotations

import numpy as np
import pytest

from rbi_core.analysis.dna_embedder import DNAEmbedder


class TestDNAEmbedderEmbed:

    def test_output_is_ndarray(self):
        dna = {"fast": 8, "slow": 21, "threshold": 0.01}
        vec = DNAEmbedder.embed(dna)
        assert isinstance(vec, np.ndarray)

    def test_output_is_normalized(self):
        dna = {"fast": 8, "slow": 21, "threshold": 0.01, "rsi_period": 14}
        vec = DNAEmbedder.embed(dna)
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-6

    def test_same_dna_produces_same_embedding(self):
        dna = {"fast": 8, "slow": 21}
        v1 = DNAEmbedder.embed(dna)
        v2 = DNAEmbedder.embed(dna)
        np.testing.assert_array_equal(v1, v2)

    def test_different_dna_produces_different_embeddings(self):
        dna_a = {"fast": 8, "slow": 21}
        dna_b = {"fast": 5, "slow": 13}
        v_a = DNAEmbedder.embed(dna_a)
        v_b = DNAEmbedder.embed(dna_b)
        assert not np.allclose(v_a, v_b)

    def test_output_is_1d(self):
        dna = {"a": 1.0, "b": 2.0, "c": 3.0}
        vec = DNAEmbedder.embed(dna)
        assert vec.ndim == 1

    def test_raises_on_empty_dna(self):
        with pytest.raises(ValueError, match="empty"):
            DNAEmbedder.embed({})
