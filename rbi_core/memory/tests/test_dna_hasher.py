"""Unit tests for DNAHasher and ChromaVectorStore — TDD RED phase."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rbi_core.memory.dna_hasher import DNAHasher, ChromaVectorStore


DNA_A = {"fast": 8, "slow": 21, "threshold": 0.01}
DNA_B = {"fast": 5, "slow": 13, "threshold": 0.02}


class TestDNAHasher:

    def test_hash_returns_hex_string(self):
        h = DNAHasher.hash(DNA_A)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex = 64 chars

    def test_same_dna_produces_same_hash(self):
        assert DNAHasher.hash(DNA_A) == DNAHasher.hash(DNA_A)

    def test_different_dna_produces_different_hashes(self):
        assert DNAHasher.hash(DNA_A) != DNAHasher.hash(DNA_B)

    def test_hash_is_key_order_independent(self):
        dna1 = {"a": 1, "b": 2}
        dna2 = {"b": 2, "a": 1}
        assert DNAHasher.hash(dna1) == DNAHasher.hash(dna2)


class TestChromaVectorStore:

    def test_store_calls_collection_add(self):
        mock_collection = MagicMock()
        store = ChromaVectorStore(collection=mock_collection)
        store.store(dna=DNA_A, embedding=[0.5, 0.5], label="valid")
        mock_collection.add.assert_called_once()

    def test_store_payload_contains_label_and_hash(self):
        mock_collection = MagicMock()
        store = ChromaVectorStore(collection=mock_collection)
        store.store(dna=DNA_A, embedding=[0.5, 0.5], label="toxic")
        call_kwargs = mock_collection.add.call_args.kwargs
        metadata = call_kwargs["metadatas"][0]
        assert metadata["label"] == "toxic"
        assert "hash" in metadata

    def test_query_similar_called_with_embedding(self):
        mock_collection = MagicMock()
        mock_collection.query.return_value = {"documents": [[]], "metadatas": [[]]}
        store = ChromaVectorStore(collection=mock_collection)
        store.query_similar(embedding=[0.5, 0.5], n_results=5)
        mock_collection.query.assert_called_once()

    def test_is_toxic_returns_true_when_similar_toxic_exists(self):
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["some"]],
            "metadatas": [[{"label": "toxic"}]],
        }
        store = ChromaVectorStore(collection=mock_collection)
        assert store.is_toxic(embedding=[0.5, 0.5]) is True

    def test_is_toxic_returns_false_when_no_toxic_similar(self):
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["some"]],
            "metadatas": [[{"label": "valid"}]],
        }
        store = ChromaVectorStore(collection=mock_collection)
        assert store.is_toxic(embedding=[0.5, 0.5]) is False
