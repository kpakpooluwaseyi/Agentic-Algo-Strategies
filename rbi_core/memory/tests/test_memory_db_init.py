"""Unit tests for MemoryDBInitializer — TDD RED phase.

ChromaDB client is mocked; no actual ChromaDB install required.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rbi_core.memory.memory_db_init import MemoryDBInitializer
from rbi_core.exceptions import MemoryDBChecksumError


class TestMemoryDBInitializerInitialize:

    def test_returns_chromadb_client(self, tmp_path):
        with patch("rbi_core.memory.memory_db_init.chromadb") as mock_chroma:
            mock_client = MagicMock()
            mock_client.list_collections.return_value = []
            mock_chroma.PersistentClient.return_value = mock_client
            init = MemoryDBInitializer()
            client = init.initialize(persist_dir=str(tmp_path))
        assert client is mock_client

    def test_creates_manifest_on_first_startup(self, tmp_path):
        with patch("rbi_core.memory.memory_db_init.chromadb") as mock_chroma:
            mock_client = MagicMock()
            mock_client.list_collections.return_value = []
            mock_chroma.PersistentClient.return_value = mock_client
            init = MemoryDBInitializer()
            init.initialize(persist_dir=str(tmp_path))
        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert "collection_count" in manifest

    def test_passes_validation_when_count_matches_manifest(self, tmp_path):
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({"collection_count": 2}))
        with patch("rbi_core.memory.memory_db_init.chromadb") as mock_chroma:
            mock_client = MagicMock()
            mock_client.list_collections.return_value = [MagicMock(), MagicMock()]
            mock_chroma.PersistentClient.return_value = mock_client
            init = MemoryDBInitializer()
            # Must not raise
            init.initialize(persist_dir=str(tmp_path))

    def test_raises_checksum_error_when_count_mismatches(self, tmp_path):
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({"collection_count": 5}))
        with patch("rbi_core.memory.memory_db_init.chromadb") as mock_chroma:
            mock_client = MagicMock()
            mock_client.list_collections.return_value = [MagicMock()]  # 1, not 5
            mock_chroma.PersistentClient.return_value = mock_client
            init = MemoryDBInitializer()
            with pytest.raises(MemoryDBChecksumError, match="collection_count"):
                init.initialize(persist_dir=str(tmp_path))

    def test_error_message_includes_expected_and_actual(self, tmp_path):
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({"collection_count": 3}))
        with patch("rbi_core.memory.memory_db_init.chromadb") as mock_chroma:
            mock_client = MagicMock()
            mock_client.list_collections.return_value = [MagicMock()]
            mock_chroma.PersistentClient.return_value = mock_client
            init = MemoryDBInitializer()
            try:
                init.initialize(persist_dir=str(tmp_path))
            except MemoryDBChecksumError as exc:
                assert "3" in str(exc)  # expected
                assert "1" in str(exc)  # actual
