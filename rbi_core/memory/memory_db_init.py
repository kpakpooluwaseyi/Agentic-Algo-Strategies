"""MemoryDBInitializer — ChromaDB startup with checksum validation.

Reads ``{persist_dir}/manifest.json`` on startup; creates it on first
run. Raises ``MemoryDBChecksumError`` if collection count mismatches.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import chromadb

from rbi_core.exceptions import MemoryDBChecksumError

logger = logging.getLogger(__name__)

_MANIFEST_FILENAME = "manifest.json"


class MemoryDBInitializer:
    """Handles ChromaDB startup and pre-flight checksum integrity check."""

    def initialize(self, persist_dir: str) -> "chromadb.PersistentClient":
        """Connect to ChromaDB, validate integrity, and return the client.

        On first startup, creates a ``manifest.json`` with the current
        collection count.  On subsequent startups, compares the actual
        collection count against the manifest.

        Args:
            persist_dir: Path to the ChromaDB persistence directory.

        Returns:
            An initialized ``chromadb.PersistentClient``.

        Raises:
            MemoryDBChecksumError: If the collection count does not match
                the stored manifest.
        """
        persist_path = Path(persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        manifest_path = persist_path / _MANIFEST_FILENAME

        client = chromadb.PersistentClient(path=str(persist_path))
        actual_count = len(client.list_collections())

        if not manifest_path.exists():
            self._write_manifest(manifest_path, actual_count)
            logger.info(
                "memory_db_manifest_created",
                extra={
                    "event": "memory_db_manifest_created",
                    "persist_dir": str(persist_dir),
                    "collection_count": actual_count,
                },
            )
        else:
            self._validate_manifest(manifest_path, actual_count)

        return client

    # ------------------------------------------------------------------

    @staticmethod
    def _write_manifest(path: Path, count: int) -> None:
        path.write_text(json.dumps({"collection_count": count}))

    @staticmethod
    def _validate_manifest(path: Path, actual_count: int) -> None:
        manifest = json.loads(path.read_text())
        expected = manifest.get("collection_count", -1)
        if actual_count != expected:
            raise MemoryDBChecksumError(
                f"ChromaDB collection_count mismatch: "
                f"expected {expected}, actual {actual_count}. "
                "DB may be corrupted — halting generation cycle."
            )
