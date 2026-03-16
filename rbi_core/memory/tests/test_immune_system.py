"""Unit tests for ToxicityTagger and ToxicityDecay — TDD RED phase (Stories 3.3 + 3.4)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from rbi_core.memory.immune_system import ToxicityTagger, ToxicityDecay


class TestToxicityTagger:

    def test_tag_calls_store_with_toxic_label(self):
        mock_store = MagicMock()
        tagger = ToxicityTagger(vector_store=mock_store)
        tagger.tag(dna={"fast": 8}, embedding=[0.5, 0.5])
        mock_store.store.assert_called_once_with(
            dna={"fast": 8}, embedding=[0.5, 0.5], label="toxic"
        )

    def test_tag_returns_dna_hash(self):
        mock_store = MagicMock()
        tagger = ToxicityTagger(vector_store=mock_store)
        result = tagger.tag(dna={"fast": 8}, embedding=[0.5, 0.5])
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex

    def test_is_blocked_returns_true_when_store_reports_toxic(self):
        mock_store = MagicMock()
        mock_store.is_toxic.return_value = True
        tagger = ToxicityTagger(vector_store=mock_store)
        assert tagger.is_blocked(embedding=[0.5, 0.5]) is True

    def test_is_blocked_returns_false_when_not_toxic(self):
        mock_store = MagicMock()
        mock_store.is_toxic.return_value = False
        tagger = ToxicityTagger(vector_store=mock_store)
        assert tagger.is_blocked(embedding=[0.3, 0.7]) is False


class TestToxicityDecay:

    def test_is_expired_true_when_older_than_decay_days(self):
        old_ts = datetime.now(timezone.utc) - timedelta(days=200)
        assert ToxicityDecay.is_expired(tagged_at=old_ts, decay_days=180) is True

    def test_is_expired_false_when_within_decay_days(self):
        recent_ts = datetime.now(timezone.utc) - timedelta(days=30)
        assert ToxicityDecay.is_expired(tagged_at=recent_ts, decay_days=180) is False

    def test_default_decay_days_is_180(self):
        recent_ts = datetime.now(timezone.utc) - timedelta(days=30)
        assert ToxicityDecay.is_expired(tagged_at=recent_ts) is False  # default 180

    def test_expired_boundary_at_exactly_decay_days(self):
        exact_ts = datetime.now(timezone.utc) - timedelta(days=180)
        # Boundary: 180 elapsed days should be expired
        assert ToxicityDecay.is_expired(tagged_at=exact_ts, decay_days=180) is True
