"""Unit tests for rbi_core.exceptions."""
from __future__ import annotations

import pytest

# These imports will fail until rbi_core/exceptions.py is created (RED phase)
from rbi_core.exceptions import (
    HeartbeatError,
    RBIError,
    StreamConsumeError,
    StreamPublishError,
)


class TestExceptionHierarchy:
    def test_rbi_error_is_base_exception(self):
        err = RBIError("base")
        assert isinstance(err, Exception)

    def test_heartbeat_error_inherits_rbi_error(self):
        err = HeartbeatError("lost")
        assert isinstance(err, RBIError)

    def test_stream_publish_error_inherits_rbi_error(self):
        err = StreamPublishError("publish fail")
        assert isinstance(err, RBIError)

    def test_stream_consume_error_inherits_rbi_error(self):
        err = StreamConsumeError("consume fail")
        assert isinstance(err, RBIError)

    def test_heartbeat_error_message(self):
        err = HeartbeatError("missed 3 heartbeats")
        assert "missed 3 heartbeats" in str(err)
