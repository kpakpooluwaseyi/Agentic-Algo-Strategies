# Tests for rbi_core.utils.logging
# TDD: RED → GREEN → REFACTOR
# Architecture: Co-located tests in tests/ subdirectory (architecture.md#Structure Patterns)

from __future__ import annotations

import json
import logging
import io
import sys
from unittest.mock import patch

import pytest

# Adjust import path for standalone test runs
sys.path.insert(0, ".")

from rbi_core.utils.logging import get_logger, _RequiredFieldsFilter


class TestGetLogger:
    """Test the get_logger factory function."""

    def test_returns_logger_instance(self):
        """get_logger returns a logging.Logger."""
        logger = get_logger("rbi_core.test")
        assert isinstance(logger, logging.Logger)

    def test_logger_name_matches_component(self):
        """Logger component name is set correctly."""
        logger = get_logger("rbi_core.execution.engine")
        assert logger.name == "rbi_core.execution.engine"

    def test_logger_has_required_fields_filter(self):
        """Logger must have _RequiredFieldsFilter attached."""
        logger = get_logger("rbi_core.test_filter")
        filter_types = [type(f) for f in logger.filters]
        assert _RequiredFieldsFilter in filter_types

    def test_logger_does_not_duplicate_filters(self):
        """Calling get_logger twice for same component does not add duplicate filters."""
        logger = get_logger("rbi_core.test_dedup")
        _ = get_logger("rbi_core.test_dedup")  # second call
        rff_count = sum(1 for f in logger.filters if isinstance(f, _RequiredFieldsFilter))
        assert rff_count == 1

class TestAsyncQueueListener:
    """Verify that QueueListener properly drains logs asynchronously."""
    
    def test_queue_listener_starts_and_drains(self):
        """QueueListener should be started if a queue handler targets standard handlers."""
        import rbi_core.utils.logging as rbi_logging
        
        # Reset initialized state
        rbi_logging._initialized = False
        if rbi_logging._listener:
            rbi_logging.flush_logs()
            
        test_stream = io.StringIO()
        target_handler = logging.StreamHandler(test_stream)
        target_handler.setFormatter(logging.Formatter("%(message)s"))
        
        # Mock the root logger to have our target handler (as if loaded from yaml)
        root = logging.getLogger()
        root.handlers = [target_handler]
        
        # Now trigger fallback init which starts listener
        rbi_logging._ensure_initialized()
        
        assert rbi_logging._listener is not None
        
        # Log via the queue
        rbi_logging.log_queue.put(
            logging.LogRecord("test", logging.INFO, "", 0, "async test message", (), None)
        )
        
        # Wait for queue to drain and flush
        rbi_logging.flush_logs()
        root.handlers = [] # cleanup
        
        assert "async test message" in test_stream.getvalue()


class TestRequiredFieldsFilter:
    """Test that required fields are injected into log records."""

    def _make_record(self, msg="test", extra=None):
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg=msg, args=(), exc_info=None
        )
        if extra:
            for k, v in extra.items():
                setattr(record, k, v)
        return record

    def test_injects_event_when_missing(self):
        """If 'event' not in extra, filter injects record message as event."""
        f = _RequiredFieldsFilter()
        record = self._make_record(msg="order_submitted")
        assert not hasattr(record, "event") or record.event == "order_submitted"
        f.filter(record)
        assert hasattr(record, "event")

    def test_injects_context_when_missing(self):
        """If 'context' not in extra, filter injects empty dict."""
        f = _RequiredFieldsFilter()
        record = self._make_record()
        f.filter(record)
        assert hasattr(record, "context")
        assert record.context == {}

    def test_preserves_existing_event(self):
        """If caller provides 'event', filter must not overwrite it."""
        f = _RequiredFieldsFilter()
        record = self._make_record(extra={"event": "custom_event", "context": {"k": "v"}})
        f.filter(record)
        assert record.event == "custom_event"
        assert record.context == {"k": "v"}

    def test_filter_returns_true(self):
        """_RequiredFieldsFilter.filter() always returns True (never drops records)."""
        f = _RequiredFieldsFilter()
        record = self._make_record()
        result = f.filter(record)
        assert result is True


class TestJsonOutputFormat:
    """Integration test — verify JSON is produced when python-json-logger is available."""

    def test_json_output_contains_required_fields(self):
        """When a message is logged, JSON output includes ts, level, component fields."""
        try:
            from pythonjsonlogger import jsonlogger  # noqa: F401
        except ImportError:
            pytest.skip("python-json-logger not installed — skipping JSON format test")

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        from pythonjsonlogger import jsonlogger
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            rename_fields={"asctime": "ts", "levelname": "level", "name": "component"},
        )
        handler.setFormatter(formatter)

        test_logger = logging.getLogger("test.json_format")
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.INFO)

        test_logger.info(
            "order_submitted",
            extra={"event": "order_submit", "context": {"symbol": "BTCUSDT"}},
        )

        output = stream.getvalue().strip()
        assert output, "No log output produced"

        data = json.loads(output)
        assert "ts" in data, f"Missing 'ts' field in: {data}"
        assert "level" in data, f"Missing 'level' field in: {data}"
        assert "component" in data, f"Missing 'component' field in: {data}"
