"""
rbi_core.utils.logging — Structured JSON Logger Factory

Usage:
    from rbi_core.utils.logging import get_logger

    logger = get_logger("rbi_core.execution")
    logger.info("order_submitted", extra={"event": "order_submit", "context": {"symbol": "BTCUSDT", "side": "buy"}})

Output (JSON):
    {"ts": "2026-03-05T01:00:00+09:00", "level": "INFO", "component": "rbi_core.execution",
     "message": "order_submitted", "event": "order_submit", "context": {"symbol": "BTCUSDT", "side": "buy"}}

Architecture compliance:
    - Required fields: ts, component, event, level, context (architecture.md#Process Patterns / Logging)
    - Async I/O: Dedicated low-priority thread via QueueHandler (NFR14)
    - Format: single-line JSON for forensic replayability
"""

from __future__ import annotations

import logging
import logging.config
import queue
import threading
from pathlib import Path
from typing import Optional

# Lazy import — only required when production logging is initialized
try:
    from pythonjsonlogger import jsonlogger  # type: ignore[import]
    _JSON_LOGGER_AVAILABLE = True
except ImportError:
    _JSON_LOGGER_AVAILABLE = False

# Module-level queue and listener for async I/O (low-priority thread)
log_queue: queue.Queue[logging.LogRecord] = queue.Queue(maxsize=10_000)
_listener: Optional[logging.handlers.QueueListener] = None  # type: ignore[name-defined]
_initialized = False
_lock = threading.Lock()


class _RequiredFieldsFilter(logging.Filter):
    """Inject default 'event' and 'context' fields if caller forgot them."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "event"):
            record.event = record.getMessage()  # type: ignore[attr-defined]
        if not hasattr(record, "context"):
            record.context = {}  # type: ignore[attr-defined]
        return True


def get_logger(component: str) -> logging.Logger:
    """
    Return a structured JSON logger for the given component name.

    Args:
        component: Dot-separated module name, e.g. "rbi_core.execution.engine"

    Returns:
        A Logger instance pre-configured with JSON formatting and required fields.
    """
    _ensure_initialized()
    logger = logging.getLogger(component)
    # Add the required-fields filter if not already present
    for f in logger.filters:
        if isinstance(f, _RequiredFieldsFilter):
            break
    else:
        logger.addFilter(_RequiredFieldsFilter())
    return logger


def _ensure_initialized() -> None:
    """Initialize logging exactly once using YAML config if available, else fallback."""
    global _initialized, _listener
    with _lock:
        if _initialized:
            return

        config_path = Path("config/logging.yaml")
        if config_path.exists() and _JSON_LOGGER_AVAILABLE:
            import yaml  # type: ignore[import]
            with config_path.open() as f:
                config = yaml.safe_load(f)
            logging.config.dictConfig(config)
        else:
            _setup_fallback_logging()

        # Start the QueueListener to drain async logs
        _start_queue_listener()

        _initialized = True

def _start_queue_listener() -> None:
    """Find file handlers attached to root/RBI loggers and start listening on the queue."""
    global _listener
    
    # We want the listener to pass records to our actual FileHandlers.
    # The queue handler itself is attached to the config, so we find the targets.
    target_handlers = []
    root = logging.getLogger()
    for h in root.handlers:
        if not isinstance(h, logging.handlers.QueueHandler):
            target_handlers.append(h)
    
    # If using dictConfig, handlers might be defined cleanly.
    if target_handlers:
        from logging.handlers import QueueListener
        _listener = QueueListener(log_queue, *target_handlers, respect_handler_level=True)
        _listener.start()
        
        # Ensure it shuts down properly
        import atexit
        atexit.register(flush_logs)

def flush_logs() -> None:
    """Stop the QueueListener to flush remaining logs. Call on graceful shutdown."""
    global _listener
    with _lock:
        if _listener is not None:
            _listener.stop()
            _listener = None


def _setup_fallback_logging() -> None:
    """Minimal JSON logging for environments without python-json-logger or YAML config."""
    root = logging.getLogger()
    if root.handlers:
        return  # Already configured by something else

    handler = logging.StreamHandler()

    if _JSON_LOGGER_AVAILABLE:
        from pythonjsonlogger import jsonlogger  # type: ignore[import]
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            rename_fields={"asctime": "ts", "levelname": "level", "name": "component"},
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
        handler.setFormatter(formatter)
    else:
        # Plaintext fallback — not for production use
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
        ))

    root.addHandler(handler)
    root.setLevel(logging.INFO)
