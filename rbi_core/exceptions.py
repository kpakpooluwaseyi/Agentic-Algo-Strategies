"""RBI Swarm custom exception hierarchy.

All domain exceptions inherit from RBIError for unified handling at service boundaries.

Hierarchy:
    RBIError
    ├── HeartbeatError       — heartbeat loss / Safe Harbor trigger
    ├── StreamPublishError   — Redis Streams XADD failure
    └── StreamConsumeError   — Redis Streams XREADGROUP failure
"""
from __future__ import annotations


class RBIError(Exception):
    """Base exception for all RBI Swarm errors."""


class HeartbeatError(RBIError):
    """Raised when the heartbeat monitor detects consecutive missed heartbeats.

    Triggers Safe Harbor Mode transition (NFR8).
    """


class StreamPublishError(RBIError):
    """Redis XADD failed after exhausting retries."""


class StreamConsumeError(RBIError):
    """Redis XREADGROUP failed or consumer group is unavailable."""


class NetworkSecurityError(RBIError):
    """Attempted connection to a non-Tailscale or unauthorized IP address."""


class ConfigurationError(RBIError):
    """Required configuration field is missing or invalid."""


class SecretsNotFoundError(RBIError):
    """Requested secret not found in the OS keyring."""


class DataSanitizationError(RBIError):
    """Input DataFrame is empty or invalid after sanitization."""


class SimulationError(RBIError):
    """VectorBT simulation failed for a given ticker or strategy DNA."""


class MemoryDBError(RBIError):
    """ChromaDB operation failed."""


class MemoryDBChecksumError(MemoryDBError):
    """ChromaDB collection count does not match the stored manifest."""
