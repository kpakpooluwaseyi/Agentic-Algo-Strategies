"""NetworkConfig — loads and validates config/network.yaml.

Required fields: ``redis_host``, ``tailscale_cidr``,
``dell_tailscale_ip``, ``mac_tailscale_ip``.
Raises ``ConfigurationError`` on missing fields or inaccessible file.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import yaml

from rbi_core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

_REQUIRED_FIELDS = (
    "redis_host",
    "tailscale_cidr",
    "dell_tailscale_ip",
    "mac_tailscale_ip",
)


@dataclass(frozen=True)
class NetworkConfig:
    """Immutable snapshot of network.yaml configuration."""

    redis_host: str
    redis_port: int
    tailscale_cidr: str
    dell_tailscale_ip: str
    mac_tailscale_ip: str

    @classmethod
    def load(cls, path: str) -> "NetworkConfig":
        """Load and validate a network YAML config file.

        Args:
            path: Absolute or relative path to ``network.yaml``.

        Returns:
            A validated, immutable ``NetworkConfig`` instance.

        Raises:
            ConfigurationError: If the file is missing or required fields are absent.
        """
        try:
            with open(path) as fh:
                raw: dict[str, Any] = yaml.safe_load(fh) or {}
        except FileNotFoundError:
            raise ConfigurationError(
                f"Network configuration file not found: {path!r}"
            )

        for field in _REQUIRED_FIELDS:
            if field not in raw or raw[field] is None:
                raise ConfigurationError(
                    f"Missing required field '{field}' in {path!r}"
                )

        logger.info(
            "network_config_loaded",
            extra={
                "event": "network_config_loaded",
                "redis_host": raw["redis_host"],
                "tailscale_cidr": raw["tailscale_cidr"],
            },
        )

        return cls(
            redis_host=str(raw["redis_host"]),
            redis_port=int(raw.get("redis_port", 6379)),
            tailscale_cidr=str(raw["tailscale_cidr"]),
            dell_tailscale_ip=str(raw["dell_tailscale_ip"]),
            mac_tailscale_ip=str(raw["mac_tailscale_ip"]),
        )
