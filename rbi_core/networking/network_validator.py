"""NetworkValidator — enforces Tailscale-only IP connections.

Raises ``NetworkSecurityError`` for any host outside the Tailscale
CGNAT range ``100.64.0.0/10``.
"""
from __future__ import annotations

import ipaddress
import logging

from rbi_core.exceptions import NetworkSecurityError

logger = logging.getLogger(__name__)

_TAILSCALE_CIDR = ipaddress.ip_network("100.64.0.0/10")


class NetworkValidator:
    """Static helpers to enforce Tailscale-only networking."""

    @staticmethod
    def is_tailscale_ip(host: str) -> bool:
        """Return True if *host* falls within the Tailscale CGNAT range.

        Args:
            host: IPv4 address string (e.g. ``"100.101.23.5"``).

        Returns:
            True if the address is within ``100.64.0.0/10``, False otherwise.
        """
        try:
            addr = ipaddress.ip_address(host)
            return addr in _TAILSCALE_CIDR
        except ValueError:
            return False

    @staticmethod
    def assert_tailscale_host(host: str) -> None:
        """Assert *host* is a Tailscale IP or raise ``NetworkSecurityError``.

        Also emits a structured JSON-compatible log on failure.

        Args:
            host: IPv4 address string to validate.

        Raises:
            NetworkSecurityError: If the host is not in the Tailscale CIDR.
        """
        if not NetworkValidator.is_tailscale_ip(host):
            logger.warning(
                "network_security_violation",
                extra={
                    "event": "network_security_violation",
                    "host": host,
                    "expected_cidr": str(_TAILSCALE_CIDR),
                },
            )
            raise NetworkSecurityError(
                f"Connection to non-Tailscale IP rejected: {host!r}. "
                f"Expected host within {_TAILSCALE_CIDR}."
            )
