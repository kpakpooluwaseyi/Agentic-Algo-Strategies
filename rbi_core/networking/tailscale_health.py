"""TailscaleHealthChecker — verifies Tailscale daemon and peer reachability.

Uses ``subprocess`` to call the ``tailscale`` CLI. Tests mock
``subprocess.run`` to avoid requiring a live Tailscale install.
"""
from __future__ import annotations

import logging
import subprocess

from rbi_core.exceptions import NetworkSecurityError

logger = logging.getLogger(__name__)


class TailscaleHealthChecker:
    """Verifies Tailscale daemon status and peer reachability."""

    def is_daemon_running(self) -> bool:
        """Return True if the Tailscale daemon responds to ``tailscale status``.

        Returns:
            True if exit code is 0, False on any failure/absence.
        """
        try:
            result = subprocess.run(
                ["tailscale", "status"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def ping_peer(self, peer_ip: str, count: int = 1) -> bool:
        """Return True if the Tailscale peer at *peer_ip* is reachable.

        Args:
            peer_ip: Tailscale IP of the peer to ping.
            count:   Number of pings to send (default 1).

        Returns:
            True if ``tailscale ping`` exits 0, False otherwise.
        """
        try:
            result = subprocess.run(
                ["tailscale", "ping", f"--c={count}", peer_ip],
                capture_output=True,
                timeout=15,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def assert_peer_reachable(self, peer_ip: str) -> None:
        """Assert *peer_ip* is reachable via Tailscale or raise ``NetworkSecurityError``.

        Args:
            peer_ip: Tailscale IP of the peer to verify.

        Raises:
            NetworkSecurityError: If the peer is unreachable.
        """
        if not self.ping_peer(peer_ip):
            logger.error(
                "tailscale_peer_unreachable",
                extra={
                    "event": "network_security_violation",
                    "peer_ip": peer_ip,
                    "context": "Tailscale peer unreachable",
                },
            )
            raise NetworkSecurityError(
                f"Tailscale peer unreachable: {peer_ip!r}. "
                "Ensure Tailscale is running and the peer is online."
            )
