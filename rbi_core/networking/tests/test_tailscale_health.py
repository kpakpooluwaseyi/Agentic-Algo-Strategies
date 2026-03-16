"""Unit tests for TailscaleHealthChecker — TDD RED phase."""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from rbi_core.networking.tailscale_health import TailscaleHealthChecker
from rbi_core.exceptions import NetworkSecurityError


@pytest.fixture
def checker():
    return TailscaleHealthChecker()


class TestIsDaemonRunning:

    def test_returns_true_when_exit_code_zero(self, checker):
        result = MagicMock()
        result.returncode = 0
        with patch("subprocess.run", return_value=result):
            assert checker.is_daemon_running() is True

    def test_returns_false_when_exit_code_nonzero(self, checker):
        result = MagicMock()
        result.returncode = 1
        with patch("subprocess.run", return_value=result):
            assert checker.is_daemon_running() is False

    def test_returns_false_when_subprocess_raises_file_not_found(self, checker):
        with patch("subprocess.run", side_effect=FileNotFoundError("tailscale not found")):
            assert checker.is_daemon_running() is False

    def test_calls_tailscale_status(self, checker):
        result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=result) as mock_run:
            checker.is_daemon_running()
            args = mock_run.call_args[0][0]
            assert "tailscale" in args
            assert "status" in args


class TestPingPeer:

    def test_returns_true_when_ping_succeeds(self, checker):
        result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=result):
            assert checker.ping_peer("100.101.1.1") is True

    def test_returns_false_when_ping_fails(self, checker):
        result = MagicMock(returncode=1)
        with patch("subprocess.run", return_value=result):
            assert checker.ping_peer("100.101.1.1") is False

    def test_returns_false_when_subprocess_raises(self, checker):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert checker.ping_peer("100.101.1.1") is False

    def test_calls_tailscale_ping_with_ip(self, checker):
        result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=result) as mock_run:
            checker.ping_peer("100.50.1.1", count=3)
            args = mock_run.call_args[0][0]
            assert "tailscale" in args
            assert "ping" in args
            assert "100.50.1.1" in args


class TestAssertPeerReachable:

    def test_does_not_raise_when_reachable(self, checker):
        result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=result):
            checker.assert_peer_reachable("100.100.1.1")  # Must not raise

    def test_raises_network_security_error_when_not_reachable(self, checker):
        result = MagicMock(returncode=1)
        with patch("subprocess.run", return_value=result):
            with pytest.raises(NetworkSecurityError, match="unreachable"):
                checker.assert_peer_reachable("100.100.1.1")

    def test_error_message_includes_ip(self, checker):
        result = MagicMock(returncode=1)
        with patch("subprocess.run", return_value=result):
            try:
                checker.assert_peer_reachable("100.55.10.10")
            except NetworkSecurityError as exc:
                assert "100.55.10.10" in str(exc)
