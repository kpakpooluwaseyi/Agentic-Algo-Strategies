"""Unit tests for NetworkValidator — TDD RED phase."""
from __future__ import annotations

import pytest

from rbi_core.networking.network_validator import NetworkValidator
from rbi_core.exceptions import NetworkSecurityError


class TestNetworkValidatorIsTailscaleIp:

    def test_returns_true_for_tailscale_ip(self):
        assert NetworkValidator.is_tailscale_ip("100.64.0.1") is True

    def test_returns_true_for_upper_tailscale_range(self):
        assert NetworkValidator.is_tailscale_ip("100.127.255.255") is True

    def test_returns_true_for_common_tailscale_ip(self):
        assert NetworkValidator.is_tailscale_ip("100.101.23.5") is True

    def test_returns_false_for_localhost(self):
        assert NetworkValidator.is_tailscale_ip("127.0.0.1") is False

    def test_returns_false_for_lan_ip(self):
        assert NetworkValidator.is_tailscale_ip("192.168.1.10") is False

    def test_returns_false_for_public_ip(self):
        assert NetworkValidator.is_tailscale_ip("8.8.8.8") is False

    def test_returns_false_for_wsl_ip(self):
        assert NetworkValidator.is_tailscale_ip("172.24.0.1") is False

    def test_returns_false_for_ip_just_outside_tailscale_range(self):
        # 100.63.255.255 is one address below 100.64.0.0/10
        assert NetworkValidator.is_tailscale_ip("100.63.255.255") is False

    def test_returns_false_for_ip_just_above_tailscale_range(self):
        # 100.128.0.0 is above 100.127.255.255
        assert NetworkValidator.is_tailscale_ip("100.128.0.0") is False


class TestNetworkValidatorAssertTailscaleHost:

    def test_does_not_raise_for_tailscale_ip(self):
        NetworkValidator.assert_tailscale_host("100.100.1.1")  # Must not raise

    def test_raises_network_security_error_for_public_ip(self):
        with pytest.raises(NetworkSecurityError, match="non-Tailscale"):
            NetworkValidator.assert_tailscale_host("8.8.8.8")

    def test_raises_network_security_error_for_localhost(self):
        with pytest.raises(NetworkSecurityError, match="non-Tailscale"):
            NetworkValidator.assert_tailscale_host("127.0.0.1")

    def test_raises_network_security_error_for_lan_ip(self):
        with pytest.raises(NetworkSecurityError, match="non-Tailscale"):
            NetworkValidator.assert_tailscale_host("192.168.0.1")

    def test_error_message_includes_host(self):
        try:
            NetworkValidator.assert_tailscale_host("10.0.0.1")
        except NetworkSecurityError as exc:
            assert "10.0.0.1" in str(exc)
