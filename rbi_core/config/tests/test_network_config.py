"""Unit tests for NetworkConfig — TDD RED phase."""
from __future__ import annotations

import os
import textwrap

import pytest

from rbi_core.config.network_config import NetworkConfig
from rbi_core.exceptions import ConfigurationError


VALID_YAML = textwrap.dedent("""\
    redis_host: "100.64.0.1"
    redis_port: 6379
    tailscale_cidr: "100.64.0.0/10"
    dell_tailscale_ip: "100.64.0.1"
    mac_tailscale_ip: "100.64.0.2"
""")


@pytest.fixture
def valid_config_file(tmp_path):
    p = tmp_path / "network.yaml"
    p.write_text(VALID_YAML)
    return str(p)


class TestNetworkConfigLoad:

    def test_loads_redis_host(self, valid_config_file):
        cfg = NetworkConfig.load(valid_config_file)
        assert cfg.redis_host == "100.64.0.1"

    def test_loads_redis_port(self, valid_config_file):
        cfg = NetworkConfig.load(valid_config_file)
        assert cfg.redis_port == 6379

    def test_loads_tailscale_cidr(self, valid_config_file):
        cfg = NetworkConfig.load(valid_config_file)
        assert cfg.tailscale_cidr == "100.64.0.0/10"

    def test_loads_dell_tailscale_ip(self, valid_config_file):
        cfg = NetworkConfig.load(valid_config_file)
        assert cfg.dell_tailscale_ip == "100.64.0.1"

    def test_loads_mac_tailscale_ip(self, valid_config_file):
        cfg = NetworkConfig.load(valid_config_file)
        assert cfg.mac_tailscale_ip == "100.64.0.2"


class TestNetworkConfigValidation:

    def test_raises_configuration_error_on_missing_redis_host(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("redis_port: 6379\ntailscale_cidr: '100.64.0.0/10'\ndell_tailscale_ip: '100.64.0.1'\nmac_tailscale_ip: '100.64.0.2'\n")
        with pytest.raises(ConfigurationError, match="redis_host"):
            NetworkConfig.load(str(p))

    def test_raises_configuration_error_on_missing_tailscale_cidr(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("redis_host: '100.64.0.1'\nredis_port: 6379\ndell_tailscale_ip: '100.64.0.1'\nmac_tailscale_ip: '100.64.0.2'\n")
        with pytest.raises(ConfigurationError, match="tailscale_cidr"):
            NetworkConfig.load(str(p))

    def test_raises_configuration_error_on_missing_dell_ip(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("redis_host: '100.64.0.1'\nredis_port: 6379\ntailscale_cidr: '100.64.0.0/10'\nmac_tailscale_ip: '100.64.0.2'\n")
        with pytest.raises(ConfigurationError, match="dell_tailscale_ip"):
            NetworkConfig.load(str(p))

    def test_raises_configuration_error_on_missing_mac_ip(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("redis_host: '100.64.0.1'\nredis_port: 6379\ntailscale_cidr: '100.64.0.0/10'\ndell_tailscale_ip: '100.64.0.1'\n")
        with pytest.raises(ConfigurationError, match="mac_tailscale_ip"):
            NetworkConfig.load(str(p))

    def test_raises_configuration_error_when_file_missing(self):
        with pytest.raises(ConfigurationError, match="not found"):
            NetworkConfig.load("/nonexistent/path/network.yaml")
