# Tests for rbi_core.utils.time_sync
# TDD: RED → GREEN → REFACTOR
# Architecture: Co-located tests in tests/ subdirectory (architecture.md#Structure Patterns)

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, ".")

from rbi_core.utils.time_sync import (
    NTPDriftError,
    NTPStatus,
    _chrony_check,
    _macos_sntp_check,
    _macos_timed_check,
    check_ntp_drift,
    get_ntp_status,
)

# ─── Fixtures ────────────────────────────────────────────────────────────────

CHRONY_OUTPUT_GOOD = """\
Reference ID    : C0A80001 (192.168.0.1)
Stratum         : 3
Ref time (UTC)  : Thu Mar 05 00:00:00 2026
System time     : 0.000012345 seconds slow of NTP time
Last offset     : -0.000012221 seconds
RMS offset      : 0.000015678 seconds
Frequency       : 1.234 ppm slow
Residual freq   : 0.001 ppm
Skew            : 0.012 ppm
Root delay      : 0.004567890 seconds
Root dispersion : 0.000123456 seconds
Update interval : 64.1 seconds
Leap status     : Normal
"""

CHRONY_OUTPUT_HIGH_DRIFT = """\
Reference ID    : C0A80001 (192.168.0.1)
System time     : 0.150000000 seconds slow of NTP time
"""

SNTP_OUTPUT_GOOD = """\
+0.012345 +/- 0.005678 time.apple.com 17.253.68.125
"""

SNTP_OUTPUT_BAD_DRIFT = """\
+0.200000 +/- 0.001000 time.apple.com 17.253.68.125
"""


# ─── Tests: _chrony_check ────────────────────────────────────────────────────

class TestChronyCheck:

    def test_parses_good_output_within_threshold(self):
        """Parses chronyc output and returns offset < 50ms."""
        mock_result = MagicMock(stdout=CHRONY_OUTPUT_GOOD, returncode=0)
        with patch("subprocess.run", return_value=mock_result):
            status = _chrony_check()
        # 0.000012345s × 1000 = ~0.012ms
        assert status.offset_ms < 1.0
        # chronyc Reference ID is hex-encoded (e.g. "C0A80001"), not dotted-decimal
        assert len(status.reference) > 0

    def test_raises_on_chrony_not_installed(self):
        """Raises RuntimeError if chronyc is not on PATH."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(RuntimeError, match="chronyc not found"):
                _chrony_check()

    def test_raises_on_timeout(self):
        """Raises NTPDriftError if chronyc times out."""
        import subprocess
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("chronyc", 5)):
            with pytest.raises(NTPDriftError, match="timed out"):
                _chrony_check()

    def test_raises_on_unparseable_output(self):
        """Raises NTPDriftError if chronyc output is garbled."""
        mock_result = MagicMock(stdout="garbage output", returncode=1)
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(NTPDriftError, match="Failed to parse"):
                _chrony_check()


# ─── Tests: check_ntp_drift ──────────────────────────────────────────────────

class TestCheckNTPDrift:

    def test_returns_offset_when_within_threshold(self):
        """Returns float offset_ms when within 50ms threshold."""
        good_status = NTPStatus(offset_ms=12.3, reference="ntp.pool", within_threshold=True, max_allowed_ms=50.0)
        with patch("rbi_core.utils.time_sync._chrony_check", return_value=good_status), \
             patch("platform.system", return_value="Linux"):
            result = check_ntp_drift(max_offset_ms=50.0)
        assert result == 12.3

    def test_raises_when_offset_exceeds_threshold(self):
        """Raises NTPDriftError when offset > 50ms."""
        bad_status = NTPStatus(offset_ms=150.0, reference="ntp.pool", within_threshold=False, max_allowed_ms=50.0)
        with patch("rbi_core.utils.time_sync._chrony_check", return_value=bad_status), \
             patch("platform.system", return_value="Linux"):
            with pytest.raises(NTPDriftError, match="150.00ms exceeds max 50.0ms"):
                check_ntp_drift(max_offset_ms=50.0)

    def test_raises_on_unsupported_platform(self):
        """Raises RuntimeError on Windows or unknown OS."""
        with patch("platform.system", return_value="Windows"):
            with pytest.raises(RuntimeError, match="Unsupported platform"):
                check_ntp_drift()


# ─── Tests: get_ntp_status ───────────────────────────────────────────────────

class TestGetNTPStatus:

    def test_returns_ntpstatus_dataclass(self):
        """Returns NTPStatus dataclass with all fields."""
        good_chrony = NTPStatus(offset_ms=5.0, reference="pool.ntp.org", within_threshold=True, max_allowed_ms=50.0)
        with patch("rbi_core.utils.time_sync._chrony_check", return_value=good_chrony), \
             patch("platform.system", return_value="Linux"):
            result = get_ntp_status()
        assert isinstance(result, NTPStatus)
        assert result.offset_ms == 5.0
        assert result.within_threshold is True

    def test_within_threshold_false_when_high_drift(self):
        """NTPStatus.within_threshold is False when offset > max."""
        high_drift = NTPStatus(offset_ms=80.0, reference="pool.ntp.org", within_threshold=False, max_allowed_ms=50.0)
        with patch("rbi_core.utils.time_sync._chrony_check", return_value=high_drift), \
             patch("platform.system", return_value="Linux"):
            status = get_ntp_status(max_offset_ms=50.0)
        assert status.within_threshold is False


# ─── Tests: Requirements file structure ──────────────────────────────────────
# ─── Tests: macOS strict fallback ───────────────────────────────────────────

class TestMacOSTimedCheck:
    
    def test_macos_timed_check_raises_when_network_time_on(self):
        """Even if Network Time is On, we must raise because we can't measure 50ms strictly."""
        mock_result = MagicMock(stdout="Network Time: On", returncode=0)
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(NTPDriftError, match="Cannot guarantee sub-50ms"):
                _macos_timed_check()

    def test_macos_timed_check_raises_when_command_fails(self):
        """If systemsetup fails or file not found, raises fallback error."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(NTPDriftError, match="Unable to verify"):
                _macos_timed_check()
class TestRequirementsFiles:
    """Validate that requirements files exist and share identical pinned packages."""

    def test_requirements_wsl_exists(self):
        """requirements-wsl.txt must exist."""
        from pathlib import Path
        assert Path("requirements-wsl.txt").exists(), "requirements-wsl.txt not found"

    def test_requirements_mac_exists(self):
        """requirements-mac.txt must exist."""
        from pathlib import Path
        assert Path("requirements-mac.txt").exists(), "requirements-mac.txt not found"

    def test_shared_packages_have_identical_pins(self):
        """Shared packages (numpy, pandas, pyarrow, msgpack) must have same pin in both files."""
        from pathlib import Path

        shared_pkgs = ["numpy", "pandas", "pyarrow", "msgpack", "zstandard"]

        def _parse_pins(filepath: str) -> dict[str, str]:
            pins = {}
            content = Path(filepath).read_text()
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                if "==" in line:
                    pkg, version = line.split("==", 1)
                    pkg = pkg.strip().lower().split("[")[0]  # strip extras
                    version = version.split("#")[0].strip()  # strip inline comments
                    pins[pkg] = version
            return pins

        wsl_pins = _parse_pins("requirements-wsl.txt")
        mac_pins = _parse_pins("requirements-mac.txt")

        for pkg in shared_pkgs:
            assert pkg in wsl_pins, f"{pkg} missing from requirements-wsl.txt"
            assert pkg in mac_pins, f"{pkg} missing from requirements-mac.txt"
            assert wsl_pins[pkg] == mac_pins[pkg], (
                f"Version mismatch for '{pkg}': "
                f"WSL={wsl_pins[pkg]!r} vs Mac={mac_pins[pkg]!r}"
            )
