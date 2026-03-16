"""
rbi_core.utils.time_sync — NTP Drift Monitor

Verifies system clock offset against NTP is within the 50ms tolerance defined by NFR10.
Uses chronyc on Linux/WSL2 and sntp/timed on macOS.

Usage:
    from rbi_core.utils.time_sync import check_ntp_drift, NTPDriftError

    try:
        offset_ms = check_ntp_drift(max_offset_ms=50.0)
        print(f"NTP offset: {offset_ms:.2f}ms ✅")
    except NTPDriftError as e:
        logger.critical("ntp_drift_exceeded", extra={"event": "ntp_check", "context": {"error": str(e)}})
        raise

Architecture compliance:
    - NFR10: Sub-50ms NTP tolerance across Mac and Dell
    - Error logged at CRITICAL level (triggers Safe Harbor awareness)
"""

from __future__ import annotations

import platform
import re
import subprocess
from dataclasses import dataclass
from typing import Optional


class NTPDriftError(Exception):
    """Raised when NTP offset exceeds acceptable threshold."""


@dataclass(frozen=True)
class NTPStatus:
    """Result of an NTP drift check."""
    offset_ms: float
    reference: str
    within_threshold: bool
    max_allowed_ms: float


def check_ntp_drift(max_offset_ms: float = 50.0) -> float:
    """
    Check system NTP drift via chronyc (Linux/WSL2) or sntp (macOS).

    Args:
        max_offset_ms: Maximum acceptable absolute offset in milliseconds. Default: 50ms (NFR10).

    Returns:
        Absolute offset in milliseconds if within threshold.

    Raises:
        NTPDriftError: If offset exceeds max_offset_ms or chrony/sntp is unavailable.
        RuntimeError: If the NTP binary is not found on PATH.
    """
    system = platform.system()

    if system == "Linux":
        status = _chrony_check()
    elif system == "Darwin":
        status = _macos_sntp_check()
    else:
        raise RuntimeError(f"Unsupported platform for NTP check: {system}")

    if not status.within_threshold or status.offset_ms > max_offset_ms:
        raise NTPDriftError(
            f"NTP offset {status.offset_ms:.2f}ms exceeds max {max_offset_ms}ms (NFR10). "
            f"Reference: {status.reference}"
        )

    return status.offset_ms


def get_ntp_status(max_offset_ms: float = 50.0) -> NTPStatus:
    """
    Return full NTP status without raising on threshold breach.
    Used by monitoring/alerting logic.
    """
    system = platform.system()
    if system == "Linux":
        status = _chrony_check()
    elif system == "Darwin":
        status = _macos_sntp_check()
    else:
        return NTPStatus(
            offset_ms=999.0,
            reference="unknown",
            within_threshold=False,
            max_allowed_ms=max_offset_ms,
        )

    return NTPStatus(
        offset_ms=status.offset_ms,
        reference=status.reference,
        within_threshold=status.offset_ms <= max_offset_ms,
        max_allowed_ms=max_offset_ms,
    )


def _chrony_check() -> NTPStatus:
    """Parse `chronyc tracking` output on Linux/WSL2."""
    try:
        result = subprocess.run(
            ["chronyc", "tracking"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "chronyc not found. Install chrony: sudo apt install chrony"
        )
    except subprocess.TimeoutExpired:
        raise NTPDriftError("chronyc timed out after 5 seconds")

    output = result.stdout
    # Example line: "System time     :  0.000012345 seconds slow of NTP time"
    match = re.search(r"System time\s*:\s*([\d.]+)\s*seconds", output)
    if not match:
        raise NTPDriftError(f"Failed to parse chronyc output: {output[:200]!r}")

    offset_s = float(match.group(1))
    offset_ms = abs(offset_s) * 1000.0

    # Extract reference server
    ref_match = re.search(r"Reference ID\s*:\s*(\S+)", output)
    reference = ref_match.group(1) if ref_match else "unknown"

    return NTPStatus(
        offset_ms=offset_ms,
        reference=reference,
        within_threshold=True,   # threshold check done by caller
        max_allowed_ms=50.0,
    )


def _macos_sntp_check() -> NTPStatus:
    """Use sntp on macOS to measure offset."""
    try:
        result = subprocess.run(
            ["sntp", "-t", "1", "time.apple.com"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        # Fall back to checking timed (macOS) via systemsetup
        return _macos_timed_check()
    except subprocess.TimeoutExpired:
        raise NTPDriftError("sntp timed out after 5 seconds")

    output = result.stdout + result.stderr
    # sntp output example: "+0.012345 +/- 0.005678 time.apple.com 17.253.68.125"
    match = re.search(r"([+-][\d.]+)\s+\+/-", output)
    if not match:
        raise NTPDriftError(f"Failed to parse sntp output: {output[:200]!r}")

    offset_ms = abs(float(match.group(1))) * 1000.0
    return NTPStatus(
        offset_ms=offset_ms,
        reference="time.apple.com",
        within_threshold=True,
        max_allowed_ms=50.0,
    )


def _macos_timed_check() -> NTPStatus:
    """Fallback: Enforce strict NFR10 failure if sntp is unavailable.
    We cannot fake a passing offset test on macOS without measuring it.
    """
    try:
        # Check if network time is enabled as a proxy for basic health
        result = subprocess.run(
            ["systemsetup", "-getusingnetworktime"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if "Network Time: On" in result.stdout:
            # We can't guarantee sub-50ms objectively without sntp.
            # To strictly enforce NFR10 as required, we must raise.
            raise NTPDriftError(
                "sntp measurement failed. Cannot guarantee sub-50ms NFR10 offset "
                "relying solely on Network Time status."
            )
    except FileNotFoundError:
        pass

    raise NTPDriftError("Unable to verify NTP drift on macOS. Ensure 'sntp' is available.")
