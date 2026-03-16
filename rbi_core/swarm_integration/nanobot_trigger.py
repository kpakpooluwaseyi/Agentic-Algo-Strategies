"""rbi_core/swarm_integration/nanobot_trigger.py — Trigger batch backtests on Dell via SSH."""
import subprocess
import json
from typing import Optional


class NanobotTrigger:
    """
    Sends backtest jobs to Dell Nanobot instance via SSH.
    Dell runs WFA and publishes updated weights to its HTTP endpoint.
    """

    def __init__(self, dell_host: str, dell_user: str, dell_ssh_key: str,
                 remote_script: str = "~/nanobot/run_wfa.py"):
        """
        Args:
            dell_host: IP or hostname of Dell WSL2 (e.g., "192.168.1.50")
            dell_user: SSH username on Dell
            dell_ssh_key: Path to SSH private key
            remote_script: Path to WFA runner script on Dell
        """
        self.dell_host = dell_host
        self.dell_user = dell_user
        self.dell_ssh_key = dell_ssh_key
        self.remote_script = remote_script

    def trigger_wfa(self, strategy_name: str, params_json: str,
                    data_range: str = "30d") -> Optional[str]:
        """
        Trigger Walk-Forward Analysis on Dell.

        Args:
            strategy_name: Name of strategy to optimize.
            params_json: JSON string of current parameter bounds.
            data_range: Lookback period for WFA (e.g., "30d", "90d").

        Returns:
            Job ID string if successful, None on failure.
        """
        cmd = [
            "ssh", "-i", self.dell_ssh_key,
            "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=no",
            f"{self.dell_user}@{self.dell_host}",
            f"python3 {self.remote_script} "
            f"--strategy {strategy_name} "
            f"--params '{params_json}' "
            f"--range {data_range} "
            f"--output-endpoint /weights/latest"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                output = result.stdout.strip()
                print(f"[NanobotTrigger] WFA triggered: {output}")
                return output
            else:
                print(f"[NanobotTrigger] SSH error: {result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            print("[NanobotTrigger] SSH timeout")
            return None
