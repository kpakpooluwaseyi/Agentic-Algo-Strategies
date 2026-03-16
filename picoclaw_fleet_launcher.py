#!/usr/bin/env python3
"""picoclaw_fleet_launcher.py — Launch N PicoClaw instances with role-cycling.

Usage:
    python picoclaw_fleet_launcher.py              # default 8 instances
    python picoclaw_fleet_launcher.py --num=4      # launch 4
    python picoclaw_fleet_launcher.py --kill       # kill all managed instances

Role allocation (cycles for N=8):
    extra_0: breeder     extra_1: backtester   extra_2: breeder
    extra_3: backtester  extra_4: breeder      extra_5: backtester
    extra_6: coder       extra_7: monitor

Idempotent: checks PID files before launching. Won't double-spawn.
"""
import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time

# === CONFIG ===
PICOCLAW_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "picoclaw_main.py")
PID_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".fleet_pids")
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fleet.log")

# Role cycle: 3 breeders, 3 backtesters, 1 coder, 1 monitor = 8 total
ROLE_CYCLE = [
    "breeder", "backtester", "breeder", "backtester",
    "breeder", "backtester", "coder", "monitor",
]


def _log(msg: str) -> None:
    """Append timestamped message to fleet.log and print to stdout."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def _pid_file(instance_id: str) -> str:
    return os.path.join(PID_DIR, f"{instance_id}.pid")


def _is_running(instance_id: str) -> bool:
    """Check if instance is already running via PID file + process check."""
    pf = _pid_file(instance_id)
    if not os.path.exists(pf):
        return False
    try:
        with open(pf) as f:
            pid = int(f.read().strip())
        # Check if process exists (signal 0 = no signal, just check)
        os.kill(pid, 0)
        return True
    except (ValueError, ProcessLookupError, PermissionError, OSError):
        # Stale PID file — clean up
        os.remove(pf)
        return False


def _write_pid(instance_id: str, pid: int) -> None:
    with open(_pid_file(instance_id), "w") as f:
        f.write(str(pid))


def launch_fleet(num_instances: int) -> None:
    """Launch N PicoClaw instances with cycling roles."""
    os.makedirs(PID_DIR, exist_ok=True)

    if not os.path.exists(PICOCLAW_SCRIPT):
        _log(f"ERROR: PicoClaw script not found at {PICOCLAW_SCRIPT}")
        _log("Set PICOCLAW_SCRIPT in this file or ensure ~/RBI_Swarm/picoclaw_main.py exists.")
        sys.exit(1)

    launched = []
    skipped = []
    role_counts: dict[str, int] = {}

    for i in range(num_instances):
        instance_id = f"extra_{i}"
        role = ROLE_CYCLE[i % len(ROLE_CYCLE)]

        if _is_running(instance_id):
            skipped.append((instance_id, role))
            _log(f"SKIP {instance_id} ({role}) — already running")
            role_counts[role] = role_counts.get(role, 0) + 1
            continue

        cmd = [
            sys.executable, PICOCLAW_SCRIPT,
            f"--instance_id={instance_id}",
            f"--role={role}",
        ]

        # Launch detached — stdout/stderr to per-instance log
        instance_log = os.path.join(PID_DIR, f"{instance_id}.log")
        with open(instance_log, "a") as log_fh:
            proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # Detach from parent
            )

        _write_pid(instance_id, proc.pid)
        launched.append((instance_id, role, proc.pid))
        role_counts[role] = role_counts.get(role, 0) + 1
        _log(f"LAUNCHED {instance_id} ({role}) PID={proc.pid}")

        # Small delay to avoid port/resource collisions
        time.sleep(0.3)

    # === Final Status ===
    total = len(launched) + len(skipped)
    role_summary = ", ".join(f"{r}={c}" for r, c in sorted(role_counts.items()))

    _log("=" * 60)
    _log(f"Launched {len(launched)} instances ({len(skipped)} already running)")
    _log(f"Total fleet: {total} instances with roles: {role_summary}")
    _log(f"PID files:   {PID_DIR}")
    _log(f"Logs:        {LOG_FILE}")
    _log("=" * 60)

    # Write fleet manifest for programmatic access
    manifest = {
        "timestamp": datetime.datetime.now().isoformat(),
        "instances": [
            {"id": iid, "role": role, "pid": pid}
            for iid, role, pid in launched
        ],
        "skipped": [
            {"id": iid, "role": role}
            for iid, role in skipped
        ],
    }
    manifest_path = os.path.join(PID_DIR, "fleet_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def kill_fleet() -> None:
    """Kill all managed PicoClaw instances."""
    if not os.path.isdir(PID_DIR):
        _log("No fleet PID directory found. Nothing to kill.")
        return

    killed = 0
    for fname in os.listdir(PID_DIR):
        if not fname.endswith(".pid"):
            continue
        pf = os.path.join(PID_DIR, fname)
        try:
            with open(pf) as f:
                pid = int(f.read().strip())
            os.kill(pid, signal.SIGTERM)
            _log(f"KILLED {fname.replace('.pid', '')} PID={pid}")
            killed += 1
        except (ValueError, ProcessLookupError, PermissionError):
            _log(f"STALE {fname} — process already gone")
        os.remove(pf)

    _log(f"Fleet shutdown: {killed} instances terminated")


def status_fleet() -> None:
    """Print status of all managed instances."""
    if not os.path.isdir(PID_DIR):
        print("No fleet running.")
        return

    print(f"{'Instance':<15} {'Role':<12} {'PID':<8} {'Status'}")
    print("-" * 50)
    for fname in sorted(os.listdir(PID_DIR)):
        if not fname.endswith(".pid"):
            continue
        instance_id = fname.replace(".pid", "")
        pf = os.path.join(PID_DIR, fname)
        try:
            with open(pf) as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)
            status = "RUNNING"
        except (ValueError, ProcessLookupError, PermissionError):
            pid = 0
            status = "DEAD"

        # Infer role from instance index
        try:
            idx = int(instance_id.split("_")[1])
            role = ROLE_CYCLE[idx % len(ROLE_CYCLE)]
        except (IndexError, ValueError):
            role = "unknown"

        print(f"{instance_id:<15} {role:<12} {pid:<8} {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PicoClaw Fleet Launcher")
    parser.add_argument("--num", type=int, default=8, help="Number of instances (default: 8)")
    parser.add_argument("--kill", action="store_true", help="Kill all managed instances")
    parser.add_argument("--status", action="store_true", help="Show fleet status")
    args = parser.parse_args()

    if args.kill:
        kill_fleet()
    elif args.status:
        status_fleet()
    else:
        launch_fleet(args.num)
