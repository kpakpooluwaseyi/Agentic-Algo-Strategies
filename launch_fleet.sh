#!/usr/bin/env bash
cd "$(dirname "$0")" && source venv/bin/activate && python picoclaw_fleet_launcher.py "$@"
