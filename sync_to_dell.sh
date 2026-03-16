#!/usr/bin/env bash
# sync_to_dell.sh
# Syncs recently updated breeder/orchestrator files from Mac to Dell
# Usage: ./sync_to_dell.sh [DELL_USER@DELL_IP]
# Example: ./sync_to_dell.sh kpakpo@100.74.67.56

DELL_TARGET=${1:-"kpakpo@100.74.67.56"}
# We remove the ~/ as scp relative paths are home-based by default
DEST_DIR="moon-dev-ai-agents-for-trading"

echo "========================================="
echo "  Deploying Breeder Innovations to Dell"
echo "  Target: $DELL_TARGET"
echo "========================================="

echo "[1/3] Syncing picoclaw_main.py..."
scp picoclaw_main.py "$DELL_TARGET:$DEST_DIR/"

echo "[2/3] Syncing rbi_core/dashboard/run.py..."
scp rbi_core/dashboard/run.py "$DELL_TARGET:$DEST_DIR/rbi_core/dashboard/"

echo "[3/3] Syncing rbi_core/strategy/base.py..."
scp rbi_core/strategy/base.py "$DELL_TARGET:$DEST_DIR/rbi_core/strategy/"


echo "========================================="
echo "  Sync Complete! ✅"
echo "========================================="
echo "To restart the breeder on the Dell, connect to it and run:"
echo "  cd $DEST_DIR"
echo "  source venv/bin/activate"
echo "  python picoclaw_main.py --instance_id dell-01 --role breeder"
