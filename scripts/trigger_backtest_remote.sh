#!/bin/bash
# trigger_backtest_remote.sh
# Called by Nanobot on Dell to trigger a backtest on the Mac and notify Discord on completion.
# Usage: ./trigger_backtest_remote.sh [strategy_name]

STRATEGY=${1:-"all"}
MAC_HOST="mac"
MAC_PROJECT="/Users/kpakpo/RBI_Swarm/moon-dev-ai-agents-for-trading"
LOG_FILE="~/backtest_$(date +%Y%m%d_%H%M%S).log"
DISCORD_TOKEN="MTQ3MDI2NjgxNTIxNjA5NTMwNA.Gz49sa.jKIz2Kfh9QOn3i7-LytvBgnTEZH1CMpniehmF8"
DISCORD_USER_ID="831076116492058624"

echo "[$(date)] Triggering backtest on Mac for strategy: $STRATEGY"

ssh "$MAC_HOST" "
  nohup bash -c '
    cd $MAC_PROJECT
    source venv/bin/activate
    python run_standardized_backtest.py > $LOG_FILE 2>&1
    EXIT_CODE=\$?

    # Get DM channel ID
    CHANNEL=\$(curl -s -X POST https://discord.com/api/v10/users/@me/channels \
      -H \"Authorization: Bot $DISCORD_TOKEN\" \
      -H \"Content-Type: application/json\" \
      -d \"{\\\"recipient_id\\\": \\\"$DISCORD_USER_ID\\\"}\" | python3 -c \"import sys,json; print(json.load(sys.stdin)[\\\"id\\\"])\")

    # Send result notification
    if [ \$EXIT_CODE -eq 0 ]; then
      MSG=\"✅ Backtest complete for *$STRATEGY*. Log: $LOG_FILE\"
    else
      MSG=\"❌ Backtest FAILED for *$STRATEGY* (exit \$EXIT_CODE). Check: $LOG_FILE\"
    fi

    curl -s -X POST \"https://discord.com/api/v10/channels/\$CHANNEL/messages\" \
      -H \"Authorization: Bot $DISCORD_TOKEN\" \
      -H \"Content-Type: application/json\" \
      -d \"{\\\"content\\\": \\\"\$MSG\\\"}\"
  ' > /dev/null 2>&1 &
  echo \$!
"

echo "[$(date)] Backtest launched on Mac. You will be notified on Discord when complete."
