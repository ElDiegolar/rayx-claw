#!/bin/bash
# swap_main_agent.sh
# Usage: ./swap_main_agent.sh [claude|minimax]

set -e

if [ $# -ne 1 ]; then
  echo "Usage: $0 [claude|minimax]"
  exit 1
fi

AGENT="$1"
CONFIG_FILE="$(dirname "$0")/config.py"

if [[ "$AGENT" != "claude" && "$AGENT" != "minimax" ]]; then
  echo "Error: agent must be 'claude' or 'minimax'"
  exit 2
fi

# Use sed to update the default model in config.py
if [[ "$AGENT" == "claude" ]]; then
  sed -i 's/^    claude_model: str = ".*"/    claude_model: str = "claude-opus-4-6"/' "$CONFIG_FILE"
  sed -i 's/^    minimax_model: str = ".*"/    minimax_model: str = "MiniMax-M2.5"/' "$CONFIG_FILE"
  echo "Main agent set to Claude."
else
  sed -i 's/^    claude_model: str = ".*"/    claude_model: str = "minimax"/' "$CONFIG_FILE"
  sed -i 's/^    minimax_model: str = ".*"/    minimax_model: str = "MiniMax-M2.5"/' "$CONFIG_FILE"
  echo "Main agent set to MiniMax."
fi
