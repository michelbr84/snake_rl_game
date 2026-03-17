#!/bin/bash
# Block dangerous commands before execution
# Triggered by: Bash tool

COMMAND="${CLAUDE_COMMAND:-}"

# Block destructive operations on checkpoints
if echo "$COMMAND" | grep -qE "rm\s+(-rf?\s+)?checkpoints"; then
    echo "BLOCKED: Cannot delete checkpoints directory. Ask user first."
    exit 1
fi

# Block deleting the entire project
if echo "$COMMAND" | grep -qE "rm\s+-rf\s+\.$|rm\s+-rf\s+/"; then
    echo "BLOCKED: Cannot delete project root."
    exit 1
fi

# Block force push to main
if echo "$COMMAND" | grep -qE "git\s+push\s+--force\s+origin\s+main"; then
    echo "BLOCKED: Force push to main not allowed. Use a branch."
    exit 1
fi
