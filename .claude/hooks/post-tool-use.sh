#!/bin/bash
# Auto-run pytest after Python file edits
# Triggered by: Edit tool on .py files

FILE_PATH="${CLAUDE_FILE_PATH:-}"

# Only run for Python files
if [[ "$FILE_PATH" == *.py ]]; then
    cd "$(git rev-parse --show-toplevel 2>/dev/null || echo ".")"

    # Check if .venv exists
    if [[ -f ".venv/Scripts/python.exe" ]]; then
        PYTHON=".venv/Scripts/python.exe"
    elif [[ -f ".venv/bin/python" ]]; then
        PYTHON=".venv/bin/python"
    else
        PYTHON="python"
    fi

    echo "Running pytest..."
    $PYTHON -m pytest tests/ -q --tb=short 2>&1 | tail -5
fi
