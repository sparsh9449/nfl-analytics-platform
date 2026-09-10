#!/bin/bash
# Start the NFL Analytics API server.
# Usage:
#   ./run_api.sh           # production (port 8000)
#   ./run_api.sh --reload  # development with auto-reload

set -e
cd "$(dirname "$0")"
source .venv/bin/activate

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

exec uvicorn api.main:app --host "$HOST" --port "$PORT" "$@"
