#!/usr/bin/env bash
# Thin supervisor: keeps the Playground runner alive across transient network
# blips (e.g. DNS getaddrinfo failures on join/re-join that crash the loop).
# Does NOT touch agent decision logic, ranges, mode, or rebuy — pure ops layer.
# Restarts the runner whenever it exits, with a short backoff.
cd "$(dirname "$0")" || exit 1
n=0
while true; do
  n=$((n+1))
  echo "[supervisor] === start #$n $(date -u +%FT%TZ) ==="
  PYTHONUNBUFFERED=1 uv run python -u examples/playground_runner.py
  code=$?
  echo "[supervisor] runner exited code=$code at $(date -u +%FT%TZ); restarting in 10s"
  sleep 10
done
