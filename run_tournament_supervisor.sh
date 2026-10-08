#!/usr/bin/env bash
# Supervised runner for [Poker] Tournament S1 (elimination).
# Explicit --competition-id so S2's ID in .env stays untouched.
# Frozen heuristic (AGENT_MODE from .env=heuristic); no --agent, no LLM.
cd "$(dirname "$0")" || exit 1
COMP=cmq6l1gnq0lkz60y9i9d9eca1
n=0
while true; do
  n=$((n+1))
  echo "[tsupervisor] === start #$n $(date -u +%FT%TZ) ==="
  PYTHONUNBUFFERED=1 AGENT_MODE=heuristic uv run python -u examples/playground_runner.py --competition-id "$COMP"
  code=$?
  echo "[tsupervisor] runner exited code=$code at $(date -u +%FT%TZ); restarting in 10s"
  # code 5 = 402 fee required (ticket exhausted) — stop, do not loop-pay.
  if [ "$code" = "5" ]; then echo "[tsupervisor] 402 fee required — STOPPING (no auto-pay)"; exit 5; fi
  sleep 10
done
