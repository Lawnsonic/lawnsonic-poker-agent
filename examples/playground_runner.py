"""examples/playground_runner.py — Playground S1 (live PvP) loop.

Wraps the existing decide() from examples/agent.py unchanged.

Endpoint diff vs the Eval benchmark runner in agent.py:

  Eval (agent.py)                 Playground (this file)
  ──────────────────────────────  ──────────────────────────────────
  POST /texas/benchmark/start     POST /texas/join
  GET  /texas/benchmark/status    GET  /texas/lobby
  GET  /texas/pending-actions     GET  /texas/pending-actions  (same)
  POST /texas/action              POST /texas/action            (same)

The loop is continuous — no fixed hand target. After each table ends the
agent re-joins the matchmaking queue automatically.

Hard limits enforced here (never crossed without user approval):
  - 402 on /texas/join → STOP immediately; print fee details; never pay.
  - Rebuy calls are not made.
  - No new agent registration without explicit user instruction.

CLI:
    uv run examples/playground_runner.py
    uv run examples/playground_runner.py --competition-id cmpy2qy65002ud9ej6b7jjq0l
    uv run examples/playground_runner.py --dry-run
    uv run examples/playground_runner.py --max-hands 10
    uv run examples/playground_runner.py --agent path/to/decide.py

Environment (loaded from .env):
    ARENA_COMPETITION_ID   competition to join
    AGENT_MODE             heuristic | analyst | llm  (default: heuristic)
    OPENAI_API_KEY         required for analyst / llm modes
    OPENROUTER_MODEL       model string (default: anthropic/claude-haiku-4.5)
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from typing import Any, Optional

from dotenv import load_dotenv

from arena_client import (
    ArenaClient,
    ArenaError,
    DEFAULT_BASE,
    load_or_register,
    load_state,
    save_state,
)
from agent import (
    POLL_INTERVAL,
    POLL_JITTER,
    _FALLBACK_REASONING,
    _normalize_action_name,
    _safe_research_context,
    _validate_pending_tables,
    decide,
    load_external_decide,
    retrieve_solver_context,
)

# How often to check /texas/lobby when no tables are pending.
LOBBY_REFRESH_S = 15.0

# Seconds with no pending tables + lobby=null before triggering a re-join.
# Conservative: in 6-max with 5 opponents each using their full 60 s clock
# back-to-back, the gap between our actions can reach ~5 min. Set well above
# that so we never false-positive re-join mid-hand.
IDLE_REJOIN_S = 360.0


# ─── Playground-specific API helpers ──────────────────────────────────

def _join(client: ArenaClient, competition_id: str) -> dict:
    """POST /texas/join.

    May raise ArenaError(402) — caller MUST stop and surface fee details;
    this function never pays automatically.
    Returns the raw response dict on success.
    """
    return client.post("/texas/join", {"competitionId": competition_id})


def _get_lobby(client: ArenaClient, competition_id: str) -> Optional[dict]:
    """GET /texas/lobby → lobby position dict, or None when not in queue.

    None means either seated at an active table, or not yet joined / between
    tables. Errors are logged and treated as None so the loop stays alive.
    """
    try:
        resp = client.get(f"/texas/lobby?competitionId={competition_id}")
        if isinstance(resp, dict):
            return resp.get("lobby")   # server returns null when not queued
    except ArenaError as e:
        print(f"[playground] lobby check error: {e}", file=sys.stderr)
    return None


def _do_join(client: ArenaClient, competition_id: str,
             label: str) -> tuple[bool, Optional[int]]:
    """Call _join and print result. Returns (joined_ok, bankroll_chips).

    Handles 402 (prints hard stop) and 409 (already queued, fine).
    Raises on any other ArenaError so the outer loop can decide.
    """
    try:
        resp = _join(client, competition_id)
        if isinstance(resp, dict):
            li   = resp.get("lobby") or {}
            part = resp.get("participant") or {}
            bk   = part.get("totalChips")
            print(f"[playground{label}] joined queue: "
                  f"pos={li.get('position')}/{li.get('total')} "
                  f"bankroll={bk}")
            return True, bk
    except ArenaError as e:
        if e.status == 409:
            print(f"[playground{label}] already in queue (409), continuing")
            return True, None
        if e.status == 402:
            print(
                f"\n[playground{label}] *** ENTRY FEE REQUIRED ***\n"
                f"  This competition requires payment before joining.\n"
                f"  Details from server: {e}\n"
                f"  Pay manually, then re-run. This runner will NOT pay automatically.",
                file=sys.stderr,
            )
            return False, None
        raise
    return False, None


def _heartbeat(label: str, pending: int, lobby: Optional[dict],
               hands: int, bankroll: Optional[int]) -> None:
    queue_str = (f"queue_pos={lobby['position']}/{lobby['total']}"
                 if isinstance(lobby, dict) else "at_table_or_idle")
    br_str    = f"bankroll={bankroll}" if bankroll is not None else "bankroll=?"
    print(f"[playground{label}] pending={pending} | {queue_str} | "
          f"hands={hands} | {br_str}")


# ─── Core loop ────────────────────────────────────────────────────────

def _run_playground_loop(
    client: ArenaClient,
    args: argparse.Namespace,
    competition_id: str,
    decide_fn: Any,
    retrieve_fn: Any,
    label: str = "",
) -> int:
    state    = load_state()
    rng      = random.Random()
    hands    = 0
    bankroll: Optional[int] = None

    last_lobby_at  = 0.0
    last_action_at = time.monotonic()   # reset whenever we POST an action
    last_hb_at     = 0.0

    # ── Initial join ──────────────────────────────────────────────────
    print(f"[playground{label}] joining competition {competition_id} ...")
    ok, bk = _do_join(client, competition_id, label)
    if not ok:
        return 5   # 402 fee required — caller should surface this
    if bk is not None:
        bankroll = bk

    # ── Poll loop ─────────────────────────────────────────────────────
    while True:
        tables: list[dict] = []
        try:
            pending = client.get(
                f"/texas/pending-actions?competitionId={competition_id}")
            tables = _validate_pending_tables(pending)
            # Prioritise tables closest to their action deadline.
            tables = sorted(tables,
                            key=lambda t: t.get("actionDeadlineAt") or 0)
        except ArenaError as e:
            print(f"[playground{label}] pending-actions error: {e}",
                  file=sys.stderr)

        # ── Act on each pending table ──────────────────────────────
        for table in tables:
            deadline_ms = table.get("actionDeadlineAt") or 0
            deadline_s  = (max(0.0, deadline_ms / 1000.0 - time.time())
                           if deadline_ms else 30.0)
            ctx = _safe_research_context(table, retrieve_fn)

            # decide() is called as a black box — no poker logic here.
            try:
                action = decide_fn(table, deadline_s=deadline_s,
                                   research_context=ctx)
            except TypeError:
                action = decide_fn(table, deadline_s=deadline_s)

            action  = _normalize_action_name(action)
            payload = {"tableId": table["tableId"], **action}

            try:
                resp = client.post("/texas/action", payload)
                hands          += 1
                last_action_at  = time.monotonic()

                # Track bankroll from action response.
                if isinstance(resp, dict):
                    bk = (resp.get("participant") or {}).get("totalChips")
                    if bk is not None:
                        bankroll = bk

                state["hands_played"] = state.get("hands_played", 0) + 1
                state["last_action"]  = {
                    "action": action["action"],
                    "amount": action.get("amount"),
                    "at":     int(time.time()),
                }
                save_state(state)

            except ArenaError as e:
                if e.status == 409:
                    continue   # stale — skip, re-poll
                if e.status == 400:
                    state["rejection_count"] = \
                        state.get("rejection_count", 0) + 1
                    save_state(state)
                    try:
                        client.post("/texas/action", {
                            "tableId":  table["tableId"],
                            "action":   "fold",
                            "message":  "fallback after illegal action",
                            "reasoning": _FALLBACK_REASONING,
                        })
                    except ArenaError:
                        pass
                    continue
                raise

        # ── --max-hands early exit ────────────────────────────────
        if args.max_hands and hands >= args.max_hands:
            print(f"[playground{label}] reached --max-hands={args.max_hands}, stopping")
            return 0

        # ── Bankroll exhausted ────────────────────────────────────
        if bankroll is not None and bankroll < 2:
            print(
                f"[playground{label}] bankroll={bankroll} is below one big blind.\n"
                f"  Season rules: no rebuy. Stopping. Contact the user to handle manually.",
                file=sys.stderr,
            )
            return 6

        # ── Periodic lobby check + re-join ────────────────────────
        now = time.monotonic()
        if not tables and now - last_lobby_at >= LOBBY_REFRESH_S:
            last_lobby_at = now
            lobby = _get_lobby(client, competition_id)
            idle  = now - last_action_at

            _heartbeat(label, len(tables), lobby, hands, bankroll)
            last_hb_at = now

            if lobby is None and idle >= IDLE_REJOIN_S:
                # No queue position + no pending tables + idle long enough
                # → table has ended; re-enter the queue.
                print(f"[playground{label}] idle {idle:.0f}s with no queue "
                      "position — re-joining ...")
                ok, bk = _do_join(client, competition_id, label)
                if not ok:
                    return 5   # 402 on re-join
                if bk is not None:
                    bankroll = bk
                last_action_at = time.monotonic()   # reset idle clock

        elif tables and now - last_hb_at >= 5.0:
            _heartbeat(label, len(tables), None, hands, bankroll)
            last_hb_at = now

        if not tables:
            time.sleep(POLL_INTERVAL + rng.uniform(-POLL_JITTER, POLL_JITTER))


# ─── Dry-run (no network) ─────────────────────────────────────────────

_DRY_SCENARIOS = [
    {   # Flop: facing a bet with top pair
        "tableId": "dry-pg-1",
        "selfSeatNumber": 1,
        "dealerSeatNumber": 3,
        "street": "Flop",
        "potChips": 40,
        "boardCards": ["Ah", "Kd", "7c"],
        "seats": [
            {"seatNumber": 1, "stackChips": 180, "holeCards": ["As", "Ks"]},
            {"seatNumber": 2, "stackChips": 200, "holeCards": []},
            {"seatNumber": 3, "stackChips": 150, "holeCards": []},
        ],
        "allowedActions": {
            "availableActions": ["call", "raise", "fold"],
            "canCall": True, "canRaise": True, "canFold": True,
            "callChips": 20,
            "raiseRange": {"min": 40, "max": 180},
        },
        "recentEvents": [],
    },
    {   # Preflop: UTG open opportunity
        "tableId": "dry-pg-2",
        "selfSeatNumber": 2,
        "dealerSeatNumber": 1,
        "street": "Preflop",
        "potChips": 3,
        "boardCards": [],
        "seats": [
            {"seatNumber": 1, "stackChips": 200, "holeCards": []},
            {"seatNumber": 2, "stackChips": 198, "holeCards": ["Kh", "Qs"]},
            {"seatNumber": 3, "stackChips": 199, "holeCards": []},
        ],
        "allowedActions": {
            "availableActions": ["raise", "fold", "check"],
            "canRaise": True, "canFold": True, "canCheck": True,
            "raiseRange": {"min": 4, "max": 198},
        },
        "recentEvents": [],
    },
    {   # River: free option with weak hand
        "tableId": "dry-pg-3",
        "selfSeatNumber": 1,
        "dealerSeatNumber": 2,
        "street": "River",
        "potChips": 120,
        "boardCards": ["2h", "7d", "Tc", "4s", "9h"],
        "seats": [
            {"seatNumber": 1, "stackChips": 80, "holeCards": ["6c", "5c"]},
            {"seatNumber": 2, "stackChips": 140, "holeCards": []},
        ],
        "allowedActions": {
            "availableActions": ["check", "bet"],
            "canCheck": True, "canBet": True,
            "betRange": {"min": 2, "max": 80},
        },
        "recentEvents": [],
    },
]


def _run_dry_run(args: argparse.Namespace,
                 decide_fn: Any, retrieve_fn: Any) -> int:
    label = " (dry-run)"
    print(f"[playground{label}] simulating join -> joined queue: pos=1/4 bankroll=1000")

    hands      = 0
    max_hands  = args.max_hands or len(_DRY_SCENARIOS)
    scenarios  = (_DRY_SCENARIOS * ((max_hands // len(_DRY_SCENARIOS)) + 1))[:max_hands]

    for i, scenario in enumerate(scenarios, 1):
        table = dict(scenario)
        table["actionDeadlineAt"] = int((time.time() + 30) * 1000)
        ctx = _safe_research_context(table, retrieve_fn)

        try:
            action = decide_fn(table, deadline_s=29.0, research_context=ctx)
        except TypeError:
            action = decide_fn(table, deadline_s=29.0)

        action = _normalize_action_name(action)
        print(f"[playground{label}] hand={i}/{max_hands} "
              f"street={table['street']} "
              f"action={action['action']} "
              f"amount={action.get('amount', '-')} "
              f"reasoning={str(action.get('reasoning',''))[:80]}")
        hands += 1

    print(f"\n[playground{label}] dry-run complete: {hands} hand(s) decided, no errors.")
    print(f"[playground{label}] analyst mode active: "
          f"{os.environ.get('AGENT_MODE','heuristic').lower() == 'analyst'}")
    return 0


# ─── Entry point ──────────────────────────────────────────────────────

def run_playground(args: argparse.Namespace,
                   decide_fn: Optional[Any] = None) -> int:
    load_dotenv()

    competition_id = (args.competition_id
                      or os.environ.get("ARENA_COMPETITION_ID"))
    if not competition_id:
        print(
            "ERROR: no competition ID specified.\n"
            "  Set ARENA_COMPETITION_ID in .env or pass --competition-id.",
            file=sys.stderr,
        )
        return 2

    decide_fn   = decide_fn or decide
    retrieve_fn = retrieve_solver_context

    if args.dry_run:
        return _run_dry_run(args, decide_fn, retrieve_fn)

    api_key = os.environ.get("ARENA_API_KEY") or None
    base    = os.environ.get("ARENA_API_BASE", DEFAULT_BASE)
    client  = ArenaClient(base, api_key=api_key)

    try:
        creds    = load_or_register(client, args.handle, args.name, args.quote)
        agent_id = creds.get("agentId") or creds.get("id") or "?"
        print(f"[playground] agent={agent_id} base={base} "
              f"mode={os.environ.get('AGENT_MODE','heuristic')}")
        return _run_playground_loop(
            client=client,
            args=args,
            competition_id=competition_id,
            decide_fn=decide_fn,
            retrieve_fn=retrieve_fn,
        )
    finally:
        client.close()


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        description="Arena Playground S1 — live PvP runner")
    p.add_argument("--competition-id", default=None,
                   help="Playground competition ID (else ARENA_COMPETITION_ID)")
    p.add_argument("--dry-run", action="store_true",
                   help="Simulated loop — no network calls")
    p.add_argument("--max-hands", type=int, default=0,
                   help="Stop after N actions (0 = run indefinitely)")
    p.add_argument("--agent", default=None,
                   help="Path to a .py file exposing decide() — "
                        "overrides the built-in heuristic")
    # Credential fallback args — only used if .arena-credentials is absent.
    p.add_argument("--handle", default="pokerkit-starter")
    p.add_argument("--name",   default="PokerKit Starter")
    p.add_argument("--quote",  default="probability over swagger")
    args = p.parse_args(argv)

    decide_fn = decide
    if args.agent:
        decide_fn = load_external_decide(args.agent)
        print(f"[playground] using external decide() from {args.agent}")

    return run_playground(args, decide_fn=decide_fn)


if __name__ == "__main__":
    sys.exit(main())
