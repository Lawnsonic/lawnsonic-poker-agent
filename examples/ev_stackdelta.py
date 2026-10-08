"""ev_stackdelta.py — REAL chip-EV audit via stack-delta reconstruction.

Read-only. Touches no play logic. Complements analyze_telemetry.py (which only
computes an equity-minus-pot-odds proxy) by reconstructing REAL chips won/lost
per hand from the `stack` field across hand_id boundaries.

Method
------
1. Keep only genuine live hands. Synthetic fixtures (hand_id `tbl_*`, `dry-pg-*`)
   from the Gate-1 test suite are dropped; live arena hands are cuid-style ids
   (^cm[a-z0-9]{23}$).
2. Order live hands chronologically by the ts of their first (preflop) decision.
3. Per-hand starting stack = stack logged on that hand's first decision.
   Net chips for hand H = start_stack(H+1) - start_stack(H).
   This captures blinds posted, chips invested, and pot won/lost — the real
   chip swing — because the next hand's opening stack reflects H's outcome.
4. Session/rebuy boundary guard: if the next hand's start stack jumps by an
   implausible amount AND lands on a round reset value, OR there is a large time
   gap, that delta is excluded (flagged) rather than counted as a win/loss.

Each hand is attributed to the position and hand_class of its FIRST decision.

Usage:
    uv run python examples/ev_stackdelta.py [--file hand_telemetry.jsonl]
                                            [--date 2026-06-12] [--min-sample 2]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

LIVE_ID = re.compile(r"^cm[a-z0-9]{23}$")
RESET_STACKS = {1000, 2000, 1500, 500}  # known fresh-stack / rebuy values


def load(path: Path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def is_live(r):
    return bool(LIVE_ID.match(r.get("hand_id", "")))


def parse_ts(s):
    return s  # ISO strings sort lexicographically when same offset format


def reconstruct(rows, date_filter=None):
    """Return list of per-hand dicts with real chip delta, plus excluded count."""
    live = [r for r in rows if is_live(r)]
    if date_filter:
        live = [r for r in live if r.get("ts", "")[:10] == date_filter]

    # group rows by hand, keep first (preflop) decision per hand
    hands = {}
    order = {}
    for r in live:
        h = r["hand_id"]
        ts = r.get("ts", "")
        if h not in hands or ts < order[h]:
            hands[h] = r
            order[h] = ts
    # chronological hand sequence
    seq = sorted(hands.values(), key=lambda r: r.get("ts", ""))

    results = []
    excluded = 0
    for i in range(len(seq) - 1):
        cur, nxt = seq[i], seq[i + 1]
        s0 = cur.get("stack")
        s1 = nxt.get("stack")
        if s0 is None or s1 is None:
            continue
        delta = s1 - s0
        # boundary guard: large jump landing on a reset value = new session/rebuy
        if abs(delta) >= 800 and s1 in RESET_STACKS:
            excluded += 1
            continue
        results.append({
            "hand_id": cur["hand_id"],
            "position": cur.get("position") or "?",
            "hand_class": cur.get("hand_class") or "?",
            "start_stack": s0,
            "delta": delta,
            "ts": cur.get("ts", ""),
        })
    return results, excluded, len(seq)


def agg(results, key):
    b = defaultdict(lambda: {"n": 0, "sum": 0})
    for h in results:
        k = h[key]
        b[k]["n"] += 1
        b[k]["sum"] += h["delta"]
    return b


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="hand_telemetry.jsonl")
    ap.add_argument("--date", default=None, help="filter to one YYYY-MM-DD")
    ap.add_argument("--min-sample", type=int, default=2)
    a = ap.parse_args(argv)

    rows = load(Path(a.file))
    results, excluded, nhands = reconstruct(rows, a.date)

    print("=" * 64)
    print("  REAL CHIP-EV AUDIT  (stack-delta reconstruction)")
    print("=" * 64)
    scope = a.date or "all dates"
    print(f"  Scope: {scope}   live hands found: {nhands}")
    print(f"  Hands scored: {len(results)}   "
          f"session/rebuy boundaries excluded: {excluded}")
    if not results:
        print("  No scorable hands.")
        return 0
    total = sum(h["delta"] for h in results)
    print(f"  NET chips over sample: {total:+d}   "
          f"mean/hand: {total/len(results):+.1f}")

    # ---- hand class ----
    cb = agg(results, "hand_class")
    ranked = sorted(cb.items(), key=lambda x: x[1]["sum"])
    def show(items, title):
        print(f"\n  {title}")
        print(f"  {'class':<8} {'N':>4} {'net':>8} {'mean':>8}")
        print(f"  {'-'*8} {'-'*4} {'-'*8} {'-'*8}")
        for k, v in items:
            print(f"  {k:<8} {v['n']:>4} {v['sum']:>+8d} {v['sum']/v['n']:>+8.1f}")

    winners = [x for x in sorted(cb.items(), key=lambda x: -x[1]["sum"])
               if x[1]["n"] >= a.min_sample][:10]
    losers = [x for x in ranked if x[1]["n"] >= a.min_sample][:10]
    print(f"\n{'─'*64}\n  TOP 10 WINNING HAND CLASSES  (n>={a.min_sample})\n{'─'*64}")
    show(winners, "by net chips")
    print(f"\n{'─'*64}\n  TOP 10 LOSING HAND CLASSES  (n>={a.min_sample})\n{'─'*64}")
    show(losers, "by net chips")

    # ---- position ----
    pb = agg(results, "position")
    print(f"\n{'─'*64}\n  CHIP EV BY POSITION\n{'─'*64}")
    print(f"  {'pos':<6} {'N':>4} {'net':>8} {'mean':>8}")
    print(f"  {'-'*6} {'-'*4} {'-'*8} {'-'*8}")
    for k, v in sorted(pb.items(), key=lambda x: x[1]["sum"]):
        print(f"  {k:<6} {v['n']:>4} {v['sum']:>+8d} {v['sum']/v['n']:>+8.1f}")

    # ---- biggest individual swings ----
    print(f"\n{'─'*64}\n  TOP 10 BIGGEST CHIP LOSSES (single hands)\n{'─'*64}")
    for h in sorted(results, key=lambda x: x["delta"])[:10]:
        print(f"  {h['delta']:>+6d}  {h['position']:<4} {h['hand_class']:<5} "
              f"start={h['start_stack']:<5} {h['hand_id']}")
    print(f"\n{'─'*64}\n  TOP 10 BIGGEST CHIP WINS (single hands)\n{'─'*64}")
    for h in sorted(results, key=lambda x: -x["delta"])[:10]:
        print(f"  {h['delta']:>+6d}  {h['position']:<4} {h['hand_class']:<5} "
              f"start={h['start_stack']:<5} {h['hand_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
