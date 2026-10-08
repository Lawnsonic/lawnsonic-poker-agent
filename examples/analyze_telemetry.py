"""analyze_telemetry.py — Poker agent telemetry failure analyzer.

Reads hand_telemetry.jsonl and produces a ranked report showing:
  - EV by position (which seat bleeds chips)
  - EV by hand class (which holdings lose most)
  - EV by board texture (dry/wet/neutral performance)
  - EV by action type (are our folds/calls/raises profitable?)
  - EV by street (preflop vs postflop leaks)
  - Opponent profile summary (who is beating us)
  - Worst individual spots (top 10 chip-bleeding decisions)

Usage:
    python analyze_telemetry.py                          # reads hand_telemetry.jsonl
    python analyze_telemetry.py --file path/to/file.jsonl
    python analyze_telemetry.py --out report.txt         # also writes to file
    python analyze_telemetry.py --min-sample 5           # filter low-sample buckets

Output is a ranked text report. Paste into Claude for strategy patches.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


# ─── Data loading ────────────────────────────────────────────────────

def load_telemetry(path: Path) -> list[dict]:
    records = []
    errors = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                errors += 1
                if errors <= 3:
                    print(f"  [warn] line {i} parse error: {e}", file=sys.stderr)
    if errors:
        print(f"  [warn] {errors} malformed lines skipped", file=sys.stderr)
    return records


# ─── EV proxy ────────────────────────────────────────────────────────
# We don't have chip-delta per decision in telemetry (no showdown result).
# Best available proxy: equity_minus_pot_odds = equity - pot_odds.
# Positive = we had the better of it. Negative = we were behind the price.
# For fold actions: proxy = 0 (neutral — we avoided a -EV call).
# This is imperfect but directionally correct for identifying leaks.

def ev_proxy(record: dict) -> float:
    action = record.get("action", "")
    equity = record.get("equity", 0.5)
    pot_odds = record.get("pot_odds", 0.0)
    if action == "fold":
        return 0.0
    return round(equity - pot_odds, 4)


# ─── Aggregator ──────────────────────────────────────────────────────

class Bucket:
    def __init__(self):
        self.count = 0
        self.ev_sum = 0.0
        self.actions: dict[str, int] = defaultdict(int)

    def add(self, record: dict):
        self.count += 1
        self.ev_sum += ev_proxy(record)
        self.actions[record.get("action", "unknown")] += 1

    @property
    def mean_ev(self) -> float:
        return round(self.ev_sum / self.count, 4) if self.count else 0.0

    def action_summary(self) -> str:
        total = sum(self.actions.values())
        parts = []
        for a, n in sorted(self.actions.items(), key=lambda x: -x[1]):
            parts.append(f"{a}:{n}({int(n/total*100)}%)")
        return " ".join(parts)


def aggregate(records: list[dict], key_fn) -> dict[str, Bucket]:
    buckets: dict[str, Bucket] = defaultdict(Bucket)
    for r in records:
        k = key_fn(r)
        if k:
            buckets[k].add(r)
    return buckets


# ─── Report sections ─────────────────────────────────────────────────

def _rank_table(buckets: dict[str, Bucket], label: str,
                min_sample: int, lines: list[str]):
    lines.append(f"\n{'─'*60}")
    lines.append(f"  {label}")
    lines.append(f"{'─'*60}")
    ranked = sorted(
        [(k, b) for k, b in buckets.items() if b.count >= min_sample],
        key=lambda x: x[1].mean_ev
    )
    if not ranked:
        lines.append(f"  (no buckets with >= {min_sample} samples)")
        return
    lines.append(f"  {'Key':<18} {'N':>5}  {'MeanEV':>8}  Actions")
    lines.append(f"  {'-'*18} {'-'*5}  {'-'*8}  {'-'*30}")
    for key, b in ranked:
        flag = "  ← LEAK" if b.mean_ev < -0.05 else ("  ← STRONG" if b.mean_ev > 0.10 else "")
        lines.append(
            f"  {key:<18} {b.count:>5}  {b.mean_ev:>+8.4f}  {b.action_summary()}{flag}"
        )


def build_report(records: list[dict], min_sample: int = 3) -> str:
    lines: list[str] = []
    n = len(records)

    lines.append("=" * 60)
    lines.append("  POKER AGENT TELEMETRY REPORT")
    lines.append("=" * 60)
    lines.append(f"  Total decisions: {n}")
    if n == 0:
        lines.append("  No records found. Run some hands first.")
        return "\n".join(lines)

    # ── Overview ──────────────────────────────────────────────────
    action_counts: dict[str, int] = defaultdict(int)
    street_counts: dict[str, int] = defaultdict(int)
    total_ev = 0.0
    for r in records:
        action_counts[r.get("action", "?")] += 1
        street_counts[r.get("street", "?")] += 1
        total_ev += ev_proxy(r)

    lines.append(f"  Mean EV proxy (all hands): {total_ev/n:+.4f}")
    lines.append(f"  Actions: " + "  ".join(
        f"{a}:{c}" for a, c in sorted(action_counts.items(), key=lambda x: -x[1])
    ))
    lines.append(f"  Streets: " + "  ".join(
        f"{s}:{c}" for s, c in sorted(street_counts.items(), key=lambda x: -x[1])
    ))

    # ── By position ───────────────────────────────────────────────
    _rank_table(
        aggregate(records, lambda r: r.get("position") or "?"),
        "EV BY POSITION  (worst = biggest leak by seat)",
        min_sample, lines
    )

    # ── By street ─────────────────────────────────────────────────
    _rank_table(
        aggregate(records, lambda r: r.get("street") or "?"),
        "EV BY STREET",
        min_sample, lines
    )

    # ── By texture ────────────────────────────────────────────────
    _rank_table(
        aggregate(records, lambda r: r.get("texture") or "?"),
        "EV BY BOARD TEXTURE",
        min_sample, lines
    )

    # ── By action ─────────────────────────────────────────────────
    _rank_table(
        aggregate(records, lambda r: r.get("action") or "?"),
        "EV BY ACTION TYPE",
        min_sample, lines
    )

    # ── By hand class ─────────────────────────────────────────────
    _rank_table(
        aggregate(records, lambda r: r.get("hand_class") or "?"),
        "EV BY HAND CLASS  (top 15 worst)",
        min_sample, lines
    )

    # ── Position × Street cross-tab ───────────────────────────────
    _rank_table(
        aggregate(records, lambda r: f"{r.get('position','?')}_{r.get('street','?')}"),
        "EV BY POSITION × STREET  (combined leak locator)",
        min_sample, lines
    )

    # ── Position × Texture ────────────────────────────────────────
    _rank_table(
        aggregate(records, lambda r: f"{r.get('position','?')}_{r.get('texture','?')}"),
        "EV BY POSITION × TEXTURE",
        min_sample, lines
    )

    # ── Opponent profiles ─────────────────────────────────────────
    lines.append(f"\n{'─'*60}")
    lines.append("  OPPONENT PROFILES  (accumulated across hands)")
    lines.append(f"{'─'*60}")
    all_opponents: dict[str, dict] = {}
    for r in records:
        opps = r.get("opponents") or {}
        for agent_id, stats in opps.items():
            if agent_id not in all_opponents:
                all_opponents[agent_id] = stats.copy()
            else:
                # Take latest snapshot (stats accumulate in the agent process)
                if stats.get("sample", 0) >= all_opponents[agent_id].get("sample", 0):
                    all_opponents[agent_id] = stats.copy()

    if not all_opponents:
        lines.append("  (no opponent data — run live hands to populate)")
    else:
        lines.append(f"  {'Agent':<30} {'VPIP':>6} {'PFR':>5} {'Agg%':>6} {'Sample':>7}  Type")
        lines.append(f"  {'-'*30} {'-'*6} {'-'*5} {'-'*6} {'-'*7}  {'-'*15}")
        for agent_id, s in sorted(all_opponents.items(),
                                   key=lambda x: x[1].get("sample", 0), reverse=True):
            vpip = s.get("vpip", 0)
            pfr  = s.get("pfr", 0)
            agg  = s.get("agg_pct", 0.0)
            samp = s.get("sample", 0)
            # Classify style
            if agg > 0.5:
                style = "AGGRO"
            elif vpip > pfr * 2:
                style = "STATION"
            elif vpip < 3:
                style = "NITTER"
            else:
                style = "BALANCED"
            lines.append(f"  {agent_id:<30} {vpip:>6} {pfr:>5} {agg:>6.2f} {samp:>7}  {style}")

    # ── Worst individual spots ────────────────────────────────────
    lines.append(f"\n{'─'*60}")
    lines.append("  TOP 10 WORST INDIVIDUAL SPOTS  (lowest EV proxy)")
    lines.append(f"{'─'*60}")
    scored = [(ev_proxy(r), r) for r in records]
    worst = sorted(scored, key=lambda x: x[0])[:10]
    for ev, r in worst:
        lines.append(
            f"  ev={ev:+.3f}  {r.get('street','?'):8}  {r.get('position','?'):4}  "
            f"{r.get('hand_class','?'):5}  {r.get('texture','?'):7}  "
            f"action={r.get('action','?')}  "
            f"board={r.get('board',[])}  "
            f"eq={r.get('equity',0):.2f}  po={r.get('pot_odds',0):.2f}"
        )

    # ── Strategy patch suggestions ────────────────────────────────
    lines.append(f"\n{'─'*60}")
    lines.append("  AUTO-GENERATED PATCH SUGGESTIONS")
    lines.append(f"{'─'*60}")
    suggestions: list[str] = []

    # Find worst position
    pos_buckets = aggregate(records, lambda r: r.get("position") or "?")
    worst_pos = min(
        [(k, b) for k, b in pos_buckets.items() if b.count >= min_sample],
        key=lambda x: x[1].mean_ev, default=None
    )
    if worst_pos and worst_pos[1].mean_ev < -0.05:
        suggestions.append(
            f"  • Position '{worst_pos[0]}' is your biggest leak "
            f"(ev={worst_pos[1].mean_ev:+.4f}, n={worst_pos[1].count}). "
            f"Tighten opening range or improve postflop play from this seat."
        )

    # Find worst texture
    tex_buckets = aggregate(records, lambda r: r.get("texture") or "?")
    worst_tex = min(
        [(k, b) for k, b in tex_buckets.items() if b.count >= min_sample],
        key=lambda x: x[1].mean_ev, default=None
    )
    if worst_tex and worst_tex[1].mean_ev < -0.05:
        suggestions.append(
            f"  • '{worst_tex[0]}' boards are hurting you "
            f"(ev={worst_tex[1].mean_ev:+.4f}). "
            f"Review bet sizing in SIZING['{worst_tex[0]}'] or fold frequency."
        )

    # Check for calling with low equity
    bad_calls = [r for r in records
                 if r.get("action") == "call"
                 and r.get("equity", 1.0) < r.get("pot_odds", 0.0) - 0.05]
    if bad_calls:
        suggestions.append(
            f"  • {len(bad_calls)} calls made with equity below pot odds "
            f"(equity < pot_odds - 0.05). These are -EV calls. "
            f"Check postflop call threshold in decide()."
        )

    # Check for checking strong hands
    missed_bets = [r for r in records
                   if r.get("action") == "check"
                   and r.get("equity", 0.0) > 0.65
                   and r.get("pot_odds", 1.0) == 0.0]
    if missed_bets:
        suggestions.append(
            f"  • {len(missed_bets)} checks with equity > 65% and no bet facing. "
            f"These are likely missed value bets. "
            f"Consider lowering the value-bet threshold in decide()."
        )

    # Aggro opponents
    aggro = [aid for aid, s in all_opponents.items()
             if s.get("agg_pct", 0) > 0.5 and s.get("sample", 0) >= 5]
    if aggro:
        suggestions.append(
            f"  • Aggro opponents detected: {aggro}. "
            f"Consider tightening call range vs these agents and widening 3-bet range."
        )

    if not suggestions:
        suggestions.append("  • No strong leak signals yet — need more hands.")
    lines.extend(suggestions)

    lines.append(f"\n{'='*60}")
    lines.append("  END OF REPORT — paste into Claude for strategy patches")
    lines.append(f"{'='*60}\n")

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Poker telemetry failure analyzer")
    parser.add_argument("--file", default="hand_telemetry.jsonl",
                        help="Path to telemetry JSONL file")
    parser.add_argument("--out", default=None,
                        help="Also write report to this file")
    parser.add_argument("--min-sample", type=int, default=3,
                        help="Minimum decisions per bucket to include in report")
    args = parser.parse_args(argv)

    path = Path(args.file)
    if not path.exists():
        print(f"ERROR: {path} not found. Run some hands first.", file=sys.stderr)
        return 1

    print(f"Loading {path}...", file=sys.stderr)
    records = load_telemetry(path)
    print(f"Loaded {len(records)} records.", file=sys.stderr)

    report = build_report(records, min_sample=args.min_sample)
    print(report)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(report, encoding="utf-8")
        print(f"\nReport written to {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())