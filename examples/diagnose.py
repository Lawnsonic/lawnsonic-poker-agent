"""Poker Diagnostics — telemetry analysis for the Heuristic Learning loop.

Reads hand_telemetry.jsonl (produced by agent.py's _log_decision()) and
generates a structured diagnostic report answering:

  1. Which position loses the most EV?
  2. Which hand classes are leaking?
  3. Which board textures cause problems?
  4. Which actions have the worst EV profile?
  5. Which opponent type hurts us most?

Usage:
    uv run python examples/diagnose.py
    uv run python examples/diagnose.py --out diagnostics.txt
    uv run python examples/diagnose.py --telemetry path/to/file.jsonl

Or via CLI:
    pokerkit diagnose
    pokerkit diagnose --out diagnostics.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional


# ─── Data loading ─────────────────────────────────────────────────────

def _load_telemetry(path: Path) -> list[dict]:
    """Load all records from hand_telemetry.jsonl."""
    if not path.exists():
        print(f"[diagnose] telemetry file not found: {path}", file=sys.stderr)
        print(f"[diagnose] run a benchmark first to generate telemetry:",
              file=sys.stderr)
        print(f"  uv run python examples/cli.py run --max-hands 50",
              file=sys.stderr)
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[diagnose] skipping malformed line {line_num}",
                      file=sys.stderr)
    return records


# ─── EV proxy calculation ────────────────────────────────────────────

def _ev_proxy(record: dict) -> float:
    """Compute an expected-value proxy for a single decision.

    This is NOT the actual chip outcome (we don't have that in telemetry).
    It's a mathematical estimate of the decision quality:

    - fold:  0  (no further investment, EV of folding is baseline)
    - check: equity × pot  (expected share of current pot)
    - call:  equity × (pot + call) - call  (expected gain minus cost)
    - bet/raise: equity × (pot + amount) - amount  (simplified, ignores
                 fold equity — conservative estimate)
    """
    action = record.get("action", "")
    equity = record.get("equity", 0.5)
    pot = record.get("pot", 0)
    call_chips = record.get("call_chips", 0)
    amount = record.get("amount") or 0

    if action == "fold":
        return 0.0
    if action == "check":
        return equity * pot
    if action == "call":
        return equity * (pot + call_chips) - call_chips
    if action in ("bet", "raise", "all-in"):
        # Conservative: doesn't account for fold equity (which is always
        # positive). Real EV is higher than this estimate.
        return equity * (pot + amount) - amount
    return 0.0


# ─── Section builders ────────────────────────────────────────────────

def _section_header(title: str) -> str:
    bar = "═" * 60
    return f"\n{bar}\n  {title}\n{bar}"


def _build_position_analysis(records: list[dict]) -> str:
    """Position Performance: hands, avg equity, action distribution, avg EV."""
    lines = [_section_header("POSITION PERFORMANCE")]

    pos_data: dict[str, dict] = defaultdict(lambda: {
        "count": 0, "equity_sum": 0.0, "ev_sum": 0.0,
        "actions": defaultdict(int), "pot_sum": 0,
    })

    for r in records:
        pos = r.get("position", "?")
        d = pos_data[pos]
        d["count"] += 1
        d["equity_sum"] += r.get("equity", 0)
        d["ev_sum"] += _ev_proxy(r)
        d["actions"][r.get("action", "?")] += 1
        d["pot_sum"] += r.get("pot", 0)

    # Sort by avg EV (worst first)
    sorted_pos = sorted(pos_data.items(),
                        key=lambda x: x[1]["ev_sum"] / max(x[1]["count"], 1))

    for pos, d in sorted_pos:
        n = d["count"]
        avg_eq = d["equity_sum"] / max(n, 1)
        avg_ev = d["ev_sum"] / max(n, 1)
        avg_pot = d["pot_sum"] / max(n, 1)

        # Action breakdown
        acts = d["actions"]
        total_acts = sum(acts.values())
        act_str = ", ".join(
            f"{a}: {c} ({c/total_acts*100:.0f}%)"
            for a, c in sorted(acts.items(), key=lambda x: -x[1])
        )

        marker = "⚠" if avg_ev < 0 else "✓"
        lines.append(f"\n  {marker} {pos:4s}  ({n} decisions)")
        lines.append(f"    Avg equity:  {avg_eq:.1%}")
        lines.append(f"    Avg EV:      {avg_ev:+.1f} chips/decision")
        lines.append(f"    Avg pot:     {avg_pot:.0f} chips")
        lines.append(f"    Actions:     {act_str}")

    return "\n".join(lines)


def _build_hand_class_analysis(records: list[dict]) -> str:
    """Hand-Class Performance: top winners and losers by EV."""
    lines = [_section_header("HAND-CLASS PERFORMANCE")]

    cls_data: dict[str, dict] = defaultdict(lambda: {
        "count": 0, "ev_sum": 0.0, "equity_sum": 0.0,
        "actions": defaultdict(int),
    })

    for r in records:
        cls = r.get("hand_class", "?")
        if not cls:
            cls = "unknown"
        d = cls_data[cls]
        d["count"] += 1
        d["ev_sum"] += _ev_proxy(r)
        d["equity_sum"] += r.get("equity", 0)
        d["actions"][r.get("action", "?")] += 1

    sorted_cls = sorted(cls_data.items(),
                        key=lambda x: x[1]["ev_sum"] / max(x[1]["count"], 1))

    # Worst 15 hand classes
    losers = [(c, d) for c, d in sorted_cls
              if d["ev_sum"] / max(d["count"], 1) < 0][:15]
    if losers:
        lines.append("\n  WORST hand classes (negative EV):")
        for rank, (cls, d) in enumerate(losers, 1):
            n = d["count"]
            avg_ev = d["ev_sum"] / max(n, 1)
            avg_eq = d["equity_sum"] / max(n, 1)
            acts = ", ".join(f"{a}:{c}" for a, c in
                            sorted(d["actions"].items(), key=lambda x: -x[1]))
            lines.append(f"    #{rank:2d}  {cls:5s}  avg EV: {avg_ev:+6.1f}  "
                         f"avg eq: {avg_eq:.0%}  n={n}  [{acts}]")

    # Best 10 hand classes
    winners = [(c, d) for c, d in reversed(sorted_cls)
               if d["ev_sum"] / max(d["count"], 1) > 0][:10]
    if winners:
        lines.append("\n  BEST hand classes (positive EV):")
        for rank, (cls, d) in enumerate(winners, 1):
            n = d["count"]
            avg_ev = d["ev_sum"] / max(n, 1)
            avg_eq = d["equity_sum"] / max(n, 1)
            lines.append(f"    #{rank:2d}  {cls:5s}  avg EV: {avg_ev:+6.1f}  "
                         f"avg eq: {avg_eq:.0%}  n={n}")

    return "\n".join(lines)


def _build_texture_analysis(records: list[dict]) -> str:
    """Texture Analysis: performance on dry / wet / neutral boards."""
    lines = [_section_header("BOARD TEXTURE ANALYSIS")]

    tex_data: dict[str, dict] = defaultdict(lambda: {
        "count": 0, "ev_sum": 0.0, "equity_sum": 0.0,
        "actions": defaultdict(int), "pot_sum": 0,
    })

    # Only include postflop decisions (texture is meaningful)
    for r in records:
        if r.get("street", "") == "preflop":
            continue
        tex = r.get("texture", "unknown")
        d = tex_data[tex]
        d["count"] += 1
        d["ev_sum"] += _ev_proxy(r)
        d["equity_sum"] += r.get("equity", 0)
        d["actions"][r.get("action", "?")] += 1
        d["pot_sum"] += r.get("pot", 0)

    for tex in ("dry", "wet", "neutral"):
        d = tex_data.get(tex)
        if not d or d["count"] == 0:
            lines.append(f"\n  {tex.upper():8s}  (no data)")
            continue
        n = d["count"]
        avg_ev = d["ev_sum"] / max(n, 1)
        avg_eq = d["equity_sum"] / max(n, 1)
        avg_pot = d["pot_sum"] / max(n, 1)

        acts = d["actions"]
        total = sum(acts.values())
        bet_pct = (acts.get("bet", 0) + acts.get("raise", 0)) / max(total, 1)
        call_pct = acts.get("call", 0) / max(total, 1)
        fold_pct = acts.get("fold", 0) / max(total, 1)
        check_pct = acts.get("check", 0) / max(total, 1)

        marker = "⚠" if avg_ev < 0 else "✓"
        lines.append(f"\n  {marker} {tex.upper():8s}  ({n} decisions)")
        lines.append(f"    Avg equity:    {avg_eq:.1%}")
        lines.append(f"    Avg EV:        {avg_ev:+.1f} chips/decision")
        lines.append(f"    Avg pot:       {avg_pot:.0f} chips")
        lines.append(f"    Bet/Raise:     {bet_pct:.0%}")
        lines.append(f"    Call:          {call_pct:.0%}")
        lines.append(f"    Check:         {check_pct:.0%}")
        lines.append(f"    Fold:          {fold_pct:.0%}")

    return "\n".join(lines)


def _build_action_analysis(records: list[dict]) -> str:
    """Action Analysis: frequency, avg equity, avg EV per action type."""
    lines = [_section_header("ACTION ANALYSIS")]

    act_data: dict[str, dict] = defaultdict(lambda: {
        "count": 0, "ev_sum": 0.0, "equity_sum": 0.0,
        "pot_sum": 0, "amount_sum": 0,
    })

    for r in records:
        action = r.get("action", "?")
        d = act_data[action]
        d["count"] += 1
        d["ev_sum"] += _ev_proxy(r)
        d["equity_sum"] += r.get("equity", 0)
        d["pot_sum"] += r.get("pot", 0)
        d["amount_sum"] += r.get("amount") or 0

    total_decisions = len(records)
    sorted_acts = sorted(act_data.items(),
                         key=lambda x: x[1]["ev_sum"] / max(x[1]["count"], 1))

    for action, d in sorted_acts:
        n = d["count"]
        freq = n / max(total_decisions, 1)
        avg_ev = d["ev_sum"] / max(n, 1)
        avg_eq = d["equity_sum"] / max(n, 1)
        avg_pot = d["pot_sum"] / max(n, 1)

        marker = "⚠" if avg_ev < 0 else "✓"
        lines.append(f"\n  {marker} {action:6s}  "
                     f"freq: {freq:.0%} ({n})  "
                     f"avg EV: {avg_ev:+.1f}  "
                     f"avg eq: {avg_eq:.0%}  "
                     f"avg pot: {avg_pot:.0f}")

        if action in ("bet", "raise") and d["amount_sum"] > 0:
            avg_amt = d["amount_sum"] / max(n, 1)
            avg_sizing = avg_amt / max(avg_pot, 1)
            lines.append(f"           avg sizing: {avg_sizing:.0%} pot "
                         f"({avg_amt:.0f} chips)")

    return "\n".join(lines)


def _build_street_analysis(records: list[dict]) -> str:
    """Street Analysis: performance by preflop / flop / turn / river."""
    lines = [_section_header("STREET ANALYSIS")]

    street_data: dict[str, dict] = defaultdict(lambda: {
        "count": 0, "ev_sum": 0.0, "equity_sum": 0.0,
        "actions": defaultdict(int),
    })

    for r in records:
        street = r.get("street", "?")
        d = street_data[street]
        d["count"] += 1
        d["ev_sum"] += _ev_proxy(r)
        d["equity_sum"] += r.get("equity", 0)
        d["actions"][r.get("action", "?")] += 1

    for street in ("preflop", "flop", "turn", "river"):
        d = street_data.get(street)
        if not d or d["count"] == 0:
            continue
        n = d["count"]
        avg_ev = d["ev_sum"] / max(n, 1)
        avg_eq = d["equity_sum"] / max(n, 1)
        acts = d["actions"]
        total = sum(acts.values())
        fold_pct = acts.get("fold", 0) / max(total, 1)

        marker = "⚠" if avg_ev < 0 else "✓"
        act_str = ", ".join(
            f"{a}: {c/total*100:.0f}%"
            for a, c in sorted(acts.items(), key=lambda x: -x[1])
        )
        lines.append(f"\n  {marker} {street:8s}  ({n} decisions)  "
                     f"avg EV: {avg_ev:+.1f}  avg eq: {avg_eq:.0%}  "
                     f"fold: {fold_pct:.0%}")
        lines.append(f"    Actions: {act_str}")

    return "\n".join(lines)


def _build_opponent_analysis(records: list[dict]) -> str:
    """Opponent-Type Analysis: bucket opponents by VPIP/AGG and show
    our performance against each archetype."""
    lines = [_section_header("OPPONENT-TYPE ANALYSIS")]

    # Collect all opponent data across records
    opp_stats: dict[str, dict] = {}
    for r in records:
        opponents = r.get("opponents") or {}
        for opp_id, stats in opponents.items():
            if opp_id not in opp_stats:
                opp_stats[opp_id] = stats

    if not opp_stats:
        lines.append("\n  No opponent data collected yet.")
        lines.append("  Run a benchmark to populate opponent profiles.")
        return "\n".join(lines)

    # Classify opponents into archetypes
    archetypes: dict[str, list[str]] = {
        "Tight Passive":    [],
        "Tight Aggressive": [],
        "Loose Passive":    [],
        "Loose Aggressive": [],
    }

    for opp_id, stats in opp_stats.items():
        vpip = stats.get("vpip", 0)
        agg = stats.get("agg_pct", 0)
        sample = stats.get("sample", 0)

        # Classify: VPIP > 5 = loose (relative threshold for bot pool)
        # AGG > 0.5 = aggressive
        is_loose = vpip > 5
        is_agg = agg > 0.5

        if is_loose and is_agg:
            archetype = "Loose Aggressive"
        elif is_loose:
            archetype = "Loose Passive"
        elif is_agg:
            archetype = "Tight Aggressive"
        else:
            archetype = "Tight Passive"

        archetypes[archetype].append(opp_id)
        lines.append(f"\n  {opp_id}:")
        lines.append(f"    VPIP: {vpip}  PFR: {stats.get('pfr', 0)}  "
                     f"AGG: {agg:.0%}  sample: {sample}")
        lines.append(f"    → {archetype}")

    lines.append("\n  Archetype distribution:")
    for arch, opps in archetypes.items():
        count = len(opps)
        lines.append(f"    {arch:20s}  {count} opponents")

    return "\n".join(lines)


def _build_leak_detector(records: list[dict]) -> str:
    """Automated Leak Detector: flags common EV-negative patterns."""
    lines = [_section_header("LEAK DETECTOR")]

    if not records:
        lines.append("\n  No data to analyze.")
        return "\n".join(lines)

    leaks_found = 0

    # ── Leak 1: Folding with positive EV (equity > pot_odds + margin) ──
    bad_folds = []
    for r in records:
        if r.get("action") == "fold":
            eq = r.get("equity", 0)
            po = r.get("pot_odds", 0)
            if eq > po + 0.10:  # folding with 10%+ equity edge
                bad_folds.append(r)

    if bad_folds:
        leaks_found += 1
        lines.append(f"\n  ⚠ LEAK: Folding profitable spots ({len(bad_folds)} instances)")
        lines.append(f"    You folded with equity significantly above pot odds.")
        for r in bad_folds[:5]:
            lines.append(f"    - {r.get('hand_class','?'):5s} {r.get('position','?'):4s} "
                         f"{r.get('street','?'):8s}  "
                         f"eq={r.get('equity',0):.0%} > po={r.get('pot_odds',0):.0%}")

    # ── Leak 2: Calling with negative EV (equity < pot_odds - margin) ──
    bad_calls = []
    for r in records:
        if r.get("action") == "call":
            eq = r.get("equity", 0)
            po = r.get("pot_odds", 0)
            if eq < po - 0.10:  # calling with 10%+ deficit
                bad_calls.append(r)

    if bad_calls:
        leaks_found += 1
        lines.append(f"\n  ⚠ LEAK: Calling unprofitable spots ({len(bad_calls)} instances)")
        lines.append(f"    You called when equity was well below pot odds.")
        for r in bad_calls[:5]:
            lines.append(f"    - {r.get('hand_class','?'):5s} {r.get('position','?'):4s} "
                         f"{r.get('street','?'):8s}  "
                         f"eq={r.get('equity',0):.0%} < po={r.get('pot_odds',0):.0%}")

    # ── Leak 3: Over-folding by position ──
    pos_folds: dict[str, dict] = defaultdict(lambda: {"total": 0, "folds": 0})
    for r in records:
        pos = r.get("position", "?")
        pos_folds[pos]["total"] += 1
        if r.get("action") == "fold":
            pos_folds[pos]["folds"] += 1

    for pos, d in pos_folds.items():
        fold_rate = d["folds"] / max(d["total"], 1)
        if fold_rate > 0.60 and d["total"] >= 5:
            leaks_found += 1
            lines.append(f"\n  ⚠ LEAK: Over-folding from {pos} "
                         f"({fold_rate:.0%} fold rate, {d['total']} decisions)")
            if pos in ("BTN", "CO"):
                lines.append(f"    Late position should be more aggressive. "
                             f"Consider widening opening range.")
            elif pos == "BB":
                lines.append(f"    BB defend range may be too narrow.")

    # ── Leak 4: Betting/raising with low equity ──
    bad_bets = []
    for r in records:
        if r.get("action") in ("bet", "raise") and r.get("street") != "preflop":
            eq = r.get("equity", 0)
            if eq < 0.35:  # betting with under 35% equity (not a good semi-bluff)
                bad_bets.append(r)

    if bad_bets:
        leaks_found += 1
        lines.append(f"\n  ⚠ LEAK: Betting/raising with weak equity "
                     f"({len(bad_bets)} instances)")
        lines.append(f"    Postflop aggression with <35% equity (not strong "
                     f"semi-bluffs).")
        for r in bad_bets[:5]:
            lines.append(f"    - {r.get('hand_class','?'):5s} {r.get('position','?'):4s} "
                         f"{r.get('street','?'):8s} {r.get('texture','?'):7s}  "
                         f"eq={r.get('equity',0):.0%}")

    # ── Leak 5: Texture mismatch — betting too much on wet boards ──
    tex_bet: dict[str, dict] = defaultdict(lambda: {"total": 0, "bets": 0})
    for r in records:
        if r.get("street") == "preflop":
            continue
        tex = r.get("texture", "?")
        tex_bet[tex]["total"] += 1
        if r.get("action") in ("bet", "raise"):
            tex_bet[tex]["bets"] += 1

    wet = tex_bet.get("wet", {"total": 0, "bets": 0})
    dry = tex_bet.get("dry", {"total": 0, "bets": 0})
    if wet["total"] >= 5 and dry["total"] >= 5:
        wet_rate = wet["bets"] / max(wet["total"], 1)
        dry_rate = dry["bets"] / max(dry["total"], 1)
        if wet_rate > dry_rate + 0.15:
            leaks_found += 1
            lines.append(f"\n  ⚠ LEAK: Betting more on wet boards ({wet_rate:.0%}) "
                         f"than dry ({dry_rate:.0%})")
            lines.append(f"    Wet boards should have MORE checking to protect "
                         f"range. Consider reducing c-bet frequency on wet textures.")

    # ── Leak 6: River decisions ──
    river_calls = [r for r in records
                   if r.get("street") == "river" and r.get("action") == "call"]
    if river_calls:
        avg_eq = sum(r.get("equity", 0) for r in river_calls) / len(river_calls)
        avg_po = sum(r.get("pot_odds", 0) for r in river_calls) / len(river_calls)
        if avg_eq < avg_po:
            leaks_found += 1
            lines.append(f"\n  ⚠ LEAK: River calls average equity ({avg_eq:.0%}) "
                         f"below pot odds ({avg_po:.0%})")
            lines.append(f"    {len(river_calls)} river calls — consider "
                         f"tightening river calling range.")

    # Summary
    if leaks_found == 0:
        lines.append("\n  ✓ No obvious leaks detected in current sample.")
        lines.append("    (More data may reveal patterns. Run more hands.)")
    else:
        lines.append(f"\n  Found {leaks_found} potential leak(s).")
        lines.append(f"  Priority: fix highest-frequency leaks first.")

    return "\n".join(lines)


# ─── Report builder ──────────────────────────────────────────────────

def build_report(records: list[dict]) -> str:
    """Build the full diagnostic report from telemetry records."""
    total = len(records)

    header = [
        "=" * 62,
        "  POKER DIAGNOSTICS — Telemetry Analysis Report",
        "  (paste alongside STRATEGY.md for Heuristic Learning loop)",
        "=" * 62,
        "",
        f"  Total decisions analyzed:  {total}",
    ]

    # Count unique hands
    hand_ids = set(r.get("hand_id", "") for r in records)
    header.append(f"  Unique hands:              {len(hand_ids)}")

    # Street distribution
    streets = defaultdict(int)
    for r in records:
        streets[r.get("street", "?")] += 1
    street_str = ", ".join(f"{s}: {c}" for s, c in
                          sorted(streets.items(), key=lambda x: -x[1]))
    header.append(f"  Street distribution:       {street_str}")

    # Action distribution
    actions = defaultdict(int)
    for r in records:
        actions[r.get("action", "?")] += 1
    action_str = ", ".join(f"{a}: {c}" for a, c in
                          sorted(actions.items(), key=lambda x: -x[1]))
    header.append(f"  Action distribution:       {action_str}")
    header.append("")

    sections = [
        "\n".join(header),
        _build_position_analysis(records),
        _build_hand_class_analysis(records),
        _build_street_analysis(records),
        _build_texture_analysis(records),
        _build_action_analysis(records),
        _build_opponent_analysis(records),
        _build_leak_detector(records),
    ]

    # Footer
    sections.append("\n" + "=" * 62)
    sections.append("  NEXT STEP — paste to coding agent:")
    sections.append('  "Read STRATEGY.md and this diagnostics report.')
    sections.append("   Which patterns do you see in the leaks?")
    sections.append("   Improve decide() in examples/agent.py.")
    sections.append('   Zero LLM calls at runtime. Bake into code."')
    sections.append("=" * 62)

    return "\n".join(sections)


# ─── CLI ─────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Poker telemetry diagnostics for HL loop")
    parser.add_argument("--telemetry", default="hand_telemetry.jsonl",
                        help="Path to telemetry JSONL file "
                             "(default: hand_telemetry.jsonl)")
    parser.add_argument("--out", default=None,
                        help="Write report to file (default: stdout)")
    args = parser.parse_args(argv)

    records = _load_telemetry(Path(args.telemetry))
    if not records:
        return 1

    report = build_report(records)

    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"wrote → {args.out}")
    else:
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
