# My Poker Strategy

> Instrumentation-first approach: collect failure data, then iterate.
>
>     pokerkit analyze --out failure_report.txt
>
> Then feed STRATEGY.md + failure_report.txt into HL loop:
>
>     "Read STRATEGY.md and failure_report.txt, then improve decide() in
>      examples/agent.py. Zero LLM calls at runtime. Bake everything into code."

---

## Starting Hand Ranges

**UTG (tightest):**
Raise: AA-99, AKs, AKo, AQs. Fold everything else.

**MP:**
Raise: AA-88, AKs, AKo, AQs, AQo, AJs, KQs. Fold everything else.

**CO:**
Raise: AA-55, AKs-ATs, AKo-AJo, KQs-KTs, KQo, QJs, QTs, JTs, T9s, 98s.

**BTN (widest):**
Open ~45% of hands. All pairs, AXs, AKo-ATo, KQs-K9s, KQo-KJo, suited connectors 76s+, Q9s+, J9s+.

**SB:**
Raise-or-fold, no limping. AA-88, AKs, AKo, AQs, AQo, AJs, ATs, KQs, KJs, QJs.

**BB:**
Open (when checked to): AA-TT, AKs, AKo, AQs.
Defend vs opens: above + 99, 88, 77, KQs, KJs, QJs, JTs, T9s, 98s.

---

## Postflop Philosophy

**When I have top pair or better:**
Value bet using texture-aware sizing. Dry boards = 33% pot. Wet boards = 66% pot.

**Dry boards (e.g. K72 rainbow):**
Bet small (33%) with range advantage. These boards favor the preflop raiser.

**Wet boards (e.g. JT9 two-tone):**
Bet larger (66%) to deny equity. Check more marginal hands to protect range.

**In position (IP) vs out of position (OOP):**
IP: Bet more often, control pot size, can value bet thinner.
OOP: Check-call strong hands, check-fold weak ones, avoid bloating pots.

---

## Aggression Style

- Bluff frequency: **low** (instrumentation phase — minimize variance)
- 3-bet range: **value-only** (premiums only: QQ+, AKs)
- C-bet frequency: **selective** (board-texture driven — 33% dry, 66% wet)

---

## Bet Sizing Rules

| Situation            | Sizing       |
|----------------------|--------------|
| Preflop open         | 2× min-raise |
| Preflop 3-bet        | 3× the open  |
| Dry flop c-bet       | 33% pot      |
| Wet flop c-bet       | 66% pot      |
| Neutral flop c-bet   | 50% pot      |
| Dry turn barrel      | 50% pot      |
| Wet turn barrel      | 75% pot      |
| River value bet      | 66% pot      |
| Wet river value      | 75% pot      |

---

## Adaptation to Opponents

Currently in **instrumentation phase** — collecting data before adapting.

After 50-100 hands of telemetry:
- If villain folds to c-bets > 60%: bet every flop regardless of cards.
- If villain calls too wide: value-bet thinly, bluff less.
- If villain is very tight (VPIP < 20%): fold more to their aggression.
- If villain is very aggressive (AGG > 60%): trap with strong hands, call wider.

Opponent stats are being tracked via `retrieve_solver_context()` and logged
to `hand_telemetry.jsonl` for analysis.

---

## What I Want to Fix

**Known weaknesses (pre-data):**
- No real equity calculation preflop (using static lookup table)
- BB defend range may be too narrow
- No 3-bet/4-bet ranges defined
- No river bluff strategy
- Position mapping is heuristic (seat number → position) — may be wrong for non-standard seating
- No stack-depth adjustments (same ranges at 10bb and 100bb)

**After first 50 hands, update this section with data-driven findings from:**
```bash
pokerkit analyze --out failure_report.txt
```
