"""
decide_v2.py — Enhanced L1 Heuristic Agent for dev.fun Arena
=============================================================
Drop this file into poker-arena-starter-kit/examples/decide_v2.py

Run with:
    uv run python examples/cli.py run --max-hands 50 --agent examples/decide_v2.py

Improvements over baseline decide():
  1. Positional awareness (BTN/CO/HJ play wider; UTG/MP play tighter)
  2. Preflop range filtering (O(1) dict lookup per position)
  3. Board texture classification (dry/wet affects c-bet frequency)
  4. Opponent profiling (VPIP, fold-to-cbet — exploits calling stations & nits)
  5. Street-aware sizing (flop/turn/river use different bet fractions)
  6. Timeout watchdog (hard deadline guard throughout)
"""
from __future__ import annotations

import time
from typing import Optional

# ─── Opponent Profile Store ────────────────────────────────────────────────
# Persists across hands within the same session.
# Keyed by seatNumber (int). Cleared on process restart.
_opponent_profiles: dict[int, dict] = {}


def _get_profile(seat: int) -> dict:
    if seat not in _opponent_profiles:
        _opponent_profiles[seat] = {
            "hands_seen":    0,
            "vpip":          0,      # times voluntarily put money in
            "fold_to_cbet":  0,      # times folded to a flop cbet
            "cbet_seen":     0,      # times faced a flop cbet
            "pfr":           0,      # preflop raises
            "total_actions": 0,
        }
    return _opponent_profiles[seat]


# Track which action-log entries we've already processed so we don't
# double-count when the cumulative log grows across calls within a hand.
_seen_action_indices: dict[str, int] = {}   # tableId -> last processed index


def _update_profiles(table: dict) -> None:
    """
    Read the action history from the table state and update opponent profiles.
    Arena exposes per-street action logs inside table['actionLog'] or
    table['actions']. We do a best-effort parse — if the key is absent we
    skip silently so we never crash the decide loop.
    """
    self_seat = table.get("selfSeatNumber")
    street = (table.get("street") or "Preflop")

    # action log key varies by Arena version — try both
    action_log = table.get("actionLog") or table.get("actions") or []

    if not action_log:
        return

    # Only process entries we haven't seen yet for this table.
    table_id = table.get("tableId") or ""
    start_idx = _seen_action_indices.get(table_id, 0)
    if start_idx >= len(action_log):
        return
    _seen_action_indices[table_id] = len(action_log)

    for entry in action_log[start_idx:]:
        if not isinstance(entry, dict):
            continue
        seat_num = entry.get("seatNumber")
        if seat_num is None or seat_num == self_seat:
            continue
        action = (entry.get("action") or "").lower()
        entry_street = (entry.get("street") or street).lower()

        prof = _get_profile(seat_num)
        prof["total_actions"] += 1

        # VPIP: voluntarily entered pot preflop
        if entry_street == "preflop" and action in ("call", "raise", "bet"):
            prof["vpip"] += 1
            prof["hands_seen"] = max(prof["hands_seen"], 1)

        # PFR: raised preflop
        if entry_street == "preflop" and action in ("raise", "bet"):
            prof["pfr"] += 1

        # Fold to cbet: folded on the flop
        if entry_street == "flop" and action == "fold":
            prof["fold_to_cbet"] += 1
            prof["cbet_seen"] += 1
        elif entry_street == "flop" and action in ("call", "raise"):
            prof["cbet_seen"] += 1


def _classify_opponent(seat: int) -> str:
    """
    Returns one of: 'nit', 'tag', 'lag', 'station', 'maniac', 'unknown'
    Needs at least 8 hands of data before classifying.
    """
    prof = _get_profile(seat)
    h = max(prof["hands_seen"], 1)
    if h < 8:
        return "unknown"

    vpip_rate = prof["vpip"] / h
    pfr_rate  = prof["pfr"]  / h

    if vpip_rate < 0.12 and pfr_rate < 0.06:
        return "nit"
    if vpip_rate < 0.25 and pfr_rate > 0.10:
        return "tag"
    if vpip_rate > 0.35 and pfr_rate > 0.22:
        return "lag"
    if vpip_rate > 0.35 and pfr_rate < 0.12:
        return "station"        # calls a lot, rarely raises
    if pfr_rate > 0.30:
        return "maniac"
    return "tag"                # default to TAG if unclear


# ─── Preflop Range Lookup (O(1) dict) ─────────────────────────────────────
# Positions: UTG, UTG1, MP, HJ, CO, BTN, SB, BB
# Ranges listed tightest (UTG) to widest (BTN).
# We use hand-class strings from _hand_class() in the baseline code.

_PREFLOP_OPEN_RANGES: dict[str, set[str]] = {
    "UTG": {
        "AA","KK","QQ","JJ","TT","99",
        "AKs","AKo","AQs","AQo","AJs","AJo","ATs",
        "KQs","KQo","KJs","QJs",
    },
    "UTG1": {
        "AA","KK","QQ","JJ","TT","99","88",
        "AKs","AKo","AQs","AQo","AJs","AJo","ATs","A9s",
        "KQs","KQo","KJs","KTs","QJs","QTs","JTs",
    },
    "MP": {
        "AA","KK","QQ","JJ","TT","99","88","77",
        "AKs","AKo","AQs","AQo","AJs","AJo","ATs","A9s","A8s",
        "KQs","KQo","KJs","KTs","QJs","QTs","JTs","T9s",
    },
    "HJ": {
        "AA","KK","QQ","JJ","TT","99","88","77","66",
        "AKs","AKo","AQs","AQo","AJs","AJo","ATs","A9s","A8s","A7s",
        "KQs","KQo","KJs","KTs","K9s","QJs","QTs","Q9s",
        "JTs","J9s","T9s","98s",
    },
    "CO": {
        "AA","KK","QQ","JJ","TT","99","88","77","66","55",
        "AKs","AKo","AQs","AQo","AJs","AJo","ATs","A9s","A8s","A7s","A6s","A5s",
        "KQs","KQo","KJs","KTs","K9s","K8s","QJs","QTs","Q9s",
        "JTs","J9s","J8s","T9s","T8s","98s","87s",
    },
    "BTN": {
        "AA","KK","QQ","JJ","TT","99","88","77","66","55","44","33","22",
        "AKs","AKo","AQs","AQo","AJs","AJo","ATs","A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s",
        "KQs","KQo","KJs","KTs","K9s","K8s","K7s","QJs","QTs","Q9s","Q8s",
        "JTs","J9s","J8s","T9s","T8s","98s","97s","87s","76s","65s",
        "KJo","KTo","QJo","JTo",
    },
    "SB": {
        # SB plays slightly tighter than BTN (OOP postflop)
        "AA","KK","QQ","JJ","TT","99","88","77","66","55","44",
        "AKs","AKo","AQs","AQo","AJs","AJo","ATs","A9s","A8s","A7s","A6s","A5s","A4s",
        "KQs","KQo","KJs","KTs","K9s","QJs","QTs","Q9s",
        "JTs","J9s","T9s","T8s","98s","87s","76s",
        "KJo","KTo","QJo",
    },
    "BB": {
        # BB defends wide — already has 1BB invested
        "AA","KK","QQ","JJ","TT","99","88","77","66","55","44","33","22",
        "AKs","AKo","AQs","AQo","AJs","AJo","ATs","A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s",
        "KQs","KQo","KJs","KTs","K9s","K8s","K7s","K6s","QJs","QTs","Q9s","Q8s",
        "JTs","J9s","J8s","J7s","T9s","T8s","T7s","98s","97s","87s","76s","65s","54s",
        "KJo","KTo","K9o","QJo","QTo","JTo","T9o","98o",
    },
}

# 3-bet ranges (tighter — only premiums + some bluff combos)
_3BET_RANGES: dict[str, set[str]] = {
    "UTG":  {"AA","KK","QQ","AKs","AKo"},
    "UTG1": {"AA","KK","QQ","JJ","AKs","AKo","AQs"},
    "MP":   {"AA","KK","QQ","JJ","AKs","AKo","AQs","AQo"},
    "HJ":   {"AA","KK","QQ","JJ","TT","AKs","AKo","AQs","AQo","AJs"},
    "CO":   {"AA","KK","QQ","JJ","TT","AKs","AKo","AQs","AQo","AJs","A5s","A4s"},
    "BTN":  {"AA","KK","QQ","JJ","TT","99","AKs","AKo","AQs","AQo","AJs","ATs","A5s","A4s","KQs"},
    "SB":   {"AA","KK","QQ","JJ","TT","AKs","AKo","AQs","AQo","AJs","A5s","KQs"},
    "BB":   {"AA","KK","QQ","JJ","TT","99","AKs","AKo","AQs","AJs","A5s","A4s","KQs","KJs"},
}


def _hand_class(hole: list[str]) -> str:
    """Identical to baseline — converts hole cards to range key like 'AKs'."""
    ranks = "23456789TJQKA"
    if len(hole) != 2:
        return ""
    r1, s1 = hole[0][0].upper(), hole[0][-1].lower()
    r2, s2 = hole[1][0].upper(), hole[1][-1].lower()
    if r1 not in ranks or r2 not in ranks:
        return ""
    if ranks.index(r1) < ranks.index(r2):
        r1, r2 = r2, r1
        s1, s2 = s2, s1
    if r1 == r2:
        return r1 + r2
    return f"{r1}{r2}{'s' if s1 == s2 else 'o'}"


def _in_open_range(hand_cls: str, position: str) -> bool:
    pos = position.upper()
    rng = _PREFLOP_OPEN_RANGES.get(pos, _PREFLOP_OPEN_RANGES["MP"])
    return hand_cls in rng


def _in_3bet_range(hand_cls: str, position: str) -> bool:
    pos = position.upper()
    rng = _3BET_RANGES.get(pos, _3BET_RANGES["MP"])
    return hand_cls in rng


# ─── Position Detection ────────────────────────────────────────────────────

def _detect_position(table: dict) -> str:
    """
    Infer position label from seat numbers and dealer button.
    Returns one of: UTG, UTG1, MP, HJ, CO, BTN, SB, BB
    Falls back to 'MP' if data is missing.
    """
    self_seat_num = table.get("selfSeatNumber")
    seats = table.get("seats") or []
    dealer_seat = table.get("dealerSeatNumber") or table.get("buttonSeatNumber")

    active_seats = sorted(
        [s["seatNumber"] for s in seats if s.get("isActive", True)],
    )
    n = len(active_seats)
    if not active_seats or self_seat_num not in active_seats:
        return "MP"

    if dealer_seat is None:
        # Fallback: use seat index parity
        idx = active_seats.index(self_seat_num)
        labels = ["BTN", "SB", "BB", "UTG", "UTG1", "MP", "HJ", "CO"]
        return labels[idx % len(labels)]

    try:
        dealer_idx = active_seats.index(dealer_seat)
    except ValueError:
        return "MP"

    self_idx = active_seats.index(self_seat_num)
    # Distance from dealer (dealer = BTN = 0)
    dist = (self_idx - dealer_idx) % n

    position_map = {
        0: "BTN",
        1: "SB",
        2: "BB",
        3: "UTG",
        4: "UTG1",
        5: "MP",
        6: "HJ",
        7: "CO",
    }
    if n <= 4:
        # Short-handed adjustments
        short_map = {0: "BTN", 1: "SB", 2: "BB", 3: "UTG"}
        return short_map.get(dist, "BTN")

    return position_map.get(dist, "MP")


def _is_in_position(position: str) -> bool:
    """True if we act last (or near last) postflop."""
    return position in ("BTN", "CO", "HJ")


# ─── Board Texture Classification ─────────────────────────────────────────

def _classify_board(board: list[str]) -> str:
    """
    Returns: 'dry', 'semi-wet', or 'wet'
    Dry   = rainbow, no straight draws, no pair
    Wet   = monotone OR flush draw + connected cards (OESD possible)
    Semi  = one draw present
    """
    if not board:
        return "dry"

    suits = [c[-1].lower() for c in board]
    ranks_str = "23456789TJQKA"
    rank_indices = []
    for c in board:
        r = c[0].upper()
        if c.startswith("10"):
            r = "T"
        if r in ranks_str:
            rank_indices.append(ranks_str.index(r))

    # Flush draw: 3+ of the same suit is monotone (very wet).
    # 2 of the same suit on a 3-card flop is a draw possibility,
    # but we need the threshold at 3 for "flush draw" on 4-5 card boards.
    max_same_suit = max(suits.count(s) for s in set(suits)) if suits else 0
    flush_draw = max_same_suit >= 3 if len(board) >= 4 else max_same_suit >= 2

    straight_draw = False
    if len(rank_indices) >= 2:
        rank_indices_sorted = sorted(set(rank_indices))
        for i in range(len(rank_indices_sorted) - 1):
            gap = rank_indices_sorted[i+1] - rank_indices_sorted[i]
            if gap <= 2:
                straight_draw = True
                break

    paired = len(rank_indices) != len(set(rank_indices))

    if paired:
        return "dry"      # paired boards reduce draws significantly
    if flush_draw and straight_draw:
        return "wet"
    if flush_draw or straight_draw:
        return "semi-wet"
    return "dry"


# ─── Sizing Helpers ────────────────────────────────────────────────────────

def _bet_size(pot: int, street: str, texture: str,
              bet_range: dict, equity: float) -> int:
    """
    Return a TO-amount bet size based on street and board texture.
    Always clamped within [min, max] from allowedActions.
    """
    min_b = int(bet_range.get("min") or max(int(pot * 0.33), 1))
    max_b = int(bet_range.get("max") or min_b)

    # Fraction of pot to bet
    if street == "Preflop":
        frac = 0.75                         # open 3x-ish
    elif street == "Flop":
        frac = 0.50 if texture == "dry" else 0.66
    elif street == "Turn":
        frac = 0.66 if texture != "wet" else 0.50
    else:                                   # River
        frac = 0.75 if equity > 0.85 else 0.50

    # Minimum meaningful open: 2.5x BB (assume BB=2 chips baseline)
    floor = int(min_b)
    target = int(pot * frac)
    target = max(target, floor)
    # Never let an open-raise exceed 25% of remaining stack unintentionally
    return max(min_b, min(target, max_b))


def _raise_size(pot: int, call_chips: int, street: str,
                raise_range: dict, equity: float) -> int:
    min_r = int(raise_range.get("min") or call_chips * 2)
    max_r = int(raise_range.get("max") or min_r)

    if street == "Preflop":
        target = int(call_chips * 3)        # standard 3x reraise
    else:
        target = int(pot * 0.66 + call_chips)

    return max(min_r, min(target, max_r))


# ─── Fallback Reasoning (must stay under 150 chars) ───────────────────────

_FALLBACK_REASONING = '{vr: "std", ke: "legal", pp: "pot control"}'


def _build_reasoning(action: str, equity: float, pot_odds: float,
                     position: str, texture: str, street: str) -> str:
    ke   = f"{int(round(equity * 100))}% eq"[:20]
    pos  = position[:6]
    pp   = f"{pos} {street[:3]} {texture[:4]}"[:28]
    sr   = ""
    if action in ("bet", "raise", "all-in"):
        sr = f'sr: "po {int(round(pot_odds*100))}% FE"'

    parts = [
        f'vr: "v2"',
        f'ke: "{ke}"',
        f'pos: "{pos}"',
        f'pp: "{pp}"',
    ]
    if sr:
        parts.append(sr)
    yaml = "{" + ", ".join(parts) + "}"
    return yaml if len(yaml) <= 150 else _FALLBACK_REASONING


# ─── Main decide() ────────────────────────────────────────────────────────

def decide(table: dict, deadline_s: float = 10.0,
           research_context: Optional[dict] = None) -> dict:
    """
    Enhanced L1 decide() with:
      - Positional preflop range filtering
      - Opponent profiling & exploit adjustments
      - Board texture-aware sizing
      - Street-differentiated thresholds
      - Hard timeout watchdog
    """
    start_t = time.monotonic()

    allowed    = table.get("allowedActions") or {}
    available  = allowed.get("availableActions") or []
    street     = table.get("street") or "Preflop"

    # ── Compute position & texture BEFORE the deadline closure ─────────────
    # (needed by _deadline_action's call to _out)
    board    = list(table.get("boardCards") or [])
    position = _detect_position(table)
    texture  = _classify_board(board)

    # ── Hard deadline guard ────────────────────────────────────────────────
    def _deadline_action() -> dict:
        if allowed.get("canCheck") or "check" in available:
            return _out("check", None, 0.5, 0.0, position,
                        texture, street, allowed, "deadline — free card")
        return _out("fold", None, 0.0, 1.0, position,
                    texture, street, allowed, "deadline — fold")

    if deadline_s < 1.5:
        return _deadline_action()

    # ── Extract table state ────────────────────────────────────────────────
    self_seat_num = table.get("selfSeatNumber")
    seats         = table.get("seats") or []
    self_seat     = next(
        (s for s in seats if s.get("seatNumber") == self_seat_num), {}
    )
    hole  = list(self_seat.get("holeCards") or [])

    pot        = int(table.get("potChips") or 0)
    call_chips = int(allowed.get("callChips") or 0)
    pot_odds   = call_chips / max(pot + call_chips, 1) if call_chips else 0.0

    # ── Position & texture (already computed above) ───────────────────────
    in_pos   = _is_in_position(position)
    hand_cls = _hand_class(hole)

    # ── Update opponent profiles ──────────────────────────────────────────
    _update_profiles(table)

    # ── Identify dominant opponent type at table ──────────────────────────
    opp_types = []
    for s in seats:
        sn = s.get("seatNumber")
        if sn is not None and sn != self_seat_num:
            opp_types.append(_classify_opponent(sn))

    has_station = "station" in opp_types
    has_nit     = "nit"     in opp_types
    has_maniac  = "maniac"  in opp_types

    # ── Equity estimation ─────────────────────────────────────────────────
    # Reuse baseline treys estimator if available; otherwise use preflop table
    equity = _estimate_equity_safe(hole, board, deadline_s)

    # ── Timeout check after equity calc ───────────────────────────────────
    if time.monotonic() - start_t > deadline_s - 0.8:
        return _deadline_action()

    # ══════════════════════════════════════════════════════════════════════
    # PREFLOP LOGIC
    # ══════════════════════════════════════════════════════════════════════
    if street == "Preflop":
        in_range   = _in_open_range(hand_cls, position)
        in_3b_range = _in_3bet_range(hand_cls, position)

        # Facing a raise (call_chips > 0 and it's more than just BB)
        if call_chips > 0:
            # All-in defense: if call commits >40% of stack, only continue
            # with premium hands. This stops the agent bleeding 100-chip
            # stacks calling off jams with garbage.
            self_stack = int(self_seat.get("stack") or 100)
            call_is_shove = call_chips >= self_stack * 0.40

            if call_is_shove:
                # Only call a jam with the top of our range
                jam_call_hands = {
                    "AA","KK","QQ","JJ","TT","99",
                    "AKs","AKo","AQs","AQo","AJs",
                }
                if hand_cls in jam_call_hands and equity > 0.55:
                    return _out("call", None, equity, pot_odds,
                                position, texture, street, allowed,
                                f"call jam {position} — premium hand {hand_cls}")
                elif in_3b_range and allowed.get("canRaise"):
                    # Re-jam with absolute premiums only
                    if hand_cls in {"AA","KK","QQ","AKs","AKo"}:
                        rr = allowed.get("raiseRange") or {}
                        amount = _raise_size(pot, call_chips, street, rr, equity)
                        return _out("raise", amount, equity, pot_odds,
                                    position, texture, street, allowed,
                                    f"re-jam {position} — premium")
                # Everything else: fold against a shove
                if "fold" in available:
                    return _out("fold", None, equity, pot_odds,
                                position, texture, street, allowed,
                                f"fold vs shove — {hand_cls} not premium enough")

            if in_3b_range and allowed.get("canRaise"):
                rr     = allowed.get("raiseRange") or {}
                amount = _raise_size(pot, call_chips, street, rr, equity)
                return _out("raise", amount, equity, pot_odds,
                            position, texture, street, allowed,
                            f"3-bet {position} premium")

            elif in_range and equity >= pot_odds + 0.04:
                return _out("call", None, equity, pot_odds,
                            position, texture, street, allowed,
                            f"call {position} — in range, price ok")

            else:
                if "fold" in available:
                    return _out("fold", None, equity, pot_odds,
                                position, texture, street, allowed,
                                f"fold {position} — outside range")
                return _out("check", None, equity, pot_odds,
                            position, texture, street, allowed,
                            "check — BB sees free flop")

        else:
            # No one has raised — open or check
            # FIX: added parentheses — original had operator precedence bug:
            #   `in_range and allowed.get("canBet") or allowed.get("canRaise")`
            #   was parsed as `(in_range and canBet) or canRaise`, which would
            #   open-raise hands outside range whenever canRaise was truthy.
            if in_range and (allowed.get("canBet") or allowed.get("canRaise")):
                br = allowed.get("betRange") or allowed.get("raiseRange") or {}
                amount = _bet_size(pot, street, texture, br, equity)
                action_name = "bet" if allowed.get("canBet") else "raise"
                return _out(action_name, amount, equity, pot_odds,
                            position, texture, street, allowed,
                            f"open {position} — in range")
            elif "check" in available:
                return _out("check", None, equity, pot_odds,
                            position, texture, street, allowed,
                            "check — outside open range")
            else:
                return _out("fold", None, equity, pot_odds,
                            position, texture, street, allowed,
                            "fold — outside range, must act")

    # ══════════════════════════════════════════════════════════════════════
    # POSTFLOP LOGIC (Flop / Turn / River)
    # ══════════════════════════════════════════════════════════════════════

    # Adjust equity thresholds by opponent type
    # vs calling stations: bet for value more, bluff less
    # vs nits: bluff more freely
    # vs maniacs: tighten calling range
    value_threshold  = 0.65   # base threshold to bet for value
    raise_threshold  = 0.78   # base threshold to raise
    call_threshold   = 0.02   # min edge over pot odds to call

    if has_station:
        value_threshold  = 0.60   # bet thinner for value (they call)
        raise_threshold  = 0.75
        # Never bluff stations — they never fold
    if has_nit:
        value_threshold  = 0.55   # bet even thinner; they fold a lot
        raise_threshold  = 0.72
    if has_maniac:
        raise_threshold  = 0.85   # only raise/reraise with very strong hands
        call_threshold   = 0.08   # need more equity edge to call their aggression

    # Positional adjustment — be more aggressive in position
    if in_pos:
        value_threshold -= 0.04
        raise_threshold -= 0.03

    # Wet board adjustment — check back more OOP on wet boards
    if texture == "wet" and not in_pos:
        value_threshold += 0.06

    # ── No bet to call (check or bet) ─────────────────────────────────────
    if call_chips == 0:
        if equity > value_threshold and (
                allowed.get("canBet") or "bet" in available):
            br     = allowed.get("betRange") or {}
            amount = _bet_size(pot, street, texture, br, equity)
            return _out("bet", amount, equity, pot_odds,
                        position, texture, street, allowed,
                        f"value bet {street} {texture}")

        # Bluff on dry boards in position with decent equity
        bluff_ok = (
            in_pos
            and texture == "dry"
            and equity > 0.35
            and not has_station          # never bluff stations
            and street in ("Flop", "Turn")
            and (allowed.get("canBet") or "bet" in available)
        )
        if bluff_ok:
            br     = allowed.get("betRange") or {}
            amount = _bet_size(pot, street, texture, br, equity)
            return _out("bet", amount, equity, pot_odds,
                        position, texture, street, allowed,
                        f"semi-bluff/cbet {street}")

        # Default: check
        if "check" in available:
            return _out("check", None, equity, pot_odds,
                        position, texture, street, allowed,
                        "check — pot control")
        return _out("fold", None, equity, pot_odds,
                    position, texture, street, allowed,
                    "fold — no free option")

    # ── Facing a bet (call / raise / fold) ────────────────────────────────
    edge = equity - pot_odds

    if equity > raise_threshold and allowed.get("canRaise"):
        rr     = allowed.get("raiseRange") or {}
        amount = _raise_size(pot, call_chips, street, rr, equity)
        return _out("raise", amount, equity, pot_odds,
                    position, texture, street, allowed,
                    f"raise for value {street}")

    elif edge >= call_threshold and "call" in available:
        return _out("call", None, equity, pot_odds,
                    position, texture, street, allowed,
                    f"call — {int(equity*100)}% covers {int(pot_odds*100)}%")

    elif "fold" in available:
        return _out("fold", None, equity, pot_odds,
                    position, texture, street, allowed,
                    f"fold — {int(equity*100)}% short of {int(pot_odds*100)}%")

    elif "check" in available:
        return _out("check", None, equity, pot_odds,
                    position, texture, street, allowed,
                    "check — last resort")

    # Absolute fallback
    return _out("fold", None, 0.0, 1.0,
                position, texture, street, allowed,
                "fold — no legal action found")


# ─── Output builder ───────────────────────────────────────────────────────

def _out(action: str, amount: Optional[int],
         equity: float, pot_odds: float,
         position: str, texture: str, street: str,
         allowed: dict, msg: str) -> dict:
    """Build the action payload. Strips amount from non-sizing actions."""
    if action in ("fold", "check"):
        amount = None
    # For call, let server compute callToAmount
    if action == "call":
        cta = allowed.get("callToAmount")
        amount = int(cta) if cta is not None else None

    reasoning = _build_reasoning(
        action, equity, pot_odds, position, texture, street
    )
    payload: dict = {
        "action":    action,
        "message":   msg[:500],
        "reasoning": reasoning,
    }
    if amount is not None:
        payload["amount"] = int(amount)
    return payload


# ─── Safe equity estimator ────────────────────────────────────────────────

_PREFLOP_EQUITY_TABLE = {
    "AA": 0.85, "KK": 0.82, "QQ": 0.80, "JJ": 0.77, "TT": 0.75,
    "99": 0.72, "88": 0.69, "77": 0.66, "66": 0.63, "55": 0.60,
    "44": 0.57, "33": 0.54, "22": 0.50,
    "AKs": 0.67, "AKo": 0.65, "AQs": 0.66, "AQo": 0.64, "AJs": 0.65,
    "AJo": 0.63, "ATs": 0.64, "KQs": 0.63, "KQo": 0.61, "KJs": 0.62,
    "QJs": 0.60, "JTs": 0.58, "T9s": 0.54, "98s": 0.52,
}


def _to_treys(card_str: str) -> str:
    if not card_str:
        return "2c"
    r = card_str[0].upper()
    if card_str.startswith("10"):
        r = "T"
        s = card_str[2].lower() if len(card_str) > 2 else "x"
    else:
        s = card_str[-1].lower()
    return r + s


def _estimate_equity_safe(hole: list[str], board: list[str],
                           deadline_s: float) -> float:
    """Wraps treys Monte Carlo with a safe fallback to preflop table."""
    cls = _hand_class(hole)
    if not board:
        return _PREFLOP_EQUITY_TABLE.get(cls, 0.45)
    if deadline_s < 2.5:
        return _PREFLOP_EQUITY_TABLE.get(cls, 0.45)
    try:
        import random as _random
        from treys import Card as TC, Evaluator as TE, Deck as TD
        ev   = TE()
        hero = [TC.new(_to_treys(c)) for c in hole]
        brd  = [TC.new(_to_treys(c)) for c in board]
        used = set(hero) | set(brd)
        rng  = _random.Random(2026)
        wins = ties = 0
        sims = 200
        for _ in range(sims):
            deck = TD()
            deck.cards = [c for c in deck.cards if c not in used]
            rng.shuffle(deck.cards)
            opp     = [deck.cards.pop(), deck.cards.pop()]
            runout  = [deck.cards.pop() for _ in range(5 - len(brd))]
            full    = brd + runout
            hr      = ev.evaluate(full, hero)
            orr     = ev.evaluate(full, opp)
            if hr < orr:
                wins += 1
            elif hr == orr:
                ties += 1
        return (wins + 0.5 * ties) / sims
    except Exception:
        return _PREFLOP_EQUITY_TABLE.get(cls, 0.45)
