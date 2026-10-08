"""
llm_decide_v1.py — Hybrid L1+L2 Agent for dev.fun Arena
=========================================================
Drop into poker-arena-starter-kit/examples/llm_decide_v1.py

Strategy: decide_v2 heuristic runs first and proposes an action.
The LLM reviews it with full table context and either approves
or overrides with a better decision. Best of both worlds:
- L1 speed and reliability as the baseline
- L2 reasoning as an enhancement layer
- Fallback to pure L1 if LLM fails or deadline is tight

Run with:
    uv run python examples/cli.py run --max-hands 50 --agent examples/llm_decide_v1.py

.env requirements:
    OPENAI_API_KEY=sk-or-your-openrouter-key-here
    OPENAI_BASE_URL=https://openrouter.ai/api/v1
    OPENROUTER_MODEL=nousresearch/hermes-4-405b
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import importlib.util
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Import proven L1 heuristic ───────────────────────────────────────────────
_v2_path = Path(__file__).parent / "decide_v2.py"
_spec    = importlib.util.spec_from_file_location("decide_v2", str(_v2_path))
_v2_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v2_mod)

heuristic_decide   = _v2_mod.decide
_detect_position   = _v2_mod._detect_position
_classify_board    = _v2_mod._classify_board
_hand_class        = _v2_mod._hand_class
_get_profile       = _v2_mod._get_profile
_update_profiles   = _v2_mod._update_profiles
_classify_opponent = _v2_mod._classify_opponent

_FALLBACK_REASONING = '{vr: "std", ke: "legal", pp: "pot control"}'

# ── Hybrid System Prompt ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are reviewing a proposed poker action. APPROVE it unless you see a CLEAR error.

ONLY override when:
- Hero has >75% equity AND proposed action is check/fold → BET instead
- Hero has <40% equity AND proposed action is call/raise → FOLD instead
- Proposed is raise preflop with garbage hand (72o, 85o, 93o type hands) → FOLD

In ALL other cases: return the PROPOSED ACTION unchanged.

CRITICAL: Most proposals are correct. Your job is APPROVAL, not creativity.
DO NOT invent new sizings. DO NOT raise when heuristic says call/check.
DO NOT bet when heuristic says fold.

amount = TOTAL chips this street. Must be within betRange/raiseRange min/max.
Omit amount for fold/check/call.

Respond with ONLY this JSON:
{"action":"X","amount":N_or_omit,"message":"short","reasoning":"{vr:\\"r\\",ke:\\"eq\\",pp:\\"plan\\"}"}"""


# ── Context builder ───────────────────────────────────────────────────────────
def retrieve_solver_context(table: dict) -> dict:
    _update_profiles(table)
    self_seat_num = table.get("selfSeatNumber")
    seats         = table.get("seats") or []
    position      = _detect_position(table)
    board         = list(table.get("boardCards") or [])
    texture       = _classify_board(board)
    self_seat     = next((s for s in seats if s.get("seatNumber") == self_seat_num), {})
    hole          = list(self_seat.get("holeCards") or [])
    hand_cls      = _hand_class(hole)

    opponent_reads = []
    for s in seats:
        sn = s.get("seatNumber")
        if sn is None or sn == self_seat_num:
            continue
        prof      = _get_profile(sn)
        archetype = _classify_opponent(sn)
        h         = max(prof["hands_seen"], 1)
        opponent_reads.append({
            "seat":      sn,
            "archetype": archetype,
            "vpip":      round(prof["vpip"] / h, 2),
            "pfr":       round(prof["pfr"] / h, 2),
            "fold_cbet": round(prof["fold_to_cbet"] / max(prof["cbet_seen"], 1), 2),
            "hands":     prof["hands_seen"],
            "stack":     s.get("stackChips"),
        })

    return {
        "position":   position,
        "texture":    texture,
        "hand_class": hand_cls,
        "opponents":  opponent_reads,
        "note":       _strategic_note(position, texture, opponent_reads, table),
    }


def _strategic_note(position, texture, opponents, table):
    allowed    = table.get("allowedActions") or {}
    call_chips = int(allowed.get("callChips") or 0)
    pot        = int(table.get("potChips") or 0)
    street     = table.get("street") or "Preflop"
    in_pos     = position in ("BTN", "CO", "HJ")

    if any(o["archetype"] == "station" for o in opponents):
        return "STATION: value bet every street, zero bluffs"
    if any(o["archetype"] == "maniac" for o in opponents):
        return "MANIAC: premiums only, trap"
    if any(o["archetype"] == "nit" for o in opponents) and street != "Preflop":
        return "NIT: bluff dry boards freely"
    if texture == "wet" and not in_pos:
        return "WET OOP: check-call, avoid inflating pot"
    if texture == "dry" and in_pos:
        return "DRY IP: c-bet 50% pot, barrel turns"
    if call_chips > pot * 0.5:
        return "BIG BET FACING: need clear equity edge"
    return "Standard: position and equity driven"


# ── LLM call ─────────────────────────────────────────────────────────────────
def _call_llm(prompt: str, remaining_s: float) -> Optional[str]:
    if remaining_s < 5.0:
        return None
    api_key  = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    model    = os.environ.get("OPENROUTER_MODEL", "nousresearch/hermes-4-405b")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model,
            max_tokens=200,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            extra_headers={
                "HTTP-Referer": "https://arena.dev.fun",
                "X-Title":      "Lawnsonic Arena Agent",
            },
        )
        raw = (resp.choices[0].message.content or "").strip()
        print(f"[hybrid] LLM: {raw[:150]}", file=sys.stderr)
        return raw
    except Exception as e:
        print(f"[hybrid] LLM failed: {e}", file=sys.stderr)
        return None


# ── JSON extraction ───────────────────────────────────────────────────────────
def _extract_json(text: str) -> Optional[dict]:
    s = text.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        s  = s[nl + 1:] if nl >= 0 else s.lstrip("`")
    if s.rstrip().endswith("```"):
        s = s.rstrip()[:-3].strip()

    first = s.find("{")
    if first < 0:
        return None
    candidate = s[first:]

    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict) and "action" in obj:
            return obj
    except Exception:
        pass

    action_m  = re.search(r'"action"\s*:\s*"([^"]+)"', candidate)
    amount_m  = re.search(r'"amount"\s*:\s*(\d+)', candidate)
    message_m = re.search(r'"message"\s*:\s*"([^"]*)"', candidate)
    if action_m:
        result = {
            "action":    action_m.group(1),
            "message":   message_m.group(1) if message_m else "LLM decision",
            "reasoning": _FALLBACK_REASONING,
        }
        if amount_m:
            result["amount"] = int(amount_m.group(1))
        return result
    return None


def _validate(action: dict, table: dict) -> dict:
    allowed   = table.get("allowedActions") or {}
    available = set(allowed.get("availableActions") or [])
    name      = action.get("action", "")

    if name not in available:
        if "check" in available:
            return {"action": "check", "message": "fallback",
                    "reasoning": _FALLBACK_REASONING}
        return {"action": "fold", "message": "fallback",
                "reasoning": _FALLBACK_REASONING}

    if name in ("fold", "check", "call"):
        action.pop("amount", None)
        return action

    try:
        amount = int(action.get("amount") or 0)
    except (TypeError, ValueError):
        amount = 0

    rng = (allowed.get("betRange")   if name == "bet"   else
           allowed.get("raiseRange") if name == "raise" else {}) or {}
    lo = int(rng.get("min") or amount or 1)
    hi = int(rng.get("max") or amount or lo)
    action["amount"] = max(lo, min(amount or lo, hi))
    return action


def _clean_reasoning(r: str) -> str:
    if r.startswith('"') and r.endswith('"'):
        r = r[1:-1].replace('\\"', '"')
    if not (r.startswith("{") and r.endswith("}")):
        return _FALLBACK_REASONING
    return r[:150]


# ── Main decide() ─────────────────────────────────────────────────────────────
def decide(table: dict, deadline_s: float = 10.0,
           research_context: Optional[dict] = None) -> dict:
    """
    Hybrid L1+L2: heuristic proposes, LLM reviews and approves/overrides.
    Falls back to pure heuristic on tight deadline or LLM failure.
    """
    start_t = time.monotonic()

    if research_context is None:
        research_context = retrieve_solver_context(table)

    # Always get heuristic proposal first
    h_action = heuristic_decide(table, deadline_s=deadline_s,
                                research_context=research_context)

    # If deadline too tight for LLM, return heuristic directly
    if deadline_s < 6.0:
        print(f"[hybrid] deadline tight ({deadline_s:.1f}s), using heuristic",
              file=sys.stderr)
        return h_action

    # Build context for LLM review
    allowed       = table.get("allowedActions") or {}
    self_seat_num = table.get("selfSeatNumber")
    seats         = table.get("seats") or []
    self_seat     = next(
        (s for s in seats if s.get("seatNumber") == self_seat_num), {}
    )

    table_summary = {
        "street":   table.get("street"),
        "pot":      table.get("potChips"),
        "board":    table.get("boardCards"),
        "hero": {
            "hole":     self_seat.get("holeCards"),
            "stack":    self_seat.get("stackChips"),
            "position": research_context.get("position"),
        },
        "opponents": [
            {"seat": s.get("seatNumber"), "stack": s.get("stackChips"),
             "bet": s.get("currentBetChips")}
            for s in seats if s.get("seatNumber") != self_seat_num
        ],
        "allowed": {
            "actions":    allowed.get("availableActions"),
            "call_chips": allowed.get("callChips"),
            "bet_range":  allowed.get("betRange"),
            "raise_range":allowed.get("raiseRange"),
        },
        "texture":    research_context.get("texture"),
        "hand_class": research_context.get("hand_class"),
        "opponents_profile": research_context.get("opponents"),
        "note":       research_context.get("note"),
    }

    prompt = (
        "TABLE STATE:\n" +
        json.dumps(table_summary, separators=(",", ":")) +
        "\n\nPROPOSED ACTION (from heuristic agent):\n" +
        json.dumps(h_action, separators=(",", ":")) +
        "\n\nApprove or override? Respond with ONLY the JSON action."
    )

    remaining = deadline_s - (time.monotonic() - start_t)
    text = _call_llm(prompt, remaining)

    if text:
        llm_action = _extract_json(text)
        if llm_action:
            llm_action = _validate(llm_action, table)
            llm_action.setdefault("message", h_action.get("message", "decision"))
            llm_action.setdefault("reasoning", _FALLBACK_REASONING)
            llm_action["message"]   = str(llm_action["message"])[:500]
            llm_action["reasoning"] = _clean_reasoning(
                str(llm_action.get("reasoning", ""))
            )
            return llm_action

    # LLM failed or returned bad JSON — use heuristic
    print("[hybrid] using heuristic (LLM unavailable)", file=sys.stderr)
    return h_action