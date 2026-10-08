"""llm_client.py — robust, env-driven LLM client for the poker agent.

Two responsibilities:

  1. _chat_completion()  — one model-agnostic call that correctly reads
     OpenRouter config from .env (OPENAI_API_KEY + OPENAI_BASE_URL +
     OPENROUTER_MODEL), passes base_url explicitly, and attaches the
     optional OpenRouter attribution headers. Returns text or None.

  2. consult_analyst()   — the L6 strategic analyst. Given a window of
     recent telemetry records, asks Haiku for ADJUSTMENT MULTIPLIERS
     (never actions). Returns a validated dict of floats clamped to safe
     ranges. Falls back to neutral multipliers on any failure.

The pure-LLM per-action path stays in llm_agent.py (llm_decide). This
module is what agent.py imports for MODE=analyst, and what llm_agent.py
can optionally reuse for its provider call.

Design rules honored:
  - LLM never returns chip amounts or actions here. Multipliers only.
  - Every failure path returns safe neutral defaults — the match never
    dies because the analyst timed out or returned garbage.
  - Model + base_url come from env so you never have to pass --model.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Optional


# ─── Config resolution (env-driven) ──────────────────────────────────

def _resolve_model() -> str:
    """Pick the model string. Priority:
       OPENROUTER_MODEL > OPENAI_MODEL > a safe default.
    Your .env sets OPENROUTER_MODEL=anthropic/claude-haiku-4.5."""
    return (
        os.environ.get("OPENROUTER_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or "anthropic/claude-haiku-4.5"
    )


def _resolve_base_url() -> Optional[str]:
    return os.environ.get("OPENAI_BASE_URL") or None


# ─── Core LLM call ────────────────────────────────────────────────────

def _chat_completion(system: str, user: str, max_tokens: int = 400,
                     temperature: float = 0.3,
                     model: Optional[str] = None) -> Optional[str]:
    """One OpenAI-compatible chat call. Works with OpenRouter, OpenAI,
    Together, Groq, vLLM — anything that speaks the chat-completions API.

    Returns the text content on success, None on any failure. Never raises.

    Fixes vs the original llm_agent.py wiring:
      - base_url passed explicitly (not left to SDK env auto-read)
      - model read from OPENROUTER_MODEL env, not hard-coded "gpt-5"
      - OpenRouter attribution headers attached when using openrouter.ai
    """
    # Anthropic-direct path (only if a real Anthropic key is set AND we are
    # not pointed at an OpenAI-compatible base like OpenRouter).
    if os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        try:
            import anthropic  # type: ignore
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model=model or os.environ.get("ANTHROPIC_MODEL")
                      or "claude-haiku-4-5-20251001",
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                timeout=8.0,  # bound a hung consult so it can't leak _analyst_busy
            )
            return "".join(
                getattr(b, "text", "") for b in resp.content
                if getattr(b, "type", None) == "text"
            ).strip()
        except Exception as e:
            print(f"[llm_client] anthropic call failed: {e}", file=sys.stderr)
            return None

    # OpenAI-compatible path (OpenRouter / OpenAI / Together / Groq / vLLM).
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI  # type: ignore
            base_url = _resolve_base_url()
            extra_headers = {}
            if base_url and "openrouter.ai" in base_url:
                # Optional but recommended OpenRouter attribution headers.
                extra_headers = {
                    "HTTP-Referer": os.environ.get(
                        "OPENROUTER_REFERER", "https://arena.dev.fun"),
                    "X-Title": os.environ.get(
                        "OPENROUTER_TITLE", "poker-arena-agent"),
                }
            client = OpenAI(base_url=base_url) if base_url else OpenAI()
            resp = client.chat.completions.create(
                model=model or _resolve_model(),
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                extra_headers=extra_headers or None,
                timeout=8.0,  # bound a hung consult so it can't leak _analyst_busy
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[llm_client] openai-compat call failed: {e}", file=sys.stderr)
            return None

    return None


def llm_available() -> bool:
    """True if any provider key is configured."""
    return bool(
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )


# ─── L6 Strategic Analyst ─────────────────────────────────────────────

# Neutral multipliers — returned on any failure. 1.0 = no change.
_NEUTRAL_ADJUSTMENTS = {
    "open_multiplier": 1.0,        # widen/tighten opening ranges
    "defend_multiplier": 1.0,      # widen/tighten BB defend
    "value_multiplier": 1.0,       # how thin to value bet
    "bluff_multiplier": 1.0,       # bluff frequency
    "wet_aggression": 1.0,         # aggression on wet boards
    "river_aggression": 1.0,       # river barrel frequency
}

# Clamp ranges — the analyst can nudge, not swing wildly. This is the
# safety rail that keeps a hallucinated 5.0 from blowing up the strategy.
_ADJUSTMENT_BOUNDS = {
    "open_multiplier":   (0.6, 1.4),
    "defend_multiplier": (0.6, 1.4),
    "value_multiplier":  (0.7, 1.3),
    "bluff_multiplier":  (0.3, 1.5),
    "wet_aggression":    (0.6, 1.4),
    "river_aggression":  (0.5, 1.4),
}


_ANALYST_SYSTEM = """You are a poker strategy analyst reviewing an agent's
recent hands in a 6-max No-Limit Hold'em benchmark. You do NOT choose
actions. You output ONLY strategic adjustment multipliers as JSON.

A multiplier of 1.0 means no change. Below 1.0 means do less of that
thing; above 1.0 means more. Stay within 0.5–1.5. Base every adjustment
on the data given — if the data is thin or neutral, return values near 1.0.

Output ONLY this JSON object on a single line, no prose, no code fences:
{"open_multiplier": <f>, "defend_multiplier": <f>, "value_multiplier": <f>,
 "bluff_multiplier": <f>, "wet_aggression": <f>, "river_aggression": <f>}

Guidance:
- If the agent bleeds chips calling too wide → lower defend_multiplier.
- If opponents fold a lot → raise bluff_multiplier.
- If opponents are calling stations → lower bluff_multiplier, raise value_multiplier.
- If losing big on wet boards → lower wet_aggression.
- If river decisions are negative EV → lower river_aggression.
"""


def _summarize_window(records: list[dict]) -> str:
    """Build a compact, token-cheap summary of the recent telemetry window
    for the analyst prompt. We send aggregates, not raw rows."""
    if not records:
        return "No hands yet."

    from collections import defaultdict
    by_pos: dict[str, list[float]] = defaultdict(list)
    by_tex: dict[str, list[float]] = defaultdict(list)
    by_street: dict[str, list[float]] = defaultdict(list)
    action_counts: dict[str, int] = defaultdict(int)
    opp_latest: dict[str, dict] = {}

    for r in records:
        ev = 0.0 if r.get("action") == "fold" else round(
            r.get("equity", 0.5) - r.get("pot_odds", 0.0), 3)
        by_pos[r.get("position", "?")].append(ev)
        by_tex[r.get("texture", "?")].append(ev)
        by_street[r.get("street", "?")].append(ev)
        action_counts[r.get("action", "?")] += 1
        for aid, stats in (r.get("opponents") or {}).items():
            if stats.get("sample", 0) >= opp_latest.get(aid, {}).get("sample", -1):
                opp_latest[aid] = stats

    def _avg(xs): return round(sum(xs) / len(xs), 3) if xs else 0.0

    lines = [f"Hands in window: {len(records)}"]
    lines.append("Mean EV-proxy by position: " + ", ".join(
        f"{k}={_avg(v)}(n{len(v)})" for k, v in sorted(by_pos.items())))
    lines.append("Mean EV-proxy by texture: " + ", ".join(
        f"{k}={_avg(v)}(n{len(v)})" for k, v in sorted(by_tex.items())))
    lines.append("Mean EV-proxy by street: " + ", ".join(
        f"{k}={_avg(v)}(n{len(v)})" for k, v in sorted(by_street.items())))
    lines.append("Action mix: " + ", ".join(
        f"{k}:{v}" for k, v in sorted(action_counts.items(), key=lambda x: -x[1])))
    if opp_latest:
        opp_bits = []
        for aid, s in list(opp_latest.items())[:5]:
            opp_bits.append(
                f"{aid}(vpip{s.get('vpip',0)},pfr{s.get('pfr',0)},"
                f"agg{s.get('agg_pct',0)},n{s.get('sample',0)})")
        lines.append("Opponents: " + ", ".join(opp_bits))
    return "\n".join(lines)


def _clamp_adjustments(raw: dict) -> dict:
    """Validate + clamp the analyst's output to safe bounds. Any missing or
    non-numeric field falls back to neutral (1.0)."""
    out = dict(_NEUTRAL_ADJUSTMENTS)
    for key, (lo, hi) in _ADJUSTMENT_BOUNDS.items():
        val = raw.get(key)
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        out[key] = round(max(lo, min(v, hi)), 3)
    return out


def consult_analyst(records: list[dict],
                    model: Optional[str] = None) -> dict:
    """Ask the analyst LLM for adjustment multipliers based on the recent
    telemetry window. ALWAYS returns a valid, clamped dict — neutral on any
    failure. This is the only place agent.py touches the LLM in MODE=analyst.

    records: a list of recent telemetry dicts (same shape _log_decision writes)
    """
    if not llm_available():
        return dict(_NEUTRAL_ADJUSTMENTS)

    summary = _summarize_window(records)
    user = (
        "Recent performance summary:\n" + summary +
        "\n\nReturn ONLY the adjustment JSON object."
    )
    text = _chat_completion(_ANALYST_SYSTEM, user,
                            max_tokens=200, temperature=0.2, model=model)
    if not text:
        return dict(_NEUTRAL_ADJUSTMENTS)

    # Parse — tolerate fences / surrounding prose.
    s = text.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        s = s[nl + 1:] if nl >= 0 else s.lstrip("`")
        s = s.strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end < 0:
        return dict(_NEUTRAL_ADJUSTMENTS)
    try:
        raw = json.loads(s[start:end + 1])
    except Exception:
        return dict(_NEUTRAL_ADJUSTMENTS)
    if not isinstance(raw, dict):
        return dict(_NEUTRAL_ADJUSTMENTS)

    return _clamp_adjustments(raw)


# ─── Self-test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Offline validation of the parse/clamp path (no network needed).
    print("Testing clamp logic...")
    assert _clamp_adjustments({}) == _NEUTRAL_ADJUSTMENTS
    assert _clamp_adjustments({"bluff_multiplier": 9.0})["bluff_multiplier"] == 1.5
    assert _clamp_adjustments({"defend_multiplier": 0.1})["defend_multiplier"] == 0.6
    assert _clamp_adjustments({"value_multiplier": "garbage"})["value_multiplier"] == 1.0
    assert _clamp_adjustments({"open_multiplier": 1.1})["open_multiplier"] == 1.1
    print("  clamp logic OK")

    print("Testing window summary...")
    fake = [
        {"position": "BTN", "texture": "wet", "street": "flop",
         "action": "call", "equity": 0.4, "pot_odds": 0.5,
         "opponents": {"bot1": {"vpip": 10, "pfr": 5, "agg_pct": 0.6, "sample": 20}}},
        {"position": "SB", "texture": "dry", "street": "preflop",
         "action": "raise", "equity": 0.7, "pot_odds": 0.0, "opponents": {}},
    ]
    print(_summarize_window(fake))
    print("\nModel resolved:", _resolve_model())
    print("Base URL resolved:", _resolve_base_url())
    print("LLM available:", llm_available())
    print("\nAll offline tests passed.")