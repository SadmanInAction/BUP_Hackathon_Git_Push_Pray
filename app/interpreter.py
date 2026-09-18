"""LLM operator-note interpreter (Groq, OpenAI-compatible chat API).

The LLM reads every note and returns one structured JSON object per note. Deterministic
code then normalises it into the Problem Statement shape: it expands time windows into
hour lists (start included, end excluded) and turns percentages into numbers. The result
still passes through app/guardrails.py before it reaches the optimizer.
"""
import json
import logging
import os

import httpx

try:  # local development convenience; in Docker/hosting, env vars are injected directly
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

log = logging.getLogger("gridwise.interpreter")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
PRIMARY_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")
HTTP_TIMEOUT_S = 9.0

SYSTEM_PROMPT = """You convert campus energy operator notes into structured directives for a 24-hour schedule (hours 0-23).

Each note maps to EXACTLY ONE directive_type:
- "solar_reduction": usable solar power is reduced during some hours (cleaning, washing, clouds, inverter/panel work, shading, maintenance).
- "minimum_battery_reserve": battery must keep at least some amount of stored energy during some hours.
- "no_charge_window": battery must not / cannot CHARGE during some hours (charger offline, isolated, disabled).
- "no_discharge_window": battery must not / cannot DISCHARGE during some hours.
- "max_grid_window": grid import/intake/draw must not exceed a limit (kWh per hour) during some hours.
- "no_op": the note does not change today's energy schedule (unrelated campus news, events, menus, deadlines, notes about other days or next week, or information with no actionable limit).

Time rules:
- Use 24-hour clock. midnight=0, noon=12, 1 PM=13, 6 PM=18, 11 PM=23.
- A window "from A until B" / "between A and B" / "A-B" gives start_hour=A and end_hour=B; the END IS EXCLUSIVE. "1 PM to 3 PM" -> start 13, end 15 (hours 13,14).
- "until midnight" / "until end of day" -> end_hour=24. "all day" -> start 0, end 24.
- A single hour ("at 5 PM", "during the 5 PM hour") -> start 17, end 18.
- Use several windows only if the note names several separate periods.

Value rules:
- solar_reduction: "remaining_solar_fraction" is the fraction of normal solar that is STILL AVAILABLE.
  "drops to 20%" -> 0.2; "80% reduction" -> 0.2; "half" -> 0.5; "one-fifth of normal" -> 0.2; "reduced by a quarter" -> 0.75; "no solar"/"fully covered" -> 0.
- minimum_battery_reserve: if given in kWh put it in "minimum_energy_kwh"; if given as a percentage/fraction of battery capacity put that percentage (0-100) in "reserve_percent_of_capacity" instead.
- max_grid_window: "max_grid_kwh" is the per-hour grid import limit in kWh.
- Never invent numbers that are not stated or directly implied by the note.

Return ONLY a JSON object:
{"interpretations": [
  {"note_index": 0,
   "directive_type": "...",
   "windows": [{"start_hour": 13, "end_hour": 15}],
   "remaining_solar_fraction": null,
   "minimum_energy_kwh": null,
   "reserve_percent_of_capacity": null,
   "max_grid_kwh": null,
   "explanation": "one short sentence"}
]}
One entry per note, in the same order as the notes. For no_op use "windows": [] and nulls."""


def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip().rstrip("%"))
        except ValueError:
            return None
    return None


def _expand_windows(windows) -> list[int]:
    hours: set[int] = set()
    for w in windows or []:
        if not isinstance(w, dict):
            continue
        start, end = _num(w.get("start_hour")), _num(w.get("end_hour"))
        if start is None or end is None:
            continue
        start, end = int(start), int(end)
        if not 0 <= start <= 23 or not 0 <= end <= 24:
            continue
        if end > start:
            hours.update(range(start, end))
        elif end < start:  # wraps past midnight, e.g. 10 PM to 2 AM
            hours.update(range(start, 24))
            hours.update(range(0, end))
        else:  # zero-length window: treat as that single hour
            hours.add(start)
    return sorted(hours)


def _no_op(i: int, explanation: str) -> dict:
    return {"note_index": i, "applies": False, "directive_type": "no_op",
            "structured_adjustment": None, "explanation": explanation}


def _normalise(item: dict, i: int, battery: dict) -> dict:
    """Map one raw LLM entry to the Problem Statement directive shape."""
    dtype = item.get("directive_type")
    explanation = str(item.get("explanation") or "").strip() or f"Interpreted as {dtype}."
    if dtype == "no_op":
        return _no_op(i, explanation)

    hours = _expand_windows(item.get("windows"))
    if not hours:
        return _no_op(i, "No valid time window could be extracted; treated as no_op.")
    adj: dict = {"hours": hours}

    if dtype == "solar_reduction":
        f = _num(item.get("remaining_solar_fraction"))
        if f is not None and 1 < f <= 100:  # model answered as a percentage
            f = f / 100
        adj["factor"] = f
    elif dtype == "minimum_battery_reserve":
        kwh = _num(item.get("minimum_energy_kwh"))
        pct = _num(item.get("reserve_percent_of_capacity"))
        if kwh is None and pct is not None:
            if 0 < pct <= 1:
                pct *= 100
            kwh = float(battery["capacity_kwh"]) * pct / 100
        adj["minimum_energy_kwh"] = round(kwh, 4) if kwh is not None else None
    elif dtype == "max_grid_window":
        adj["max_grid_kwh"] = _num(item.get("max_grid_kwh"))
    elif dtype not in ("no_charge_window", "no_discharge_window"):
        return _no_op(i, "Unsupported directive type returned by the model; treated as no_op.")

    # Missing required numbers are left as None; app/guardrails.py turns that entry into no_op.
    return {"note_index": i, "applies": True, "directive_type": dtype,
            "structured_adjustment": adj, "explanation": explanation}


def _call_llm(model: str, user_prompt: str, api_key: str) -> list:
    resp = httpx.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": user_prompt}],
        },
        timeout=HTTP_TIMEOUT_S,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    data = json.loads(content)
    items = data.get("interpretations") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("model output missing 'interpretations' list")
    return items


def interpret_notes(operator_notes: list[str], hours: list[dict],
                    battery: dict) -> list[dict]:
    """
    Returns one dict per note, in note_index order:
    {"note_index": int, "applies": bool, "directive_type": str,
     "structured_adjustment": dict|None, "explanation": str}
    """
    n = len(operator_notes)
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        log.warning("GROQ_API_KEY not set; all notes treated as no_op")
        return [_no_op(i, "LLM not configured; note treated as no_op.") for i in range(n)]

    notes_text = "\n".join(f"[{i}] {note}" for i, note in enumerate(operator_notes))
    user_prompt = (f"Battery capacity_kwh = {battery['capacity_kwh']}.\n"
                   f"There are {n} operator notes:\n{notes_text}")

    items = None
    for model in (PRIMARY_MODEL, FALLBACK_MODEL):
        try:
            items = _call_llm(model, user_prompt, api_key)
            break
        except Exception as exc:  # never log the exception text: it could echo headers
            log.warning("LLM call failed model=%s error=%s", model, type(exc).__name__)
    if items is None:
        return [_no_op(i, "LLM unavailable; note treated as no_op.") for i in range(n)]

    by_index: dict[int, dict] = {}
    for pos, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        idx = item.get("note_index", pos)
        idx = int(idx) if isinstance(idx, (int, float)) and not isinstance(idx, bool) else pos
        if 0 <= idx < n:
            by_index.setdefault(idx, item)

    return [_normalise(by_index[i], i, battery) if i in by_index
            else _no_op(i, "Model returned no interpretation for this note; treated as no_op.")
            for i in range(n)]
