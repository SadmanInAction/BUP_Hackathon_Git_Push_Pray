"""
GridWise operator-note interpreter.

Flow:  operator_notes --> LLM (strict JSON) --> deterministic guardrails --> directives

The LLM does the language understanding (which directive, which window, which
number). Everything it returns is treated as untrusted data and passed through
plain-Python validation before anything downstream sees it. Any note the model
fails to interpret safely becomes a flagged no_op; this module never raises.

Configuration (environment variables; no secrets are ever logged):
    GRIDWISE_LLM_PROVIDER   comma-separated provider order, tried in turn.
                            Values: groq, gemini, ollama. Default: "groq,gemini".
                            (ollama is opt-in: CPU-only inference is far too slow
                            for the judge's 30 s timeout.)
    GROQ_API_KEY            free key from https://console.groq.com
    GROQ_MODELS             comma-separated models, tried in order; each has its own
                            free-tier rate limit, so on HTTP 429 the next one is used.
                            Default "qwen/qwen3.8-27b,openai/gpt-oss-120b,openai/gpt-oss-20b"
    GEMINI_API_KEY          free key from https://aistudio.google.com
    GEMINI_MODEL            default "gemini-2.5-flash"
    OLLAMA_HOST             default "http://localhost:11434"
    OLLAMA_MODEL            default "qwen2.5:3b"
    GRIDWISE_LLM_TIMEOUT    per-call timeout in seconds, default 6
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
import threading
import time
import urllib.request
from typing import Any, Callable, Optional

try:  # local development convenience; in Docker/hosting, env vars are injected directly
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

log = logging.getLogger("gridwise.interpreter")

ALLOWED_TYPES = (
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
)

# Which numeric field (if any) each directive carries besides "hours".
NUMERIC_FIELD = {
    "solar_reduction": "factor",
    "minimum_battery_reserve": "minimum_energy_kwh",
    "no_charge_window": None,
    "no_discharge_window": None,
    "max_grid_window": "max_grid_kwh",
}

GUARDRAIL_TAG = "[guardrail]"

# An LLM call takes a (system_prompt, user_prompt) pair and returns raw text.
LLMCallable = Callable[[str, str], str]


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """Convert campus energy operator notes into directives for today's 24-hour schedule (hours 0-23, 24-hour clock). Each note maps to exactly one directive_type:
- solar_reduction: usable solar reduced. Adjustment {start_hour,end_hour,hours,factor}. factor = fraction of solar that REMAINS, 0..1: "drops to 20%" -> 0.2, "80% reduction" -> 0.2, "half" -> 0.5, "one-fifth" -> 0.2, none left -> 0.
- minimum_battery_reserve: battery must keep at least some energy. {start_hour,end_hour,hours,minimum_energy_kwh}. Convert a % of capacity to kWh using the given capacity.
- no_charge_window: battery may not charge / charger unavailable. {start_hour,end_hour,hours}
- no_discharge_window: battery may not discharge. {start_hour,end_hour,hours}
- max_grid_window: grid import capped per hour. {start_hour,end_hour,hours,max_grid_kwh}
- no_op: does not affect today's solar, battery or grid (events, admin news, future dates). structured_adjustment null, applies false.
Times: start_hour/end_hour are the stated boundaries (noon 12, 1 PM 13, midnight 0; "until midnight" -> end 24). Start INCLUDED, end EXCLUDED: "1 PM to 3 PM" -> hours [13,14].
If AM/PM is not stated, pick the physically sensible reading: solar exists only in daylight (about 6-18), so for solar "one until three" is 13-15; "evening"/"tonight" is PM, "morning" is AM.
applies is true for every type except no_op. Never invent values or types.
Reply with ONLY JSON: {"interpretations":[{"note_index","applies","directive_type","structured_adjustment","explanation"}]}, one entry per note, in order, explanation under 15 words."""

# Few-shot examples: three taken from the public sample pack, the rest written to
# cover the remaining directive types. They illustrate the mapping; they are not
# used for matching.
FEW_SHOT_USER = """Battery capacity_kwh: 200
Notes:
[0] Facilities will wash the rooftop solar panels from noon until 2 PM. During cleaning, usable solar should be treated as roughly 25% of the forecast.
[1] Expect an 80% reduction in rooftop solar between 11 AM and 2 PM because of inverter work.
[2] Keep at least 50% of the battery capacity stored in the battery from 6 PM until 9 PM for emergency operations.
[3] The sports office moved next month's registration deadline.
[4] The battery charger is offline 01:00-04:00.
[5] Discharging is locked out between 5 and 7 in the evening.
[6] Feeder limit: no more than 150 kWh from the grid per hour, 18:00 to 20:00."""

FEW_SHOT_ASSISTANT = json.dumps(separators=(",", ":"), obj={
    "interpretations": [
        {"note_index": 0, "applies": True, "directive_type": "solar_reduction",
         "structured_adjustment": {"start_hour": 12, "end_hour": 14, "hours": [12, 13], "factor": 0.25},
         "explanation": "25% of solar remains."},
        {"note_index": 1, "applies": True, "directive_type": "solar_reduction",
         "structured_adjustment": {"start_hour": 11, "end_hour": 14, "hours": [11, 12, 13], "factor": 0.2},
         "explanation": "80% reduction leaves 20%."},
        {"note_index": 2, "applies": True, "directive_type": "minimum_battery_reserve",
         "structured_adjustment": {"start_hour": 18, "end_hour": 21, "hours": [18, 19, 20], "minimum_energy_kwh": 100},
         "explanation": "50% of 200 kWh is 100 kWh."},
        {"note_index": 3, "applies": False, "directive_type": "no_op",
         "structured_adjustment": None,
         "explanation": "Not about today's energy schedule."},
        {"note_index": 4, "applies": True, "directive_type": "no_charge_window",
         "structured_adjustment": {"start_hour": 1, "end_hour": 4, "hours": [1, 2, 3]},
         "explanation": "Charger offline."},
        {"note_index": 5, "applies": True, "directive_type": "no_discharge_window",
         "structured_adjustment": {"start_hour": 17, "end_hour": 19, "hours": [17, 18]},
         "explanation": "Discharge locked out."},
        {"note_index": 6, "applies": True, "directive_type": "max_grid_window",
         "structured_adjustment": {"start_hour": 18, "end_hour": 20, "hours": [18, 19], "max_grid_kwh": 150},
         "explanation": "Grid capped at 150 kWh."},
    ]
})


def build_user_prompt(operator_notes: list[str], battery: dict) -> str:
    cap = battery.get("capacity_kwh") if isinstance(battery, dict) else None
    cap_txt = _fmt_num(cap) if _is_finite_number(cap) else "unknown"
    lines = [f"Battery capacity_kwh: {cap_txt}", "Notes:"]
    for i, note in enumerate(operator_notes):
        # Keep each note on one line so indices stay unambiguous.
        lines.append(f"[{i}] {' '.join(str(note).split())}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# LLM providers (stdlib only, so no extra dependencies)
# --------------------------------------------------------------------------- #

def _timeout() -> float:
    try:
        return float(os.environ.get("GRIDWISE_LLM_TIMEOUT", "6"))
    except ValueError:
        return 6.0


def _post_json(url: str, payload: dict, headers: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_timeout()) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _messages(system: str, user: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": FEW_SHOT_USER},
        {"role": "assistant", "content": FEW_SHOT_ASSISTANT},
        {"role": "user", "content": user},
    ]


MAX_RATE_LIMIT_WAIT_S = 8.0
DEFAULT_GROQ_MODELS = "qwen/qwen3.8-27b,openai/gpt-oss-120b,openai/gpt-oss-20b"


def _reasoning_args(model: str) -> dict:
    # Reasoning tokens cost latency and free-tier quota; this task doesn't need them.
    if "gpt-oss" in model:
        return {"reasoning_effort": "low"}
    if "qwen3" in model:
        return {"reasoning_effort": "none"}
    return {}


def call_groq(system: str, user: str) -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set")
    models = [m.strip() for m in os.environ.get("GROQ_MODELS", DEFAULT_GROQ_MODELS).split(",") if m.strip()]
    last_exc: Exception = RuntimeError("no GROQ_MODELS configured")
    for attempt in range(2):
        retry_after = []
        for model in models:
            try:
                data = _post_json(
                    "https://api.groq.com/openai/v1/chat/completions",
                    {
                        "model": model,
                        "messages": _messages(system, user),
                        "temperature": 0,
                        "response_format": {"type": "json_object"},
                        **_reasoning_args(model),
                    },
                    # Groq sits behind Cloudflare, which rejects urllib's default User-Agent.
                    {"Authorization": f"Bearer {key}", "User-Agent": "gridwise/1.0"},
                )
                return data["choices"][0]["message"]["content"]
            except Exception as exc:
                # Each model has its own free-tier quota: on rate limit / retired model /
                # server error, move straight on to the next one.
                code = getattr(exc, "code", "")
                log.warning("groq model %s failed (%s)", model, code or type(exc).__name__)
                last_exc = exc
                if code == 429:
                    headers = getattr(exc, "headers", None)
                    retry_after.append(_to_number(headers.get("retry-after") if headers else None) or 2.0)
        # Every model rate-limited: wait out the shortest Retry-After once, if it is short.
        if attempt == 0 and retry_after and min(retry_after) <= MAX_RATE_LIMIT_WAIT_S:
            time.sleep(min(retry_after) + 0.2)
        else:
            break
    raise last_exc


def call_gemini(system: str, user: str) -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    contents = [
        {"role": "user", "parts": [{"text": FEW_SHOT_USER}]},
        {"role": "model", "parts": [{"text": FEW_SHOT_ASSISTANT}]},
        {"role": "user", "parts": [{"text": user}]},
    ]
    data = _post_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        },
        # Key goes in a header, never in the URL, so it can't leak via logs.
        {"x-goog-api-key": key},
    )
    return data["candidates"][0]["content"]["parts"][0]["text"]


def call_ollama(system: str, user: str) -> str:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    data = _post_json(
        f"{host}/api/chat",
        {
            "model": os.environ.get("OLLAMA_MODEL", "qwen2.5:3b"),
            "messages": _messages(system, user),
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        },
        {},
    )
    return data["message"]["content"]


PROVIDERS: dict[str, LLMCallable] = {
    "groq": call_groq,
    "gemini": call_gemini,
    "ollama": call_ollama,
}


def _provider_chain() -> list[str]:
    raw = os.environ.get("GRIDWISE_LLM_PROVIDER", "groq,gemini")
    return [p.strip().lower() for p in raw.split(",") if p.strip().lower() in PROVIDERS]


def default_llm(system: str, user: str) -> str:
    """Try each configured provider in order; raise only if all fail."""
    errors = []
    for name in _provider_chain():
        try:
            return PROVIDERS[name](system, user)
        except Exception as exc:  # network, HTTP, missing key, bad envelope...
            # Log the exception type/status only: never the key or full payload.
            detail = getattr(exc, "code", "") or type(exc).__name__
            log.warning("LLM provider %s failed (%s)", name, detail)
            errors.append(f"{name}:{detail}")
    raise RuntimeError("all LLM providers failed: " + ", ".join(errors or ["none configured"]))


# --------------------------------------------------------------------------- #
# Parsing untrusted model text
# --------------------------------------------------------------------------- #

def extract_json(text: Any) -> Any:
    """Best-effort extraction of a JSON value from model output. Returns None on failure."""
    if not isinstance(text, str):
        return None
    s = text.strip()
    # Strip markdown code fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL | re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()
    try:
        return json.loads(s)
    except (ValueError, RecursionError):
        pass
    # Fall back to the outermost {...} or [...] span.
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = s.find(open_c), s.rfind(close_c)
        if 0 <= i < j:
            try:
                return json.loads(s[i:j + 1])
            except (ValueError, RecursionError):
                continue
    return None


def _entries_from_payload(payload: Any) -> list:
    """Pull the list of per-note entries out of whatever container the model used."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("interpretations", "directive_interpretation", "directives", "results", "notes"):
            if isinstance(payload.get(key), list):
                return payload[key]
        if "directive_type" in payload:  # a single bare entry
            return [payload]
    return []


# --------------------------------------------------------------------------- #
# Deterministic guardrails
# --------------------------------------------------------------------------- #

def _is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _to_number(x: Any) -> Optional[float]:
    if _is_finite_number(x):
        return float(x)
    if isinstance(x, str):
        m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*(%|kwh)?\s*", x, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def _fmt_num(x: float) -> Any:
    return int(x) if float(x).is_integer() else round(float(x), 6)


def _to_int(x: Any) -> Optional[int]:
    v = _to_number(x)
    return int(v) if v is not None and float(v).is_integer() else None


def _to_hour(x: Any) -> Optional[int]:
    v = _to_int(x)
    return v if v is not None and 0 <= v <= 24 else None


def hours_from_window(start: Any, end: Any) -> Optional[list[int]]:
    """[start, end) on a 24h clock, wrapping past midnight. None if unusable."""
    s, e = _to_hour(start), _to_hour(end)
    if s is None or e is None:
        return None
    s, e = s % 24, e % 24
    if s == e:
        return None  # zero-length or full-day: ambiguous, defer to explicit hours list
    if s < e:
        return list(range(s, e))
    return sorted(list(range(s, 24)) + list(range(0, e)))


def clean_hours(raw: Any) -> Optional[list[int]]:
    if not isinstance(raw, list):
        return None
    out = set()
    for h in raw:
        v = _to_hour(h)
        if v is None or v > 23:
            return None  # one bad hour means the list itself is untrustworthy
        out.add(v)
    return sorted(out) or None


def no_op_entry(index: int, explanation: str) -> dict:
    return {
        "note_index": index,
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": explanation,
    }


def _fail(index: int, reason: str) -> dict:
    log.warning("note %d fell back to no_op: %s", index, reason)
    return no_op_entry(index, f"{GUARDRAIL_TAG} Could not reliably interpret this note ({reason}); treated as no_op.")


def _clean_explanation(raw: Any, default: str) -> str:
    if isinstance(raw, str) and raw.strip():
        return " ".join(raw.split())[:300]
    return default


def validate_entry(raw: Any, index: int, battery: dict) -> dict:
    """Turn one untrusted model entry into a spec-compliant entry (or a flagged no_op)."""
    if not isinstance(raw, dict):
        return _fail(index, "entry is not an object")

    dtype = raw.get("directive_type")
    if isinstance(dtype, str):
        dtype = dtype.strip().lower().replace("-", "_").replace(" ", "_")
    if dtype not in ALLOWED_TYPES:
        return _fail(index, "unsupported directive_type")

    if dtype == "no_op":
        return no_op_entry(index, _clean_explanation(
            raw.get("explanation"), "This note does not affect today's energy schedule."))

    adj = raw.get("structured_adjustment")
    if not isinstance(adj, dict):
        return _fail(index, "missing structured_adjustment")

    # Hours: prefer the stated window boundaries (end excluded) over the model's
    # own enumeration, since off-by-one enumeration is the most common error.
    hours = hours_from_window(adj.get("start_hour"), adj.get("end_hour"))
    if hours is None:
        hours = clean_hours(adj.get("hours"))
    if not hours:
        return _fail(index, "no valid hours")

    clean: dict = {"hours": hours}
    field = NUMERIC_FIELD[dtype]
    if field is not None:
        value = _to_number(adj.get(field))
        if value is None:
            return _fail(index, f"missing or non-numeric {field}")

        if dtype == "solar_reduction":
            if 1 < value <= 100:
                value = value / 100.0  # model gave a percentage instead of a fraction
            if not 0 <= value <= 1:
                return _fail(index, "factor outside [0, 1]")
            value = round(value, 6)

        elif dtype == "minimum_battery_reserve":
            cap = _to_number(battery.get("capacity_kwh")) if isinstance(battery, dict) else None
            if value < 0:
                return _fail(index, "negative reserve")
            if cap is not None and value > cap:
                return _fail(index, "reserve exceeds battery capacity")

        elif dtype == "max_grid_window":
            if value < 0:
                return _fail(index, "negative grid cap")

        clean[field] = _fmt_num(value)

    return {
        "note_index": index,
        "applies": True,
        "directive_type": dtype,
        "structured_adjustment": clean,
        "explanation": _clean_explanation(raw.get("explanation"), f"Interpreted as {dtype}."),
    }


def assemble(payload: Any, n_notes: int, battery: dict) -> list[dict]:
    """Map model entries onto note indices 0..n-1: exactly one entry per note, in order."""
    by_index: dict[int, Any] = {}
    entries = _entries_from_payload(payload)
    for pos, entry in enumerate(entries):
        idx = entry.get("note_index") if isinstance(entry, dict) else None
        idx = _to_int(idx)
        if idx is None and len(entries) == n_notes:
            idx = pos  # model dropped note_index but kept order and count
        if idx is None or not 0 <= idx < n_notes or idx in by_index:
            continue  # out of range or duplicate: first one wins
        by_index[idx] = entry

    return [
        validate_entry(by_index[i], i, battery) if i in by_index else _fail(i, "model returned no entry")
        for i in range(n_notes)
    ]


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

RETRY_BUDGET_S = 10.0

_cache: dict[tuple, list[dict]] = {}
_cache_lock = threading.Lock()
_CACHE_MAX = 512


def _cache_key(notes: list[str], battery: dict) -> tuple:
    cap = battery.get("capacity_kwh") if isinstance(battery, dict) else None
    return (tuple(notes), repr(cap))


def interpret_notes(operator_notes: list[str], hours: list[dict], battery: dict,
                    *, llm: Optional[LLMCallable] = None) -> list[dict]:
    """
    Returns one dict per note, in note_index order, shaped exactly as:
    {
      "note_index": int,
      "applies": bool,                     # False only for no_op
      "directive_type": str,               # one of the 6 allowed types
      "structured_adjustment": dict|None,  # None only when directive_type=no_op
      "explanation": str
    }
    Never raises. Notes that cannot be interpreted safely become no_op entries
    whose explanation starts with "[guardrail]".

    `hours` is accepted for interface stability; interpretation does not need it.
    `llm` lets tests inject a fake model; by default the configured provider chain is used.
    """
    try:
        notes = [n if isinstance(n, str) else str(n) for n in (operator_notes or [])]
        battery = battery if isinstance(battery, dict) else {}
        if not notes:
            return []

        use_cache = llm is None
        key = _cache_key(notes, battery)
        if use_cache:
            with _cache_lock:
                hit = _cache.get(key)
            if hit is not None:
                return copy.deepcopy(hit)

        call = llm or default_llm
        user_prompt = build_user_prompt(notes, battery)
        payload = None
        started = time.monotonic()
        for attempt in range(2):  # one retry, only while well inside the judge's 30 s limit
            if attempt and time.monotonic() - started > RETRY_BUDGET_S:
                break
            try:
                payload = extract_json(call(SYSTEM_PROMPT, user_prompt))
            except Exception as exc:
                log.warning("LLM call failed on attempt %d (%s)", attempt + 1, type(exc).__name__)
                payload = None
            if _entries_from_payload(payload):
                break

        result = assemble(payload, len(notes), battery)

        # Only cache fully clean results, so transient failures get retried next time.
        if use_cache and not any(e["explanation"].startswith(GUARDRAIL_TAG) for e in result):
            with _cache_lock:
                if len(_cache) >= _CACHE_MAX:
                    _cache.clear()
                _cache[key] = copy.deepcopy(result)
        return result

    except Exception as exc:  # absolute last line of defence: never crash the API
        log.error("interpret_notes internal error (%s)", type(exc).__name__)
        n = len(operator_notes) if isinstance(operator_notes, list) else 0
        return [_fail(i, "internal error") for i in range(n)]
