"""Deterministic guardrails applied to interpreter output before optimization.

Problem Statement section 08: LLM output is untrusted structured data. Any entry
that fails validation is replaced by a safe no_op, so an invalid model output can
never invent a constraint or crash the service.
"""
import math

SAFE_NO_OP_EXPLANATION = "Interpretation could not be validated; treated as no_op."

# directive_type -> required numeric field (besides "hours"), or None
_REQUIRED_FIELDS = {
    "solar_reduction": "factor",
    "minimum_battery_reserve": "minimum_energy_kwh",
    "no_charge_window": None,
    "no_discharge_window": None,
    "max_grid_window": "max_grid_kwh",
}


def no_op(note_index: int, explanation: str = SAFE_NO_OP_EXPLANATION) -> dict:
    return {
        "note_index": note_index,
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": explanation,
    }


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _clean_hours(raw) -> list[int] | None:
    if not isinstance(raw, list) or not raw:
        return None
    hours = []
    for h in raw:
        if isinstance(h, bool):
            return None
        if isinstance(h, float) and h.is_integer():
            h = int(h)
        if not isinstance(h, int) or not 0 <= h <= 23:
            return None
        hours.append(h)
    return sorted(set(hours))


def _validate_entry(entry, note_index: int, capacity_kwh: float) -> dict:
    if not isinstance(entry, dict):
        return no_op(note_index)
    dtype = entry.get("directive_type")
    explanation = entry.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        explanation = f"Interpreted as {dtype}."

    if dtype == "no_op":
        return no_op(note_index, explanation)
    if dtype not in _REQUIRED_FIELDS:
        return no_op(note_index)

    adj = entry.get("structured_adjustment")
    if not isinstance(adj, dict):
        return no_op(note_index)
    hours = _clean_hours(adj.get("hours"))
    if hours is None:
        return no_op(note_index)

    clean_adj = {"hours": hours}
    field = _REQUIRED_FIELDS[dtype]
    if field is not None:
        value = adj.get(field)
        if not _is_number(value) or value < 0:
            return no_op(note_index)
        if dtype == "solar_reduction" and value > 1:
            return no_op(note_index)
        if dtype == "minimum_battery_reserve" and value > capacity_kwh:
            return no_op(note_index)
        clean_adj[field] = value

    return {
        "note_index": note_index,
        "applies": True,
        "directive_type": dtype,
        "structured_adjustment": clean_adj,
        "explanation": explanation,
    }


def validate_interpretation(raw, n_notes: int, battery: dict) -> list[dict]:
    """Return exactly n_notes validated entries in note_index order 0..n-1."""
    by_index: dict[int, object] = {}
    if isinstance(raw, list):
        for pos, entry in enumerate(raw):
            idx = entry.get("note_index", pos) if isinstance(entry, dict) else pos
            if isinstance(idx, int) and not isinstance(idx, bool) and 0 <= idx < n_notes:
                by_index.setdefault(idx, entry)
    capacity = battery["capacity_kwh"]
    return [_validate_entry(by_index.get(i), i, capacity) for i in range(n_notes)]
