"""
Tests for app/interpreter.py.

Three groups:
  1. Public sample cases vs expected_output.directive_interpretation   (live LLM)
  2. Hand-written paraphrases of every directive type                   (live LLM)
  3. Malformed / hostile model output never crashes the interpreter     (offline, fake LLM)

Live tests are skipped automatically when no provider is reachable.
Run only the offline tests with:  pytest tests -m "not live"
"""

import json
import math
import os
import urllib.request
from pathlib import Path

import pytest

from app import interpreter
from app.interpreter import ALLOWED_TYPES, GUARDRAIL_TAG, interpret_notes

SAMPLES = json.loads((Path(__file__).parent / "sample_cases.json").read_text(encoding="utf-8"))["cases"]
BATTERY = {"capacity_kwh": 200, "initial_energy_kwh": 100, "minimum_energy_kwh": 30,
           "max_charge_kwh_per_hour": 50, "max_discharge_kwh_per_hour": 50}
HOURS = [{"hour": h, "demand_kwh": 100, "solar_kwh": 0, "tariff_bdt_per_kwh": 8} for h in range(24)]
TOL = 0.01


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _llm_available() -> bool:
    for name in interpreter._provider_chain():
        if name == "groq" and os.environ.get("GROQ_API_KEY"):
            return True
        if name == "gemini" and (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            return True
        if name == "ollama":
            host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
            try:
                with urllib.request.urlopen(f"{host}/api/tags", timeout=2) as r:
                    models = [m["name"] for m in json.loads(r.read())["models"]]
                want = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
                if any(m == want or m.startswith(want + ":") for m in models):
                    return True
            except Exception:
                pass
    return False


live = pytest.mark.skipif(not _llm_available(), reason="no LLM provider reachable")


def assert_shape(result, n_notes):
    """The contract from the Problem Statement, checked on every result."""
    assert isinstance(result, list) and len(result) == n_notes
    for i, e in enumerate(result):
        assert set(e) == {"note_index", "applies", "directive_type", "structured_adjustment", "explanation"}
        assert e["note_index"] == i
        assert e["directive_type"] in ALLOWED_TYPES
        assert isinstance(e["explanation"], str) and e["explanation"]
        if e["directive_type"] == "no_op":
            assert e["applies"] is False and e["structured_adjustment"] is None
            continue
        assert e["applies"] is True
        adj = e["structured_adjustment"]
        hours = adj["hours"]
        assert hours and all(isinstance(h, int) and not isinstance(h, bool) and 0 <= h <= 23 for h in hours)
        assert hours == sorted(set(hours))
        expected_keys = {"hours"} | ({interpreter.NUMERIC_FIELD[e["directive_type"]]} - {None})
        assert set(adj) == expected_keys
        for k in expected_keys - {"hours"}:
            assert isinstance(adj[k], (int, float)) and math.isfinite(adj[k]) and adj[k] >= 0
        if e["directive_type"] == "solar_reduction":
            assert 0 <= adj["factor"] <= 1


def assert_matches(got, expected):
    assert got["applies"] == expected["applies"], got
    assert got["directive_type"] == expected["directive_type"], got
    exp_adj = expected["structured_adjustment"]
    if exp_adj is None:
        assert got["structured_adjustment"] is None
        return
    assert got["structured_adjustment"]["hours"] == exp_adj["hours"], got
    for k, v in exp_adj.items():
        if k != "hours":
            assert abs(got["structured_adjustment"][k] - v) <= TOL, got


def fake(text):
    """A fake LLM that always returns `text`, regardless of prompt."""
    return lambda system, user: text


def entry(i, dtype, adj=None, applies=True, explanation="x"):
    return {"note_index": i, "applies": applies, "directive_type": dtype,
            "structured_adjustment": adj, "explanation": explanation}


def wrap(*entries):
    return json.dumps({"interpretations": list(entries)})


# --------------------------------------------------------------------------- #
# 1. Public sample cases (live)
# --------------------------------------------------------------------------- #

@live
@pytest.mark.live
@pytest.mark.parametrize("case", SAMPLES, ids=[c["id"] for c in SAMPLES])
def test_public_samples(case):
    inp = case["input"]
    got = interpret_notes(inp["operator_notes"], inp["hours"], inp["battery"])
    assert_shape(got, len(inp["operator_notes"]))
    for g, e in zip(got, case["expected_output"]["directive_interpretation"]):
        assert_matches(g, e)


# --------------------------------------------------------------------------- #
# 2. Paraphrases written from scratch (live). Battery capacity is 200 kWh.
# --------------------------------------------------------------------------- #

S, R, NC, ND, G, NOP = ("solar_reduction", "minimum_battery_reserve", "no_charge_window",
                        "no_discharge_window", "max_grid_window", "no_op")

PARAPHRASES = [
    # solar_reduction
    ("PV production will drop to about 20% between 13:00 and 15:00.", S, [13, 14], 0.2),
    ("Panel washing from one until three in the afternoon will leave roughly one-fifth of normal solar output.", S, [13, 14], 0.2),
    ("Panel washing from one until three will leave roughly one-fifth of normal solar output.", S, [13, 14], 0.2),
    ("Expect an 80% reduction in rooftop solar during the 1-3 PM maintenance window.", S, [13, 14], 0.2),
    ("Heavy haze is expected to cut rooftop generation by three quarters from 9 AM to 11 AM.", S, [9, 10], 0.25),
    ("A crane will fully shade the solar array from 15:00 to 17:00, so assume no PV at all.", S, [15, 16], 0.0),
    ("Only 40 percent of the predicted solar yield will be usable from 10 in the morning until 1 in the afternoon.", S, [10, 11, 12], 0.4),
    # minimum_battery_reserve
    ("Hold a minimum of 75 kWh of stored energy from 20:00 to 23:00 as backup for the labs.", R, [20, 21, 22], 75),
    ("Between 4 PM and 7 PM the battery must not fall below a quarter of its capacity.", R, [16, 17, 18], 50),
    ("Keep 60 kWh in the battery tonight from 7 until 10.", R, [19, 20, 21], 60),
    ("The medical centre needs 120 kilowatt-hours kept in storage from 5 o'clock in the evening until 8 PM.", R, [17, 18, 19], 120),
    # no_charge_window
    ("The battery can't accept any charge from 03:00 to 06:00 while the BMS firmware is updated.", NC, [3, 4, 5], None),
    ("Charging of the storage system is suspended between 9 AM and noon.", NC, [9, 10, 11], None),
    # no_discharge_window
    ("From 8 PM to 11 PM no energy may be drawn out of the battery.", ND, [20, 21, 22], None),
    ("Battery output is blocked from 7 AM to 9 AM for relay calibration.", ND, [7, 8], None),
    # max_grid_window
    ("The utility has asked us to keep hourly imports at or below 120 kWh between 5 PM and 8 PM.", G, [17, 18, 19], 120),
    ("Due to substation work, grid draw is limited to one hundred kWh each hour from 14:00 until 16:00.", G, [14, 15], 100),
    # no_op distractors, including ones that mention energy words or clock times
    ("The engineering faculty will host a robotics workshop on Friday.", NOP, None, None),
    ("Next month the solar vendor will send over a new maintenance contract for review.", NOP, None, None),
    ("Security guards will change shifts at 6 PM tonight.", NOP, None, None),
]


@live
@pytest.mark.live
@pytest.mark.parametrize("note,dtype,hours,value", PARAPHRASES, ids=[p[0][:40] for p in PARAPHRASES])
def test_paraphrases(note, dtype, hours, value):
    got = interpret_notes([note], HOURS, BATTERY)
    assert_shape(got, 1)
    e = got[0]
    assert e["directive_type"] == dtype, e
    if dtype == NOP:
        return
    assert e["structured_adjustment"]["hours"] == hours, e
    field = interpreter.NUMERIC_FIELD[dtype]
    if field:
        assert abs(e["structured_adjustment"][field] - value) <= TOL, e


@live
@pytest.mark.live
def test_mixed_batch_keeps_order():
    first_of = {dtype: note for note, dtype, _, _ in reversed(PARAPHRASES)}
    notes = [first_of[NOP], first_of[G], first_of[S]]
    got = interpret_notes(notes, HOURS, BATTERY)
    assert_shape(got, 3)
    assert [e["directive_type"] for e in got] == [NOP, G, S]


# --------------------------------------------------------------------------- #
# 3. Guardrails against malformed model output (offline)
# --------------------------------------------------------------------------- #

GARBAGE_OUTPUTS = [
    "",
    "Sure! Here is the interpretation you asked for.",
    "{not valid json",
    "null",
    "42",
    "[]",
    '{"interpretations": "nope"}',
    '{"interpretations": [null, 5, "text"]}',
    json.dumps({"interpretations": [entry(0, "reduce_everything", {"hours": [1]})]}),
    json.dumps({"interpretations": [entry(0, "solar_reduction", None)]}),
    json.dumps({"interpretations": [entry(0, "solar_reduction", {"hours": [13], "factor": "lots"})]}),
    json.dumps({"interpretations": [entry(0, "solar_reduction", {"hours": [13], "factor": 1.5e9})]}),
    json.dumps({"interpretations": [entry(0, "no_charge_window", {"hours": [25, -1]})]}),
    json.dumps({"interpretations": [entry(0, "no_charge_window", {"hours": []})]}),
    json.dumps({"interpretations": [entry(0, "max_grid_window", {"hours": [3], "max_grid_kwh": -5})]}),
    json.dumps({"interpretations": [entry(0, "minimum_battery_reserve", {"hours": [3], "minimum_energy_kwh": 9999})]}),
    json.dumps({"interpretations": [entry(0, "minimum_battery_reserve", {"hours": [3], "minimum_energy_kwh": True})]}),
    '{"interpretations": [{"note_index": 0, "directive_type": "max_grid_window", "structured_adjustment": {"hours": [3], "max_grid_kwh": NaN}}]}',
    "[" * 5000 + "]" * 5000,
]


@pytest.mark.parametrize("raw", GARBAGE_OUTPUTS, ids=range(len(GARBAGE_OUTPUTS)))
def test_garbage_output_fails_safe_to_flagged_no_op(raw):
    got = interpret_notes(["Do not charge the battery between 2 PM and 4 PM."], HOURS, BATTERY, llm=fake(raw))
    assert_shape(got, 1)
    assert got[0]["directive_type"] == "no_op"
    assert got[0]["explanation"].startswith(GUARDRAIL_TAG)


@pytest.mark.parametrize("bad_return", [None, 123, b"bytes", {"a": 1}])
def test_non_string_llm_return_is_safe(bad_return):
    got = interpret_notes(["note a", "note b"], HOURS, BATTERY, llm=lambda s, u: bad_return)
    assert_shape(got, 2)
    assert all(e["directive_type"] == "no_op" for e in got)


def test_llm_exception_is_safe_and_retried():
    calls = []

    def boom(system, user):
        calls.append(1)
        raise TimeoutError("provider down")

    got = interpret_notes(["a", "b", "c"], HOURS, BATTERY, llm=boom)
    assert_shape(got, 3)
    assert all(e["explanation"].startswith(GUARDRAIL_TAG) for e in got)
    assert len(calls) == 2  # one retry, then give up


def test_retry_recovers_after_one_bad_response():
    responses = iter(["garbage", wrap(entry(0, "no_charge_window", {"hours": [14, 15]}))])
    got = interpret_notes(["x"], HOURS, BATTERY, llm=lambda s, u: next(responses))
    assert got[0]["directive_type"] == "no_charge_window"


def test_end_exclusive_window_overrides_bad_enumeration():
    # Model states the window correctly but enumerates it end-inclusive.
    raw = wrap(entry(0, "solar_reduction", {"start_hour": 13, "end_hour": 15, "hours": [13, 14, 15], "factor": 0.2}))
    got = interpret_notes(["x"], HOURS, BATTERY, llm=fake(raw))
    assert got[0]["structured_adjustment"] == {"hours": [13, 14], "factor": 0.2}


def test_window_wrapping_midnight():
    raw = wrap(entry(0, "no_discharge_window", {"start_hour": 22, "end_hour": 2}))
    got = interpret_notes(["x"], HOURS, BATTERY, llm=fake(raw))
    assert got[0]["structured_adjustment"] == {"hours": [0, 1, 22, 23]}


def test_window_until_midnight_as_24():
    raw = wrap(entry(0, "no_charge_window", {"start_hour": 21, "end_hour": 24}))
    got = interpret_notes(["x"], HOURS, BATTERY, llm=fake(raw))
    assert got[0]["structured_adjustment"] == {"hours": [21, 22, 23]}


def test_hours_list_is_deduplicated_sorted_and_coerced():
    raw = wrap(entry(0, "no_charge_window", {"hours": [15, "14", 14.0, 15]}))
    got = interpret_notes(["x"], HOURS, BATTERY, llm=fake(raw))
    assert got[0]["structured_adjustment"] == {"hours": [14, 15]}


def test_factor_given_as_percent_is_normalised():
    raw = wrap(entry(0, "solar_reduction", {"hours": [12], "factor": "25%"}))
    got = interpret_notes(["x"], HOURS, BATTERY, llm=fake(raw))
    assert got[0]["structured_adjustment"]["factor"] == 0.25


def test_extra_keys_are_stripped_and_applies_is_derived_from_type():
    raw = wrap(
        entry(0, "max_grid_window", {"hours": [18], "max_grid_kwh": 150, "tariff": 1, "demand_kwh": 0}, applies=False),
        entry(1, "no_op", {"hours": [1]}, applies=True),
    )
    got = interpret_notes(["a", "b"], HOURS, BATTERY, llm=fake(raw))
    assert_shape(got, 2)
    assert got[0]["applies"] is True and got[0]["structured_adjustment"] == {"hours": [18], "max_grid_kwh": 150}
    assert got[1]["applies"] is False and got[1]["structured_adjustment"] is None


def test_directive_type_spelling_is_normalised_but_not_invented():
    raw = wrap(entry(0, "No-Charge Window", {"hours": [2]}), entry(1, "no_charging", {"hours": [2]}))
    got = interpret_notes(["a", "b"], HOURS, BATTERY, llm=fake(raw))
    assert got[0]["directive_type"] == "no_charge_window"
    assert got[1]["directive_type"] == "no_op" and got[1]["explanation"].startswith(GUARDRAIL_TAG)


def test_missing_duplicate_and_out_of_range_indices():
    raw = wrap(
        entry(2, "no_charge_window", {"hours": [1]}),
        entry(2, "no_discharge_window", {"hours": [5]}),  # duplicate: ignored
        entry(7, "no_charge_window", {"hours": [1]}),     # no such note: ignored
        entry(0, "no_op", None, applies=False),
        # note 1 missing entirely
    )
    got = interpret_notes(["a", "b", "c"], HOURS, BATTERY, llm=fake(raw))
    assert_shape(got, 3)
    assert got[0]["directive_type"] == "no_op" and not got[0]["explanation"].startswith(GUARDRAIL_TAG)
    assert got[1]["directive_type"] == "no_op" and got[1]["explanation"].startswith(GUARDRAIL_TAG)
    assert got[2]["directive_type"] == "no_charge_window"


def test_markdown_fenced_and_bare_list_payloads():
    fenced = "Here you go:\n```json\n" + wrap(entry(0, "no_charge_window", {"hours": [3]})) + "\n```"
    assert interpret_notes(["x"], HOURS, BATTERY, llm=fake(fenced))[0]["directive_type"] == "no_charge_window"
    bare = json.dumps([{"directive_type": "no_discharge_window", "structured_adjustment": {"hours": [4]}}])
    assert interpret_notes(["x"], HOURS, BATTERY, llm=fake(bare))[0]["directive_type"] == "no_discharge_window"


@pytest.mark.parametrize("notes,battery", [(None, BATTERY), ([], BATTERY), (["x"], None), (["x"], "junk"),
                                           ([123, None], {"capacity_kwh": "abc"})])
def test_bad_inputs_do_not_crash(notes, battery):
    got = interpret_notes(notes, HOURS, battery, llm=fake("garbage"))
    assert_shape(got, len(notes or []))


def test_prompt_contains_capacity_and_every_note():
    seen = {}

    def spy(system, user):
        seen["user"] = user
        return "{}"

    interpret_notes(["first\nnote", "second"], HOURS, {"capacity_kwh": 260}, llm=spy)
    assert "capacity_kwh: 260" in seen["user"]
    assert "[0] first note" in seen["user"] and "[1] second" in seen["user"]


def test_reasoning_args_per_model_family():
    assert interpreter._reasoning_args("openai/gpt-oss-120b") == {"reasoning_effort": "low"}
    assert interpreter._reasoning_args("qwen/qwen3.8-27b") == {"reasoning_effort": "none"}
    assert interpreter._reasoning_args("some/other-model") == {}


def _http_error(code, retry_after=None):
    import email.message
    import urllib.error

    headers = email.message.Message()
    if retry_after is not None:
        headers["retry-after"] = str(retry_after)
    return urllib.error.HTTPError("https://api.groq.com", code, "err", headers, None)


def test_groq_rotates_models_and_waits_out_rate_limit_once(monkeypatch):
    calls = []
    sleeps = []
    ok_body = {"choices": [{"message": {"content": wrap(entry(0, "no_charge_window", {"hours": [2]}))}}]}

    def fake_post(url, payload, headers):
        calls.append(payload["model"])
        if len(calls) <= 2:  # both models rate-limited on the first pass
            raise _http_error(429, retry_after=1)
        return ok_body

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODELS", "m1,m2")
    monkeypatch.setattr(interpreter, "_post_json", fake_post)
    monkeypatch.setattr(interpreter.time, "sleep", sleeps.append)
    out = interpreter.call_groq("sys", "user")
    assert "no_charge_window" in out
    assert calls == ["m1", "m2", "m1"]
    assert sleeps and sleeps[0] <= interpreter.MAX_RATE_LIMIT_WAIT_S + 1


def test_groq_does_not_wait_on_long_retry_after_or_non_429(monkeypatch):
    sleeps = []
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODELS", "m1,m2")
    monkeypatch.setattr(interpreter.time, "sleep", sleeps.append)

    def long_wait(url, payload, headers):
        raise _http_error(429, retry_after=60)

    monkeypatch.setattr(interpreter, "_post_json", long_wait)
    with pytest.raises(Exception):
        interpreter.call_groq("sys", "user")

    def server_error(url, payload, headers):
        raise _http_error(503)

    monkeypatch.setattr(interpreter, "_post_json", server_error)
    with pytest.raises(Exception):
        interpreter.call_groq("sys", "user")
    assert sleeps == []
