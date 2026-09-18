"""Public sample-case validation for GridWise.

Run:  python -m pytest -v tests/test_public_cases.py

- Optimizer tests feed each sample's ground-truth directives straight into the
  optimizer and replay the plan exactly as the judge does (Problem Statement 09/11).
- API tests run each sample through the full HTTP pipeline. The ground-truth
  interpretation/application tests are skipped while app/interpreter.py is the stub.
"""
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.interpreter as interpreter
import app.main as main
from app.optimizer import optimize_schedule

TOL = 0.01
CASES = json.loads((Path(__file__).parent / "sample_cases.json").read_text(encoding="utf-8"))["cases"]
IDS = [c["id"] for c in CASES]
USING_STUB = getattr(interpreter, "IS_STUB", False) or not os.getenv("GROQ_API_KEY")

client = TestClient(main.app)


def replay_and_check(inp: dict, directives: list[dict], out: dict) -> None:
    """Independently replay a plan against the rules and the given directives."""
    hours = {h["hour"]: h for h in inp["hours"]}
    bat = inp["battery"]
    plan = out["hourly_plan"]

    assert len(plan) == 24
    assert sorted(p["hour"] for p in plan) == list(range(24))

    factor, floor = {}, {h: bat["minimum_energy_kwh"] for h in range(24)}
    no_chg, no_dis, grid_cap = set(), set(), {}
    for d in directives:
        if not d["applies"]:
            continue
        adj = d["structured_adjustment"]
        for h in adj["hours"]:
            t = d["directive_type"]
            if t == "solar_reduction":
                factor[h] = min(factor.get(h, 1.0), adj["factor"])
            elif t == "minimum_battery_reserve":
                floor[h] = max(floor[h], adj["minimum_energy_kwh"])
            elif t == "no_charge_window":
                no_chg.add(h)
            elif t == "no_discharge_window":
                no_dis.add(h)
            elif t == "max_grid_window":
                grid_cap[h] = min(grid_cap.get(h, adj["max_grid_kwh"]), adj["max_grid_kwh"])

    energy = bat["initial_energy_kwh"]
    for p in sorted(plan, key=lambda p: p["hour"]):
        h, act, amt = p["hour"], p["battery_action"], p["battery_kwh"]
        for k in ("grid_kwh", "solar_used_kwh", "battery_kwh", "battery_energy_after_kwh"):
            assert p[k] >= -TOL, f"hour {h}: negative {k}"
        assert act in ("charge", "discharge", "idle")
        chg = amt if act == "charge" else 0.0
        dis = amt if act == "discharge" else 0.0
        if act == "idle":
            assert abs(amt) <= TOL, f"hour {h}: idle with battery_kwh={amt}"
        assert chg <= bat["max_charge_kwh_per_hour"] + TOL, f"hour {h}: charge rate"
        assert dis <= bat["max_discharge_kwh_per_hour"] + TOL, f"hour {h}: discharge rate"
        if h in no_chg:
            assert chg <= TOL, f"hour {h}: charged in no_charge_window"
        if h in no_dis:
            assert dis <= TOL, f"hour {h}: discharged in no_discharge_window"

        eff_solar = hours[h]["solar_kwh"] * factor.get(h, 1.0)
        assert p["solar_used_kwh"] <= eff_solar + TOL, f"hour {h}: solar overuse"

        balance = p["grid_kwh"] + p["solar_used_kwh"] + dis - hours[h]["demand_kwh"] - chg
        assert abs(balance) <= TOL, f"hour {h}: energy balance off by {balance}"

        energy = energy + chg - dis
        assert abs(p["battery_energy_after_kwh"] - energy) <= TOL, f"hour {h}: battery transition"
        assert floor[h] - TOL <= energy <= bat["capacity_kwh"] + TOL, f"hour {h}: battery bounds"
        if h in grid_cap:
            assert p["grid_kwh"] <= grid_cap[h] + TOL, f"hour {h}: grid cap exceeded"

    assert abs(energy - bat["initial_energy_kwh"]) <= TOL, "end-of-day neutrality"

    total_grid = sum(p["grid_kwh"] for p in plan)
    total_cost = sum(p["grid_kwh"] * hours[p["hour"]]["tariff_bdt_per_kwh"] for p in plan)
    assert abs(out["total_grid_kwh"] - total_grid) <= TOL
    assert abs(out["total_cost_bdt"] - total_cost) <= TOL
    assert abs(out["peak_grid_kwh"] - max(p["grid_kwh"] for p in plan)) <= TOL


# ------------------------------------------------------------- optimizer ---

@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_optimizer_with_ground_truth_directives(case):
    inp, exp = case["input"], case["expected_output"]
    hours = sorted(inp["hours"], key=lambda h: h["hour"])
    out = optimize_schedule(hours, inp["battery"], exp["directive_interpretation"])
    replay_and_check(inp, exp["directive_interpretation"], out)
    assert not out["relaxed"]
    # Cost quality: must be at least as good as the organizer's reference plan.
    assert out["total_cost_bdt"] <= exp["total_cost_bdt"] + TOL


# ---------------------------------------------------------- API pipeline ---

def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_api_schema_and_self_consistency(case):
    inp = case["input"]
    r = client.post("/optimize-energy", json=inp)
    assert r.status_code == 200, r.text
    out = r.json()
    assert set(out) == {"scenario_id", "directive_interpretation", "hourly_plan",
                        "total_grid_kwh", "total_cost_bdt", "peak_grid_kwh", "plan_summary"}
    assert out["scenario_id"] == inp["scenario_id"]
    di = out["directive_interpretation"]
    assert [d["note_index"] for d in di] == list(range(len(inp["operator_notes"])))
    for d in di:
        assert (d["directive_type"] == "no_op") == (not d["applies"])
        if not d["applies"]:
            assert d["structured_adjustment"] is None
    # Plan must obey whatever the service itself reported.
    replay_and_check(inp, di, out)


@pytest.mark.skipif(USING_STUB, reason="LLM not configured (GROQ_API_KEY unset)")
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_api_ground_truth_interpretation(case):
    out = client.post("/optimize-energy", json=case["input"]).json()
    for got, want in zip(out["directive_interpretation"],
                         case["expected_output"]["directive_interpretation"], strict=True):
        assert got["directive_type"] == want["directive_type"], got
        assert got["applies"] == want["applies"]
        if want["structured_adjustment"] is None:
            assert got["structured_adjustment"] is None
            continue
        for k, v in want["structured_adjustment"].items():
            if k == "hours":
                assert got["structured_adjustment"]["hours"] == v
            else:
                assert abs(got["structured_adjustment"][k] - v) <= TOL, (k, got)


@pytest.mark.skipif(USING_STUB, reason="LLM not configured (GROQ_API_KEY unset)")
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_api_ground_truth_application_and_cost(case):
    exp = case["expected_output"]
    out = client.post("/optimize-energy", json=case["input"]).json()
    # The judge replays against the TRUE directives, not our reported ones.
    replay_and_check(case["input"], exp["directive_interpretation"], out)
    assert out["total_cost_bdt"] <= exp["total_cost_bdt"] + TOL


# ------------------------------------------------------ request validation ---

def _sample():
    return json.loads(json.dumps(CASES[0]["input"]))


@pytest.mark.parametrize("mutate", [
    lambda b: b.pop("battery"),
    lambda b: b.update(operator_notes=[]),
    lambda b: b.update(operator_notes=["a", "b", "c", "d"]),
    lambda b: b.update(operator_notes=["   "]),
    lambda b: b["hours"].pop(),
    lambda b: b["hours"].__setitem__(1, dict(b["hours"][0])),  # duplicate hour 0
    lambda b: b["hours"][3].update(demand_kwh="lots"),
    lambda b: b["battery"].update(capacity_kwh=-1),
], ids=["no_battery", "no_notes", "four_notes", "blank_note", "23_hours",
        "duplicate_hour", "non_numeric", "negative"])
def test_structurally_invalid_request_is_400(mutate):
    body = _sample()
    mutate(body)
    r = client.post("/optimize-energy", json=body)
    assert r.status_code == 400
    assert "Traceback" not in r.text


def test_malformed_json_is_400():
    r = client.post("/optimize-energy", content=b"{not json",
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 400


def test_shuffled_hours_accepted():
    body = _sample()
    body["hours"].reverse()
    r = client.post("/optimize-energy", json=body)
    assert r.status_code == 200
    assert [p["hour"] for p in r.json()["hourly_plan"]] == list(range(24))


# ------------------------------------------------ guardrails / robustness ---

def test_bad_interpreter_output_becomes_no_op(monkeypatch):
    def bad(notes, hours, battery):
        return [
            {"note_index": 0, "applies": True, "directive_type": "turbo_mode",
             "structured_adjustment": {"hours": [1]}, "explanation": "x"},
            {"note_index": 1, "applies": True, "directive_type": "solar_reduction",
             "structured_adjustment": {"hours": [30], "factor": 3}, "explanation": "x"},
        ]
    monkeypatch.setattr(main, "interpret_notes", bad)
    out = client.post("/optimize-energy", json=_sample()).json()
    assert [d["directive_type"] for d in out["directive_interpretation"]] == ["no_op", "no_op"]


def test_interpreter_crash_degrades_gracefully(monkeypatch):
    def boom(notes, hours, battery):
        raise RuntimeError("provider down, key=sk-should-never-leak")
    monkeypatch.setattr(main, "interpret_notes", boom)
    r = client.post("/optimize-energy", json=_sample())
    assert r.status_code == 200
    assert "sk-should-never-leak" not in r.text


def test_infeasible_directives_are_relaxed_not_crashed(monkeypatch):
    def impossible(notes, hours, battery):
        return [{"note_index": i, "applies": True, "directive_type": "max_grid_window",
                 "structured_adjustment": {"hours": list(range(24)), "max_grid_kwh": 0},
                 "explanation": "x"} for i in range(len(notes))]
    monkeypatch.setattr(main, "interpret_notes", impossible)
    r = client.post("/optimize-energy", json=_sample())
    assert r.status_code == 200
    out = r.json()
    assert out["hourly_plan"] and "relaxed" in out["plan_summary"]
