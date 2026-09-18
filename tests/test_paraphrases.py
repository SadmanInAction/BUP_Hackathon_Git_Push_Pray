"""Paraphrase robustness for the LLM interpreter (requires GROQ_API_KEY).

Run:  python -m pytest -v tests/test_paraphrases.py
"""
import os

import pytest

from app.guardrails import validate_interpretation
from app.interpreter import interpret_notes

pytestmark = pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY unset")

BATTERY = {"capacity_kwh": 200, "initial_energy_kwh": 100, "minimum_energy_kwh": 20,
           "max_charge_kwh_per_hour": 50, "max_discharge_kwh_per_hour": 50}
HOURS = [{"hour": h, "demand_kwh": 100, "solar_kwh": 0, "tariff_bdt_per_kwh": 8} for h in range(24)]

# (note, directive_type, hours, {numeric field: value})
CASES = [
    # Problem Statement 11.4 paraphrases
    ("PV production will drop to about 20% between 13:00 and 15:00.", "solar_reduction", [13, 14], {"factor": 0.2}),
    ("Panel washing from one until three will leave roughly one-fifth of normal solar output.", "solar_reduction", [13, 14], {"factor": 0.2}),
    ("Expect an 80% reduction in rooftop solar during the 1-3 PM maintenance window.", "solar_reduction", [13, 14], {"factor": 0.2}),
    # Problem Statement 4.2 examples
    ("Solar output will drop to about 20% from 1 PM to 3 PM.", "solar_reduction", [13, 14], {"factor": 0.2}),
    ("Do not charge the battery between 2 PM and 4 PM.", "no_charge_window", [14, 15], {}),
    ("Keep at least 120 kWh in reserve from 6 PM until 9 PM.", "minimum_battery_reserve", [18, 19, 20], {"minimum_energy_kwh": 120}),
    ("The cafeteria menu changes tomorrow.", "no_op", None, {}),
    # Extra variations
    ("Solar will be cut in half from 9 AM to noon due to haze.", "solar_reduction", [9, 10, 11], {"factor": 0.5}),
    ("Maintain a battery reserve of 25% of capacity between 17:00 and 20:00.", "minimum_battery_reserve", [17, 18, 19], {"minimum_energy_kwh": 50}),
    ("The feeder can supply no more than 120 kWh per hour from 5 PM to 8 PM.", "max_grid_window", [17, 18, 19], {"max_grid_kwh": 120}),
    ("Battery discharge is prohibited between 7 and 9 in the evening.", "no_discharge_window", [19, 20], {}),
    ("The charger is offline from midnight until 3 AM.", "no_charge_window", [0, 1, 2], {}),
    ("Grid draw must stay at or below 90 kWh from 10 PM until midnight.", "max_grid_window", [22, 23], {"max_grid_kwh": 90}),
    ("The physics department is hosting a guest lecture this afternoon.", "no_op", None, {}),
    ("Next week the solar panels will be cleaned.", "no_op", None, {}),
]


@pytest.mark.parametrize("note,dtype,hours,values", CASES, ids=[c[0][:40] for c in CASES])
def test_paraphrase(note, dtype, hours, values):
    raw = interpret_notes([note], HOURS, BATTERY)
    d = validate_interpretation(raw, 1, BATTERY)[0]
    assert d["directive_type"] == dtype, d
    if dtype == "no_op":
        assert d["applies"] is False and d["structured_adjustment"] is None
        return
    assert d["applies"] is True
    assert d["structured_adjustment"]["hours"] == hours, d
    for k, v in values.items():
        assert abs(d["structured_adjustment"][k] - v) <= 0.01, d
