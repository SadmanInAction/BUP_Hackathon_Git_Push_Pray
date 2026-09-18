"""Request/response models for POST /optimize-energy.

Field names and types follow the Problem Statement, sections 07 (request) and 10 (response).
"""
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DirectiveType = Literal[
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]

# Finite, non-negative numbers only (rejects NaN / Infinity / negatives).
NonNegFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]


# ---------------------------------------------------------------- request ---

class HourIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hour: int = Field(ge=0, le=23, strict=True)
    demand_kwh: NonNegFloat
    solar_kwh: NonNegFloat
    tariff_bdt_per_kwh: NonNegFloat


class BatteryIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    capacity_kwh: NonNegFloat
    initial_energy_kwh: NonNegFloat
    minimum_energy_kwh: NonNegFloat
    max_charge_kwh_per_hour: NonNegFloat
    max_discharge_kwh_per_hour: NonNegFloat

    @model_validator(mode="after")
    def _consistent(self):
        if self.minimum_energy_kwh > self.capacity_kwh:
            raise ValueError("minimum_energy_kwh must not exceed capacity_kwh")
        if not (self.minimum_energy_kwh <= self.initial_energy_kwh <= self.capacity_kwh):
            raise ValueError("initial_energy_kwh must be between minimum_energy_kwh and capacity_kwh")
        return self


class OptimizeRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    scenario_id: str
    operator_notes: list[str] = Field(min_length=1, max_length=3)
    hours: list[HourIn] = Field(min_length=24, max_length=24)
    battery: BatteryIn

    @field_validator("operator_notes")
    @classmethod
    def _notes_non_empty(cls, notes: list[str]) -> list[str]:
        if any(not n.strip() for n in notes):
            raise ValueError("operator_notes entries must be non-empty strings")
        return notes

    @field_validator("hours")
    @classmethod
    def _hours_cover_day(cls, hours: list[HourIn]) -> list[HourIn]:
        if sorted(h.hour for h in hours) != list(range(24)):
            raise ValueError("hours must contain each hour 0-23 exactly once")
        return sorted(hours, key=lambda h: h.hour)


# --------------------------------------------------------------- response ---

class DirectiveInterpretation(BaseModel):
    note_index: int
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: Optional[dict]
    explanation: str


class HourPlan(BaseModel):
    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: Literal["charge", "discharge", "idle"]
    battery_kwh: float
    battery_energy_after_kwh: float


class OptimizeResponse(BaseModel):
    scenario_id: str
    directive_interpretation: list[DirectiveInterpretation]
    hourly_plan: list[HourPlan]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str


class HealthResponse(BaseModel):
    status: str
