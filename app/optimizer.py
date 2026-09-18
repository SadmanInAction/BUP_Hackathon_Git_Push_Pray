"""24-hour battery/grid scheduling as a mixed-integer linear program (PuLP + CBC).

Rules implemented from Problem Statement sections 05.3 and 09.
"""
import pulp

ROUND = 4               # decimals in the returned plan (judge tolerance is 0.01)
ACTION_EPS = 1e-6       # battery flows below this are treated as idle
CYCLE_PENALTY = 1e-6    # tiny tie-breaker: avoids pointless battery cycling
SLACK_PENALTY = 1e6     # used only by the relaxed fallback model
SOLVER_TIME_LIMIT_S = 20


class OptimizationError(Exception):
    pass


def _merge_directives(hours: list[dict], battery: dict, directives: list[dict]) -> dict:
    """Merge every applicable directive, per type, into per-hour limits."""
    factor: dict[int, float] = {}
    floor = {h: float(battery["minimum_energy_kwh"]) for h in range(24)}
    no_charge: set[int] = set()
    no_discharge: set[int] = set()
    grid_cap: dict[int, float] = {}

    for d in directives:
        if not d.get("applies"):
            continue
        adj = d.get("structured_adjustment") or {}
        dtype = d.get("directive_type")
        for h in adj.get("hours", []):
            if dtype == "solar_reduction":
                f = min(max(float(adj["factor"]), 0.0), 1.0)
                factor[h] = min(factor.get(h, 1.0), f)  # overlapping reductions: keep the stricter
            elif dtype == "minimum_battery_reserve":
                floor[h] = max(floor[h], float(adj["minimum_energy_kwh"]))
            elif dtype == "no_charge_window":
                no_charge.add(h)
            elif dtype == "no_discharge_window":
                no_discharge.add(h)
            elif dtype == "max_grid_window":
                cap = float(adj["max_grid_kwh"])
                grid_cap[h] = min(grid_cap.get(h, cap), cap)

    solar = {h: float(hours[h]["solar_kwh"]) * factor.get(h, 1.0) for h in range(24)}
    return {"solar": solar, "floor": floor, "no_charge": no_charge,
            "no_discharge": no_discharge, "grid_cap": grid_cap}


def _solve(hours, battery, m, relaxed: bool):
    H = range(24)
    cap = float(battery["capacity_kwh"])
    e0 = float(battery["initial_energy_kwh"])
    max_c = float(battery["max_charge_kwh_per_hour"])
    max_d = float(battery["max_discharge_kwh_per_hour"])

    prob = pulp.LpProblem("gridwise", pulp.LpMinimize)
    grid = pulp.LpVariable.dicts("grid", H, lowBound=0)
    sol = {h: pulp.LpVariable(f"solar_{h}", lowBound=0, upBound=m["solar"][h]) for h in H}
    chg = {h: pulp.LpVariable(f"chg_{h}", lowBound=0,
                              upBound=0 if h in m["no_charge"] else max_c) for h in H}
    dis = {h: pulp.LpVariable(f"dis_{h}", lowBound=0,
                              upBound=0 if h in m["no_discharge"] else max_d) for h in H}
    energy = pulp.LpVariable.dicts("energy", H, lowBound=0, upBound=cap)
    is_chg = pulp.LpVariable.dicts("is_chg", H, cat="Binary")
    slack_floor = pulp.LpVariable.dicts("slack_floor", H, lowBound=0) if relaxed else None
    slack_grid = pulp.LpVariable.dicts("slack_grid", H, lowBound=0) if relaxed else None

    objective = pulp.lpSum(grid[h] * float(hours[h]["tariff_bdt_per_kwh"]) for h in H)
    objective += CYCLE_PENALTY * pulp.lpSum(chg[h] + dis[h] for h in H)
    if relaxed:
        objective += SLACK_PENALTY * pulp.lpSum(slack_floor[h] + slack_grid[h] for h in H)
    prob += objective

    for h in H:
        prev = e0 if h == 0 else energy[h - 1]
        prob += grid[h] + sol[h] + dis[h] == float(hours[h]["demand_kwh"]) + chg[h]
        prob += energy[h] == prev + chg[h] - dis[h]
        # Exactly one of charge / discharge / idle per hour.
        prob += chg[h] <= max_c * is_chg[h]
        prob += dis[h] <= max_d * (1 - is_chg[h])
        if relaxed:
            prob += energy[h] + slack_floor[h] >= m["floor"][h]
        else:
            prob += energy[h] >= m["floor"][h]
        if h in m["grid_cap"]:
            if relaxed:
                prob += grid[h] <= m["grid_cap"][h] + slack_grid[h]
            else:
                prob += grid[h] <= m["grid_cap"][h]
    prob += energy[23] == e0  # end-of-day neutrality

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=SOLVER_TIME_LIMIT_S))
    if pulp.LpStatus[status] != "Optimal":
        return None
    return {h: (sol[h].value() or 0.0, chg[h].value() or 0.0, dis[h].value() or 0.0) for h in H}


def _build_plan(hours, battery, m, flows) -> list[dict]:
    """Turn raw LP values into a plan that is exactly self-consistent.

    battery energy is replayed from the rounded battery amounts and grid_kwh is
    derived from the energy-balance equation, so validity never depends on
    solver floating-point noise.
    """
    energy = float(battery["initial_energy_kwh"])
    plan = []
    for h in range(24):
        s, c, d = flows[h]
        net = c - d
        if net > ACTION_EPS:
            action, amount = "charge", round(net, ROUND)
        elif net < -ACTION_EPS:
            action, amount = "discharge", round(-net, ROUND)
        else:
            action, amount = "idle", 0.0
        if amount == 0:
            action = "idle"
        charge = amount if action == "charge" else 0.0
        discharge = amount if action == "discharge" else 0.0

        solar_used = min(max(round(s, ROUND), 0.0), m["solar"][h])
        grid = float(hours[h]["demand_kwh"]) + charge - discharge - solar_used
        if grid < 0:  # numerical noise: shed the excess from solar instead
            solar_used = max(solar_used + grid, 0.0)
            grid = 0.0
        energy = energy + charge - discharge

        plan.append({
            "hour": h,
            "grid_kwh": round(grid, ROUND),
            "solar_used_kwh": round(solar_used, ROUND),
            "battery_action": action,
            "battery_kwh": amount,
            "battery_energy_after_kwh": round(energy, ROUND),
        })
    return plan


def optimize_schedule(hours: list[dict], battery: dict, directives: list[dict]) -> dict:
    """Minimise grid cost subject to GridWise rules plus all applicable directives.

    `hours` must hold 24 entries ordered by hour 0..23. Only directives with
    applies=True are used. Returns hourly_plan plus totals computed from it.
    """
    m = _merge_directives(hours, battery, directives)
    relaxed = False
    flows = _solve(hours, battery, m, relaxed=False)
    if flows is None:
        # Should not happen for valid judge scenarios; keeps the service up if an
        # interpreted directive is infeasible by violating it as little as possible.
        relaxed = True
        flows = _solve(hours, battery, m, relaxed=True)
    if flows is None:
        raise OptimizationError("no feasible schedule found")

    plan = _build_plan(hours, battery, m, flows)
    total_grid = sum(p["grid_kwh"] for p in plan)
    total_cost = sum(p["grid_kwh"] * float(hours[p["hour"]]["tariff_bdt_per_kwh"]) for p in plan)
    return {
        "hourly_plan": plan,
        "total_grid_kwh": round(total_grid, ROUND),
        "total_cost_bdt": round(total_cost, ROUND),
        "peak_grid_kwh": max(p["grid_kwh"] for p in plan),
        "relaxed": relaxed,
    }
