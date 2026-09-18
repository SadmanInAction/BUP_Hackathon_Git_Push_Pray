"""GridWise API: GET /health and POST /optimize-energy."""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.guardrails import no_op, validate_interpretation
from app.interpreter import interpret_notes
from app.optimizer import optimize_schedule
from app.schemas import HealthResponse, OptimizeRequest, OptimizeResponse

# The judge times out at 30 s; leave room for the optimizer (~50 ms) and network.
INTERPRETER_TIMEOUT_S = 22

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("gridwise")

app = FastAPI(title="GridWise LLM-Assisted Energy Optimizer", version="1.0.0")
_interpreter_pool = ThreadPoolExecutor(max_workers=8)


@app.exception_handler(RequestValidationError)
async def _bad_request(request: Request, exc: RequestValidationError):
    # Only field locations and messages - never echo input values back.
    errors = [{"loc": [str(p) for p in e.get("loc", [])], "msg": e.get("msg", "")}
              for e in exc.errors()]
    log.info("400 %s %s: %d validation error(s)", request.method, request.url.path, len(errors))
    return JSONResponse(status_code=400, content={"error": "invalid_request", "details": errors})


@app.exception_handler(Exception)
async def _internal_error(request: Request, exc: Exception):
    # Log only the exception type: messages may contain provider responses or config.
    log.error("500 %s %s: %s", request.method, request.url.path, type(exc).__name__)
    return JSONResponse(status_code=500, content={"error": "internal_error"})


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


def _interpret(req: OptimizeRequest, hours: list[dict], battery: dict) -> list[dict]:
    """Call the LLM interpreter with a timeout; any failure degrades to no_op."""
    n = len(req.operator_notes)
    try:
        future = _interpreter_pool.submit(interpret_notes, list(req.operator_notes), hours, battery)
        raw = future.result(timeout=INTERPRETER_TIMEOUT_S)
    except FutureTimeout:
        log.warning("scenario=%s interpreter timed out; using no_op fallback", req.scenario_id)
        raw = [no_op(i, "Interpreter timed out; note treated as no_op.") for i in range(n)]
    except Exception as exc:
        log.warning("scenario=%s interpreter failed (%s); using no_op fallback",
                    req.scenario_id, type(exc).__name__)
        raw = [no_op(i, "Interpreter unavailable; note treated as no_op.") for i in range(n)]
    return validate_interpretation(raw, n, battery)


def _summary(directives: list[dict], result: dict) -> str:
    applied = [d["directive_type"] for d in directives if d["applies"]]
    ignored = sum(1 for d in directives if not d["applies"])
    parts = []
    if applied:
        parts.append("Applied " + ", ".join(applied))
    if ignored:
        parts.append(f"ignored {ignored} unrelated note(s)")
    text = "; ".join(parts) + ". " if parts else ""
    text += ("Cost-minimising schedule: the battery shifts energy from cheaper to "
             "higher-tariff hours, solar is used first, and the battery ends the day "
             f"at its initial level. Total grid {result['total_grid_kwh']:g} kWh, "
             f"cost {result['total_cost_bdt']:g} BDT, peak {result['peak_grid_kwh']:g} kWh.")
    if result.get("relaxed"):
        text += " Some interpreted directives were infeasible together and were relaxed minimally."
    return text[0].upper() + text[1:]


@app.post("/optimize-energy", response_model=OptimizeResponse)
def optimize_energy(req: OptimizeRequest):
    # Handled here (not only by the global handler) so no traceback reaches the logs.
    try:
        return _optimize(req)
    except Exception as exc:
        log.error("500 /optimize-energy scenario=%s: %s", req.scenario_id, type(exc).__name__)
        return JSONResponse(status_code=500, content={"error": "internal_error"})


def _optimize(req: OptimizeRequest) -> dict:
    started = time.perf_counter()
    hours = [h.model_dump() for h in req.hours]  # already sorted 0..23 by the schema
    battery = req.battery.model_dump()

    directives = _interpret(req, hours, battery)
    result = optimize_schedule(hours, battery, directives)

    response = {
        "scenario_id": req.scenario_id,
        "directive_interpretation": directives,
        "hourly_plan": result["hourly_plan"],
        "total_grid_kwh": result["total_grid_kwh"],
        "total_cost_bdt": result["total_cost_bdt"],
        "peak_grid_kwh": result["peak_grid_kwh"],
        "plan_summary": _summary(directives, result),
    }
    log.info("200 /optimize-energy scenario=%s notes=%d directives=%s cost=%.2f relaxed=%s %.0fms",
             req.scenario_id, len(req.operator_notes),
             [d["directive_type"] for d in directives], result["total_cost_bdt"],
             result["relaxed"], (time.perf_counter() - started) * 1000)
    return response
