# GridWise — LLM-Assisted 24-Hour Energy Optimization API

BUP CSE Fest 2026 Hackathon — Online Preliminary. Team: Git Push & Pray.

One HTTP service that reads a 24-hour campus energy scenario plus 1–3 natural-language
operator notes, turns the notes into structured directives with an LLM, validates them with
deterministic guardrails, and returns a minimum-cost valid 24-hour battery/grid schedule.

| | |
|---|---|
| Health endpoint | `GET /health` → `{"status":"ok"}` |
| Main endpoint | `POST /optimize-energy` |
| Port | `8000` (override with env `PORT`) |
| Public URL | https://gridwise-e43e.onrender.com (Render, Singapore) |
| Docker image | `docker.io/dockersakib/gridwise:v1.0.0` (digest `sha256:6b3982acf5bb7a51030c51b4d890caf1d2d0868165a7a789f88616b808e6d707`) |

## Architecture

```
operator_notes ──► LLM interpreter ──► guardrails ──► MILP optimizer ──► response
                   app/interpreter.py  app/guardrails.py  app/optimizer.py   app/main.py
```

1. **Request validation** (`app/schemas.py`, Pydantic): exactly 24 unique hours 0–23, 1–3
   non-empty notes, finite non-negative numbers, `minimum ≤ initial ≤ capacity`.
   Structural errors → HTTP 400 (field locations only, never echoed values or stack traces).
2. **LLM interpretation** (`app/interpreter.py`): one entry per note —
   `solar_reduction`, `minimum_battery_reserve`, `no_charge_window`, `no_discharge_window`,
   `max_grid_window`, or `no_op`. Model/provider: Groq `qwen/qwen3.8-27b` with fallbacks (see below).
   Called with a 22 s timeout; timeouts and provider errors degrade to `no_op` instead of failing.
3. **Guardrails** (`app/guardrails.py`): LLM output is untrusted. Exactly one entry per note in
   `note_index` order; only supported types; hours unique ints 0–23, ascending; `factor ∈ [0,1]`;
   reserve finite, `≥ 0`, `≤ capacity`; `max_grid_kwh` finite, `≥ 0`; `applies` is `false` only for
   `no_op` (with `null` adjustment). Any entry failing a check becomes a safe `no_op`.
4. **Optimizer** (`app/optimizer.py`, PuLP + CBC): directives of the same type are merged
   (solar factors per hour, union of no-charge / no-discharge hours, max of reserve floors,
   min of grid caps). Then a MILP minimizes `Σ grid_kwh[h] · tariff[h]` subject to:
   energy balance, `solar_used ≤ effective solar`, battery bounds with reserve floors, rate limits
   (0 in no-charge / no-discharge windows), grid caps, state transitions, and
   **end-of-day neutrality** (`E[23] = initial_energy_kwh`). One binary per hour forbids
   charging and discharging in the same hour. A 1e-6 cost on battery throughput breaks ties
   against pointless cycling.
5. **Plan assembly**: battery energy is replayed from the rounded battery amounts, and `grid_kwh` is
   derived from the energy-balance equation, so every hour is exactly self-consistent.
   `total_grid_kwh`, `total_cost_bdt`, and `peak_grid_kwh` are computed from the final `hourly_plan`.
6. **Safety net**: if the interpreted directives are jointly infeasible (should never happen for
   valid judge scenarios), a fallback model relaxes reserve floors / grid caps with a heavy penalty,
   so the service still answers with 200 and states this in `plan_summary`.

Solve time is about 30 ms per scenario. On all 10 public samples, the cost with the ground-truth
directives equals the organizer's reference cost.

## Local quickstart (clean machine)

Requires Python 3.11+ and git.

```bash
git clone https://github.com/SadmanInAction/BUP_Hackathon_Git_Push_Pray.git
cd BUP_Hackathon_Git_Push_Pray
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then set GROQ_API_KEY (see Environment variables)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

PuLP ships the CBC solver binary, so no separate solver install is needed.

## Environment variables

| Name | Required | Purpose |
|---|---|---|
| `PORT` | no (default `8000`) | Port the server binds to (Docker image) |
| `GROQ_API_KEY` | yes | Groq API key (free at https://console.groq.com, no card) |
| `GROQ_MODELS` | no | Comma-separated model fallback chain (default `qwen/qwen3.8-27b,openai/gpt-oss-120b,openai/gpt-oss-20b`) |
| `GEMINI_API_KEY` | no | Optional extra fallback provider (free key at https://aistudio.google.com), used only if every Groq model fails |
| `GRIDWISE_LLM_PROVIDER` | no | Provider order, default `groq,gemini` (`ollama` is also supported for local experiments, but CPU inference is too slow for the 30 s judge limit) |

Never commit `.env`; it is in `.gitignore` and `.dockerignore`.

## Model / provider

- **Provider:** [Groq](https://groq.com) (OpenAI-compatible chat completions API).
- **Primary model:** `qwen/qwen3.8-27b` (`reasoning_effort=none`), falling back to `openai/gpt-oss-120b` and then `openai/gpt-oss-20b` (`reasoning_effort=low`).
  Each Groq model has its own free-tier rate limit, so on a 429 or error the next model is tried at once.
  If all three are rate-limited, the service waits once for the shortest Retry-After (8 s maximum).
- **Why:** free, no card needed, median latency about 0.4 s (p95 about 1 s, well under the 5 s target).
  Each of the three models, alone, answered all public samples and all 55 paraphrase / held-out notes in our tests correctly.
- **LLM role:** interprets every operator note in a single call (JSON mode, temperature 0, few-shot prompt).
  The model returns the full directive entry (type, `hours`, `factor` / `minimum_energy_kwh` / `max_grid_kwh`,
  explanation), plus the window boundaries `start_hour`/`end_hour` exactly as stated in the note.
- **Interpreter guardrails** (`app/interpreter.py`, before `app/guardrails.py` runs again in `main.py`):
  `hours` are rebuilt from `start_hour`/`end_hour` (end exclusive, wrapping past midnight), which removes
  the most common model error (end-inclusive windows); unsupported types, non-numeric or out-of-range values,
  and missing/duplicate/out-of-range `note_index` values become `no_op` entries flagged with `[guardrail]`;
  a percentage `factor` such as `25` is normalised to `0.25`; `applies` is always derived from the type.
  Unparseable output is retried once. `interpret_notes` never raises.
- **Caching:** identical note sets reuse the previous interpretation.
- **Failure mode:** if the provider fails or times out, notes become `no_op`; the service still returns a valid schedule and never crashes.

## API examples

```bash
curl -s http://localhost:8000/health
# {"status":"ok"}

curl -s -X POST http://localhost:8000/optimize-energy \
  -H "Content-Type: application/json" \
  --data @tests/sample_request.json
```

Against the live deployment, replace `http://localhost:8000` with `https://gridwise-e43e.onrender.com`.

`tests/sample_request.json` is the input of public sample `SAMPLE-01`. Response shape:

```json
{
  "scenario_id": "SAMPLE-01",
  "directive_interpretation": [
    {"note_index": 0, "applies": true, "directive_type": "solar_reduction",
     "structured_adjustment": {"hours": [12, 13], "factor": 0.25}, "explanation": "..."},
    {"note_index": 1, "applies": false, "directive_type": "no_op",
     "structured_adjustment": null, "explanation": "..."}
  ],
  "hourly_plan": [
    {"hour": 0, "grid_kwh": 90.0, "solar_used_kwh": 0.0, "battery_action": "idle",
     "battery_kwh": 0.0, "battery_energy_after_kwh": 110.0}
  ],
  "total_grid_kwh": 2692.5, "total_cost_bdt": 38365.0, "peak_grid_kwh": 175.0,
  "plan_summary": "..."
}
```
(`hourly_plan` is truncated here; the real response always has 24 entries.)

Status codes: `200` success · `400` malformed JSON or structurally invalid request ·
`500` controlled internal error (`{"error":"internal_error"}`, no stack trace).

## Public-sample tests

```bash
python -m pytest -v
```

`tests/test_paraphrases.py` checks 15 paraphrased notes (including the Problem Statement's own examples).
`tests/test_interpreter.py` checks the interpreter on its own: all 10 public samples, 20 more paraphrases of
every directive type (number words, 24 h / AM-PM / no-AM-PM times, % of capacity, tricky distractors), and
43 offline tests that feed deliberately malformed model output (invalid JSON, `NaN`, invented types,
hour 25, duplicate indices, provider exceptions, rate limits) and check it always fails safe.
The LLM-dependent tests are skipped when no provider key is set (`python -m pytest -m "not live"` runs only the offline ones;
set `LIVE_TEST_DELAY=2` to pace live tests under the free-tier rate limit).

`tests/test_public_cases.py` runs all 10 cases in `tests/sample_cases.json`. For each case it
replays the plan the way the judge does: 24 unique hours; energy balance, effective solar,
battery bounds, rate limits, transitions and directive windows checked every hour; end-of-day
neutrality; totals recomputed from `hourly_plan` (tolerance 0.01). It also checks that cost is no
higher than the reference, that interpretation matches the ground truth, and that request
validation (400s), guardrails, and provider-failure handling work.
Expected result: all tests pass.

## Docker fallback

```bash
docker pull docker.io/dockersakib/gridwise:v1.0.0
docker run --rm -p 8000:8000 -e GROQ_API_KEY=<your-groq-key> docker.io/dockersakib/gridwise:v1.0.0
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/optimize-energy -H "Content-Type: application/json" --data @tests/sample_request.json
```

The image binds `0.0.0.0:8000`, runs as a non-root user, and contains no secrets. Credentials are
passed only at runtime (`-e` / `--env-file`). Pin by digest if preferred:
`docker.io/dockersakib/gridwise@sha256:6b3982acf5bb7a51030c51b4d890caf1d2d0868165a7a789f88616b808e6d707`. Build locally with `docker build -t gridwise .`.
Run the tests inside the container with `docker run --rm gridwise python -m pytest -q`.

## Dependencies / credits

- [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/): HTTP API
- [Pydantic](https://docs.pydantic.dev/): request validation
- [PuLP](https://coin-or.github.io/pulp/) with the bundled [COIN-OR CBC](https://github.com/coin-or/Cbc) solver: MILP optimization
- pytest, httpx: tests
- Groq API (called with Python's standard `urllib`), optional Google Gemini API, python-dotenv
- Hosting: [Render](https://render.com) free web service (runs the Docker Hub image)
- AI coding assistant (Claude Code) used during development, as permitted by the rulebook.

## Known limitations

- Interpretation quality depends on the hosted LLM being reachable. If it is unavailable or times
  out, notes fall back to `no_op`: the service stays up, but those directives are not applied.
- Directives that are infeasible together are relaxed with a penalty rather than rejected.
- Output values are rounded to 4 decimals (the judge tolerance is 0.01).
- Hosted on the Render free tier, which sleeps after 15 min idle; an external pinger hits /health every 5 min to keep it warm. A cold start takes about 1 min.
- Groq free tier: about 1,000 requests per day per model; the three-model chain and caching spread the load.

## Secret handling

No keys in the repo, the image, logs or responses. Logs record only scenario id, directive types,
cost and latency; errors log only the exception type.
