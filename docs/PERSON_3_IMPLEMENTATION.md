# Person 3: backend and route reconstruction handoff

## Architecture

The service flow is `track rows → directed road graph → spatial candidates → Person 2 candidate probabilities → HMM emissions/transitions → log-space Viterbi → connected arc sequence → [latitude, longitude] route coordinates`. The same `RouteReconstructionService.reconstruct(trip_id)` is used by FastAPI and offline evaluation. Prediction code never opens `.route` files.

## Verified source data

- `.track`: longitude, latitude, sequence/time; rows are retained in file order.
- `.nodes`: longitude, latitude; node ID is the zero-based row number.
- `.arcs`: directed from-node and to-node indices; arc ID is the zero-based row number.
- `.route`: ordered zero-based arc IDs, read only by the post-prediction evaluation script.

The ten trip IDs are discovered from the `student_test` partition at runtime. No separate approved-ID manifest is present. The server sorts complete trips lexically and returns the first ten; the present dataset has ten, IDs `00000000`, `00000007`, `00000009`, `00000040`, `00000052`, `00000056`, `00000065`, `00000076`, `00000077`, and `00000091`. The team should confirm these are the designated evaluation IDs. Ground truth is not returned by the trip-list API.

## Road network and candidates

`src/road_network.py` keeps directed adjacency and the original row IDs, computes local metric segment lengths/bearings, uses a spatial KD-tree over arc midpoints, and provides bounded directed Dijkstra connectivity/path queries. Invalid arcs are logged and skipped without renumbering valid arc IDs.

`src/candidate_generation.py` projects points into a trip-local metric coordinate system, measures GPS-to-segment distance, and retains up to the configured closest candidates inside the primary radius. It calculates the same seven features saved in Person 2's schema: `distance_m`, heading difference sine/cosine, `length_m`, `step_distance_m`, and turn sine/cosine. If the primary radius has no result, only roads inside the configured fallback radius are emitted, with a warning. If that radius also has no road, prediction fails clearly.

## Person 2 model and emissions

`src/model_predictor.py` loads `models/best_model.joblib` and `models/feature_schema.json`. No scaler or preprocessing artifact exists or is required. The adapter checks the exact feature order, verifies positive class `1`, requires `predict_proba`, and returns scores with the original `candidate_id` and `observation_id`.

Person 2's positive-class probabilities are clipped by a small epsilon, normalized across the candidates for each observation, and converted to log emissions. If artifacts are missing or incompatible, production startup fails. A geometric distance/heading scorer is available only when `PATHSNAP_DEV_FALLBACK=true`; the health response identifies that mode.

## HMM, transition model, and Viterbi

The hidden state at each observation is one candidate directed arc. Emissions come from the trained scorer. A transition uses directed shortest path distance between the previous arc's end and next arc's start, plus the partial lengths on both candidate arcs. That route distance is compared with GPS displacement and elapsed time; impossible speed or disconnected transitions are rejected in the primary pass. When no path remains from any currently viable state, a second bounded search may allow a longer **connected directed graph path**, with a strong speed/distance penalty and an explicit warning. It never adds a nonexistent edge. If no directed path is found within the configured maximum, reconstruction fails rather than inventing a route.

`src/viterbi.py` applies the standard max-sum recurrence to log emissions and log transitions, stores predecessor indices, and backtracks the best complete sequence. Empty candidate sets, malformed matrices, and disconnected paths produce explicit failure results. Single-observation tracks are supported.

The final ordered route arc list includes directed network arcs used to connect selected candidate arcs. Route coordinates come from the source nodes as `[latitude, longitude]` arrays.

## API contract

Backend URL by default: `http://localhost:8000` for local development. In the two-laptop setup, Laptop 1 binds `0.0.0.0:8000` and Laptop 2 sets `VITE_API_BASE_URL` to Laptop 1's private address. See [TWO_LAPTOP_DEPLOYMENT.md](TWO_LAPTOP_DEPLOYMENT.md).

### `GET /health`

```json
{"status":"ok","model_name":"extra_trees","version":"1.0.0","scoring_mode":"trained_model"}
```

### `GET /api/evaluation/trips`

```json
{"trips":[{"trip_id":"00000007","label":"Evaluation trip 02"}]}
```

The IDs and labels are generated from available files. The example is illustrative.

### `POST /api/evaluation/predict`

Request:

```json
{"trip_id":"00000007"}
```

Response follows the frontend schema: `trip_id`, `gps_trace`, `predicted_route`, optional `road_network`, `inference_time_ms`, and `decoder`. Coordinates are `[latitude, longitude]`. Additional diagnostics include `status`, `scoring_mode`, matched and expanded arc IDs, matched coordinates, candidate count, log probability, warnings, and server-computed route comparison metrics. Ground truth is loaded only after prediction finishes and its raw route is not returned.

Unknown IDs return 404; invalid requests return 422; missing service resources return 503; a genuine no-path result returns 422. Internal file paths and exception details are not exposed by the API.

## Setup and run

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt\requirements.txt
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Set `frontend/.env` to the backend host when using a separate machine:

```text
VITE_API_BASE_URL=http://<backend-host>:8000
```

The existing UI calls these endpoints and displays service readiness, prediction errors, and post-prediction route metrics. For local development it defaults to `http://localhost:8000`.

Run tests and post-prediction evaluation:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe src\evaluate_routes.py
```

Evaluation writes trip-level metrics to `reports/route_evaluation.csv` and an aggregate to `reports/route_evaluation_summary.json`. It first predicts; only after a successful prediction does it open that trip's `.route` for comparison.

## Configuration

Environment variables (defaults shown):

- `PATHSNAP_DATASET_ROOT=./dataset`
- `PATHSNAP_MODEL_PATH=./models/best_model.joblib`
- `PATHSNAP_FEATURE_SCHEMA=./models/feature_schema.json`
- `PATHSNAP_HOST=0.0.0.0`, `PATHSNAP_PORT=8000`
- `PATHSNAP_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173`
- `PATHSNAP_CANDIDATE_RADIUS_M=120`, `PATHSNAP_FALLBACK_RADIUS_M=350`, `PATHSNAP_MAX_CANDIDATES=8`
- `PATHSNAP_MAX_SPEED_MPS=75`
- `PATHSNAP_DEV_FALLBACK=false`

Default HMM settings are also saved in `models/inference/hmm_config.json`. The current implementation keeps one trip's large network in an LRU cache at a time.

## Evaluation run and limitations

On the local ten-trip student evaluation set, the trained model completed **9/10 trips (90%)**. For those nine successful reconstructions, mean route-edge precision was **0.845**, recall **0.992**, F1 **0.907**, and sequence similarity **0.840**. Mean inference time was about **3.54 seconds per successful trip**. The run used 15 explicitly warned long-but-connected transition fallbacks across successful trips. These are actual results from `reports/route_evaluation.csv`; failed trips are excluded from the route-quality averages and remain in the CSV with their failure reason.

One trip (`00000000`) still has no complete path within the bounded directed graph search. The backend returns a clear 422 for that trip instead of returning an invented route. Long connected transitions are explicit and appear in warnings. The network represents each arc as a straight line between its two nodes; no intermediate road shape is provided. The model's labels are geometric proxies, and its candidate scores are not calibrated probabilities. These limits affect route quality and should be considered before treating the evaluation averages as general performance.

## Person 4 handoff

- Start the backend from the repository root with the Uvicorn command above.
- Backend URL is `http://<backend-host>:8000`; set the frontend's `VITE_API_BASE_URL` accordingly.
- The UI endpoints are `GET /health`, `GET /api/evaluation/trips`, and `POST /api/evaluation/predict` with `{ "trip_id": "..." }`.
- `predicted_route`, `gps_trace`, and each `road_network` polyline use `[latitude, longitude]` coordinate arrays.
- CORS permits the Vite origins on localhost by default; add the remote frontend origin to `PATHSNAP_CORS_ORIGINS` if needed.
- The UI displays readiness, prediction errors, and post-prediction route metrics. A failed reconstruction returns an HTTP error that the UI displays.

