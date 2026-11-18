# PathSnap backend — Person 3

## HMM + Viterbi decoder

The decoder treats each directed road arc as a hidden HMM state. For each GPS fix it generates nearby arc candidates, computes a Gaussian GPS-to-road emission likelihood (plus a heading consistency term), and scores transitions using the difference between observed GPS displacement and directed shortest-path distance in the road graph. Viterbi runs in log space and returns an ordered list of arc row IDs and node IDs.

### Dataset schema used

- `.track`: longitude, latitude, timestamp
- `.nodes`: longitude, latitude; row index is node ID
- `.arcs`: source node ID, target node ID; row index is arc ID
- `.route`: ground truth arc IDs; **not read by prediction endpoints**

### Run

From repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

Endpoints:

- `GET /health`
- `GET /api/evaluation/trips`
- `POST /api/evaluation/predict` with `{"trip_id":"00000000"}`

By default the API lists only `dataset/Student folder/student_test`. Set `PATHSNAP_EVALUATION_SPLIT` only according to the official evaluation protocol. The endpoint does not use `.route` labels. Set `PATHSNAP_DATA_DIR` to override the dataset parent folder.

### Artifacts

```bash
python -m backend.export_artifacts
```

This exports `models/hmm_config.json` and `models/manifest.json`. The HMM decoder is deterministic and configuration-driven, so these are inference configuration artifacts, not learned model weights/checkpoints. Do not claim the configuration is a trained model.

### Frontend

The `frontend/` directory in this patch is copied from the separate UI repo and retains the dark black/ivory interface. Configure Vercel's `VITE_API_BASE_URL` to the public URL of this FastAPI service; `localhost` only works for local development.

## Vercel deployment note

The current deployable UI is a separate Vite app in `frontend/`. The `VITE_API_BASE_URL` environment variable must point to a deployed FastAPI service. Deploying the static frontend does not deploy the Python API; host the API separately (for example, on a Python-capable service) and set CORS accordingly.
