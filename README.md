# PATHSNAP-DISTRIBUTED-MAP-MATCHING-ROUTE-RECONSTRUCTION
## Backend and route reconstruction

The backend uses the Person 2 trained Extra Trees candidate scorer with directed road-network candidates and an HMM/Viterbi decoder. The API implements the existing evaluator frontend contract. See [docs/PERSON_3_IMPLEMENTATION.md](docs/PERSON_3_IMPLEMENTATION.md) for the verified data schemas, model adapter, API examples, configuration options, evaluation results, and Person 4 handoff.

Laptop 1/Laptop 2 LAN setup, CORS, firewall notes, and the cross-device test procedure are documented in [docs/TWO_LAPTOP_DEPLOYMENT.md](docs/TWO_LAPTOP_DEPLOYMENT.md).

### Install and run (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt\requirements.txt
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

The frontend defaults to `http://localhost:8000`. For a backend on another machine, set `VITE_API_BASE_URL` in `frontend/.env` to that machine's reachable address.

### Tests and evaluation

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe src\evaluate_routes.py
```

Evaluation predicts first, then reads ground-truth route files for comparison. It writes trip-level results under `reports/route_evaluation.csv` and aggregate results to `reports/route_evaluation_summary.json`.
