# PATHSNAP model component handoff

This component exports candidate scores to the Person 3 HMM/Viterbi backend. The backend uses the candidate scorer together with directed graph connectivity to produce an ordered route. See `docs/PERSON_3_IMPLEMENTATION.md` for the API and route-decoder handoff.

## Retrain the Person 2 model

From the repository root with the model dependencies installed:

```powershell
python -m pip install -r requirements.txt/requirements.txt
python src/train_models.py
```

The pipeline reads only `dataset/Student folder/train`. It reserves 25% of complete trip IDs for a held-out validation set, compares geometric nearest-arc baseline, Logistic Regression, Extra Trees, and HistGradientBoosting, then runs seeded Optuna TPE tuning with grouped 3-fold validation over training trips. Optimization trials are written to `reports/optimization_trials.csv`.

## Person 3 API

```python
from pathlib import Path
from src.model_predictor import CandidateScorer, ordered_route_coordinates

scorer = CandidateScorer(Path("models/best_model.joblib"))
scores = scorer.score_candidates(records)
# records: one candidate per (observation_id, candidate_id), with the seven
# feature columns in models/feature_schema.json
# output columns: observation_id (if supplied), candidate_id, score

coords = ordered_route_coordinates(nodes_lonlat, arcs, decoded_arc_ids)
# coords is [[latitude, longitude], ...], in decoded route arc order.
# Use it for the frontend's `predicted_route` field.
```

`candidate_id` is the row index into that trip's `.arcs` file (the same zero-based arc ID used by `.route`). Candidate scorer outputs retain IDs. The frontend response contract expects `[latitude, longitude]` coordinate pairs. Person 3's decoder must generate ordered arc IDs; the scorer does not itself produce a final route.

## Limits

Labels are built with minimum-cost monotone alignment from GPS observations to ordered route-segment geometry. This is a geometric proxy, not independently verified observation-level ground truth, and may be wrong on loops, parallel roads, uneven sampling, or sparse route segments. Candidate metrics must not be presented as end-to-end route accuracy. The ten `student_test` trips are never read by training, model selection, or tuning.
