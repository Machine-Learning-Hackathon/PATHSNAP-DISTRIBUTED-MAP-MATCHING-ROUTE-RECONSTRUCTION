# Model selection report

- The candidate-label alignment distance has median 4.9 m and 95th percentile 18.0 m over 176,017 positive labels in eligible trips.
- 80 supplied training trips were audited; 77 passed the alignment quality rule (per-trip median <= 50 m and 95th percentile <= 100 m). See label_alignment_audit.csv.
- Held-out validation trips: 20 (seed 42); groups have no overlap.
- Selected: **extra_trees**; held-out top-1: 0.8358; F1: 0.6891; balanced accuracy: 0.9102.
- Tuning change in validation top-1 for extra_trees: -0.0003; the original model is retained when that change is negative.
- Optuna TPE: 12 trials; best grouped-CV top-1 0.8104; tuned validation top-1 0.8355.
- Training is capped deterministically at at most 500 GPS observations per training trip. Validation retains all observations. Candidate rows from a trip never cross splits.
- Labels use minimum-cost monotone geometric alignment to ordered route segments. This remains a proxy and can be ambiguous at loops or repeated roads. These metrics are candidate selection metrics against proxy labels, not end-to-end route accuracy.
- The ten student evaluation trips are not read by the pipeline.

## Learning setup
Each example is a GPS observation paired with a candidate directed road arc. The binary target indicates whether that arc is selected by monotone geometric alignment to the ordered ground-truth route. Candidate features are perpendicular GPS-to-segment distance, movement heading relative to segment bearing (sine/cosine), segment length, GPS step distance, and turn angle (sine/cosine). Candidate IDs stay alongside scores for downstream decoding.

## Model results
The nearest-geometry baseline chooses the closest arc. Logistic Regression, Extra Trees, and HistGradientBoosting are trained on the same trip-level split. Top-k accuracy ranks candidates within each observation; row metrics classify each candidate.

| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 | Top-1 | Top-3 | Inference (ms) | Model size (bytes) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nearest_geometry_baseline | 0.9112 | 0.8970 | 0.5522 | 0.8789 | 0.6782 | 0.6224 | 0.9509 | 0.0 | 0 |
| logistic_regression | 0.8752 | 0.8940 | 0.4571 | 0.9180 | 0.6103 | 0.8145 | 0.9447 | 17.6 | 140 |
| extra_trees | 0.9128 | 0.9102 | 0.5556 | 0.9070 | 0.6891 | 0.8358 | 0.9630 | 553.3 | 91921824 |
| hist_gradient_boosting | 0.9506 | 0.8500 | 0.7955 | 0.7221 | 0.7570 | 0.8330 | 0.9658 | 408.9 | 1072 |
| tuned_extra_trees | 0.9058 | 0.9107 | 0.5334 | 0.9171 | 0.6745 | 0.8355 | 0.9631 | 622.6 | 71708184 |

Accuracy is the fraction of candidate rows classified correctly; precision and recall describe positive route-arc predictions; F1 balances precision and recall. Balanced accuracy averages sensitivity across the positive and negative classes, reducing the effect of the many negative candidates. Top-1 and top-3 measure whether the aligned route arc ranks first or appears within the three highest-scored candidates for each GPS observation. Inference time covers held-out candidate scoring; estimated model size is based on fitted estimator structures.

## Tuning and selection
Optuna TPE searched 12 trials over the model selected before tuning, with GroupKFold(3) on training trips. The objective was mean top-1 candidate accuracy. The tuned validation result is compared with the original model in the first section; the original is retained when tuning does not improve held-out ranking. Trial-level parameters and scores are in optimization_trials.csv.

## Use and limitations
Load models/best_model.joblib with `CandidateScorer` from src/model_predictor.py. Supply one row per candidate with the seven feature columns in models/feature_schema.json plus candidate_id; output contains candidate_id and score. The model produces candidate scores, not a final reconstructed route. Person 3's decoder should choose an ordered connected arc sequence and convert it to `[latitude, longitude]` coordinates using `ordered_route_coordinates(...)` for the frontend response. The geometric labels can still be ambiguous at loops and parallel roads; these results are not route-level evaluation.