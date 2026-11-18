# Data and interface audit

## Raw format

A sample from each raw file and the existing exploration code confirms:

- `.track`: longitude, latitude, timestamp/sequence (three numeric columns).
- `.nodes`: longitude, latitude (two numeric columns).
- `.arcs`: directed from-node index, to-node index; indices point into the `.nodes` rows.
- `.route`: ordered arc row indices; indices point into `.arcs` rows.

All 80 training trips were parsed by the candidate pipeline; their route and arc indices were within range. Existing raw-data summaries report no missing or invalid GPS coordinates, no duplicate observations, and ordered sequence values for the audited trips.

## Existing feature script issue

`src/explore_dataset.py` reads track columns as longitude, latitude, timestamp. `src/feature_engineering.py` reads the same raw file as latitude, longitude, sequence. This is a genuine longitude/latitude reversal, not just a different header. That script was left unchanged to preserve Person 1's work; the model pipeline independently reads raw track files in the verified order.

## Label integrity

The route file has one arc sequence per trip rather than one label per GPS row. Labels are therefore formed by minimum-cost monotone alignment from each GPS point to the ordered route segment geometries. This allows multiple GPS observations per arc and skips route arcs as needed; it does not assume one-to-one correspondence.

Alignment distance is used as a quality check and never as a model input. Three of 80 trips failed the data-derived quality rule (per-trip median alignment distance at most 50 m and 95th percentile at most 100 m) and were excluded from supervised training. The per-trip results are in `label_alignment_audit.csv`.

## Validation and frontend contract

Model comparison uses a seeded 75/25 split by complete trip ID. Optuna uses three-fold `GroupKFold` over the training trips only. Student evaluation trips are not accessed by the training scripts.

`frontend/src/lib/api.ts` defines `predicted_route` as an ordered array of `[latitude, longitude]` pairs. The model preserves candidate arc IDs for Person 3's decoder; `ordered_route_coordinates(...)` converts the decoded ordered arc IDs to that coordinate order. This component does not implement the API server or decoder.
