
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_DIR / "reports"
FEATURE_DIR = REPORT_DIR / "features"

summary_files = [
    REPORT_DIR / "training_feature_summary.csv",
    REPORT_DIR / "student_test_feature_summary.csv"
]

summaries = []

for file in summary_files:
    if file.exists():
        df = pd.read_csv(file)
        df["dataset"] = file.stem.replace(
            "_feature_summary", ""
        )
        summaries.append(df)

if not summaries:
    raise FileNotFoundError("Feature summary files not found.")

data = pd.concat(summaries, ignore_index=True)

print("\n========== FEATURE VALIDATION ==========")

print("\n1. Dataset sizes")
print(data.groupby("dataset")["trip_id"].count())

print("\n2. Missing values")
print(data.isnull().sum())

print("\n3. Distance validation")
print(data["total_distance_m"].describe())

print("\n4. Path efficiency validation")

invalid_efficiency = data[
    (data["path_efficiency"] < 0)
    | (data["path_efficiency"] > 1.000001)
]

print("Invalid path efficiency values:", len(invalid_efficiency))

print("\n5. GPS point validation")

print(
    "Trips with fewer than 2 points:",
    (data["valid_points"] < 2).sum()
)

print("\n6. Extremely large step distances")

large_steps = data[
    data["maximum_step_distance_m"] > 1000
]

print("Trips with steps above 1 km:", len(large_steps))

print("\n7. Training/test column comparison")

train = summaries[0]
test = summaries[1]

train_columns = set(train.columns)
test_columns = set(test.columns)

print("Common columns:", sorted(train_columns & test_columns))
print("Training-only columns:", sorted(train_columns - test_columns))
print("Test-only columns:", sorted(test_columns - train_columns))

# Save validation findings
issues = []

for _, row in data.iterrows():
    reasons = []

    if pd.isna(row["path_efficiency"]):
        reasons.append("Missing path efficiency")
    elif not 0 <= row["path_efficiency"] <= 1.000001:
        reasons.append("Invalid path efficiency")

    if row["maximum_step_distance_m"] > 1000:
        reasons.append("Step distance above 1 km")

    if row["valid_points"] < 2:
        reasons.append("Insufficient GPS points")

    if reasons:
        issues.append({
            "trip_id": row["trip_id"],
            "dataset": row["dataset"],
            "issues": "; ".join(reasons)
        })

pd.DataFrame(issues).to_csv(
    REPORT_DIR / "feature_validation_issues.csv",
    index=False
)

print("\nValidation report saved:")
print(REPORT_DIR / "feature_validation_issues.csv")
