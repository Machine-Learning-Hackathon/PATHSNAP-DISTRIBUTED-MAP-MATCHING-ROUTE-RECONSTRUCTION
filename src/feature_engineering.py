
from pathlib import Path
import math
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
REPORT_DIR = PROJECT_DIR / "reports"
FEATURE_DIR = REPORT_DIR / "features"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)

EARTH_RADIUS = 6371000


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(
        math.radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * EARTH_RADIUS * math.asin(math.sqrt(a))


def calculate_heading(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(
        math.radians, [lat1, lon1, lat2, lon2]
    )

    dlon = lon2 - lon1

    x = math.sin(dlon) * math.cos(lat2)
    y = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    )

    return (math.degrees(math.atan2(x, y)) + 360) % 360


def calculate_turn_angle(previous, current):
    difference = abs(current - previous)
    return min(difference, 360 - difference)


def identify_split(track_file):
    parts = [part.lower() for part in track_file.parts]

    if "student_test" in parts or "test" in parts:
        return "student_test"

    if any("train" in part for part in parts):
        return "training"

    return "other"


def process_trip(track_file):
    df = pd.read_csv(
        track_file,
        sep=r"\s+",
        header=None,
        names=["latitude", "longitude", "sequence"]
    )

    for column in ["latitude", "longitude", "sequence"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    original_count = len(df)
    df = df.dropna().copy()

    df = df[
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    ].copy()

    df = df.sort_values("sequence").reset_index(drop=True)

    if len(df) < 2:
        return None, f"Only {len(df)} valid GPS points"

    distances = [0.0]
    headings = [float("nan")]
    turn_angles = [float("nan")]

    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i - 1, ["latitude", "longitude"]]
        lat2, lon2 = df.loc[i, ["latitude", "longitude"]]

        distance = haversine(lat1, lon1, lat2, lon2)
        heading = calculate_heading(lat1, lon1, lat2, lon2)

        distances.append(distance)
        headings.append(heading)

        angle = (
            calculate_turn_angle(headings[i - 1], heading)
            if i >= 2 else float("nan")
        )

        turn_angles.append(angle)

    df["distance_m"] = distances
    df["cumulative_distance_m"] = df["distance_m"].cumsum()
    df["heading_deg"] = headings
    df["turn_angle_deg"] = turn_angles

    start = df.iloc[0]
    end = df.iloc[-1]

    straight_distance = haversine(
        start["latitude"], start["longitude"],
        end["latitude"], end["longitude"]
    )

    total_distance = df["distance_m"].sum()

    summary = {
        "trip_id": track_file.stem,
        "source_file": str(track_file.relative_to(DATASET_DIR)),
        "original_points": original_count,
        "valid_points": len(df),
        "total_distance_m": total_distance,
        "straight_distance_m": straight_distance,
        "average_step_distance_m": df["distance_m"].iloc[1:].mean(),
        "maximum_step_distance_m": df["distance_m"].iloc[1:].max(),
        "average_turn_angle_deg": df["turn_angle_deg"].mean(),
        "path_efficiency": (
            straight_distance / total_distance
            if total_distance > 0 else 0
        )
    }

    return df, summary


def main():
    track_files = sorted(DATASET_DIR.rglob("*.track"))

    summaries = {
        "training": [],
        "student_test": [],
        "other": []
    }

    skipped = []

    print(f"Found {len(track_files)} track files.\n")

    for track_file in track_files:
        split = identify_split(track_file)

        features, result = process_trip(track_file)

        if features is None:
            skipped.append({
                "trip_id": track_file.stem,
                "source_file": str(track_file.relative_to(DATASET_DIR)),
                "split": split,
                "reason": result
            })
            print(f"Skipped {track_file.stem}: {result}")
            continue

        summaries[split].append(result)

        output_folder = FEATURE_DIR / split
        output_folder.mkdir(parents=True, exist_ok=True)

        output_file = output_folder / f"features_{track_file.stem}.csv"
        features.to_csv(output_file, index=False)

        print(
            f"{split} | {track_file.stem} | "
            f"{len(features)} points | "
            f"{result['total_distance_m'] / 1000:.2f} km"
        )

    for split, records in summaries.items():
        if records:
            pd.DataFrame(records).to_csv(
                REPORT_DIR / f"{split}_feature_summary.csv",
                index=False
            )

    all_records = [
        {**record, "split": split}
        for split, records in summaries.items()
        for record in records
    ]

    if all_records:
        pd.DataFrame(all_records).to_csv(
            REPORT_DIR / "all_feature_summary.csv",
            index=False
        )

    pd.DataFrame(skipped).to_csv(
        REPORT_DIR / "skipped_trips.csv",
        index=False
    )

    print("\nFEATURE ENGINEERING FINISHED")
    for split, records in summaries.items():
        print(f"{split}: {len(records)} trips processed")

    print(f"Skipped trips: {len(skipped)}")
    print(f"Reports saved in: {REPORT_DIR}")


if __name__ == "__main__":
    main()
