
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASET_ROOT = Path("../dataset/Student folder")

def haversine(lon1, lat1, lon2, lat2):
    """Calculate distance between GPS points in metres."""

    R = 6371000

    lon1, lat1, lon2, lat2 = map(
        np.radians, [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * R * np.arcsin(np.sqrt(a))


def analyse_trip(trip_folder):

    trip_id = trip_folder.name

    track = pd.read_csv(
        trip_folder / f"{trip_id}.track",
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "timestamp"]
    )

    dt = track["timestamp"].diff()

    distance = haversine(
        track["longitude"].shift(),
        track["latitude"].shift(),
        track["longitude"],
        track["latitude"]
    )

    speed = distance / dt

    return {
        "trip_id": trip_id,
        "mean_interval": dt.mean(),
        "max_interval": dt.max(),
        "gaps_over_5s": (dt > 5).sum(),
        "mean_speed_mps": speed.mean(),
        "max_speed_mps": speed.max(),
        "gps_points": len(track)
    }


if __name__ == "__main__":

    train_path = DATASET_ROOT / "train"

    results = []

    for folder in sorted(train_path.iterdir()):

        if folder.is_dir():
            results.append(analyse_trip(folder))

    df = pd.DataFrame(results)

    print("\nGPS QUALITY ANALYSIS")
    print(df.describe().round(2))

    print("\nTrips with largest timestamp gaps:")
    print(
        df.sort_values(
            "max_interval", ascending=False
        ).head(10).to_string(index=False)
    )

    df.to_csv(
        "../reports/gps_quality_summary.csv",
        index=False
    )

    Path("../reports").mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))

    plt.hist(df["gps_points"], bins=15)

    plt.xlabel("Number of GPS observations")
    plt.ylabel("Number of trips")
    plt.title("Distribution of GPS trajectory lengths")

    plt.savefig("../reports/gps_distribution.png")
    
