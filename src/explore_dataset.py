
from pathlib import Path
import pandas as pd
import numpy as np

# Change this if your extracted folder has a different location
DATASET_ROOT = Path("../dataset/Student folder")

def load_trip(trip_folder):
    """Load all four files belonging to one trip."""

    trip_id = trip_folder.name

    track = pd.read_csv(
        trip_folder / f"{trip_id}.track",
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "timestamp"]
    )

    nodes = pd.read_csv(
        trip_folder / f"{trip_id}.nodes",
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude"]
    )

    arcs = pd.read_csv(
        trip_folder / f"{trip_id}.arcs",
        sep=r"\s+",
        header=None,
        names=["from_node_id", "to_node_id"]
    )

    route = pd.read_csv(
        trip_folder / f"{trip_id}.route",
        sep=r"\s+",
        header=None,
        names=["arc_id"]
    )

    return track, nodes, arcs, route


def explore_partition(partition):
    partition_path = DATASET_ROOT / partition

    print(f"\n{'=' * 50}")
    print(f"EXPLORING: {partition.upper()}")
    print(f"{'=' * 50}")

    trip_folders = sorted(
        [p for p in partition_path.iterdir() if p.is_dir()]
    )

    print("Number of trips:", len(trip_folders))

    results = []

    for folder in trip_folders:
        try:
            track, nodes, arcs, route = load_trip(folder)

            results.append({
                "trip_id": folder.name,
                "gps_observations": len(track),
                "road_nodes": len(nodes),
                "road_arcs": len(arcs),
                "route_length": len(route),
                "missing_gps": track.isnull().any(axis=1).sum(),
                "duplicate_gps": track.duplicated().sum(),
                "invalid_coordinates": (
                    ~track["longitude"].between(-180, 180)
                    | ~track["latitude"].between(-90, 90)
                ).sum(),
                "timestamps_ordered": (
                    track["timestamp"].diff().dropna() >= 0
                ).all()
            })

        except Exception as e:
            print(f"Error in trip {folder.name}: {e}")

    summary = pd.DataFrame(results)

    print("\nTRIP SUMMARY:")
    print(summary.to_string(index=False))

    print("\nSTATISTICAL SUMMARY:")
    print(summary.describe())

    return summary


if __name__ == "__main__":

    train_summary = explore_partition("train")

    test_summary = explore_partition("student_test")

    train_summary.to_csv(
        "../reports/train_summary.csv",
        index=False
    )

    test_summary.to_csv(
        "../reports/student_test_summary.csv",
        index=False
    )

    print("\nDataset exploration completed!")
