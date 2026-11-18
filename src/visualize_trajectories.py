
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset" / "Student folder" / "student_test"
REPORT_DIR = PROJECT_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Number of trips to visualize
MAX_TRIPS = 10

def load_track(track_file):
    """Load GPS latitude, longitude and sequence number."""
    df = pd.read_csv(
        track_file,
        sep=r"\s+",
        header=None,
        names=["latitude", "longitude", "sequence"]
    )

    return df.dropna()


def visualize_trip(trip_folder):
    """Plot one GPS trajectory."""
    track_file = trip_folder / f"{trip_folder.name}.track"

    if not track_file.exists():
        return None

    df = load_track(track_file)

    if df.empty:
        return None

    plt.figure(figsize=(9, 7))

    plt.plot(
        df["longitude"],
        df["latitude"],
        linewidth=1,
        label="GPS trajectory"
    )

    plt.scatter(
        df["longitude"].iloc[0],
        df["latitude"].iloc[0],
        color="green",
        label="Start",
        zorder=3
    )

    plt.scatter(
        df["longitude"].iloc[-1],
        df["latitude"].iloc[-1],
        color="red",
        label="End",
        zorder=3
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"GPS Trajectory - Trip {trip_folder.name}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_file = REPORT_DIR / f"trajectory_{trip_folder.name}.png"

    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    return len(df)


def main():
    trip_folders = sorted(
        folder for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    )

    if not trip_folders:
        print("No trip folders found.")
        return

    total = 0

    for trip_folder in trip_folders[:MAX_TRIPS]:
        points = visualize_trip(trip_folder)

        if points is not None:
            print(f"Trip {trip_folder.name}: {points} GPS points visualized")
            total += 1

    print(f"\nVisualization completed.")
    print(f"Trips visualized: {total}")
    print(f"Output directory: {REPORT_DIR}")


if __name__ == "__main__":
    main()
