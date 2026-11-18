
from pathlib import Path
from collections import Counter
import csv

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset" / "Student folder" / "student_test"
REPORT_DIR = PROJECT_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def read_nodes(file_path):
    """Read latitude and longitude of road-network nodes."""
    nodes = []

    with open(file_path, "r") as file:
        for line in file:
            values = line.split()

            if len(values) >= 2:
                try:
                    latitude = float(values[0])
                    longitude = float(values[1])
                    nodes.append((latitude, longitude))
                except ValueError:
                    continue

    return nodes


def read_arcs(file_path):
    """Read connections between node indices."""
    arcs = []

    with open(file_path, "r") as file:
        for line in file:
            values = line.split()

            if len(values) >= 2:
                try:
                    source = int(values[0])
                    destination = int(values[1])
                    arcs.append((source, destination))
                except ValueError:
                    continue

    return arcs


def analyze_network(nodes, arcs):
    """Calculate road-network connectivity statistics."""

    node_count = len(nodes)
    parent = list(range(node_count))
    rank = [0] * node_count

    in_degree = Counter()
    out_degree = Counter()

    valid_arcs = 0
    invalid_arcs = 0

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        root_a = find(a)
        root_b = find(b)

        if root_a == root_b:
            return

        if rank[root_a] < rank[root_b]:
            root_a, root_b = root_b, root_a

        parent[root_b] = root_a

        if rank[root_a] == rank[root_b]:
            rank[root_a] += 1

    for source, destination in arcs:

        # Check whether endpoints are valid node indices
        if not (
            0 <= source < node_count
            and 0 <= destination < node_count
        ):
            invalid_arcs += 1
            continue

        valid_arcs += 1

        out_degree[source] += 1
        in_degree[destination] += 1

        # Treat connections as undirected for
        # weakly connected component analysis
        union(source, destination)

    total_degree = {
        i: in_degree[i] + out_degree[i]
        for i in range(node_count)
    }

    component_sizes = Counter(
        find(i) for i in range(node_count)
    )

    components = len(component_sizes)

    largest_component = (
        max(component_sizes.values())
        if component_sizes else 0
    )

    isolated_nodes = sum(
        1 for degree in total_degree.values()
        if degree == 0
    )

    maximum_degree = (
        max(total_degree.values())
        if total_degree else 0
    )

    average_degree = (
        sum(total_degree.values()) / node_count
        if node_count else 0
    )

    return {
        "nodes": node_count,
        "total_arcs": len(arcs),
        "valid_arcs": valid_arcs,
        "invalid_arcs": invalid_arcs,
        "components": components,
        "largest_component": largest_component,
        "largest_component_percent": (
            largest_component / node_count * 100
            if node_count else 0
        ),
        "isolated_nodes": isolated_nodes,
        "average_degree": average_degree,
        "maximum_degree": maximum_degree,
    }


def main():

    results = []

    trip_folders = sorted(
        folder for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    )

    for folder in trip_folders:

        trip_id = folder.name

        nodes_file = folder / f"{trip_id}.nodes"
        arcs_file = folder / f"{trip_id}.arcs"

        if not nodes_file.exists() or not arcs_file.exists():
            continue

        nodes = read_nodes(nodes_file)
        arcs = read_arcs(arcs_file)

        if not nodes:
            print(f"Skipping {trip_id}: no valid nodes")
            continue

        stats = analyze_network(nodes, arcs)
        stats["trip_id"] = trip_id

        results.append(stats)

        print(
            f"Trip {trip_id}: "
            f"{stats['nodes']} nodes, "
            f"{stats['valid_arcs']} valid arcs, "
            f"{stats['components']} components"
        )

    if not results:
        print("No road-network data found.")
        return

    output_file = REPORT_DIR / "road_network_summary.csv"

    fieldnames = [
        "trip_id",
        "nodes",
        "total_arcs",
        "valid_arcs",
        "invalid_arcs",
        "components",
        "largest_component",
        "largest_component_percent",
        "isolated_nodes",
        "average_degree",
        "maximum_degree"
    ]

    with open(output_file, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nROAD NETWORK ANALYSIS COMPLETED")
    print(f"Trips analysed: {len(results)}")
    print(f"Report saved: {output_file}")


if __name__ == "__main__":
    main()
