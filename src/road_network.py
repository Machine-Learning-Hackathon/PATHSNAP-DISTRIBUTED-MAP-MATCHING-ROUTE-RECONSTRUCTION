"""Directed road graph loaded from one trip's `.nodes` and `.arcs` files."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import heapq
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree

LOGGER = logging.getLogger(__name__)
EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in metres."""
    p1, p2 = np.radians([lat1, lat2])
    dl = np.radians(lon2 - lon1)
    dp = p2 - p1
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


@dataclass(frozen=True, slots=True)
class RoadArc:
    """A directed source arc. arc_id is its original zero-based `.arcs` row."""
    arc_id: int
    source: int
    target: int
    length_m: float
    bearing_rad: float


class RoadNetwork:
    """Directed adjacency graph, preserving source node and arc row IDs."""
    def __init__(self, nodes_lonlat: np.ndarray, arc_rows: np.ndarray, trip_id: str = ""):
        self.trip_id = trip_id
        raw_nodes = np.asarray(nodes_lonlat, dtype=float)
        if raw_nodes.ndim != 2 or raw_nodes.shape[1] != 2:
            raise ValueError("nodes must have [longitude, latitude] columns")
        self.nodes_lonlat = raw_nodes
        valid_nodes = np.isfinite(raw_nodes).all(axis=1) & (np.abs(raw_nodes[:, 0]) <= 180) & (np.abs(raw_nodes[:, 1]) <= 90)
        self.valid_node = valid_nodes
        self.lat0 = float(np.mean(raw_nodes[valid_nodes, 1])) if valid_nodes.any() else 0.0
        lon_scale = EARTH_RADIUS_M * np.cos(np.radians(self.lat0)) * np.pi / 180
        lat_scale = EARTH_RADIUS_M * np.pi / 180
        self.nodes_xy = np.column_stack((raw_nodes[:, 0] * lon_scale, raw_nodes[:, 1] * lat_scale))
        self.arc_rows = np.asarray(arc_rows, dtype=np.int64)
        if self.arc_rows.ndim != 2 or self.arc_rows.shape[1] != 2:
            raise ValueError("arcs must have [from_node_id, to_node_id] columns")
        self.arcs: dict[int, RoadArc] = {}
        self.adjacency: dict[int, list[tuple[int, int, float]]] = {}
        self.incoming: dict[int, list[int]] = {}
        for arc_id, pair in enumerate(self.arc_rows):
            source, target = map(int, pair)
            if source < 0 or target < 0 or source >= len(raw_nodes) or target >= len(raw_nodes):
                LOGGER.warning("Trip %s: skipping arc row %d with out-of-range node index", trip_id, arc_id)
                continue
            if not valid_nodes[source] or not valid_nodes[target]:
                LOGGER.warning("Trip %s: skipping arc row %d with invalid endpoint coordinates", trip_id, arc_id)
                continue
            vec = self.nodes_xy[target] - self.nodes_xy[source]
            length = float(np.linalg.norm(vec))
            arc = RoadArc(arc_id, source, target, length, float(np.arctan2(vec[1], vec[0])))
            self.arcs[arc_id] = arc
            self.adjacency.setdefault(source, []).append((target, arc_id, length))
            self.incoming.setdefault(target, []).append(arc_id)
        if not self.arcs:
            raise ValueError("road network contains no arcs with valid coordinates")
        self._spatial_arc_ids = np.fromiter(self.arcs.keys(), dtype=np.int64)
        self._spatial_midpoints = np.vstack([(self.nodes_xy[self.arcs[i].source] + self.nodes_xy[self.arcs[i].target]) / 2 for i in self._spatial_arc_ids])
        self._spatial_tree = cKDTree(self._spatial_midpoints)

    @classmethod
    def from_trip_folder(cls, folder: Path) -> "RoadNetwork":
        """Load `.nodes` and `.arcs` without reinterpreting their row IDs."""
        trip_id = folder.name
        nodes = np.loadtxt(folder / f"{trip_id}.nodes", dtype=float, ndmin=2)
        arcs = np.loadtxt(folder / f"{trip_id}.arcs", dtype=np.int64, ndmin=2)
        return cls(nodes, arcs, trip_id)

    def lonlat_to_xy(self, lon: float, lat: float) -> np.ndarray:
        return np.array([lon * EARTH_RADIUS_M * np.cos(np.radians(self.lat0)) * np.pi / 180,
                         lat * EARTH_RADIUS_M * np.pi / 180], dtype=float)

    def xy_to_lonlat(self, point: Iterable[float]) -> tuple[float, float]:
        x, y = point
        lon = x / (EARTH_RADIUS_M * np.cos(np.radians(self.lat0)) * np.pi / 180)
        lat = y / (EARTH_RADIUS_M * np.pi / 180)
        return float(lon), float(lat)

    def nearby_arc_ids(self, lon: float, lat: float, count: int = 64) -> list[int]:
        """Return nearest arc midpoints; final radius filtering uses exact segment distance."""
        k = min(max(1, int(count)), len(self._spatial_arc_ids))
        _, indices = self._spatial_tree.query(self.lonlat_to_xy(lon, lat), k=k)
        indices = np.atleast_1d(indices)
        return [int(self._spatial_arc_ids[i]) for i in indices if i < len(self._spatial_arc_ids)]

    def arc_geometry(self, arc_id: int) -> list[list[float]]:
        arc = self.arcs[arc_id]
        return [[float(self.nodes_lonlat[arc.source, 1]), float(self.nodes_lonlat[arc.source, 0])],
                [float(self.nodes_lonlat[arc.target, 1]), float(self.nodes_lonlat[arc.target, 0])]]

    def shortest_distances_to_targets(self, source: int, targets: set[int], cutoff_m: float) -> dict[int, float]:
        """Bounded directed Dijkstra; stops once all requested target nodes are settled."""
        if not targets:
            return {}
        found: dict[int, float] = {source: 0.0} if source in targets else {}
        distances = {source: 0.0}
        queue = [(0.0, source)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != distances.get(node):
                continue
            if distance > cutoff_m:
                break
            if node in targets:
                found[node] = distance
                if len(found) == len(targets):
                    break
            for target, _, length in self.adjacency.get(node, ()):
                proposed = distance + length
                if proposed <= cutoff_m and proposed < distances.get(target, float("inf")):
                    distances[target] = proposed
                    heapq.heappush(queue, (proposed, target))
        return found

    def shortest_path_arcs(self, source: int, target: int, cutoff_m: float) -> list[int] | None:
        """Return directed arc IDs in a bounded shortest path, or None if unreachable."""
        if source == target:
            return []
        distances = {source: 0.0}
        predecessor: dict[int, tuple[int, int]] = {}
        queue = [(0.0, source)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != distances.get(node):
                continue
            if distance > cutoff_m:
                break
            if node == target:
                break
            for nxt, arc_id, length in self.adjacency.get(node, ()):
                proposed = distance + length
                if proposed <= cutoff_m and proposed < distances.get(nxt, float("inf")):
                    distances[nxt] = proposed
                    predecessor[nxt] = (node, arc_id)
                    heapq.heappush(queue, (proposed, nxt))
        if target not in predecessor:
            return None
        path: list[int] = []
        node = target
        while node != source:
            node, arc_id = predecessor[node]
            path.append(arc_id)
        return list(reversed(path))

    def connected(self, source_arc_id: int, target_arc_id: int, cutoff_m: float = 200.0) -> bool:
        """Whether the end of one directed arc reaches the start of another."""
        first, second = self.arcs[source_arc_id], self.arcs[target_arc_id]
        return bool(self.shortest_distances_to_targets(first.target, {second.source}, cutoff_m).get(second.source) is not None)

    def route_coordinates(self, ordered_arc_ids: Iterable[int]) -> list[list[float]]:
        """Expand directed arc IDs to an ordered route as [latitude, longitude]."""
        coordinates: list[list[float]] = []
        for arc_id in ordered_arc_ids:
            geometry = self.arc_geometry(int(arc_id))
            for point in geometry:
                if not coordinates or coordinates[-1] != point:
                    coordinates.append(point)
        return coordinates
