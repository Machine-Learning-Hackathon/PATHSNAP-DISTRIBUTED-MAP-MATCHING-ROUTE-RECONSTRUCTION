"""HMM map matcher with Gaussian emissions and Viterbi decoding.

Input formats (same as the repository's exploration code):
  .track: longitude latitude timestamp
  .nodes: longitude latitude (zero-based row index is node_id)
  .arcs: source_node_id target_node_id (zero-based row index is arc_id)

The decoder returns a sequence of arc row IDs. Geometric calculations use a
local equirectangular projection centred on the road network, in metres.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, log, radians
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

EARTH_RADIUS_M = 6_371_000.0


def _project_lonlat(coords: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    lon = np.radians(coords[:, 0])
    lat = np.radians(coords[:, 1])
    x = EARTH_RADIUS_M * (lon - radians(lon0)) * cos(radians(lat0))
    y = EARTH_RADIUS_M * (lat - radians(lat0))
    return np.column_stack((x, y))


def _bearing(a: np.ndarray, b: np.ndarray) -> float:
    return atan2(b[0] - a[0], b[1] - a[1])


def _angle_difference(a: float, b: float) -> float:
    d = abs(a - b) % (2 * np.pi)
    return min(d, 2 * np.pi - d)


def _project_to_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray):
    vector = end - start
    length_sq = float(vector @ vector)
    if length_sq <= 1e-12:
        return start, 0.0, float(np.linalg.norm(point - start))
    fraction = float(np.clip(((point - start) @ vector) / length_sq, 0.0, 1.0))
    projected = start + fraction * vector
    return projected, fraction, float(np.linalg.norm(point - projected))


@dataclass(frozen=True)
class Candidate:
    arc_id: int
    source: int
    target: int
    fraction: float
    distance_m: float
    heading_rad: float
    projected_xy: tuple[float, float]


@dataclass
class HMMConfig:
    candidate_radius_m: float = 80.0
    max_candidates: int = 8
    gps_sigma_m: float = 20.0
    transition_beta_m: float = 30.0
    heading_weight: float = 0.35
    max_graph_expansions: int = 100_000


class HMMMapMatcher:
    """HMM whose hidden states are directed road arcs."""

    def __init__(self, nodes_lonlat: Sequence[Sequence[float]],
                 arcs: Sequence[Sequence[int]], config: HMMConfig | None = None):
        self.nodes = np.asarray(nodes_lonlat, dtype=float)
        self.arcs = np.asarray(arcs, dtype=int)
        self.config = config or HMMConfig()
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2 or not len(self.nodes):
            raise ValueError("nodes must be a non-empty Nx2 [longitude, latitude] array")
        if self.arcs.ndim != 2 or self.arcs.shape[1] != 2 or not len(self.arcs):
            raise ValueError("arcs must be a non-empty Mx2 [source, target] array")
        if not np.isfinite(self.nodes).all():
            raise ValueError("nodes contain non-finite coordinates")
        if np.any(self.arcs < 0) or np.any(self.arcs >= len(self.nodes)):
            raise ValueError("arc endpoint references an invalid node row")
        if self.config.max_candidates < 1 or self.config.candidate_radius_m <= 0:
            raise ValueError("candidate configuration must be positive")
        if self.config.gps_sigma_m <= 0 or self.config.transition_beta_m <= 0:
            raise ValueError("HMM sigma and beta must be positive")

        self.lon0 = float(np.mean(self.nodes[:, 0]))
        self.lat0 = float(np.mean(self.nodes[:, 1]))
        self.nodes_xy = _project_lonlat(self.nodes, self.lon0, self.lat0)
        vectors = self.nodes_xy[self.arcs[:, 1]] - self.nodes_xy[self.arcs[:, 0]]
        self.arc_lengths = np.linalg.norm(vectors, axis=1)
        self.arc_headings = np.arctan2(vectors[:, 0], vectors[:, 1])
        self.tree = cKDTree(self.nodes_xy)
        self.outgoing: list[list[int]] = [[] for _ in range(len(self.nodes))]
        self.incident: list[list[int]] = [[] for _ in range(len(self.nodes))]
        for arc_id, (source, target) in enumerate(self.arcs):
            self.outgoing[int(source)].append(arc_id)
            self.incident[int(source)].append(arc_id)
            self.incident[int(target)].append(arc_id)
        self._distance_cache: dict[tuple[int, int], float] = {}

    def _candidate_states(self, gps_xy: np.ndarray, gps_heading: float | None):
        nearby_nodes = self.tree.query_ball_point(gps_xy, self.config.candidate_radius_m)
        arc_ids = {arc for node in nearby_nodes for arc in self.incident[node]}
        if not arc_ids:
            _, nearest = self.tree.query(gps_xy, k=min(3, len(self.nodes)))
            arc_ids = {arc for node in np.atleast_1d(nearest) for arc in self.incident[int(node)]}
        scored: list[Candidate] = []
        for arc_id in arc_ids:
            source, target = map(int, self.arcs[arc_id])
            projected, fraction, distance = _project_to_segment(
                gps_xy, self.nodes_xy[source], self.nodes_xy[target]
            )
            if distance <= self.config.candidate_radius_m:
                scored.append(Candidate(
                    int(arc_id), source, target, fraction, distance,
                    float(self.arc_headings[arc_id]),
                    (float(projected[0]), float(projected[1]))
                ))
        if not scored:
            # Always retain the nearest directed arc as a deterministic fallback.
            for arc_id, (source, target) in enumerate(self.arcs):
                projected, fraction, distance = _project_to_segment(
                    gps_xy, self.nodes_xy[source], self.nodes_xy[target]
                )
                scored.append(Candidate(
                    arc_id, int(source), int(target), fraction, distance,
                    float(self.arc_headings[arc_id]),
                    (float(projected[0]), float(projected[1]))
                ))
        scored.sort(key=lambda c: c.distance_m)
        return scored[:self.config.max_candidates]

    def _shortest_path_distance(self, start: int, end: int) -> float:
        key = (start, end)
        if key in self._distance_cache:
            return self._distance_cache[key]
        if start == end:
            return 0.0
        import heapq
        queue = [(0.0, start)]
        distances = {start: 0.0}
        expansions = 0
        result = float("inf")
        while queue and expansions < self.config.max_graph_expansions:
            distance, node = heapq.heappop(queue)
            if distance != distances.get(node):
                continue
            if node == end:
                result = distance
                break
            expansions += 1
            for arc_id in self.outgoing[node]:
                nxt = int(self.arcs[arc_id, 1])
                candidate_distance = distance + float(self.arc_lengths[arc_id])
                if candidate_distance < distances.get(nxt, float("inf")):
                    distances[nxt] = candidate_distance
                    heapq.heappush(queue, (candidate_distance, nxt))
        self._distance_cache[key] = result
        return result

    def _network_distance(self, left: Candidate, right: Candidate) -> float:
        if left.arc_id == right.arc_id and right.fraction >= left.fraction:
            return (right.fraction - left.fraction) * self.arc_lengths[left.arc_id]
        middle = self._shortest_path_distance(left.target, right.source)
        if not np.isfinite(middle):
            return float("inf")
        return ((1.0 - left.fraction) * self.arc_lengths[left.arc_id]
                + middle + right.fraction * self.arc_lengths[right.arc_id])

    def _emission_logp(self, candidate: Candidate, gps_heading: float | None) -> float:
        sigma = self.config.gps_sigma_m
        score = -0.5 * (candidate.distance_m / sigma) ** 2 - log(sigma)
        if gps_heading is not None:
            angle = _angle_difference(candidate.heading_rad, gps_heading)
            score -= self.config.heading_weight * (angle / np.pi) ** 2
        return score

    def _transition_logp(self, previous: Candidate, current: Candidate,
                         observed_distance: float) -> float:
        network_distance = self._network_distance(previous, current)
        if not np.isfinite(network_distance):
            return -1e6
        beta = self.config.transition_beta_m
        return -abs(network_distance - observed_distance) / beta - log(beta)

    def decode(self, gps_lonlat: Sequence[Sequence[float]],
               timestamps: Sequence[float] | None = None) -> dict:
        gps = np.asarray(gps_lonlat, dtype=float)
        if gps.ndim != 2 or gps.shape[1] != 2 or not len(gps):
            raise ValueError("gps_lonlat must be a non-empty Nx2 [longitude, latitude] array")
        if not np.isfinite(gps).all():
            raise ValueError("GPS contains non-finite coordinates")
        times = None if timestamps is None else np.asarray(timestamps, dtype=float)
        if times is not None:
            if len(times) != len(gps) or not np.isfinite(times).all():
                raise ValueError("timestamps must match GPS rows and be finite")
            if np.any(np.diff(times) < 0):
                raise ValueError("timestamps must be non-decreasing")

        gps_xy = _project_lonlat(gps, self.lon0, self.lat0)
        headings: list[float | None] = [None] * len(gps)
        steps = np.zeros(len(gps), dtype=float)
        for i in range(1, len(gps)):
            delta = gps_xy[i] - gps_xy[i - 1]
            steps[i] = float(np.linalg.norm(delta))
            if steps[i] > 0.1:
                headings[i] = float(np.arctan2(delta[0], delta[1]))

        candidates = [self._candidate_states(p, headings[i]) for i, p in enumerate(gps_xy)]
        scores = [np.array([self._emission_logp(c, headings[0]) for c in candidates[0]])]
        back = [np.full(len(candidates[0]), -1, dtype=int)]
        for t in range(1, len(gps)):
            row_scores = np.full(len(candidates[t]), -np.inf)
            row_back = np.full(len(candidates[t]), -1, dtype=int)
            for j, current in enumerate(candidates[t]):
                emission = self._emission_logp(current, headings[t])
                for i, previous in enumerate(candidates[t - 1]):
                    score = (scores[t - 1][i]
                             + self._transition_logp(previous, current, steps[t])
                             + emission)
                    if score > row_scores[j]:
                        row_scores[j], row_back[j] = score, i
            scores.append(row_scores)
            back.append(row_back)

        state = int(np.argmax(scores[-1]))
        state_path = [state]
        for t in range(len(gps) - 1, 0, -1):
            state = int(back[t][state])
            if state < 0:
                raise RuntimeError("Viterbi backtracking failed")
            state_path.append(state)
        state_path.reverse()
        chosen = [candidates[t][state_path[t]] for t in range(len(gps))]

        arc_ids: list[int] = []
        for state in chosen:
            if not arc_ids or arc_ids[-1] != state.arc_id:
                arc_ids.append(state.arc_id)
        node_ids: list[int] = []
        if chosen:
            node_ids.append(chosen[0].source)
            for state in chosen:
                if node_ids[-1] != state.target:
                    node_ids.append(state.target)
        return {
            "arc_ids": arc_ids,
            "node_ids": node_ids,
            "matched_points_lonlat": [
                [float(c.projected_xy[0] / (EARTH_RADIUS_M * cos(radians(self.lat0))) * 180 / np.pi + self.lon0),
                 float(c.projected_xy[1] / EARTH_RADIUS_M * 180 / np.pi + self.lat0)]
                for c in chosen
            ],
            "candidate_arc_ids": [[c.arc_id for c in row] for row in candidates],
            "viterbi_score": float(np.max(scores[-1])),
            "candidate_count_mean": float(np.mean([len(row) for row in candidates])),
        }
