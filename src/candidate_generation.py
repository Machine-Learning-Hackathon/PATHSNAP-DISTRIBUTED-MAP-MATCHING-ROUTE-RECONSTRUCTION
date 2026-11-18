"""Spatial candidate generation and Person 2-compatible feature records."""
from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import Sequence

import numpy as np

from .road_network import RoadNetwork

LOGGER = logging.getLogger(__name__)
FEATURE_COLUMNS = ("distance_m", "heading_cos", "heading_sin", "length_m", "step_distance_m", "turn_cos", "turn_sin")


@dataclass(frozen=True, slots=True)
class Candidate:
    observation_id: int
    arc_id: int
    distance_m: float
    heading_cos: float
    heading_sin: float
    length_m: float
    step_distance_m: float
    turn_cos: float
    turn_sin: float
    fraction: float
    fallback: bool = False

    def feature_record(self) -> dict[str, float | int]:
        return {"observation_id": self.observation_id, "candidate_id": self.arc_id,
                "distance_m": self.distance_m, "heading_cos": self.heading_cos,
                "heading_sin": self.heading_sin, "length_m": self.length_m,
                "step_distance_m": self.step_distance_m, "turn_cos": self.turn_cos,
                "turn_sin": self.turn_sin}


@dataclass(frozen=True, slots=True)
class CandidateConfig:
    search_radius_m: float = 120.0
    fallback_radius_m: float = 350.0
    max_candidates: int = 8
    spatial_pool_size: int = 96

    def __post_init__(self) -> None:
        if self.search_radius_m <= 0 or self.fallback_radius_m < self.search_radius_m:
            raise ValueError("candidate radii must be positive and fallback_radius_m >= search_radius_m")
        if self.max_candidates < 1 or self.spatial_pool_size < self.max_candidates:
            raise ValueError("candidate counts must be positive and the spatial pool at least max_candidates")


def generate_candidates(track: np.ndarray, network: RoadNetwork, config: CandidateConfig | None = None) -> list[list[Candidate]]:
    """Find geometrically plausible directed arcs for each ordered [lon,lat,time] row.

    Uses a local metric projection for point-to-segment distance. If no segment is
    within the primary radius, candidates are emitted only within the separately
    configured fallback radius and marked as geometric fallbacks. Beyond that,
    the observation receives an empty candidate set and reconstruction fails.
    """
    cfg = config or CandidateConfig()
    observations = np.asarray(track, dtype=float)
    if observations.ndim != 2 or observations.shape[1] != 3:
        raise ValueError("track must contain [longitude, latitude, timestamp] rows")
    if not np.isfinite(observations).all():
        raise ValueError("track contains missing or non-finite values")
    lon, lat = observations[:, 0], observations[:, 1]
    if np.any(np.abs(lon) > 180) or np.any(np.abs(lat) > 90):
        raise ValueError("track contains invalid longitude or latitude")
    points = np.column_stack((lon * 6371000.0 * np.cos(np.radians(network.lat0)) * np.pi / 180,
                              lat * 6371000.0 * np.pi / 180))
    movements = np.diff(points, axis=0, prepend=points[:1])
    headings = np.arctan2(movements[:, 1], movements[:, 0])
    step_distances = np.linalg.norm(movements, axis=1)
    turns = np.diff(headings, prepend=headings[:1])
    output: list[list[Candidate]] = []
    for observation_id, point in enumerate(points):
        nearby = network.nearby_arc_ids(float(lon[observation_id]), float(lat[observation_id]), cfg.spatial_pool_size)
        scored: list[tuple[float, int, float, float]] = []
        for arc_id in nearby:
            arc = network.arcs[arc_id]
            start, end = network.nodes_xy[arc.source], network.nodes_xy[arc.target]
            vector = end - start
            denominator = max(float(np.dot(vector, vector)), 1e-9)
            fraction = float(np.clip(np.dot(point - start, vector) / denominator, 0.0, 1.0))
            distance = float(np.linalg.norm(point - (start + fraction * vector)))
            scored.append((distance, arc_id, fraction, arc.bearing_rad))
        scored.sort(key=lambda item: (item[0], item[1]))
        primary = [item for item in scored if item[0] <= cfg.search_radius_m]
        is_fallback = False
        if not primary:
            primary = [item for item in scored if item[0] <= cfg.fallback_radius_m]
            is_fallback = bool(primary)
            if is_fallback:
                LOGGER.warning("Trip %s observation %d only has roads within fallback radius %.0fm", network.trip_id, observation_id, cfg.fallback_radius_m)
        chosen = primary[:cfg.max_candidates]
        candidates = []
        for distance, arc_id, fraction, bearing in chosen:
            difference = float(headings[observation_id] - bearing)
            arc = network.arcs[arc_id]
            candidates.append(Candidate(observation_id, arc_id, distance, float(np.cos(difference)),
                float(np.sin(difference)), arc.length_m, float(step_distances[observation_id]),
                float(np.cos(turns[observation_id])), float(np.sin(turns[observation_id])), fraction, is_fallback))
        output.append(candidates)
    return output
