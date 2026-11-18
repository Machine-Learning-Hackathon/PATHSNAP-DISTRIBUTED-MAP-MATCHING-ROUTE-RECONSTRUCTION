"""HMM emission and directed road-network transition scoring."""
from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import Sequence

import numpy as np

from .candidate_generation import Candidate
from .road_network import RoadNetwork

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HMMConfig:
    probability_epsilon: float = 1e-9
    max_speed_mps: float = 75.0
    transition_scale_m: float = 30.0
    min_search_distance_m: float = 180.0
    max_search_distance_m: float = 5000.0

    def __post_init__(self) -> None:
        if not 0 < self.probability_epsilon < 1:
            raise ValueError("probability_epsilon must be between zero and one")
        if self.max_speed_mps <= 0 or self.transition_scale_m <= 0:
            raise ValueError("speed and transition scale must be positive")


def emission_log_probabilities(scores: Sequence[float], epsilon: float = 1e-9) -> np.ndarray:
    """Normalize independent positive-class scores within one observation, then log them."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("emission scores must be a non-empty vector")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("emission scores must be finite and non-negative")
    clipped = np.clip(values, epsilon, 1.0)
    probabilities = clipped / clipped.sum()
    return np.log(np.maximum(probabilities, epsilon))


def transition_log_matrix(
    previous: Sequence[Candidate], current: Sequence[Candidate],
    previous_gps_xy: np.ndarray, current_gps_xy: np.ndarray,
    time_delta_s: float, network: RoadNetwork, config: HMMConfig | None = None,
    allow_long_connectors: bool = False,
) -> np.ndarray:
    """Score consecutive candidate pairs from directed network distance and GPS motion.

    Unreachable directed paths are `-inf`, so Viterbi cannot silently jump across a
    disconnected component. Bounded Dijkstra is run once for each distinct previous
    candidate end node and stops after all current candidate start nodes are found.
    """
    cfg = config or HMMConfig()
    if not previous or not current:
        raise ValueError("transition scoring requires candidates at both observations")
    if not np.isfinite(time_delta_s) or time_delta_s <= 0:
        time_delta_s = 1.0
    displacement = float(np.linalg.norm(np.asarray(current_gps_xy) - np.asarray(previous_gps_xy)))
    time_limit = time_delta_s * cfg.max_speed_mps + 100.0
    cutoff = cfg.max_search_distance_m if allow_long_connectors else min(cfg.max_search_distance_m, max(cfg.min_search_distance_m, displacement * 3.0 + 50.0), time_limit)
    target_nodes = {network.arcs[c.arc_id].source for c in current}
    paths_by_source: dict[int, dict[int, float]] = {}
    for candidate in previous:
        source = network.arcs[candidate.arc_id].target
        if source not in paths_by_source:
            paths_by_source[source] = network.shortest_distances_to_targets(source, target_nodes, cutoff)
    matrix = np.full((len(previous), len(current)), -np.inf, dtype=float)
    for i, first in enumerate(previous):
        first_arc = network.arcs[first.arc_id]
        for j, second in enumerate(current):
            second_arc = network.arcs[second.arc_id]
            if first.arc_id == second.arc_id:
                route_distance = abs(second.fraction - first.fraction) * first_arc.length_m
            else:
                gap = paths_by_source[first_arc.target].get(second_arc.source)
                if gap is None:
                    continue
                route_distance = (1.0 - first.fraction) * first_arc.length_m + gap + second.fraction * second_arc.length_m
            speed = route_distance / time_delta_s
            if speed > cfg.max_speed_mps and not allow_long_connectors:
                continue
            scale = max(cfg.transition_scale_m, displacement * 0.5, 10.0)
            score = -abs(route_distance - displacement) / scale
            # Expanded-graph recovery remains connected but is heavily penalized
            # when the implied distance/speed is implausible.
            speed_scale = 10.0 if allow_long_connectors else 30.0
            score -= max(0.0, speed - 35.0) / speed_scale
            matrix[i, j] = score
    return matrix
