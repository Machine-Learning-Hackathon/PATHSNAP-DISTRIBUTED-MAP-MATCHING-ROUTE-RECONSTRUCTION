"""Reusable GPS-to-candidate-score-to-HMM-to-Viterbi route pipeline."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np

from .candidate_generation import Candidate, CandidateConfig, generate_candidates
from .evaluation_metrics import route_sequence_metrics
from .hmm import HMMConfig, emission_log_probabilities, transition_log_matrix
from .model_predictor import CandidateScorer
from .road_network import RoadNetwork
from .viterbi import viterbi_decode

LOGGER = logging.getLogger(__name__)


class ReconstructionError(RuntimeError):
    """Expected data, scoring, or path failure during inference."""


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    dataset_root: Path
    model_path: Path
    schema_path: Path
    evaluation_partition: str = "student_test"
    candidate: CandidateConfig = CandidateConfig()
    hmm: HMMConfig = HMMConfig()
    allow_geometric_fallback: bool = False
    fallback_sigma_m: float = 25.0
    max_display_road_segments: int = 300


class RouteReconstructionService:
    """Shared inference service; ground-truth `.route` files are never read here."""
    def __init__(self, config: PipelineConfig, scorer: CandidateScorer | None = None):
        self.config = config
        self.scorer = scorer
        self.scoring_mode = "trained_model"
        self.model_error: str | None = None
        if scorer is None:
            try:
                self.scorer = CandidateScorer(config.model_path, config.schema_path)
            except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
                self.model_error = str(exc)
                if not config.allow_geometric_fallback:
                    raise RuntimeError(f"Could not load the required Person 2 model artifacts: {exc}") from exc
                self.scoring_mode = "geometric_fallback"
                LOGGER.warning("Explicit development fallback enabled: %s", exc)
        self.partition_dir = config.dataset_root / "Student folder" / config.evaluation_partition
        if not self.partition_dir.is_dir():
            raise FileNotFoundError(f"Evaluation partition is unavailable: {self.partition_dir}")

    @lru_cache(maxsize=1)
    def _network(self, trip_id: str) -> RoadNetwork:
        folder = self.partition_dir / trip_id
        if not folder.is_dir():
            raise FileNotFoundError("Trip does not exist")
        return RoadNetwork.from_trip_folder(folder)

    def list_trips(self) -> list[dict[str, str]]:
        """Return exactly ten deterministic evaluation options; never expose labels/routes."""
        trips = []
        for folder in sorted(p for p in self.partition_dir.iterdir() if p.is_dir()):
            trip_id = folder.name
            required = [folder / f"{trip_id}.track", folder / f"{trip_id}.nodes", folder / f"{trip_id}.arcs"]
            if all(path.is_file() for path in required):
                trips.append({"trip_id": trip_id, "label": f"Evaluation trip {len(trips) + 1:02d}"})
        trips.sort(key=lambda item: item["trip_id"])
        if len(trips) > 10:
            LOGGER.warning("Found %d usable evaluation trips; using the first ten trip IDs in lexical order", len(trips))
        return trips[:10]

    def health_status(self) -> dict[str, object]:
        """Report artifact and on-disk evaluation readiness without loading every graph."""
        trips = self.list_trips()
        networks_available = sum(
            int((self.partition_dir / trip["trip_id"] / f"{trip['trip_id']}.nodes").is_file()
                and (self.partition_dir / trip["trip_id"] / f"{trip['trip_id']}.arcs").is_file())
            for trip in trips
        )
        labels_available = sum((self.partition_dir / trip["trip_id"] / f"{trip['trip_id']}.route").is_file() for trip in trips)
        model_loaded = self.scorer is not None
        ready = len(trips) == 10 and networks_available == 10 and labels_available == 10 and model_loaded
        return {
            "status": "ok" if ready else "degraded",
            "model_loaded": model_loaded,
            "scoring_mode": self.scoring_mode,
            "dataset_available": self.partition_dir.is_dir() and len(trips) == 10,
            "evaluation_trip_count": len(trips),
            "road_network_files_available": networks_available,
            "ground_truth_files_available": labels_available,
            "road_network_loading": "loaded on demand per selected trip",
            "ready_for_inference": ready,
        }

    def _load_track(self, folder: Path) -> tuple[np.ndarray, list[str]]:
        path = folder / f"{folder.name}.track"
        try:
            track = np.loadtxt(path, dtype=float, ndmin=2)
        except (OSError, ValueError) as exc:
            raise ReconstructionError("Trip track is missing or malformed") from exc
        if track.shape[1] != 3:
            raise ReconstructionError("Track must have longitude, latitude, and sequence/time columns")
        finite = np.isfinite(track).all(axis=1)
        coordinate_ok = (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
        valid = finite & coordinate_ok
        warnings: list[str] = []
        if not valid.all():
            warnings.append(f"Skipped {int((~valid).sum())} observations with missing or invalid values")
            track = track[valid]
        if not len(track):
            raise ReconstructionError("Trip has no valid GPS observations")
        if len(track) > 1 and np.any(np.diff(track[:, 2]) < 0):
            warnings.append("Sequence/time values decrease; source row order was preserved and invalid time gaps use a 1 second default")
        return track, warnings

    def _score(self, candidates: list[list[Candidate]]) -> list[np.ndarray]:
        records = [candidate.feature_record() for group in candidates for candidate in group]
        if not records or self.scoring_mode == "geometric_fallback":
            self.scoring_mode = "geometric_fallback"
            return [emission_log_probabilities([np.exp(-0.5 * (c.distance_m / self.config.fallback_sigma_m) ** 2) * max(0.05, (1 + c.heading_cos) / 2) for c in group]) for group in candidates]
        assert self.scorer is not None
        scored = self.scorer.score_candidates(records)
        emissions: list[np.ndarray] = []
        cursor = 0
        for group in candidates:
            probabilities = scored.score.iloc[cursor:cursor + len(group)].to_numpy(dtype=float)
            if len(group):
                emissions.append(emission_log_probabilities(probabilities, self.config.hmm.probability_epsilon))
            else:
                emissions.append(np.array([], dtype=float))
            cursor += len(group)
        return emissions

    def reconstruct(self, trip_id: str) -> dict[str, Any]:
        """Reconstruct one trip without accessing its ground-truth route file."""
        started = time.perf_counter()
        folder = self.partition_dir / trip_id
        if not folder.is_dir():
            raise FileNotFoundError("Trip does not exist")
        track, warnings = self._load_track(folder)
        network = self._network(trip_id)
        candidates = generate_candidates(track, network, self.config.candidate)
        missing = [i for i, group in enumerate(candidates) if not group]
        if missing:
            raise ReconstructionError(f"No road candidates found for GPS observation(s): {missing[:10]}")
        if any(c.fallback for group in candidates for c in group):
            warnings.append("Some observations used a nearest-road candidate within the configured fallback radius")
        emissions = self._score(candidates)
        gps_xy = [network.lonlat_to_xy(float(row[0]), float(row[1])) for row in track]
        transitions = []
        reachable = np.isfinite(emissions[0])
        long_connector_layers: set[int] = set()
        for i in range(1, len(candidates)):
            dt = float(track[i, 2] - track[i - 1, 2])
            if not np.isfinite(dt) or dt <= 0:
                dt = 1.0
            matrix = transition_log_matrix(candidates[i - 1], candidates[i], gps_xy[i - 1], gps_xy[i], dt, network, self.config.hmm)
            if not np.isfinite(matrix[reachable]).any():
                # Only after physically plausible graph transitions fail, search
                # farther through the directed graph. These edges remain real
                # network paths and receive a strong distance/speed penalty.
                expanded = transition_log_matrix(candidates[i - 1], candidates[i], gps_xy[i - 1], gps_xy[i], dt, network, self.config.hmm, allow_long_connectors=True)
                if np.isfinite(expanded[reachable]).any():
                    matrix = expanded
                    long_connector_layers.add(i)
                    warnings.append(f"Used a connected but physically long road-network transition before observation {i}")
            transitions.append(matrix)
            reachable = np.isfinite(matrix[reachable]).any(axis=0)
        decoded = viterbi_decode(emissions, transitions)
        if decoded.status != "success":
            raise ReconstructionError(decoded.message or "No valid connected road path exists")
        chosen = [group[index] for group, index in zip(candidates, decoded.state_indices)]
        matched_arc_ids = [candidate.arc_id for candidate in chosen]
        route_arc_ids: list[int] = []
        for index, candidate in enumerate(chosen):
            if index == 0:
                route_arc_ids.append(candidate.arc_id)
                continue
            previous = chosen[index - 1]
            if previous.arc_id != candidate.arc_id:
                first_arc, next_arc = network.arcs[previous.arc_id], network.arcs[candidate.arc_id]
                displacement = float(np.linalg.norm(gps_xy[index] - gps_xy[index - 1]))
                dt = max(float(track[index, 2] - track[index - 1, 2]), 1.0)
                cutoff = self.config.hmm.max_search_distance_m if index in long_connector_layers else min(self.config.hmm.max_search_distance_m, max(self.config.hmm.min_search_distance_m, displacement * 3 + 50), dt * self.config.hmm.max_speed_mps + 100)
                connector = network.shortest_path_arcs(first_arc.target, next_arc.source, cutoff)
                if connector is None:
                    raise ReconstructionError("Decoded transition became disconnected during route expansion")
                route_arc_ids.extend(connector)
                route_arc_ids.append(candidate.arc_id)
        route_arc_ids = [arc_id for i, arc_id in enumerate(route_arc_ids) if i == 0 or arc_id != route_arc_ids[i - 1]]
        matched = []
        for candidate in chosen:
            arc = network.arcs[candidate.arc_id]
            point = network.nodes_xy[arc.source] + candidate.fraction * (network.nodes_xy[arc.target] - network.nodes_xy[arc.source])
            lon, lat = network.xy_to_lonlat(point)
            matched.append([lat, lon])
        display_ids = []
        stride = max(1, len(candidates) // max(1, self.config.max_display_road_segments))
        for i in range(0, len(candidates), stride):
            display_ids.extend(c.arc_id for c in candidates[i])
        display_ids = list(dict.fromkeys(display_ids))[:self.config.max_display_road_segments]
        road_network = [network.arc_geometry(arc_id) for arc_id in display_ids]
        result = {
            "trip_id": trip_id,
            "gps_trace": [[float(row[1]), float(row[0])] for row in track],
            "predicted_route": network.route_coordinates(route_arc_ids),
            "road_network": road_network,
            "inference_time_ms": (time.perf_counter() - started) * 1000,
            "decoder": "HMM + Viterbi",
            "status": "success",
            "scoring_mode": self.scoring_mode,
            "matched_arc_ids": matched_arc_ids,
            "route_arc_ids": route_arc_ids,
            "matched_coordinates": matched,
            "warnings": warnings,
            "candidate_count": int(sum(map(len, candidates))),
            "log_probability": decoded.log_probability,
        }
        # Evaluation is deliberately post-prediction: the route file is opened only
        # after candidates, model scores, HMM, Viterbi, and route expansion are done.
        reference_path = folder / f"{trip_id}.route"
        if reference_path.is_file():
            try:
                reference = np.loadtxt(reference_path, dtype=int, ndmin=1).tolist()
                result["evaluation"] = route_sequence_metrics(route_arc_ids, reference)
            except (OSError, ValueError) as exc:
                LOGGER.warning("Trip %s prediction completed, but post-prediction evaluation failed: %s", trip_id, exc)
                result["evaluation"] = None
                warnings.append("Ground-truth comparison was unavailable after prediction")
        else:
            result["evaluation"] = None
        return result
