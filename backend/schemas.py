"""Pydantic request and response contracts for the existing evaluator UI."""
from __future__ import annotations
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str
    model_name: str | None = None
    version: str | None = None
    scoring_mode: str | None = None
    model_loaded: bool = False
    dataset_available: bool = False
    evaluation_trip_count: int = 0
    road_network_files_available: int = 0
    ground_truth_files_available: int = 0
    road_network_loading: str | None = None
    ready_for_inference: bool = False


class TripOption(BaseModel):
    trip_id: str
    label: str | None = None


class TripListResponse(BaseModel):
    trips: list[TripOption]


class PredictRequest(BaseModel):
    trip_id: str = Field(min_length=1, max_length=64)

    @field_validator("trip_id")
    @classmethod
    def validate_trip_id(cls, value: str) -> str:
        if value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError("trip_id must be a simple directory identifier")
        return value


LatLon = tuple[float, float]


class EvaluationMetrics(BaseModel):
    route_edge_precision: float
    route_edge_recall: float
    route_edge_f1: float
    sequence_similarity: float
    predicted_route_arc_count: int
    ground_truth_route_arc_count: int
    comparison_available: bool = True


class PredictionResponse(BaseModel):
    trip_id: str
    gps_trace: list[LatLon]
    predicted_route: list[LatLon]
    road_network: list[list[LatLon]] = Field(default_factory=list)
    inference_time_ms: float
    decoder: str
    status: str = "success"
    scoring_mode: str
    matched_arc_ids: list[int]
    route_arc_ids: list[int]
    matched_coordinates: list[LatLon]
    candidate_count: int
    warnings: list[str] = Field(default_factory=list)
    log_probability: float | None = None
    evaluation: EvaluationMetrics | None = None


class ErrorResponse(BaseModel):
    detail: str
