"""FastAPI app implementing the evaluator frontend API contract."""
from __future__ import annotations
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.config import BackendSettings
from backend.schemas import ErrorResponse, HealthResponse, PredictRequest, PredictionResponse, TripListResponse
from src.route_reconstruction import ReconstructionError, RouteReconstructionService

LOGGER = logging.getLogger(__name__)


def create_app(settings: BackendSettings | None = None, inference_service: Any | None = None) -> FastAPI:
    config = settings or BackendSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if inference_service is not None:
            app.state.service = inference_service
        else:
            try:
                app.state.service = RouteReconstructionService(config.pipeline_config())
            except Exception as exc:
                LOGGER.exception("Backend resources failed to load")
                raise RuntimeError("PATHSNAP inference resources are unavailable; check the configured dataset and model artifacts") from exc
        yield
        app.state.service = None

    application = FastAPI(title="PATHSNAP Route Reconstruction API", version="1.0.0", lifespan=lifespan)
    application.add_middleware(CORSMiddleware, allow_origins=list(config.cors_origins), allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request):
        service = getattr(request.app.state, "service", None)
        if service is None:
            return HealthResponse(status="starting", version="1.0.0", ready_for_inference=False)
        status = service.health_status()
        model_name = "geometric-development-fallback" if service.scoring_mode == "geometric_fallback" else "person2-extra-trees"
        if service.scoring_mode != "geometric_fallback":
            try:
                metadata = json.loads((Path(__file__).resolve().parents[1] / "models" / "model_metadata.json").read_text(encoding="utf8"))
                model_name = metadata.get("model", model_name)
            except (OSError, ValueError):
                pass
        return HealthResponse(**status, model_name=model_name, version="1.0.0")

    @application.get("/api/evaluation/trips", response_model=TripListResponse)
    def trips(request: Request):
        service = getattr(request.app.state, "service", None)
        if service is None:
            raise HTTPException(status_code=503, detail="Inference service is not ready")
        return {"trips": service.list_trips()}

    @application.post("/api/evaluation/predict", response_model=PredictionResponse, responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
    def predict(payload: PredictRequest, request: Request):
        service = getattr(request.app.state, "service", None)
        if service is None:
            raise HTTPException(status_code=503, detail="Inference service is not ready")
        available = {trip["trip_id"] for trip in service.list_trips()}
        if payload.trip_id not in available:
            raise HTTPException(status_code=404, detail="Evaluation trip not found")
        try:
            return service.reconstruct(payload.trip_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Evaluation trip files are unavailable") from None
        except ReconstructionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except Exception as exc:
            LOGGER.exception("Route reconstruction failed for trip %s", payload.trip_id)
            raise HTTPException(status_code=500, detail="Route reconstruction failed") from exc

    return application


app = create_app()
