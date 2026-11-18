"""FastAPI endpoints for PathSnap route reconstruction."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.hmm_viterbi import HMMConfig, HMMMapMatcher

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.getenv("PATHSNAP_DATA_DIR", ROOT / "dataset" / "Student folder"))
EVALUATION_SPLIT = os.getenv("PATHSNAP_EVALUATION_SPLIT", "student_test")
MODEL_DIR = ROOT / "models"

app = FastAPI(title="PathSnap Route Reconstruction API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    trip_id: str = Field(min_length=1, max_length=128)


def _load_numeric(path: Path, columns: int) -> np.ndarray:
    data = np.loadtxt(path, dtype=float)
    data = np.asarray(data)
    if data.ndim == 1:
        if data.size != columns:
            raise ValueError(f"{path.name} must contain {columns} columns")
        data = data.reshape(1, columns)
    if data.ndim != 2 or data.shape[1] != columns:
        raise ValueError(f"{path.name} must contain exactly {columns} columns")
    return data


def _trip_folder(trip_id: str) -> Path:
    if not trip_id.isdigit():
        raise HTTPException(status_code=400, detail="Invalid trip_id")
    folder = DATA_ROOT / EVALUATION_SPLIT / trip_id
    if not folder.is_dir():
        raise HTTPException(status_code=404, detail="Evaluation trip not found")
    return folder


def _load_trip(trip_id: str):
    folder = _trip_folder(trip_id)
    gps = _load_numeric(folder / f"{trip_id}.track", 3)
    nodes = _load_numeric(folder / f"{trip_id}.nodes", 2)
    arcs = _load_numeric(folder / f"{trip_id}.arcs", 2).astype(int)
    # .track is longitude, latitude, timestamp. Do not load .route here.
    return gps[:, :2], gps[:, 2], nodes, arcs


def _config() -> HMMConfig:
    path = MODEL_DIR / "hmm_config.json"
    return HMMConfig(**json.loads(path.read_text())) if path.exists() else HMMConfig()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_name": "HMM + Viterbi", "version": "0.1.0"}


@app.get("/api/evaluation/trips")
def evaluation_trips() -> dict[str, list[dict[str, str]]]:
    partition = DATA_ROOT / EVALUATION_SPLIT
    if not partition.is_dir():
        return {"trips": []}
    trips = []
    for folder in sorted(partition.iterdir()):
        trip_id = folder.name
        if (folder.is_dir() and trip_id.isdigit()
                and all((folder / f"{trip_id}{suffix}").is_file()
                        for suffix in (".track", ".nodes", ".arcs"))):
            trips.append({"trip_id": trip_id, "label": trip_id})
    return {"trips": trips}


@app.post("/api/evaluation/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        gps, timestamps, nodes, arcs = _load_trip(request.trip_id)
        decoded = HMMMapMatcher(nodes, arcs, _config()).decode(gps, timestamps)
        node_ids = decoded["node_ids"]
        route_lonlat = nodes[node_ids].tolist() if node_ids else []
        return {
            "trip_id": request.trip_id,
            "gps_trace": gps[:, [1, 0]].tolist(),  # frontend uses [lat, lon]
            "predicted_route": [[lat, lon] for lon, lat in route_lonlat],
            "predicted_arc_ids": decoded["arc_ids"],
            "route_node_ids": node_ids,
            "road_network": [
                [[float(nodes[s, 1]), float(nodes[s, 0])],
                 [float(nodes[t, 1]), float(nodes[t, 0])]]
                for s, t in arcs
            ],
            "inference_time_ms": (time.perf_counter() - started) * 1000.0,
            "decoder": "HMM + Viterbi",
            "viterbi_score": decoded["viterbi_score"],
        }
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
