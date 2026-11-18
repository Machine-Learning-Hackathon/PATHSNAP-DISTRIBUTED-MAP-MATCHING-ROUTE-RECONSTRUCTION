"""Environment-backed backend settings; paths are relative to the project root by default."""
from __future__ import annotations
from dataclasses import dataclass
import os
import json
import logging
from pathlib import Path

from src.candidate_generation import CandidateConfig
from src.hmm import HMMConfig
from src.route_reconstruction import PipelineConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


def _path_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    path = Path(value).expanduser() if value else default
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True, slots=True)
class BackendSettings:
    dataset_root: Path
    model_path: Path
    schema_path: Path
    host: str = "0.0.0.0"
    port: int = 8000
    allow_geometric_fallback: bool = False
    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")
    candidate_radius_m: float = 120.0
    fallback_radius_m: float = 350.0
    max_candidates: int = 8
    max_speed_mps: float = 75.0

    @classmethod
    def from_env(cls) -> "BackendSettings":
        origins = tuple(item.strip() for item in os.getenv("PATHSNAP_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if item.strip())
        return cls(
            dataset_root=_path_env("PATHSNAP_DATASET_ROOT", PROJECT_ROOT / "dataset"),
            model_path=_path_env("PATHSNAP_MODEL_PATH", PROJECT_ROOT / "models" / "best_model.joblib"),
            schema_path=_path_env("PATHSNAP_FEATURE_SCHEMA", PROJECT_ROOT / "models" / "feature_schema.json"),
            host=os.getenv("PATHSNAP_HOST", "0.0.0.0"),
            port=int(os.getenv("PATHSNAP_PORT", "8000")),
            allow_geometric_fallback=os.getenv("PATHSNAP_DEV_FALLBACK", "false").lower() in {"1", "true", "yes"},
            cors_origins=origins,
            candidate_radius_m=float(os.getenv("PATHSNAP_CANDIDATE_RADIUS_M", "120")),
            fallback_radius_m=float(os.getenv("PATHSNAP_FALLBACK_RADIUS_M", "350")),
            max_candidates=int(os.getenv("PATHSNAP_MAX_CANDIDATES", "8")),
            max_speed_mps=float(os.getenv("PATHSNAP_MAX_SPEED_MPS", "75")),
        )

    def pipeline_config(self) -> PipelineConfig:
        hmm_values = {
            "probability_epsilon": 1e-9,
            "max_speed_mps": self.max_speed_mps,
            "transition_scale_m": 30.0,
            "min_search_distance_m": 180.0,
            "max_search_distance_m": 5000.0,
        }
        hmm_path = PROJECT_ROOT / "models" / "inference" / "hmm_config.json"
        if hmm_path.is_file():
            try:
                saved = json.loads(hmm_path.read_text(encoding="utf8"))
                hmm_values.update({key: saved[key] for key in hmm_values if key in saved})
            except (OSError, ValueError) as exc:
                LOGGER.warning("Could not read HMM configuration %s: %s", hmm_path.name, exc)
        hmm_values["max_speed_mps"] = self.max_speed_mps
        return PipelineConfig(
            dataset_root=self.dataset_root,
            model_path=self.model_path,
            schema_path=self.schema_path,
            candidate=CandidateConfig(self.candidate_radius_m, self.fallback_radius_m, self.max_candidates, max(96, self.max_candidates)),
            hmm=HMMConfig(**hmm_values),
            allow_geometric_fallback=self.allow_geometric_fallback,
        )
