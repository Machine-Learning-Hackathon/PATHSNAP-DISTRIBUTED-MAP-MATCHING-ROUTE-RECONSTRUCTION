"""Load the candidate scorer used by the HMM/route decoder."""
from pathlib import Path
import json
import joblib
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
FEATURES=["distance_m","heading_cos","heading_sin","length_m","step_distance_m","turn_cos","turn_sin"]

class CandidateScorer:
    """Adapter around Person 2's binary candidate classifier."""
    def __init__(self, model_path=None, schema_path=None):
        model_file=Path(model_path or ROOT/"models"/"best_model.joblib")
        schema_file=Path(schema_path or ROOT/"models"/"feature_schema.json")
        if not model_file.is_file():
            raise FileNotFoundError(f"Required trained candidate model not found: {model_file}")
        if not schema_file.is_file():
            raise FileNotFoundError(f"Required feature schema not found: {schema_file}")
        self.feature_schema=json.loads(schema_file.read_text(encoding="utf8"))
        self.features=list(self.feature_schema.get("features", []))
        if self.features != FEATURES:
            raise ValueError(f"Model feature schema mismatch: expected {FEATURES}, got {self.features}")
        self.positive_class=1
        self.model=joblib.load(model_file)
        classes=list(getattr(self.model,"classes_",[]))
        if self.positive_class not in classes:
            raise ValueError(f"Model has no positive candidate class {self.positive_class}: {classes}")
        self.positive_column=classes.index(self.positive_class)
        if not hasattr(self.model,"predict_proba"):
            raise TypeError("Candidate model must expose predict_proba for HMM emissions")
    def score_candidates(self, records):
        """Score candidate rows and retain observation and original arc identifiers."""
        frame=records.copy() if isinstance(records,pd.DataFrame) else pd.DataFrame(records)
        missing=set(self.features)-set(frame.columns)
        if missing: raise ValueError(f"Missing model features: {sorted(missing)}")
        if "candidate_id" not in frame: raise ValueError("candidate_id is required")
        if frame.empty:
            out=frame[[c for c in ["observation_id","candidate_id"] if c in frame]].copy()
            out["score"]=pd.Series(dtype=float)
            return out
        values=frame[self.features].apply(pd.to_numeric,errors="coerce")
        if values.isnull().any().any(): raise ValueError("Candidate feature records contain non-numeric or missing values")
        probs=self.model.predict_proba(values)[:,self.positive_column]
        if not pd.Series(probs).between(0,1).all(): raise ValueError("Model returned invalid candidate probabilities")
        out=frame[[c for c in ["observation_id","candidate_id"] if c in frame]].copy()
        out["score"]=probs
        return out


def ordered_route_coordinates(nodes_lonlat, arcs, ordered_arc_ids):
    """Return route geometry as ordered [latitude, longitude] coordinates."""
    coords=[]
    for arc_id in ordered_arc_ids:
        source,dest=arcs[int(arc_id)]
        p0=[float(nodes_lonlat[source][1]),float(nodes_lonlat[source][0])]
        p1=[float(nodes_lonlat[dest][1]),float(nodes_lonlat[dest][0])]
        if not coords or coords[-1] != p0: coords.append(p0)
        if coords[-1] != p1: coords.append(p1)
    return coords
