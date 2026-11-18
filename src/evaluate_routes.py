"""Evaluate predictions first, then read reference routes for post-hoc metrics."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.config import BackendSettings
from src.route_reconstruction import RouteReconstructionService, ReconstructionError
from src.evaluation_metrics import route_sequence_metrics

def main() -> None:
    settings=BackendSettings.from_env()
    service=RouteReconstructionService(settings.pipeline_config())
    records=[]
    for option in service.list_trips():
        trip_id=option["trip_id"]
        try:
            # Ground truth is not read until the inference call is complete.
            result=service.reconstruct(trip_id)
            predicted=result["route_arc_ids"]
            metrics=result.get("evaluation")
            if not metrics:
                raise ValueError("Post-prediction ground-truth comparison is unavailable")
            records.append({"trip_id":trip_id,"status":"success","route_edge_precision":metrics["route_edge_precision"],"route_edge_recall":metrics["route_edge_recall"],"route_edge_f1":metrics["route_edge_f1"],"sequence_similarity":metrics["sequence_similarity"],"predicted_route_arcs":metrics["predicted_route_arc_count"],"reference_route_arcs":metrics["ground_truth_route_arc_count"],"inference_time_ms":result["inference_time_ms"],"scoring_mode":result["scoring_mode"],"warnings":"; ".join(result["warnings"]),"error":""})
        except (ReconstructionError,FileNotFoundError,ValueError) as exc:
            records.append({"trip_id":trip_id,"status":"failed","route_edge_precision":0.0,"route_edge_recall":0.0,"route_edge_f1":0.0,"sequence_similarity":0.0,"predicted_route_arcs":0,"reference_route_arcs":0,"inference_time_ms":np.nan,"scoring_mode":service.scoring_mode,"warnings":"","error":str(exc)})
    report=pd.DataFrame(records);out=ROOT/"reports";out.mkdir(exist_ok=True);report.to_csv(out/"route_evaluation.csv",index=False)
    successful=report[report.status.eq("success")]
    long_count=int(successful.warnings.str.count("Used a connected but physically long road-network transition").sum()) if len(successful) else 0
    summary={"evaluation_trips":len(report),"successful_reconstructions":len(successful),"failed_trip_ids":report.loc[report.status.eq("failed"),"trip_id"].tolist(),"success_rate":float(len(successful)/len(report)) if len(report) else 0.0,"mean_route_edge_precision":float(successful.route_edge_precision.mean()) if len(successful) else None,"mean_route_edge_recall":float(successful.route_edge_recall.mean()) if len(successful) else None,"mean_route_edge_f1":float(successful.route_edge_f1.mean()) if len(successful) else None,"mean_sequence_similarity":float(successful.sequence_similarity.mean()) if len(successful) else None,"mean_inference_time_ms":float(successful.inference_time_ms.mean()) if len(successful) else None,"long_connected_fallback_transitions":long_count,"scoring_mode":service.scoring_mode}
    (out/"route_evaluation_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
    print(report.to_string(index=False));print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
