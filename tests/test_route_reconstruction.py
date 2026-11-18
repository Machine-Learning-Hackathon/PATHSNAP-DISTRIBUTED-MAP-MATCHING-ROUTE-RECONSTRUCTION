import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.candidate_generation import CandidateConfig
from src.hmm import HMMConfig
from src.route_reconstruction import PipelineConfig,RouteReconstructionService

class GeometryScorer:
    def score_candidates(self,records):
        frame=pd.DataFrame(records)
        # Transparent synthetic score for integration testing only.
        return frame[["observation_id","candidate_id"]].assign(score=np.exp(-frame.distance_m/20))

class ReconstructionTests(unittest.TestCase):
    def test_complete_synthetic_trip_pipeline_without_ground_truth(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/"Student folder"/"student_test"/"syn001";folder.mkdir(parents=True)
            np.savetxt(folder/"syn001.nodes",[[0,0],[.001,0],[.002,0]])
            np.savetxt(folder/"syn001.arcs",[[0,1],[1,2]],fmt="%d")
            np.savetxt(folder/"syn001.track",[[.0003,0,1],[.0007,0,2],[.0013,0,3],[.0017,0,4]])
            # No .route file is present or needed by prediction.
            cfg=PipelineConfig(root,Path("unused"),Path("unused"),candidate=CandidateConfig(80,200,4,32),hmm=HMMConfig())
            service=RouteReconstructionService(cfg,scorer=GeometryScorer())
            result=service.reconstruct("syn001")
            self.assertEqual(result["status"],"success")
            self.assertEqual(len(result["matched_arc_ids"]),4)
            self.assertGreaterEqual(len(result["route_arc_ids"]),1)
            self.assertTrue(all(len(point)==2 for point in result["predicted_route"]))
            self.assertEqual(result["decoder"],"HMM + Viterbi")

if __name__=="__main__":unittest.main()
