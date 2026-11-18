import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from model_predictor import CandidateScorer, ordered_route_coordinates
import json
class PipelineTests(unittest.TestCase):
    def test_candidate_id_and_probability_output(self):
        model=ExtraTreesClassifier(n_estimators=4,random_state=1).fit([[0,0,0,1,0,1,0],[1,0,0,1,0,1,0]],[0,1])
        path=ROOT/"models"/"test_model.joblib"
        import joblib
        joblib.dump(model,path)
        try:
            row={"observation_id":8,"candidate_id":42,"distance_m":1,"heading_cos":0,"heading_sin":0,"length_m":1,"step_distance_m":0,"turn_cos":1,"turn_sin":0}
            result=CandidateScorer(path).score_candidates(pd.DataFrame([row]))
            self.assertEqual(result.candidate_id.tolist(),[42])
            self.assertTrue(result.score.between(0,1).all())
        finally: path.unlink(missing_ok=True)
    def test_route_geometry_is_latitude_longitude_and_ordered(self):
        coords=ordered_route_coordinates(np.array([[10.,20.],[11.,21.],[12.,22.]]),np.array([[0,1],[1,2]]),[0,1])
        self.assertEqual(coords,[[20.,10.],[21.,11.],[22.,12.]])
    def test_trip_split_does_not_leak_groups(self):
        from sklearn.model_selection import GroupShuffleSplit
        groups=np.array(["a","a","b","b","c","c"])
        tr,va=next(GroupShuffleSplit(test_size=.33,random_state=42).split(np.zeros((6,1)),groups=groups))
        self.assertFalse(set(groups[tr]) & set(groups[va]))
    def test_exported_model_scores_a_held_out_trip(self):
        metadata=json.loads((ROOT/"models"/"model_metadata.json").read_text())
        held_out=metadata["validation_trips"][0]
        cache=pd.read_csv(ROOT/"reports"/"features"/"model_candidates_train_v2.csv",dtype={"trip_id":str})
        rows=cache[cache.trip_id.eq(held_out)].head(4).copy()
        scorer=CandidateScorer(ROOT/"models"/"best_model.joblib")
        scores=scorer.score_candidates(rows)
        self.assertEqual(scores.candidate_id.tolist(),rows.candidate_id.tolist())
        self.assertTrue(scores.score.between(0,1).all())
if __name__=="__main__": unittest.main()
