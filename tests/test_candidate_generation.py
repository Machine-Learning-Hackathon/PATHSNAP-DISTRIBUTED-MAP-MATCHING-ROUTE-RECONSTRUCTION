import unittest
import numpy as np
from tests.fixtures import network_fixture
from src.candidate_generation import CandidateConfig,generate_candidates

class CandidateGenerationTests(unittest.TestCase):
    def test_candidates_keep_order_and_original_arc_ids(self):
        track=np.array([[.0005,.00002,1.],[.0015,.00002,2.]])
        result=generate_candidates(track,network_fixture(),CandidateConfig(40,350,4,32))
        self.assertEqual(len(result),2)
        self.assertEqual([c.observation_id for c in result[0]],[0]*len(result[0]))
        self.assertEqual(result[0][0].arc_id,0)
        self.assertTrue(all(c.distance_m<40 for c in result[0]))
    def test_empty_candidates_past_explicit_fallback_radius(self):
        track=np.array([[.5,.5,1.]])
        result=generate_candidates(track,network_fixture(),CandidateConfig(20,30,3,32))
        self.assertEqual(result,[[]])
    def test_invalid_coordinates_fail_clearly(self):
        with self.assertRaisesRegex(ValueError,"invalid longitude"):
            generate_candidates(np.array([[181.,0.,1.]]),network_fixture())

if __name__=="__main__":unittest.main()
