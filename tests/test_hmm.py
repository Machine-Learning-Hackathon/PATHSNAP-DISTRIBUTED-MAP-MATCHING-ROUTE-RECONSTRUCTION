import unittest
import numpy as np
from tests.fixtures import network_fixture
from src.candidate_generation import Candidate
from src.hmm import emission_log_probabilities,transition_log_matrix

class HMMTests(unittest.TestCase):
    def test_emissions_are_normalized_in_log_space(self):
        values=emission_log_probabilities([.25,.75])
        self.assertAlmostEqual(float(np.exp(values).sum()),1.0)
        self.assertTrue(np.isfinite(values).all())
    def test_connected_transition_has_finite_score(self):
        graph=network_fixture()
        a=Candidate(0,0,0,1,0,111,55,1,0,.5)
        b=Candidate(1,1,0,1,0,111,55,1,0,.5)
        matrix=transition_log_matrix([a],[b],np.array([55.,0]),np.array([166.,0]),2.,graph)
        self.assertTrue(np.isfinite(matrix[0,0]))
    def test_disconnected_transition_is_not_fabricated(self):
        graph=network_fixture()
        a=Candidate(0,0,0,1,0,111,0,1,0,.5)
        b=Candidate(1,3,0,1,0,111,0,1,0,.5)
        matrix=transition_log_matrix([a],[b],np.array([0.,0]),np.array([111.,0]),2.,graph)
        self.assertTrue(np.isneginf(matrix[0,0]))
    def test_long_connector_recovery_is_explicit_and_still_connected(self):
        graph=network_fixture()
        a=Candidate(0,0,0,1,0,111,0,1,0,0.0)
        b=Candidate(1,1,0,1,0,111,0,1,0,1.0)
        gps0=np.array([0.,0.]);gps1=np.array([222.,0.])
        strict=transition_log_matrix([a],[b],gps0,gps1,1.,graph)
        expanded=transition_log_matrix([a],[b],gps0,gps1,1.,graph,allow_long_connectors=True)
        self.assertTrue(np.isneginf(strict[0,0]))
        self.assertTrue(np.isfinite(expanded[0,0]))

if __name__=="__main__":unittest.main()
