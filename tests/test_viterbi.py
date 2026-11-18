import unittest
import numpy as np
from src.viterbi import viterbi_decode

class ViterbiTests(unittest.TestCase):
    def test_initialization_recurrence_and_backtracking(self):
        emissions=[np.log([.9,.1]),np.log([.1,.9]),np.log([.1,.9])]
        transitions=[np.log([[.99,.01],[.01,.99]]),np.log([[.99,.01],[.01,.99]])]
        result=viterbi_decode(emissions,transitions)
        self.assertEqual(result.status,"success")
        self.assertEqual(result.state_indices,[1,1,1])
    def test_single_observation(self):
        result=viterbi_decode([np.log([.2,.8])],[])
        self.assertEqual(result.state_indices,[1])
    def test_empty_or_disconnected_path_fails_explicitly(self):
        self.assertEqual(viterbi_decode([],[]).status,"failed")
        self.assertEqual(viterbi_decode([np.array([0.]),np.array([0.])],[np.array([[-np.inf]])]).status,"failed")
        self.assertEqual(viterbi_decode([np.array([])],[]).status,"failed")

if __name__=="__main__":unittest.main()
