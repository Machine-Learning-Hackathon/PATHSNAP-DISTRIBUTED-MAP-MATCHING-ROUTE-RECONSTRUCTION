import unittest
import numpy as np
from tests.fixtures import network_fixture
from src.road_network import RoadNetwork,haversine_m

class RoadNetworkTests(unittest.TestCase):
    def test_directed_graph_keeps_original_ids(self):
        graph=network_fixture()
        self.assertEqual(graph.arcs[2].source,2)
        self.assertTrue(graph.connected(0,1))
        self.assertFalse(graph.connected(1,0))
        self.assertFalse(graph.connected(0,3))
    def test_distance_and_geometry_coordinates(self):
        self.assertAlmostEqual(haversine_m(0,0,0,.001),111.2,delta=.5)
        self.assertEqual(network_fixture().arc_geometry(0),[[0.0,0.0],[0.0,0.001]])
    def test_invalid_arc_is_skipped_without_renumbering(self):
        graph=RoadNetwork(np.array([[0.,0.],[.001,0.]]),np.array([[0,1],[1,9]]))
        self.assertIn(0,graph.arcs)
        self.assertNotIn(1,graph.arcs)

if __name__=="__main__":unittest.main()
