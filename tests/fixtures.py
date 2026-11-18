"""Small directed graph fixtures shared by unit tests."""
import numpy as np
from src.road_network import RoadNetwork


def network_fixture():
    nodes=np.array([[0.0,0.0],[0.001,0.0],[0.002,0.0],[0.002,0.001],[1.0,1.0],[1.001,1.0]])
    arcs=np.array([[0,1],[1,2],[2,3],[4,5]])
    return RoadNetwork(nodes,arcs,"synthetic")
