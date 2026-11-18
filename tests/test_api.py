import unittest
from fastapi.testclient import TestClient
from backend.app import create_app
from backend.config import BackendSettings

class FakeService:
    scoring_mode="trained_model"
    def health_status(self):return {"status":"ok","scoring_mode":self.scoring_mode,"model_loaded":True,"dataset_available":True,"evaluation_trip_count":10,"road_network_files_available":10,"ground_truth_files_available":10,"road_network_loading":"loaded on demand per selected trip","ready_for_inference":True}
    def list_trips(self):return [{"trip_id":f"{i:08d}","label":f"Evaluation trip {i+1:02d}"} for i in range(10)]
    def reconstruct(self,trip_id):
        return {"trip_id":trip_id,"gps_trace":[[1.,2.]],"predicted_route":[[1.,2.],[1.1,2.1]],"road_network":[],"inference_time_ms":2.3,"decoder":"HMM + Viterbi","status":"success","scoring_mode":"trained_model","matched_arc_ids":[3],"route_arc_ids":[3],"matched_coordinates":[[1.,2.]],"candidate_count":1,"warnings":[],"evaluation":{"route_edge_precision":.9,"route_edge_recall":.8,"route_edge_f1":.847,"sequence_similarity":.75,"predicted_route_arc_count":10,"ground_truth_route_arc_count":11,"comparison_available":True}}

class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client=TestClient(create_app(BackendSettings.from_env(),FakeService()))
        self.client.__enter__()
    def tearDown(self):
        self.client.__exit__(None,None,None)
    def test_health_and_evaluation_trip_list(self):
        health=self.client.get("/health").json()
        self.assertEqual(health["status"],"ok")
        self.assertTrue(health["model_loaded"])
        self.assertTrue(health["ready_for_inference"])
        trips=self.client.get("/api/evaluation/trips").json()["trips"]
        self.assertEqual(len(trips),10)
        self.assertEqual(trips[0]["trip_id"],"00000000")
    def test_predict_contract_and_lat_lon_order(self):
        response=self.client.post("/api/evaluation/predict",json={"trip_id":"00000000"})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json()["predicted_route"],[[1.,2.],[1.1,2.1]])
        self.assertAlmostEqual(response.json()["evaluation"]["route_edge_f1"],.847)
    def test_unknown_trip_and_path_traversal_rejected(self):
        self.assertEqual(self.client.post("/api/evaluation/predict",json={"trip_id":"99999999"}).status_code,404)
        self.assertEqual(self.client.post("/api/evaluation/predict",json={"trip_id":"../secret"}).status_code,422)

if __name__=="__main__":unittest.main()
