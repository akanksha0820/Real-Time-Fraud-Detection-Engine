from fastapi.testclient import TestClient
import api.main as api_module
from python.common import ROOT

class FakeOperations:
    def summary(self):
        return {"pipeline":{"consumed_messages":10,"persisted_transactions":5,
                "persisted_decisions":5,"pending_outbox":0,"dead_letters":0},
                "decisions":[],"cases":[]}
    def review_queue(self, limit=100): return [{"case_id":1,"priority":"HIGH"}]
    def merchant_risk(self, limit=20): return []
    def model_governance(self): return {"registry":[],"monitoring_runs":[]}
    def update_case(self, case_id, **values): return {"case_id":case_id,**values}

def test_operations_api_contract(monkeypatch):
    monkeypatch.setattr(api_module,"operations",lambda:FakeOperations())
    client=TestClient(api_module.app)
    assert client.get("/operations/summary").json()["pipeline"]["pending_outbox"] == 0
    assert client.get("/operations/review-queue").json()[0]["priority"] == "HIGH"
    assert client.get("/operations/model-governance").status_code == 200
    response=client.patch("/operations/cases/1",json={"status":"RESOLVED","outcome":"FRAUD"})
    assert response.status_code == 200 and response.json()["status"] == "RESOLVED"

def test_operations_schema_contains_reconciliation_views():
    sql=(ROOT/"sql/schema.sql").read_text(encoding="utf-8")
    assert "fraud.v_pipeline_health" in sql
    assert "fraud.v_review_queue" in sql
    assert "fraud.investigation_cases" in sql
