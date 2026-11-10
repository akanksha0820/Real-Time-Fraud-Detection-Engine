import pandas as pd
from python.inference.state import InMemoryStateStore

def test_state_store_contract_commit_and_idempotency():
    state = InMemoryStateStore()
    record = (pd.Timestamp("2026-01-01"), 100.0, 10.0, 20.0, "D1", "M1")
    with state.lock():
        state.commit("C1", "M1", "D1", record, [], [])
        state.save_decision("T1", {"decision": "ALLOW"})
    assert state.get_customer_events("C1") == [record]
    assert state.get_merchant_events("M1") == [record[0]]
    assert state.get_last_customer_event("C1") == record
    assert state.get_customer_device_count("C1", "D1") == 1
    assert state.get_customer_merchant_count("C1", "M1") == 1
    assert state.get_device_global_count("D1") == 1
    assert state.get_decision("T1") == {"decision": "ALLOW"}
