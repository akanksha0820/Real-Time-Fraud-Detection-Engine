from python.inference.state import InMemoryStateStore


def test_in_memory_state_reset_clears_all_state():
    state = InMemoryStateStore()
    state.device_global_counts["D1"] = 2
    state.decisions["T1"] = {"decision": "ALLOW"}
    state.customer_events["C1"].append((1,))
    state.reset()
    assert not state.device_global_counts
    assert not state.decisions
    assert not state.customer_events
