"""Snapshot streaming health, decisions and case workload as JSON."""
import json
from python.common import ROOT, write_json
from python.monitoring.operations import OperationsRepository

def main():
    report = OperationsRepository().summary()
    pipeline = report["pipeline"]
    report["alerts"] = [name for name, triggered in {
        "OUTBOX_BACKLOG": pipeline["pending_outbox"] > 0,
        "PIPELINE_RECONCILIATION": pipeline["persisted_transactions"] != pipeline["persisted_decisions"],
        "DEAD_LETTER_PRESENT": pipeline["dead_letters"] > 0,
    }.items() if triggered]
    write_json(report, ROOT / "reports/operational_report.json")
    print(json.dumps(report, indent=2, default=str))

if __name__ == "__main__": main()
