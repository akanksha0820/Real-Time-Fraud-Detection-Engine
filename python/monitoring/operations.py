"""Operational queries and analyst case commands backed by PostgreSQL."""
from __future__ import annotations

import os


class OperationsRepository:
    def __init__(self, dsn=None):
        import psycopg
        from psycopg.rows import dict_row
        self.psycopg, self.dict_row = psycopg, dict_row
        self.dsn = dsn or os.getenv("POSTGRES_DSN", "postgresql://fraud:fraud@localhost:5433/fraud")

    def _rows(self, query, params=()):
        with self.psycopg.connect(self.dsn, row_factory=self.dict_row) as connection:
            return connection.execute(query, params).fetchall()

    def ping(self):
        return self._rows("SELECT 1 AS ready")[0]["ready"] == 1

    def summary(self):
        pipeline = self._rows("SELECT * FROM fraud.v_pipeline_health")[0]
        decisions = self._rows("""SELECT decision,COUNT(*) AS transactions,
            SUM(t.amount_inr) AS amount_inr FROM fraud.risk_decisions d
            JOIN fraud.transactions t USING(transaction_id) GROUP BY decision ORDER BY decision""")
        cases = self._rows("""SELECT status,COUNT(*) AS cases FROM fraud.investigation_cases
            GROUP BY status ORDER BY status""")
        return {"pipeline": pipeline, "decisions": decisions, "cases": cases}

    def hourly_decisions(self, limit=168):
        return self._rows("""SELECT metric_hour,decision,transactions,amount_inr,average_probability
            FROM fraud.v_hourly_decision_metrics ORDER BY metric_hour DESC LIMIT %s""", (limit,))

    def review_queue(self, limit=100):
        return self._rows("""SELECT * FROM fraud.v_review_queue
            ORDER BY CASE priority WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 WHEN 'MEDIUM' THEN 3 ELSE 4 END,
            created_at LIMIT %s""", (limit,))

    def merchant_risk(self, limit=20):
        return self._rows("""SELECT * FROM fraud.v_merchant_risk_concentration
            ORDER BY reviews DESC,blocks DESC,amount_inr DESC LIMIT %s""", (limit,))

    def model_governance(self):
        registry = self._rows("""SELECT model_version,stage,registered_at,approved_by,approved_at,
            offline_metrics FROM fraud.model_registry ORDER BY registered_at DESC""")
        monitoring = self._rows("""SELECT model_version,window_start,window_end,labeled_transactions,
            metrics,segment_metrics,alerts,created_at FROM fraud.model_monitoring_runs
            ORDER BY created_at DESC LIMIT 10""")
        return {"registry": registry, "monitoring_runs": monitoring}

    def update_case(self, case_id, status, assigned_to=None, outcome=None, notes=None):
        resolved = status in {"RESOLVED", "CLOSED"}
        rows = self._rows("""UPDATE fraud.investigation_cases SET status=%s,assigned_to=%s,
            outcome=%s,notes=%s,updated_at=now(),resolved_at=CASE WHEN %s THEN now() ELSE NULL END
            WHERE case_id=%s RETURNING *""", (status, assigned_to, outcome, notes, resolved, case_id))
        if not rows:
            raise KeyError(case_id)
        return rows[0]
