"""Register candidates and require explicit, gated approval for champion promotion."""
import argparse,hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
import psycopg
from python.common import ROOT

def sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()

def register(version,stage,artifact,metadata,approved_by=None):
    dsn=os.getenv("POSTGRES_DSN","postgresql://fraud:fraud@localhost:5433/fraud")
    approved_at=datetime.now(timezone.utc) if stage=="CHAMPION" else None
    if stage=="CHAMPION" and not approved_by: raise ValueError("CHAMPION registration requires --approved-by")
    with psycopg.connect(dsn) as connection:
        connection.execute("""INSERT INTO fraud.model_registry
            (model_version,stage,artifact_path,artifact_sha256,offline_metrics,approved_by,approved_at)
            VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s)
            ON CONFLICT (model_version) DO UPDATE SET artifact_path=excluded.artifact_path,
            artifact_sha256=excluded.artifact_sha256,offline_metrics=excluded.offline_metrics""",
            (version,stage,str(artifact),sha256(artifact),json.dumps(metadata),approved_by,approved_at))

def promote(version,approved_by):
    dsn=os.getenv("POSTGRES_DSN","postgresql://fraud:fraud@localhost:5433/fraud")
    with psycopg.connect(dsn) as connection:
        candidate=connection.execute("SELECT offline_metrics FROM fraud.model_registry WHERE model_version=%s AND stage='CHALLENGER'",(version,)).fetchone()
        if not candidate: raise ValueError("Version is not a registered CHALLENGER")
        champion=connection.execute("SELECT offline_metrics FROM fraud.model_registry WHERE stage='CHAMPION'").fetchone()
        candidate_metrics=candidate[0]
        if champion:
            champion_metrics=champion[0]
            if candidate_metrics.get("pr_auc",0) < champion_metrics.get("pr_auc",0)-.02:
                raise ValueError("Promotion blocked: challenger PR-AUC regression exceeds 0.02")
            if candidate_metrics.get("brier_score",1) > champion_metrics.get("brier_score",1)+.005:
                raise ValueError("Promotion blocked: challenger Brier score regression exceeds 0.005")
        latest=connection.execute("SELECT alerts FROM fraud.model_monitoring_runs ORDER BY created_at DESC LIMIT 1").fetchone()
        blocking={"PIPELINE_RECONCILIATION","CALIBRATION_DEGRADATION","PR_AUC_DEGRADATION",
                  "INSUFFICIENT_MATURED_LABELS","SINGLE_CLASS_LABEL_WINDOW"}
        blocking.add("INSUFFICIENT_FRAUD_LABELS")
        alerts=set(latest[0] if latest else [])
        if alerts & blocking: raise ValueError(f"Promotion blocked by monitoring alerts: {sorted(alerts & blocking)}")
        connection.execute("UPDATE fraud.model_registry SET stage='ARCHIVED' WHERE stage='CHAMPION'")
        connection.execute("""UPDATE fraud.model_registry SET stage='CHAMPION',approved_by=%s,approved_at=now()
            WHERE model_version=%s""",(approved_by,version))
    print(f"[PROMOTED] {version} approved_by={approved_by}")

def main():
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True)
    add=sub.add_parser("register"); add.add_argument("--version"); add.add_argument("--stage",choices=["CHAMPION","CHALLENGER"],default="CHALLENGER"); add.add_argument("--artifact",type=Path,default=ROOT/"artifacts/model.joblib"); add.add_argument("--metadata",type=Path,default=ROOT/"artifacts/model_metadata.json"); add.add_argument("--approved-by")
    pro=sub.add_parser("promote"); pro.add_argument("--version",required=True); pro.add_argument("--approved-by",required=True)
    args=parser.parse_args()
    if args.command=="promote": return promote(args.version,args.approved_by)
    metadata=json.loads(args.metadata.read_text(encoding="utf-8")); version=args.version or metadata["model_version"]
    register(version,args.stage,args.artifact,metadata,args.approved_by); print(f"[REGISTERED] {version} stage={args.stage}")

if __name__=="__main__": main()
