"""Measure realized model performance only on matured delayed labels."""
import argparse,json,os
import numpy as np
import pandas as pd
import psycopg
from psycopg.rows import dict_row
from sklearn.metrics import average_precision_score,brier_score_loss,log_loss,roc_auc_score
from python.common import ROOT,write_json

def expected_calibration_error(labels, probability, bins=10):
    frame=pd.DataFrame({"y":labels,"p":probability})
    frame["bin"]=pd.cut(frame.p,np.linspace(0,1,bins+1),include_lowest=True)
    grouped=frame.groupby("bin",observed=False).agg(n=("y","size"),actual=("y","mean"),predicted=("p","mean")).dropna()
    return float(((grouped.n/len(frame))*(grouped.actual-grouped.predicted).abs()).sum())

def evaluate(frame, minimum_labels=200, minimum_fraud_labels=25):
    alerts=[]
    fraud_labels=int(frame.is_fraud.sum()) if len(frame) else 0
    metrics={"labeled_transactions":len(frame),"fraud_labels":fraud_labels,
             "fraud_rate":float(frame.is_fraud.mean()) if len(frame) else None}
    if len(frame)<minimum_labels:
        alerts.append("INSUFFICIENT_MATURED_LABELS")
    if fraud_labels<minimum_fraud_labels:
        alerts.append("INSUFFICIENT_FRAUD_LABELS")
    if len(frame) and frame.is_fraud.nunique()==2:
        y=frame.is_fraud.astype(int); p=frame.fraud_probability.astype(float)
        metrics.update({"pr_auc":float(average_precision_score(y,p)),"roc_auc":float(roc_auc_score(y,p)),
                        "brier_score":float(brier_score_loss(y,p)),"log_loss":float(log_loss(y,p,labels=[0,1])),
                        "ece":expected_calibration_error(y,p),
                        "investigation_capture_rate":float(frame.loc[y.eq(1),"decision"].isin(["REVIEW","BLOCK"]).mean())})
    else: alerts.append("SINGLE_CLASS_LABEL_WINDOW")
    metrics["metric_status"]="VALIDATED" if not alerts else "PROVISIONAL"
    return metrics,alerts

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--minimum-labels",type=int,default=200); parser.add_argument("--minimum-fraud-labels",type=int,default=25); parser.add_argument("--no-mlflow",action="store_true"); args=parser.parse_args()
    dsn=os.getenv("POSTGRES_DSN","postgresql://fraud:fraud@localhost:5433/fraud")
    query="""SELECT t.transaction_id,t.event_timestamp,t.payment_channel,t.amount_inr,
        d.fraud_probability,d.decision,d.model_version,l.is_fraud,l.observed_at
        FROM fraud.transactions t JOIN fraud.risk_decisions d USING(transaction_id)
        JOIN fraud.fraud_labels l USING(transaction_id) ORDER BY t.event_timestamp"""
    with psycopg.connect(dsn) as connection:
        cursor=connection.execute(query)
        frame=pd.DataFrame(cursor.fetchall(),columns=[column.name for column in cursor.description])
    metrics,alerts=evaluate(frame,args.minimum_labels,args.minimum_fraud_labels)
    metadata=json.loads((ROOT/"artifacts/model_metadata.json").read_text(encoding="utf-8"))
    if "pr_auc" in metrics and metrics["pr_auc"] < metadata.get("pr_auc",0)-.10: alerts.append("PR_AUC_DEGRADATION")
    if "brier_score" in metrics and metrics["brier_score"] > max(.02,metadata.get("brier_score",0)*2): alerts.append("CALIBRATION_DEGRADATION")
    segments={}
    for name,part in frame.groupby("payment_channel"):
        if len(part)>=20:
            segment_metrics,segment_alerts=evaluate(part,20,min(5,args.minimum_fraud_labels))
            segments[name]={"metrics":segment_metrics,"alerts":segment_alerts}
    report={"model_version":metadata.get("model_version","unknown"),"window_start":str(frame.event_timestamp.min()) if len(frame) else None,
            "window_end":str(frame.event_timestamp.max()) if len(frame) else None,"metrics":metrics,"segments":segments,"alerts":sorted(set(alerts))}
    with psycopg.connect(dsn) as connection:
        connection.execute("""INSERT INTO fraud.model_monitoring_runs
            (model_version,window_start,window_end,labeled_transactions,metrics,segment_metrics,alerts)
            VALUES (%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb)""",
            (report["model_version"],report["window_start"],report["window_end"],len(frame),json.dumps(metrics),json.dumps(segments),json.dumps(report["alerts"])))
    write_json(report,ROOT/"reports/production_performance.json")
    if not args.no_mlflow:
        try:
            import mlflow
            tracking_uri=f"sqlite:///{(ROOT/'mlflow.db').as_posix()}"
            mlflow.set_tracking_uri(tracking_uri); mlflow.set_experiment("fraud-production-monitoring")
            with mlflow.start_run(run_name=report["model_version"]):
                mlflow.log_metrics({k:v for k,v in metrics.items() if isinstance(v,(int,float)) and v is not None}); mlflow.log_dict(report,"production_performance.json")
        except ImportError: print("[SKIP] MLflow is not installed; PostgreSQL report was still persisted")
        except Exception as exc: print(f"[MLFLOW WARNING] {exc}; PostgreSQL report was still persisted")
    print(json.dumps(report,indent=2,default=str))

if __name__=="__main__": main()
