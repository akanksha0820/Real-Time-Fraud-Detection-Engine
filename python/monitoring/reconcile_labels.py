"""Load delayed confirmed labels separately from live transaction events."""
import argparse, os
from datetime import datetime, timezone
import pandas as pd
import psycopg
from python.common import ROOT

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--delay-days",type=int,default=30)
    parser.add_argument("--as-of",default=datetime.now(timezone.utc).date().isoformat())
    args=parser.parse_args()
    dsn=os.getenv("POSTGRES_DSN","postgresql://fraud:fraud@localhost:5433/fraud")
    with psycopg.connect(dsn) as connection:
        ids={row[0] for row in connection.execute("SELECT transaction_id FROM fraud.transactions")}
    source=pd.read_parquet(ROOT/"data/raw/transactions.parquet",
                           columns=["transaction_id","event_timestamp","is_fraud","fraud_scenario"])
    source=source[source.transaction_id.isin(ids)].copy()
    source["event_timestamp"]=pd.to_datetime(source.event_timestamp,utc=True)
    source["observed_at"]=source.event_timestamp+pd.Timedelta(days=args.delay_days)
    eligible=source[source.observed_at<=pd.Timestamp(args.as_of,tz="UTC")]
    inserted=0
    with psycopg.connect(dsn) as connection:
        for row in eligible.itertuples(index=False):
            cursor=connection.execute("""INSERT INTO fraud.fraud_labels
                (transaction_id,is_fraud,fraud_scenario,label_source,observed_at,confidence)
                VALUES (%s,%s,%s,'SIMULATION',%s,1.0) ON CONFLICT (transaction_id) DO NOTHING""",
                (row.transaction_id,bool(row.is_fraud),row.fraud_scenario,row.observed_at.to_pydatetime()))
            inserted+=cursor.rowcount
    print(f"[LABELS] eligible={len(eligible):,} inserted={inserted:,} delayed_days={args.delay_days}")

if __name__=="__main__": main()
