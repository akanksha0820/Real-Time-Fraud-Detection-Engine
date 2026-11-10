"""Train an unsupervised Isolation Forest as a novel-pattern investigation signal."""

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from python.common import ROOT, load_settings
from python.training.train_model import NUMERIC_FEATURES


def main() -> None:
    settings = load_settings()
    data = pd.read_parquet(ROOT / "data/processed/transaction_features.parquet")
    train = data[pd.to_datetime(data.event_timestamp) < pd.Timestamp(settings["model"]["validation_start"])]
    clean = train.loc[train.is_fraud.eq(0), NUMERIC_FEATURES].fillna(0)
    model = IsolationForest(n_estimators=250, contamination="auto", random_state=settings["seed"], n_jobs=-1)
    model.fit(clean)
    joblib.dump({"model": model, "features": NUMERIC_FEATURES}, ROOT / "artifacts/anomaly_model.joblib")
    print("[SAVED] artifacts/anomaly_model.joblib")


if __name__ == "__main__": main()
