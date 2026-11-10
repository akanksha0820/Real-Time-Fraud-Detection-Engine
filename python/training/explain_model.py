"""Generate global SHAP importance for the trained XGBoost component."""

import pandas as pd
import shap

from python.common import ROOT
from python.training.train_model import FEATURES
import joblib


def main() -> None:
    bundle = joblib.load(ROOT / "artifacts/model.joblib")
    data = pd.read_parquet(ROOT / "data/processed/transaction_features.parquet").tail(5000)
    preprocessor = bundle["pipeline"].named_steps["preprocessor"]
    model = bundle["pipeline"].named_steps["classifier"]
    matrix = preprocessor.transform(data[FEATURES])
    sample = matrix[:1000]
    values = shap.TreeExplainer(model).shap_values(sample)
    names = preprocessor.get_feature_names_out()
    importance = pd.DataFrame({"feature": names, "mean_abs_shap": abs(values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(ROOT / "reports/shap_feature_importance.csv", index=False)
    print(importance.head(20).to_string(index=False))
    print("[SAVED] reports/shap_feature_importance.csv")


if __name__ == "__main__": main()
