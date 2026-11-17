"""Real cross-validated classifier predicting the primary endpoint
from trial covariates -- the real modeling step underneath the
resume's AUC claim."""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

FEATURES = ["age", "baseline_severity", "biomarker_a", "biomarker_b", "biomarker_c", "biomarker_d",
            "comorbidity_count", "bmi"]


def train_and_evaluate(df: pd.DataFrame, cv: int = 5, seed: int = 937) -> dict:
    X = df[FEATURES].values
    y = df["primary_endpoint_responder"].values

    model = GradientBoostingClassifier(random_state=seed, n_estimators=150, max_depth=3)
    probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, probs)

    return {"auc": round(float(auc), 4), "n_patients": len(df), "n_features": len(FEATURES)}
