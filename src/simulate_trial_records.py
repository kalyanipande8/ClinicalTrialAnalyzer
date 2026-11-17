"""Simulate 2,600 clinical trial patient records with 10 covariates
and a primary binary endpoint driven by a known subset of true
predictive covariates -- the ground truth the statistical-analysis
and endpoint-prediction modules must recover."""
from __future__ import annotations

import numpy as np
import pandas as pd

N_PATIENTS = 2600
SEED = 937
TRUE_PREDICTORS = ["biomarker_a", "biomarker_b", "baseline_severity"]


def simulate(n: int = N_PATIENTS, seed: int = SEED):
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "patient_id": [f"PT-{i:05d}" for i in range(n)],
        "age": rng.integers(18, 85, n), "sex": rng.choice(["M", "F"], n),
        "treatment_arm": rng.choice(["Active", "Placebo"], n),
        "baseline_severity": np.round(rng.normal(50, 12, n), 2),
        "biomarker_a": np.round(rng.normal(0, 1, n), 3), "biomarker_b": np.round(rng.normal(0, 1, n), 3),
        "biomarker_c": np.round(rng.normal(0, 1, n), 3), "biomarker_d": np.round(rng.normal(0, 1, n), 3),
        "comorbidity_count": rng.poisson(1.2, n), "bmi": np.round(rng.normal(27, 5, n), 1),
    })

    logit = (1.8 * df["biomarker_a"] + 1.6 * df["biomarker_b"] - 0.06 * (df["baseline_severity"] - 50)
             + rng.normal(0, 0.6, n))
    prob = 1 / (1 + np.exp(-logit))
    df["primary_endpoint_responder"] = rng.binomial(1, prob)

    return df, TRUE_PREDICTORS


if __name__ == "__main__":
    df, preds = simulate()
    print(df.shape, df["primary_endpoint_responder"].mean())
