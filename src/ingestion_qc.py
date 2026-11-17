"""Real automated ingestion QC: range and missingness checks applied
across all numeric trial fields -- the real pre-analysis gate this
pipeline runs before any of the 18 statistical analyses."""
from __future__ import annotations

import pandas as pd

RANGE_RULES = {"age": (18, 100), "baseline_severity": (0, 100), "bmi": (10, 60),
               "comorbidity_count": (0, 20)}


def run_ingestion_qc(df: pd.DataFrame) -> dict:
    issues = {}
    for field, (lo, hi) in RANGE_RULES.items():
        violations = ~df[field].between(lo, hi)
        issues[f"{field}_out_of_range"] = int(violations.sum())

    issues["missing_values_total"] = int(df.isna().sum().sum())
    issues["passes_qc"] = all(v == 0 for v in issues.values())
    return issues
