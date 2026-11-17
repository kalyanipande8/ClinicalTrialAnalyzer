"""18 real, independently computed statistical analyses over the
trial dataset -- descriptive statistics, group comparisons,
correlations, and treatment-effect tests -- registered in a flat list
so adding a 19th analysis needs a new function, not pipeline changes."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

NUMERIC_COVARIATES = ["age", "baseline_severity", "biomarker_a", "biomarker_b", "biomarker_c",
                      "biomarker_d", "comorbidity_count", "bmi"]


def _descriptive_stats(df):
    return df[NUMERIC_COVARIATES].describe().to_dict()


def _responder_rate_overall(df):
    return {"responder_rate_pct": round(100 * df["primary_endpoint_responder"].mean(), 2)}


def _responder_rate_by_arm(df):
    return df.groupby("treatment_arm")["primary_endpoint_responder"].mean().round(4).to_dict()


def _responder_rate_by_sex(df):
    return df.groupby("sex")["primary_endpoint_responder"].mean().round(4).to_dict()


def _treatment_effect_ttest(df):
    active = df.loc[df["treatment_arm"] == "Active", "primary_endpoint_responder"]
    placebo = df.loc[df["treatment_arm"] == "Placebo", "primary_endpoint_responder"]
    t, p = stats.ttest_ind(active, placebo)
    return {"t_stat": round(float(t), 4), "p_value": float(p)}


def _chi_square_sex_responder(df):
    table = pd.crosstab(df["sex"], df["primary_endpoint_responder"])
    chi2, p, _, _ = stats.chi2_contingency(table)
    return {"chi2": round(float(chi2), 4), "p_value": float(p)}


def _make_correlation_fn(cov):
    def fn(df):
        r, p = stats.pointbiserialr(df["primary_endpoint_responder"], df[cov])
        return {"covariate": cov, "r": round(float(r), 4), "p_value": float(p)}
    return fn


def _age_group_summary(df):
    bins = [18, 40, 60, 100]
    groups = pd.cut(df["age"], bins=bins)
    return df.groupby(groups, observed=True)["primary_endpoint_responder"].mean().round(4).to_dict()


def _bmi_category_summary(df):
    bins = [0, 25, 30, 60]
    groups = pd.cut(df["bmi"], bins=bins)
    return df.groupby(groups, observed=True)["primary_endpoint_responder"].mean().round(4).to_dict()


def _comorbidity_correlation(df):
    r, p = stats.pointbiserialr(df["primary_endpoint_responder"], df["comorbidity_count"])
    return {"r": round(float(r), 4), "p_value": float(p)}


def _severity_by_arm_ttest(df):
    active = df.loc[df["treatment_arm"] == "Active", "baseline_severity"]
    placebo = df.loc[df["treatment_arm"] == "Placebo", "baseline_severity"]
    t, p = stats.ttest_ind(active, placebo)
    return {"t_stat": round(float(t), 4), "p_value": float(p)}


ANALYSES = [
    {"name": "descriptive_statistics", "fn": _descriptive_stats},
    {"name": "overall_responder_rate", "fn": _responder_rate_overall},
    {"name": "responder_rate_by_arm", "fn": _responder_rate_by_arm},
    {"name": "responder_rate_by_sex", "fn": _responder_rate_by_sex},
    {"name": "treatment_effect_ttest", "fn": _treatment_effect_ttest},
    {"name": "chi_square_sex_vs_responder", "fn": _chi_square_sex_responder},
    {"name": "correlation_biomarker_a", "fn": _make_correlation_fn("biomarker_a")},
    {"name": "correlation_biomarker_b", "fn": _make_correlation_fn("biomarker_b")},
    {"name": "correlation_biomarker_c", "fn": _make_correlation_fn("biomarker_c")},
    {"name": "correlation_biomarker_d", "fn": _make_correlation_fn("biomarker_d")},
    {"name": "correlation_baseline_severity", "fn": _make_correlation_fn("baseline_severity")},
    {"name": "correlation_age", "fn": _make_correlation_fn("age")},
    {"name": "correlation_bmi", "fn": _make_correlation_fn("bmi")},
    {"name": "age_group_responder_summary", "fn": _age_group_summary},
    {"name": "bmi_category_responder_summary", "fn": _bmi_category_summary},
    {"name": "comorbidity_correlation", "fn": _comorbidity_correlation},
    {"name": "baseline_severity_balance_by_arm", "fn": _severity_by_arm_ttest},
    {"name": "age_balance_check", "fn": lambda df: {
        "active_mean_age": round(float(df.loc[df['treatment_arm'] == 'Active', 'age'].mean()), 2),
        "placebo_mean_age": round(float(df.loc[df['treatment_arm'] == 'Placebo', 'age'].mean()), 2)}},
]

assert len(ANALYSES) == 18


def run_all_analyses(df: pd.DataFrame) -> dict:
    return {a["name"]: a["fn"](df) for a in ANALYSES}
