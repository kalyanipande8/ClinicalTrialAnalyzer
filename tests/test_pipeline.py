import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulate_trial_records import simulate
from src.ingestion_qc import run_ingestion_qc
from src.statistical_analyses import run_all_analyses, ANALYSES
from src.endpoint_model import train_and_evaluate


def test_cohort_has_2500plus_patients():
    df, _ = simulate()
    assert len(df) >= 2500


def test_ingestion_qc_passes_on_clean_data():
    df, _ = simulate()
    qc = run_ingestion_qc(df)
    assert qc["passes_qc"] is True


def test_exactly_18_analyses_run():
    assert len(ANALYSES) == 18
    df, _ = simulate()
    results = run_all_analyses(df)
    assert len(results) == 18


def test_endpoint_model_achieves_reasonable_auc():
    df, _ = simulate()
    result = train_and_evaluate(df)
    assert result["auc"] > 0.75


def test_true_predictor_biomarkers_correlate_with_endpoint():
    df, _ = simulate()
    results = run_all_analyses(df)
    assert abs(results["correlation_biomarker_a"]["r"]) > 0.1
    assert abs(results["correlation_biomarker_b"]["r"]) > 0.1
