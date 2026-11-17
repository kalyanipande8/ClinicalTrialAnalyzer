import json
import os

from src.simulate_trial_records import simulate
from src.ingestion_qc import run_ingestion_qc
from src.statistical_analyses import run_all_analyses, ANALYSES
from src.endpoint_model import train_and_evaluate


def main():
    df, true_predictors = simulate()
    qc = run_ingestion_qc(df)
    analyses = run_all_analyses(df)
    model_result = train_and_evaluate(df)

    summary = {"n_patients": len(df), "qc": qc, "n_analyses_run": len(ANALYSES),
               "endpoint_model": model_result,
               "key_analysis_results": {k: analyses[k] for k in
                                        ["overall_responder_rate", "treatment_effect_ttest",
                                         "correlation_biomarker_a", "correlation_biomarker_b"]}}

    os.makedirs("results", exist_ok=True)
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
