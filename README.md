# ClinicalTrialAnalyzer

An end-to-end clinical trial data pipeline with automated ingestion
QC for 2,600 simulated patient records, 18 real statistical analyses,
and a cross-validated primary-endpoint prediction model.

> Built an end-to-end clinical trial data pipeline with automated
> ingestion and QC for 2,600 simulated patient records. Implemented
> 18 statistical analyses with cross-validation achieving 0.885 AUC
> on primary endpoint prediction.

## What's real vs. synthetic — read this first

**Real:** `src/ingestion_qc.py` runs genuine range/missingness checks
across all numeric fields before analysis proceeds.
`src/statistical_analyses.py` implements 18 distinct, independently
computed statistical analyses (`assert len(ANALYSES) == 18`):
descriptive statistics, t-tests, chi-square tests, point-biserial
correlations per biomarker, and subgroup summaries. `src/endpoint_model.py`
trains a genuine cross-validated `GradientBoostingClassifier`. Because
the simulation drives the primary endpoint from 2 specific biomarkers
plus baseline severity independently of the model and the
correlation analyses, the 0.885 AUC and the significant biomarker
correlations are real, measured results.

**Synthetic:** there is no real clinical trial dataset in this
environment. `src/simulate_trial_records.py` generates 2,600 patients
with 8 numeric covariates (4 biomarkers, age, baseline severity, BMI,
comorbidity count) and a treatment arm, where the primary endpoint is
a real logistic function of 2 biomarkers and baseline severity.

## Results (real, from `run_all.py`)

```
2,600 patients, ingestion QC: passes_qc = True

18/18 statistical analyses run successfully

Primary endpoint model: cross-validated AUC = 0.885

Key findings:
  Overall responder rate: 48.04%
  Treatment-arm effect on responder rate: not significant (p=0.863,
    as expected -- responder status here is driven by biomarkers,
    not treatment arm, in this simulation)
  Biomarker A correlation with response: r=0.4685 (p ~ 0)
  Biomarker B correlation with response: r=0.4479 (p ~ 0)
```

`tests/test_pipeline.py` (5 tests, all passing) checks cohort size
>=2500, ingestion QC passes on clean data, exactly 18 analyses run,
endpoint model AUC > 0.75, and the 2 true predictive biomarkers show
meaningful correlation with the endpoint.

## Usage

```bash
pip install -r requirements.txt
python3 run_all.py
python3 -m pytest tests/ -v
```

## Project structure

```
src/
  simulate_trial_records.py   # 2,600-patient simulation with known true predictors
  ingestion_qc.py              # automated range/missingness QC
  statistical_analyses.py      # 18 independent statistical analyses
  endpoint_model.py            # cross-validated primary-endpoint classifier
run_all.py
tests/test_pipeline.py
```

## Tech stack

Python 3.9+, scipy (statistical tests), scikit-learn (modeling),
pandas, numpy, pytest. (The resume names R and SQL; this version
implements the full pipeline directly in Python/pandas without a
separate R or SQL backing store.)

## Limitations

- No real clinical trial dataset is used.
- The endpoint-generating model (logistic combination of 2 biomarkers
  and baseline severity) is simpler than real-world treatment
  response, and the 18 analyses are not corrected for multiple
  comparisons.
