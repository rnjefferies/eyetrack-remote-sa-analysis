# EyeTrack Remote-SA Analysis

Reproducible analysis package for **Chapter 5, Study 2** of the thesis *Towards Passive Measurement of Situation
Awareness in the Remote Operation of Live Vehicles*: a passive, behaviour-only
situation-awareness (SA) indicator for remote / teleoperated driving, built from pre-probe
gaze and vehicle-control features across 37 operators. Each notebook reproduces a specific
part of the chapter and its supplement, and carries a markdown header stating its purpose,
the packages it uses, and the justification for the methods applied.

This is the **Study 2** analysis package. The data-collection and processing tools that
turn live gaze, audio, and 1000 Hz telemetry into the feature tables used here are in the
companion repository
**[eyetrack-remote-sa-tools](https://github.com/rnjefferies/eyetrack-remote-sa-tools)**; the
Study 1 analyses (offline versus real-time SA, and confidence) are in
**[query-remote-sa-analysis](https://github.com/rnjefferies/query-remote-sa-analysis)**.

> **Data included.** The de-identified, analysis-ready feature tables are committed here
> (one row per SA probe; anonymised operator codes such as `01_RT`; no raw recordings or
> identifying information).

## Repository layout

```
notebooks/   analysis notebooks 01–09 (run top to bottom)
scripts/     helper analyses driven by notebook 09
data/        de-identified per-probe feature tables + video-coding data
dashboard/   optional SA-state monitor app + its prebuilt inputs
outputs/     (ships empty) figures and result tables are written here when you run the notebooks
```

## Quick start

Run every command from the **repository root**, in a fresh virtual environment:

```bash
python3 -m venv .venv && source .venv/bin/activate   # Python 3.11 used for the reported run
pip install -r requirements.txt
jupyter lab          # open notebooks/ and run 01–09 top to bottom
```

- **Working directory.** Launch from the repository root so the `data/`, `scripts/`,
  `outputs/`, and `dashboard/` paths resolve. (Each notebook's first cell also re-anchors to
  the root, so notebooks work wherever you open them; the `dashboard/` scripts do not, so run
  those from the root — see below.)
- **Packages** (all pinned in `requirements.txt`): `pandas`, `numpy (<2)`, `scipy`,
  `scikit-learn`, `xgboost`, `statsmodels`, `shap`, `matplotlib`, `seaborn`, and `jupyterlab`
  to run the notebooks, plus `dash` + `plotly` for the optional dashboard.
- All analyses are deterministic (`random_state = 42`).

## Notebooks

| Notebook | Chapter 5 section | Produces |
|---|---|---|
| `01_Descriptives.ipynb` | §3.1 | participant + SA-outcome tables, predictor descriptives, figure panel |
| `02_Outcome_Structure_PCA.ipynb` | §2–3 | PCA of the three outcomes; within/between-operator coupling |
| `03_Accuracy_Detector.ipynb` | §3.2, §3.5 | SA-error detector (XGBoost), nested-CV comparison, out-of-fold SHAP, parsimony |
| `04_Latency_Detector.ipynb` | §3.3 | delayed-response detector (random forest), SHAP, threshold sensitivity |
| `05_Confidence_Detector.ipynb` | §3.4 | confidence regression (null) + low-confidence detector (logistic), calibration |
| `06_GLMM_Analysis.ipynb` | §3.7 | crossed-random-effect GLMMs (logit + log-latency), coefficient forest plots |
| `07_Video_Coding_Analysis.ipynb` | §3.10 | blinded antecedent coding, Fisher-exact marker/collision tests |
| `08_Lean_IMU_Analysis.ipynb` | §3.10 | objective head-lean (IMU) corroboration, Mann–Whitney contrasts |
| `09_Supplementary_Analyses.ipynb` | supplement | single runnable reproduction of the supplementary notes/tables (drives the helper `*.py` scripts) |

**Modelling scope.** The three detectors and the GLMMs share one sample: the perceptually
demanding **Sign + Animal** probes (*n* = 439 probes, 37 operators).

## Data files

| File | Contents |
|---|---|
| `data/route1…6_ml_ready.csv` | one row per probe: pre-probe gaze + control features and outcomes |
| `data/master_routes_1_to_6_latency.csv` | trial-level master (all query types) |
| `data/master_routes_1_to_6_combined.csv` | stacked, scaled all-query master |
| `data/Video_Coded_Analysis.csv`, `data/CODING_SHEET_antecedent_COMPLETE.csv` | blinded antecedent video coding |
| `data/_lean_imu_features.csv` | cached head-lean IMU features (raw archive not distributed) |
| `dashboard/SA_Dashboard_*.csv` | pre-built inputs for the optional dashboard |

Helper scripts (`*.py`) are the single source of truth for several supplementary analyses
and are executed in place by `09_Supplementary_Analyses.ipynb`.

## Interactive dashboard (optional)

`sa_dashboard_app_v3.py` is a Dash/Plotly demo of the layered SA-state monitor: a per-operator
triage board, live recent-state dials, a probe timeline, and the triggered confidence
read-out. It reads the committed `SA_Dashboard_*.csv`, so it runs out of the box. Run it from
the **repository root** (the `dashboard/` prefix is part of the command; the data paths are
relative to the root, so `cd dashboard` first will not work):

```bash
python dashboard/sa_dashboard_app_v3.py        # then open http://127.0.0.1:8051
```

To rebuild the dashboard data from the route CSVs (e.g. after changing a model), again from the
repository root:

```bash
python dashboard/_build_dashboard_data.py   # regenerates the three SA_Dashboard_*.csv
python dashboard/sa_dashboard_app_v3.py
```

`_build_dashboard_data.py` applies the same eyes-forward road-gaze and angular-saccade
corrections as the notebooks, and uses the validated detector hyperparameters.

## Citation

If you use this code or data, please cite the thesis (R. Jefferies, *Towards Passive Measurement of Situation
Awareness in the Remote Operation of Live Vehicles*). Full citation and DOI to be added on
deposit.

## Licence

Code released under the MIT Licence (see [`LICENSE`](LICENSE)). The de-identified data are
shared for reproduction of the reported analyses.
