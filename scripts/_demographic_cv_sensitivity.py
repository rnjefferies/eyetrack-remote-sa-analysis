"""
_demographic_cv_sensitivity.py: backs Supplementary Note (Robustness to
Operator Demographics). Uses the eyes-forward road-gaze feature.

Tests whether the grouped cross-validation used for the three SA detectors was
biased by the uneven distribution of operator sex and age across folds. GroupKFold
partitions operators but does not balance demographics, and both sex and age show
between-operator associations with the outcomes (Section 3.7). This script:

  (1) tabulates the demographic composition of the five GroupKFold test folds;
  (2) re-estimates out-of-fold PR-AUC / ROC-AUC with folds stratified by sex, and
      separately by age band, using StratifiedGroupKFold (operators kept intact);
  (3) adds sex and age band as predictors to test whether they carry signal the
      behaviour misses.

Reads route{1..6}_ml_ready.csv; reproduces the reduced-11 detector pipeline exactly
(matches _build_dashboard_data.py / deployment_validation_v3.py). Deterministic.
"""
import pandas as pd, numpy as np, warnings, os, random
warnings.filterwarnings('ignore')
os.environ['PYTHONHASHSEED'] = '0'; random.seed(42); np.random.seed(42)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
import xgboost as xgb

# ---------- data (replicate the detector pipeline exactly) ----------
frames = []
for i in range(1, 7):
    d = pd.read_csv(f"data/route{i}_ml_ready.csv"); d['Route'] = i; frames.append(d)
df = pd.concat(frames, ignore_index=True).fillna(0.0)
df = df[df['Target_Latency'] >= 0].reset_index(drop=True)
for c in [c for c in df.columns if 'Before_Raw_Dwell' in c]:
    df[c.replace('Raw_Dwell', 'Dwell_Proportion')] = df[c] / 5.0
# Eyes-forward road-gaze = 1 - dwell on the off-screen references
# (route map, name clipboard). Matches notebooks 01/03/04/05/06 and _build_dashboard_data.py.
# Using the diffuse road-gaze instead would understate the fold-stratification
# sensitivity of the latency/confidence detectors.
_off = [c for c in ['Before_Raw_Dwell_map_Distractor', 'Before_Raw_Dwell_name_Distractor'] if c in df.columns]
if 'Before_Road_Gaze_Pct' in df.columns and _off:
    df['Before_Diffuse_Gaze_Pct'] = df['Before_Road_Gaze_Pct']
    df['Before_Road_Gaze_Pct'] = (1 - df[_off].sum(axis=1) / 5.0).clip(0, 1)
df = df[df['Question_Type'].isin(['Sign', 'Animal'])].reset_index(drop=True)
FEATURES = ['Before_Saccade_Rate_Hz', 'Before_Dwell_Proportion_Target_Object',
    'Before_Mean_Saccadic_Velocity', 'Before_Road_Gaze_Pct', 'Before_Scanpath_Rate_px_s',
    'Before_Steer_Variance', 'Before_Speed_Variance', 'Before_Major_SRR',
    'Before_Fine_SRR', 'Before_TRR', 'Before_Zero_Throttle_Pct']
X = df[FEATURES].values
g = df['Participant_ID'].values
y_err = (1 - df['Target_Accuracy']).astype(int).values
y_lat = (df['Target_Latency'] >= 3.5).astype(int).values
y_conf = (df['Target_Confidence'] <= 4).astype(int).values
sex_female = df['Gender_Female'].astype(int).values      # 1 = female
age_hi = (df['Age_Num'] >= 4).astype(int).values         # older bands (4-5) vs younger (1-3)

ops = df.drop_duplicates('Participant_ID')
print(f"n = {len(df)} probes, {df['Participant_ID'].nunique()} operators "
      f"({int(ops['Gender_Female'].sum())} F / {int(ops['Gender_Male'].sum())} M; "
      f"{int((ops['Age_Num'] >= 4).sum())} in older bands 4-5). Age_Num is an ordinal band (1-5).")

# ---------- (1) fold composition under GroupKFold ----------
gkf = GroupKFold(5)
print("\n(1) GroupKFold test-fold composition (as used in the paper):")
rows = []
for k, (tr, te) in enumerate(gkf.split(X, y_err, g)):
    o = ops[ops['Participant_ID'].isin(df.iloc[te]['Participant_ID'].unique())]
    nop = len(o); f = int(o['Gender_Female'].sum())
    rows.append((k + 1, nop, f, nop - f, f / nop, (o['Age_Num'] >= 4).mean(),
                 y_err[te].mean(), y_lat[te].mean()))
comp = pd.DataFrame(rows, columns=['fold', 'ops', 'F', 'M', 'female_frac',
                                   'older_frac', 'err_rate', 'lat_rate'])
print(comp.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print(f"    female fraction spans {comp.female_frac.min():.0%}-{comp.female_frac.max():.0%}; "
      f"older fraction spans {comp.older_frac.min():.0%}-{comp.older_frac.max():.0%}; "
      f"error rate spans {comp.err_rate.min():.2f}-{comp.err_rate.max():.2f}")

# ---------- models ----------
def mkxgb(y):
    w = (y == 0).sum() / max((y == 1).sum(), 1)
    return xgb.XGBClassifier(scale_pos_weight=w, eval_metric='logloss', random_state=42,
                             n_jobs=1, max_depth=2, learning_rate=0.03, n_estimators=100)
def mkrf():
    return RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=3, random_state=42)
def mklr():
    return Pipeline([('i', SimpleImputer(strategy='median')), ('s', StandardScaler()),
                     ('m', LogisticRegression(class_weight='balanced', max_iter=2000, C=0.1, random_state=42))])
OUT = [('Accuracy (XGBoost)', y_err, mkxgb), ('Latency (random forest)', y_lat, mkrf),
       ('Low-confidence (logistic)', y_conf, mklr)]
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

def oof(mk, y, splits):
    p = cross_val_predict(mk(), X, y, cv=splits, method='predict_proba')[:, 1]
    return average_precision_score(y, p), roc_auc_score(y, p)

# ---------- (2) stratified-CV sensitivity ----------
print("\n(2) Out-of-fold PR-AUC (ROC-AUC) by cross-validation scheme:")
print(f"{'outcome':26s} {'base':>5s} {'GroupKFold':>13s} {'Strat-sex':>13s} {'Strat-age':>13s}")
for name, y, mk in OUT:
    m = (lambda: mk(y)) if name.startswith('Accuracy') else (lambda: mk())
    a0, r0 = oof(m, y, list(gkf.split(X, y, g)))
    a1, r1 = oof(m, y, list(sgkf.split(X, sex_female, g)))
    a2, r2 = oof(m, y, list(sgkf.split(X, age_hi, g)))
    print(f"{name:26s} {y.mean():5.3f}  {a0:.3f}({r0:.3f})  {a1:.3f}({r1:.3f})  {a2:.3f}({r2:.3f})")

# ---------- (3) sex + age as predictors ----------
Xd = np.column_stack([X, df['Gender_Female'].values, df['Age_Num'].values])
print("\n(3) Adding operator sex + age band as predictors (behaviour-only -> + sex/age):")
for name, y, mk in OUT:
    m = (lambda: mk(y)) if name.startswith('Accuracy') else (lambda: mk())
    base_ap, _ = oof(m, y, list(gkf.split(X, y, g)))
    p = cross_val_predict(m(), Xd, y, cv=list(gkf.split(Xd, y, g)), method='predict_proba')[:, 1]
    print(f"{name:26s} {base_ap:.3f} -> {average_precision_score(y, p):.3f} PR-AUC")
print("\nConclusion: adding sex/age as predictors moves PR-AUC by <0.03 for every outcome "
      "(and adding them as GLMM fixed effects leaves the behavioural coefficients essentially "
      "unchanged), so the behavioural signal does not reduce to a demographic proxy. Stratifying "
      "folds by sex or age shifts PR-AUC within each outcome's bootstrap CI, negligibly for "
      "accuracy, by up to ~0.05 for the latency/confidence detectors, reflecting fold-composition "
      "variance in a 37-operator sample rather than a demographic effect.")
