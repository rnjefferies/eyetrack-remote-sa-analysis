"""
Feature-block comparison for the three SA-outcome detectors.

Goal: test whether FEWER, theory-grouped features do as well as (or better than)
the full candidate set, and which theoretical block carries the signal.

Design:
  * Targets (rare-event, grouped by participant):
        Accuracy   : y = 1 - Target_Accuracy            (incorrect SA response)
        Latency    : y = Target_Latency >= 3.5s          (cognitive freeze)
        Confidence : y = Target_Confidence <= 4          (low confidence / unsure)
  * A SINGLE common estimator is used for every block so the comparison isolates
    the feature block, not the model family:
        balanced L2 logistic regression (median impute -> standard scale).
  * Evaluation = grouped out-of-fold (GroupKFold-5 cross_val_predict), the same
    "primary" estimator the notebooks use.  Reported: OOF PR-AUC (+ lift over base
    rate), ROC-AUC.  Participant-cluster bootstrap 95% CI on PR-AUC (B=400).
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import average_precision_score, roc_auc_score

CSV = "data/master_routes_1_to_6_latency.csv"
df = pd.read_csv(CSV)

_off = [c for c in ['Before_Raw_Dwell_map_Distractor', 'Before_Raw_Dwell_name_Distractor'] if c in df.columns]
df['Before_Road_Gaze_Pct'] = (1 - df[_off].sum(axis=1) / 5.0).clip(0, 1)
# (2) Sign + Animal modelling scope (n = 439), as used by the detectors/GLMM.
df = df[df['Question_Type'].isin(['Sign', 'Animal'])].reset_index(drop=True)

# ---- feature blocks -------------------------------------------------------
GAZE = ['Before_Saccade_Rate_Hz', 'Before_Dwell_Proportion_Target_Object',
        'Before_Mean_Saccadic_Velocity', 'Before_Road_Gaze_Pct',
        'Before_Scanpath_Rate_px_s']
CONTROL = ['Before_Steer_Variance', 'Before_Speed_Variance', 'Before_Major_SRR',
           'Before_Fine_SRR', 'Before_TRR', 'Before_Zero_Throttle_Pct']
PRIOR_EXP = ['Video_Game_Hours_Num', 'Driving_Hours_Num']
distractor_cols = [c for c in df.columns
                   if 'Before_Dwell_Proportion' in c and 'Distractor' in c]
leaky = ['Before_Dwell_Proportion_' + x + '_Distractor' for x in
         ['r2', 'cone', 'rwheel', 'Sign', 'lwheel', 'bumper', 'Animal']]
DISTRACT = [c for c in distractor_cols if c not in leaky]

PARSIMONIOUS = ['Before_Dwell_Proportion_Target_Object', 'Before_Road_Gaze_Pct',
                'Before_Major_SRR']            # minimal cross-block set

BLOCKS = {
    'Gaze/attention'      : GAZE,
    'Manual-control'      : CONTROL,
    'Prior experience'    : PRIOR_EXP,
    'Distractor-dwell'    : DISTRACT,
    'Gaze + Control'      : GAZE + CONTROL,
    'Parsimonious (3)'    : PARSIMONIOUS,
    'Target-dwell only(1)': ['Before_Dwell_Proportion_Target_Object'],
    'FULL (current)'      : GAZE + CONTROL + PRIOR_EXP + DISTRACT,
}

def make_est():
    return Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('sc',  StandardScaler()),
                     ('lr',  LogisticRegression(class_weight='balanced',
                                                max_iter=4000, random_state=42))])

def oof_proba(X, y, groups):
    return cross_val_predict(make_est(), X, y, groups=groups,
                             cv=GroupKFold(5), method='predict_proba')[:, 1]

def boot_ap_ci(y, p, groups, B=400, seed=42):
    rng = np.random.default_rng(seed)
    pids = np.unique(groups)
    gidx = {g: np.where(groups == g)[0] for g in pids}
    aps = []
    for _ in range(B):
        samp = rng.choice(pids, size=len(pids), replace=True)
        idx = np.concatenate([gidx[g] for g in samp])
        if len(np.unique(y[idx])) < 2:
            continue
        aps.append(average_precision_score(y[idx], p[idx]))
    return (np.percentile(aps, 2.5), np.percentile(aps, 97.5)) if aps else (np.nan, np.nan)

def run_target(name, y, base_df):
    sub = base_df.loc[y.index]
    groups = sub['Participant_ID'].values
    yv = y.values
    base = yv.mean()
    print("\n" + "=" * 78)
    print(f"TARGET: {name}    n={len(yv)}  positives={int(yv.sum())} "
          f"({base:.1%})  participants={len(np.unique(groups))}")
    print("=" * 78)
    print(f"{'Block':<22}{'k':>3}{'PR-AUC':>9}{'  95% CI':>16}{'lift':>7}{'ROC':>7}")
    print("-" * 78)
    rows = []
    for bname, cols in BLOCKS.items():
        cols = [c for c in cols if c in sub.columns]
        X = sub[cols]
        p = oof_proba(X, yv, groups)
        ap = average_precision_score(yv, p)
        roc = roc_auc_score(yv, p)
        lo, hi = boot_ap_ci(yv, p, groups)
        print(f"{bname:<22}{len(cols):>3}{ap:>9.3f}  [{lo:.3f},{hi:.3f}]"
              f"{ap/base:>7.2f}{roc:>7.3f}")
        rows.append([name, bname, len(cols), base, ap, lo, hi, ap/base, roc])
    return rows

allrows = []
# Accuracy
y = (1 - df['Target_Accuracy']).astype(int)
allrows += run_target('Accuracy (incorrect SA)', y, df)
# Latency
d2 = df[df['Target_Latency'] >= 0].copy()
yl = (d2['Target_Latency'] >= 3.5).astype(int)
allrows += run_target('Latency (freeze >=3.5s)', yl, d2)
# Confidence
yc = (df['Target_Confidence'] <= 4).astype(int)
allrows += run_target('Confidence (unsure <=4)', yc, df)

out = pd.DataFrame(allrows, columns=['Target', 'Block', 'k', 'BaseRate',
                                     'PR_AUC', 'CI_lo', 'CI_hi', 'Lift', 'ROC_AUC'])
out.to_csv("outputs/_feature_block_comparison.csv", index=False)
print("\nSaved -> outputs/_feature_block_comparison.csv")
