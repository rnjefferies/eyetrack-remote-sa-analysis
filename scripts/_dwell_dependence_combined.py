#!/usr/bin/env python3
# Two-panel target-dwell SHAP dependence: accuracy (XGBoost) and latency (random
# forest) detectors, OOF, Sign+Animal n=439. Shows the same saturating non-linear
# shape in both outcomes at very different magnitudes. -> outputs/Supp_TargetDwell_Dependence.png
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings
warnings.filterwarnings('ignore')

CANDIDATE = [
    'Before_Saccade_Rate_Hz', 'Before_Dwell_Proportion_Target_Object',
    'Before_Mean_Saccadic_Velocity', 'Before_Road_Gaze_Pct', 'Before_Scanpath_Rate_px_s',
    'Before_Steer_Variance', 'Before_Speed_Variance', 'Before_Major_SRR',
    'Before_Fine_SRR', 'Before_TRR', 'Before_Zero_Throttle_Pct',
]
DWELL = 'Before_Dwell_Proportion_Target_Object'

frames = []
for i in range(1, 7):
    fn = f"data/route{i}_ml_ready.csv"
    if os.path.exists(fn):
        frames.append(pd.read_csv(fn))
df = pd.concat(frames, ignore_index=True).fillna(0.0)
for c in [c for c in df.columns if 'Before_Raw_Dwell' in c]:
    df[c.replace('Raw_Dwell', 'Dwell_Proportion')] = df[c] / 5.0
df = df[df['Question_Type'].isin(['Sign', 'Animal'])].copy()
feats = [f for f in CANDIDATE if f in df.columns]
X = df[feats]; groups = df['Participant_ID']
di = feats.index(DWELL)


def pos_vals(sv):
    if isinstance(sv, list):
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


def oof_dwell_shap(y, make):
    sx_list, xv_list = [], []
    for tr, te in GroupKFold(5).split(X, y, groups):
        if len(np.unique(y.iloc[tr])) < 2:
            continue
        m = make(y.iloc[tr]); m.fit(X.iloc[tr], y.iloc[tr])
        sv = pos_vals(shap.TreeExplainer(m).shap_values(X.iloc[te]))
        sx_list.append(sv[:, di]); xv_list.append(X.iloc[te][DWELL].values)
    return np.concatenate(sx_list), np.concatenate(xv_list)


def make_xgb(ytr):
    w = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    return xgb.XGBClassifier(scale_pos_weight=w, eval_metric='logloss', random_state=42,
                             max_depth=2, learning_rate=0.03, n_estimators=100)


def make_rf(ytr=None):
    return RandomForestClassifier(class_weight='balanced', random_state=42,
                                  n_estimators=100, max_depth=3)


y_acc = (1 - df['Target_Accuracy']).astype(int)
y_lat = (df['Target_Latency'] >= 3.5).astype(int)
panels = [
    ("Accuracy model (errors)", y_acc, make_xgb, '#3a7ca5', '+ = push toward SA error'),
    ("Latency model (delayed responses)", y_lat, make_rf, '#7b9e3f', '+ = push toward delayed response'),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
rng = np.random.default_rng(0)
for ax, (title, y, make, col, ylab) in zip(axes, panels):
    sx, xv = oof_dwell_shap(y, make)
    b1, b0 = np.polyfit(xv, sx, 1)
    xx = np.linspace(xv.min(), xv.max(), 200)
    lo = lowess(sx, xv, frac=0.4, return_sorted=True)
    zero = xv == 0
    xj = xv + np.where(zero, rng.uniform(-0.004, 0.004, len(xv)), 0)
    ax.axhline(0, color='#bbb', lw=.8)
    ax.scatter(xj, sx, s=14, alpha=.4, color=col, edgecolor='none', label='OOF probes')
    ax.plot(xx, b0 + b1 * xx, '--', color='#888', lw=1.6, label='straight-line (linear) fit')
    ax.plot(lo[:, 0], lo[:, 1], color='#e4572e', lw=2.3, label='actual shape (LOWESS)')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Pre-query target-object dwell (proportion of 5 s window)')
    ax.set_ylabel(f'SHAP value  ({ylab})')
    ax.legend(frameon=False, fontsize=8.5)
fig.suptitle('Target-object dwell shows the same saturating (threshold) effect in both detectors\n'
             '(out-of-fold SHAP, Sign + Animal, n=439; note the differing vertical scales)',
             fontsize=12.5, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('outputs/Supp_TargetDwell_Dependence.png', dpi=140, bbox_inches='tight'); plt.close()
print("saved outputs/Supp_TargetDwell_Dependence.png (two-panel: accuracy + latency)")
