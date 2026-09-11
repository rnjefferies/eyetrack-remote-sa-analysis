# Permutation test: are the three OOF detectors better than a stratified-dummy / no-skill null?
# No-skill baselines: ROC-AUC = 0.5, PR-AUC = positive prevalence.
# Test: shuffle the outcome, re-run the SAME GroupKFold-5 OOF pipeline, recompute PR-AUC/ROC-AUC.
#   - FREE shuffle      -> tests vs complete no-information (textbook chance test)
#   - WITHIN-OPERATOR   -> preserves each operator's base rate; tests within-operator feature signal
# p = (1 + #{perm >= observed}) / (B + 1)
import warnings, numpy as np, pandas as pd, time
warnings.filterwarnings("ignore")
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import average_precision_score, roc_auc_score

B = 1000
RNG = np.random.default_rng(42)
df = pd.read_csv("data/master_routes_1_to_6_latency.csv")
df = df[df['Question_Type'].isin(['Sign', 'Animal'])].reset_index(drop=True)
g = df['Participant_ID'].to_numpy(); cv = GroupKFold(5)
GAZE = ['Before_Saccade_Rate_Hz', 'Before_Dwell_Proportion_Target_Object', 'Before_Mean_Saccadic_Velocity', 'Before_Road_Gaze_Pct', 'Before_Scanpath_Rate_px_s']
CONTROL = ['Before_Steer_Variance', 'Before_Speed_Variance', 'Before_Major_SRR', 'Before_Fine_SRR', 'Before_TRR', 'Before_Zero_Throttle_Pct']
REDUCED = GAZE + CONTROL

def make(kind, y):
    if kind == 'xgb':
        w = (y == 0).sum() / max((y == 1).sum(), 1)
        return xgb.XGBClassifier(scale_pos_weight=w, eval_metric='logloss', random_state=42, max_depth=2, learning_rate=0.03, n_estimators=100)
    if kind == 'rf':
        return RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=3, random_state=42)
    return Pipeline([('i', SimpleImputer(strategy='median')), ('s', StandardScaler()), ('m', LogisticRegression(class_weight='balanced', max_iter=2000, C=0.1, random_state=42))])

X_all = df[REDUCED].to_numpy()
def oof_scores(y, kind):
    p = cross_val_predict(make(kind, y), X_all, y, cv=cv, groups=g, method='predict_proba')[:, 1]
    return average_precision_score(y, p), roc_auc_score(y, p)

def perm_within(y):
    yp = y.copy()
    for grp in np.unique(g):
        idx = np.where(g == grp)[0]
        yp[idx] = RNG.permutation(y[idx])
    return yp

outcomes = [
    ('Accuracy (incorrect)', 'xgb', (1 - df['Target_Accuracy']).astype(int).to_numpy()),
    ('Latency (delayed >=3.5s)', 'rf', (df['Target_Latency'] >= 3.5).astype(int).to_numpy()),
    ('Confidence (unsure <=4)', 'lr', (df['Target_Confidence'] <= 4).astype(int).to_numpy()),
]

print(f"Permutation test vs no-skill null | Sign+Animal n={len(df)} | reduced-{len(REDUCED)} feats | B={B}\n")
for name, kind, y in outcomes:
    t0 = time.time()
    base = y.mean()
    ap_obs, roc_obs = oof_scores(y, kind)
    ap_free = np.empty(B); ap_within = np.empty(B)
    roc_free = np.empty(B); roc_within = np.empty(B)
    for b in range(B):
        ap_free[b], roc_free[b] = oof_scores(RNG.permutation(y), kind)
        ap_within[b], roc_within[b] = oof_scores(perm_within(y), kind)
    def pval(obs, null):
        return (1 + np.sum(null >= obs)) / (B + 1)
    print(f"== {name} | learner={kind} | base rate(=PR-AUC null)={base:.3f} ==")
    print(f"   PR-AUC observed = {ap_obs:.3f}  (lift {ap_obs/base:.2f}x over base)")
    print(f"     free-perm null mean={ap_free.mean():.3f} (95% {np.percentile(ap_free,2.5):.3f}-{np.percentile(ap_free,97.5):.3f})  p={pval(ap_obs,ap_free):.4f}")
    print(f"     within-op null mean={ap_within.mean():.3f} (95% {np.percentile(ap_within,2.5):.3f}-{np.percentile(ap_within,97.5):.3f})  p={pval(ap_obs,ap_within):.4f}")
    print(f"   ROC-AUC observed = {roc_obs:.3f}  (null=0.500)")
    print(f"     free-perm  p={pval(roc_obs,roc_free):.4f}   within-op p={pval(roc_obs,roc_within):.4f}")
    print(f"   [{time.time()-t0:.0f}s]\n", flush=True)
print("DONE")
