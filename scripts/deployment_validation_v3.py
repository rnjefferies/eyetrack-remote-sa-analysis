"""
deployment_validation_v3.py: backs the §3.11 Results sentence
("...the Red-state failure rate reading about 39% out of fold against about 44%
in sample and still well above the 30% base rate."). Reproduced as §B12 in
Supplementary_Analyses.ipynb. The rest (T2b/T3/T4) remains exploratory support,
not headline results.

Tests indicator transfer on the present sample with a FULLY NESTED
leave-operators-out pipeline. Everything is self-contained: it reads the raw route CSVs
and reproduces the build's pipeline, so the SA_Dashboard_* files and app are untouched.

Nested design (outer GroupKFold-5 over operators):
  * each outer fold TRAINS the dials, FITS the isotonic calibration, and SETS the
    cut-points on the inner (training) operators only, then evaluates on the held-out ones.
  * isotonic for the dials is fit on an inner GroupKFold-4 OOF, so the calibration never
    sees the operators it is scored on (the gap the in-sample build leaves open).

Reported:
  T1/T2  in-sample vs nested-OOF recall + state-conditional failure rates (the optimism,
         and whether the recall-targeted cut-points transfer).
  T2b    a composite -> P(any_fail) calibration layer: reliability in-sample vs nested.
  T3     operating-point selection out of sample: alert-budget and cost-weighted.
  T4     per-operator offset vs a global threshold (DEMONSTRATION ONLY, underpowered).
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve, brier_score_loss
import xgboost as xgb

# ---------- data (replicate _build_dashboard_data.py exactly) ----------
frames = []
for i in range(1, 7):
    d = pd.read_csv(f"data/route{i}_ml_ready.csv"); d['Route'] = i; frames.append(d)
df = pd.concat(frames, ignore_index=True).fillna(0.0)
df = df[df['Target_Latency'] >= 0].reset_index(drop=True)
for c in [c for c in df.columns if 'Before_Raw_Dwell' in c]:
    df[c.replace('Raw_Dwell', 'Dwell_Proportion')] = df[c] / 5.0
df = df[df['Question_Type'].isin(['Sign', 'Animal'])].reset_index(drop=True)
FEATURES = ['Before_Saccade_Rate_Hz', 'Before_Dwell_Proportion_Target_Object',
    'Before_Mean_Saccadic_Velocity', 'Before_Road_Gaze_Pct', 'Before_Scanpath_Rate_px_s',
    'Before_Steer_Variance', 'Before_Speed_Variance', 'Before_Major_SRR',
    'Before_Fine_SRR', 'Before_TRR', 'Before_Zero_Throttle_Pct']
df['evn'] = df['Event'].astype(str).str.extract(r'(\d+)').astype(int)
df = df.sort_values(['Participant_ID', 'Route', 'evn']).reset_index(drop=True)
X = df[FEATURES].values
groups = df['Participant_ID'].values
y_err = (1 - df['Target_Accuracy']).astype(int).values
y_lat = (df['Target_Latency'] >= 3.5).astype(int).values
y_any = ((y_err == 1) | (y_lat == 1)).astype(int)
df['any_fail'] = y_any
SPAN, DIAL_CAP = 3, 0.90
N = len(df)
print(f"n={N}  ops={df.Participant_ID.nunique()}  base any_fail={y_any.mean():.3f}")

def mkxgb(y):
    w = (y == 0).sum() / max((y == 1).sum(), 1)
    return xgb.XGBClassifier(scale_pos_weight=w, eval_metric='logloss', random_state=42,
                             max_depth=2, learning_rate=0.03, n_estimators=100)
def mkrf():
    return RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=3, random_state=42)

def band_series(comp, sub):
    """trailing EWMA (span 3) within each operator, respecting the session order of `sub`."""
    s = pd.Series(comp, index=sub.index)
    return s.groupby(sub['Participant_ID']).transform(lambda x: x.ewm(span=SPAN, adjust=False).mean()).values

def cutpoints(band, y, ra=0.85, rr=0.50):
    pr, rc, th = precision_recall_curve(y, band)
    a = float(th[np.argmin(np.abs(rc[:-1] - ra))]); r = float(th[np.argmin(np.abs(rc[:-1] - rr))])
    return min(a, r), r

def metrics(flag, y):
    pos = y.sum()
    rec = (flag & (y == 1)).sum() / pos if pos else np.nan
    prec = (flag & (y == 1)).sum() / flag.sum() if flag.sum() else np.nan
    return rec, prec, flag.mean()

# ---------- IN-SAMPLE reference (what the build/manuscript report) ----------
me = mkxgb(y_err); ml = mkrf()
raw_e = cross_val_predict(mkxgb(y_err), X, y_err, groups=groups, cv=GroupKFold(5), method='predict_proba')[:, 1]
raw_l = cross_val_predict(mkrf(),      X, y_lat, groups=groups, cv=GroupKFold(5), method='predict_proba')[:, 1]
ce = np.clip(IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(raw_e, y_err).transform(raw_e), 0, DIAL_CAP)
cl = np.clip(IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(raw_l, y_lat).transform(raw_l), 0, DIAL_CAP)
comp_is = np.maximum(ce, cl)
band_is = band_series(comp_is, df)
A_is, R_is = cutpoints(band_is, y_any)
comp_cal_is = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(comp_is, y_any).transform(comp_is)

# ---------- NESTED leave-operators-out ----------
band_ne = np.full(N, np.nan); comp_ne = np.full(N, np.nan); comp_cal_ne = np.full(N, np.nan)
amber_ne = np.zeros(N, bool); red_ne = np.zeros(N, bool)
fold_id = np.full(N, -1, int)
fold_cuts = []
outer = GroupKFold(5)
for k, (inn, out) in enumerate(outer.split(X, y_any, groups)):
    fold_id[out] = k
    sub_i, sub_o = df.iloc[inn], df.iloc[out]
    gi = groups[inn]
    m_e = mkxgb(y_err[inn]).fit(X[inn], y_err[inn])
    m_l = mkrf().fit(X[inn], y_lat[inn])
    oe = cross_val_predict(mkxgb(y_err[inn]), X[inn], y_err[inn], groups=gi, cv=GroupKFold(4), method='predict_proba')[:, 1]
    ol = cross_val_predict(mkrf(),            X[inn], y_lat[inn], groups=gi, cv=GroupKFold(4), method='predict_proba')[:, 1]
    ie = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(oe, y_err[inn])
    il = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(ol, y_lat[inn])
    cei = np.clip(ie.transform(oe), 0, DIAL_CAP); cli = np.clip(il.transform(ol), 0, DIAL_CAP)
    ceo = np.clip(ie.transform(m_e.predict_proba(X[out])[:, 1]), 0, DIAL_CAP)
    clo = np.clip(il.transform(m_l.predict_proba(X[out])[:, 1]), 0, DIAL_CAP)
    comp_i = np.maximum(cei, cli); comp_o = np.maximum(ceo, clo)
    band_i = band_series(comp_i, sub_i); band_o = band_series(comp_o, sub_o)
    comp_ne[out] = comp_o; band_ne[out] = band_o
    A, R = cutpoints(band_i, y_any[inn]); fold_cuts.append((A, R))
    amber_ne[out] = band_o >= A; red_ne[out] = band_o >= R
    # composite -> P(any_fail) calibration layer, fit on inner
    ic = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(comp_i, y_any[inn])
    comp_cal_ne[out] = ic.transform(comp_o)

def state_rates(amber, red, y):
    g = ~amber; a = amber & ~red
    fr = lambda m: y[m].mean() if m.sum() else np.nan
    return fr(g), fr(a), fr(red), g.mean(), a.mean(), red.mean()

print("\n===== T1/T2  recall + state-conditional failure rates =====")
for lab, band, A, R, am, rd in [
    ('IN-SAMPLE ', band_is, A_is, R_is, band_is >= A_is, band_is >= R_is),
    ('NESTED-OOF', band_ne, None, None, amber_ne, red_ne)]:
    ra, pa, fa = metrics(am, y_any); rr, pr_, fr_ = metrics(rd, y_any)
    gG, gA, gR, pG, pA, pR = state_rates(am, rd, y_any)
    print(f"{lab}: Amber recall={ra:.3f} rate={fa:.3f} | Red recall={rr:.3f} prec={pr_:.3f} rate={fr_:.3f}")
    print(f"           P(fail): Green={gG:.3f} Amber={gA:.3f} Red={gR:.3f}   (base {y_any.mean():.3f})")
fc = np.array(fold_cuts)
print(f"per-fold cut-points: Amber {fc[:,0].min():.3f}-{fc[:,0].max():.3f}  Red {fc[:,1].min():.3f}-{fc[:,1].max():.3f}")
# per-fold held-out recall scatter (nested)
ar, rr_ = [], []
for k in range(5):
    m = fold_id == k
    ar.append(metrics(amber_ne[m], y_any[m])[0]); rr_.append(metrics(red_ne[m], y_any[m])[0])
print(f"per-fold held-out recall SD: Amber {np.std(ar):.3f} (range {min(ar):.2f}-{max(ar):.2f})  Red {np.std(rr_):.3f} (range {min(rr_):.2f}-{max(rr_):.2f})")

print("\n===== T2b  composite -> P(any_fail) calibration reliability =====")
print(f"Brier: in-sample={brier_score_loss(y_any, comp_cal_is):.4f}  nested-OOF={brier_score_loss(y_any, comp_cal_ne):.4f}")
# reliability by decile (nested)
q = pd.qcut(comp_cal_ne, 5, duplicates='drop')
rel = pd.DataFrame({'p': comp_cal_ne, 'y': y_any, 'q': q}).groupby('q').agg(pred=('p', 'mean'), obs=('y', 'mean'), n=('y', 'size'))
print("nested reliability (5 bins):")
print(rel.to_string())

print("\n===== T3  operating-point selection out of sample =====")
# band_ne / comp_cal_ne are defined for EVERY probe (each from the fold where it was held out),
# so band_ne[inn] are honest OOF values for the inner operators -> set the operating point there.
for b in (0.10, 0.20):
    fl = np.zeros(N, bool)
    for inn, out in outer.split(X, y_any, groups):
        t = np.quantile(band_ne[inn], 1 - b)          # top-b of the inner band
        fl[out] = band_ne[out] >= t
    rr, pr_, fr_ = metrics(fl, y_any)
    print(f"alert-budget top {int(b*100)}% (band): realised rate={fr_:.3f} recall={rr:.3f} prec={pr_:.3f}")
for ratio in (5, 10):
    fl = np.zeros(N, bool)
    for inn, out in outer.split(X, y_any, groups):
        p, yi = comp_cal_ne[inn], y_any[inn]          # calibrated P(any_fail) on inner
        cand = np.unique(p)
        costs = [ratio * ((p < c) & (yi == 1)).sum() + ((p >= c) & (yi == 0)).sum() for c in cand]
        t = cand[int(np.argmin(costs))]
        fl[out] = comp_cal_ne[out] >= t
    rr, pr_, fr_ = metrics(fl, y_any)
    print(f"cost-weighted C_miss:C_fa={ratio}:1 (calib prob): realised rate={fr_:.3f} recall={rr:.3f} prec={pr_:.3f}")

print("\n===== T4  per-operator offset vs global threshold (DEMO, underpowered) =====")
opdf = pd.DataFrame({'pid': groups, 'band': band_ne, 'y': y_any})
gmed = np.median(band_ne)
op_off = opdf.groupby('pid')['band'].median() - gmed          # each op's baseline shift
# global Red on nested band at target ~50% recall
_, Rg = cutpoints(band_ne, y_any)
def per_op_recall(flag):
    t = pd.DataFrame({'pid': groups, 'f': flag, 'y': y_any})
    t = t[t.y == 1]
    return t.groupby('pid')['f'].mean()
glob = per_op_recall(band_ne >= Rg)
adj_band = band_ne - opdf['pid'].map(op_off).values
_, Ra = cutpoints(adj_band, y_any)
offs = per_op_recall(adj_band >= Ra)
print(f"global Red: mean per-op recall={glob.mean():.3f} SD={glob.std():.3f}")
print(f"+per-op offset: mean per-op recall={offs.mean():.3f} SD={offs.std():.3f}")
print("(offset baseline is the operator's OWN median band -> mild leakage, ~12 probes/op: demonstration only)")
