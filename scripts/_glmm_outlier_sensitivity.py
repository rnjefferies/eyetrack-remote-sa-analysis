"""
Outlier-sensitivity check for the inferential GLMM.
Re-fits the crossed-RE binary GLMM (operator + item) on the Sign+Animal scope
with (a) RAW predictors and (b) predictors WINSORIZED at the 1st/99th percentile,
then compares ORs. A genuine effect should survive winsorizing; an outlier-driven
one will collapse. Validates against the published notebook ORs first.
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import statsmodels.api as sm

df0 = pd.read_csv("data/master_routes_1_to_6_latency.csv")
df0 = df0[df0['Question_Type'].isin(['Sign', 'Animal'])].copy()

PREDS = {
    'tgt_dwell': 'Before_Dwell_Proportion_Target_Object',
    'speed_var': 'Before_Speed_Variance',
    'major_srr': 'Before_Major_SRR',
    'trr':       'Before_TRR',
    'saccade':   'Before_Saccade_Rate_Hz',
    'road_gaze': 'Before_Road_Gaze_Pct',
}
df0['Item'] = df0['Route_ID'].astype(str) + "_" + df0['Event'].astype(str)
df0['incorrect'] = (1 - df0['Target_Accuracy']).astype(int)
df0['unsure'] = (df0['Target_Confidence'] <= 4).astype(int)

def winsorize(s, lo=0.01, hi=0.99):
    ql, qh = s.quantile(lo), s.quantile(hi)
    return s.clip(ql, qh)

def build(df, wins):
    d = df.copy()
    zcols = []
    for k, col in PREDS.items():
        x = winsorize(d[col]) if wins else d[col]
        d['z_' + k] = (x - x.mean()) / x.std()
        zcols.append('z_' + k)
    return d, zcols

def fit_bin(df, zcols, outcome):
    FIXED = ' + '.join(zcols)
    vc = {'P': '0 + C(Participant_ID)', 'I': '0 + C(Item)'}
    r = sm.BinomialBayesMixedGLM.from_formula(f'{outcome} ~ {FIXED}', vc, df).fit_vb()
    out = {}
    for i, nm in enumerate(r.model.exog_names):
        if nm == 'Intercept':
            continue
        mn, sd = r.fe_mean[i], r.fe_sd[i]
        out[nm] = (np.exp(mn), np.exp(mn - 1.96 * sd), np.exp(mn + 1.96 * sd))
    return out

for outcome in ['incorrect', 'unsure']:
    dR, zc = build(df0, wins=False)
    dW, _ = build(df0, wins=True)
    raw = fit_bin(dR, zc, outcome)
    win = fit_bin(dW, zc, outcome)
    print("=" * 76)
    print(f"OUTCOME: {outcome.upper()}   (Sign+Animal, n={len(df0)})")
    print("=" * 76)
    print(f"{'predictor':<14}{'RAW OR [95% CI]':>26}{'WINSORIZED OR [95% CI]':>30}")
    print("-" * 76)
    for nm in zc:
        ro, rlo, rhi = raw[nm]; wo, wlo, whi = win[nm]
        rsig = '*' if (rlo > 1 or rhi < 1) else ' '
        wsig = '*' if (wlo > 1 or whi < 1) else ' '
        print(f"{nm:<14}{ro:6.2f} [{rlo:4.2f},{rhi:4.2f}]{rsig:>2}"
              f"{wo:14.2f} [{wlo:4.2f},{whi:4.2f}]{wsig:>2}")
    print()
