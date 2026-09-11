import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
frames=[]
for i in range(1,7): frames.append(pd.read_csv(f"data/route{i}_ml_ready.csv"))
df=pd.concat(frames,ignore_index=True)
df=df[df['Target_Latency']>=0].reset_index(drop=True)
for c in [c for c in df.columns if 'Before_Raw_Dwell' in c]:
    df[c.replace('Raw_Dwell','Dwell_Proportion')]=df[c]/5.0
df=df[df['Question_Type'].isin(['Sign','Animal'])].reset_index(drop=True)
F=['Before_Saccade_Rate_Hz','Before_Dwell_Proportion_Target_Object','Before_Mean_Saccadic_Velocity',
   'Before_Road_Gaze_Pct','Before_Scanpath_Rate_px_s','Before_Steer_Variance','Before_Speed_Variance',
   'Before_Major_SRR','Before_Fine_SRR','Before_TRR','Before_Zero_Throttle_Pct']
X=df[F]; g=df['Participant_ID']; y=(df['Target_Confidence']<=4).astype(int); cv=GroupKFold(5)
m=Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler()),('m',LogisticRegression(class_weight='balanced',max_iter=2000,C=0.1,random_state=42))])
p=cross_val_predict(m,X,y,groups=g,cv=cv,method='predict_proba')[:,1]
pr=average_precision_score(y,p); roc=roc_auc_score(y,p); base=y.mean()
# cluster (participant) bootstrap 95% CI on PR-AUC
ids=df['Participant_ID'].values; uids=np.unique(ids); rng=np.random.default_rng(42); boots=[]
for _ in range(2000):
    samp=rng.choice(uids,len(uids),replace=True)
    idx=np.concatenate([np.where(ids==u)[0] for u in samp])
    if y.values[idx].sum()>0: boots.append(average_precision_score(y.values[idx],p[idx]))
lo,hi=np.percentile(boots,[2.5,97.5])
print("CONFIDENCE detector, Sign+Animal (n=%d, pos=%d, base=%.3f):"%(len(y),int(y.sum()),base))
print("  PR-AUC %.3f  95%% CI [%.3f, %.3f]   (vs chance %.3f, %.2f-fold lift)"%(pr,lo,hi,base,pr/base))
print("  ROC-AUC %.3f"%roc)
