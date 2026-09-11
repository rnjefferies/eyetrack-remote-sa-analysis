# §3.3b full-vs-reduced parsimony on the harmonised Sign+Animal scope (n=439),
# preferred configs (matches the §3.2-3.4 headlines). Source for the §3.3b table.
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import average_precision_score, roc_auc_score
df = pd.read_csv("data/master_routes_1_to_6_latency.csv")
df = df[df['Question_Type'].isin(['Sign','Animal'])].reset_index(drop=True)
g = df['Participant_ID'].to_numpy(); cv = GroupKFold(5)
GAZE=['Before_Saccade_Rate_Hz','Before_Dwell_Proportion_Target_Object','Before_Mean_Saccadic_Velocity','Before_Road_Gaze_Pct','Before_Scanpath_Rate_px_s']
CONTROL=['Before_Steer_Variance','Before_Speed_Variance','Before_Major_SRR','Before_Fine_SRR','Before_TRR','Before_Zero_Throttle_Pct']
INDIV=['Video_Game_Hours_Num','Driving_Hours_Num']
distractor=[c for c in df.columns if 'Before_Dwell_Proportion' in c and 'Distractor' in c]
leaky=['Before_Dwell_Proportion_'+x+'_Distractor' for x in ['r2','cone','rwheel','Sign','lwheel','bumper','Animal']]
DISTRACT=[c for c in distractor if c not in leaky]
FULL=GAZE+CONTROL+INDIV+DISTRACT; REDUCED=GAZE+CONTROL
def model(kind,y):
    if kind=='xgb':
        w=(y==0).sum()/max((y==1).sum(),1)
        return xgb.XGBClassifier(scale_pos_weight=w,eval_metric='logloss',random_state=42,max_depth=2,learning_rate=0.03,n_estimators=100)
    if kind=='rf': return RandomForestClassifier(class_weight='balanced',n_estimators=100,max_depth=3,random_state=42)
    return Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler()),('m',LogisticRegression(class_weight='balanced',max_iter=2000,C=0.1,random_state=42))])
def prauc(feats,kind,y):
    X=df[feats].to_numpy(); p=cross_val_predict(model(kind,y),X,y,cv=cv,groups=g,method='predict_proba')[:,1]
    return average_precision_score(y,p)
print(f"Sign+Animal n={len(df)} | FULL={len(FULL)} feats, REDUCED={len(REDUCED)} feats, DISTRACT={DISTRACT}\n")
print(f"{'detector':12s}{'base':>7}{'FULL':>9}{'REDUCED':>9}{'verdict':>20}")
for name,kind,ytgt in [('Accuracy','xgb',(1-df['Target_Accuracy']).astype(int).to_numpy()),
                       ('Freeze','rf',(df['Target_Latency']>=3.5).astype(int).to_numpy()),
                       ('Confidence','lr',(df['Target_Confidence']<=4).astype(int).to_numpy())]:
    b=ytgt.mean(); f=prauc(FULL,kind,ytgt); r=prauc(REDUCED,kind,ytgt)
    v='reduced wins' if r>f+0.005 else ('tie' if abs(r-f)<=0.005 else 'full edge')
    print(f"{name:12s}{b:>7.3f}{f:>9.3f} ({f/b:.2f}){r:>9.3f} ({r/b:.2f})   {v}")
