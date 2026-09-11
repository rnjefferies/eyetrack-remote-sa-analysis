import pandas as pd, numpy as np
from scipy.stats import spearmanr, mannwhitneyu, fisher_exact
m=pd.read_csv('data/master_routes_1_to_6_latency.csv')
m['incorrect']=(m['Target_Accuracy']==0).astype(int)
m['conf']=pd.to_numeric(m['Target_Confidence'],errors='coerce')
m['overconf_err']=((m['conf']>=6)&(m['incorrect']==1)).astype(int)
m['collision']=pd.to_numeric(m['Target_Post_Q_Collision'],errors='coerce').fillna(0).astype(int)
m['speeding']=(pd.to_numeric(m['During_Mean_Speed_mph'],errors='coerce')>4).astype(int)
m['hazard']=(pd.to_numeric(m['During_Hazard_Encountered'],errors='coerce').fillna(0)>0).astype(int)
m['hazard_any']=((pd.to_numeric(m['During_Hazard_Encountered'],errors='coerce').fillna(0)>0)|
                 (pd.to_numeric(m['Before_Hazard_Encountered'],errors='coerce').fillna(0)>0)).astype(int)

# ---------- OPERATOR LEVEL (n=37) ----------
op=m.groupby('Participant_ID').agg(
    n=('incorrect','size'), overconf=('overconf_err','sum'), errors=('incorrect','sum'),
    collisions=('collision','sum'), speeding=('speeding','sum'),
    hazard=('hazard_any','sum'), mean_speed=('During_Mean_Speed_mph','mean')).reset_index()
for c in ['overconf','errors','collisions','speeding','hazard']:
    op[c+'_rate']=op[c]/op['n']
# continuous overconfidence index = mean confidence on that operator's INCORRECT trials (NaN if no errors)
mc=m[m.incorrect==1].groupby('Participant_ID').conf.mean().rename('conf_when_wrong')
op=op.merge(mc, on='Participant_ID', how='left')

def partial_spearman(x,y,z):
    d=pd.DataFrame({'x':x,'y':y,'z':z}).dropna()
    rx,ry,rz=d.x.rank(),d.y.rank(),d.z.rank()
    ex=rx-np.polyval(np.polyfit(rz,rx,1),rz); ey=ry-np.polyval(np.polyfit(rz,ry,1),rz)
    r=np.corrcoef(ex,ey)[0,1]; n=len(d)
    from scipy.stats import t as tdist
    tt=r*np.sqrt((n-3)/(1-r**2)); p=2*tdist.sf(abs(tt),n-3)
    return r,p,n

print("="*72)
print("OPERATOR-LEVEL (n=37): overconfident-error rate vs risky driving")
print("="*72)
for out in ['collisions_rate','speeding_rate','hazard_rate','mean_speed']:
    r,p=spearmanr(op['overconf_rate'], op[out])
    pr,pp,_=partial_spearman(op['overconf_rate'], op[out], op['errors_rate'])
    print(f"  overconf_rate vs {out:16s}: rho={r:+.2f} p={p:.3f}  | partial(ctrl error_rate) rho={pr:+.2f} p={pp:.3f}")
print("\n  (reference) overall error_rate vs collisions_rate: rho=%.2f p=%.3f"%spearmanr(op.errors_rate,op.collisions_rate))
print("\nCONTINUOUS overconfidence index = mean confidence on incorrect trials (n operators with >=1 error):")
sub=op.dropna(subset=['conf_when_wrong'])
print(f"  available for {len(sub)}/37 operators")
for out in ['collisions_rate','speeding_rate','hazard_rate','mean_speed']:
    r,p=spearmanr(sub['conf_when_wrong'], sub[out])
    print(f"  conf_when_wrong vs {out:16s}: rho={r:+.2f} p={p:.3f}")

# ---------- TRIAL LEVEL (better powered) ----------
print("\n"+"="*72)
print("TRIAL-LEVEL (n=659): does an overconfident error co-occur with risk on the SAME query?")
print("="*72)
for out in ['collision','speeding','hazard_any']:
    a=m[m.overconf_err==1][out]; b=m[m.overconf_err==0][out]
    _,p=fisher_exact([[int(a.sum()),len(a)-int(a.sum())],[int(b.sum()),len(b)-int(b.sum())]])
    print(f"  {out:12s}: overconf-err trials {int(a.sum())}/{len(a)} ({100*a.mean():.0f}%) vs others {int(b.sum())}/{len(b)} ({100*b.mean():.1f}%)  Fisher p={p:.3f}")
