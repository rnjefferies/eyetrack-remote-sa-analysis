"""
Figure S5: Post-Query Collisions and the Operator Stability Buffer.
(a) post-query collision rate by SA outcome, overall rate dashed;
(b) operator-level pre-probe steering variance vs collision rate, with a LOWESS
    guide line and a Spearman rank correlation. The LOWESS line is illustrative;
    the reported association is the rank correlation. Includes an outlier check.
Scope: the Sign + Animal modelling sample (n = 439, 37 operators).
Output: outputs/FigureS5_collisions_stability.png
"""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess

df = pd.read_csv('data/master_routes_1_to_6_latency.csv')
d = df[df['Question_Type'].isin(['Sign', 'Animal'])]
C = 'Target_Post_Q_Collision'

fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.2))
cats = [('correct', d[d.Target_Accuracy == 1], '#2ca02c'),
        ('incorrect', d[d.Target_Accuracy == 0], '#d62728'),
        ('delayed', d[d.Target_Latency >= 3.5], '#1f77b4'),
        ('unsure', d[d.Target_Confidence <= 4], '#9467bd')]
a.bar([c[0] for c in cats], [c[1][C].mean() * 100 for c in cats],
      color=[c[2] for c in cats], width=0.7)
ov = d[C].mean() * 100
a.axhline(ov, ls='--', color='gray', lw=1.3, label=f'overall ({ov:.1f}%)')
a.set_ylabel('Post-query collision rate (%)'); a.set_title('Collision rate by SA outcome')
a.legend(); a.set_ylim(0, 16)

op = d.groupby('Participant_ID').agg(steer=('Before_Steer_Variance', 'mean'),
                                     coll=(C, 'mean')).reset_index()
op['coll'] *= 100
r, p = spearmanr(op.steer, op.coll)
b.scatter(op.steer, op.coll, s=55, color='#2f3b52', alpha=0.85, edgecolor='white', linewidth=0.5)
lo = lowess(op.coll, op.steer, frac=0.75)
b.plot(lo[:, 0], lo[:, 1], color='#d62728', lw=2.2, label='LOWESS (illustrative)')
b.set_xlabel('Operator mean pre-probe steering variance')
b.set_ylabel('Operator collision rate (%)')
b.set_title(f'Operator buffer (Spearman rho = +{r:.2f}, p = {p:.3f})')
b.legend(loc='upper left')
plt.tight_layout()
out = 'outputs/FigureS5_collisions_stability.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'Saved -> {out}')

imax, ims = op['coll'].idxmax(), op['steer'].idxmax()
print(f'Operator buffer: Spearman rho = {r:+.3f} (p = {p:.3f}), n = {len(op)} operators')
for lbl, idx in [('drop highest-collision op', [imax]),
                 ('drop highest-steering op', [ims]),
                 ('drop both', [imax, ims])]:
    rr, pp = spearmanr(op.drop(idx).steer, op.drop(idx).coll)
    print(f'  {lbl:26s}: rho = {rr:+.3f}, p = {pp:.3f}')
