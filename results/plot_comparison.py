from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from model.ChangeForest.ChangeForest import run_changeforest
from model.iCID.iCID import run_iCID
import path

# import example participant
base_path = Path(path.DATA_PATH, '88acdccfe1ab13225e2cb86a3fe13ba4c63d4ce9f3a38f219381c124a2ff6edc')
gsr_pf = pd.read_csv(base_path / "GSR.csv")
bvp_pf = pd.read_csv(base_path / "BVP.csv")
gsr_valid = gsr_pf.dropna(subset=['shortNTPTime'])

# normalize time
X = gsr_pf.loc[gsr_pf['shortNTPTime'].notna(), 'shortNTPTime'].values
begin = X[0]
X = (X - X[0]) / 1000

# feature cols
gsr_feature_cols = ['GSR_clean', 'GSR_tonic', 'GSR_phasic', 'GSR_avg', 'GSR_std']

# features during HRI time
Y_gsr = gsr_pf[gsr_feature_cols].values[gsr_pf['shortNTPTime'].notna()]

# apply algorithms
pred_gsr_icid, _, _ = run_iCID(Y_gsr, win_size=50, alpha=1)
pred_gsr_cf = run_changeforest(
            gsr_pf, gsr_feature_cols,
            minimal_relative_segment_length=0.05,
            minimal_gain_to_split=100
        )

# plot
fig, ax = plt.subplots(figsize=(10, 2))
val = gsr_valid['GSR'].ffill().values
ax.plot(X, val,  label='GSR', linewidth=1.5)


r = gsr_pf.dropna(subset=['emotion_HRI']).iloc[0]
ax.axvline(x=((r['shortNTPTime'] - begin) / 1000), color='black', label = 'Ground Truth', linewidth=1.5, linestyle='-')
rows = list(gsr_pf.dropna(subset=['emotion_HRI']).iterrows())
for _, row in rows[1:]:
    ax.axvline(x=((row['shortNTPTime'] - begin) / 1000), color='black', linewidth=1.5, linestyle='-')
color_icid = 'orange'
style_icid = '-'
color_cf   = 'green'
style_cf   = '-'


ax.axvline(x=X[pred_gsr_icid[0]], color=color_icid, linestyle=style_icid, linewidth=1.5, alpha=0.9, label='iCID')
for idx in pred_gsr_icid[1:]:
    if idx < len(X):
        ax.axvline(x=X[idx], color=color_icid, linestyle=style_icid, linewidth=1.5, alpha=0.9)

ax.axvline(x=X[pred_gsr_cf[0]], color=color_cf, linestyle=style_cf, linewidth=1.5, alpha=0.9, label='ChangeForest')
for idx in pred_gsr_cf[1:]:
    if idx < len(X):
        ax.axvline(x=X[idx], color=color_cf, linestyle=style_cf, linewidth=1.5, alpha=0.9)
ax.set_xlabel('Time (s)')
ax.set_ylabel('GSR (µS)')
ax.legend(loc='upper right', fontsize='small')

plt.tight_layout()
out_path = Path(path.PLOTS_PATH) / "comparison.png"
fig.savefig(out_path, dpi=300, bbox_inches='tight')
