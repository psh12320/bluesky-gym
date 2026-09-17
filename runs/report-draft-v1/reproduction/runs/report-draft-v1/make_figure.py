from pathlib import Path
import sys, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from atc.compare import load_evaluation
sources=[('Classical','#64748b','runs/goal-route-choice-v1-fast-joint-sa-200'),('Original reward','#2563eb','runs/fast-reference-v1-ma25k-sa/validation-200'),('Arrival reward 250','#c2410c','runs/fast-reference-reach250-v1-ma25k-sa/validation-200')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(6.26,2.8),layout='constrained')
for label,color,prefix in sources:
    meta,rows=load_evaluation(ROOT/prefix)
    assert meta['seed']==2026 and meta['episodes']==200
    vals=np.sort([r['flight_time'] for r in rows])
    axs[0].step(np.r_[0,vals],np.r_[0,np.arange(1,201)/2],where='post',color=color,lw=1.4,label=label)
    vals=np.array([r['intrusion_time'] for r in rows])
    x=np.unique(np.r_[0,vals])
    axs[1].step(x,[100*np.mean(vals>v) for v in x],where='post',color=color,lw=1.4)
axs[0].set(title='Flight duration',xlabel='Seconds',ylabel='At or below duration (%)',xlim=(0,3000),ylim=(0,101))
axs[1].set(title='Conflict exposure',xlabel='Intrusion seconds',ylabel='Above exposure (%)',xlim=(0,450),ylim=(0,36))
for ax in axs:ax.grid(alpha=.2)
fig.suptitle('SA development distributions: 200 scenarios',fontsize=11)
fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncols=3,fontsize=8,frameon=False)
folder=Path(__file__).resolve().parent
fig.savefig(folder/'sa-distributions.png',dpi=240)
fig.savefig(folder/'sa-distributions.svg')
plt.close(fig)
