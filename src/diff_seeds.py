"""
DIFF TEST: do the headline conclusions hold on a DISJOINT set of seeds?

The reported results use seeds 0-9. This re-runs the identical protocol on seeds
10-19 and compares per-dataset winners. A conclusion that moves when the seeds move
is not a conclusion.

Result at the time of writing: winners agree on 10 of 11 datasets. The ECE best-arm
count is global 9 / regional 2 on seeds 0-9 and global 8 / regional 3 on seeds 10-19;
the single flip is powerplant, whose global-vs-regional gap is small (0.0197 vs
0.0218). The heteroscedastic GP wins zero datasets on ECE in BOTH seed sets.
"""
import sys, json, numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'/Users/srikarmk/Programming/Aurora-GP/src')
from pathlib import Path
P=Path('/Users/srikarmk/Programming/Aurora-GP')

import hetero_fair as HF
HF.SEEDS=tuple(range(10,20))                      # DISJOINT from the reported 0-9
screen={d['dataset']:d['het_ratio'] for d in json.load(open(P/'results/hetero_screen.json'))}
files=[(p.stem,p) for p in sorted((P/'data/real').glob('*.npz'))]
files+=[(p.stem,p) for p in sorted((P/'data').glob('*.npz'))]
R={}
for name,path in files:
    z=np.load(path,allow_pickle=True)
    R[name]={'het_ratio':screen.get(name),'seeds':{str(s):HF.run(z['X'],z['y'],s) for s in HF.SEEDS}}
    print(f'  {name} done',flush=True)
    json.dump(R,open(P/'results/hetero_fair_seeds10_19.json','w'),indent=2)
for met in ('ece','nll'):
    tal={}
    for n,d in R.items():
        c=list(d['seeds'].values())
        col={k:np.mean([x[k][met] for x in c if k in x]) for k in ('global','het_gp','regional')}
        w=min(col,key=col.get); tal[w]=tal.get(w,0)+1
    print(f'{met.upper()} best-arm count (seeds 10-19): {tal}')
