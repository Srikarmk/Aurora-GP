"""
DIFF TEST: do the headline conclusions hold on a DISJOINT set of seeds?

The reported results use seeds 0-9. This re-runs the identical protocol on seeds
10-19 and compares per-dataset winners. A conclusion that moves when the seeds move
is not a conclusion.

Significance on both seed sets, with multiple-comparison correction, is computed by
significance.py from the two result files.
"""
import sys, json, numpy as np, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))
P = Path(__file__).parent.parent

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
