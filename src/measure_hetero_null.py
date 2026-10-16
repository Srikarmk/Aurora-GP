"""
A permutation null for het_ratio.

het_ratio (measure_hetero.py) is not 1 for data with no input-dependent noise: the p90/p10
of kNN-smoothed squared residuals depends on the residual distribution's tails and on
the test-set size. Shuffling the residuals over X keeps both and destroys any relation
to X. Reported per dataset over seeds 0-9: observed ratio, mean null ratio, their
quotient, and the share of 100 permutations at least as large as the observed ratio.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import json, sys, warnings, numpy as np
from pathlib import Path
from multiprocessing import Pool

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))
P = Path(__file__).parent.parent

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from measure_hetero import CAP, K
from mechanism import fit_exact_hypers, exact_gp_predict
from structured_data import GENERATORS

SEEDS = range(10)
N_PERM = 100


def load(name):
    if name in GENERATORS:
        X, y, _ = GENERATORS[name](seed=0); return X, y
    f = P / 'data' / f'{name}.npz'
    f = f if f.exists() else P / 'data' / 'real' / f'{name}.npz'
    z = np.load(f, allow_pickle=True); return z['X'], z['y']


def job(arg):
    name, seed = arg
    X, y = load(name)
    rs = np.random.RandomState(seed)
    if len(X) > CAP:
        i = rs.choice(len(X), CAP, replace=False); X, y = X[i], y[i]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)
    sx, sy = StandardScaler(), StandardScaler()
    Xs = sx.fit_transform(X_tr); ys = sy.fit_transform(y_tr.reshape(-1, 1)).ravel()
    Xq = sx.transform(X_te)
    ell, sf, sn = fit_exact_hypers(Xs, ys, seed)
    mu, _ = exact_gp_predict(Xs, ys, Xq, ell, sf, sn)
    r2 = (y_te - sy.inverse_transform(mu.reshape(-1, 1)).ravel()) ** 2
    idx = NearestNeighbors(n_neighbors=min(K, len(Xq))).fit(Xq).kneighbors(Xq, return_distance=False)

    def ratio(v):
        lo, hi = np.percentile(v[idx].mean(1), [10, 90]); return float(hi / max(lo, 1e-12))
    obs = ratio(r2)
    prs = np.random.RandomState(1000 + seed)
    null = np.array([ratio(prs.permutation(r2)) for _ in range(N_PERM)])
    return name, seed, obs, float(null.mean()), float(np.mean(null >= obs))


def main():
    names = sorted([f.stem for f in (P / 'data').glob('*.npz')] +
                   [f.stem for f in (P / 'data' / 'real').glob('*.npz')]) + list(GENERATORS)
    with Pool(min(16, os.cpu_count() or 4)) as pool:
        res = pool.map(job, [(n, s) for n in names for s in SEEDS])
    R = {}
    for n, s, obs, null, p in res:
        R.setdefault(n, {'obs': [], 'null': [], 'perm_p': []})
        R[n]['obs'].append(obs); R[n]['null'].append(null); R[n]['perm_p'].append(p)
    for n, v in R.items():
        v['obs_mean'] = float(np.mean(v['obs'])); v['null_mean'] = float(np.mean(v['null']))
        v['relative'] = float(np.mean(np.array(v['obs']) / np.array(v['null'])))
        v['seeds_p_below_05'] = int(sum(p < .05 for p in v['perm_p']))
    json.dump(R, open(P / 'results' / 'hetero_screen_null.json', 'w'), indent=2)
    for n, v in sorted(R.items(), key=lambda kv: -kv[1]['relative']):
        print(f"{n:28s} obs {v['obs_mean']:9.2f}  null {v['null_mean']:5.2f}  "
              f"obs/null {v['relative']:7.2f}  seeds with perm p<.05: {v['seeds_p_below_05']}/10")


if __name__ == '__main__':
    main()
