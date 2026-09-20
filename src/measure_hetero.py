"""
Measure heteroscedasticity, do not assume it.

FINDINGS.md concludes that routing helps only where noise varies across the input
space. That makes "does this dataset actually have input-dependent noise?" a
prerequisite question, not a footnote -- and it was never asked of the original five
benchmarks.

Procedure per dataset: fit an exact GP, take held-out residuals, then measure whether
residual variance depends on X.

  het_ratio  p90/p10 of kNN-smoothed local residual variance. 1.0 = homoscedastic.
  bp_r2      R^2 of regressing squared residuals on X (Breusch-Pagan style).
             Near 0 means X carries no information about the noise level.

Caveat recorded up front: residual variance that varies with X can come from genuine
heteroscedastic noise OR from residual model bias. These measures do not separate
them. For the question at hand -- would a region-dependent noise term help -- the
distinction does not matter, but it does matter for calling a dataset
"heteroscedastic", so the term used here is "input-dependent residual variance".
"""
import numpy as np, json, sys, warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))
PROJECT_ROOT = Path(__file__).parent.parent

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from mechanism import fit_exact_hypers, exact_gp_predict

CAP = 4000
K = 40


def measure(X, y, seed=0):
    rs = np.random.RandomState(seed)
    if len(X) > CAP:
        i = rs.choice(len(X), CAP, replace=False); X, y = X[i], y[i]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)
    sx, sy = StandardScaler(), StandardScaler()
    Xs = sx.fit_transform(X_tr); ys = sy.fit_transform(y_tr.reshape(-1, 1)).ravel()
    Xq = sx.transform(X_te)
    ell, sf, sn = fit_exact_hypers(Xs, ys, seed)
    mu, _ = exact_gp_predict(Xs, ys, Xq, ell, sf, sn)
    p = sy.inverse_transform(mu.reshape(-1, 1)).ravel()
    r2 = (y_te - p) ** 2

    nb = NearestNeighbors(n_neighbors=min(K, len(Xq))).fit(Xq)
    loc = r2[nb.kneighbors(Xq, return_distance=False)].mean(1)
    lo, hi = np.percentile(loc, [10, 90])
    het_ratio = float(hi / max(lo, 1e-12))

    bp = LinearRegression().fit(Xq, r2 / max(r2.mean(), 1e-12))
    ss = float(bp.score(Xq, r2 / max(r2.mean(), 1e-12)))
    return {'het_ratio': het_ratio, 'bp_r2': max(ss, 0.0),
            'n': int(len(X)), 'd': int(X.shape[1])}


def main():
    rows = []
    for d in sorted((PROJECT_ROOT / 'data').glob('*.npz')):
        z = np.load(d, allow_pickle=True)
        rows.append(('original', d.stem, measure(z['X'], z['y'])))
    for d in sorted((PROJECT_ROOT / 'data' / 'real').glob('*.npz')):
        z = np.load(d, allow_pickle=True)
        rows.append(('real', d.stem, measure(z['X'], z['y'])))
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))
    from structured_data import GENERATORS
    for k, g in GENERATORS.items():
        X, y, _ = g(seed=0)
        rows.append(('synthetic', k, measure(X, y)))

    rows.sort(key=lambda r: -r[2]['het_ratio'])
    print("=" * 84)
    print("INPUT-DEPENDENT RESIDUAL VARIANCE  (het_ratio = p90/p10 of local resid. var)")
    print("=" * 84)
    print(f"{'group':11s} {'dataset':26s} {'n':>7s} {'d':>4s} {'het_ratio':>11s} {'bp_r2':>8s}")
    print("-" * 84)
    for grp, name, m in rows:
        print(f"{grp:11s} {name:26s} {m['n']:7d} {m['d']:4d} {m['het_ratio']:11.2f} {m['bp_r2']:8.3f}")
    out = PROJECT_ROOT / 'results' / 'hetero_screen.json'
    json.dump([{'group': g, 'dataset': n, **m} for g, n, m in rows], open(out, 'w'), indent=2)
    print("\nsaved ->", out)


if __name__ == '__main__':
    main()
