"""
Two experiments that decide whether either finding is real.

EXPERIMENT A - "rank sweep": is the calibration advantage caused by the
low-rank approximation gap?

    Hyperparameters (lengthscale, signal variance, noise) are fitted ONCE per
    (dataset, seed) by exact marginal likelihood, then held FIXED while only the
    Nystrom rank m varies, from m=25 up to the exact GP (m = n_train). Nothing
    else changes - same data, same kernel, same noise. So any trend in ECE is
    attributable to the approximation gap alone, not to different tuning.

    Prediction if the mechanism is real: ECE degrades monotonically toward the
    exact GP's as m grows, and the predictive variance shrinks toward it.

EXPERIMENT B - "oracle routing": is there ANY headroom for region-aware routing?

    For every test point we evaluate all three models, then compare:
      uniform-best  : the single best model overall   (chosen on validation)
      oracle        : the best model per region       (chosen on validation)
      importance    : AURORA's actual routing
      random        : routing shuffled, proportions preserved
    The oracle is the best any region-aware router could do with these models and
    this partition. If oracle ~= uniform-best, region-aware routing has nothing
    to win, regardless of how good the importance criterion is.
"""

import numpy as np
import json, time, sys, warnings
from pathlib import Path
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))
PROJECT_ROOT = Path(__file__).parent.parent

from fair_benchmark import (m_rmse, m_nll, m_ece, m_r2, all_metrics, RFF, Nystrom,
                            MAX_N, JITTER, quiet)
from region_identification import RegionIdentifier

RANKS = [25, 50, 100, 200, 400, 800, 1600]


# ---------------------------------------------------------------------------
# Exact GP in numpy, with hyperparameters we control (the m = n endpoint)
# ---------------------------------------------------------------------------
def rbf(A, B, ell):
    d2 = np.sum(A ** 2, 1)[:, None] + np.sum(B ** 2, 1)[None, :] - 2 * A @ B.T
    return np.exp(-np.maximum(d2, 0) / (2 * ell ** 2))


def exact_gp_nlml(theta, Xs, ys):
    ell, sf, sn = np.exp(theta)
    n = len(ys)
    K = sf ** 2 * rbf(Xs, Xs, ell) + (sn ** 2 + JITTER) * np.eye(n)
    try:
        L = cholesky(K, lower=True)
    except np.linalg.LinAlgError:
        return 1e10
    a = cho_solve((L, True), ys)
    v = 0.5 * (ys @ a) + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)
    return v if np.isfinite(v) else 1e10


def fit_exact_hypers(Xs, ys, seed, n_fit=1500):
    """Fit (ell, sf, sn) by exact marginal likelihood on a subsample."""
    rs = np.random.RandomState(seed)
    idx = rs.choice(len(Xs), min(n_fit, len(Xs)), replace=False)
    Xf, yf = Xs[idx], ys[idx]
    ell0 = np.median(pdist(Xf[rs.choice(len(Xf), min(800, len(Xf)), replace=False)]))
    ell0 = ell0 if np.isfinite(ell0) and ell0 > 1e-6 else 1.0
    bounds = [(np.log(1e-2), np.log(1e3)), (-4, 4), (np.log(1e-3), np.log(3.0))]
    best, bv = None, np.inf
    for m in (1.0, 0.5, 2.0):
        x0 = np.array([np.log(ell0 * m), 0.0, np.log(0.3)])
        try:
            r = minimize(exact_gp_nlml, x0, args=(Xf, yf), method='L-BFGS-B',
                         bounds=bounds, options={'maxiter': 60})
            if r.fun < bv:
                bv, best = r.fun, r.x
        except Exception:
            pass
    return np.exp(best if best is not None else np.array([np.log(ell0), 0., np.log(.3)]))


def exact_gp_predict(Xs_tr, ys_tr, Xs_te, ell, sf, sn):
    n = len(ys_tr)
    K = sf ** 2 * rbf(Xs_tr, Xs_tr, ell) + (sn ** 2 + JITTER) * np.eye(n)
    L = cholesky(K, lower=True)
    Ks = sf ** 2 * rbf(Xs_te, Xs_tr, ell)
    mean = Ks @ cho_solve((L, True), ys_tr)
    V = solve_triangular(L, Ks.T, lower=True)
    var = sf ** 2 - np.sum(V ** 2, 0) + sn ** 2
    return mean, np.sqrt(np.maximum(var, 1e-12))


def nystrom_fixed(Xs_tr, ys_tr, Xs_te, m, ell, sf, sn, seed):
    """Nystrom at rank m with the SAME hyperparameters as the exact GP."""
    rs = np.random.RandomState(seed)
    m = min(m, len(Xs_tr))
    Z = Xs_tr[rs.choice(len(Xs_tr), m, replace=False)]
    Kmm = rbf(Z, Z, ell) + JITTER * np.eye(m)
    Lm = cholesky(Kmm, lower=True)
    Phi = sf * solve_triangular(Lm, rbf(Xs_tr, Z, ell).T, lower=True).T
    Phs = sf * solve_triangular(Lm, rbf(Xs_te, Z, ell).T, lower=True).T
    sn2 = sn ** 2
    A = Phi.T @ Phi / sn2 + np.eye(m)
    La = cholesky(A, lower=True)
    mu_w = cho_solve((La, True), Phi.T @ ys_tr / sn2)
    mean = Phs @ mu_w
    var = np.einsum('ij,ji->i', Phs, cho_solve((La, True), Phs.T)) + sn2
    return mean, np.sqrt(np.maximum(var, 1e-12))


def z_dispersion(y, p, s):
    """std of standardized residuals. 1.0 = calibrated, >1 = overconfident."""
    return float(np.std((y - p) / np.maximum(s, 1e-12)))


# ---------------------------------------------------------------------------
def experiment_A(X, y, seed):
    rs = np.random.RandomState(seed)
    if len(X) > MAX_N:
        i = rs.choice(len(X), MAX_N, replace=False); X, y = X[i], y[i]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=seed)
    _, X_te, _, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=seed)

    sx, sy = StandardScaler(), StandardScaler()
    Xs = sx.fit_transform(X_tr); ys = sy.fit_transform(y_tr.reshape(-1, 1)).ravel()
    Xt = sx.transform(X_te)
    scale = sy.scale_[0]

    ell, sf, sn = fit_exact_hypers(Xs, ys, seed)
    out = {'hypers': {'lengthscale': float(ell), 'signal_std': float(sf),
                      'noise_std': float(sn)}, 'n_train': len(X_tr), 'ranks': {}}

    for m in RANKS:
        if m > len(Xs):
            continue
        mu, sd = nystrom_fixed(Xs, ys, Xt, m, ell, sf, sn, seed)
        p = sy.inverse_transform(mu.reshape(-1, 1)).ravel(); s = sd * scale
        out['ranks'][str(m)] = {**all_metrics(y_te, p, s),
                                'mean_std': float(s.mean()),
                                'z_disp': z_dispersion(y_te, p, s)}
    mu, sd = exact_gp_predict(Xs, ys, Xt, ell, sf, sn)
    p = sy.inverse_transform(mu.reshape(-1, 1)).ravel(); s = sd * scale
    out['exact'] = {**all_metrics(y_te, p, s), 'mean_std': float(s.mean()),
                    'z_disp': z_dispersion(y_te, p, s), 'rank': len(Xs)}
    return out


def experiment_B(X, y, seed):
    rs = np.random.RandomState(seed)
    if len(X) > MAX_N:
        i = rs.choice(len(X), MAX_N, replace=False); X, y = X[i], y[i]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=seed)

    # three fitted models (same protocol as the fair benchmark)
    models = {'rff': RFF(1000, random_state=seed).fit(X_tr, y_tr),
              'nys': Nystrom(500, random_state=seed).fit(X_tr, y_tr),
              'nys_hi': Nystrom(1500, random_state=seed).fit(X_tr, y_tr)}

    sx, sy = StandardScaler(), StandardScaler()
    Xs = sx.fit_transform(X_tr); ys = sy.fit_transform(y_tr.reshape(-1, 1)).ravel()
    ell, sf, sn = fit_exact_hypers(Xs, ys, seed)

    def gp_pred(Xq):
        mu, sd = exact_gp_predict(Xs, ys, sx.transform(Xq), ell, sf, sn)
        return sy.inverse_transform(mu.reshape(-1, 1)).ravel(), sd * sy.scale_[0]

    names = ['rff', 'nys', 'nys_hi', 'gp']
    pv, pt = {}, {}
    for k, mo in models.items():
        pv[k] = mo.predict(X_va); pt[k] = mo.predict(X_te)
    pv['gp'] = gp_pred(X_va); pt['gp'] = gp_pred(X_te)

    with quiet():
        ri = RegionIdentifier(max_gp_samples=5000, random_state=seed)
        ri.fit(X_tr, y_tr, n_gp_iter=30)
        reg_va, _ = ri.predict_regions(X_va, adaptive_thresholds=False)
        reg_te, _ = ri.predict_regions(X_te, adaptive_thresholds=False)

    # per-region ECE of every model on the test set (the matrix nobody computed)
    matrix = {}
    for k in names:
        p, s = pt[k]
        matrix[k] = {'overall': m_ece(y_te, p, s)}
        for r, nm in {0: 'low', 1: 'medium', 2: 'high'}.items():
            mk = reg_te == r
            matrix[k][nm] = m_ece(y_te[mk], p[mk], s[mk]) if mk.sum() > 5 else None

    def assemble(choice, regions):
        p = np.zeros(len(y_te)); s = np.zeros(len(y_te))
        for r in (0, 1, 2):
            mk = regions == r
            if mk.sum():
                pk, sk = pt[choice[r]]
                p[mk], s[mk] = pk[mk], sk[mk]
        return p, s

    # uniform-best: one model for everything, selected on validation ECE
    best_uni = min(names, key=lambda k: m_ece(y_va, *pv[k]))
    p, s = pt[best_uni]
    res = {'uniform_best': {'model': best_uni, **all_metrics(y_te, p, s)}}

    # oracle: best model PER REGION, still selected on validation
    choice = {}
    for r in (0, 1, 2):
        mk = reg_va == r
        if mk.sum() > 5:
            choice[r] = min(names, key=lambda k: m_ece(y_va[mk], pv[k][0][mk], pv[k][1][mk]))
        else:
            choice[r] = best_uni
    p, s = assemble(choice, reg_te)
    res['oracle_region'] = {'choice': {str(k): v for k, v in choice.items()},
                            **all_metrics(y_te, p, s)}

    # AURORA's actual routing, and the shuffled control, with the same models
    aur = {0: 'rff', 1: 'nys', 2: 'gp'}
    p, s = assemble(aur, reg_te)
    res['importance_routing'] = all_metrics(y_te, p, s)
    p, s = assemble(aur, np.random.RandomState(seed).permutation(reg_te))
    res['random_routing'] = all_metrics(y_te, p, s)

    res['per_region_ece'] = matrix
    res['region_counts'] = {nm: int((reg_te == r).sum())
                            for r, nm in {0: 'low', 1: 'medium', 2: 'high'}.items()}
    return res


def main(seeds=(0, 1, 2, 3, 4)):
    out_dir = PROJECT_ROOT / 'results' / 'mechanism'
    out_dir.mkdir(parents=True, exist_ok=True)
    A, B = {}, {}
    for f in sorted((PROJECT_ROOT / 'data').glob('*.npz')):
        ds = f.stem
        d = np.load(f, allow_pickle=True); X, y = d['X'], d['y']
        A[ds], B[ds] = {}, {}
        for seed in seeds:
            t0 = time.time()
            print(f"[A {ds}] seed {seed}", flush=True)
            A[ds][str(seed)] = experiment_A(X, y, seed)
            print(f"[B {ds}] seed {seed}", flush=True)
            B[ds][str(seed)] = experiment_B(X, y, seed)
            print(f"   done {time.time()-t0:.1f}s", flush=True)
            json.dump(A, open(out_dir / 'rank_sweep.json', 'w'), indent=2)
            json.dump(B, open(out_dir / 'oracle_routing.json', 'w'), indent=2)
    print("saved ->", out_dir)


if __name__ == '__main__':
    main()
