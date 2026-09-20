"""
AURORA - Fair re-run of the benchmark.

This file replaces the evaluation protocol, not the idea. Each change below
maps to a specific defect found in the audit (see AUDIT.md):

  Fix 1  Kernel lengthscale is fitted by marginal likelihood in the SAME space
         the kernel operates in (standardized). The original took the median
         pairwise distance of RAW X and applied it to standardized X, which is
         wrong by up to 155,000x and collapsed the baselines to constant
         predictors.                       [approximation_benchmark.py:85-88]

  Fix 2  Every method gets the same tuning budget. RFF and Nystrom now fit
         lengthscale, signal variance AND noise by marginal likelihood, exactly
         as the exact GP does. Previously sigma_noise was hardcoded to 0.1 and
         there was no signal variance at all.   [approximations.py:17,143,211]

  Fix 3  Correct sparse-GP predictive variance, including the data-dependent
         term, and no variance clipping. The original omitted the data term and
         clipped variance to [sigma^2, 3.0], so a magic constant was setting the
         reported calibration.                  [approximations.py:218-219]

  Fix 4  One split per (dataset, seed), shared by every method. The original
         evaluated AURORA on 2000 points and the baselines on 9146/9787 for
         protein and sarcos.                    [aurora_final.py:349-352]

  Fix 5  Capacity is selected on a validation split; test is touched once, for
         reporting. The original selected the winning config on the test set.
                                                [tune_aurora.py:187,240,290]

  Fix 6  A single ECE estimator (exact norm.ppf) for every method. The original
         used an unseeded Monte-Carlo z-score for the exact GP only, making that
         row non-reproducible and a different metric.   [gp_baseline.py:254]

  Fix 7  5 seeds, reported as mean +/- std. The original was a single seed with
         no error bars.

It also adds the control the original never ran: identical models with the
region assignment randomly permuted. If importance-based routing carries signal,
AURORA must beat that control.
"""

import numpy as np
import json, time, warnings, sys, os, io, contextlib
from pathlib import Path
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.spatial.distance import pdist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))
PROJECT_ROOT = Path(__file__).parent.parent

from gp_baseline import GaussianProcessBaseline      # repo's own exact GP, unchanged
from region_identification import RegionIdentifier   # repo's own Stage 1, unchanged

MAX_N = 8000          # dataset cap, applied identically to every method
JITTER = 1e-6


@contextlib.contextmanager
def quiet():
    """The repo's classes print heavily; keep the benchmark log readable."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


# ---------------------------------------------------------------------------
# Metrics - ONE implementation, used by every method (Fix 6)
# ---------------------------------------------------------------------------
def m_rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def m_nll(y, p, s):
    v = np.maximum(s, 1e-12) ** 2
    return float(np.mean(0.5 * np.log(2 * np.pi * v) + (y - p) ** 2 / (2 * v)))


def m_ece(y, p, s, n_bins=10):
    """Mean |nominal - empirical| coverage of central predictive intervals."""
    err = np.abs(y - p)
    tot = 0.0
    for c in np.linspace(0.1, 0.9, n_bins):
        tot += abs((err <= s * norm.ppf((1 + c) / 2)).mean() - c) / n_bins
    return float(tot)


def m_r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def all_metrics(y, p, s):
    return {'rmse': m_rmse(y, p), 'nll': m_nll(y, p, s),
            'ece': m_ece(y, p, s), 'r2': m_r2(y, p)}


# ---------------------------------------------------------------------------
# Shared linear-Gaussian model:  f = Phi(theta) w,  w ~ N(0, I),  y = f + eps
# RFF and Nystrom differ only in how Phi is built, so they get identical
# hyperparameter fitting and an identical (correct) predictive variance.
# ---------------------------------------------------------------------------
class FeatureGP:
    """Sparse GP in feature space with marginal-likelihood hyperparameters."""

    def __init__(self, capacity, random_state=42, n_hyp=2000):
        self.capacity = capacity
        self.random_state = random_state
        self.n_hyp = n_hyp          # subsample used for hyperparameter fitting only
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.theta = None
        self.training_time = 0.0

    # --- subclasses implement these -------------------------------------
    def _init_basis(self, Xs):
        raise NotImplementedError

    def _features(self, Xs, log_ell, log_sf):
        raise NotImplementedError

    # --- shared machinery ------------------------------------------------
    def _posterior(self, Phi, y, sn2):
        """Return (chol of A, Phi^T y) where A = I + Phi^T Phi / sn2."""
        D = Phi.shape[1]
        A = Phi.T @ Phi / sn2 + np.eye(D)
        L = cholesky(A, lower=True)
        return L, Phi.T @ y

    def _neg_log_ml(self, theta, Xs, y):
        log_ell, log_sf, log_sn = theta
        sn2 = np.exp(2 * log_sn)
        try:
            Phi = self._features(Xs, log_ell, log_sf)
            L, Phity = self._posterior(Phi, y, sn2)
        except np.linalg.LinAlgError:
            return 1e10
        n = len(y)
        v = cho_solve((L, True), Phity)
        quad = (y @ y - (Phity @ v) / sn2) / sn2
        logdet = n * np.log(sn2) + 2 * np.sum(np.log(np.diag(L)))
        val = 0.5 * (n * np.log(2 * np.pi) + logdet + quad)
        return val if np.isfinite(val) else 1e10

    def fit(self, X_train, y_train):
        t0 = time.time()
        Xs = self.scaler_X.fit_transform(X_train)
        ys = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        self._init_basis(Xs)

        # hyperparameter fitting on a subsample (identical policy for RFF/Nystrom)
        rs = np.random.RandomState(self.random_state)
        if len(Xs) > self.n_hyp:
            idx = rs.choice(len(Xs), self.n_hyp, replace=False)
            Xh, yh = Xs[idx], ys[idx]
        else:
            Xh, yh = Xs, ys

        # Fix 1: the median heuristic is computed in the space the kernel sees.
        sub = rs.choice(len(Xh), min(1000, len(Xh)), replace=False)
        ell0 = np.median(pdist(Xh[sub]))
        ell0 = ell0 if np.isfinite(ell0) and ell0 > 1e-6 else 1.0

        bounds = [(np.log(1e-2), np.log(1e3)),    # lengthscale
                  (-4.0, 4.0),                    # log signal std
                  (np.log(1e-3), np.log(3.0))]    # log noise std
        best, best_v = None, np.inf
        for l0 in (ell0, ell0 * 0.5, ell0 * 2.0):
            x0 = np.array([np.log(l0), 0.0, np.log(0.3)])
            try:
                r = minimize(self._neg_log_ml, x0, args=(Xh, yh),
                             method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 60})
                if r.fun < best_v:
                    best_v, best = r.fun, r.x
            except Exception:
                continue
        self.theta = best if best is not None else np.array([np.log(ell0), 0.0, np.log(0.3)])

        # final fit on ALL training data with the selected hyperparameters
        log_ell, log_sf, log_sn = self.theta
        self.sn2 = np.exp(2 * log_sn)
        Phi = self._features(Xs, log_ell, log_sf)
        self.L, Phity = self._posterior(Phi, ys, self.sn2)
        self.mu_w = cho_solve((self.L, True), Phity / self.sn2)
        self.training_time = time.time() - t0
        return self

    def predict(self, X_test, return_std=True):
        Xs = self.scaler_X.transform(X_test)
        log_ell, log_sf, _ = self.theta
        Phi = self._features(Xs, log_ell, log_sf)
        mean = self.scaler_y.inverse_transform((Phi @ self.mu_w).reshape(-1, 1)).ravel()
        if not return_std:
            return mean
        # Fix 3: full predictive variance, data term included, no clipping.
        V = cho_solve((self.L, True), Phi.T)
        var = np.einsum('ij,ji->i', Phi, V) + self.sn2
        std = np.sqrt(np.maximum(var, 1e-12)) * self.scaler_y.scale_
        return mean, std

    @property
    def fitted(self):
        log_ell, log_sf, log_sn = self.theta
        return {'lengthscale': float(np.exp(log_ell)),
                'signal_std': float(np.exp(log_sf)),
                'noise_std': float(np.exp(log_sn))}


class RFF(FeatureGP):
    """Random Fourier Features, hyperparameters fitted by marginal likelihood."""

    def _init_basis(self, Xs):
        rs = np.random.RandomState(self.random_state)
        self.W = rs.randn(Xs.shape[1], self.capacity)   # omega = W / ell
        self.b = rs.uniform(0, 2 * np.pi, self.capacity)

    def _features(self, Xs, log_ell, log_sf):
        proj = (Xs @ self.W) / np.exp(log_ell) + self.b
        return np.exp(log_sf) * np.sqrt(2.0 / self.capacity) * np.cos(proj)


class Nystrom(FeatureGP):
    """Nystrom / subset-of-regressors, hyperparameters fitted by marginal likelihood."""

    def _init_basis(self, Xs):
        rs = np.random.RandomState(self.random_state)
        m = min(self.capacity, len(Xs))
        self.Z = Xs[rs.choice(len(Xs), m, replace=False)]
        self.m = m

    @staticmethod
    def _rbf(A, B, ell):
        d2 = (np.sum(A ** 2, 1)[:, None] + np.sum(B ** 2, 1)[None, :]
              - 2 * A @ B.T)
        return np.exp(-np.maximum(d2, 0) / (2 * ell ** 2))

    def _features(self, Xs, log_ell, log_sf):
        ell = np.exp(log_ell)
        Kmm = self._rbf(self.Z, self.Z, ell) + JITTER * np.eye(self.m)
        Lm = cholesky(Kmm, lower=True)
        Knm = self._rbf(Xs, self.Z, ell)
        # Phi Phi^T = sf^2 * Knm Kmm^-1 Kmn  (exactly the Nystrom kernel)
        return np.exp(log_sf) * solve_triangular(Lm, Knm.T, lower=True).T


# ---------------------------------------------------------------------------
# AURORA, rebuilt on properly fitted components
# ---------------------------------------------------------------------------
class AuroraFair:
    """Region-routed predictor. Stage 1 is the repo's own RegionIdentifier."""

    def __init__(self, models, region_identifier):
        self.models = models                      # {0: low, 1: medium, 2: high}
        self.ri = region_identifier

    def predict(self, X_test, permute_seed=None):
        with quiet():
            regions, _ = self.ri.predict_regions(X_test, adaptive_thresholds=False)
        if permute_seed is not None:
            # Control: keep the region proportions, destroy the importance signal.
            regions = np.random.RandomState(permute_seed).permutation(regions)
        p = np.zeros(len(X_test))
        s = np.zeros(len(X_test))
        for r in (0, 1, 2):
            mask = regions == r
            if mask.sum():
                pr, sr = self.models[r].predict(X_test[mask], return_std=True)
                p[mask], s[mask] = pr, sr
        return p, s, regions


# ---------------------------------------------------------------------------
# One (dataset, seed) cell of the experiment
# ---------------------------------------------------------------------------
def run_cell(X, y, seed, log=print):
    """All methods share one split; hyperparameters from train, capacity from val."""
    rs = np.random.RandomState(seed)
    if len(X) > MAX_N:
        idx = rs.choice(len(X), MAX_N, replace=False)
        X, y = X[idx], y[idx]

    # Fix 4: one split, every method sees exactly these points.
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=seed)

    out = {'n_train': len(X_tr), 'n_val': len(X_va), 'n_test': len(X_te)}

    # --- capacity selected on VALIDATION (Fix 5) ------------------------
    def pick(cls, grid):
        best, best_nll, chosen = None, np.inf, None
        for c in grid:
            m = cls(capacity=c, random_state=seed).fit(X_tr, y_tr)
            p, s = m.predict(X_va)
            v = m_nll(y_va, p, s)
            if v < best_nll:
                best_nll, best, chosen = v, m, c
        return best, chosen

    t0 = time.time()
    rff, rff_c = pick(RFF, [200, 1000])
    nys, nys_c = pick(Nystrom, [200, 500])
    approx_time = time.time() - t0

    # --- exact GP: the repo's own class, unchanged ----------------------
    t0 = time.time()
    with quiet():
        gp = GaussianProcessBaseline(max_train_size=5000, random_state=seed, use_gpu=False)
        gp.fit(X_tr, y_tr, n_iter=50)
    gp_time = time.time() - t0

    # --- Stage 1 region identifier: the repo's own class, unchanged -----
    t0 = time.time()
    with quiet():
        ri = RegionIdentifier(max_gp_samples=5000, random_state=seed)
        ri.fit(X_tr, y_tr, n_gp_iter=30)
    ri_time = time.time() - t0

    # high tier for the all-approximation variant
    nys_hi = Nystrom(capacity=1500, random_state=seed).fit(X_tr, y_tr)

    # --- evaluate every arm on the SAME test set ------------------------
    res = {}

    p, s = rff.predict(X_te);  res['rff_uniform'] = all_metrics(y_te, p, s)
    res['rff_uniform'].update(train_time=approx_time / 2, capacity=rff_c, **rff.fitted)

    p, s = nys.predict(X_te);  res['nystrom_uniform'] = all_metrics(y_te, p, s)
    res['nystrom_uniform'].update(train_time=approx_time / 2, capacity=nys_c, **nys.fitted)

    p, s = gp.predict(X_te);   res['exact_gp'] = all_metrics(y_te, p, s)
    res['exact_gp'].update(train_time=gp_time)

    # AURORA with the exact GP in the high tier (the repo's Config 3, repaired)
    a_gp = AuroraFair({0: rff, 1: nys, 2: gp}, ri)
    p, s, reg = a_gp.predict(X_te)
    res['aurora_gp'] = all_metrics(y_te, p, s)
    res['aurora_gp'].update(train_time=approx_time + gp_time + ri_time,
                            per_region=_per_region(y_te, p, s, reg))

    # Control: same three models, routing signal destroyed
    p, s, _ = a_gp.predict(X_te, permute_seed=seed)
    res['aurora_gp_shuffled'] = all_metrics(y_te, p, s)

    # AURORA with no exact GP anywhere - isolates routing from the GP swap
    a_ap = AuroraFair({0: rff, 1: nys, 2: nys_hi}, ri)
    p, s, reg = a_ap.predict(X_te)
    res['aurora_approx'] = all_metrics(y_te, p, s)
    res['aurora_approx'].update(train_time=approx_time + nys_hi.training_time + ri_time,
                                per_region=_per_region(y_te, p, s, reg))

    p, s, _ = a_ap.predict(X_te, permute_seed=seed)
    res['aurora_approx_shuffled'] = all_metrics(y_te, p, s)

    out['methods'] = res
    return out


def _per_region(y, p, s, reg):
    names = {0: 'low', 1: 'medium', 2: 'high'}
    d = {}
    for r, nm in names.items():
        mask = reg == r
        d[nm] = ({'n': int(mask.sum()), **all_metrics(y[mask], p[mask], s[mask])}
                 if mask.sum() > 5 else {'n': int(mask.sum())})
    return d


def main(seeds=(0, 1, 2, 3, 4), out_name='fair'):
    data_dir = PROJECT_ROOT / 'data'
    out_dir = PROJECT_ROOT / 'results' / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    all_res = {}
    for f in sorted(data_dir.glob('*.npz')):
        name = f.stem
        d = np.load(f, allow_pickle=True)
        X, y = d['X'], d['y']
        all_res[name] = {}
        for seed in seeds:
            t0 = time.time()
            print(f"[{name}] seed {seed} ...", flush=True)
            try:
                all_res[name][str(seed)] = run_cell(X, y, seed)
                print(f"[{name}] seed {seed} done in {time.time()-t0:.1f}s", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                all_res[name][str(seed)] = {'error': str(e)}
            with open(out_dir / 'raw_results.json', 'w') as fh:
                json.dump(all_res, fh, indent=2)
    print("saved ->", out_dir / 'raw_results.json')
    return all_res


if __name__ == '__main__':
    main()
