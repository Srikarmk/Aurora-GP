"""
PIVOT EXPERIMENT — approximate GP inference silently corrupts calibration metrics.

The claim: a practitioner who writes a textbook exact GP in GPyTorch and reports
ECE/NLL is, above a training-set size they never chose, reporting properties of an
approximation rather than of their model. Nothing in user code signals the switch.

Two independent mechanisms are in play, and this separates them:

  A. `gpytorch.settings.max_cholesky_size` (default 800). Above it, GPyTorch replaces
     Cholesky with iterative CG / stochastic Lanczos quadrature. The user never names
     this number; it is a library default keyed to training-set size.

  B. `gpytorch.settings.fast_pred_var()` — LOVE. An explicit opt-in, widely copied
     from the GPyTorch docs, that approximates the predictive covariance. The original
     gp_baseline.py:157 calls it.

Design: sweep n_train across the 800 boundary. For each (dataset, n, seed) fit ONE
model, then predict four ways — the only thing varying is the inference path:

    repo_default   fast_pred_var ON,  max_cholesky_size 800   (what gp_baseline.py does)
    love_off       fast_pred_var OFF, max_cholesky_size 800   (isolates B)
    cholesky_on    fast_pred_var ON,  max_cholesky_size huge  (isolates A)
    exact          fast_pred_var OFF, max_cholesky_size huge  (ground truth)

Reported per cell: ECE, NLL, and the pointwise relative error in predictive std
against `exact`, which measures the corruption directly and is independent of how
well the model happens to fit.
"""
import numpy as np, json, sys, time, warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))
PROJECT_ROOT = Path(__file__).parent.parent

import torch, gpytorch
from sklearn.model_selection import train_test_split
from fair_benchmark import m_ece, m_nll, m_rmse, quiet
from gp_baseline import GaussianProcessBaseline

N_GRID = [200, 400, 600, 800, 1000, 1500, 2500, 4000]
SEEDS = (0, 1, 2)
BIG = 100000
_OFF = dict(covar_root_decomposition=False, log_prob=False, solves=False)

MODES = {
    'repo_default': (True,  800),
    'love_off':     (False, 800),
    'cholesky_on':  (True,  BIG),
    'exact':        (False, BIG),
}


def predict_mode(g, X_test, love, chol):
    Xs = g.scaler_X.transform(X_test)
    t = torch.tensor(Xs, dtype=torch.float32).to(g.device)
    g.model.eval(); g.likelihood.eval()
    ctx = [torch.no_grad(), gpytorch.settings.max_cholesky_size(chol),
           gpytorch.settings.fast_pred_var(love)]
    if chol == BIG:
        ctx.append(gpytorch.settings.fast_computations(**_OFF))
    import contextlib
    with contextlib.ExitStack() as st:
        for c in ctx:
            st.enter_context(c)
        d = g.likelihood(g.model(t))
        m = d.mean.cpu().numpy(); v = d.variance.cpu().numpy()
    return (g.scaler_y.inverse_transform(m.reshape(-1, 1)).ravel(),
            np.sqrt(np.maximum(v, 1e-12)) * g.scaler_y.scale_)


def run(X, y, n_train, seed):
    rs = np.random.RandomState(seed)
    need = n_train + 1000
    if len(X) > need:
        i = rs.choice(len(X), need, replace=False); X, y = X[i], y[i]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, train_size=n_train, test_size=min(1000, len(X) - n_train), random_state=seed)

    with quiet():
        g = GaussianProcessBaseline(max_train_size=100000, random_state=seed, use_gpu=False)
        g.fit(X_tr, y_tr, n_iter=50)

    out = {}
    _, s_exact = predict_mode(g, X_te, *MODES['exact'])
    for name, (love, chol) in MODES.items():
        p, s = predict_mode(g, X_te, love, chol)
        out[name] = {
            'ece': m_ece(y_te, p, s), 'nll': m_nll(y_te, p, s), 'rmse': m_rmse(y_te, p),
            'mean_std': float(s.mean()),
            # direct measure of corruption: pointwise relative error vs exact
            'std_relerr_median': float(np.median(np.abs(s - s_exact) / s_exact)),
            'std_relerr_max': float(np.max(np.abs(s - s_exact) / s_exact)),
        }
    out['n_train'] = int(len(X_tr)); out['n_test'] = int(len(X_te))
    return out


def main():
    out_dir = PROJECT_ROOT / 'results' / 'inference'
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    for f in sorted((PROJECT_ROOT / 'data').glob('*.npz')):
        ds = f.stem
        d = np.load(f, allow_pickle=True); X, y = d['X'], d['y']
        res[ds] = {}
        for n in N_GRID:
            if len(X) < n + 200:
                continue
            res[ds][str(n)] = {}
            for seed in SEEDS:
                res[ds][str(n)][str(seed)] = run(X, y, n, seed)
            e = np.mean([res[ds][str(n)][str(s)]['repo_default']['ece'] for s in SEEDS])
            x = np.mean([res[ds][str(n)][str(s)]['exact']['ece'] for s in SEEDS])
            r = np.mean([res[ds][str(n)][str(s)]['repo_default']['std_relerr_median'] for s in SEEDS])
            print(f"[{ds:26s} n={n:5d}] ECE repo {e:.4f} vs exact {x:.4f}"
                  f"   median std rel.err {r*100:6.2f}%", flush=True)
            json.dump(res, open(out_dir / 'corruption.json', 'w'), indent=2)
    print("saved ->", out_dir / 'corruption.json')


if __name__ == '__main__':
    main()
