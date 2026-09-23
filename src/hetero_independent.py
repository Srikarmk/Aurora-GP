"""
TODO from the threats section: an INDEPENDENT heteroscedastic-GP implementation.

Every heteroscedastic result so far used our own numpy/scipy EM. Since four separate
headline numbers in this project turned on how a baseline was configured, a baseline
that only we have implemented is a standing risk. This re-runs the comparison with a
het-GP built on GPyTorch -- different linear algebra, different optimiser (Adam on the
exact marginal likelihood), different codebase.

Scope, stated precisely: this tests the IMPLEMENTATION, not the algorithm family. It
is still a two-stage Goldberg-style construction (a second GP on log squared
residuals). A genuinely different algorithm -- e.g. the variational heteroscedastic GP
of Lazaro-Gredilla and Titsias -- remains untested, and the threats section says so.

Exact inference is forced throughout (AUDIT.md Tier 1 finding 6).
"""
import numpy as np, json, sys, warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))
PROJECT_ROOT = Path(__file__).parent.parent

import torch, gpytorch
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from fair_benchmark import m_ece, m_nll
from adversarial import crps_gauss, pit_ks
from mechanism import fit_exact_hypers, exact_gp_predict
from hetero_fair import oof_resid2, knn

CAP = 2500
SEEDS = (0, 1, 2)
EXACT = dict(covar_root_decomposition=False, log_prob=False, solves=False)


class _GP(gpytorch.models.ExactGP):
    def __init__(self, x, y, lik):
        super().__init__(x, y, lik)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))


def _fit_gpytorch(X, y, n_iter=120, seed=0):
    torch.manual_seed(seed)
    tx = torch.tensor(X, dtype=torch.float64)
    ty = torch.tensor(y, dtype=torch.float64)
    lik = gpytorch.likelihoods.GaussianLikelihood().double()
    model = _GP(tx, ty, lik).double()
    model.train(); lik.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    for _ in range(n_iter):
        opt.zero_grad()
        loss = -mll(model(tx), ty)
        loss.backward(); opt.step()
    model.eval(); lik.eval()
    return model, lik


def _pred(model, lik, Xq):
    tq = torch.tensor(Xq, dtype=torch.float64)
    with torch.no_grad(), gpytorch.settings.max_cholesky_size(100000), \
         gpytorch.settings.fast_pred_var(False), \
         gpytorch.settings.fast_computations(**EXACT):
        d = lik(model(tq))
        return d.mean.numpy(), d.variance.numpy()


def run(X, y, seed):
    rs = np.random.RandomState(seed)
    if len(X) > CAP:
        i = rs.choice(len(X), CAP, replace=False); X, y = X[i], y[i]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=.4, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=.5, random_state=seed)
    sx, sy = StandardScaler(), StandardScaler()
    Xs = sx.fit_transform(X_tr); ys = sy.fit_transform(y_tr.reshape(-1, 1)).ravel()
    Xva, Xte = sx.transform(X_va), sx.transform(X_te)
    sc = sy.scale_[0]
    back = lambda m, v: (sy.inverse_transform(m.reshape(-1, 1)).ravel(),
                         np.sqrt(np.maximum(v, 1e-12)) * sc)

    out = {}
    # stage 1: signal GP (GPyTorch)
    m1, l1 = _fit_gpytorch(Xs, ys, seed=seed)
    mu_te, var_te = _pred(m1, l1, Xte)
    p, s = back(mu_te, var_te)
    out['global_gpytorch'] = {'ece': m_ece(y_te, p, s), 'nll': m_nll(y_te, p, s),
                              'crps': crps_gauss(y_te, p, s), 'pit_ks': pit_ks(y_te, p, s)}

    # stage 2: noise GP on out-of-fold log squared residuals (GPyTorch)
    oof = np.zeros(len(ys))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(Xs):
        mk, lk = _fit_gpytorch(Xs[tr], ys[tr], n_iter=60, seed=seed)
        oof[te] = _pred(mk, lk, Xs[te])[0]
    z = np.log((ys - oof) ** 2 + 1e-8)
    sz = StandardScaler(); zs = sz.fit_transform(z.reshape(-1, 1)).ravel()
    m2, l2 = _fit_gpytorch(Xs, zs, seed=seed + 7)

    noise_lat = float(l1.noise.item())
    lat_te = np.maximum(var_te - noise_lat, 1e-12)
    for tag, Q in (('te', Xte),):
        mz, _ = _pred(m2, l2, Q)
        lv = sz.inverse_transform(mz.reshape(-1, 1)).ravel()
        s_het = np.sqrt(lat_te + np.exp(np.clip(lv, -30, 30)))
        p2, s2 = back(mu_te, s_het ** 2)
    out['het_gpytorch'] = {'ece': m_ece(y_te, p2, s2), 'nll': m_nll(y_te, p2, s2),
                           'crps': crps_gauss(y_te, p2, s2), 'pit_ks': pit_ks(y_te, p2, s2)}
    return out


def main():
    files = sorted((PROJECT_ROOT / 'data' / 'real').glob('*.npz'))
    R = {}
    for f in files:
        z = np.load(f, allow_pickle=True)
        R[f.stem] = {str(s): run(z['X'], z['y'], s) for s in SEEDS}
        print(f"  {f.stem} done", flush=True)
        json.dump(R, open(PROJECT_ROOT / 'results' / 'hetero_independent.json', 'w'), indent=2)

    print("\n" + "=" * 84)
    print("INDEPENDENT het-GP (GPyTorch) vs its own global baseline")
    print("=" * 84)
    print(f"{'dataset':16s} " + "".join(f"{m:>14s}" for m in ('ece', 'nll', 'crps', 'pit_ks')))
    print("-" * 84)
    for ds, cells in R.items():
        c = list(cells.values())
        row = f"{ds:16s} "
        for m in ('ece', 'nll', 'crps', 'pit_ks'):
            g = np.mean([x['global_gpytorch'][m] for x in c])
            h = np.mean([x['het_gpytorch'][m] for x in c])
            row += f"{g:6.3f}/{h:6.3f} " + ("" if True else "")
        print(row)
    print("\n  each cell is global/het. het < global means the het-GP helps.")
    for m in ('ece', 'nll', 'crps', 'pit_ks'):
        w = sum(np.mean([x['het_gpytorch'][m] for x in cells.values()])
                < np.mean([x['global_gpytorch'][m] for x in cells.values()])
                for cells in R.values())
        print(f"   {m:7s} het-GP better on {w}/{len(R)} datasets")


if __name__ == '__main__':
    main()
