"""
Turning the failure into a method: shrinkage-regularised regional noise.

Attack E showed regional noise loses at small $n$ and wins by $n=4000$. The mechanism
is estimation variance: a per-region noise estimated from a few hundred residuals is
itself noisy, and that noise costs more than the regional structure buys. This is a
textbook bias-variance problem, and it has a textbook fix.

Instead of using the raw per-region estimate, shrink it toward the global estimate:

    sigma^2_r(lambda) = lambda * sigma^2_global + (1 - lambda) * sigma^2_r

with lambda selected on validation NLL. The endpoints are exactly the two arms already
compared -- lambda=1 is global noise, lambda=0 is regional noise -- so with honest
validation selection this cannot be worse than either, and should beat both wherever
the truth is intermediate. The interesting question is whether the selected lambda
tracks n in the way the diagnosis predicts: strong shrinkage when data is scarce,
weak when it is plentiful.

Also reports an empirical-Bayes lambda per region, which needs no validation search:

    lambda_r = (s_r^2 / n_r) / (s_r^2 / n_r + tau^2)

where s_r^2/n_r is the sampling variance of the region's estimate and tau^2 is the
between-region variance. Regions with few points shrink harder, automatically.
"""
import numpy as np, json, sys, warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))
PROJECT_ROOT = Path(__file__).parent.parent

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fair_benchmark import m_ece, m_nll
from mechanism import fit_exact_hypers, exact_gp_predict
from hetero_fair import oof_resid2, knn
from adversarial import battery

N_GRID = [300, 600, 1200, 2400, 4000]
SEEDS = (0, 1, 2)
DATASETS = ['calhousing', 'powerplant', 'wine_white', 'airfoil', 'energy']
LAMBDAS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]


def run(X, y, n_train, seed):
    rs = np.random.RandomState(seed)
    need = n_train + 1200
    if len(X) > need:
        i = rs.choice(len(X), need, replace=False); X, y = X[i], y[i]
    X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, train_size=n_train, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_rest, y_rest, test_size=.5, random_state=seed)
    sx, sy = StandardScaler(), StandardScaler()
    Xs = sx.fit_transform(X_tr); ys = sy.fit_transform(y_tr.reshape(-1, 1)).ravel()
    Xva, Xte = sx.transform(X_va), sx.transform(X_te)
    sc = sy.scale_[0]
    back = lambda m, s: (sy.inverse_transform(m.reshape(-1, 1)).ravel(), s * sc)

    ell, sf, sn = fit_exact_hypers(Xs, ys, seed)
    mu_va, sd_va = exact_gp_predict(Xs, ys, Xva, ell, sf, sn)
    mu_te, sd_te = exact_gp_predict(Xs, ys, Xte, ell, sf, sn)
    lat_va = np.maximum(sd_va ** 2 - sn ** 2, 1e-12)
    lat_te = np.maximum(sd_te ** 2 - sn ** 2, 1e-12)
    r2 = oof_resid2(Xs, ys, ell, sf, np.full(len(ys), sn ** 2), seed=seed)
    g_var = float(np.mean(r2))

    out = {'global': battery(y_te, *back(mu_te, sd_te))}

    def build(n_reg, k, lam=None, eb=False):
        v_tr = knn(Xs, r2, Xs, k)
        cuts = np.percentile(v_tr, np.linspace(0, 100, n_reg + 1)[1:-1])
        lab = lambda q: np.digitize(q, cuts)
        l_tr = lab(v_tr)
        raw, cnt = {}, {}
        for r in range(n_reg):
            m = l_tr == r
            raw[r] = float(np.mean(r2[m])) if m.sum() > 5 else g_var
            cnt[r] = int(m.sum())
        if eb:
            tau2 = max(np.var(list(raw.values())), 1e-12)
            nz = {}
            for r in range(n_reg):
                m = l_tr == r
                s2 = float(np.var(r2[m])) if m.sum() > 5 else 0.0
                se = s2 / max(cnt[r], 1)
                lr = se / (se + tau2)
                nz[r] = lr * g_var + (1 - lr) * raw[r]
        else:
            nz = {r: lam * g_var + (1 - lam) * raw[r] for r in range(n_reg)}
        f = lambda Q, lat: np.sqrt(lat + np.array(
            [max(nz.get(int(r), g_var), 1e-12) for r in lab(knn(Xs, r2, Q, k))]))
        return f

    # raw regional (lambda = 0) and shrunk (lambda selected on validation)
    for tag, grid in (('regional', [0.0]), ('shrunk', LAMBDAS)):
        best = (np.inf, None, None)
        for n_reg in (2, 3, 5, 10):
            for k in (20, 50, 100):
                for lam in grid:
                    f = build(n_reg, k, lam=lam)
                    v = m_nll(y_va, *back(mu_va, f(Xva, lat_va)))
                    if np.isfinite(v) and v < best[0]:
                        best = (v, f(Xte, lat_te), (n_reg, k, lam))
        out[tag] = battery(y_te, *back(mu_te, best[1]))
        out[tag + '_cfg'] = list(best[2])

    # empirical Bayes: no validation search over lambda
    best = (np.inf, None, None)
    for n_reg in (2, 3, 5, 10):
        for k in (20, 50, 100):
            f = build(n_reg, k, eb=True)
            v = m_nll(y_va, *back(mu_va, f(Xva, lat_va)))
            if np.isfinite(v) and v < best[0]:
                best = (v, f(Xte, lat_te), (n_reg, k))
    out['eb'] = battery(y_te, *back(mu_te, best[1]))
    out['eb_cfg'] = list(best[2])
    return out


def main():
    R = {}
    for ds in DATASETS:
        z = np.load(PROJECT_ROOT / 'data' / 'real' / f'{ds}.npz', allow_pickle=True)
        X, y = z['X'], z['y']
        R[ds] = {}
        for n in N_GRID:
            if len(X) < n + 400:
                continue
            R[ds][str(n)] = {str(s): run(X, y, n, s) for s in SEEDS}
        print(f"  {ds} done", flush=True)
        json.dump(R, open(PROJECT_ROOT / 'results' / 'shrinkage.json', 'w'), indent=2)

    ARMS = ['global', 'regional', 'shrunk', 'eb']
    for met in ('ece10', 'nll', 'crps'):
        print("\n" + "=" * 92)
        print(f"{met.upper()} by training-set size (mean over datasets and seeds)")
        print("=" * 92)
        print(f"{'n_train':>8s} " + "".join(f"{a:>12s}" for a in ARMS) + f"{'best':>10s}{'sel. lambda':>13s}")
        print("-" * 92)
        for n in N_GRID:
            vals, lams = {}, []
            for a in ARMS:
                v = [x[a][met] for ds in R if str(n) in R[ds] for x in R[ds][str(n)].values()]
                if v:
                    vals[a] = float(np.mean(v))
            lams = [x['shrunk_cfg'][2] for ds in R if str(n) in R[ds]
                    for x in R[ds][str(n)].values()]
            if not vals:
                continue
            b = min(vals, key=vals.get)
            print(f"{n:>8d} " + "".join(f"{vals.get(a, float('nan')):12.4f}" for a in ARMS)
                  + f"{b:>10s}{np.mean(lams):13.2f}")
    print("\n  lambda = 1 is pure global noise, lambda = 0 is pure regional.")


if __name__ == '__main__':
    main()
