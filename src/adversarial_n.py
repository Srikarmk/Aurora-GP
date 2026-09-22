"""
ATTACK E: is the negative result an artifact of training-set size?

Regional noise estimates a variance per region from training residuals. With ~1500
training points split three ways, each estimate uses a few hundred residuals. If the
conclusion is really "not enough data per region", then growing n should close the gap
and eventually reverse it. Sweep n_train and watch the gap.

Also reported: the gap's trend. A flat or widening gap rules out the sample-size
explanation; a narrowing one means the result is n-limited and must be stated as such.
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

N_GRID = [300, 600, 1200, 2400, 4000]
SEEDS = (0, 1, 2)
DATASETS = ['calhousing', 'powerplant', 'wine_white', 'airfoil']


def run(X, y, n_train, seed):
    rs = np.random.RandomState(seed)
    need = n_train + 1200
    if len(X) > need:
        i = rs.choice(len(X), need, replace=False); X, y = X[i], y[i]
    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y, train_size=n_train, random_state=seed)
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
    g_ece = m_ece(y_te, *back(mu_te, sd_te))
    g_nll = m_nll(y_te, *back(mu_te, sd_te))

    r2 = oof_resid2(Xs, ys, ell, sf, np.full(len(ys), sn ** 2), seed=seed)
    best = (np.inf, None)
    for n_reg in (2, 3, 5, 10):
        for k in (20, 50, 100):
            v_tr = knn(Xs, r2, Xs, k)
            cuts = np.percentile(v_tr, np.linspace(0, 100, n_reg + 1)[1:-1])
            lab = lambda q: np.digitize(q, cuts)
            l_tr = lab(v_tr)
            nz = {r: float(np.mean(r2[l_tr == r])) if (l_tr == r).sum() > 5 else sn ** 2
                  for r in range(n_reg)}
            f = lambda Q, lat: np.sqrt(lat + np.array(
                [nz.get(int(r), sn ** 2) for r in lab(knn(Xs, r2, Q, k))]))
            v = m_nll(y_va, *back(mu_va, f(Xva, lat_va)))
            if np.isfinite(v) and v < best[0]:
                best = (v, f(Xte, lat_te))
    r_ece = m_ece(y_te, *back(mu_te, best[1]))
    r_nll = m_nll(y_te, *back(mu_te, best[1]))
    return {'global_ece': g_ece, 'regional_ece': r_ece,
            'global_nll': g_nll, 'regional_nll': r_nll, 'n_test': int(len(y_te))}


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
        json.dump(R, open(PROJECT_ROOT / 'results' / 'adversarial_n.json', 'w'), indent=2)

    print("\n" + "=" * 88)
    print("ATTACK E — does the global-vs-regional gap close as n grows?")
    print("  gap = regional - global   (positive = global better = conclusion holds)")
    print("=" * 88)
    print(f"{'n_train':>8s} " + "".join(f"{d[:11]:>13s}" for d in DATASETS) + f"{'mean gap':>11s}")
    print("-" * 88)
    for n in N_GRID:
        row, vals = "", []
        for ds in DATASETS:
            if str(n) not in R.get(ds, {}):
                row += f"{'-':>13s}"; continue
            c = list(R[ds][str(n)].values())
            gap = np.mean([x['regional_ece'] - x['global_ece'] for x in c])
            vals.append(gap); row += f"{gap:+13.4f}"
        print(f"{n:>8d} " + row + (f"{np.mean(vals):+11.4f}" if vals else ""))
    print("\n  A gap that stays positive as n grows rules out sample size as the")
    print("  explanation for the negative result.")


if __name__ == '__main__':
    main()
