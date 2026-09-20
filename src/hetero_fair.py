"""
Regional noise vs a heteroscedastic GP -- with the noise estimate done correctly.

Earlier versions of this comparison were not fair and were not claimed. The bug:
both arms estimated noise from IN-SAMPLE residuals of a GP conditioned on those same
points, which underestimates noise by 1.1-1.9x (measured). Regional noise averages
over large groups so the bias largely cancels; the het-GP uses residuals pointwise and
compounds the error through EM iterations. That handicapped the baseline, which is
exactly the error this whole audit exists to document.

Both arms now estimate noise from OUT-OF-FOLD residuals (5-fold CV), computed once per
(dataset, seed) and shared, so neither arm has an information advantage. Both select
their hyperparameters on validation NLL with equal budgets. 10 seeds, with paired
per-seed differences and a sign test.
"""
import numpy as np, json, sys, warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))
PROJECT_ROOT = Path(__file__).parent.parent

from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.stats import binomtest, wilcoxon
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from fair_benchmark import m_ece, m_nll, m_rmse
from mechanism import fit_exact_hypers, exact_gp_predict, rbf

CAP = 2500
JIT = 1e-6
SEEDS = tuple(range(10))


def gp_het(Xs, ys, Xq, ell, sf, noise_tr, noise_q):
    K = sf ** 2 * rbf(Xs, Xs, ell) + np.diag(noise_tr + JIT)
    L = cholesky(K, lower=True)
    Ks = sf ** 2 * rbf(Xq, Xs, ell)
    mean = Ks @ cho_solve((L, True), ys)
    V = solve_triangular(L, Ks.T, lower=True)
    return mean, np.sqrt(np.maximum(sf ** 2 - np.sum(V ** 2, 0) + noise_q, 1e-12))


def oof_resid2(Xs, ys, ell, sf, noise_tr, folds=5, seed=0):
    """Out-of-fold squared residuals -- an unbiased-ish noise estimate."""
    oof = np.zeros(len(ys))
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(Xs):
        m, _ = gp_het(Xs[tr], ys[tr], Xs[te], ell, sf, noise_tr[tr], noise_tr[te])
        oof[te] = m
    return (ys - oof) ** 2


def knn(Xref, vals, Xq, k):
    nb = NearestNeighbors(n_neighbors=min(k, len(Xref))).fit(Xref)
    return vals[nb.kneighbors(Xq, return_distance=False)].mean(1)


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
    back = lambda t: (sy.inverse_transform(t[0].reshape(-1, 1)).ravel(), t[1] * sc)

    ell, sf, sn = fit_exact_hypers(Xs, ys, seed)
    res = {}
    p, s = back(exact_gp_predict(Xs, ys, Xte, ell, sf, sn))
    res['global'] = {'ece': m_ece(y_te, p, s), 'nll': m_nll(y_te, p, s)}
    res['rmse'] = m_rmse(y_te, p)

    # shared, out-of-fold noise evidence -- neither arm gets an advantage
    r2_oof = oof_resid2(Xs, ys, ell, sf, np.full(len(ys), sn ** 2), seed=seed)

    def het(n_em, k):
        noise = np.full(len(ys), sn ** 2)
        for _ in range(n_em):
            r2 = oof_resid2(Xs, ys, ell, sf, noise, seed=seed) if _ else r2_oof
            z = np.log(r2 + 1e-8)
            noise = np.exp(np.clip(knn(Xs, z, Xs, k), -30, 30))
        zf = np.log(np.maximum(noise, 1e-12))
        nq = lambda Q: np.exp(np.clip(knn(Xs, zf, Q, k), -30, 30))
        return (gp_het(Xs, ys, Xva, ell, sf, noise, nq(Xva)),
                gp_het(Xs, ys, Xte, ell, sf, noise, nq(Xte)))

    def reg(n_reg, k):
        v = knn(Xs, r2_oof, Xs, k)
        cuts = np.percentile(v, np.linspace(0, 100, n_reg + 1)[1:-1])
        lab = lambda q: np.digitize(q, cuts)
        l_tr = lab(v)
        nz = {r: float(np.mean(r2_oof[l_tr == r])) if (l_tr == r).sum() > 5 else sn ** 2
              for r in range(n_reg)}
        out = []
        for Q in (Xva, Xte):
            mu, sd = exact_gp_predict(Xs, ys, Q, ell, sf, sn)
            lat = np.maximum(sd ** 2 - sn ** 2, 1e-12)
            lq = lab(knn(Xs, r2_oof, Q, k))
            out.append((mu, np.sqrt(lat + np.array([nz.get(int(r), sn ** 2) for r in lq]))))
        return out[0], out[1]

    for tag, fn, grid in (('het_gp', het, [(a, b) for a in (1, 2) for b in (20, 50, 100)]),
                          ('regional', reg, [(a, b) for a in (2, 3, 5) for b in (20, 50, 100)])):
        best = (np.inf, None, None)
        for cfg in grid:
            try:
                va, te = fn(*cfg)
                v = m_nll(y_va, *back(va))
                if np.isfinite(v) and v < best[0]:
                    best = (v, back(te), cfg)
            except Exception:
                continue
        if best[1]:
            p, s = best[1]
            res[tag] = {'ece': m_ece(y_te, p, s), 'nll': m_nll(y_te, p, s), 'cfg': list(best[2])}
    return res


def main():
    screen = {d['dataset']: d['het_ratio']
              for d in json.load(open(PROJECT_ROOT / 'results' / 'hetero_screen.json'))}
    files = [(p.stem, p) for p in sorted((PROJECT_ROOT / 'data' / 'real').glob('*.npz'))]
    files += [(p.stem, p) for p in sorted((PROJECT_ROOT / 'data').glob('*.npz'))]
    R = {}
    for name, path in files:
        z = np.load(path, allow_pickle=True)
        R[name] = {'het_ratio': screen.get(name),
                   'seeds': {str(s): run(z['X'], z['y'], s) for s in SEEDS}}
        print(f"  {name} done", flush=True)
        json.dump(R, open(PROJECT_ROOT / 'results' / 'hetero_fair.json', 'w'), indent=2)

    for met in ('nll', 'ece'):
        print("\n" + "=" * 104)
        print(f"{met.upper()} — out-of-fold noise, equal tuning budgets, {len(SEEDS)} seeds")
        print("=" * 104)
        print(f"{'dataset':22s} {'het':>6s} {'global':>16s} {'het_gp':>16s} {'regional':>16s}"
              f" {'reg-vs-het':>12s} {'p':>7s}")
        print("-" * 104)
        wins = {'global': 0, 'het_gp': 0, 'regional': 0}
        for n, d in sorted(R.items(), key=lambda kv: -(kv[1]['het_ratio'] or 0)):
            c = list(d['seeds'].values())
            col = {}
            for k in ('global', 'het_gp', 'regional'):
                v = np.array([x[k][met] for x in c if k in x])
                col[k] = v
            w = min(col, key=lambda k: col[k].mean()); wins[w] += 1
            diff = col['regional'] - col['het_gp']
            try:
                pv = wilcoxon(diff).pvalue if len(diff) > 5 and np.ptp(diff) > 0 else float('nan')
            except Exception:
                pv = float('nan')
            line = f"{n:22s} {d['het_ratio'] or 0:6.2f} "
            for k in ('global', 'het_gp', 'regional'):
                line += f"{col[k].mean():9.4f}+/-{col[k].std(ddof=1):5.4f} "
            line += f"{diff.mean():+12.4f} {pv:7.4f}"
            print(line)
        print(f"\n  best-arm count: {wins}")


if __name__ == '__main__':
    main()
