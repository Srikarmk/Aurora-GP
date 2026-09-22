"""
Does the routing conclusion survive at larger n?

Attack E caught the regional-noise result depending on a training-set size that the
main experiments never varied. The routing experiments in sections 4 and 5 were run at
the same size. Arguing that they must be unaffected -- because orthogonality concerns
the predictive-variance decomposition rather than a noise estimate -- is exactly the
kind of reasoning that attack E just falsified elsewhere, so it is checked directly.

Re-runs the oracle-routing headroom test at n_train up to 4800 and reports:
  - the oracle-minus-uniform ECE gap as a function of n
  - the per-region spread in predictive std across model fidelities as a function of n
The orthogonality mechanism predicts both stay flat.
"""
import numpy as np, json, sys, warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))
PROJECT_ROOT = Path(__file__).parent.parent

import routing_recheck                       # noqa: F401 -- exact GP inference
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fair_benchmark import m_ece, RFF, Nystrom, quiet
from mechanism import fit_exact_hypers, exact_gp_predict
from region_identification import RegionIdentifier

N_GRID = [600, 1200, 2400, 4800]
SEEDS = (0, 1, 2)
DATASETS = ['calhousing', 'powerplant', 'wine_white']
NAMES = ['rff', 'nys', 'nys_hi', 'gp']


def run(X, y, n_train, seed):
    rs = np.random.RandomState(seed)
    need = n_train + 1600
    if len(X) > need:
        i = rs.choice(len(X), need, replace=False); X, y = X[i], y[i]
    X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, train_size=n_train, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_rest, y_rest, test_size=.5, random_state=seed)

    M = {'rff': RFF(1000, random_state=seed).fit(X_tr, y_tr),
         'nys': Nystrom(500, random_state=seed).fit(X_tr, y_tr),
         'nys_hi': Nystrom(1500, random_state=seed).fit(X_tr, y_tr)}
    sx, sy = StandardScaler(), StandardScaler()
    Xs = sx.fit_transform(X_tr); ys = sy.fit_transform(y_tr.reshape(-1, 1)).ravel()
    ell, sf, sn = fit_exact_hypers(Xs, ys, seed)

    def gp(Q):
        mu, sd = exact_gp_predict(Xs, ys, sx.transform(Q), ell, sf, sn)
        return sy.inverse_transform(mu.reshape(-1, 1)).ravel(), sd * sy.scale_[0]

    pv = {k: m.predict(X_va) for k, m in M.items()}; pv['gp'] = gp(X_va)
    pt = {k: m.predict(X_te) for k, m in M.items()}; pt['gp'] = gp(X_te)

    with quiet():
        ri = RegionIdentifier(max_gp_samples=5000, random_state=seed)
        ri.fit(X_tr, y_tr, n_gp_iter=30)
        rv, _ = ri.predict_regions(X_va, adaptive_thresholds=False)
        rt, _ = ri.predict_regions(X_te, adaptive_thresholds=False)

    best_uni = min(NAMES, key=lambda k: m_ece(y_va, *pv[k]))
    uni = m_ece(y_te, *pt[best_uni])

    ch = {}
    for r in np.unique(rt):
        m = rv == r
        ch[r] = (min(NAMES, key=lambda k: m_ece(y_va[m], pv[k][0][m], pv[k][1][m]))
                 if m.sum() > 10 else best_uni)
    p = np.zeros(len(y_te)); s = np.zeros(len(y_te))
    for r in np.unique(rt):
        m = rt == r
        pk, sk = pt[ch[r]]
        p[m], s[m] = pk[m], sk[m]
    orc = m_ece(y_te, p, s)

    # spread in predictive std across fidelities, within region
    spreads = []
    for r in np.unique(rt):
        m = rt == r
        if m.sum() < 10:
            continue
        w = [pt[k][1][m].mean() for k in NAMES]
        spreads.append((max(w) - min(w)) / max(np.mean(w), 1e-12))
    return {'uniform': uni, 'oracle': orc, 'gap': orc - uni,
            'std_spread_rel': float(np.mean(spreads))}


def main():
    R = {}
    for ds in DATASETS:
        z = np.load(PROJECT_ROOT / 'data' / 'real' / f'{ds}.npz', allow_pickle=True)
        X, y = z['X'], z['y']
        R[ds] = {}
        for n in N_GRID:
            if len(X) < n + 600:
                continue
            R[ds][str(n)] = {str(s): run(X, y, n, s) for s in SEEDS}
            print(f"  {ds} n={n} done", flush=True)
        json.dump(R, open(PROJECT_ROOT / 'results' / 'routing_largen.json', 'w'), indent=2)

    print("\n" + "=" * 86)
    print("Does routing headroom appear at larger n?  gap = oracle - uniform")
    print("  (negative would mean routing finally helps)")
    print("=" * 86)
    print(f"{'n_train':>8s} " + "".join(f"{d[:12]:>15s}" for d in DATASETS)
          + f"{'mean gap':>11s}{'std spread':>12s}")
    print("-" * 86)
    for n in N_GRID:
        row, gaps, spr = "", [], []
        for ds in DATASETS:
            if str(n) not in R.get(ds, {}):
                row += f"{'-':>15s}"; continue
            c = list(R[ds][str(n)].values())
            g = np.mean([x['gap'] for x in c]); gaps.append(g)
            spr += [x['std_spread_rel'] for x in c]
            row += f"{g:+15.4f}"
        if gaps:
            print(f"{n:>8d} " + row + f"{np.mean(gaps):+11.4f}{np.mean(spr)*100:11.2f}%")
    print("\n  std spread = relative range of mean predictive std across the four")
    print("  fidelities within a region. Flat and small confirms orthogonality at all n.")


if __name__ == '__main__':
    main()
