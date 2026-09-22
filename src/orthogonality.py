"""
Table 2 of the paper: approximation fidelity and predictive uncertainty are orthogonal.

This is the paper's central mechanism and was previously produced by an ad-hoc command
rather than a script in the repository -- found during the self-audit. It is now
regenerable by reproduce.sh like every other result.

Claim: four models spanning a wide range of approximation fidelity, evaluated at the
same inputs, emit nearly identical predictive standard deviations, so a router has
nothing to choose between along the axis calibration depends on.
"""
import numpy as np, json, sys, warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))
PROJECT_ROOT = Path(__file__).parent.parent

import routing_recheck                      # noqa: F401 -- exact GP inference
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fair_benchmark import m_ece, RFF, Nystrom
from mechanism import fit_exact_hypers, exact_gp_predict
from structured_data import GENERATORS

ARMS = ['rff', 'nys', 'nys_hi', 'gp']


def run(ds, seed):
    X, y, reg = GENERATORS[ds](seed=seed)
    i = np.arange(len(X))
    i_tr, i_te = train_test_split(i, test_size=.4, random_state=seed)
    X_tr, y_tr = X[i_tr], y[i_tr]
    X_te, y_te, r_te = X[i_te], y[i_te], reg[i_te]

    M = {'rff': RFF(1000, random_state=seed).fit(X_tr, y_tr),
         'nys': Nystrom(500, random_state=seed).fit(X_tr, y_tr),
         'nys_hi': Nystrom(1500, random_state=seed).fit(X_tr, y_tr)}
    sx, sy = StandardScaler(), StandardScaler()
    Xs = sx.fit_transform(X_tr); ys = sy.fit_transform(y_tr.reshape(-1, 1)).ravel()
    ell, sf, sn = fit_exact_hypers(Xs, ys, seed)
    mu, sd = exact_gp_predict(Xs, ys, sx.transform(X_te), ell, sf, sn)
    P = {k: m.predict(X_te) for k, m in M.items()}
    P['gp'] = (sy.inverse_transform(mu.reshape(-1, 1)).ravel(), sd * sy.scale_[0])

    out = {}
    for r in np.unique(r_te):
        m = r_te == r
        out[int(r)] = {
            'ece': {k: m_ece(y_te[m], P[k][0][m], P[k][1][m]) for k in ARMS},
            'mean_std': {k: float(P[k][1][m].mean()) for k in ARMS},
            'n': int(m.sum()),
        }
    return out


def main(seeds=(0, 1, 2)):
    R = {ds: {str(s): run(ds, s) for s in seeds} for ds in ('hetero_extreme', 'combined')}
    out = PROJECT_ROOT / 'results' / 'structured'
    out.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(out / 'orthogonality.json', 'w'), indent=2)

    for ds, cells in R.items():
        c = list(cells.values())
        print("=" * 92)
        print(f"{ds}: per-region ECE and mean predictive std across four fidelities "
              f"({len(c)} seeds)")
        print("=" * 92)
        print(f"{'region':8s}" + "".join(f"{k:>11s}" for k in ARMS)
              + f"{'ECE spread':>12s}{'mean std (all arms)':>22s}")
        print("-" * 92)
        for r in sorted(c[0]):
            e = {k: np.mean([x[r]['ece'][k] for x in c]) for k in ARMS}
            s = {k: np.mean([x[r]['mean_std'][k] for x in c]) for k in ARMS}
            spread = max(e.values()) - min(e.values())
            rng = f"{min(s.values()):.3f}-{max(s.values()):.3f}"
            print(f"{str(r):<8s}" + "".join(f"{e[k]:11.4f}" for k in ARMS)
                  + f"{spread:12.4f}{rng:>22s}")
        allsp = [max(np.mean([x[r]['ece'][k] for x in c]) for k in ARMS)
                 - min(np.mean([x[r]['ece'][k] for x in c]) for k in ARMS) for r in c[0]]
        print(f"  max ECE spread across models within any region: {max(allsp):.4f}")
        print()
    print("saved ->", out / 'orthogonality.json')


if __name__ == '__main__':
    main()
