"""
The constructive counterpart to structured_routing.py.

structured_routing.py shows that routing approximation FIDELITY by region gains
nothing (+0.0003 ECE) even with the true partition and an oracle model choice. The
reason is visible in the per-region predictive standard deviations: on hetero_extreme
the true noise varies 50x across regions (0.020 / 0.141 / 1.000) while every candidate
model -- RFF, Nystrom at two ranks, and the exact GP -- emits the same ~0.563
everywhere. Their per-region ECE differs by at most 0.009 while the calibration error
itself is 0.5. There is nothing to route between.

Approximation fidelity and predictive uncertainty are orthogonal: a sparse GP's
predictive variance is dominated by the shared global noise term, not by the
approximation gap. So no amount of routing between homoscedastic models of differing
rank can produce region-appropriate uncertainty.

This script changes the knob. Same partition, same GP posterior mean, same latent
variance -- only the NOISE term is made regional, fitted on training residuals within
each region. Nothing here uses test labels.
"""
import numpy as np, json, sys, warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))
PROJECT_ROOT = Path(__file__).parent.parent

import routing_recheck           # noqa: F401 -- exact GP inference
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fair_benchmark import m_ece, m_nll, m_rmse
from mechanism import fit_exact_hypers, exact_gp_predict
from structured_data import GENERATORS


def run(ds, seed):
    X, y, reg = GENERATORS[ds](seed=seed)
    i = np.arange(len(X))
    i_tr, i_te = train_test_split(i, test_size=0.4, random_state=seed)
    X_tr, y_tr, r_tr = X[i_tr], y[i_tr], reg[i_tr]
    X_te, y_te, r_te = X[i_te], y[i_te], reg[i_te]

    sx, sy = StandardScaler(), StandardScaler()
    Xs = sx.fit_transform(X_tr); ys = sy.fit_transform(y_tr.reshape(-1, 1)).ravel()
    ell, sf, sn = fit_exact_hypers(Xs, ys, seed)

    mu, sd = exact_gp_predict(Xs, ys, sx.transform(X_te), ell, sf, sn)
    p = sy.inverse_transform(mu.reshape(-1, 1)).ravel()
    s_global = sd * sy.scale_[0]

    # strip the global noise the GP added, leaving the latent function variance
    latent = np.maximum(s_global ** 2 - (sn * sy.scale_[0]) ** 2, 1e-12)

    # per-region noise from TRAINING residuals only
    mu_tr, _ = exact_gp_predict(Xs, ys, Xs, ell, sf, sn)
    p_tr = sy.inverse_transform(mu_tr.reshape(-1, 1)).ravel()
    s_region = np.zeros(len(y_te))
    noises = {}
    for r in np.unique(r_tr):
        nr = float(np.std(y_tr[r_tr == r] - p_tr[r_tr == r]))
        noises[int(r)] = nr
        s_region[r_te == r] = np.sqrt(latent[r_te == r] + nr ** 2)

    return {
        'global': {'ece': m_ece(y_te, p, s_global), 'nll': m_nll(y_te, p, s_global)},
        'regional_noise': {'ece': m_ece(y_te, p, s_region), 'nll': m_nll(y_te, p, s_region)},
        'rmse': m_rmse(y_te, p),          # identical for both: only variance changes
        'fitted_region_noise': noises,
    }


def main(seeds=(0, 1, 2)):
    out = PROJECT_ROOT / 'results' / 'structured'
    out.mkdir(parents=True, exist_ok=True)
    R = {ds: {str(s): run(ds, s) for s in seeds} for ds in GENERATORS}
    json.dump(R, open(out / 'noise_routing.json', 'w'), indent=2)

    print("=" * 92)
    print("ROUTING THE NOISE MODEL BY REGION (same partition, same mean, same latent var)")
    print("=" * 92)
    print(f"{'dataset':22s} {'ECE global':>12s} {'ECE regional':>14s} {'ratio':>8s}"
          f" {'NLL global':>12s} {'NLL regional':>14s}")
    print("-" * 92)
    for ds, cells in R.items():
        c = list(cells.values())
        g = np.mean([x['global']['ece'] for x in c])
        q = np.mean([x['regional_noise']['ece'] for x in c])
        gn = np.mean([x['global']['nll'] for x in c])
        qn = np.mean([x['regional_noise']['nll'] for x in c])
        print(f"{ds:22s} {g:12.4f} {q:14.4f} {g/max(q,1e-9):7.2f}x {gn:12.3f} {qn:14.3f}")
    print("\n  For comparison, routing approximation FIDELITY on the same partitions")
    print("  gained +0.0003 ECE -- i.e. nothing. See results/structured/results.json.")
    print("saved ->", out / 'noise_routing.json')


if __name__ == '__main__':
    main()
