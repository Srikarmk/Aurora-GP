"""
Is "regional noise" actually worth anything, or is it a worse heteroscedastic GP?

Heteroscedastic GPs are established (Goldberg et al. 1997; Kersting et al. 2007). If
a standard two-stage het-GP dominates regional noise everywhere, then the finding in
FINDINGS.md reduces to "use a heteroscedastic GP on heteroscedastic data", which is
textbook and not a contribution. This runs that comparison.

It also closes the loop on region identification. FINDINGS.md section 2 shows
RegionIdentifier recovers the true partition only 41-49% of the time (chance 33%)
because it scores by predictive uncertainty + sparsity. Now that we know the partition
exists to track NOISE, we can target it directly: partition by local residual
variance, estimated from training residuals alone.

Compared here, all with exact GP inference and no test labels anywhere:

  global          one noise parameter                        (the bar)
  het_gp          two-stage heteroscedastic GP               (the established method)
  regional_true   regional noise, TRUE generative partition  (upper bound)
  regional_learn  regional noise, noise-targeted partition   (the practical method)
"""
import numpy as np, json, sys, warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))
PROJECT_ROOT = Path(__file__).parent.parent

import routing_recheck                      # noqa: F401 -- exact GP inference
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from fair_benchmark import m_ece, m_nll, m_rmse
from mechanism import fit_exact_hypers, exact_gp_predict
from structured_data import GENERATORS

K_NN = 40


def _gp(Xs, ys, Xq, seed):
    ell, sf, sn = fit_exact_hypers(Xs, ys, seed)
    mu, sd = exact_gp_predict(Xs, ys, Xq, ell, sf, sn)
    return mu, sd, sn


def local_resid_var(Xs_tr, resid2, Xq, k=K_NN):
    """kNN-smoothed local residual variance -- training residuals only."""
    nb = NearestNeighbors(n_neighbors=min(k, len(Xs_tr))).fit(Xs_tr)
    idx = nb.kneighbors(Xq, return_distance=False)
    return resid2[idx].mean(1)


def run(ds, seed):
    X, y, reg = GENERATORS[ds](seed=seed)
    i = np.arange(len(X))
    i_tr, i_te = train_test_split(i, test_size=0.4, random_state=seed)
    X_tr, y_tr, r_tr = X[i_tr], y[i_tr], reg[i_tr]
    X_te, y_te, r_te = X[i_te], y[i_te], reg[i_te]

    sx, sy = StandardScaler(), StandardScaler()
    Xs = sx.fit_transform(X_tr); ys = sy.fit_transform(y_tr.reshape(-1, 1)).ravel()
    Xq = sx.transform(X_te); yscale = sy.scale_[0]

    mu_te, sd_te, sn = _gp(Xs, ys, Xq, seed)
    p = sy.inverse_transform(mu_te.reshape(-1, 1)).ravel()
    s_global = sd_te * yscale
    latent = np.maximum(s_global ** 2 - (sn * yscale) ** 2, 1e-12)

    ell, sf, sn2 = fit_exact_hypers(Xs, ys, seed)
    mu_tr, _ = exact_gp_predict(Xs, ys, Xs, ell, sf, sn2)
    p_tr = sy.inverse_transform(mu_tr.reshape(-1, 1)).ravel()
    res2 = (y_tr - p_tr) ** 2

    out = {'rmse': m_rmse(y_te, p)}
    out['global'] = {'ece': m_ece(y_te, p, s_global), 'nll': m_nll(y_te, p, s_global)}

    # --- established baseline: two-stage heteroscedastic GP ------------------
    z = np.log(res2 + 1e-8)
    sz = StandardScaler(); zs = sz.fit_transform(z.reshape(-1, 1)).ravel()
    mu_z, _, _ = _gp(Xs, zs, Xq, seed + 991)
    log_var = sz.inverse_transform(mu_z.reshape(-1, 1)).ravel()
    s_het = np.sqrt(latent + np.exp(np.clip(log_var, -30, 30)))
    out['het_gp'] = {'ece': m_ece(y_te, p, s_het), 'nll': m_nll(y_te, p, s_het)}

    # --- regional noise, true partition --------------------------------------
    s_true = np.zeros(len(y_te))
    for r in np.unique(r_tr):
        nr = np.std(y_tr[r_tr == r] - p_tr[r_tr == r])
        s_true[r_te == r] = np.sqrt(latent[r_te == r] + nr ** 2)
    out['regional_true'] = {'ece': m_ece(y_te, p, s_true), 'nll': m_nll(y_te, p, s_true)}

    # --- regional noise, NOISE-TARGETED partition (learned) ------------------
    v_tr = local_resid_var(Xs, res2, Xs)
    cuts = np.percentile(v_tr, [100 / 3, 200 / 3])
    lab_tr = np.digitize(v_tr, cuts)
    lab_te = np.digitize(local_resid_var(Xs, res2, Xq), cuts)
    s_learn = np.zeros(len(y_te))
    for r in np.unique(lab_tr):
        m_tr = lab_tr == r; m_te = lab_te == r
        if m_te.sum() == 0:
            continue
        nr = np.std(y_tr[m_tr] - p_tr[m_tr])
        s_learn[m_te] = np.sqrt(latent[m_te] + nr ** 2)
    s_learn[s_learn == 0] = s_global[s_learn == 0]
    out['regional_learn'] = {'ece': m_ece(y_te, p, s_learn), 'nll': m_nll(y_te, p, s_learn)}

    best = max(np.mean(lab_te == perm[r_te]) for perm in
               [np.array(q) for q in [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]])
    out['partition_recovery'] = float(best)
    return out


def main(seeds=(0, 1, 2)):
    R = {ds: {str(s): run(ds, s) for s in seeds} for ds in GENERATORS}
    out = PROJECT_ROOT / 'results' / 'structured'
    out.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(out / 'hetero_baseline.json', 'w'), indent=2)

    for metric in ('ece', 'nll'):
        print("=" * 100)
        print(f"{metric.upper()} — regional noise vs an established heteroscedastic GP")
        print("=" * 100)
        print(f"{'dataset':22s} {'global':>11s} {'het_gp':>11s} {'regional_true':>15s}"
              f" {'regional_learn':>16s} {'recovery':>10s}")
        print("-" * 100)
        for ds, cells in R.items():
            c = list(cells.values())
            f = lambda k: np.mean([x[k][metric] for x in c])
            rec = np.mean([x['partition_recovery'] for x in c])
            print(f"{ds:22s} {f('global'):11.4f} {f('het_gp'):11.4f}"
                  f" {f('regional_true'):15.4f} {f('regional_learn'):16.4f} {rec*100:9.1f}%")
        print()


if __name__ == '__main__':
    main()
