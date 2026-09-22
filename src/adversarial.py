"""
ADVERSARIAL AUDIT of this paper's own conclusions.

The headline -- a single global noise parameter beats regional and heteroscedastic
noise on real data -- rests on ECE with ten central-interval bins. That is one metric,
one granularity, one sample size and one tuning protocol. Each is a place the
conclusion could be an artifact. This mounts four attacks.

  A. METRIC DEPENDENCE. ECE over central intervals is not a proper scoring rule, and
     it is known to reward underconfidence: intervals that are too wide can score
     well. If "global wins" survives only under ECE, it is not a result. Tested
     against CRPS (proper), NLL (proper), the Kolmogorov-Smirnov statistic of the
     probability integral transform (a strictly sharper calibration test than binned
     ECE), 90% interval coverage error, and ECE at 5 / 10 / 20 bins.

  B. SHARPNESS. If global noise wins by being underconfident rather than correct, its
     mean predictive width should exceed the others'. Reported alongside.

  C. GRANULARITY. Regional noise was tested with 2, 3 or 5 regions. As the count
     grows it approaches a smooth input-dependent noise model, so if the conclusion is
     really about granularity rather than concept, more regions should rescue it.
     Swept to 20.

  D. TUNING PARITY. The global arm takes its noise from marginal likelihood while the
     other arms select theirs on validation. That asymmetry favours whichever arm gets
     the extra freedom. A global arm whose noise is also validation-selected is added.
"""
import numpy as np, json, sys, warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))
PROJECT_ROOT = Path(__file__).parent.parent

from scipy.stats import norm, kstest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from fair_benchmark import m_ece, m_nll
from mechanism import fit_exact_hypers, exact_gp_predict
from hetero_fair import oof_resid2, knn, gp_het

CAP = 2500
SEEDS = (0, 1, 2, 3, 4)
REGION_GRID = (2, 3, 5, 10, 20)


# ---------------- metric battery (attack A) ----------------
def crps_gauss(y, mu, sd):
    z = (y - mu) / sd
    return float(np.mean(sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))


def pit_ks(y, mu, sd):
    """KS distance of the PIT from uniform. Sharper than binned ECE."""
    return float(kstest(norm.cdf((y - mu) / sd), 'uniform').statistic)


def cov_err(y, mu, sd, level=0.9):
    z = norm.ppf((1 + level) / 2)
    return float(abs(np.mean(np.abs(y - mu) <= z * sd) - level))


def battery(y, mu, sd):
    return {'ece10': m_ece(y, mu, sd, 10), 'ece5': m_ece(y, mu, sd, 5),
            'ece20': m_ece(y, mu, sd, 20), 'nll': m_nll(y, mu, sd),
            'crps': crps_gauss(y, mu, sd), 'pit_ks': pit_ks(y, mu, sd),
            'cov90_err': cov_err(y, mu, sd), 'mean_width': float(np.mean(sd))}


# ---------------- arms ----------------
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
    back = lambda m, s: (sy.inverse_transform(m.reshape(-1, 1)).ravel(), s * sc)

    ell, sf, sn = fit_exact_hypers(Xs, ys, seed)
    res = {}

    mu_te, sd_te = exact_gp_predict(Xs, ys, Xte, ell, sf, sn)
    res['global'] = battery(y_te, *back(mu_te, sd_te))

    # --- attack D: global noise also selected on validation ------------------
    mu_va, sd_va = exact_gp_predict(Xs, ys, Xva, ell, sf, sn)
    lat_va = np.maximum(sd_va ** 2 - sn ** 2, 1e-12)
    lat_te = np.maximum(sd_te ** 2 - sn ** 2, 1e-12)
    best = (np.inf, None)
    for mult in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0):
        s_va = np.sqrt(lat_va + (sn * mult) ** 2)
        v = m_nll(y_va, *back(mu_va, s_va))
        if v < best[0]:
            best = (v, mult)
    res['global_tuned'] = battery(y_te, *back(mu_te, np.sqrt(lat_te + (sn * best[1]) ** 2)))
    res['global_tuned_mult'] = best[1]

    # --- attack C: regional noise across a granularity sweep -----------------
    r2 = oof_resid2(Xs, ys, ell, sf, np.full(len(ys), sn ** 2), seed=seed)
    best = (np.inf, None, None)
    for n_reg in REGION_GRID:
        for k in (20, 50, 100):
            v_tr = knn(Xs, r2, Xs, k)
            cuts = np.percentile(v_tr, np.linspace(0, 100, n_reg + 1)[1:-1])
            lab = lambda q: np.digitize(q, cuts)
            l_tr = lab(v_tr)
            nz = {r: float(np.mean(r2[l_tr == r])) if (l_tr == r).sum() > 5 else sn ** 2
                  for r in range(n_reg)}
            g = lambda Q, lat: np.sqrt(lat + np.array(
                [nz.get(int(r), sn ** 2) for r in lab(knn(Xs, r2, Q, k))]))
            v = m_nll(y_va, *back(mu_va, g(Xva, lat_va)))
            if np.isfinite(v) and v < best[0]:
                best = (v, g(Xte, lat_te), (n_reg, k))
    res['regional'] = battery(y_te, *back(mu_te, best[1]))
    res['regional_cfg'] = list(best[2])

    # per-granularity, to see whether more regions ever help
    per = {}
    for n_reg in REGION_GRID:
        bb = (np.inf, None)
        for k in (20, 50, 100):
            v_tr = knn(Xs, r2, Xs, k)
            cuts = np.percentile(v_tr, np.linspace(0, 100, n_reg + 1)[1:-1])
            lab = lambda q: np.digitize(q, cuts)
            l_tr = lab(v_tr)
            nz = {r: float(np.mean(r2[l_tr == r])) if (l_tr == r).sum() > 5 else sn ** 2
                  for r in range(n_reg)}
            g = lambda Q, lat: np.sqrt(lat + np.array(
                [nz.get(int(r), sn ** 2) for r in lab(knn(Xs, r2, Q, k))]))
            v = m_nll(y_va, *back(mu_va, g(Xva, lat_va)))
            if v < bb[0]:
                bb = (v, g(Xte, lat_te))
        per[str(n_reg)] = battery(y_te, *back(mu_te, bb[1]))
    res['granularity'] = per
    return res


def main():
    files = [(p.stem, p) for p in sorted((PROJECT_ROOT / 'data' / 'real').glob('*.npz'))]
    R = {}
    for name, path in files:
        z = np.load(path, allow_pickle=True)
        R[name] = {str(s): run(z['X'], z['y'], s) for s in SEEDS}
        print(f"  {name} done", flush=True)
        json.dump(R, open(PROJECT_ROOT / 'results' / 'adversarial.json', 'w'), indent=2)

    ARMS = ['global', 'global_tuned', 'regional']
    METS = ['ece5', 'ece10', 'ece20', 'pit_ks', 'cov90_err', 'crps', 'nll']
    print("\n" + "=" * 100)
    print("ATTACK A/B/D — best arm per dataset under each metric (lower is better)")
    print("=" * 100)
    print(f"{'metric':11s} " + "".join(f"{a:>15s}" for a in ARMS) + "   winner-count")
    print("-" * 100)
    for met in METS:
        tal = {}
        means = {a: [] for a in ARMS}
        for n, cells in R.items():
            c = list(cells.values())
            v = {a: np.mean([x[a][met] for x in c]) for a in ARMS}
            for a in ARMS:
                means[a].append(v[a])
            w = min(v, key=v.get); tal[w] = tal.get(w, 0) + 1
        print(f"{met:11s} " + "".join(f"{np.mean(means[a]):15.4f}" for a in ARMS)
              + "   " + ", ".join(f"{k}:{v}" for k, v in sorted(tal.items())))
    print("\nmean predictive width (attack B — is global winning by being wide?)")
    for a in ARMS:
        w = np.mean([np.mean([x[a]['mean_width'] for x in cells.values()])
                     for cells in R.values()])
        print(f"   {a:14s} {w:.4f}")

    print("\n" + "=" * 100)
    print("ATTACK C — does finer granularity rescue regional noise?")
    print("=" * 100)
    print(f"{'n_regions':>10s} " + "".join(f"{m:>11s}" for m in ('ece10', 'pit_ks', 'crps', 'nll')))
    print("-" * 60)
    for g in REGION_GRID:
        row = [np.mean([np.mean([x['granularity'][str(g)][m] for x in cells.values()])
                        for cells in R.values()]) for m in ('ece10', 'pit_ks', 'crps', 'nll')]
        print(f"{g:>10d} " + "".join(f"{v:11.4f}" for v in row))
    base = [np.mean([np.mean([x['global'][m] for x in cells.values()])
                     for cells in R.values()]) for m in ('ece10', 'pit_ks', 'crps', 'nll')]
    print(f"{'global':>10s} " + "".join(f"{v:11.4f}" for v in base))


if __name__ == '__main__':
    main()
