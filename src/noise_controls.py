"""
Controls for the regional-noise result (paper Sec. 6), on the hetero_fair.py protocol:
11 datasets, seeds 0-9, identical splits and fits.

Does the regional arm win only because it is re-calibrated on validation data and the
global arm is not? Three global recalibrations, each given the same validation data:
  oof_global     one noise level set to the mean out-of-fold noise evidence
                 (regional noise with a single region, no selection)
  global_fine    sigma_n multiplier from 61 values in [0.5, 2], chosen on validation NLL
  global_scale   total predictive sd rescaled by its validation-optimal factor

Does selection over configurations drive it? regional_fixed uses one configuration
(3 regions, k=50) with no selection.

Does the noise floor drive it? regional_nofloor drops the 0.05 sn^2 floor; floor
statistics record how often the floor binds for the selected configuration.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import json, sys, warnings, numpy as np
from pathlib import Path
from multiprocessing import Pool

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))
P = Path(__file__).parent.parent

from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fair_benchmark import m_ece, m_nll
from mechanism import fit_exact_hypers, exact_gp_predict
from hetero_fair import oof_noise, region_noise, knn, crps_gauss, FLOOR, CAP, SEEDS

FILES = {f.stem: f for f in list((P / 'data' / 'real').glob('*.npz')) + list((P / 'data').glob('*.npz'))}
GRID = [(a, b) for a in (2, 3, 5, 10) for b in (20, 50, 100)]


def job(arg):
    name, seed = arg
    z = np.load(FILES[name], allow_pickle=True); X, y = z['X'], z['y']
    rs = np.random.RandomState(seed)
    if len(X) > CAP:
        i = rs.choice(len(X), CAP, replace=False); X, y = X[i], y[i]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=.4, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=.5, random_state=seed)
    sx, sy = StandardScaler(), StandardScaler()
    Xs = sx.fit_transform(X_tr); ys = sy.fit_transform(y_tr.reshape(-1, 1)).ravel()
    Xva, Xte = sx.transform(X_va), sx.transform(X_te); sc = sy.scale_[0]
    back = lambda m, s: (sy.inverse_transform(m.reshape(-1, 1)).ravel(), s * sc)
    met = lambda yy, m, s: {'ece': m_ece(yy, *back(m, s)), 'nll': m_nll(yy, *back(m, s)),
                            'crps': crps_gauss(yy, *back(m, s))}

    ell, sf, sn = fit_exact_hypers(Xs, ys, seed); sn2 = sn ** 2
    mva, sva = exact_gp_predict(Xs, ys, Xva, ell, sf, sn)
    mte, ste = exact_gp_predict(Xs, ys, Xte, ell, sf, sn)
    latva, latte = np.maximum(sva ** 2 - sn2, 1e-12), np.maximum(ste ** 2 - sn2, 1e-12)
    e = oof_noise(Xs, ys, ell, sf, np.full(len(ys), sn2), seed=seed)
    floor = FLOOR * sn2
    out = {'global': met(y_te, mte, ste), 'sn': float(sn),
           'oof_mean_over_sn2': float(e.mean() / sn2)}

    out['oof_global'] = met(y_te, mte, np.sqrt(latte + max(float(e.mean()), floor)))
    mults = np.linspace(0.5, 2.0, 61)
    m = mults[int(np.argmin([m_nll(y_va, *back(mva, np.sqrt(latva + sn2 * k ** 2))) for k in mults]))]
    out['global_fine'] = met(y_te, mte, np.sqrt(latte + sn2 * m ** 2))
    c = float(np.sqrt(np.mean(((y_va - back(mva, sva)[0]) / back(mva, sva)[1]) ** 2)))
    out['global_scale'] = met(y_te, mte, ste * c)

    def reg(n_reg, k, Q, use_floor=True):
        v = knn(Xs, e, Xs, k)
        cuts = np.percentile(v, np.linspace(0, 100, n_reg + 1)[1:-1])
        lab = lambda q: np.digitize(q, cuts)
        raw = {r: float(np.mean(e[lab(v) == r])) for r in range(n_reg)}
        nz = region_noise(e, lab(v), n_reg, sn2) if use_floor else \
            {r: max(raw[r], 1e-6 * sn2) for r in range(n_reg)}
        return np.array([nz.get(int(r), sn2) for r in lab(knn(Xs, e, Q, k))]), raw

    cfg = min(GRID, key=lambda g: (lambda val: val if np.isfinite(val) else np.inf)(
        m_nll(y_va, *back(mva, np.sqrt(latva + reg(*g, Xva)[0])))))
    nte, raw = reg(*cfg, Xte)
    out['regional'] = met(y_te, mte, np.sqrt(latte + nte)); out['regional_cfg'] = list(cfg)
    out['floor_regions'] = int(sum(r < floor for r in raw.values()))
    out['frac_test_on_floor'] = float(np.mean(np.isclose(nte, floor)))
    out['regional_nofloor'] = met(y_te, mte, np.sqrt(latte + reg(*cfg, Xte, use_floor=False)[0]))
    out['regional_fixed'] = met(y_te, mte, np.sqrt(latte + reg(3, 50, Xte)[0]))
    return name, seed, out


def main():
    jobs = [(n, s) for n in sorted(FILES) for s in SEEDS]
    with Pool(min(16, os.cpu_count() or 4)) as pool:
        res = pool.map(job, jobs)
    R = {}
    for n, s, o in res:
        R.setdefault(n, {})[str(s)] = o
    json.dump(R, open(P / 'results' / 'noise_controls.json', 'w'), indent=2)

    arms = ('global', 'oof_global', 'global_fine', 'global_scale', 'regional',
            'regional_fixed', 'regional_nofloor')
    print(f"{'dataset':26s}" + "".join(f"{a:>17s}" for a in arms) + f"{'floor%':>8s}")
    for n in sorted(R):
        c = list(R[n].values())
        print(f"{n:26s}" + "".join(f"{np.mean([x[a]['ece'] for x in c]):17.4f}" for a in arms)
              + f"{100 * np.mean([x['frac_test_on_floor'] for x in c]):8.1f}")
    print("\nregional vs each control on ECE, Wilcoxon p (datasets where regional is better):")
    for a in arms[:4] + arms[5:]:
        if a == 'regional':
            continue
        sig = []
        for n in sorted(R):
            c = list(R[n].values())
            d = np.array([x['regional']['ece'] - x[a]['ece'] for x in c]) if a in arms[:4] else \
                np.array([x[a]['ece'] - x['global']['ece'] for x in c])
            if np.ptp(d) > 0 and wilcoxon(d).pvalue < .05 and d.mean() < 0:
                sig.append(n)
        print(f"  {'regional vs ' + a if a in arms[:4] else a + ' vs global':34s} {sig}")


if __name__ == '__main__':
    main()
