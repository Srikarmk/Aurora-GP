"""
Does region-aware routing help when regional structure is real by construction?

Decomposes the question so a negative result is interpretable:

  uniform_best       one model everywhere (validation-selected)      <- the bar
  true_oracle        TRUE generative partition + best model per region
                     -> is there ANY exploitable regional structure?
  learned_oracle     RegionIdentifier partition + best model per region
                     -> can region identification find it?
  importance         RegionIdentifier partition + AURORA's fixed assignment
                     -> does AURORA's criterion work?
  random             partition shuffled, proportions preserved       <- the control

If true_oracle does not beat uniform_best, the idea fails even on data purpose-built
for it. If true_oracle wins but learned_oracle does not, the failure is region
identification -- which is fixable, and a different paper.

Exact GP inference throughout (see AUDIT.md Tier 1 finding 6).
"""
import numpy as np, json, sys, time, warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))
PROJECT_ROOT = Path(__file__).parent.parent

import routing_recheck            # noqa: F401 -- patches predict() to exact inference
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fair_benchmark import m_ece, all_metrics, RFF, Nystrom, quiet
from mechanism import fit_exact_hypers, exact_gp_predict
from region_identification import RegionIdentifier
from structured_data import GENERATORS

NAMES = ['rff', 'nys', 'nys_hi', 'gp']


def run_cell(X, y, reg_true, seed):
    idx = np.arange(len(X))
    i_tr, i_tmp = train_test_split(idx, test_size=0.4, random_state=seed)
    i_va, i_te = train_test_split(i_tmp, test_size=0.5, random_state=seed)
    X_tr, y_tr = X[i_tr], y[i_tr]
    X_va, y_va = X[i_va], y[i_va]
    X_te, y_te = X[i_te], y[i_te]

    models = {'rff': RFF(1000, random_state=seed).fit(X_tr, y_tr),
              'nys': Nystrom(500, random_state=seed).fit(X_tr, y_tr),
              'nys_hi': Nystrom(1500, random_state=seed).fit(X_tr, y_tr)}

    sx, sy = StandardScaler(), StandardScaler()
    Xs = sx.fit_transform(X_tr); ys = sy.fit_transform(y_tr.reshape(-1, 1)).ravel()
    ell, sf, sn = fit_exact_hypers(Xs, ys, seed)

    def gp_pred(Xq):
        mu, sd = exact_gp_predict(Xs, ys, sx.transform(Xq), ell, sf, sn)
        return sy.inverse_transform(mu.reshape(-1, 1)).ravel(), sd * sy.scale_[0]

    pv = {k: m.predict(X_va) for k, m in models.items()}; pv['gp'] = gp_pred(X_va)
    pt = {k: m.predict(X_te) for k, m in models.items()}; pt['gp'] = gp_pred(X_te)

    with quiet():
        ri = RegionIdentifier(max_gp_samples=5000, random_state=seed)
        ri.fit(X_tr, y_tr, n_gp_iter=30)
        reg_va_L, _ = ri.predict_regions(X_va, adaptive_thresholds=False)
        reg_te_L, _ = ri.predict_regions(X_te, adaptive_thresholds=False)
    reg_va_T, reg_te_T = reg_true[i_va], reg_true[i_te]

    def assemble(choice, regions):
        p = np.zeros(len(y_te)); s = np.zeros(len(y_te))
        for r in np.unique(regions):
            m = regions == r
            pk, sk = pt[choice[r]]
            p[m], s[m] = pk[m], sk[m]
        return p, s

    def oracle(reg_va_, reg_te_, fallback):
        ch = {}
        for r in np.unique(reg_te_):
            m = reg_va_ == r
            ch[r] = (min(NAMES, key=lambda k: m_ece(y_va[m], pv[k][0][m], pv[k][1][m]))
                     if m.sum() > 10 else fallback)
        return ch

    best_uni = min(NAMES, key=lambda k: m_ece(y_va, *pv[k]))
    res = {}
    p, s = pt[best_uni]
    res['uniform_best'] = {'model': best_uni, **all_metrics(y_te, p, s)}

    ch = oracle(reg_va_T, reg_te_T, best_uni)
    res['true_oracle'] = {'choice': {str(k): v for k, v in ch.items()},
                          **all_metrics(y_te, *assemble(ch, reg_te_T))}

    ch = oracle(reg_va_L, reg_te_L, best_uni)
    res['learned_oracle'] = {'choice': {str(k): v for k, v in ch.items()},
                             **all_metrics(y_te, *assemble(ch, reg_te_L))}

    aur = {0: 'rff', 1: 'nys', 2: 'gp'}
    res['importance'] = all_metrics(y_te, *assemble(aur, reg_te_L))
    res['random'] = all_metrics(
        y_te, *assemble(aur, np.random.RandomState(seed).permutation(reg_te_L)))

    # how well does the learned partition recover the true one?
    agree = max(np.mean(reg_te_L == perm[reg_te_T]) for perm in
                [np.array(p) for p in
                 [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]])
    res['partition_recovery'] = float(agree)
    return res


def main(seeds=(0, 1, 2)):
    out_dir = PROJECT_ROOT / 'results' / 'structured'
    out_dir.mkdir(parents=True, exist_ok=True)
    R = {}
    for name, gen in GENERATORS.items():
        R[name] = {}
        for seed in seeds:
            t0 = time.time()
            X, y, reg = gen(seed=seed)
            R[name][str(seed)] = run_cell(X, y, reg, seed)
            print(f"[{name}] seed {seed} done {time.time()-t0:.1f}s", flush=True)
            json.dump(R, open(out_dir / 'results.json', 'w'), indent=2)

    print("\n" + "=" * 100)
    print("ROUTING ON DATA WITH REGIONAL STRUCTURE TRUE BY CONSTRUCTION (ECE, 3 seeds)")
    print("=" * 100)
    print(f"{'dataset':22s} {'uniform':>10s} {'true_oracle':>12s} {'learned_orc':>12s}"
          f" {'importance':>11s} {'random':>10s} {'recovery':>9s}")
    print("-" * 100)
    gaps = []
    for name, cells in R.items():
        c = list(cells.values())
        f = lambda k: np.mean([x[k]['ece'] for x in c])
        rec = np.mean([x['partition_recovery'] for x in c])
        gaps.append(f('true_oracle') - f('uniform_best'))
        print(f"{name:22s} {f('uniform_best'):10.4f} {f('true_oracle'):12.4f}"
              f" {f('learned_oracle'):12.4f} {f('importance'):11.4f} {f('random'):10.4f}"
              f" {rec*100:8.1f}%")
    print(f"\n  mean (true_oracle - uniform_best) = {np.mean(gaps):+.4f}")
    print("  -> " + ("REGIONAL STRUCTURE IS EXPLOITABLE" if np.mean(gaps) < -0.002
                     else "no gain even with the true partition and an oracle model choice"))
    print("saved ->", out_dir / 'results.json')


if __name__ == '__main__':
    main()
