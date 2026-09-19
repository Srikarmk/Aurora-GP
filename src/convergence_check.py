"""
Is the original exact-GP baseline converged?

gp_baseline.py:114 optimizes with Adam at lr=0.1 for n_iter steps (50 in
run_baseline_experiments, 30 inside AURORA). This varies ONLY n_iter, holding
data, splits, model and seed fixed, and reports how the exact GP's calibration
moves.

This matters because the fair benchmark's exact-GP arm used that baseline, and a
properly fitted exact GP (L-BFGS on the exact marginal likelihood, as used in
mechanism.py) came out dramatically better calibrated on the same splits:

    robot_arm  0.2701 (baseline, 50 iters)  vs  0.0168 (properly fitted)
    synthetic  0.2738                       vs  0.0243
    sarcos     0.3178                       vs  0.0932
    protein    0.1170                       vs  0.0246

If ECE falls toward the properly-fitted values as n_iter grows, then the
"sparse approximations are better calibrated than exact GPs" result in
RESULTS_FAIR.md section 4 is an artifact of an undertrained optimizer, not a
property of sparse GPs, and must be withdrawn.
"""
import numpy as np, sys, json, time, warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))
PROJECT_ROOT = Path(__file__).parent.parent

from sklearn.model_selection import train_test_split
from fair_benchmark import m_ece, m_rmse, m_nll, MAX_N, quiet
from gp_baseline import GaussianProcessBaseline

ITERS = [50, 150, 400, 1000]
SEEDS = (0, 1, 2)


def main():
    out_dir = PROJECT_ROOT / 'results' / 'mechanism'
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    print(f"{'dataset':26s} {'n_iter':>7s} {'ECE':>17s} {'NLL':>17s} {'RMSE':>17s} {'noise':>8s}", flush=True)
    print("-" * 100, flush=True)

    for f in sorted((PROJECT_ROOT / 'data').glob('*.npz')):
        name = f.stem
        d = np.load(f, allow_pickle=True)
        X, y = d['X'], d['y']
        acc = {it: {'ece': [], 'nll': [], 'rmse': [], 'noise': []} for it in ITERS}

        for seed in SEEDS:
            rs = np.random.RandomState(seed)
            if len(X) > MAX_N:
                i = rs.choice(len(X), MAX_N, replace=False)
                Xc, yc = X[i], y[i]
            else:
                Xc, yc = X, y
            # identical split policy to fair_benchmark.run_cell
            X_tr, X_tmp, y_tr, y_tmp = train_test_split(Xc, yc, test_size=0.4, random_state=seed)
            _, X_te, _, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=seed)

            for it in ITERS:
                with quiet():
                    g = GaussianProcessBaseline(max_train_size=5000, random_state=seed, use_gpu=False)
                    g.fit(X_tr, y_tr, n_iter=it)
                    p, s = g.predict(X_te)
                acc[it]['ece'].append(m_ece(y_te, p, s))
                acc[it]['nll'].append(m_nll(y_te, p, s))
                acc[it]['rmse'].append(m_rmse(y_te, p))
                acc[it]['noise'].append(float(g.likelihood.noise.item()))

        res[name] = {str(it): {k: [float(x) for x in v] for k, v in acc[it].items()} for it in ITERS}
        for it in ITERS:
            a = acc[it]
            lab = name if it == ITERS[0] else ''
            print(f"{lab:26s} {it:7d} "
                  f"{np.mean(a['ece']):7.4f}+/-{np.std(a['ece'], ddof=1):7.4f} "
                  f"{np.mean(a['nll']):7.3f}+/-{np.std(a['nll'], ddof=1):7.3f} "
                  f"{np.mean(a['rmse']):7.4f}+/-{np.std(a['rmse'], ddof=1):7.4f} "
                  f"{np.mean(a['noise']):8.4f}", flush=True)
        print(flush=True)
        json.dump(res, open(out_dir / 'convergence.json', 'w'), indent=2)

    print("saved ->", out_dir / 'convergence.json', flush=True)


if __name__ == '__main__':
    main()
