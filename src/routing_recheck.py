"""
Re-run the routing headroom test with EXACT GP inference.

Why this exists: gp_baseline.py:157 calls gpytorch.settings.fast_pred_var(), and
GPyTorch falls back to iterative CG/Lanczos above max_cholesky_size (default 800).
Both approximate the predictive VARIANCE -- the quantity ECE measures and the
quantity RegionIdentifier builds its importance scores from. Above n=800 the
partition itself is therefore computed from approximated uncertainties. Measured
region agreement between approximate and exact inference:

    concrete   (n=618)  100.0%      <- below the threshold, unaffected
    synthetic  (n=3000)  99.8%
    sarcos     (n=4800)  88.8%
    robot_arm  (n=4800)  79.3%
    protein    (n=4800)  70.2%      <- a third of points change region

So the oracle-routing result in results/mechanism/ was computed on a partition that
is wrong on two datasets. This re-runs it with exact inference throughout.

The original gp_baseline.py is NOT modified -- it is the evidence the audit cites.
The corrected predict() is monkeypatched in here instead, so the fix is visible and
auditable at the point of use.
"""
import numpy as np, json, sys, time, warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))
PROJECT_ROOT = Path(__file__).parent.parent

import torch, gpytorch
from gp_baseline import GaussianProcessBaseline

_EXACT = dict(covar_root_decomposition=False, log_prob=False, solves=False)


def _exact_predict(self, X_test, return_std=True):
    """Drop-in for GaussianProcessBaseline.predict using exact Cholesky inference."""
    Xs = self.scaler_X.transform(X_test)
    t = torch.tensor(Xs, dtype=torch.float32).to(self.device)
    self.model.eval(); self.likelihood.eval()
    with torch.no_grad(), gpytorch.settings.max_cholesky_size(100000), \
         gpytorch.settings.fast_pred_var(False), \
         gpytorch.settings.fast_computations(**_EXACT):
        d = self.likelihood(self.model(t))
        m = d.mean.cpu().numpy(); v = d.variance.cpu().numpy()
    mean = self.scaler_y.inverse_transform(m.reshape(-1, 1)).ravel()
    if not return_std:
        return mean
    return mean, np.sqrt(np.maximum(v, 1e-12)) * self.scaler_y.scale_


GaussianProcessBaseline.predict = _exact_predict          # applied before import below

from mechanism import experiment_B                        # noqa: E402  (uses patched predict)


def main(seeds=(0, 1, 2, 3, 4)):
    out_dir = PROJECT_ROOT / 'results' / 'mechanism'
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    for f in sorted((PROJECT_ROOT / 'data').glob('*.npz')):
        ds = f.stem
        d = np.load(f, allow_pickle=True); X, y = d['X'], d['y']
        res[ds] = {}
        for seed in seeds:
            t0 = time.time()
            print(f"[{ds}] seed {seed}", flush=True)
            res[ds][str(seed)] = experiment_B(X, y, seed)
            print(f"   done {time.time()-t0:.1f}s", flush=True)
            json.dump(res, open(out_dir / 'oracle_routing_exact.json', 'w'), indent=2)

    print("\n" + "=" * 92)
    print("ROUTING HEADROOM WITH EXACT INFERENCE (compare results/mechanism/report.txt)")
    print("=" * 92)
    print(f"{'dataset':26s} {'uniform_best':>15s} {'oracle_region':>15s} {'importance':>15s} {'random':>15s}")
    print("-" * 92)
    gaps = []
    for ds, cells in res.items():
        c = list(cells.values())
        g = lambda k: np.mean([x[k]['ece'] for x in c])
        gaps.append(g('oracle_region') - g('uniform_best'))
        print(f"{ds:26s} {g('uniform_best'):15.4f} {g('oracle_region'):15.4f}"
              f" {g('importance_routing'):15.4f} {g('random_routing'):15.4f}")
    print(f"\n  mean (oracle_region - uniform_best) = {np.mean(gaps):+.4f}"
          f"   -> {'routing has headroom' if np.mean(gaps) < -0.002 else 'NO headroom for routing'}")
    print("saved ->", out_dir / 'oracle_routing_exact.json')


if __name__ == '__main__':
    main()
