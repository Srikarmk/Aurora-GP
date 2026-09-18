"""Aggregate the mechanism study (src/mechanism.py) over seeds."""
import json, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MD = PROJECT_ROOT / 'results' / 'mechanism'
RANKS = [25, 50, 100, 200, 400, 800, 1600]


def mstd(v, p=4):
    v = np.asarray(v, float)
    return f"{v.mean():.{p}f}+/-{v.std(ddof=1):.{p}f}" if len(v) > 1 else f"{v.mean():.{p}f}"


def main():
    A = json.load(open(MD / 'rank_sweep.json'))
    B = json.load(open(MD / 'oracle_routing.json'))
    L = []
    W = L.append

    W("=" * 100)
    W("EXPERIMENT A — rank sweep with hyperparameters HELD FIXED")
    W("  Only the Nystrom rank changes. lengthscale/signal/noise are fitted once per")
    W("  (dataset, seed) by exact marginal likelihood and shared by every row.")
    W("  z_disp = std of (y-mu)/sigma;  1.0 = calibrated, >1 = overconfident.")
    W("=" * 100)
    for ds, seeds in A.items():
        cells = list(seeds.values())
        if not cells:
            continue
        h = cells[0]['hypers']
        W(f"\n### {ds}   (n_train {cells[0]['n_train']}, {len(cells)} seeds; "
          f"fitted ell={np.mean([c['hypers']['lengthscale'] for c in cells]):.2f} "
          f"sig_f={np.mean([c['hypers']['signal_std'] for c in cells]):.2f} "
          f"sig_n={np.mean([c['hypers']['noise_std'] for c in cells]):.2f})")
        W(f"{'rank':>8s} {'ECE':>16s} {'NLL':>16s} {'RMSE':>16s} {'mean_std':>10s} {'z_disp':>8s}")
        W("-" * 82)
        for m in RANKS:
            got = [c['ranks'][str(m)] for c in cells if str(m) in c.get('ranks', {})]
            if not got:
                continue
            W(f"{m:>8d} {mstd([g['ece'] for g in got]):>16s} {mstd([g['nll'] for g in got],3):>16s}"
              f" {mstd([g['rmse'] for g in got]):>16s} {np.mean([g['mean_std'] for g in got]):10.3f}"
              f" {np.mean([g['z_disp'] for g in got]):8.3f}")
        ex = [c['exact'] for c in cells]
        W(f"{'exact':>8s} {mstd([g['ece'] for g in ex]):>16s} {mstd([g['nll'] for g in ex],3):>16s}"
          f" {mstd([g['rmse'] for g in ex]):>16s} {np.mean([g['mean_std'] for g in ex]):10.3f}"
          f" {np.mean([g['z_disp'] for g in ex]):8.3f}   (rank {ex[0]['rank']})")
        # where is the optimum?
        curve = []
        for m in RANKS:
            got = [c['ranks'][str(m)]['ece'] for c in cells if str(m) in c.get('ranks', {})]
            if got:
                curve.append((m, float(np.mean(got))))
        curve.append(('exact', float(np.mean([g['ece'] for g in ex]))))
        best = min(curve, key=lambda t: t[1])
        W(f"   best ECE at rank {best[0]} ({best[1]:.4f});  exact GP = {curve[-1][1]:.4f}"
          f"   -> {'LOW RANK WINS' if best[0] != 'exact' else 'exact wins'}")

    W("\n" + "=" * 100)
    W("EXPERIMENT B — is there any headroom for region-aware routing?")
    W("  uniform_best : single best model everywhere      (selected on validation)")
    W("  oracle_region: best model PER REGION             (selected on validation)")
    W("  importance   : AURORA's routing, same models")
    W("  random       : routing shuffled, proportions preserved")
    W("=" * 100)
    W(f"\n{'dataset':26s} {'uniform_best':>17s} {'oracle_region':>17s} {'importance':>17s} {'random':>17s}")
    W("-" * 98)
    head = []
    for ds, seeds in B.items():
        cells = list(seeds.values())
        if not cells:
            continue
        g = lambda k: [c[k]['ece'] for c in cells]
        W(f"{ds:26s} {mstd(g('uniform_best')):>17s} {mstd(g('oracle_region')):>17s}"
          f" {mstd(g('importance_routing')):>17s} {mstd(g('random_routing')):>17s}")
        head.append(np.mean(g('oracle_region')) - np.mean(g('uniform_best')))
    if head:
        W(f"\n  mean (oracle_region - uniform_best) ECE = {np.mean(head):+.4f}"
          f"   -> {'routing has headroom' if np.mean(head) < -0.002 else 'NO headroom for routing'}")

    W("\n  Model chosen per region by the oracle (validation ECE):")
    for ds, seeds in B.items():
        picks = [tuple(c['oracle_region']['choice'][k] for k in ('0', '1', '2'))
                 for c in seeds.values()]
        uni = [c['uniform_best']['model'] for c in seeds.values()]
        W(f"    {ds:26s} low/med/high = {picks}   uniform_best={uni}")

    W("\n" + "=" * 100)
    W("Per-region ECE of every model (each model evaluated on ALL test points)")
    W("  If one column wins in every region, region-aware routing cannot help.")
    W("=" * 100)
    for ds, seeds in B.items():
        cells = list(seeds.values())
        W(f"\n### {ds}")
        W(f"{'region':10s} " + "".join(f"{k:>12s}" for k in ('rff', 'nys', 'nys_hi', 'gp')) + "   best")
        W("-" * 68)
        for reg in ('low', 'medium', 'high', 'overall'):
            vals = {}
            for k in ('rff', 'nys', 'nys_hi', 'gp'):
                v = [c['per_region_ece'][k][reg] for c in cells
                     if c['per_region_ece'][k].get(reg) is not None]
                if v:
                    vals[k] = float(np.mean(v))
            if not vals:
                continue
            best = min(vals, key=vals.get)
            W(f"{reg:10s} " + "".join(f"{vals.get(k, float('nan')):12.4f}"
                                      for k in ('rff', 'nys', 'nys_hi', 'gp')) + f"   {best}")

    txt = "\n".join(L)
    print(txt)
    (MD / 'report.txt').write_text(txt)
    print(f"\nsaved -> {MD / 'report.txt'}")


if __name__ == '__main__':
    main()
