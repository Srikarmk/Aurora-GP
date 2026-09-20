"""Aggregate the inference-corruption sweep."""
import json, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
F = PROJECT_ROOT / 'results' / 'inference' / 'corruption.json'
MODES = ['repo_default', 'love_off', 'cholesky_on', 'exact']


def main():
    R = json.load(open(F))
    L = []; W = L.append
    W("=" * 104)
    W("PREDICTIVE-VARIANCE CORRUPTION vs TRAINING SET SIZE")
    W("  One fitted model per cell; only the inference path varies. 3 seeds.")
    W("  rel.err = median over test points of |sigma_mode - sigma_exact| / sigma_exact")
    W("=" * 104)
    for ds, ns in R.items():
        W(f"\n### {ds}")
        W(f"{'n_train':>8s} {'ECE repo':>10s} {'ECE exact':>10s} {'ECE ratio':>10s}"
          f" {'rel.err repo':>13s} {'rel.err LOVE off':>17s} {'rel.err chol raised':>20s}")
        W("-" * 96)
        for n, cells in ns.items():
            c = list(cells.values())
            f = lambda m, k: np.mean([x[m][k] for x in c])
            er, ex = f('repo_default', 'ece'), f('exact', 'ece')
            W(f"{int(n):8d} {er:10.4f} {ex:10.4f} {er/max(ex,1e-9):9.2f}x"
              f" {f('repo_default','std_relerr_median')*100:12.2f}%"
              f" {f('love_off','std_relerr_median')*100:16.2f}%"
              f" {f('cholesky_on','std_relerr_median')*100:19.2f}%")

    W("\n" + "=" * 104)
    W("WHICH SETTING IS RESPONSIBLE?")
    W("  love_off    = fast_pred_var OFF, max_cholesky_size left at its 800 default")
    W("  cholesky_on = fast_pred_var ON,  max_cholesky_size raised above n")
    W("=" * 104)
    rows = {m: [] for m in MODES}
    for ds, ns in R.items():
        for n, cells in ns.items():
            if int(n) <= 800:
                continue
            for m in MODES:
                rows[m].append(np.mean([x[m]['std_relerr_median'] for x in cells.values()]))
    W(f"\n  Across all cells with n_train > 800 ({len(rows['exact'])} cells):")
    for m in MODES:
        W(f"    {m:14s} median rel.err in sigma = {np.median(rows[m])*100:7.2f}%")
    W("\n  Raising max_cholesky_size alone does NOT help; turning fast_pred_var off does.")
    W("  fast_pred_var() is a no-op below max_cholesky_size, which is why the error is")
    W("  exactly 0.00% at every n <= 800 and only appears above it.")

    W("\n" + "=" * 104)
    W("ONSET")
    W("=" * 104)
    below = [];  above = []
    for ds, ns in R.items():
        for n, cells in ns.items():
            v = np.mean([x['repo_default']['std_relerr_median'] for x in cells.values()])
            (below if int(n) <= 800 else above).append(v)
    W(f"  n_train <= 800 : max rel.err over {len(below)} cells = {max(below)*100:.4f}%")
    W(f"  n_train >  800 : min rel.err over {len(above)} cells = {min(above)*100:.4f}%,"
      f" max = {max(above)*100:.2f}%")

    txt = "\n".join(L)
    print(txt)
    (F.parent / 'report.txt').write_text(txt)
    print(f"\nsaved -> {F.parent / 'report.txt'}")


if __name__ == '__main__':
    main()
