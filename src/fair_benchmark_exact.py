"""
The fair benchmark re-run with EXACT GP inference.

Identical protocol to fair_benchmark.py. The only difference is that
routing_recheck imports a corrected GaussianProcessBaseline.predict (exact
Cholesky, no fast_pred_var) before the benchmark runs -- see AUDIT.md Tier 1
finding 6. Results go to results/fair_exact/ so the two are comparable side by
side.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import routing_recheck            # noqa: F401 -- patches predict() on import
import fair_benchmark

if __name__ == '__main__':
    fair_benchmark.main(out_name='fair_exact')
