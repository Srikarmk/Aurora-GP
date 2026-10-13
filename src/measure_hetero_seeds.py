"""
het_ratio over ten seeds.

measure_hetero.py reports one seed, and on the small datasets that single draw is not
representative (yacht ranges 3.5-20 across seeds). This repeats the same measurement
on seeds 0-9 and reports mean, sd, min and max per dataset. The single-seed file is
kept because hetero_fair.py and diff_seeds.py read it.
"""
import numpy as np, json, sys, warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))
PROJECT_ROOT = Path(__file__).parent.parent

from measure_hetero import measure
from structured_data import GENERATORS

SEEDS = range(10)


def main():
    sets = []
    for d in sorted((PROJECT_ROOT / 'data').glob('*.npz')):
        z = np.load(d, allow_pickle=True); sets.append(('original', d.stem, z['X'], z['y']))
    for d in sorted((PROJECT_ROOT / 'data' / 'real').glob('*.npz')):
        z = np.load(d, allow_pickle=True); sets.append(('real', d.stem, z['X'], z['y']))
    for k, g in GENERATORS.items():
        X, y, _ = g(seed=0); sets.append(('synthetic', k, X, y))

    out = {}
    for group, name, X, y in sets:
        r = [measure(X, y, seed=s)['het_ratio'] for s in SEEDS]
        out[name] = {'group': group, 'het_ratio': r, 'mean': float(np.mean(r)),
                     'sd': float(np.std(r, ddof=1)), 'min': float(min(r)), 'max': float(max(r))}
        print(f"{name:28s} mean {np.mean(r):9.2f}  sd {np.std(r, ddof=1):8.2f}  "
              f"range {min(r):8.2f}-{max(r):8.2f}", flush=True)
    json.dump(out, open(PROJECT_ROOT / 'results' / 'hetero_screen_seeds.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
