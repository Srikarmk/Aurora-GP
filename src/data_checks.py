"""
Numbers quoted in the paper's protocol section and data/README.md that describe the
data and the metric rather than any model: the ECE noise floor by test-set size, and
the duplicate / censoring rates under the noise-comparison protocol.
"""
import json, sys, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
P = Path(__file__).parent.parent
from fair_benchmark import m_ece
from hetero_fair import CAP, SEEDS
from sklearn.model_selection import train_test_split


def ece_floor(n, reps=4000):
    rs = np.random.RandomState(0)
    e = [m_ece(rs.randn(n), np.zeros(n), np.ones(n)) for _ in range(reps)]
    return {'mean': float(np.mean(e)), 'p95': float(np.percentile(e, 95))}


def twin_rate(X, y):
    """Share of test points with an exact training twin, hetero_fair protocol."""
    out = []
    for seed in SEEDS:
        rs = np.random.RandomState(seed)
        Xs, ys = X, y
        if len(X) > CAP:
            i = rs.choice(len(X), CAP, replace=False); Xs, ys = X[i], y[i]
        X_tr, X_tmp = train_test_split(Xs, test_size=.4, random_state=seed)
        _, X_te = train_test_split(X_tmp, test_size=.5, random_state=seed)
        tr = {r.tobytes() for r in np.ascontiguousarray(X_tr)}
        out.append(np.mean([r.tobytes() in tr for r in np.ascontiguousarray(X_te)]))
    return float(np.mean(out))


def main():
    R = {'ece_floor': {str(n): ece_floor(n) for n in (62, 154, 206, 301, 400, 500)}}
    for name, path in (('sarcos', 'data/sarcos.npz'), ('wine_white', 'data/real/wine_white.npz'),
                       ('protein', 'data/protein.npz'), ('powerplant', 'data/real/powerplant.npz'),
                       ('concrete', 'data/concrete.npz')):
        z = np.load(P / path, allow_pickle=True)
        R[name] = {'test_twin_rate': twin_rate(z['X'], z['y']),
                   'duplicate_X_rows': int(len(z['X']) - len(np.unique(z['X'], axis=0)))}
    y = np.load(P / 'data/real/calhousing.npz')['y']
    R['calhousing'] = {'censored_frac': float(np.mean(y >= y.max() - 1e-6))}
    json.dump(R, open(P / 'results' / 'data_checks.json', 'w'), indent=2)
    print(json.dumps(R, indent=1))


if __name__ == '__main__':
    main()
