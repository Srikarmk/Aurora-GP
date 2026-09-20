"""
Datasets with regional structure that is true BY CONSTRUCTION.

The five benchmarks in data/ were never checked for the property AURORA targets:
input-dependent noise, density, or smoothness. If they are broadly homogeneous,
"routing does not help" says nothing about the method -- the test simply did not
exercise it. These generators make the premise true and controllable, so the
question becomes answerable.

Each returns X, y, and the TRUE generative region label, which lets us separate
"is there structure to exploit" from "can region identification find it".
"""
import numpy as np


def _tertiles(x0):
    """Region label from position: 0 / 1 / 2 across the first coordinate."""
    r = np.ones(len(x0), dtype=int)
    r[x0 < -1 / 3] = 0
    r[x0 > 1 / 3] = 2
    return r


def hetero_extreme(n=6000, seed=0, ratio=50.0):
    """Noise std varies by ratio across three regions. Everything else constant."""
    rs = np.random.RandomState(seed)
    X = rs.uniform(-1, 1, size=(n, 2))
    reg = _tertiles(X[:, 0])
    f = np.sin(3 * X[:, 0]) * np.cos(3 * X[:, 1])
    base = 0.02
    sigma = np.choose(reg, [base, base * np.sqrt(ratio), base * ratio])
    return X, f + rs.randn(n) * sigma, reg


def varying_smoothness(n=6000, seed=0):
    """Effective lengthscale varies smoothly: nearly flat on the left, highly
    oscillatory on the right. Homoscedastic noise."""
    rs = np.random.RandomState(seed)
    X = rs.uniform(-1, 1, size=(n, 2))
    reg = _tertiles(X[:, 0])
    ramp = (X[:, 0] + 1) / 2                       # 0 -> 1, smooth
    f = np.sin(2 * X[:, 0]) + ramp * 0.7 * np.sin(14 * X[:, 0]) * np.cos(10 * X[:, 1])
    return X, f + rs.randn(n) * 0.05, reg


def varying_density(n=6000, seed=0, frac=0.85):
    """Most samples concentrated in a small blob; the rest sparse. Smooth f."""
    rs = np.random.RandomState(seed)
    n_blob = int(n * frac)
    blob = rs.randn(n_blob, 2) * 0.07 + np.array([0.55, 0.0])
    sparse = rs.uniform(-1, 1, size=(n - n_blob, 2))
    X = np.clip(np.vstack([blob, sparse]), -1, 1)
    idx = rs.permutation(n); X = X[idx]
    reg = _tertiles(X[:, 0])
    f = np.sin(3 * X[:, 0]) * np.cos(3 * X[:, 1])
    return X, f + rs.randn(n) * 0.05, reg


def combined(n=6000, seed=0):
    """All three effects at once -- the most favourable possible case."""
    rs = np.random.RandomState(seed)
    n_blob = int(n * 0.7)
    blob = rs.randn(n_blob, 2) * 0.09 + np.array([0.55, 0.0])
    sparse = rs.uniform(-1, 1, size=(n - n_blob, 2))
    X = np.clip(np.vstack([blob, sparse]), -1, 1)
    X = X[rs.permutation(n)]
    reg = _tertiles(X[:, 0])
    ramp = (X[:, 0] + 1) / 2
    f = np.sin(2 * X[:, 0]) + ramp * 0.7 * np.sin(14 * X[:, 0]) * np.cos(10 * X[:, 1])
    sigma = np.choose(reg, [0.02, 0.02 * np.sqrt(30), 0.02 * 30])
    return X, f + rs.randn(n) * sigma, reg


GENERATORS = {
    'hetero_extreme': hetero_extreme,
    'varying_smoothness': varying_smoothness,
    'varying_density': varying_density,
    'combined': combined,
}


def describe(name, seed=0):
    """Verify the intended structure is actually present."""
    X, y, reg = GENERATORS[name](seed=seed)
    from sklearn.neighbors import NearestNeighbors
    nb = NearestNeighbors(n_neighbors=11).fit(X)
    d = nb.kneighbors(X)[0][:, 1:].mean(1)
    out = {'name': name, 'n': len(X)}
    for r in (0, 1, 2):
        m = reg == r
        out[f'region{r}'] = {
            'n': int(m.sum()),
            'y_std': float(y[m].std()),
            'mean_knn_dist': float(d[m].mean()),
        }
    out['density_ratio'] = float(d[reg == 0].mean() / d[reg == 2].mean())
    return out


if __name__ == '__main__':
    import json
    for k in GENERATORS:
        print(json.dumps(describe(k), indent=None))
