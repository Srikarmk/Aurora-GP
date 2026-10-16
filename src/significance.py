"""
Significance for the noise-model comparison and the routing control, with multiple-
comparison correction.

Noise models (results/hetero_fair.json, seeds 0-9; hetero_fair_seeds10_19.json):
per (metric, arm) family of eleven datasets, a paired two-sided Wilcoxon signed-rank
test of arm vs global, Holm and Benjamini-Hochberg adjusted, for seeds 0-9, 10-19 and
the twenty pooled. Seeds are resampled splits of the same data, not new data, so a
Nadeau-Bengio corrected resampled t-test is also reported for regional ECE.

Routing (results/mechanism/oracle_routing_exact.json): importance vs random routing,
paired t-test over five seeds (a Wilcoxon test cannot go below p=0.0625 with five),
Holm adjusted over the five benchmarks.

Also reports how much train and test sets overlap across seeds.
"""
import itertools, json, numpy as np
from pathlib import Path
from scipy.stats import wilcoxon, ttest_rel, t as tdist
from sklearn.model_selection import train_test_split

P = Path(__file__).parent.parent
ARMS = ('regional', 'het_gp')
METRICS = ('ece', 'nll', 'crps')
CAP = 2500


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    a = np.minimum(np.maximum.accumulate((m - np.arange(m)) * p[o]), 1)
    out = np.empty(m); out[o] = a; return out


def bh(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    a = np.minimum.accumulate(((m / (np.arange(m) + 1)) * p[o])[::-1])[::-1]
    out = np.empty(m); out[o] = np.minimum(a, 1); return out


def wil(d):
    d = np.asarray(d)
    return float(wilcoxon(d).pvalue) if np.ptp(d) > 0 else 1.0


def nadeau_bengio(d, test_train_ratio):
    """Corrected resampled t-test (Nadeau & Bengio 2003)."""
    d = np.asarray(d); J = len(d)
    se = np.sqrt(d.var(ddof=1) * (1 / J + test_train_ratio))
    return float(2 * tdist.sf(abs(d.mean() / se), J - 1))


def diffs(sets, ds, arm, m):
    return np.concatenate([[R[ds]['seeds'][k][arm][m] - R[ds]['seeds'][k]['global'][m]
                            for k in sorted(R[ds]['seeds'], key=int)] for R in sets])


def noise():
    A = json.load(open(P / 'results' / 'hetero_fair.json'))
    B = json.load(open(P / 'results' / 'hetero_fair_seeds10_19.json'))
    ds = sorted(A)
    out = {}
    for label, sets in (('seeds0-9', [A]), ('seeds10-19', [B]), ('pooled', [A, B])):
        out[label] = {}
        for m in METRICS:
            for arm in ARMS:
                d = {k: diffs(sets, k, arm, m) for k in ds}
                p = [wil(d[k]) for k in ds]
                h, b = holm(p), bh(p)
                rows = {k: {'mean_diff': float(d[k].mean()), 'p': p[i], 'holm': float(h[i]),
                            'bh': float(b[i])} for i, k in enumerate(ds)}
                if m == 'ece' and arm == 'regional':
                    for k in ds:   # 60/20/20 split: n_test / n_train = 1/3
                        rows[k]['nadeau_bengio_p'] = nadeau_bengio(d[k], 1 / 3)
                count = lambda col: {'better': sum(rows[k][col] < .05 and rows[k]['mean_diff'] < 0 for k in ds),
                                     'worse': sum(rows[k][col] < .05 and rows[k]['mean_diff'] > 0 for k in ds)}
                out[label][f'{m}/{arm}'] = {'datasets': rows, 'raw': count('p'),
                                            'holm': count('holm'), 'bh': count('bh')}
    return out


def overlap():
    files = {f.stem: f for f in list((P / 'data' / 'real').glob('*.npz')) + list((P / 'data').glob('*.npz'))}
    out = {}
    for name, f in sorted(files.items()):
        N = len(np.load(f, allow_pickle=True)['y'])
        S = []
        for seed in range(10):
            rs = np.random.RandomState(seed); idx = np.arange(N)
            if N > CAP:
                idx = rs.choice(N, CAP, replace=False)
            tr, tmp = train_test_split(idx, test_size=.4, random_state=seed)
            _, te = train_test_split(tmp, test_size=.5, random_state=seed)
            S.append((set(tr), set(te)))
        pair = lambda j: float(np.mean([len(a[j] & b[j]) / len(a[j]) for a, b in itertools.combinations(S, 2)]))
        out[name] = {'n': N, 'train_overlap': pair(0), 'test_overlap': pair(1)}
    return out


def routing():
    R = json.load(open(P / 'results' / 'mechanism' / 'oracle_routing_exact.json'))
    ds = sorted(R)
    g = lambda d, a: np.array([R[d][s][a]['ece'] for s in sorted(R[d], key=int)])
    p = [float(ttest_rel(g(d, 'importance_routing'), g(d, 'random_routing')).pvalue) for d in ds]
    h = holm(p)
    return {d: {'mean_diff': float(np.mean(g(d, 'importance_routing') - g(d, 'random_routing'))),
                'p': p[i], 'holm': float(h[i])} for i, d in enumerate(ds)}


def main():
    res = {'noise': noise(), 'overlap': overlap(), 'routing_importance_vs_random': routing()}
    json.dump(res, open(P / 'results' / 'significance.json', 'w'), indent=2)
    for label, fam in res['noise'].items():
        print(f"\n== {label}")
        for key, v in fam.items():
            sig = [k for k, r in v['datasets'].items() if r['holm'] < .05]
            print(f"  {key:14s} raw {v['raw']}  holm {v['holm']}  bh {v['bh']}  holm-sig: {sig}")
    print("\n== regional ECE, Nadeau-Bengio p (seeds 0-9 / pooled)")
    for k, r in res['noise']['seeds0-9']['ece/regional']['datasets'].items():
        print(f"  {k:26s} {r['nadeau_bengio_p']:.4f} / "
              f"{res['noise']['pooled']['ece/regional']['datasets'][k]['nadeau_bengio_p']:.4f}")
    print("\n== split overlap across seeds 0-9")
    for k, r in res['overlap'].items():
        print(f"  {k:26s} n={r['n']:6d} train {r['train_overlap']:.2f} test {r['test_overlap']:.2f}")
    print("\n== importance vs random routing")
    for k, r in res['routing_importance_vs_random'].items():
        print(f"  {k:26s} diff {r['mean_diff']:+.4f} p {r['p']:.4f} holm {r['holm']:.4f}")


if __name__ == '__main__':
    main()
