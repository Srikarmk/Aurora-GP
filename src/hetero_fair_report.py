"""
Paper Table 5 from results/hetero_fair.json: per-dataset means, paired Wilcoxon
p-values against the global arm, and CRPS relative to global (scale-free).
"""
import json, sys, numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

P = Path(__file__).parent.parent
ARMS = ('global', 'het_gp', 'regional')


def pval(a, b):
    d = np.asarray(a) - np.asarray(b)
    return wilcoxon(d).pvalue if np.ptp(d) > 0 else 1.0


def main(path=P / 'results' / 'hetero_fair.json'):
    R = json.load(open(path))
    wins = {m: dict.fromkeys(ARMS, 0) for m in ('ece', 'nll', 'crps')}
    sig = {m: {a: [0, 0] for a in ARMS[1:]} for m in ('ece', 'nll', 'crps')}
    print(f"{'dataset':26s}{'het':>6s}" + "".join(f"{a + ' ' + m:>17s}" for m in ('ece',) for a in ARMS)
          + f"{'p het':>8s}{'p reg':>8s}" + f"{'crps het/g':>12s}{'crps reg/g':>12s}{'nll best':>10s}")
    for name, d in sorted(R.items(), key=lambda kv: -(kv[1]['het_ratio'] or 0)):
        c = list(d['seeds'].values())
        v = {m: {a: np.array([x[a][m] for x in c]) for a in ARMS} for m in ('ece', 'nll', 'crps')}
        for m in wins:
            wins[m][min(ARMS, key=lambda a: v[m][a].mean())] += 1
            for a in ARMS[1:]:
                if pval(v[m][a], v[m]['global']) < 0.05:
                    sig[m][a][0 if v[m][a].mean() < v[m]['global'].mean() else 1] += 1
        ratio = {a: np.mean(v['crps'][a] / v['crps']['global']) for a in ARMS[1:]}
        print(f"{name:26s}{d['het_ratio'] or 0:6.2f}"
              + "".join(f"{v['ece'][a].mean():17.4f}" for a in ARMS)
              + f"{pval(v['ece']['het_gp'], v['ece']['global']):8.3f}"
              + f"{pval(v['ece']['regional'], v['ece']['global']):8.3f}"
              + f"{ratio['het_gp']:12.4f}{ratio['regional']:12.4f}"
              + f"{min(ARMS, key=lambda a: v['nll'][a].mean()):>10s}")
    for m in wins:
        print(f"\n  best-arm count ({m}): {wins[m]}")
        print(f"  significantly better / worse than global at p<0.05 ({m}): {sig[m]}")


if __name__ == '__main__':
    main(*sys.argv[1:])
