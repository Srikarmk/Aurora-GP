"""Write the paper's result tables as LaTeX (paper/tables/*.tex) from results/."""
import json, numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

P = Path(__file__).parent.parent
T = P / 'paper' / 'tables'; T.mkdir(parents=True, exist_ok=True)
L = lambda f: json.load(open(P / 'results' / f))
tex = lambda s: s.replace('_', r'\_')
ORIGIN = {'robot_arm': 'simulated', 'energy': 'simulated',
          'synthetic_heteroscedastic': 'constructed'}


def pval(a, b):
    d = np.asarray(a) - np.asarray(b)
    return wilcoxon(d).pvalue if np.ptp(d) > 0 else 1.0


HEAD = {
    'orthogonality': ('llrrrrrr',
        'dataset & region & RFF & Nystr\\"om-500 & Nystr\\"om-1500 & exact GP & ECE spread & $\\sigma$ spread \\\\'),
    'screen': ('llrrr',
        'origin & dataset & het (mean) & sd & range \\\\'),
    'noise': ('llrrrrr',
        'dataset & origin & $n_{\\text{train}}$ & het & global & het-GP & regional \\\\'),
    'nsweep': ('rrrrrr',
        '$n_{\\text{train}}$ & calhousing & powerplant & wine\\_white & airfoil & mean \\\\'),
    'shrinkage': ('rrrrrr',
        '$n_{\\text{train}}$ & datasets & regional & shrunk & EB & $\\bar\\lambda$ \\\\'),
}


def write(name, body):
    cols, head = HEAD[name]
    (T / f'{name}.tex').write_text(
        f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head}\n\\midrule\n{body}"
        "\\bottomrule\n\\end{tabular}\n")
    print('wrote', name)


def orthogonality():
    R = L('structured/orthogonality.json')
    arms = ['rff', 'nys', 'nys_hi', 'gp']
    rows = []
    for ds, cells in R.items():
        c = list(cells.values())
        for j, r in enumerate(sorted(c[0], key=int)):
            e = [np.mean([x[r]['ece'][a] for x in c]) for a in arms]
            s = [np.mean([x[r]['mean_std'][a] for x in c]) for a in arms]
            sp = 100 * (max(s) - min(s)) / np.mean(s)
            lead = rf'\texttt{{{tex(ds)}}}' if j == 0 else ''
            rows.append(f"{lead} & {r} & " + " & ".join(f"{v:.4f}" for v in e)
                        + f" & {max(e) - min(e):.4f} & {sp:.2f}\\% \\\\")
        rows.append(r'\midrule')
    write('orthogonality', "\n".join(rows[:-1]) + "\n")


def noise():
    R = L('hetero_fair.json')
    seeds = {k: v for k, v in L('hetero_screen_seeds.json').items()} \
        if (P / 'results' / 'hetero_screen_seeds.json').exists() else {}
    arms = ('global', 'het_gp', 'regional')
    rows, wins = [], {m: dict.fromkeys(arms, 0) for m in ('ece', 'nll', 'crps')}
    order = sorted(R, key=lambda d: -(seeds.get(d, {}).get('mean') or R[d]['het_ratio']))
    for d in order:
        c = list(R[d]['seeds'].values())
        v = {m: {a: np.array([x[a][m] for x in c]) for a in arms} for m in wins}
        for m in wins:
            wins[m][min(arms, key=lambda a: v[m][a].mean())] += 1
        best = min(arms, key=lambda a: v['ece'][a].mean())
        cell = []
        for a in arms:
            s = f"{v['ece'][a].mean():.4f}"
            if a != 'global' and pval(v['ece'][a], v['ece']['global']) < 0.05:
                s += r'$^{\uparrow}$' if v['ece'][a].mean() < v['ece']['global'].mean() else r'$^{\downarrow}$'
            cell.append(rf'\textbf{{{s}}}' if a == best else s)
        h = seeds.get(d)
        het = f"{h['mean']:.1f}$\\pm${h['sd']:.1f}" if h else f"{R[d]['het_ratio']:.2f}"
        n = c[0]['n']['train']
        rows.append(f"{tex(d.replace('synthetic_heteroscedastic', 'synth_het'))} & "
                    f"{ORIGIN.get(d, 'measured')} & {n} & {het} & " + " & ".join(cell) + r" \\")
    rows.append(r'\midrule')
    for m, lab in (('ece', r'$\ece$'), ('nll', 'NLL'), ('crps', 'CRPS')):
        w = wins[m]; top = max(w.values())
        rows.append(rf"\multicolumn{{4}}{{l}}{{\emph{{best-arm count}} ({lab})}} & "
                    + " & ".join(rf"\textbf{{{w[a]}}}" if w[a] == top else str(w[a]) for a in arms)
                    + r" \\")
    write('noise', "\n".join(rows) + "\n")


def nsweep():
    R = L('adversarial_n.json')
    ds = list(R)
    ns = sorted({int(n) for d in R.values() for n in d})
    rows = []
    for n in ns:
        cells, gaps = [], []
        for d in ds:
            if str(n) not in R[d]:
                cells.append(''); continue
            g = [c['regional_ece'] - c['global_ece'] for c in R[d][str(n)].values()]
            gaps.append(np.mean(g))
            cells.append(f"${np.mean(g):+.4f}\\pm{np.std(g, ddof=1) / np.sqrt(len(g)):.4f}$")
        rows.append(f"{n} & " + " & ".join(cells) + f" & ${np.mean(gaps):+.4f}$ \\\\")
    write('nsweep', "\n".join(rows) + "\n")


def shrinkage():
    R = L('shrinkage.json')
    arms = ('regional', 'shrunk', 'eb')
    rows = []
    for n in sorted({int(n) for d in R.values() for n in d}):
        ds = [d for d in R if str(n) in R[d]]
        r = {a: [R[d][str(n)][k][a]['crps'] / R[d][str(n)][k]['global']['crps']
                 for d in ds for k in R[d][str(n)]] for a in arms}
        m = {a: np.mean(r[a]) for a in arms}; best = min(m, key=m.get)
        lam = np.mean([R[d][str(n)][k]['shrunk_cfg'][2] for d in ds for k in R[d][str(n)]])
        cell = [f"{m[a]:.4f}$\\pm${np.std(r[a], ddof=1) / np.sqrt(len(r[a])):.4f}" for a in arms]
        cell = [rf"\textbf{{{c}}}" if a == best else c for a, c in zip(arms, cell)]
        rows.append(f"{n} & {len(ds)} & " + " & ".join(cell) + f" & {lam:.2f} \\\\")
    write('shrinkage', "\n".join(rows) + "\n")


def screen():
    R = L('hetero_screen_seeds.json')
    grp = lambda d, g: 'constructed' if g == 'synthetic' or d == 'synthetic_heteroscedastic' \
        else ORIGIN.get(d, 'measured')
    rows = [f"{grp(d, v['group'])} & {tex(d)} & {v['mean']:.2f} & {v['sd']:.2f} & "
            f"{v['min']:.2f}--{v['max']:.2f} \\\\"
            for d, v in sorted(R.items(), key=lambda kv: -kv[1]['mean'])]
    write('screen', "\n".join(rows) + "\n")


if __name__ == '__main__':
    orthogonality()
    noise()
    nsweep()
    shrinkage()
    screen()
