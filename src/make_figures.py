"""Generate the paper's figures as PDFs from stored results."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
from pathlib import Path

P = Path(__file__).parent.parent
FIG = P / 'paper' / 'figs'; FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 9, 'legend.fontsize': 7.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#8a8984', 'axes.labelcolor': '#0b0b0b',
    'xtick.color': '#4a4944', 'ytick.color': '#4a4944',
    'grid.color': '#e1e0d9', 'grid.linewidth': 0.6,
    'figure.dpi': 200, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42,
})
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6']


def xticks_n(ax, ns):
    ax.xaxis.set_major_locator(FixedLocator(ns))
    ax.xaxis.set_major_formatter(FixedFormatter([str(n) for n in ns]))
    ax.xaxis.set_minor_locator(NullLocator())


def fig_corruption():
    R = json.load(open(P / 'results/inference/corruption.json'))
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for i, (ds, ns) in enumerate(sorted(R.items())):
        xs = sorted(int(n) for n in ns)
        cells = [[100 * ns[str(n)][s]['repo_default']['std_relerr_median'] for s in ns[str(n)]]
                 for n in xs]
        ys = np.array([np.mean(c) for c in cells])
        lo = ys - np.array([min(c) for c in cells]); hi = np.array([max(c) for c in cells]) - ys
        label = 'synthetic\\_het.' if ds.startswith('synthetic') else ds
        ax.errorbar(xs, ys, yerr=[lo, hi], marker='o', ms=3, lw=1.3, capsize=1.5, elinewidth=.7,
                    color=C[i % len(C)], label=label.replace('\\_', '_'),
                    zorder=6 if ds == 'concrete' else 3)
    ax.axvline(800, color='#d03b3b', lw=1.0, ls='--', zorder=1)
    ax.text(740, 0.4, 'default\nmax_cholesky_size', color='#d03b3b', fontsize=6.5,
            ha='right', va='center')
    ax.set_xscale('log'); ax.set_yscale('symlog', linthresh=0.1, linscale=0.6)
    ax.set_ylim(-0.02, 1500)
    ax.yaxis.set_major_locator(FixedLocator([0, 0.1, 1, 10, 100, 1000]))
    ax.yaxis.set_major_formatter(FixedFormatter(['0', '0.1', '1', '10', '100', '1000']))
    ax.yaxis.set_minor_locator(NullLocator())
    xticks_n(ax, [200, 400, 800, 1500, 4000])
    ax.set_xlabel('training set size $n$'); ax.set_ylabel('median error in $\\sigma$ (%)')
    ax.grid(axis='y', alpha=.7); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.0, 1.0), handlelength=1.4)
    fig.savefig(FIG / 'corruption.pdf'); plt.close(fig)


def fig_orthogonality():
    R = json.load(open(P / 'results/structured/orthogonality.json'))
    arms = ['rff', 'nys', 'nys_hi', 'gp']
    labels = ['RFF', 'Nyström-500', 'Nyström-1500', 'exact GP']
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
    for ax, ds in zip(axes, ['hetero_extreme', 'combined']):
        c = list(R[ds].values())
        regs = sorted(c[0].keys(), key=int)
        for j, a in enumerate(arms):
            x = np.arange(len(regs)) + (j - 1.5) * 0.13
            v = [[s[r]['ece'][a] for s in c] for r in regs]
            ys = np.array([np.mean(t) for t in v])
            err = [ys - [min(t) for t in v], [max(t) for t in v] - ys]
            ax.errorbar(x, ys, yerr=err, fmt='os^D'[j], ms=4, capsize=1.5, elinewidth=.8,
                        color=C[j], label=labels[j])
        ax.set_xticks(range(len(regs))); ax.set_xticklabels([f'region {r}' for r in regs])
        ax.set_xlim(-0.5, len(regs) - 0.5)
        ax.set_title(ds, fontsize=9, family='monospace')
        ax.set_ylabel('per-region ECE'); ax.grid(axis='y', alpha=.7); ax.set_axisbelow(True)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, handlelength=1.2, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(w_pad=2.0, rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / 'orthogonality.pdf'); plt.close(fig)


def fig_nsweep():
    R = json.load(open(P / 'results/adversarial_n.json'))
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ns = sorted({int(n) for d in R.values() for n in d})
    for i, (ds, d) in enumerate(sorted(R.items())):
        xs = [n for n in ns if str(n) in d]
        g = [[c['regional_ece'] - c['global_ece'] for c in d[str(n)].values()] for n in xs]
        ax.errorbar(xs, [np.mean(v) for v in g],
                    yerr=[np.std(v, ddof=1) / np.sqrt(len(v)) for v in g],
                    marker='o', ms=2.5, lw=1.0, capsize=1.5, elinewidth=.7,
                    color=C[i % len(C)], alpha=.85, label=ds)
    mean = [np.mean([np.mean([c['regional_ece'] - c['global_ece']
                              for c in d[str(n)].values()])
                     for d in R.values() if str(n) in d]) for n in ns]
    ax.plot(ns, mean, lw=2.2, color='#0b0b0b', marker='s', ms=3.5,
            label='mean of datasets present', zorder=5)
    ax.axhline(0, color='#8a8984', lw=0.9)
    ax.set_xscale('log'); xticks_n(ax, ns)
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, max(hi, 0.012))
    ax.text(1500, ax.get_ylim()[1], 'global better', fontsize=6.5, color='#4a4944', va='top')
    ax.text(310, ax.get_ylim()[0], 'regional better', fontsize=6.5, color='#4a4944', va='bottom')
    ax.set_xlabel('training set size $n$'); ax.set_ylabel('ECE gap (regional $-$ global)')
    ax.grid(axis='y', alpha=.7); ax.set_axisbelow(True)
    ax.legend(frameon=False, handlelength=1.4, loc='lower center', bbox_to_anchor=(0.5, 1.0),
              ncol=3, fontsize=6.5, columnspacing=1.0)
    fig.savefig(FIG / 'nsweep.pdf'); plt.close(fig)

for f in (fig_corruption, fig_orthogonality, fig_nsweep):
    f(); print('wrote', f.__name__)
print('figures ->', FIG)
