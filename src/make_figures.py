"""Generate the paper's figures as PDFs from stored results."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

P = Path(__file__).parent.parent
FIG = P / 'paper' / 'figs'; FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 9, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': '#0b0b0b',
    'xtick.color': '#898781', 'ytick.color': '#898781',
    'grid.color': '#e1e0d9', 'grid.linewidth': 0.6,
    'figure.dpi': 200, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6']


def fig_corruption():
    R = json.load(open(P / 'results/inference/corruption.json'))
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for i, (ds, ns) in enumerate(sorted(R.items())):
        xs = sorted(int(n) for n in ns)
        ys = [100 * np.mean([ns[str(n)][s]['repo_default']['std_relerr_median']
                             for s in ns[str(n)]]) for n in xs]
        ax.plot(xs, ys, marker='o', ms=3, lw=1.4, color=C[i % len(C)],
                label=ds.replace('synthetic_heteroscedastic', 'synthetic')[:12])
    ax.axvline(800, color='#d03b3b', lw=1.0, ls='--')
    ax.text(830, ax.get_ylim()[1] * 0.55, 'max\\_cholesky\\_size = 800',
            color='#d03b3b', fontsize=7, rotation=90, va='center')
    ax.set_xscale('log'); ax.set_yscale('symlog', linthresh=1)
    ax.set_xlabel('training set size'); ax.set_ylabel('median error in $\\sigma$ (\\%)')
    ax.grid(axis='y', alpha=.7); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='upper left', handlelength=1.4)
    fig.savefig(FIG / 'corruption.pdf'); plt.close(fig)


def fig_orthogonality():
    R = json.load(open(P / 'results/structured/orthogonality.json'))
    arms = ['rff', 'nys', 'nys_hi', 'gp']
    labels = ['RFF', 'Nys-500', 'Nys-1500', 'exact GP']
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), sharey=False)
    for ax, ds in zip(axes, ['hetero_extreme', 'combined']):
        c = list(R[ds].values())
        regs = sorted(c[0].keys(), key=int)
        x = np.arange(len(regs))
        for j, a in enumerate(arms):
            ys = [np.mean([s[r]['ece'][a] for s in c]) for r in regs]
            ax.plot(x, ys, marker='os^D'[j], ms=4, lw=1.2, color=C[j], label=labels[j])
        ax.set_xticks(x); ax.set_xticklabels([f'region {r}' for r in regs])
        ax.set_title(ds.replace('_', '\\_'), fontsize=9)
        ax.set_ylabel('ECE'); ax.grid(axis='y', alpha=.7); ax.set_axisbelow(True)
    axes[0].legend(frameon=False, handlelength=1.4, loc='lower left')
    fig.savefig(FIG / 'orthogonality.pdf'); plt.close(fig)


def fig_nsweep():
    R = json.load(open(P / 'results/adversarial_n.json'))
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ns = sorted({int(n) for d in R.values() for n in d})
    for i, (ds, d) in enumerate(sorted(R.items())):
        xs = [n for n in ns if str(n) in d]
        ys = [np.mean([c['regional_ece'] - c['global_ece'] for c in d[str(n)].values()])
              for n in xs]
        ax.plot(xs, ys, marker='o', ms=3, lw=1.2, color=C[i % len(C)], alpha=.75,
                label=ds[:11])
    mean = [np.mean([np.mean([c['regional_ece'] - c['global_ece']
                              for c in d[str(n)].values()])
                     for d in R.values() if str(n) in d]) for n in ns]
    ax.plot(ns, mean, lw=2.2, color='#0b0b0b', label='mean', zorder=5)
    ax.axhline(0, color='#c3c2b7', lw=1.0)
    ax.set_xscale('log')
    ax.set_xlabel('training set size'); ax.set_ylabel('ECE gap (regional $-$ global)')
    ax.grid(axis='y', alpha=.7); ax.set_axisbelow(True)
    ax.legend(frameon=False, handlelength=1.4, ncol=2, fontsize=7)
    fig.savefig(FIG / 'nsweep.pdf'); plt.close(fig)


for f in (fig_corruption, fig_orthogonality, fig_nsweep):
    f(); print('wrote', f.__name__)
print('figures ->', FIG)
