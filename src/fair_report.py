"""Aggregate the fair benchmark: mean +/- std over seeds, plus paired contrasts.

Because every method sees an identical split within a seed, per-seed differences
are paired and can be reported directly.
"""
import json, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RAW = PROJECT_ROOT / 'results' / 'fair' / 'raw_results.json'

ORDER = ['exact_gp', 'rff_uniform', 'nystrom_uniform',
         'aurora_gp', 'aurora_gp_shuffled',
         'aurora_approx', 'aurora_approx_shuffled']
LABEL = {'exact_gp': 'Exact GP', 'rff_uniform': 'Uniform RFF',
         'nystrom_uniform': 'Uniform Nystrom', 'aurora_gp': 'AURORA (exact-GP tier)',
         'aurora_gp_shuffled': '  \\_ routing shuffled (control)',
         'aurora_approx': 'AURORA (all-approx tiers)',
         'aurora_approx_shuffled': '  \\_ routing shuffled (control)'}


def collect(raw):
    """-> vals[dataset][method][metric] = array over seeds"""
    out = {}
    for ds, seeds in raw.items():
        out[ds] = {}
        for m in ORDER:
            acc = {k: [] for k in ('rmse', 'nll', 'ece', 'r2', 'train_time')}
            for s, cell in sorted(seeds.items()):
                if 'error' in cell or m not in cell.get('methods', {}):
                    continue
                v = cell['methods'][m]
                for k in acc:
                    if k in v:
                        acc[k].append(v[k])
            out[ds][m] = {k: np.array(v) for k, v in acc.items() if v}
    return out


def fmt(a, p=4):
    return f"{a.mean():.{p}f}+/-{a.std(ddof=1):.{p}f}" if len(a) > 1 else f"{a.mean():.{p}f}"


def main():
    raw = json.load(open(RAW))
    V = collect(raw)
    lines = []
    W = lines.append

    W("=" * 104)
    W("FAIR RE-RUN — mean +/- std over seeds, identical split per seed for every method")
    W("=" * 104)
    for ds in V:
        n = [c for c in raw[ds].values() if 'error' not in c]
        if not n:
            continue
        W(f"\n### {ds}   (train {n[0]['n_train']} / val {n[0]['n_val']} / test {n[0]['n_test']}, "
          f"{len(n)} seeds)")
        W(f"{'method':34s} {'RMSE':>16s} {'NLL':>16s} {'ECE':>16s} {'train s':>9s}")
        W("-" * 96)
        for m in ORDER:
            d = V[ds].get(m)
            if not d or 'rmse' not in d:
                continue
            t = f"{d['train_time'].mean():9.1f}" if 'train_time' in d else " " * 9
            W(f"{LABEL[m]:34s} {fmt(d['rmse']):>16s} {fmt(d['nll'],3):>16s} {fmt(d['ece']):>16s} {t}")

    # ---- the two questions the paper actually has to answer ----
    W("\n" + "=" * 104)
    W("Q1  Does region-aware routing beat a uniform approximation of the same family?")
    W("    paired per-seed ECE difference, AURORA(all-approx) - Uniform Nystrom  (negative = AURORA better)")
    W("=" * 104)
    W(f"{'dataset':28s} {'AURORA ECE':>16s} {'Uniform ECE':>16s} {'paired diff':>18s} {'AURORA wins':>12s}")
    W("-" * 96)
    agg = []
    for ds in V:
        a, b = V[ds].get('aurora_approx', {}), V[ds].get('nystrom_uniform', {})
        if 'ece' not in a or 'ece' not in b:
            continue
        d = a['ece'] - b['ece']
        agg.append(d.mean())
        W(f"{ds:28s} {fmt(a['ece']):>16s} {fmt(b['ece']):>16s} {fmt(d):>18s} {f'{(d<0).sum()}/{len(d)}':>12s}")
    if agg:
        W(f"\n    mean paired ECE difference across datasets: {np.mean(agg):+.4f}"
          f"   ({'AURORA better' if np.mean(agg) < 0 else 'AURORA WORSE'})")

    W("\n" + "=" * 104)
    W("Q2  Does the importance signal carry information? (same models, routing randomly permuted)")
    W("    paired per-seed ECE difference, AURORA - shuffled control  (negative = real signal)")
    W("=" * 104)
    W(f"{'dataset':28s} {'variant':22s} {'AURORA':>14s} {'shuffled':>14s} {'paired diff':>18s} {'wins':>8s}")
    W("-" * 96)
    agg2 = []
    for ds in V:
        for base, ctrl, nm in [('aurora_gp', 'aurora_gp_shuffled', 'exact-GP tier'),
                               ('aurora_approx', 'aurora_approx_shuffled', 'all-approx tiers')]:
            a, b = V[ds].get(base, {}), V[ds].get(ctrl, {})
            if 'ece' not in a or 'ece' not in b:
                continue
            d = a['ece'] - b['ece']
            if nm == 'all-approx tiers':
                agg2.append(d.mean())
            W(f"{ds:28s} {nm:22s} {a['ece'].mean():14.4f} {b['ece'].mean():14.4f} "
              f"{fmt(d):>18s} {f'{(d<0).sum()}/{len(d)}':>8s}")
    if agg2:
        W(f"\n    mean paired ECE difference vs shuffled control: {np.mean(agg2):+.4f}")

    W("\n" + "=" * 104)
    W("Q3  Is AURORA ever on the efficiency frontier? (vs the exact GP it contains)")
    W("=" * 104)
    W(f"{'dataset':28s} {'RMSE  GP -> AURORA':>26s} {'NLL  GP -> AURORA':>26s} {'train s  GP -> AURORA':>26s}")
    W("-" * 108)
    for ds in V:
        g, a = V[ds].get('exact_gp', {}), V[ds].get('aurora_gp', {})
        if 'rmse' not in g or 'rmse' not in a:
            continue
        rm = f"{g['rmse'].mean():.4f} -> {a['rmse'].mean():.4f}"
        nl = f"{g['nll'].mean():.3f} -> {a['nll'].mean():.3f}"
        tt = f"{g['train_time'].mean():.1f} -> {a['train_time'].mean():.1f}"
        W(f"{ds:28s} {rm:>26s} {nl:>26s} {tt:>26s}")

    txt = "\n".join(lines)
    print(txt)
    out = PROJECT_ROOT / 'results' / 'fair' / 'report.txt'
    out.write_text(txt)
    print(f"\nsaved -> {out}")


if __name__ == '__main__':
    main()
