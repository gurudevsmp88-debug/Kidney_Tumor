"""Reproduce every number, table and figure of the paper draft (CPU, synthetic benchmark).

Usage: python experiments/run_paper_experiments.py [--quick]
Writes results/*.json, paper/tables/*.tex and paper/figures/*.pdf.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from panolidar.benchmark import calibrate, evaluate  # noqa: E402
from panolidar.cli import load_config  # noqa: E402

RESULTS = os.path.join(ROOT, 'results')
TABLES = os.path.join(ROOT, 'paper', 'tables')
FIGURES = os.path.join(ROOT, 'paper', 'figures')
BINS = ['0-10deg', '10-30deg', '30-60deg', '60-90deg']
NAMES = {
    'prior': 'Prior only', 'prior_aligned': 'Prior + scale alignment', 'stereo': 'Stereo only',
    'fuse_constant': 'Fusion, constant $\\sigma_s$', 'fuse_confidence': 'Fusion, confidence $\\sigma_s$',
    'select_ours': 'Selection (min.\\ $\\sigma$)', 'ours_closed_form': 'Ours, closed form ($\\sigma_{px}{=}0.5$)',
    'ours': 'Ours (calibrated)',
}
ABLATIONS = {
    'ours': 'Full model', 'ours_closed_form': 'Uncalibrated closed form', 'ours_sigma_at_stereo': '$\\sigma$ at stereo depth',
    'ours_quartic_gate': 'Quartic gate instead of mixture', 'ours_no_align': 'No scale alignment',
}


def save(name, obj):
    with open(os.path.join(RESULTS, name), 'w') as f:
        json.dump(obj, f, indent=2)


def pct(x):
    return f'{100 * x:.2f}'


def bold_best(values, lower=True):
    nums = [float(v) for v in values]
    best = min(nums) if lower else max(nums)
    return [f'\\textbf{{{v}}}' if n == best else v for v, n in zip(values, nums)]


def main_table(res):
    rows = ['prior', 'prior_aligned', 'stereo', 'fuse_constant', 'fuse_confidence', 'select_ours',
            'ours_closed_form', 'ours']
    cols = []
    for r in rows:
        a = res[r]
        cols.append([pct(a[b]['median_rel']) for b in BINS] + [pct(a['all']['median_rel']),
                     pct(a['all']['within_0.2m']), f"{a['all']['calib_1s']:.2f}", f"{a['all']['nll']:.2f}",
                     f"{a['all']['ause']:.3f}"])
    # bold best per column (calibration: closest to 0.683)
    for j in range(len(cols[0])):
        vals = [c[j] for c in cols]
        if j == 6:
            best = min(vals, key=lambda v: abs(float(v) - 0.683))
            vals = [f'\\textbf{{{v}}}' if v == best else v for v in vals]
        elif j == 5:
            vals = bold_best(vals, lower=False)
        else:
            vals = bold_best(vals)
        for c, v in zip(cols, vals):
            c[j] = v
    lines = [r'\begin{tabular}{l ccccc c ccc}', r'\toprule',
             r' & \multicolumn{5}{c}{Median rel.\ error (\%) $\downarrow$, by angle to baseline} & $<$20\,cm (\%) $\uparrow$ & '
             r'Cal.\ $1\sigma$ & NLL $\downarrow$ & AUSE $\downarrow$ \\',
             r'\cmidrule(lr){2-6}', r'Method & 0--10$^\circ$ & 10--30$^\circ$ & 30--60$^\circ$ & 60--90$^\circ$ & all & all & (0.68) & all & all \\',
             r'\midrule']
    for r, c in zip(rows, cols):
        if r == 'fuse_constant':
            lines.append(r'\midrule')
        lines.append(NAMES[r] + ' & ' + ' & '.join(c) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def ablation_table(res):
    lines = [r'\begin{tabular}{l ccc}', r'\toprule',
             r'Variant & Med.\ rel.\ (\%) $\downarrow$ & NLL $\downarrow$ & Cal.\ $1\sigma$ \\', r'\midrule']
    for k, label in ABLATIONS.items():
        a = res[k]['all']
        lines.append(f"{label} & {pct(a['median_rel'])} & {a['nll']:.2f} & {a['calib_1s']:.2f} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def transfer_table(transfer):
    methods = ['prior_aligned', 'stereo', 'fuse_constant', 'fuse_confidence', 'ours']
    settings = list(transfer.keys())
    lines = [r'\begin{tabular}{l ' + 'cc' * len(settings) + '}', r'\toprule',
             ' & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{s}}}' for s in settings) + r' \\',
             ''.join(f'\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(len(settings))),
             'Method' + ' & Med.\\,(\\%) & NLL' * len(settings) + r' \\', r'\midrule']
    for m in methods:
        cells = []
        for s in settings:
            a = transfer[s][m]['all']
            cells += [pct(a['median_rel']), f"{a['nll']:.2f}"]
        lines.append(NAMES[m] + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def figures(res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif'})
    x = range(len(BINS))
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    for key, style in [('prior_aligned', ('#7f7f7f', 's', '--')), ('stereo', ('#1f77b4', 'o', ':')),
                       ('fuse_constant', ('#ff7f0e', '^', '-.')), ('ours', ('#d62728', 'D', '-'))]:
        ys = [100 * res[key][b]['median_rel'] for b in BINS]
        ax.plot(x, ys, color=style[0], marker=style[1], linestyle=style[2], label=NAMES[key].replace('$', '').replace('\\sigma_s', 'σ'), lw=1.2, ms=3.5)
    ax.set_yscale('log')
    from matplotlib.ticker import FuncFormatter, LogLocator
    ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.set_xticks(list(x))
    ax.set_xticklabels(['0–10°', '10–30°', '30–60°', '60–90°'])
    ax.set_xlabel('angle between ray and baseline')
    ax.set_ylabel('median rel. error (%)')
    ax.grid(alpha=0.3, which='both')
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'error_vs_angle.pdf'))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='tiny run for smoke testing')
    args = parser.parse_args()
    for d in (RESULTS, TABLES, FIGURES):
        os.makedirs(d, exist_ok=True)
    cfg = load_config()
    H, W, B = (64, 128, 4.0) if args.quick else (256, 512, 4.0)
    cal_seeds = list(range(1000, 1002 if args.quick else 1006))
    test_seeds = list(range(0, 2 if args.quick else 10))
    t0 = time.time()

    precision = calibrate(cfg, cal_seeds, H, W, B)
    save('calibration.json', {'resolution': [H, W], 'baseline': B, 'seeds': cal_seeds, **precision.to_dict(),
                              'fitted_baselines': precision.baselines})
    print(f'calibration {precision.to_dict()} ({time.time() - t0:.0f}s)', flush=True)

    main_res = evaluate(cfg, precision, test_seeds, H, W, B)
    save('main.json', main_res)
    with open(os.path.join(TABLES, 'main.tex'), 'w') as f:
        f.write(main_table(main_res))
    with open(os.path.join(TABLES, 'ablation.tex'), 'w') as f:
        f.write(ablation_table(main_res))
    figures(main_res)
    print(f'main done ({time.time() - t0:.0f}s)', flush=True)

    transfer_methods = ['ours', 'fuse_constant', 'fuse_confidence']
    tseeds = test_seeds[:2] if args.quick else test_seeds[:5]
    transfer = {}
    for label, (h, w, b) in {'$b{=}2$\\,m': (H, W, 2.0), '$b{=}8$\\,m': (H, W, 8.0),
                             f'{H // 2}$\\times${W // 2}': (H // 2, W // 2, B),
                             f'{2 * H}$\\times${2 * W}': (2 * H, 2 * W, B)}.items():
        seeds = tseeds[:3] if h > H else tseeds
        transfer[label] = evaluate(cfg, precision, seeds, h, w, b, methods=transfer_methods)
        print(f'transfer {label} done ({time.time() - t0:.0f}s)', flush=True)
    save('transfer.json', transfer)
    with open(os.path.join(TABLES, 'transfer.tex'), 'w') as f:
        f.write(transfer_table(transfer))
    print(f'all done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
