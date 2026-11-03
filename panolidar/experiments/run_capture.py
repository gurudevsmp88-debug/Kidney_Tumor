"""Capture-effect experiments: exposure change, sensor noise, moving objects. Writes results/capture.json
and paper/tables/capture.tex. Note: the precision model is calibrated with the L1 cost; ZNCC reuses it."""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from panolidar.benchmark import calibrate, evaluate_capture, zncc_config  # noqa: E402
from panolidar.calibration import PrecisionModel  # noqa: E402
from panolidar.cli import load_config  # noqa: E402

SETTINGS = {
    'clean': {},
    'exposure': {'gain': 0.6, 'bias': 0.15},
    'noise': {'noise': 0.03},
    'moving cars': {'moving': True},
    'all': {'gain': 0.6, 'bias': 0.15, 'noise': 0.03, 'moving': True},
}
NAMES = {'fuse_confidence_l1': 'Confidence $\\sigma_s$, L1', 'fuse_confidence_zncc': 'Confidence $\\sigma_s$, ZNCC',
         'ours_l1': 'Ours, L1', 'ours_zncc_no_gate': 'Ours, ZNCC, w/o outlier gate', 'ours_zncc': 'Ours, ZNCC'}


def write_table(res):
    settings = list(res)
    lines = [r'\begin{tabular}{l ' + 'c' * len(settings) + '}', r'\toprule',
             'Method & ' + ' & '.join(settings) + r' \\', r'\midrule']
    best = {s: min(res[s][m]['all']['median_rel'] for m in NAMES) for s in settings}
    for m, label in NAMES.items():
        cells = []
        for s in settings:
            v = res[s][m]['all']['median_rel']
            cell = f'{100 * v:.2f}'
            cells.append(f'\\textbf{{{cell}}}' if v == best[s] else cell)
        lines.append(label + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    with open(os.path.join(ROOT, 'paper', 'tables', 'capture.tex'), 'w') as f:
        f.write('\n'.join(lines))


def main():
    torch.set_num_threads(2)
    cfg = load_config()
    cal = json.load(open(os.path.join(ROOT, 'results', 'calibration.json')))
    precision = PrecisionModel.from_config({k: cal[k] for k in ('edges', 'scales', 'kappa')})
    precision.baselines = cal['fitted_baselines']
    seeds = list(range(0, 8))
    t0 = time.time()
    precision_zncc = calibrate(zncc_config(cfg), list(range(1000, 1006)), 256, 512, 4.0)
    with open(os.path.join(ROOT, 'results', 'calibration_zncc.json'), 'w') as f:
        json.dump({**precision_zncc.to_dict(), 'fitted_baselines': precision_zncc.baselines}, f, indent=2)
    print('zncc calibration', precision_zncc.to_dict(), f'({time.time() - t0:.0f}s)', flush=True)
    res = {}
    for label, kw in SETTINGS.items():
        res[label] = evaluate_capture(cfg, precision, seeds, 256, 512, precision_zncc=precision_zncc, **kw)
        print(label, ' | '.join(f"{m} {100 * r['all']['median_rel']:.2f}/{r['all']['calib_1s']:.2f}"
                                for m, r in res[label].items()), f'({time.time() - t0:.0f}s)', flush=True)
    with open(os.path.join(ROOT, 'results', 'capture.json'), 'w') as f:
        json.dump(res, f, indent=2)
    write_table(res)
    print('all done')


if __name__ == '__main__':
    main()
