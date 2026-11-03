"""Calibrating the stereo precision from a noisy city teacher instead of ground truth.
Writes results/teacher_calibration.json and paper/tables/teacher_calibration.tex."""
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from panolidar.benchmark import calibrate_from_teacher, evaluate  # noqa: E402
from panolidar.calibration import PrecisionModel  # noqa: E402
from panolidar.cli import load_config  # noqa: E402

H, W, B = 256, 512, 4.0
NAMES = {'gt': 'Ground truth (oracle)', 'naive': 'Teacher, naive', 'min_views': 'Teacher, $\\geq$2 views',
         'covariance': 'Teacher, covariance-aware (ours)'}


def main():
    torch.set_num_threads(2)
    cfg = load_config()
    cal = json.load(open(os.path.join(ROOT, 'results', 'calibration.json')))
    models = {'gt': PrecisionModel.from_config({k: cal[k] for k in ('edges', 'scales', 'kappa')})}
    t0 = time.time()
    # teacher from 0.5 px matching on 2048-px-wide panoramas
    sigma_t = 0.5 * 2 * math.pi / 2048
    for mode in ('naive', 'min_views', 'covariance'):
        models[mode] = calibrate_from_teacher(cfg, list(range(1000, 1006)), H, W, B, sigma_t, mode=mode)
        print(mode, models[mode].to_dict(), f'({time.time() - t0:.0f}s)', flush=True)
    res = {}
    for name, pm in models.items():
        r = evaluate(cfg, pm, list(range(0, 10)), H, W, B, methods=['ours'])
        res[name] = {'precision': pm.to_dict(), 'metrics': r['ours']['all']}
        a = r['ours']['all']
        print(f"{name:11s} med {100 * a['median_rel']:.2f} cal {a['calib_1s']:.2f} nll {a['nll']:.2f}", flush=True)
    with open(os.path.join(ROOT, 'results', 'teacher_calibration.json'), 'w') as f:
        json.dump(res, f, indent=2)
    lines = [r'\begin{tabular}{l c ccc}', r'\toprule',
             r'Calibration data & $\kappa$ & Med.\ rel.\ (\%) & Cal.\ $1\sigma$ & NLL \\', r'\midrule']
    for name, label in NAMES.items():
        a, p = res[name]['metrics'], res[name]['precision']
        lines.append(f"{label} & {p['kappa']:.2f} & {100 * a['median_rel']:.2f} & {a['calib_1s']:.2f} & {a['nll']:.2f} \\\\")
        if name == 'gt':
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}']
    with open(os.path.join(ROOT, 'paper', 'tables', 'teacher_calibration.tex'), 'w') as f:
        f.write('\n'.join(lines))
    print('all done')


if __name__ == '__main__':
    main()
