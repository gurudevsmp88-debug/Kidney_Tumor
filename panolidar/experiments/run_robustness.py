"""Pose-noise and mixed-baseline experiments (where geometry-blind uncertainties cannot know the geometry)."""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from panolidar.benchmark import calibrate_baselines_with_pose_noise, evaluate_robustness  # noqa: E402
from panolidar.calibration import PrecisionModel  # noqa: E402
from panolidar.cli import load_config  # noqa: E402


def main():
    cfg = load_config()
    cal = json.load(open(os.path.join(ROOT, 'results', 'calibration.json')))
    precision = PrecisionModel.from_config({k: cal[k] for k in ('edges', 'scales', 'kappa')})
    precision.baselines = cal['fitted_baselines']
    H, W = 256, 512
    seeds = list(range(0, 8))
    t0 = time.time()
    out = {}
    for label, (rot_deg, trans) in {'pose 0.1deg/2cm': (0.1, 0.02), 'pose 0.3deg/5cm': (0.3, 0.05),
                                    'pose 1deg/10cm': (1.0, 0.10)}.items():
        rot = rot_deg * 3.14159265 / 180
        out[label] = evaluate_robustness(cfg, precision, seeds, H, W, (4.0,), sigma_rot=rot, sigma_trans=trans)
        refit = calibrate_baselines_with_pose_noise(cfg, list(range(1000, 1006)), H, W, 4.0, rot, trans)
        refit_res = evaluate_robustness(cfg, precision, seeds, H, W, (4.0,), sigma_rot=rot, sigma_trans=trans,
                                        baseline_sigmas=refit)
        out[label]['fuse_constant_refit'] = refit_res['fuse_constant']
        out[label]['fuse_confidence_refit'] = refit_res['fuse_confidence']
        print(label, f'({time.time() - t0:.0f}s)', flush=True)
    out['partners +2m & -8m'] = evaluate_robustness(cfg, precision, seeds, H, W, (2.0, -8.0))
    print('mixed baselines', f'({time.time() - t0:.0f}s)', flush=True)
    with open(os.path.join(ROOT, 'results', 'robustness.json'), 'w') as f:
        json.dump(out, f, indent=2)
    for label, res in out.items():
        print(label)
        for k, m in res.items():
            a = m['all']
            print(f"   {k:18s} med {100 * a['median_rel']:.2f}  w20 {100 * a['within_0.2m']:.1f}  "
                  f"cal {a['calib_1s']:.2f}  nll {a['nll']:.2f}  ause {a['ause']:.3f}")
    write_table()
    print('all done')


def write_table(path_json=None, path_tex=None):
    path_json = path_json or os.path.join(ROOT, 'results', 'robustness.json')
    path_tex = path_tex or os.path.join(ROOT, 'paper', 'tables', 'robustness.tex')
    res = json.load(open(path_json))
    settings = [k for k in res if k.startswith('pose')]
    rows = [('fuse_constant', 'Constant $\\sigma_s$'), ('fuse_constant_refit', 'Constant $\\sigma_s$, refit'),
            ('fuse_confidence', 'Confidence $\\sigma_s$'), ('fuse_confidence_refit', 'Confidence $\\sigma_s$, refit'),
            ('ours_no_pose_term', 'Ours w/o pose term'), ('ours', 'Ours')]
    labels = [s.replace('pose ', '').replace('deg', '$^\\circ$') for s in settings]
    lines = [r'\begin{tabular}{l ' + 'ccc' * len(settings) + '}', r'\toprule',
             ' & ' + ' & '.join(f'\\multicolumn{{3}}{{c}}{{{l}}}' for l in labels) + r' \\',
             ''.join(f'\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}' for i in range(len(settings))),
             'Method' + ' & Med. & Cal. & NLL' * len(settings) + r' \\', r'\midrule']
    best = {}
    for s in settings:
        vals = {k: res[s][k]['all'] for k, _ in rows}
        best[s] = (min(v['median_rel'] for v in vals.values()), min(abs(v['calib_1s'] - 0.683) for v in vals.values()),
                   min(v['nll'] for v in vals.values()))
    for key, label in rows:
        cells = []
        for s in settings:
            a = res[s][key]['all']
            b = best[s]
            med = f"{100 * a['median_rel']:.2f}"
            cal = f"{a['calib_1s']:.2f}"
            nll = f"{a['nll']:.2f}"
            cells += [f'\\textbf{{{med}}}' if a['median_rel'] == b[0] else med,
                      f'\\textbf{{{cal}}}' if abs(a['calib_1s'] - 0.683) == b[1] else cal,
                      f'\\textbf{{{nll}}}' if a['nll'] == b[2] else nll]
        lines.append(label + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    with open(path_tex, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
