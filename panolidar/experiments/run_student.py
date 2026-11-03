"""City-as-teacher student experiment (CPU, synthetic streets).

Trains the reference student with a realistic noisy teacher (points displaced by draws from their own
triangulation covariance) under three target weightings, and evaluates on held-out streets:
  variance   teacher covariance inside the Gaussian NLL (ours)
  uniform    all teacher targets weighted equally
  min_views  uniform, but only targets seen by >= 2 teacher views (common heuristic)
Writes results/student.json and paper/tables/student.tex.
"""
import json
import math
import os
import sys
import time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from panolidar.cli import load_config  # noqa: E402
from panolidar.experiments import make_training_sample, student_loss  # noqa: E402
from panolidar.model import PanoLidarNet  # noqa: E402

H, W = 64, 128
SIGMA_ANG = 0.5 * 2 * math.pi / W
TEACHER_SIGMA_ANG = 0.5 * 2 * math.pi / 2048  # teacher reconstructed from high-resolution panoramas
TRAIN_SEEDS = range(0, 100000, 7)
TEST_SEEDS = list(range(900000, 900012))


def sample(seed):
    return make_training_sample(seed, H, W, SIGMA_ANG, teacher_sigma_ang=TEACHER_SIGMA_ANG, teacher_noise=True)


def run_variant(args):
    weighting, steps, seed = args
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    cfg = load_config()
    model = PanoLidarNet(sigma_px=cfg['stereo']['sigma_px'], stereo_kwargs=cfg['train']['stereo'])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    seeds = list(TRAIN_SEEDS)
    t0 = time.time()
    for step in range(steps):
        s = sample(seeds[(seed * 997 + step) % len(seeds)])
        pose = lambda p: (p[0][None], p[1][None])
        out = model(s['image1'][None], s['image2'][None], pose(s['pose1']), pose(s['pose2']))
        loss, _ = student_loss(out, s['targets'], weighting)
        if not torch.isfinite(loss):
            continue
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0:
            print(f'[{weighting}] step {step} loss {loss.item():.3f} ({time.time() - t0:.0f}s)', flush=True)
    return weighting, evaluate(model)


@torch.no_grad()
def evaluate(model):
    model.eval()
    rel1, cal1, rel2, nll1 = [], [], [], []
    ratios = []
    for seed in TEST_SEEDS:
        s = sample(seed)
        pose = lambda p: (p[0][None], p[1][None])
        out = model(s['image1'][None], s['image2'][None], pose(s['pose1']), pose(s['pose2']))
        gt = s['gt_depth']
        m = torch.isfinite(gt)
        mu, sig = out['mu'][0], out['sigma'][0]
        e = mu - gt.log()
        rel1.append(((mu.exp() - gt).abs() / gt)[m].median().item())
        cal1.append((e[m].abs() < sig[m]).float().mean().item())
        nll1.append((torch.log(sig[m]) + 0.5 * (e[m] / sig[m]) ** 2 + 0.5 * math.log(2 * math.pi)).mean().item())
        d2 = s['clean_targets']['depth2']
        m2 = torch.isfinite(d2) & m
        if m2.sum() > 10:
            rel2.append(((out['layer2_mu'][0].exp() - d2).abs() / d2)[m2].median().item())
            ratios.append((mu.exp()[m2], d2[m2]))
    mean = lambda x: sum(x) / len(x)
    return {'layer1_median_rel': mean(rel1), 'layer1_calib_1s': mean(cal1), 'layer1_nll': mean(nll1),
            'layer2_median_rel': mean(rel2) if rel2 else float('nan')}


def amodal_heuristic():
    """Layer-2 baseline: predicted first surface times the median ratio d2/d1 measured on training data."""
    r = []
    for seed in list(TRAIN_SEEDS)[:30]:
        t = sample(seed)['clean_targets']
        m = torch.isfinite(t['depth1']) & torch.isfinite(t['depth2'])
        r.append((t['depth2'] / t['depth1'])[m])
    ratio = torch.cat(r).median().item()
    rel = []
    for seed in TEST_SEEDS:
        s = sample(seed)
        d1, d2 = s['gt_depth'], s['clean_targets']['depth2']
        m = torch.isfinite(d1) & torch.isfinite(d2)
        if m.sum() > 10:
            rel.append(((d1 * ratio - d2).abs() / d2)[m].median().item())
    return ratio, sum(rel) / len(rel)


def write_table(res, ratio, heur):
    names = {'uniform': 'Uniform targets', 'min_views': 'Uniform, $\\geq$2 teacher views',
             'variance': 'Covariance-weighted NLL (ours)'}
    lines = [r'\begin{tabular}{l ccc c}', r'\toprule',
             r'Teacher weighting & Med.\ rel.\ (\%) & Cal.\ $1\sigma$ & NLL & Amodal med.\ rel.\ (\%) \\', r'\midrule']
    for k in ('uniform', 'min_views', 'variance'):
        a = res[k]
        lines.append(f"{names[k]} & {100 * a['layer1_median_rel']:.2f} & {a['layer1_calib_1s']:.2f} & "
                     f"{a['layer1_nll']:.2f} & {100 * a['layer2_median_rel']:.1f} \\\\")
    lines += [r'\midrule',
              f'Amodal heuristic ($d_2 = {ratio:.2f}\\,d_1^{{\\mathrm{{GT}}}}$) & -- & -- & -- & {100 * heur:.1f} \\\\',
              r'\bottomrule', r'\end{tabular}']
    with open(os.path.join(ROOT, 'paper', 'tables', 'student.tex'), 'w') as f:
        f.write('\n'.join(lines))


def main(steps=400):
    t0 = time.time()
    with Pool(3) as pool:
        results = dict(pool.map(run_variant, [(w, steps, 0) for w in ('variance', 'uniform', 'min_views')]))
    ratio, heur = amodal_heuristic()
    out = {'steps': steps, 'resolution': [H, W], 'variants': results, 'amodal_heuristic': {'ratio': ratio, 'median_rel': heur}}
    with open(os.path.join(ROOT, 'results', 'student.json'), 'w') as f:
        json.dump(out, f, indent=2)
    write_table(results, ratio, heur)
    print(json.dumps(out, indent=2))
    print(f'all done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
