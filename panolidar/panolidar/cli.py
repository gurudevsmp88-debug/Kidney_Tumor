"""python -m panolidar {benchmark,train-synthetic,reconstruct} ..."""
import argparse
import json
import os

import torch
import yaml

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'default.yaml')


def load_config(path=None):
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def build_parser():
    parser = argparse.ArgumentParser(prog='python -m panolidar')
    parser.add_argument('--config', default=None)
    sub = parser.add_subparsers(dest='command', required=True)

    b = sub.add_parser('benchmark', help='prior vs stereo vs fused on synthetic streets')
    b.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    b.add_argument('--baseline', type=float, default=4.0)
    b.add_argument('--prior-rel-std', type=float, default=0.06)
    b.add_argument('--out', default=None, help='write the table as JSON')

    t = sub.add_parser('train-synthetic', help='train the reference student with a synthetic city teacher')
    t.add_argument('--steps', type=int, default=100)
    t.add_argument('--lr', type=float, default=1e-3)
    t.add_argument('--save', default=None, help='checkpoint path')

    r = sub.add_parser('reconstruct', help='training-free reconstruction of one panorama of a posed collection')
    r.add_argument('--scene', required=True, help='folder with poses.json (see panolidar/data.py)')
    r.add_argument('--ref', type=int, default=0, help='index of the panorama to reconstruct')
    r.add_argument('--src', type=int, default=None, help='second panorama (default: nearest within baseline range)')
    r.add_argument('--single', action='store_true', help='use only the reference panorama (prior only)')
    r.add_argument('--height', type=int, default=None)
    r.add_argument('--width', type=int, default=None)
    r.add_argument('--out', required=True, help='output .ply point cloud; depth/sigma saved next to it as .npy')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    if args.command == 'benchmark':
        from .experiments import fusion_benchmark, format_benchmark
        table = fusion_benchmark(cfg, tuple(args.seeds), args.baseline, args.prior_rel_std)
        print(format_benchmark(table))
        if args.out:
            with open(args.out, 'w') as f:
                json.dump(table, f, indent=2)
        return table

    if args.command == 'train-synthetic':
        from .experiments import train_synthetic
        model, history = train_synthetic(cfg, steps=args.steps, lr=args.lr)
        if args.save:
            torch.save(model.state_dict(), args.save)
            print(f'saved {args.save}')
        return history

    if args.command == 'reconstruct':
        import numpy as np
        from .data import PanoramaCollection, write_ply
        from .pipeline import reconstruct
        h = args.height or cfg['resolution']['height']
        w = args.width or cfg['resolution']['width']
        col = PanoramaCollection(args.scene, h, w)
        prior = col.prior(args.ref, cfg['fusion']['prior_sigma'])
        if prior is None:
            raise SystemExit('reconstruct needs a prior depth (prior_depth in poses.json), e.g. from a panoramic '
                             'depth model; stereo alone leaves the epipolar cones empty.')
        image1 = col.image(args.ref)
        src = None if args.single else (args.src if args.src is not None else col.nearest_partner(
            args.ref, cfg['pairing']['min_baseline'], cfg['pairing']['max_baseline']))
        image2 = col.image(src) if src is not None else None
        res = reconstruct(image1, col.pose(args.ref), prior[0], prior[1], image2,
                          col.pose(src) if src is not None else None, cfg)
        n = write_ply(args.out, res['points'], image1.permute(1, 2, 0))
        stem = os.path.splitext(args.out)[0]
        np.save(stem + '_depth.npy', res['depth'].numpy())
        np.save(stem + '_sigma_log.npy', res['sigma_log'].numpy())
        print(f"ref {args.ref}, src {src}, scale offset {res['scale_offset']:+.4f} (log), {n} points -> {args.out}")
        gt = col.depth(args.ref)
        if gt is not None:
            from .metrics import depth_metrics
            print(json.dumps(depth_metrics(res['depth'], gt, res['sigma_log']), indent=2))
        return res


if __name__ == '__main__':
    main()
