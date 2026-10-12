"""Command-line interface: python -m src <command> [options]."""
import argparse
import json
import os

from .config import get_device, load_config, set_seed
from .models import AVAILABLE_MODELS


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--config', default=None, help='Path to YAML config (default: configs/config.yaml)')
    common.add_argument('--data-root', default=None,
                        help='Dataset directory (overrides KIDNEY_DATA_ROOT and data.root)')
    common.add_argument('--arch', choices=AVAILABLE_MODELS, default=None, help='Model architecture')
    common.add_argument('--device', default=None, help='auto | cpu | cuda')
    common.add_argument('--batch-size', type=int, default=None)
    common.add_argument('--num-workers', type=int, default=None)
    common.add_argument('--checkpoint', default=None, help='Checkpoint path to save/load')
    common.add_argument('--output-dir', default=None, help='Where JSON results are written')

    parser = argparse.ArgumentParser(prog='python -m src', description='Kidney tumor classification pipeline')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare', parents=[common], help='Download, split by patient and resize the dataset')
    train = sub.add_parser('train', parents=[common], help='Train a model')
    train.add_argument('--epochs', type=int, default=None)
    train.add_argument('--lr', type=float, default=None)
    train.add_argument('--augmentation', choices=['standard', 'robust'], default=None)
    train.add_argument('--no-pretrained', action='store_true', help='Train from random init')
    sub.add_parser('evaluate', parents=[common], help='Evaluate a checkpoint on the test split')
    unc = sub.add_parser('uncertainty', parents=[common], help='MC-dropout uncertainty on the test split')
    unc.add_argument('--mc-samples', type=int, default=None)
    rob = sub.add_parser('robustness', parents=[common], help='Robustness under input perturbations')
    rob.add_argument('--mc-samples', type=int, default=None)
    return parser


def apply_overrides(cfg, args):
    if args.arch:
        cfg['model']['architecture'] = args.arch
    if args.device:
        cfg['device'] = args.device
    if args.batch_size:
        cfg['data']['batch_size'] = args.batch_size
    if args.num_workers is not None:
        cfg['data']['num_workers'] = args.num_workers
    if args.checkpoint:
        cfg['evaluate']['checkpoint'] = args.checkpoint
    if args.output_dir:
        cfg['evaluate']['output_dir'] = args.output_dir
    if getattr(args, 'epochs', None):
        cfg['train']['epochs'] = args.epochs
    if getattr(args, 'lr', None):
        cfg['train']['lr'] = args.lr
    if getattr(args, 'augmentation', None):
        cfg['train']['augmentation'] = args.augmentation
    if getattr(args, 'no_pretrained', False):
        cfg['model']['pretrained'] = False
    if getattr(args, 'mc_samples', None):
        cfg['uncertainty']['mc_samples'] = args.mc_samples
        cfg['robustness']['mc_samples'] = args.mc_samples
    return cfg


def _save_json(cfg, name, payload):
    out_dir = cfg['evaluate']['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'Saved {path}')


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = apply_overrides(load_config(args.config, args.data_root), args)
    set_seed(cfg['seed'])
    device = get_device(cfg['device'])
    arch = cfg['model']['architecture']
    print(f"Command: {args.command} | data root: {cfg['data']['root']} | arch: {arch} | device: {device}")

    if args.command == 'prepare':
        from .data import prepare_dataset
        prepare_dataset(cfg)
        return

    from .data import build_dataloaders
    from .models import create_model, load_model
    from .train import default_checkpoint_path
    ckpt = cfg['evaluate']['checkpoint'] or default_checkpoint_path(cfg)
    num_classes = cfg['model']['num_classes']

    if args.command == 'train':
        from .evaluate import evaluate, format_metrics
        from .train import fit
        loaders = build_dataloaders(cfg)
        model = create_model(arch, num_classes, cfg['model']['pretrained'])
        model, history = fit(model, loaders['train'], loaders['val'], cfg['train'], device, ckpt)
        metrics = evaluate(model, loaders['test'], device)
        print(format_metrics(metrics, f'TEST SET METRICS - {arch}'))
        _save_json(cfg, f'{arch}_train_summary.json', {'history': history, 'test_metrics': metrics})
        return

    loaders = build_dataloaders(cfg, splits=('test',))
    model = load_model(arch, ckpt, num_classes, device)

    if args.command == 'evaluate':
        from .evaluate import evaluate, format_metrics
        metrics = evaluate(model, loaders['test'], device)
        print(format_metrics(metrics, f'TEST SET METRICS - {arch}'))
        _save_json(cfg, f'{arch}_test_metrics.json', metrics)
    elif args.command == 'uncertainty':
        from .uncertainty import mc_dropout_predict, summarize
        summary = summarize(mc_dropout_predict(model, loaders['test'], device, cfg['uncertainty']['mc_samples']))
        print(json.dumps(summary, indent=2))
        _save_json(cfg, f'{arch}_uncertainty.json', summary)
    elif args.command == 'robustness':
        from .robustness import run_robustness_suite
        results = run_robustness_suite(model, loaders['test'], device,
                                       cfg['robustness']['perturbations'], cfg['robustness']['mc_samples'])
        _save_json(cfg, f'{arch}_robustness.json', results)


if __name__ == '__main__':
    main()
