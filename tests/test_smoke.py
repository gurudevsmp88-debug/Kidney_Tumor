"""CPU smoke test: runs every pipeline stage on random dummy tensors (no dataset, no downloads)."""
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from src import cli
from src.config import load_config
from src.data import KidneyDataset, build_dataloaders, get_transforms, patient_level_split
from src.evaluate import evaluate
from src.models import create_model, load_model
from src.robustness import PERTURBATIONS, run_robustness_suite
from src.train import fit
from src.uncertainty import mc_dropout_predict, summarize

DEVICE = torch.device('cpu')
IMG = 64


def dummy_loader(n=8, batch_size=4):
    torch.manual_seed(0)
    images = torch.randn(n, 3, IMG, IMG)
    labels = torch.tensor([0, 1] * (n // 2))
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


def test_patient_level_split_has_no_leakage():
    splits = patient_level_split(1000, slices_per_patient=50)
    sets = [set(splits[f'{s}_indices']) for s in ('train', 'val', 'test')]
    assert sum(len(s) for s in sets) == 1000
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    for idx in splits['patient_groups'].values():  # each patient in exactly one split
        assert sum(set(idx) <= s for s in sets) == 1


def test_transforms_shapes():
    img = Image.fromarray(np.random.randint(0, 255, (IMG, IMG, 3), dtype=np.uint8))
    for aug in ('standard', 'robust'):
        train_tf, eval_tf = get_transforms(aug)
        assert train_tf(img).shape == (3, IMG, IMG)
        assert eval_tf(img).shape == (3, IMG, IMG)


def test_models_forward():
    x = torch.randn(2, 3, IMG, IMG)
    for arch in ('resnet18', 'unet'):
        model = create_model(arch, num_classes=2, pretrained=False).eval()
        assert model(x).shape == (2, 2)


def test_perturbations_preserve_shape():
    img = torch.randn(3, IMG, IMG)
    for name, value in [('noise', 0.1), ('contrast', 1.2), ('blur', 1.0), ('rotate', 10), ('translate', [0.1, 0.0])]:
        key = {'noise': 'intensity', 'contrast': 'factor', 'blur': 'sigma', 'rotate': 'angle', 'translate': 'shift'}[name]
        assert PERTURBATIONS[name](img, **{key: value}).shape == img.shape


def test_end_to_end_pipeline(tmp_path):
    cfg = load_config()
    tcfg = dict(cfg['train'], epochs=2, patience=5)
    loader = dummy_loader()
    ckpt = str(tmp_path / 'ckpt' / 'unet_best.pth')

    model = create_model('unet', num_classes=2, pretrained=False)
    model, history = fit(model, loader, loader, tcfg, DEVICE, ckpt, progress=False)
    assert os.path.exists(ckpt) and len(history['train_loss']) == 2

    metrics = evaluate(model, loader, DEVICE)
    assert 0.0 <= metrics['accuracy'] <= 1.0
    assert sum(metrics['confusion_matrix'].values()) == 8

    reloaded = load_model('unet', ckpt, num_classes=2, device=DEVICE)
    result = mc_dropout_predict(reloaded, loader, DEVICE, n_samples=3)
    assert result['mean_probs'].shape == (8, 2)
    assert np.allclose(result['mean_probs'].sum(1), 1.0, atol=1e-5)
    assert summarize(result)['unc'] >= 0.0

    pert_cfg = {'noise': {'param': 'intensity', 'values': [0.1]},
                'translate': {'param': 'shift', 'values': [[0.1, 0.0]]}}
    results = run_robustness_suite(reloaded, loader, DEVICE, pert_cfg, n_samples=2)
    assert set(results) == {'clean', 'noise_0.1', 'translate_[0.1, 0.0]'}


def _write_dummy_dataset(root, n=4):
    for split in ('train', 'val', 'test'):
        os.makedirs(root / split)
        records = []
        for i in range(n):
            name = f'{split}_{i}.png'
            arr = np.random.randint(0, 255, (IMG, IMG), dtype=np.uint8)
            Image.fromarray(arr).save(root / split / name)
            records.append({'filename': name, 'label': ['kidney_normal', 'kidney_tumor'][i % 2]})
        (root / f'{split}_labels.json').write_text(json.dumps(records))


def test_configurable_dataset_path_and_cli(tmp_path, monkeypatch):
    root = tmp_path / 'my_kidney_data'
    _write_dummy_dataset(root)

    cfg = load_config(data_root=str(root))
    cfg['data'].update(batch_size=2, num_workers=0)
    loaders = build_dataloaders(cfg)
    images, labels = next(iter(loaders['test']))
    assert images.shape == (2, 3, IMG, IMG) and set(labels.tolist()) <= {0, 1}
    assert isinstance(loaders['train'].dataset, KidneyDataset)

    monkeypatch.setenv('KIDNEY_DATA_ROOT', str(root))
    assert load_config()['data']['root'] == str(root)

    common = ['--arch', 'unet', '--device', 'cpu', '--num-workers', '0', '--batch-size', '2',
              '--checkpoint', str(tmp_path / 'unet.pth'), '--output-dir', str(tmp_path / 'results')]
    cli.main(['train', '--data-root', str(root), '--epochs', '1', '--no-pretrained'] + common)
    cli.main(['evaluate', '--data-root', str(root)] + common)
    cli.main(['uncertainty', '--data-root', str(root), '--mc-samples', '2'] + common)
    assert (tmp_path / 'results' / 'unet_test_metrics.json').exists()
    assert (tmp_path / 'results' / 'unet_uncertainty.json').exists()
