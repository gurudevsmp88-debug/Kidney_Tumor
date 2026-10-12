"""Data preparation, dataset and dataloaders (from notebooks 02, 03, 09)."""
import json
import os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SPLITS = ('train', 'val', 'test')


# ----------------------------------------------------------------------------
# Patient-level splitting (notebook 02)
# ----------------------------------------------------------------------------

def patient_level_split(num_samples, slices_per_patient=50, ratios=(0.7, 0.15, 0.15), seed=42):
    """Group consecutive slices into pseudo-patients and split by patient, not by slice."""
    patient_groups = defaultdict(list)
    for idx in range(num_samples):
        patient_groups[idx // slices_per_patient].append(idx)

    patients = sorted(patient_groups.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(patients)

    split_1 = int(ratios[0] * len(patients))
    split_2 = int((ratios[0] + ratios[1]) * len(patients))
    split_patients = {
        'train': patients[:split_1],
        'val': patients[split_1:split_2],
        'test': patients[split_2:],
    }
    splits = {f'{name}_indices': [i for pid in pids for i in patient_groups[pid]]
              for name, pids in split_patients.items()}
    splits['patient_groups'] = {str(k): v for k, v in patient_groups.items()}
    return splits


def prepare_dataset(cfg):
    """Download the HF dataset, split by patient, resize and write images + labels to data.root."""
    from datasets import load_dataset  # optional dependency, only needed here
    from tqdm import tqdm

    dcfg = cfg['data']
    root = dcfg['root']
    ds = load_dataset(dcfg['hf_dataset'])['train']
    ds.set_format('numpy')
    label_feature = ds.features.get('label') if hasattr(ds, 'features') else None

    splits = patient_level_split(len(ds), dcfg['slices_per_patient'], dcfg['split'], cfg['seed'])
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, 'splits.json'), 'w') as f:
        json.dump(splits, f)

    size = dcfg['image_size']
    for split in SPLITS:
        os.makedirs(os.path.join(root, split), exist_ok=True)
        labels_list = []
        for idx in tqdm(splits[f'{split}_indices'], desc=f'Processing {split}'):
            sample = ds[int(idx)]
            label = sample['label']
            if label_feature is not None and hasattr(label_feature, 'int2str') and not isinstance(label, str):
                label = label_feature.int2str(int(label))
            img = Image.fromarray(np.asarray(sample['image']).astype(np.uint8))
            img = img.resize((size, size), Image.Resampling.LANCZOS)
            filename = f'{split}_{idx}.png'
            img.save(os.path.join(root, split, filename))
            labels_list.append({'filename': filename, 'label': str(label)})
        with open(os.path.join(root, f'{split}_labels.json'), 'w') as f:
            json.dump(labels_list, f)
        print(f'{split.upper()} - saved {len(labels_list)} images to {os.path.join(root, split)}')
    return splits


# ----------------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------------

class GaussianNoiseTransform:
    """Add Gaussian noise to a tensor (notebook 09)."""

    def __init__(self, std=0.02):
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std


def get_transforms(augmentation='standard'):
    """Return (train_transform, eval_transform)."""
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    eval_tf = transforms.Compose([transforms.ToTensor(), normalize])

    if augmentation == 'standard':  # notebook 03
        train_tf = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    elif augmentation == 'robust':  # notebook 09
        train_tf = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
            normalize,
            GaussianNoiseTransform(std=0.02),
        ])
    else:
        raise ValueError(f'Unknown augmentation: {augmentation}')
    return train_tf, eval_tf


# ----------------------------------------------------------------------------
# Dataset / loaders
# ----------------------------------------------------------------------------

class KidneyDataset(Dataset):
    """Images in `data_dir` with a list of {'filename', 'label'} records."""

    def __init__(self, labels_list, data_dir, transform=None,
                 class_names=('kidney_normal', 'kidney_tumor')):
        self.labels_list = labels_list
        self.data_dir = data_dir
        self.transform = transform
        self.label_to_idx = {name: i for i, name in enumerate(class_names)}

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, idx):
        item = self.labels_list[idx]
        image = Image.open(os.path.join(self.data_dir, item['filename'])).convert('RGB')
        label = self.label_to_idx[item['label']]
        if self.transform:
            image = self.transform(image)
        return image, label


def load_labels(root, split):
    path = os.path.join(root, f'{split}_labels.json')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found. Set the dataset path with --data-root / KIDNEY_DATA_ROOT '
            f'or data.root in the config, or run `python -m src prepare` first.')
    with open(path, 'r') as f:
        return json.load(f)


def build_dataloaders(cfg, splits=SPLITS, augmentation=None):
    """Return a dict split -> DataLoader built from cfg['data']['root']."""
    dcfg = cfg['data']
    train_tf, eval_tf = get_transforms(augmentation or cfg['train']['augmentation'])
    loaders = {}
    for split in splits:
        dataset = KidneyDataset(load_labels(dcfg['root'], split),
                                os.path.join(dcfg['root'], split),
                                train_tf if split == 'train' else eval_tf,
                                dcfg['class_names'])
        loaders[split] = DataLoader(dataset, batch_size=dcfg['batch_size'],
                                    shuffle=(split == 'train'), num_workers=dcfg['num_workers'])
    return loaders
