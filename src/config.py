"""Config loading and device/seed helpers."""
import copy
import os
import random

import numpy as np
import torch
import yaml

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'configs', 'config.yaml')


def load_config(path=None, data_root=None):
    """Load YAML config. Dataset root precedence: data_root arg > KIDNEY_DATA_ROOT env > YAML."""
    with open(path or DEFAULT_CONFIG, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    env_root = os.environ.get('KIDNEY_DATA_ROOT')
    if data_root:
        cfg['data']['root'] = data_root
    elif env_root:
        cfg['data']['root'] = env_root
    return cfg


def get_device(name='auto'):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
