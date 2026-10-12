"""Input perturbations and robustness evaluation under MC dropout (notebook 02)."""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torchvision.transforms.functional import affine, rotate

# Clamp range of ImageNet-normalized pixels.
CLAMP_MIN, CLAMP_MAX = -2.1, 2.8


def add_gaussian_noise(image, intensity):
    return torch.clamp(image + torch.randn_like(image) * intensity, CLAMP_MIN, CLAMP_MAX)


def contrast_shift(image, factor):
    return torch.clamp(image * factor, CLAMP_MIN, CLAMP_MAX)


def apply_blur(image, sigma):
    image_np = image.detach().cpu().numpy()
    blurred = np.stack([gaussian_filter(channel, sigma=sigma) for channel in image_np])
    return torch.from_numpy(blurred).to(image.dtype)


def rotate_image(image, angle):
    return rotate(image, angle)


def translate_image(image, shift):
    h, w = image.shape[-2:]
    dx, dy = int(shift[0] * w), int(shift[1] * h)
    return affine(image, angle=0.0, translate=[dx, dy], scale=1.0, shear=[0.0, 0.0])


PERTURBATIONS = {
    'noise': add_gaussian_noise,
    'contrast': contrast_shift,
    'blur': apply_blur,
    'rotate': rotate_image,
    'translate': translate_image,
}


def run_robustness_suite(model, loader, device, perturbation_cfg, n_samples=10):
    """Evaluate clean + every configured perturbation level.

    perturbation_cfg: {name: {'param': str, 'values': list}} as in configs/config.yaml.
    Returns {key: summary} where key is 'clean' or '<name>_<value>'.
    """
    from .uncertainty import mc_dropout_predict, summarize

    results = {'clean': summarize(mc_dropout_predict(model, loader, device, n_samples))}
    print(f"  clean -> acc={results['clean']['acc']:.4f}")
    for name, spec in perturbation_cfg.items():
        fn = PERTURBATIONS[name]
        for value in spec['values']:
            params = {spec['param']: value}
            summary = summarize(mc_dropout_predict(model, loader, device, n_samples, fn, params))
            key = f'{name}_{value}'
            results[key] = summary
            print(f"  {key} -> acc={summary['acc']:.4f}, unc_delta={summary['unc_delta']:.4f}")
    return results
