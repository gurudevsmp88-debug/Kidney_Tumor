"""Monte Carlo dropout uncertainty (notebook 02)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


def enable_dropout(model):
    """Put the model in eval mode but keep dropout layers stochastic."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    return model


@torch.no_grad()
def mc_dropout_predict(model, loader, device, n_samples=10, perturbation=None, params=None):
    """Run `n_samples` stochastic forward passes per batch, optionally on perturbed inputs.

    `perturbation(image, **params)` is applied per image (C, H, W) before each pass.
    Returns a dict of numpy arrays: preds, mean_probs, confidence, uncertainty, entropy, labels.
    """
    enable_dropout(model)
    params = params or {}
    samples, all_labels = [], []
    for images, labels in loader:
        batch_samples = []
        for _ in range(n_samples):
            inputs = images
            if perturbation is not None:
                inputs = torch.stack([perturbation(img, **params) for img in images])
            probs = F.softmax(model(inputs.to(device)), dim=1)
            batch_samples.append(probs.cpu().numpy())
        samples.append(np.stack(batch_samples))  # (n_samples, B, C)
        all_labels.append(np.asarray(labels))
    model.eval()

    samples = np.concatenate(samples, axis=1)  # (n_samples, N, C)
    mean_probs = samples.mean(axis=0)
    std_probs = samples.std(axis=0)
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-12), axis=1)
    return {
        'preds': mean_probs.argmax(axis=1),
        'mean_probs': mean_probs,
        'confidence': mean_probs.max(axis=1),
        'uncertainty': std_probs[:, 1],
        'entropy': entropy,
        'labels': np.concatenate(all_labels),
    }


def summarize(result):
    """Aggregate an mc_dropout_predict result into scalar metrics."""
    preds, labels, unc = result['preds'], result['labels'], result['uncertainty']
    errors = preds != labels
    unc_err = float(unc[errors].mean()) if errors.any() else 0.0
    unc_ok = float(unc[~errors].mean()) if (~errors).any() else 0.0
    return {
        'acc': float(accuracy_score(labels, preds)),
        'conf': float(result['confidence'].mean()),
        'unc': float(unc.mean()),
        'entropy': float(result['entropy'].mean()),
        'unc_on_errors': unc_err,
        'unc_on_correct': unc_ok,
        'unc_delta': unc_err - unc_ok,
    }
