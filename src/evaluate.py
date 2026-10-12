"""Test-set prediction and metrics (notebooks 03, 09)."""
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


@torch.no_grad()
def predict(model, loader, device):
    """Return (preds, labels, probs) as numpy arrays."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for images, labels in loader:
        outputs = model(images.to(device))
        probs = torch.softmax(outputs, dim=1)
        all_preds.append(probs.argmax(1).cpu().numpy())
        all_labels.append(np.asarray(labels))
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels), np.concatenate(all_probs)


def compute_metrics(labels, preds, probs):
    """Binary metrics with tumor (class 1) as the positive class."""
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    try:
        auc = float(roc_auc_score(labels, probs[:, 1]))
    except ValueError:  # only one class present
        auc = float('nan')
    return {
        'accuracy': float(accuracy_score(labels, preds)),
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'recall': float(recall_score(labels, preds, zero_division=0)),
        'f1': float(f1_score(labels, preds, zero_division=0)),
        'roc_auc': auc,
        'confusion_matrix': {
            'true_negatives': int(cm[0, 0]), 'false_positives': int(cm[0, 1]),
            'false_negatives': int(cm[1, 0]), 'true_positives': int(cm[1, 1]),
        },
    }


def evaluate(model, loader, device):
    preds, labels, probs = predict(model, loader, device)
    return compute_metrics(labels, preds, probs)


def format_metrics(metrics, title='TEST SET METRICS'):
    cm = metrics['confusion_matrix']
    return '\n'.join([
        '=' * 50, title, '=' * 50,
        f"Accuracy:  {metrics['accuracy']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall:    {metrics['recall']:.4f}",
        f"F1-Score:  {metrics['f1']:.4f}",
        f"ROC-AUC:   {metrics['roc_auc']:.4f}",
        f"TN: {cm['true_negatives']}, FP: {cm['false_positives']}",
        f"FN: {cm['false_negatives']}, TP: {cm['true_positives']}",
    ])
