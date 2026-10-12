"""Training loop with early stopping and best-checkpoint saving (notebooks 03, 09)."""
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train_one_epoch(model, loader, criterion, optimizer, device, progress=True):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc='Training', leave=False, disable=not progress):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, progress=True):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc='Validating', leave=False, disable=not progress):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), 100.0 * correct / total


def build_optimizer(model, tcfg):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=tcfg['lr'], weight_decay=tcfg.get('weight_decay', 0.0))
    if tcfg.get('scheduler', 'cosine') == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg['epochs'])
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=tcfg['step_size'], gamma=tcfg['gamma'])
    return optimizer, scheduler


def fit(model, train_loader, val_loader, tcfg, device, checkpoint_path, progress=True):
    """Train `model`; keep the best val-accuracy weights at `checkpoint_path` and reload them."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = build_optimizer(model, tcfg)
    os.makedirs(os.path.dirname(checkpoint_path) or '.', exist_ok=True)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc, patience_counter = -1.0, 0
    for epoch in range(tcfg['epochs']):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, progress)
        val_loss, val_acc = validate(model, val_loader, criterion, device, progress)
        scheduler.step()
        for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[key].append(value)
        print(f"Epoch {epoch + 1}/{tcfg['epochs']}: train loss {train_loss:.4f} acc {train_acc:.2f}% | "
              f"val loss {val_loss:.4f} acc {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc, patience_counter = val_acc, 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f'  best model saved to {checkpoint_path}')
        else:
            patience_counter += 1
            if patience_counter >= tcfg['patience']:
                print(f'  early stopping at epoch {epoch + 1}')
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    history['best_val_acc'] = best_val_acc
    return model, history


def default_checkpoint_path(cfg):
    return os.path.join(cfg['train']['checkpoint_dir'], f"{cfg['model']['architecture']}_best.pth")
