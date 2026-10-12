# Kidney Tumor Classification

Binary CT-slice classifier (`kidney_normal` vs `kidney_tumor`). The notebooks (`01`–`10`) are kept
unchanged for reference; the code from notebooks **02, 03 and 09** is refactored into the `src/` package.

## Layout

| Module | Contents |
| --- | --- |
| `src/data.py` | patient-level split, dataset preparation, `KidneyDataset`, standard/robust transforms, dataloaders |
| `src/models.py` | `create_model` (resnet18/50, efficientnet_b0/b3, densenet121, vgg19, vit_b16, xception, mobilenetv3, vit, unet) |
| `src/train.py` | training loop with early stopping and best-checkpoint saving |
| `src/evaluate.py` | test predictions and metrics (accuracy, precision, recall, F1, ROC-AUC, confusion matrix) |
| `src/uncertainty.py` | Monte Carlo dropout predictions and uncertainty summaries |
| `src/robustness.py` | noise / contrast / blur / rotation / translation perturbations and the robustness suite |
| `src/cli.py` | argparse CLI (`python -m src ...`) |
| `configs/config.yaml` | all settings |

## Usage

```bash
pip install -r requirements.txt

# Dataset path: --data-root  >  KIDNEY_DATA_ROOT env var  >  data.root in configs/config.yaml
python -m src prepare     --data-root /path/to/kidney_data      # download + patient split + resize
python -m src train       --data-root /path/to/kidney_data --arch resnet18 --augmentation robust
python -m src evaluate    --data-root /path/to/kidney_data --arch resnet18
python -m src uncertainty --data-root /path/to/kidney_data --arch resnet18 --mc-samples 10
python -m src robustness  --data-root /path/to/kidney_data --arch resnet18
```

The dataset directory must contain `train/`, `val/`, `test/` and `{train,val,test}_labels.json`
(the layout `prepare` writes). Use `--config` to point at a different YAML file.

## Tests

```bash
python -m pytest tests -q
```

`tests/test_smoke.py` runs every stage on CPU using random dummy tensors and a tiny generated
image folder; it needs no dataset or pretrained-weight downloads.
