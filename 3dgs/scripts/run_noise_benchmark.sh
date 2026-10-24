#!/usr/bin/env bash
# Noisy-capture benchmark: vanilla vs split-half densification on Mip-NeRF 360 (needs a CUDA GPU).
#
# Usage: bash scripts/run_noise_benchmark.sh /path/to/mipnerf360 [output_dir]
# Trains every scene at every noise level with both densification scores, renders the
# held-out test views (--eval) and evaluates them against the *clean* ground truth.
set -euo pipefail

DATA=${1:?usage: run_noise_benchmark.sh /path/to/mipnerf360 [output_dir]}
OUT=${2:-output/noise_benchmark}
SCENES=${SCENES:-"bicycle garden stump room counter kitchen bonsai"}
NOISE_LEVELS=${NOISE_LEVELS:-"0.0 0.05 0.1 0.2"}
METHODS=${METHODS:-"vanilla split_half"}
cd "$(dirname "$0")/.."

for scene in $SCENES; do
  case $scene in
    bicycle|flowers|garden|stump|treehill) images=images_4 ;;
    *) images=images_2 ;;
  esac
  for noise in $NOISE_LEVELS; do
    for method in $METHODS; do
      model="$OUT/$scene/noise_${noise}/$method"
      python train.py -s "$DATA/$scene" -i "$images" -m "$model" --eval --disable_viewer --quiet \
        --densify_score "$method" --train_noise_std "$noise" --test_iterations -1
      python render.py -m "$model" --skip_train --quiet
      python metrics.py -m "$model"
      python - "$model" <<'PY'
import sys
from plyfile import PlyData
model = sys.argv[1]
n = PlyData.read(f"{model}/point_cloud/iteration_30000/point_cloud.ply").elements[0].count
open(f"{model}/num_gaussians.txt", "w").write(f"{n}\n")
print(f"{model}: {n} Gaussians")
PY
    done
  done
done
