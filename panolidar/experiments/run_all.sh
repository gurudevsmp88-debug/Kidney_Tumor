#!/usr/bin/env bash
# Re-run every experiment of the paper in dependency order (CPU, ~30 min).
set -euo pipefail
cd "$(dirname "$0")/.."
python experiments/run_paper_experiments.py
python experiments/run_robustness.py
python experiments/run_capture.py
python experiments/run_teacher_calibration.py
python experiments/make_figures.py
echo "ALL EXPERIMENTS DONE"
