# panolidar: LiDAR-grade 3D from one or two panoramas

Research code for **epipole-aware fusion of panoramic depth priors with spherical triangulation,
trained with a city-scale teacher**.

## Idea

- **Two high-resolution panoramas a few metres apart can triangulate to cm level**, but not
  everywhere. The depth uncertainty of a ray is
  `sigma_z^2 = sigma_ang^2 |b|^2 (sin^2 alpha + sin^2 beta cos^2 gamma) / sin^4 gamma`
  (derived in `panolidar/uncertainty.py`; far field `~ sqrt(2) z^2 sigma_ang / (|b| sin alpha)`).
  It blows up near the **epipoles**, i.e. along the direction of travel, which is where the
  street continues.
- **Monocular panoramic depth models** (Depth Any Panoramas, DA360, PaGeR) cover every ray, but
  at roughly 5–7% relative error, i.e. metres at 20 m.
- **The method:** per ray, fuse both sources in log-depth, weighting stereo by the *closed-form*
  triangulation uncertainty. The pipeline also aligns the prior's global scale to well-conditioned
  rays, and gates out stereo that disagrees with the prior (occlusions, mismatches). The output
  is metric depth with a calibrated per-ray σ, like a LiDAR spec sheet. With one panorama it
  degrades gracefully to the prior.
- **City as teacher:** a dense city capture is reconstructed once with multi-view geometry. Each
  teacher point gets a triangulation covariance (`multiview_point_covariance`). A student that
  sees only 1–2 panoramas is supervised with uncertainty-weighted NLL on the visible surface
  **and** on the first hidden surface behind it (amodal layer: behind cars, around corners),
  which only other panoramas of the city can see.

## Layout

| File | Contents |
| --- | --- |
| `panolidar/sphere.py` | ERP rays/angles/pixels, wrap-around sampling, poses, back/projection |
| `panolidar/uncertainty.py` | closed-form two-view depth σ, multi-view point covariance |
| `panolidar/stereo.py` | spherical plane-sweep stereo, coarse-to-fine, match confidence |
| `panolidar/fusion.py` | scale alignment, gated precision-weighted log-depth fusion |
| `panolidar/pipeline.py` | training-free reconstruction (any external prior + stereo + fusion) |
| `panolidar/teacher.py` | city teacher points, two-layer (visible + amodal) targets, NLL loss |
| `panolidar/model.py` | reference learned student (small CNN; swap in a foundation model) |
| `panolidar/synthetic.py` | procedural street-canyon scenes with exact depth |
| `panolidar/metrics.py` | LiDAR-style metrics (within 5/10/20 cm, calibration, Chamfer) |
| `panolidar/data.py` | posed panorama folders (`poses.json`), PLY export |
| `panolidar/calibration.py` | geometry-conditioned calibration `a(c)·U^κ` (+ pose term), fitted baselines |
| `panolidar/benchmark.py` | evaluation protocols used in the paper |
| `panolidar/kitti360.py` | MEI fisheye pair → equirect, LiDAR → ERP depth, KITTI-360 readers |
| `panolidar/cli.py` | `python -m panolidar benchmark / train-synthetic / reconstruct` |

## Usage

```bash
pip install -r requirements.txt
python -m pytest tests -q                         # CPU tests (incl. Monte Carlo checks of the sigma formula)
python -m panolidar benchmark --seeds 0 1 2 3 4 5 6 7 8 9
python -m panolidar train-synthetic --steps 200 --save student.pt

# your data: folder with poses.json + panoramas + prior depth .npy (format in panolidar/data.py)
python -m panolidar reconstruct --scene /path/to/city_block --ref 12 --out ref12.ply           # nearest partner
python -m panolidar reconstruct --scene /path/to/city_block --ref 12 --single --out ref12.ply  # 1 panorama
```

`reconstruct` writes a coloured PLY plus `*_depth.npy` and `*_sigma_log.npy`. If the folder
also has ground-truth depth (e.g. projected LiDAR), it prints the metrics. Priors come from any
panoramic depth model run beforehand (e.g. Depth Any Panoramas), saved as metric depth `.npy`.

## Paper draft and experiments

`paper/main.pdf` is a CVPR-format draft (official CVPR 2027 author kit) built from generated tables.
Reproduce every number on CPU:

```bash
python experiments/run_paper_experiments.py   # calibration, main table, ablations, transfer  (~3 min)
python experiments/run_robustness.py          # relative-pose noise, mixed baselines          (~4 min)
python experiments/run_capture.py             # exposure change, sensor noise, moving cars     (~15 min)
python experiments/run_teacher_calibration.py # calibration from a noisy city teacher          (~5 min)
python experiments/run_student.py 400         # learned student (did not converge on CPU)
bash experiments/run_all.sh                   # everything used in the paper, in order
python experiments/make_figures.py            # qualitative figure
cd paper && latexmk -pdf main.tex
```

Honest summary of the synthetic results (256x512, details in the paper):

- **Fusion works:** median relative error falls from 6.95% (simulated prior) to 1.61%, and the
  uncertainties are calibrated (71% of errors fall within 1σ).
- **No advantage in-distribution:** a geometry-blind σ ∝ 1/confidence, fitted on the same
  calibration data, is just as accurate (1.60%). The match confidence already collapses at the
  epipoles. The same tie holds when transferring to other baselines and resolutions.
- **The closed form matters under relative-pose error:** at 1°/10 cm the error drops from 3.93% to
  3.39% (best baseline), 1σ coverage is 0.67 vs ≤0.46, and NLL is −1.54 vs ≥−0.82. At 0.3°/5 cm only
  calibration and NLL improve.
- **Exposure-robust matching:** with the ZNCC cost, the confidence-based σ breaks down (5.29% vs
  2.80% clean, 6.72% vs 4.56% with exposure change, noise and moving cars combined); the closed form
  does not.
- **Calibration from a noisy city teacher** matches calibration from ground truth (1.61–1.62%).
- **The learned student did not converge** in 400 CPU steps (`results/student.json`, median errors
  of 31–830%). Those numbers say nothing about teacher weighting and are not in the paper.
- **Nothing here is real data.** KITTI-360 evaluation (`panolidar/kitti360.py`, untested on the real
  dataset) is the decisive missing experiment.

## Known limitations

- Poses are assumed known (GPS/IMU or SfM). Pose error enters the σ budget and is not modelled yet.
- One panorama cannot be LiDAR-accurate in general; the single-view path returns the prior with its σ.
- Moving objects (cars, people) break two-view consistency. The gate rejects them as outliers, but
  their depth then comes from the prior.
