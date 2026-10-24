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
| `panolidar/cli.py` | `python -m panolidar benchmark / train-synthetic / reconstruct` |

## Usage

```bash
pip install -r requirements.txt
python -m pytest tests -q                         # 15 CPU tests (incl. Monte Carlo check of the sigma formula)
python -m panolidar benchmark --seeds 0 1 2 3 4 5 6 7 8 9
python -m panolidar train-synthetic --steps 200 --save student.pt

# your data: folder with poses.json + panoramas + prior depth .npy (format in panolidar/data.py)
python -m panolidar reconstruct --scene /path/to/city_block --ref 12 --out ref12.ply           # nearest partner
python -m panolidar reconstruct --scene /path/to/city_block --ref 12 --single --out ref12.ply  # 1 panorama
```

`reconstruct` writes a coloured PLY plus `*_depth.npy` and `*_sigma_log.npy`. If the folder
also has ground-truth depth (e.g. projected LiDAR), it prints the metrics. Priors come from any
panoramic depth model run beforehand (e.g. Depth Any Panoramas), saved as metric depth `.npy`.

## Results so far (synthetic streets only, 128x256, baseline 4 m along the street)

`python -m panolidar benchmark --seeds 0..9`. The simulated prior has ~6% smooth relative error
plus up to ±10% global scale error. Median relative error per angle between ray and baseline:

| method | 0–10° | 10–30° | 30–60° | 60–90° | all | calib. 1σ (ideal 0.68) |
| --- | --- | --- | --- | --- | --- | --- |
| prior | 0.060 | 0.053 | 0.056 | 0.058 | 0.057 | 0.54 |
| prior, scale-aligned | 0.053 | 0.039 | 0.038 | 0.040 | 0.039 | 0.70 |
| stereo only | 0.390 | 0.057 | 0.026 | 0.018 | 0.022 | 0.77 |
| **fused** | **0.053** | **0.036** | **0.025** | **0.018** | **0.021** | **0.72** |

The fused result matches or beats the better source in every bin and stays calibrated. Absolute
accuracy at 256 px width is far from LiDAR. The angular precision, and hence σ, scales with 1/width,
so 8K panoramas should be about 30x better in theory; this has not been measured. **These are
synthetic results with a simulated prior; nothing here has been run on real panoramas yet.**

## Still needed for a paper

1. **Real benchmark with LiDAR:** KITTI-360 (two 180° fisheyes, about a full sphere, plus accumulated
   LiDAR incl. occluded regions). Needs a fisheye-to-ERP loader (not written yet).
2. **Real prior:** run Depth Any Panoramas / DA360 / PaGeR and feed their depth as the prior.
3. **Real teacher:** SfM + MVS (e.g. COLMAP spherical, or 2DGS) on the dense city capture. The
   teacher depth maps replace the synthetic ones in `build_teacher_points`. Validate the teacher
   against any LiDAR you have.
4. **Student:** replace the small CNN with a foundation backbone. Learn the stereo features (the
   cost volume runs on learned features already). Train on the city, test on *unseen* areas or cities.
5. **Baselines:** Splatter-360, PanSplat, Depth Any Panoramas, DA360, PaGeR, MASt3R/VGGT on cube faces.
6. **Ablations:** analytic σ vs learned σ vs none; with/without scale alignment; gating; amodal layer;
   baseline length; panorama resolution.

## Known limitations

- Poses are assumed known (GPS/IMU or SfM). Pose error enters the σ budget and is not modelled yet.
- One panorama cannot be LiDAR-accurate in general; the single-view path returns the prior with its σ.
- Moving objects (cars, people) break two-view consistency. The gate rejects them as outliers, but
  their depth then comes from the prior.
