# Split-Half Densification: noise-unbiased adaptive density control for 3DGS

This folder is the official 3D Gaussian Splatting code (graphdeco-inria, commit `54c035f`,
unmodified in the commit that added it) plus one new method, **split-half densification (SHD)**.
Upstream's own README is `README.md`; the license is `LICENSE.md` (non-commercial research use).

## Problem

3DGS grows its model with adaptive density control: a Gaussian is cloned or split when the
view-averaged norm of its screen-space positional gradient exceeds a threshold,

    s_vanilla = mean_v || g_v ||,   g_v = dL_v / d mu_2D.

When the training images are noisy (low light, high ISO, small sensors, medical or scientific
cameras), the noise contributes to every `g_v`. A norm is always non-negative, so this
contribution does not average out across views: `E||g|| >= ||E g||`, with a gap that grows with
the noise. Under the L1 loss used by 3DGS, the gradient is `sign(residual)` per pixel, so a
Gaussian that already fits the clean signal still gets a gradient of size
`~ sqrt(footprint) / N` once noise dominates its residual. Densification therefore reacts to
noise, not just to under-fitting. Prior work reports the symptom, thin, unstable primitives that
fit sensor noise (e.g. LL-Gaussian, *3DGS in the Dark*, NeurIPS 2024). The existing remedies
denoise the inputs first, add a noise model, or drop heuristic densification (3DGS-MCMC,
NOVA-GS). In the searches done so far we found none that corrects the densification statistic itself.

## Method

For each training view, split the pixels into two random disjoint halves A and B (optionally in
`b x b` blocks), and compute the positional gradients of the per-pixel photometric loss on each
half, `g_A` and `g_B`. Accumulate the cross term instead of the norm:

    s_SHD = 2 * sqrt( max(0, mean_v <g_A,v , g_B,v>) ).

**Why it is unbiased.** Write `g_A = J_A^T phi(r_A)`, where `r_A` are the residuals on half A and
`phi` is the per-pixel loss derivative (`sign` for L1). The Jacobian `J` depends only on the
current render. If pixel noise is independent across pixels, `r_A` and `r_B` are conditionally
independent given the render, so

    E[ <g_A, g_B> ] = < E[g_A], E[g_B] >.

The noise cancels in expectation. With interleaved random halves `E[g_A] ~= E[g_B] ~= E[g]/2`, so
`4 <g_A, g_B>` estimates `||E g||^2`, the squared gradient of the noise-free expected
objective. Only independence between disjoint pixel sets is used, not linearity, so the
argument holds for L1, L2 and Charbonnier alike. (SSIM couples pixels within an 11x11 window,
so the statistic is computed on the L1 term only; the optimization itself is unchanged.) It is
the same idea as half-set correlation in cryo-EM (Fourier shell correlation) and Noise2Noise,
applied to the densification signal.

Byproduct: `||g_A - g_B||^2` estimates the gradient noise energy, giving a per-Gaussian
**gradient SNR** that is logged to TensorBoard (`split_half/median_gradient_snr`).

In the toy below, both scores agree exactly on clean images; whether SHD is neutral on clean real
scenes still has to be measured. The extra cost is two extra
rasterizer backward passes per iteration during densification only (the first 15k of 30k
iterations); `--split_half_every k` amortizes it.

## Code

| File | Change |
| --- | --- |
| `utils/split_half.py` | new: split masks, half losses, half gradients, score, SNR, noise injection (pure PyTorch) |
| `scene/gaussian_model.py` | `add_split_half_stats`; `densify_and_prune(..., densify_score=...)` |
| `train.py` | computes `g_A`, `g_B` before `loss.backward()`; optional noise injection on training images |
| `arguments/__init__.py` | `--densify_score {vanilla,split_half}`, `--split_half_block`, `--split_half_every`, `--train_noise_std`, `--train_noise_seed` |
| `split_half_toy.py` | CPU toy experiment (differentiable 2D splatting) |
| `tests_split_half/` | CPU unit tests, including a Monte Carlo check of the unbiasedness identity |
| `scripts/run_noise_benchmark.sh` | GPU benchmark: scenes x noise levels x {vanilla, split_half} |

Defaults reproduce vanilla 3DGS exactly.

## Usage

```bash
git submodule update --init --recursive        # CUDA rasterizer, simple-knn, fused-ssim
conda env create -f environment.yml && conda activate gaussian_splatting

# vanilla vs split-half on a noisy capture (noise added to training views only; test GT stays clean)
python train.py -s <scene> --eval --train_noise_std 0.1 --densify_score vanilla    -m out/vanilla
python train.py -s <scene> --eval --train_noise_std 0.1 --densify_score split_half -m out/shd
python render.py -m out/shd --skip_train && python metrics.py -m out/shd

bash scripts/run_noise_benchmark.sh /path/to/mipnerf360     # full sweep

python split_half_toy.py                                    # CPU toy, ~10 s
python -m pytest tests_split_half -q                        # CPU tests
```

## Evidence so far (CPU toy only)

`python split_half_toy.py --noise 0 0.02 0.05 0.1 0.2 --views 200`. There are 36 Gaussians; 18 are
displaced by 0.25–1.5 px ("mis-fit", need densification) and 18 sit at the truth ("well-fit").
AUC = probability that a mis-fit Gaussian outscores a well-fit one.

| noise σ | vanilla score, well-fit | SHD score, well-fit | vanilla AUC | SHD AUC |
| --- | --- | --- | --- | --- |
| 0.00 | 0 | 0 | 1.000 | 1.000 |
| 0.02 | 6.2e-5 | 1.1e-5 | 1.000 | 1.000 |
| 0.05 | 6.2e-5 | 1.1e-5 | 1.000 | 1.000 |
| 0.10 | 6.3e-5 | 1.5e-5 | 0.988 | 1.000 |
| 0.20 | 6.3e-5 | 0.9e-5 | 0.833 | 1.000 |

The noise floor that vanilla assigns to correctly fitted Gaussians is 4–7x lower with SHD. What
remains is Monte Carlo error that shrinks with the number of views. SHD's ranking stays perfect
at σ = 0.2, while vanilla's ranking degrades. Both lose sensitivity to small mis-fits at high
noise (TP rate 0.39 at σ = 0.2), because the L1 sign gradient itself shrinks; SHD does not fix
that. **This is a toy. It shows the estimator does what the math says; it does not show
better novel-view synthesis.**

## What a CVPR submission still needs

1. **Main results** (GPU): Mip-NeRF 360, Tanks & Temples, Deep Blending with synthetic noise
   (Gaussian, and Poisson–Gaussian shot+read noise at several ISO-like levels). Report
   PSNR/SSIM/LPIPS against clean GT, number of Gaussians, training time, and memory.
2. **Real noise**: RawNeRF scenes (raw low-light captures), LOM, or another real low-light set.
   Ideally run in the linear raw domain, where pixel noise is closest to independent.
3. **Baselines**: 3DGS, AbsGS, Taming-3DGS, Revising Densification (Bulò et al.), 3DGS-MCMC,
   denoise-then-3DGS (e.g. Restormer/NAFNet on the inputs), and LL-Gaussian / NOVA-GS /
   *3DGS in the Dark* for low light. SHD is orthogonal to input denoising and noise-robust losses,
   so also report "baseline + SHD".
4. **Ablations**: block size (pixel-independent vs. demosaiced noise), split ratio, `every k`,
   L1 vs L2, statistic on L1 only vs. masked SSIM, SHD for densification only vs. also for pruning.
5. **Analysis**: Gaussian count and the scale/anisotropy histogram vs. noise level (does SHD
   remove the thin noise-fitting primitives?); gradient-SNR maps; the clean-data result
   (SHD should be neutral there; if it is not, explain why).
6. **Literature check** before writing: search for "half-split", "cross-correlation",
   "Noise2Noise" + "densification" / "Gaussian splatting". The searches done here found
   cross-view *voting* densification (FastGS, GC-GS, Structure-Aware Densification) and
   uncertainty-based pruning (PUP 3D-GS), but no split-half estimator.

## Risks to be aware of

- The gain may be small on clean benchmarks. The paper has to be framed around noisy captures.
- Reviewers may see a one-statistic change as incremental. It needs strong real-noise results,
  the unbiasedness analysis, and ideally an extension: SHD for pruning, or SHD-weighted
  opacity reset.
- Real camera noise after the ISP is spatially correlated. Use `--split_half_block` > 1, and show
  how robust the method is to this.
- Not yet run on a GPU. `train.py` calls `torch.autograd.grad` twice through the CUDA rasterizer
  with `retain_graph=True`; check this on the first run.
