# 4DGS (4DGaussians): temporal interpolation on coffee_martini

This folder covers the 4DGS half of Assignment 3 (E9 247, temporal interpolation with dynamic 3DGS), plus the shared
evaluation that scores every method with the same GT, masks and metrics.
It holds only what we changed or added. The method itself is the official code,
[hustvl/4DGaussians](https://github.com/hustvl/4DGaussians), at commit `843d5ac636c37e4b611242287754f3d4ed150144`.

```
4dgs/
├── patches/4dgs_temporal_interpolation.patch   # all our edits to the official code (4 modified files + 2 new configs)
├── 4DGaussians_modified/                       # the same 6 files after the patch (drop-in copies)
├── scripts/
│   ├── extract_frames.py     # cam*.mp4 -> camXX/images/0000..0049.png at 1352x1014 (PIL LANCZOS)
│   ├── check_projection.py   # sanity check: init points projected into cameras vs GT colours
│   ├── check_splits.py       # sanity check: train/test frames do not overlap, cam00 never in training
│   ├── render_4dgs.py        # renders train cams + cam00 at train and test frames, plus arbitrary t
│   ├── compare.py            # render | GT | error x4 images per test frame
│   └── finish_4dgs.py        # 4DGS-only mask / eval / report (same definitions as run_all_eval.py)
├── evaluation/
│   ├── run_all_eval.py       # shared evaluation for all methods: masks, metrics, tables, plots, crops
│   └── masks/                # the per-camera dynamic masks (camXX.png), mask_stats.csv, params.json
├── results/
│   ├── summary_4dgs.csv      # averages per k, setting, train/test frames; full / dynamic / static
│   ├── per_image_4dgs.csv    # every scored image
│   └── figures/              # per-frame PSNR (k=8), crops on cam00, mask overlays
└── splits.txt
```

## Protocol (identical to the STGS runs)

| Item | Value |
|---|---|
| Scene | Neural 3D Video, `coffee_martini`, frames 0–49, 1352×1014 (2× LANCZOS downsample of 2704×2028) |
| Held-out camera | cam00 (`eval_index=0`, the dataset's standard test camera). Never in the training loss. |
| Training cameras | the other 17: cam01–cam20, including cam13 (cam03, cam15 and cam17 are not in the dataset) |
| Gap k | 2, 4, 8. Train frames: `f % k == 0`. Test frames: `f % k == k/2` and `f <` last train frame (see `splits.txt`) |
| Training images | 425 / 221 / 119 for k = 2 / 4 / 8 |
| Time | t = frame / 50 for every k (`--time_denom 50`) |
| Init cloud | sparse COLMAP points of frame 0 only, filtered 6,880 → 5,940 points (inside the 2nd–98th percentile box per axis, reprojection error below the 95th percentile), saved as `points3D_downsample2.ply` (the file name the loader expects) |
| Hardware | NVIDIA RTX 6000 Ada Generation 48 GB, PyTorch 1.13.1+cu117, Python 3.7.16 |

The init cloud has exactly the same points and colours as the STGS one, so `../stgs/scripts/make_init_ply.py --mode frame0`
reproduces it (the STGS file only adds a `t` field). The COLMAP reconstruction of frame 0 used all 18 frame-0 images,
including cam00's. That affects only the initial point positions, never the loss.

## What we changed in the official code, and why

| File | Change | Why |
|---|---|---|
| `scene/neural_3D_dataset_NDC.py` | new arguments `num_frames`, `time_denom`, `train_stride`, `frame_subset` (`train` / `test` / `all`) and a `frame_selected()` rule; reads the pre-extracted PNGs | the default loader uses 300 frames, t = idx/300 and every frame. We need 50 frames, t = frame/50, and only the k-strided train frames. |
| `scene/dataset_readers.py` (`readdynerfInfo`) | builds the train set (17 cams, train frames) and the test set (cam00, test frames); asserts there is no train/test frame overlap; `maxtime = num_frames` | enforces the split |
| `scene/__init__.py` | passes `train_stride`, `num_frames`, `time_denom` to the reader | |
| `arguments/__init__.py` | new options `--train_stride`, `--num_frames`, `--time_denom` (the three lines appear twice; that's harmless, same values) | command-line control, saved in `cfg_args` |
| `arguments/dynerf/coffee_martini_a3.py` (new) | inherits `default.py`, sets the HexPlane temporal resolution to 25 | the default 150 cells is for 300 frames (0.5 cells per frame); 25 keeps that ratio for 50 frames |
| `arguments/dynerf/coffee_martini_smoke.py` (new) | 400-iteration smoke test | pipeline check only |

Deviations from the 4DGS defaults: the 50-frame k-strided split, t = frame/50, the sparse frame-0 init cloud (no
dense cloud was available) and temporal grid 25 instead of 150. Everything else is the repo's dynerf `default.py`:
3,000 coarse + 14,000 fine iterations, batch 4, densify until 10k, opacity reset 60,000, spatial grid 64³,
output dim 16, multires [1, 2], `plane_tv_weight` 0.0002, `time_smoothness_weight` 0.001, `l1_time_planes` 0.0001.
Seed 6666 for every k.

## Reproduce

### 0. Get the code and apply the patch
```bash
git clone https://github.com/hustvl/4DGaussians.git
cd 4DGaussians
git checkout 843d5ac636c37e4b611242287754f3d4ed150144
git submodule update --init --recursive
git apply /path/to/L3DVIG_ASG_3/4dgs/patches/4dgs_temporal_interpolation.patch
```
Then build the environment and submodules as described in the official README.

### 1. Data
Put `coffee_martini` (the `cam*.mp4` videos and `poses_bounds.npy`) in `data/dynerf/coffee_martini`, then extract frames 0–49:
```bash
python /path/to/L3DVIG_ASG_3/4dgs/scripts/extract_frames.py --datadir data/dynerf/coffee_martini
```
Put the frame-0 init cloud at `data/dynerf/coffee_martini/points3D_downsample2.ply` (see Protocol).
Optional checks (run from `4DGaussians/`): `python /path/to/scripts/check_splits.py`.

### 2. Training (run from `4DGaussians/`; only `--train_stride`, `-m` and `--port` change)
k=8
```bash
python train.py -s data/dynerf/coffee_martini -m experiments/runs/4dgs/k8 --port 6018 \
  --train_stride 8 --num_frames 50 --time_denom 50 --configs arguments/dynerf/coffee_martini_a3.py
```
k=4
```bash
python train.py -s data/dynerf/coffee_martini -m experiments/runs/4dgs/k4 --port 6019 \
  --train_stride 4 --num_frames 50 --time_denom 50 --configs arguments/dynerf/coffee_martini_a3.py
```
k=2
```bash
python train.py -s data/dynerf/coffee_martini -m experiments/runs/4dgs/k2 --port 6020 \
  --train_stride 2 --num_frames 50 --time_denom 50 --configs arguments/dynerf/coffee_martini_a3.py
```
The log prints the train frames and the held-out frames. Wall-clock time was about 48 / 42 / 56 min for
k = 8 / 4 / 2, and the final Gaussian counts were 94,766 / 100,580 / 96,893.

### 3. Rendering (run from `4DGaussians/`)
This writes `<out>/camXX/NNNN.png` for training cameras and cam00 at both train and test frames. It also renders
cam00 at arbitrary times, to check that rendering is continuous in t.
k=8
```bash
python /path/to/scripts/render_4dgs.py -m experiments/runs/4dgs/k8 --configs arguments/dynerf/coffee_martini_a3.py \
  --iteration 14000 --out experiments/renders/4dgs/k8 --arbitrary_t 0.30 0.37 0.3701 0.44
```
k=4
```bash
python /path/to/scripts/render_4dgs.py -m experiments/runs/4dgs/k4 --configs arguments/dynerf/coffee_martini_a3.py \
  --iteration 14000 --out experiments/renders/4dgs/k4 --arbitrary_t 0.30 0.37 0.3701 0.44
```
k=2
```bash
python /path/to/scripts/render_4dgs.py -m experiments/runs/4dgs/k2 --configs arguments/dynerf/coffee_martini_a3.py \
  --iteration 14000 --out experiments/renders/4dgs/k2 --arbitrary_t 0.30 0.37 0.3701 0.44
```

### 4. Evaluation (all methods)
`evaluation/run_all_eval.py` expects this layout (override with `--root`, `--gt`, `--renders method=path`, `--out`):
```
ROOT/4dgs/experiments/data/coffee_martini/camXX/images/NNNN.png     GT
ROOT/<method>/experiments/renders/<method>/k{2,4,8}/camXX/NNNN.png   renders (4dgs, stgs, ...)
```
```bash
python evaluation/run_all_eval.py --root ROOT --methods 4dgs stgs
```
It has three stages:
- **mask:** RAFT `raft_large` flow between consecutive GT frames. A pixel is dynamic if |flow| > 1.0 px in at least
  3 of the 49 frame pairs. Then blobs under 300 px are removed and the mask is dilated 7×7. That gives one mask per
  camera, the same for every method and every k: 8.9% of pixels on average (0.4%–16.1%). The masks used in the report
  are in `evaluation/masks/`.
- **eval:** PSNR, SSIM (11×11 Gaussian, σ = 1.5) and LPIPS (VGG) per image, on the full image, inside the mask
  (dynamic) and outside it (static). Masked PSNR comes from the MSE over masked pixels. Masked SSIM and LPIPS average
  their spatial maps, computed on the full images, over the masked pixels.
- **report:** summary tables, LaTeX rows, the per-frame PSNR plot for k=8, crops on the fastest-moving 320×320
  region of cam00, and mask overlays.

## Results (test frames, from `results/summary_4dgs.csv`)

| k | Train cams PSNR / SSIM / LPIPS | cam00 PSNR / SSIM / LPIPS | Train cams, dynamic | cam00, dynamic |
|---|---|---|---|---|
| 2 | 31.25 / 0.918 / 0.227 | 27.73 / 0.897 / 0.230 | 31.47 / 0.916 / 0.214 | 30.11 / 0.912 / 0.222 |
| 4 | 31.10 / 0.916 / 0.229 | 27.37 / 0.895 / 0.235 | 31.15 / 0.909 / 0.222 | 29.40 / 0.910 / 0.224 |
| 8 | 30.77 / 0.913 / 0.236 | 27.10 / 0.892 / 0.240 | 29.63 / 0.892 / 0.232 | 28.23 / 0.891 / 0.233 |

`summary_4dgs.csv` also has the same metrics on the training frames (`kind=train`), which serve as a control.
