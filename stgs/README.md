# STGS (SpacetimeGaussians): temporal interpolation on coffee_martini

This folder covers the STGS half of Assignment 3 (E9 247, temporal interpolation with dynamic 3DGS).
It holds only what we changed or added. The method itself is the official code,
[oppo-us-research/SpacetimeGaussians](https://github.com/oppo-us-research/SpacetimeGaussians), at commit
`427abfc58309a4a5213843dd673fb22c4529306c`, run with the **full** model and its unchanged
`configs/n3d_full/coffee_martini.json`.

```
stgs/
├── patches/stgs_temporal_interpolation.patch   # all our edits to the official code (4 files)
├── SpacetimeGaussians_modified/                # the same 4 files after the patch (drop-in copies)
├── scripts/
│   ├── prepare_data.py       # STGS data folder (colmap_0..49) from the shared 1352x1014 GT PNGs
│   ├── make_init_ply.py      # init point cloud (frame-0 COLMAP points, same 5,940 points as 4DGS)
│   ├── render_stgs.py        # renders every camera / frame set + arbitrary t, same layout as 4DGS
│   ├── plot_perframe_k8.py   # per-frame PSNR plot, k=8 (4DGS and STGS)
│   └── crops_method.py       # visual result: GT | render | error map on a fast-moving crop of cam00
└── results/
    ├── summary_stgs.csv      # averages per k, setting, train/test frames; full / dynamic / static
    ├── per_image_stgs.csv    # every scored image
    └── figures/              # psnr_k8_traincams.png, psnr_k8_cam00.png, crops_cam00_stgs.png
```

## Protocol (identical to the 4DGS runs)

| Item | Value |
|---|---|
| Scene | Neural 3D Video, `coffee_martini`, frames 0–49, 1352×1014 (the same GT PNGs for every method) |
| Held-out camera | cam00 (STGS's own `--eval` split also holds out cam00). Never used in training. |
| Training cameras | the other 17: cam01–cam20, including cam13 (cam03, cam15 and cam17 are not in the dataset) |
| Gap k | 2, 4, 8. Train frames: `f % k == 0`. Test frames: `f % k == k/2` and `f <` last train frame |
| Training images | 425 / 221 / 119 for k = 2 / 4 / 8 |
| Time | t = frame / 50 (`duration = 50`). STGS renders any real-valued t. |
| Init cloud | frame-0 COLMAP points (5,940, t = 0), the same cloud the 4DGS runs use |
| Metrics | PSNR, SSIM, LPIPS (VGG) on full images and inside the dynamic mask. These come from the team's shared evaluation script and per-camera RAFT masks, so all methods share one mask. |
| Hardware | NVIDIA RTX 6000 Ada Generation, 48 GB |

## What we changed in the official code, and why

All edits are in `patches/stgs_temporal_interpolation.patch` (about 20 lines).

| File | Change | Why |
|---|---|---|
| `thirdparty/gaussian_splatting/arguments/__init__.py` | new options `--train_stride` (k) and `--initply` | to choose k and the init cloud from the command line |
| `thirdparty/gaussian_splatting/scene/dataset_readers.py` (`readColmapSceneInfo`) | keeps only training views with `frame % k == 0`; loads `--initply` instead of `points3D_total50.ply` | by default STGS trains on all 50 frames. Its default init cloud merges the COLMAP points of all 50 frames, which includes the test frames, so it would leak them. |
| `thirdparty/gaussian_splatting/scene/__init__.py` | passes both options to the reader | |
| `train.py` | samples the training time only from frames that have training views (was `randint(0, 49)`); fixed seed 6666 | otherwise it would pick test times that have no views; the seed matches the 4DGS runs |

Deviations from the STGS defaults are only these: the k-strided training views, the frame-0 init cloud instead of
`points3D_total50.ply`, `--resolution 1` (our PNGs are already 1352×1014; the config says 2), and seed 6666.
Everything else stays the same: full model, 30,000 iterations, batch 2, the same densification and
learning rates for every k.

## Reproduce

### 0. Get the code and apply the patch
```bash
git clone https://github.com/oppo-us-research/SpacetimeGaussians.git
cd SpacetimeGaussians
git checkout 427abfc58309a4a5213843dd673fb22c4529306c
git apply /path/to/L3DVIG_ASG_3/stgs/patches/stgs_temporal_interpolation.patch
```
Then follow the official README to build the environment and the CUDA rasterizers. We ran everything in one conda env.

Set these paths (adjust them to your machine):
```bash
STGS=/path/to/SpacetimeGaussians
E=/path/to/stgs_experiments          # data, init clouds, runs and renders go here
SCR=/path/to/L3DVIG_ASG_3/stgs/scripts
GT=/path/to/coffee_martini_gt         # camXX/images/0000..0049.png, 1352x1014 (shared with 4DGS)
COLMAP=/path/to/coffee_martini_colmap # output of STGS script/pre_n3d.py (colmap_<j>/sparse/0, poses_bounds.npy)
```

### 1. Data folder and init cloud (once)
```bash
python $SCR/prepare_data.py --gt $GT --colmap $COLMAP --out $E/data/coffee_martini
```
```bash
python $SCR/make_init_ply.py --mode frame0 --colmap $COLMAP --out $E/init/points_frame0.ply
```
`make_init_ply.py` imports from `~/perception/tools/stgs/SpacetimeGaussians` (variable `STGS` at the top), so
edit that line if your checkout is elsewhere.

### 2. Training (run from `$STGS`; only `--train_stride` and `--model_path` change)
k=8
```bash
python train.py --eval --config configs/n3d_full/coffee_martini.json --resolution 1 \
  --train_stride 8 --initply $E/init/points_frame0.ply \
  --model_path $E/runs/stgs/k8 --source_path $E/data/coffee_martini/colmap_0 \
  2>&1 | tee $E/runs/stgs/k8.log
```
k=4
```bash
python train.py --eval --config configs/n3d_full/coffee_martini.json --resolution 1 \
  --train_stride 4 --initply $E/init/points_frame0.ply \
  --model_path $E/runs/stgs/k4 --source_path $E/data/coffee_martini/colmap_0 \
  2>&1 | tee $E/runs/stgs/k4.log
```
k=2
```bash
python train.py --eval --config configs/n3d_full/coffee_martini.json --resolution 1 \
  --train_stride 2 --initply $E/init/points_frame0.ply \
  --model_path $E/runs/stgs/k2 --source_path $E/data/coffee_martini/colmap_0 \
  2>&1 | tee $E/runs/stgs/k2.log
```
Check: the log prints the train frames (k=8: 0, 8, …, 48). `cameras.json` should list 17 cameras × the train frames
(119 / 221 / 425 views).

### 3. Rendering (run from `$STGS`)
This writes `<out>/camXX/NNNN.png` for training cameras and cam00 at both train and test frames. It also renders
cam00 at arbitrary times in `<out>/arbitrary_t/`, to check that rendering is continuous in t.
k=8
```bash
python $SCR/render_stgs.py --model_path $E/runs/stgs/k8 \
  --source_path $E/data/coffee_martini/colmap_0 --configpath configs/n3d_full/coffee_martini.json \
  --valloader colmapvalid --test_iteration 30000 --out $E/renders/stgs/k8 \
  --arbitrary_t 0.30 0.37 0.3701 0.44
```
k=4
```bash
python $SCR/render_stgs.py --model_path $E/runs/stgs/k4 \
  --source_path $E/data/coffee_martini/colmap_0 --configpath configs/n3d_full/coffee_martini.json \
  --valloader colmapvalid --test_iteration 30000 --out $E/renders/stgs/k4 \
  --arbitrary_t 0.30 0.37 0.3701 0.44
```
k=2
```bash
python $SCR/render_stgs.py --model_path $E/runs/stgs/k2 \
  --source_path $E/data/coffee_martini/colmap_0 --configpath configs/n3d_full/coffee_martini.json \
  --valloader colmapvalid --test_iteration 30000 --out $E/renders/stgs/k2 \
  --arbitrary_t 0.30 0.37 0.3701 0.44
```
`--test_iteration 30000` is required because the config sets 10000. Rendering uses the fused full renderer, the
same one STGS's `test.py` uses. Do not use the metrics from STGS's own `test.py`: it scores cam00 on all 50 frames.

### 4. Evaluation and figures
The renders are scored by the shared evaluation [`../4dgs/evaluation/run_all_eval.py`](../4dgs/evaluation/run_all_eval.py)
(same GT, same per-camera dynamic masks for all methods; see the 4DGS README). That produces `per_image_<method>.csv`
and `summary_all.csv`; the STGS rows are in `results/`. The figures:
```bash
python $SCR/plot_perframe_k8.py --methods 4dgs stgs --k 8 --eval <dir with per_image_*.csv> --out <figures dir>
```
```bash
python $SCR/crops_method.py --method stgs --renders $E/renders/stgs --gt $GT \
  --flowmags <masks dir>/flowmags_cam00.npy --out <figures dir>
```
`crops_method.py` picks the same 320×320 fast-moving crop of cam00 as the shared script (x 488–808, y 415–735) and
the test frame with the most motion in it (frames 1 / 2 / 4 for k = 2 / 4 / 8).

## Results (test frames, from `results/summary_stgs.csv`)

| k | Train cams PSNR / SSIM / LPIPS | cam00 PSNR / SSIM / LPIPS | Train cams, dynamic | cam00, dynamic |
|---|---|---|---|---|
| 2 | 30.11 / 0.914 / 0.248 | 27.59 / 0.891 / 0.246 | 29.14 / 0.885 / 0.263 | 28.39 / 0.884 / 0.250 |
| 4 | 30.08 / 0.915 / 0.249 | 26.47 / 0.893 / 0.249 | 28.86 / 0.881 / 0.271 | 27.69 / 0.880 / 0.262 |
| 8 | 29.53 / 0.907 / 0.261 | 27.20 / 0.887 / 0.259 | 26.97 / 0.844 / 0.299 | 26.65 / 0.851 / 0.271 |

`summary_stgs.csv` also has the same metrics on the training frames (`kind=train`), which serve as a control.
