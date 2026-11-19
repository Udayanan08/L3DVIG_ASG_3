"""Run from 4DGaussians/:
python ../experiments/scripts/render_4dgs.py -m <run_dir> --configs arguments/dynerf/coffee_martini_a3.py \
    --iteration 14000 --out ../experiments/renders/4dgs/k8 [--arbitrary_t 0.30 0.37 0.3701 0.44]"""
import os, sys
sys.path.insert(0, os.getcwd())
import torch, torchvision, mmcv
from argparse import ArgumentParser
from tqdm import tqdm
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
from gaussian_renderer import render, GaussianModel
from scene import Scene
from scene.dataset import FourDGSdataset
from scene.neural_3D_dataset_NDC import Neural3D_NDC_Dataset
from utils.general_utils import safe_state
from utils.params_utils import merge_hparams

SETS = {  # name: (loader split, frame subset)
    "traincams_testframes":  ("train", "test"),
    "cam00_testframes":      ("test",  "test"),
    "traincams_trainframes": ("train", "train"),
    "cam00_trainframes":     ("test",  "train"),
}
parser = ArgumentParser()
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)
hyperparam = ModelHiddenParams(parser)
parser.add_argument("--iteration", default=-1, type=int)
parser.add_argument("--configs", type=str)
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
parser.add_argument("--arbitrary_t", nargs="*", type=float, default=None)
args = get_combined_args(parser)
if args.configs:                      # same as render.py: restores the config's hyper-parameters
    args = merge_hparams(args, mmcv.Config.fromfile(args.configs))
safe_state(args.quiet)

camname = lambda p: os.path.basename(os.path.dirname(os.path.dirname(p)))
with torch.no_grad():
    dataset, pipe, hyper = model.extract(args), pipeline.extract(args), hyperparam.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    k, nf, td = dataset.train_stride, dataset.num_frames, dataset.time_denom
    print(f"k={k} frames={nf} time_denom={td} iter={scene.loaded_iter} temporal_res={hyper.kplanes_config['resolution'][3]}")

    def make(split, subset):
        return Neural3D_NDC_Dataset(dataset.source_path, split, 1.0, time_scale=1,
                                    scene_bbox_min=[-2.5, -2.0, -1.0], scene_bbox_max=[2.5, 2.0, 1.0],
                                    eval_index=0, num_frames=nf, time_denom=td,
                                    train_stride=k, frame_subset=subset)
    def draw(cam):
        return render(cam, gaussians, pipe, bg, stage="fine", cam_type="dynerf")["render"].clamp(0, 1)

    for name in args.sets:
        split, subset = SETS[name]
        ds = make(split, subset)
        cams = sorted({camname(p) for p in ds.image_paths})
        print(f"{name}: {len(ds)} images, {len(cams)} cameras, frames {sorted(set(ds.image_frames))}")
        assert ("cam00" in cams) == (split == "test"), "held-out camera leaked into / missing from a set"
        fds = FourDGSdataset(ds, dataset, "dynerf")
        for i in tqdm(range(len(fds)), desc=name):
            d = os.path.join(args.out, camname(ds.image_paths[i])); os.makedirs(d, exist_ok=True)
            torchvision.utils.save_image(draw(fds[i]), os.path.join(d, f"{ds.image_frames[i]:04d}.png"))

    if getattr(args, "arbitrary_t", None):              # t = frame/50, e.g. frame 4 is t=0.08
        fds = FourDGSdataset(make("test", "test"), dataset, "dynerf")
        cam, prev = fds[0], None
        d = os.path.join(args.out, "arbitrary_t"); os.makedirs(d, exist_ok=True)
        for t in args.arbitrary_t:
            cam.time = float(t)
            img = draw(cam)
            diff = float((img - prev).abs().mean()) if prev is not None else float("nan")
            print(f"t={t:.4f}  mean|diff to previous render|={diff:.5f}")
            torchvision.utils.save_image(img, os.path.join(d, f"cam00_t{t:.4f}.png")); prev = img
