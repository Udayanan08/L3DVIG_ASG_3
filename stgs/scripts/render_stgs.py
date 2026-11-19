"""Render an STGS run into the same layout as render_4dgs.py: <out>/camXX/NNNN.png.
Run from SpacetimeGaussians/:
python ../experiments/scripts/render_stgs.py --model_path ../experiments/runs/stgs/k8 \
    --source_path ../experiments/data/coffee_martini/colmap_0 --configpath configs/n3d_full/coffee_martini.json \
    --test_iteration 30000 --valloader colmapvalid --out ../experiments/renders/stgs/k8 [--arbitrary_t 0.30 0.37 0.3701 0.44]
Frames/stride/init cloud come from the run's cfg_args (train_stride, duration, initply)."""
import os, sys
sys.path.insert(0, os.getcwd())
sys.path.append("./thirdparty/gaussian_splatting")
import argparse
import torch, torchvision
from tqdm import tqdm

SETS = {  # name: (held-out camera?, frame subset)
    "traincams_testframes":  (False, "test"),
    "cam00_testframes":      (True,  "test"),
    "traincams_trainframes": (False, "train"),
    "cam00_trainframes":     (True,  "train"),
}
own = argparse.ArgumentParser()
own.add_argument("--out", type=str, required=True)
own.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
own.add_argument("--arbitrary_t", nargs="*", type=float, default=None)
own_args, rest = own.parse_known_args()
sys.argv = sys.argv[:1] + rest  # remaining flags go to the STGS test parser

from thirdparty.gaussian_splatting.scene import Scene
from thirdparty.gaussian_splatting.scene.dataset_readers import readColmapSceneInfo
from thirdparty.gaussian_splatting.utils.camera_utils import loadCamv2
from thirdparty.gaussian_splatting.helper3dg import gettestparse
from helper_train import getrenderpip, getmodel, trbfunction

FUSED = {"train_ours_full": "test_ours_full_fused", "train_ours_lite": "test_ours_lite"}  # same choice as test.py

def frame_selected(idx, k, num_frames, subset):  # identical rule to the 4DGS loader
    last_train = ((num_frames - 1) // k) * k
    is_train = idx % k == 0
    is_test = k > 1 and idx % k == k // 2 and idx < last_train
    return is_train if subset == "train" else is_test

args, dataset, pipe, _ = gettestparse()
assert args.test_iteration > 0, "pass --test_iteration explicitly (the n3d config sets 10000)"
k, nf = dataset.train_stride, args.duration
with torch.no_grad():
    gaussians = getmodel(dataset.model)(dataset.sh_degree, args.rgbfunction)
    scene = Scene(dataset, gaussians, load_iteration=args.test_iteration, shuffle=False, duration=nf, loader="colmapvalid")
    if gaussians.rgbdecoder is not None:
        gaussians.rgbdecoder.cuda()
        gaussians.rgbdecoder.eval()
    if gaussians.ts is None:
        cam = scene.getTestCameras()[0]
        gaussians.ts = torch.ones(1, 1, cam.image_height, cam.image_width).cuda()
    render, GRsetting, GRzer = getrenderpip(FUSED.get(args.rdpip, args.rdpip))
    bg = torch.zeros(9, dtype=torch.float32, device="cuda")
    print(f"k={k} frames={nf} iter={scene.loaded_iter} init={dataset.initply} renderer={FUSED.get(args.rdpip, args.rdpip)}")

    # every camera at every frame, no train/test split; images are loaded lazily per view
    infos = readColmapSceneInfo(dataset.source_path, dataset.images, False, duration=nf, initply=dataset.initply).train_cameras
    heldout = sorted({c.image_name for c in infos})[0]
    assert heldout == "cam00", heldout
    rays = {}
    for i, ci in enumerate(infos):  # rays are per camera, built from its frame-0 view as in Scene
        if ci.pose is not None:
            cam = loadCamv2(dataset, i, ci, 1.0)
            rays[ci.image_name] = torch.cat([cam.rayo, cam.rayd], dim=1).cuda()

    def draw(i, ci, t=None):
        cam = loadCamv2(dataset, i, ci, 1.0)
        cam.rays = rays[ci.image_name]
        if t is not None:
            cam.timestamp = t
        return render(cam, gaussians, pipe, bg, scaling_modifier=1.0, basicfunction=trbfunction,
                      GRsetting=GRsetting, GRzer=GRzer)["render"].clamp(0, 1)

    for name in own_args.sets:
        want_heldout, subset = SETS[name]
        views = [(i, ci) for i, ci in enumerate(infos)
                 if (ci.image_name == heldout) == want_heldout
                 and frame_selected(int(round(ci.timestamp * nf)), k, nf, subset)]
        frames = sorted({int(round(ci.timestamp * nf)) for _, ci in views})
        print(f"{name}: {len(views)} images, {len({ci.image_name for _, ci in views})} cameras, frames {frames}")
        for i, ci in tqdm(views, desc=name):
            d = os.path.join(own_args.out, ci.image_name); os.makedirs(d, exist_ok=True)
            torchvision.utils.save_image(draw(i, ci), os.path.join(d, f"{int(round(ci.timestamp * nf)):04d}.png"))

    if own_args.arbitrary_t:  # t = frame/50, e.g. frame 4 is t=0.08
        i, ci = next((i, ci) for i, ci in enumerate(infos) if ci.image_name == heldout)
        d = os.path.join(own_args.out, "arbitrary_t"); os.makedirs(d, exist_ok=True)
        prev = None
        for t in own_args.arbitrary_t:
            img = draw(i, ci, float(t))
            diff = float((img - prev).abs().mean()) if prev is not None else float("nan")
            print(f"t={t:.4f}  mean|diff to previous render|={diff:.5f}")
            torchvision.utils.save_image(img, os.path.join(d, f"{heldout}_t{t:.4f}.png")); prev = img
