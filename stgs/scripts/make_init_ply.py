"""Write an STGS init cloud (x, y, z, t, normals, rgb).

--mode frame0       sparse COLMAP points of frame 0 only, t = 0. Same 5940 points as the 4DGS runs.
--mode trainframes  sparse COLMAP points of the train frames of stride k (idx % k == 0), t = idx/50.
Each frame is filtered like the 4DGS cloud: inside the 2nd-98th percentile box per axis and
reprojection error below the 95th percentile (both over that frame's points).
"""
import os, sys, argparse
import numpy as np
STGS = os.path.expanduser("~/perception/tools/stgs/SpacetimeGaussians")
sys.path[:0] = [STGS, os.path.join(STGS, "thirdparty/gaussian_splatting")]
from thirdparty.gaussian_splatting.scene.colmap_loader import read_points3D_binary
from thirdparty.gaussian_splatting.scene.dataset_readers import storePly

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["frame0", "trainframes"], required=True)
p.add_argument("--k", type=int, default=1)
p.add_argument("--colmap", default=os.path.expanduser("~/perception/tools/data/coffee_martini"))
p.add_argument("--frames", type=int, default=50)
p.add_argument("--out", required=True)
a = p.parse_args()

frames = [0] if a.mode == "frame0" else [j for j in range(a.frames) if j % a.k == 0]
allxyzt, allrgb = [], []
for j in frames:
    xyz, rgb, err = read_points3D_binary(os.path.join(a.colmap, f"colmap_{j}", "sparse", "0", "points3D.bin"))
    err = err.reshape(-1)
    keep = np.all((xyz >= np.percentile(xyz, 2, 0)) & (xyz <= np.percentile(xyz, 98, 0)), 1)
    keep &= err < np.percentile(err, 95)
    allxyzt.append(np.concatenate([xyz[keep], np.full((keep.sum(), 1), j / a.frames)], 1))
    allrgb.append(rgb[keep])
    print(f"frame {j}: {len(xyz)} -> {keep.sum()}")
xyzt, rgb = np.concatenate(allxyzt), np.concatenate(allrgb)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
storePly(a.out, xyzt, rgb)
print(f"{a.out}: {len(xyzt)} points from frames {frames}")
