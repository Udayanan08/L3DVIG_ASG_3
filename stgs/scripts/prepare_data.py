"""Build the STGS data folder from the GT PNGs shared with the 4DGS runs.

colmap_<j>/images/camXX.png  -> 4dgs/experiments/data/coffee_martini/camXX/images/<j:04d>.png (1352x1014, LANCZOS)
colmap_0/sparse/0/{cameras,images}.bin  copied from the shared COLMAP output (poses come from poses_bounds.npy)
poses_bounds.npy -> shared copy
No points3D.* is copied: the init cloud is always passed explicitly with --initply.
"""
import os, glob, shutil, argparse

p = argparse.ArgumentParser()
p.add_argument("--gt", default=os.path.expanduser("~/perception/tools/4dgs/experiments/data/coffee_martini"))
p.add_argument("--colmap", default=os.path.expanduser("~/perception/tools/data/coffee_martini"))
p.add_argument("--out", default=os.path.expanduser("~/perception/tools/stgs/experiments/data/coffee_martini"))
p.add_argument("--frames", type=int, default=50)
a = p.parse_args()

cams = sorted(os.path.basename(d) for d in glob.glob(os.path.join(a.gt, "cam??")) if os.path.isdir(d))
assert len(cams) == 18 and cams[0] == "cam00", cams
os.makedirs(a.out, exist_ok=True)

dst = os.path.join(a.out, "poses_bounds.npy")
if not os.path.lexists(dst):
    os.symlink(os.path.realpath(os.path.join(a.colmap, "poses_bounds.npy")), dst)

sparse = os.path.join(a.out, "colmap_0", "sparse", "0")
os.makedirs(sparse, exist_ok=True)
for f in ("cameras.bin", "images.bin"):
    shutil.copy2(os.path.join(a.colmap, "colmap_0", "sparse", "0", f), sparse)

for j in range(a.frames):
    d = os.path.join(a.out, f"colmap_{j}", "images")
    os.makedirs(d, exist_ok=True)
    for c in cams:
        src = os.path.join(a.gt, c, "images", f"{j:04d}.png")
        assert os.path.exists(src), src
        dst = os.path.join(d, f"{c}.png")
        if not os.path.lexists(dst):
            os.symlink(src, dst)
print(f"{len(cams)} cams x {a.frames} frames linked into {a.out}")
