import os, glob, numpy as np, cv2
from plyfile import PlyData
root = os.path.expanduser("~/perception/tools/4dgs/experiments/data/coffee_martini")
figdir = os.path.expanduser("~/perception/tools/4dgs/experiments/figures"); os.makedirs(figdir, exist_ok=True)
names = [os.path.basename(p)[:-4] for p in sorted(glob.glob(root + "/cam*.mp4"))]
pa = np.load(root + "/poses_bounds.npy")
poses = pa[:, :-2].reshape(-1, 3, 5)
focal = poses[0, 2, -1] / 2.0
poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)  # as in loader
W, H = 1352, 1014
v = PlyData.read(root + "/points3D_sparse_frame0.ply")["vertex"]
xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
col = np.stack([v["red"], v["green"], v["blue"]], 1) / 255.0
for i in [1, 9, 17]:
    P = poses[i]; R = -P[:3, :3]; R[:, 0] = -R[:, 0]; T = -P[:3, 3].dot(R)
    Xc = xyz @ R + T
    z = Xc[:, 2]; z_safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = focal * Xc[:, 0] / z_safe + W / 2; w = focal * Xc[:, 1] / z_safe + H / 2
    ok = (z > 0) & (u >= 0) & (u < W) & (w >= 0) & (w < H)
    img = cv2.cvtColor(cv2.imread(f"{root}/{names[i]}/images/0000.png"), cv2.COLOR_BGR2RGB)
    ui, wi = u[ok].astype(int), w[ok].astype(int)
    sampled = img[wi, ui] / 255.0
    err = np.abs(sampled - col[ok]).mean()
    rng = np.random.default_rng(0)
    base = np.abs(sampled - col[ok][rng.permutation(ok.sum())]).mean()
    print(f"{names[i]}: in-image {ok.sum()}/{len(xyz)} | color err {err:.3f} vs shuffled {base:.3f}")
    d = z[ok]; dn = ((d - d.min()) / (np.ptp(d) + 1e-9) * 255).astype(np.uint8)
    cm = cv2.applyColorMap(dn.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
    vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
    for x, y, c in zip(ui, wi, cm):
        cv2.circle(vis, (int(x), int(y)), 3, tuple(int(q) for q in c), -1)
    cv2.imwrite(f"{figdir}/projection_{names[i]}.png", vis)
