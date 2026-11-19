#!/usr/bin/env python3
"""Run: python experiments/scripts/finish_4dgs.py [--stages mask eval report]"""
import os, glob, csv, argparse
import numpy as np, cv2, torch, torch.nn.functional as F
from PIL import Image

R = os.path.expanduser("~/perception/tools/4dgs/experiments")
GT, REND = f"{R}/data/coffee_martini", f"{R}/renders/4dgs"
MASK, EVAL, FIG = f"{R}/masks", f"{R}/eval", f"{R}/figures"
TAU, MIN_PAIRS, DIL, MIN_BLOB, CROP, GAIN = 1.0, 3, 7, 300, 320, 4
METHOD, DEV = "4dgs", "cuda"
for d in (MASK, EVAL, FIG): os.makedirs(d, exist_ok=True)
cams = sorted(os.path.basename(d) for d in glob.glob(f"{GT}/cam*")
              if os.path.isdir(f"{d}/images") and len(os.listdir(f"{d}/images")) >= 50)
gtp = lambda cam, f: f"{GT}/{cam}/images/{f:04d}.png"

# ------------------------------------------------------------------ stage 1: mask
def stage_mask():
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(DEV).eval()
    stats = []
    for cam in cams:
        frames = [np.asarray(Image.open(gtp(cam, f)).convert("RGB")) for f in range(50)]
        H, W = frames[0].shape[:2]; Hp = (H + 7) // 8 * 8
        prep = lambda a: F.pad(torch.from_numpy(a).permute(2, 0, 1)[None].float().div(255) * 2 - 1,
                               (0, 0, 0, Hp - H), mode="replicate").to(DEV)
        count = np.zeros((H, W), np.int32); mags = []
        with torch.no_grad():
            for f in range(49):
                flow = model(prep(frames[f]), prep(frames[f + 1]))[-1][0, :, :H]
                mag = flow.norm(dim=0).cpu().numpy()
                count += (mag > TAU); mags.append(mag.astype(np.float16))
        raw = (count >= MIN_PAIRS).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
        keep = np.zeros_like(raw)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] >= MIN_BLOB: keep[lab == i] = 1
        m = cv2.dilate(keep, np.ones((DIL, DIL), np.uint8))
        Image.fromarray(m * 255).save(f"{MASK}/{cam}.png")
        stats.append((cam, float(m.mean())))
        if cam == "cam00": np.save(f"{MASK}/flowmags_cam00.npy", np.stack(mags))
        if cam in ("cam00", "cam01"):
            ov = frames[25].copy(); ov[m > 0] = (0.5 * ov[m > 0] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
            Image.fromarray(ov).save(f"{FIG}/mask_overlay_{cam}.png")
        print(f"mask {cam}: dynamic fraction {m.mean():.4f}", flush=True)
    with open(f"{MASK}/mask_stats.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["cam", "dynamic_fraction"]); w.writerows(stats)
    print(f"MASK DEF: RAFT raft_large, |flow|>{TAU}px in >={MIN_PAIRS}/49 pairs (t->t+1), "
          f"blobs<{MIN_BLOB}px removed, {DIL}x{DIL} dilation. Mean dynamic fraction "
          f"{np.mean([s[1] for s in stats]):.4f}")

# ------------------------------------------------------------------ stage 2: eval
def stage_eval():
    import lpips
    net = lpips.LPIPS(net="vgg", spatial=True, verbose=False).to(DEV).eval()
    x_ = torch.arange(11, dtype=torch.float32) - 5
    g = torch.exp(-x_**2 / (2 * 1.5**2)); g /= g.sum()
    Wn = (g[:, None] * g[None]).expand(3, 1, 11, 11).contiguous().to(DEV)
    def ssim_map(x, y):
        f = lambda t: F.conv2d(t, Wn, padding=5, groups=3)
        m1, m2 = f(x), f(y)
        s1, s2, s12 = f(x*x) - m1*m1, f(y*y) - m2*m2, f(x*y) - m1*m2
        C1, C2 = 0.01**2, 0.03**2
        return (((2*m1*m2 + C1) * (2*s12 + C2)) / ((m1*m1 + m2*m2 + C1) * (s1 + s2 + C2))).mean(1)[0]
    load = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB")).copy()).float().div(255).permute(2, 0, 1)[None].to(DEV)
    nan3 = (float("nan"),) * 3
    def stats(se, sm, lp, m):
        if int(m.sum()) == 0: return nan3
        return 10 * np.log10(1 / se[m].mean().item()), sm[m].mean().item(), lp[m].mean().item()
    rows = []
    for k in (2, 4, 8):
        last = (49 // k) * k
        sets = {"test": range(k // 2, last, k), "train": range(0, last + 1, k)}
        for cam in sorted(os.path.basename(d) for d in glob.glob(f"{REND}/k{k}/cam*") if os.path.isdir(d)):
            mp = f"{MASK}/{cam}.png"
            assert os.path.exists(mp), f"run the mask stage first ({mp})"
            mask = torch.from_numpy(np.asarray(Image.open(mp).convert("L")) > 127).to(DEV)
            for kind, frames in sets.items():
                for f in frames:
                    rp = f"{REND}/k{k}/{cam}/{f:04d}.png"
                    if not os.path.exists(rp): print("MISSING", rp); continue
                    x, y = load(rp), load(gtp(cam, f))
                    assert x.shape == y.shape, (rp, tuple(x.shape), tuple(y.shape))
                    with torch.no_grad():
                        se = ((x - y)**2).mean(1)[0]; sm = ssim_map(x, y); lp = net(x*2-1, y*2-1)[0, 0]
                    rows.append([METHOD, k, cam, f, kind,
                                 *stats(se, sm, lp, torch.ones_like(mask)),
                                 *stats(se, sm, lp, mask), *stats(se, sm, lp, ~mask)])
            print("eval done", k, cam, flush=True)
    with open(f"{EVAL}/per_image_{METHOD}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method","k","cam","frame","kind","psnr","ssim","lpips",
                    "psnr_dyn","ssim_dyn","lpips_dyn","psnr_sta","ssim_sta","lpips_sta"])
        w.writerows(rows)
    print("wrote", len(rows), "rows")

# ---------------------------------------------------------------- stage 3: report
def stage_report():
    import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = pd.read_csv(f"{EVAL}/per_image_{METHOD}.csv")
    d["setting"] = np.where(d.cam == "cam00", "cam00", "traincams")
    pd.set_option("display.width", 200)
    cols = [c for c in d.columns if c.startswith(("psnr", "ssim", "lpips"))]
    g = d.groupby(["kind", "setting", "k"])[cols].mean().round(4)
    g.to_csv(f"{EVAL}/summary_{METHOD}.csv"); print(g.to_string())
    # LaTeX rows (test frames): full / dynamic / static
    with open(f"{EVAL}/latex_rows_{METHOD}.txt", "w") as fh:
        for suf, name in (("", "FULL"), ("_dyn", "DYNAMIC"), ("_sta", "STATIC")):
            fh.write(f"% {name}\n")
            for k in (2, 4, 8):
                v = [g.loc[("test", s, k)][m + suf] for s in ("traincams", "cam00") for m in ("psnr", "ssim", "lpips")]
                fh.write(f"& $k = {k}$ & {v[0]:.2f} & {v[1]:.4f} & {v[2]:.4f} & {v[3]:.2f} & {v[4]:.4f} & {v[5]:.4f} \\\\\n")
    print("LaTeX rows ->", f"{EVAL}/latex_rows_{METHOD}.txt")
    # per-frame PSNR, k=8
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for a, (title, sel) in zip(ax, (("Training cameras", d.cam != "cam00"), ("Held-out cam00", d.cam == "cam00"))):
        s = d[(d.k == 8) & sel].groupby(["frame", "kind"])["psnr"].mean().reset_index().sort_values("frame")
        a.plot(s.frame, s.psnr, "-", color="gray", lw=1)
        for kind, mk, lab in (("train", "o", "train frames"), ("test", "x", "test (interpolated)")):
            t = s[s.kind == kind]; a.scatter(t.frame, t.psnr, marker=mk, s=40, label=lab, zorder=3)
        a.set_title(f"4DGS k=8, {title}"); a.set_xlabel("frame"); a.set_ylabel("PSNR (dB)"); a.legend()
    plt.tight_layout(); plt.savefig(f"{FIG}/psnr_k8_{METHOD}.png", dpi=150)
    for name, sel in (("traincams", d.cam != "cam00"), ("cam00", d.cam == "cam00")):  # individual files for LaTeX
        s = d[(d.k == 8) & sel].groupby(["frame", "kind"])["psnr"].mean().reset_index().sort_values("frame")
        plt.figure(figsize=(5.5, 3.6)); plt.plot(s.frame, s.psnr, "-", color="gray", lw=1)
        for kind, mk in (("train", "o"), ("test", "x")):
            t = s[s.kind == kind]; plt.scatter(t.frame, t.psnr, marker=mk, s=40, label=kind, zorder=3)
        plt.xlabel("frame"); plt.ylabel("PSNR (dB)"); plt.legend(); plt.tight_layout()
        plt.savefig(f"{FIG}/psnr_k8_{name}_{METHOD}.png", dpi=150); plt.close()
    # crops on the fastest-moving region of cam00: GT | render | error (x GAIN)
    mp = f"{MASK}/flowmags_cam00.npy"
    if os.path.exists(mp):
        mags = np.load(mp).astype(np.float32); H, W = mags.shape[1:]
        sm = cv2.blur(mags.max(0), (CROP, CROP))
        sm[:CROP // 2] = 0; sm[-CROP // 2:] = 0; sm[:, :CROP // 2] = 0; sm[:, -CROP // 2:] = 0
        cy, cx = np.unravel_index(sm.argmax(), sm.shape)
        y0, x0 = cy - CROP // 2, cx - CROP // 2; sl = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
        print(f"crop window cam00: x {x0}..{x0 + CROP}, y {y0}..{y0 + CROP}")
        rows = []
        for k in (2, 4, 8):
            tf = list(range(k // 2, (49 // k) * k, k))
            f = max(tf, key=lambda t: mags[t][sl].mean())
            r = cv2.imread(f"{REND}/k{k}/cam00/{f:04d}.png")[sl]; gtc = cv2.imread(gtp("cam00", f))[sl]
            e = cv2.applyColorMap(np.clip(np.abs(r.astype(int) - gtc.astype(int)).mean(2) * GAIN, 0, 255).astype("uint8"), cv2.COLORMAP_JET)
            row = np.hstack([gtc, r, e]); cv2.putText(row, f"k={k} frame {f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            rows.append(row)
        cv2.imwrite(f"{FIG}/crops_cam00_{METHOD}.png", np.vstack(rows))
        print("crops ->", f"{FIG}/crops_cam00_{METHOD}.png (GT | render | error x{GAIN})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stages", nargs="+", default=["mask", "eval", "report"])
    a = ap.parse_args()
    if "mask" in a.stages: stage_mask()
    if "eval" in a.stages: stage_eval()
    if "report" in a.stages: stage_report()