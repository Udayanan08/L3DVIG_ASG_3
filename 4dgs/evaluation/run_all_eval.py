#!/usr/bin/env python3
"""
One script for the whole evaluation of Assignment 3 (all methods, same GT, same mask, same metrics).

Run:   python run_all_eval.py                       # everything
       python run_all_eval.py --stages eval report  # only some stages (mask | eval | report)
       python run_all_eval.py --force               # recompute even if outputs exist

Stages
  mask   RAFT on consecutive GT frames -> ONE static mask per camera (shared by every method and k).
         |flow| > 1.0 px in >= 3 of 49 pairs, blobs < 300 px removed, 7x7 dilation.
  eval   PSNR / SSIM / LPIPS(VGG) per image: full, dynamic (mask), static (~mask).
         Test frames and train frames, all cameras, k in {2,4,8}, every method found on disk.
  report summary tables, LaTeX rows, per-frame PSNR plot (k=8), crops on a fast-moving region,
         mask overlays, arbitrary-t continuity numbers.

Layout assumed (override with flags):
  ROOT = ~/perception/tools
  GT      ROOT/4dgs/experiments/data/coffee_martini/camXX/images/NNNN.png   (1352x1014, frames 0-49)
  renders ROOT/<m>/experiments/renders/<m>/k{2,4,8}/camXX/NNNN.png          for m in 4dgs, stgs, ex4dgs
  output  ROOT/evaluation/final/   (nothing in your existing evaluation/ files is touched)

Needs: torch, torchvision>=0.12 (RAFT), lpips, opencv-python, pillow, numpy, pandas, matplotlib.
"""
import os, glob, csv, json, argparse
import numpy as np, cv2, torch, torch.nn.functional as F
from PIL import Image

ROOT = os.path.expanduser("~/perception/tools")
ap = argparse.ArgumentParser()
ap.add_argument("--root", default=ROOT)
ap.add_argument("--gt", default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--methods", nargs="+", default=["4dgs", "stgs", "ex4dgs"])
ap.add_argument("--renders", nargs="*", default=[], help="override: method=/path/to/renders (dir containing k2 k4 k8)")
ap.add_argument("--stages", nargs="+", default=["mask", "eval", "report"], choices=["mask", "eval", "report"])
ap.add_argument("--force", action="store_true")
ap.add_argument("--tau", type=float, default=1.0)
ap.add_argument("--min_pairs", type=int, default=3)
ap.add_argument("--dilate", type=int, default=7)
ap.add_argument("--min_blob", type=int, default=300)
ap.add_argument("--crop", type=int, default=320)
ap.add_argument("--gain", type=int, default=4)
A = ap.parse_args()

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GT = A.gt or f"{A.root}/4dgs/experiments/data/coffee_martini"
OUT = A.out or f"{A.root}/evaluation/final"
MASK, EVAL, FIG = f"{OUT}/masks", f"{OUT}/eval", f"{OUT}/figures"
for d in (MASK, EVAL, FIG):
    os.makedirs(d, exist_ok=True)
REND = {m: f"{A.root}/{m}/experiments/renders/{m}" for m in A.methods}
for s in A.renders:
    m, p = s.split("=", 1)
    REND[m] = os.path.expanduser(p)
REND = {m: p for m, p in REND.items() if os.path.isdir(p)}
KS = (2, 4, 8)
gtp = lambda cam, f: f"{GT}/{cam}/images/{f:04d}.png"
cams = sorted(os.path.basename(d) for d in glob.glob(f"{GT}/cam*")
              if os.path.isdir(f"{d}/images") and len(glob.glob(f"{d}/images/*.png")) >= 50)
assert cams, f"no GT frames found under {GT}/camXX/images (run extract_frames.py first)"
print(f"GT: {GT} | {len(cams)} cameras | methods found: {list(REND)} | out: {OUT} | device: {DEV}")


def test_frames(k):  # midpoints strictly between two training frames
    return list(range(k // 2, (49 // k) * k, k))


def train_frames(k):
    return list(range(0, (49 // k) * k + 1, k))


# ------------------------------------------------------------------------ stage 1: mask
def stage_mask():
    if not A.force and all(os.path.exists(f"{MASK}/{c}.png") for c in cams):
        print("mask: all masks exist, skipping (use --force to recompute)")
        return
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(DEV).eval()
    stats = []
    for cam in cams:
        frames = [np.asarray(Image.open(gtp(cam, f)).convert("RGB")).copy() for f in range(50)]
        H, W = frames[0].shape[:2]
        Hp, Wp = (H + 7) // 8 * 8, (W + 7) // 8 * 8
        prep = lambda a: F.pad(torch.from_numpy(a).permute(2, 0, 1)[None].float().div(255) * 2 - 1,
                               (0, Wp - W, 0, Hp - H), mode="replicate").to(DEV)
        count = np.zeros((H, W), np.int32)
        mags = []
        with torch.no_grad():
            for f in range(49):
                flow = model(prep(frames[f]), prep(frames[f + 1]))[-1][0, :, :H, :W]
                mag = flow.norm(dim=0).cpu().numpy()
                count += (mag > A.tau)
                mags.append(mag.astype(np.float16))
        raw = (count >= A.min_pairs).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
        keep = np.zeros_like(raw)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] >= A.min_blob:
                keep[lab == i] = 1
        m = cv2.dilate(keep, np.ones((A.dilate, A.dilate), np.uint8))
        Image.fromarray(m * 255).save(f"{MASK}/{cam}.png")
        drift = float(np.abs(frames[0].astype(np.float32) - frames[49].astype(np.float32)).mean())
        stats.append((cam, float(m.mean()), drift))
        if cam == "cam00":
            np.save(f"{MASK}/flowmags_cam00.npy", np.stack(mags))
        if cam in ("cam00", "cam01"):
            ov = frames[25].copy()
            ov[m > 0] = (0.5 * ov[m > 0] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
            Image.fromarray(ov).save(f"{FIG}/mask_overlay_{cam}.png")
        print(f"mask {cam}: dynamic fraction {m.mean():.4f} | mean|frame0-frame49| {drift:.2f}", flush=True)
    with open(f"{MASK}/mask_stats.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cam", "dynamic_fraction", "mean_abs_diff_frame0_frame49"])
        w.writerows(stats)
    fr = [s[1] for s in stats]
    desc = (f"RAFT raft_large, |flow|>{A.tau}px in >={A.min_pairs}/49 pairs (t->t+1), blobs<{A.min_blob}px removed, "
            f"{A.dilate}x{A.dilate} dilation. Mean dynamic fraction {np.mean(fr):.4f} (min {min(fr):.4f}, max {max(fr):.4f})")
    json.dump({"definition": desc}, open(f"{MASK}/params.json", "w"))
    print("MASK DEF:", desc)
    print("CHECK: a camera with dynamic fraction ~0 and tiny frame0-vs-frame49 difference is frozen/static (cam13?).")


# ------------------------------------------------------------------------ stage 2: eval
def stage_eval():
    import lpips
    net = lpips.LPIPS(net="vgg", spatial=True, verbose=False).to(DEV).eval()
    x_ = torch.arange(11, dtype=torch.float32) - 5
    g = torch.exp(-x_ ** 2 / (2 * 1.5 ** 2))
    g /= g.sum()
    Wn = (g[:, None] * g[None]).expand(3, 1, 11, 11).contiguous().to(DEV)

    def ssim_map(x, y):
        f = lambda t: F.conv2d(t, Wn, padding=5, groups=3)
        m1, m2 = f(x), f(y)
        s1, s2, s12 = f(x * x) - m1 * m1, f(y * y) - m2 * m2, f(x * y) - m1 * m2
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        return (((2 * m1 * m2 + C1) * (2 * s12 + C2)) / ((m1 * m1 + m2 * m2 + C1) * (s1 + s2 + C2))).mean(1)[0]

    load = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB")).copy()).float().div(255).permute(2, 0, 1)[None].to(DEV)
    nan3 = (float("nan"),) * 3

    def stats(se, sm, lp, m):
        if int(m.sum()) == 0:
            return nan3
        return 10 * np.log10(1 / max(se[m].mean().item(), 1e-12)), sm[m].mean().item(), lp[m].mean().item()

    masks = {c: torch.from_numpy(np.asarray(Image.open(f"{MASK}/{c}.png").convert("L")) > 127).to(DEV) for c in cams}
    for method, root in REND.items():
        out = f"{EVAL}/per_image_{method}.csv"
        if os.path.exists(out) and not A.force:
            print(f"eval {method}: {out} exists, skipping (use --force)")
            continue
        rows, missing = [], 0
        for k in KS:
            for cam in sorted(os.path.basename(d) for d in glob.glob(f"{root}/k{k}/cam*") if os.path.isdir(d)):
                if cam not in masks:
                    continue
                for kind, frames in (("test", test_frames(k)), ("train", train_frames(k))):
                    for f in frames:
                        rp = f"{root}/k{k}/{cam}/{f:04d}.png"
                        if not os.path.exists(rp):
                            missing += 1
                            print("MISSING", rp)
                            continue
                        x, y = load(rp), load(gtp(cam, f))
                        assert x.shape == y.shape, f"{rp}: render {tuple(x.shape)} vs GT {tuple(y.shape)} (must both be 1352x1014)"
                        with torch.no_grad():
                            se = ((x - y) ** 2).mean(1)[0]
                            sm = ssim_map(x, y)
                            lp = net(x * 2 - 1, y * 2 - 1)[0, 0]
                        m = masks[cam]
                        rows.append([method, k, cam, f, kind, *stats(se, sm, lp, torch.ones_like(m)),
                                     *stats(se, sm, lp, m), *stats(se, sm, lp, ~m)])
                print(f"eval {method} k={k} {cam} done", flush=True)
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["method", "k", "cam", "frame", "kind", "psnr", "ssim", "lpips", "psnr_dyn", "ssim_dyn",
                        "lpips_dyn", "psnr_sta", "ssim_sta", "lpips_sta"])
            w.writerows(rows)
        print(f"eval {method}: wrote {len(rows)} rows ({missing} missing files) -> {out}")


# ------------------------------------------------------------------------ stage 3: report
def stage_report():
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = sorted(glob.glob(f"{EVAL}/per_image_*.csv"))
    assert files, "no per_image_*.csv found: run the eval stage first"
    d = pd.concat(pd.read_csv(f) for f in files)
    methods = [m for m in A.methods if m in set(d.method)] + [m for m in sorted(set(d.method)) if m not in A.methods]
    d["setting"] = np.where(d.cam == "cam00", "cam00", "traincams")
    pd.set_option("display.width", 220)
    cols = [c for c in d.columns if c.startswith(("psnr", "ssim", "lpips"))]
    g = d.groupby(["method", "kind", "setting", "k"])[cols].mean().round(4)
    g.to_csv(f"{EVAL}/summary_all.csv")
    print(g.to_string())

    # frames counted per group (sanity check that every method was scored on the same images)
    cnt = d[d.kind == "test"].groupby(["method", "setting", "k"]).size().unstack("method")
    print("\n#test images per (setting,k):\n", cnt.to_string())

    # LaTeX rows
    with open(f"{EVAL}/latex_rows.txt", "w") as fh:
        for suf, name in (("", "FULL"), ("_dyn", "DYNAMIC"), ("_sta", "STATIC")):
            fh.write(f"% ===== {name} (test frames; cols: traincams P/S/L, cam00 P/S/L)\n")
            for m in methods:
                fh.write(f"% {m}\n")
                for k in KS:
                    try:
                        v = [g.loc[(m, "test", s, k)][q + suf] for s in ("traincams", "cam00") for q in ("psnr", "ssim", "lpips")]
                    except KeyError:
                        continue
                    fh.write(f"& $k = {k}$ & {v[0]:.2f} & {v[1]:.4f} & {v[2]:.4f} & {v[3]:.2f} & {v[4]:.4f} & {v[5]:.4f} \\\\\n")
        fh.write("% ===== SPATIAL CONTROL: full image at TRAIN frames (cols: traincams P/S/L, cam00 P/S/L)\n")
        for m in methods:
            fh.write(f"% {m}\n")
            for k in KS:
                try:
                    v = [g.loc[(m, "train", s, k)][q] for s in ("traincams", "cam00") for q in ("psnr", "ssim", "lpips")]
                except KeyError:
                    continue
                fh.write(f"{k} & {v[0]:.2f} & {v[1]:.4f} & {v[2]:.4f} & {v[3]:.2f} & {v[4]:.4f} & {v[5]:.4f} \\\\\n")
    print("LaTeX rows ->", f"{EVAL}/latex_rows.txt")

    # per-frame PSNR for k=8: one line per method, train frames = circles, test frames = crosses
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for name, sel in (("traincams", d.cam != "cam00"), ("cam00", d.cam == "cam00")):
        plt.figure(figsize=(6.2, 3.8))
        for i, m in enumerate(methods):
            s = d[(d.k == 8) & sel & (d.method == m)].groupby(["frame", "kind"])["psnr"].mean().reset_index().sort_values("frame")
            if s.empty:
                continue
            plt.plot(s.frame, s.psnr, "-", color=colors[i % 10], lw=1, alpha=0.6)
            t = s[s.kind == "train"]
            plt.scatter(t.frame, t.psnr, marker="o", s=34, color=colors[i % 10], label=f"{m} train frames", zorder=3)
            t = s[s.kind == "test"]
            plt.scatter(t.frame, t.psnr, marker="x", s=44, color=colors[i % 10], label=f"{m} test frames", zorder=3)
        plt.xlabel("frame")
        plt.ylabel("PSNR (dB)")
        plt.title(f"k=8, {'training cameras' if name == 'traincams' else 'held-out cam00'}")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(f"{FIG}/psnr_k8_{name}.png", dpi=150)
        plt.close()
    print("per-frame plots ->", f"{FIG}/psnr_k8_traincams.png, psnr_k8_cam00.png")

    # crops on the fastest-moving region of cam00: GT | render per method | error per method
    mp = f"{MASK}/flowmags_cam00.npy"
    if os.path.exists(mp):
        mags = np.load(mp).astype(np.float32)
        C = A.crop
        sm = cv2.blur(mags.max(0), (C, C))
        sm[:C // 2] = 0
        sm[-C // 2:] = 0
        sm[:, :C // 2] = 0
        sm[:, -C // 2:] = 0
        cy, cx = np.unravel_index(sm.argmax(), sm.shape)
        y0, x0 = cy - C // 2, cx - C // 2
        sl = (slice(y0, y0 + C), slice(x0, x0 + C))
        print(f"crop window cam00: x {x0}..{x0 + C}, y {y0}..{y0 + C}")
        rows, used = [], {}
        for k in KS:
            tf = test_frames(k)
            f = max(tf, key=lambda t: mags[t][sl].mean())  # test frame with most motion inside the window
            used[k] = f
            gtc = cv2.imread(gtp("cam00", f))[sl]
            ims, errs = [gtc], []
            for m in methods:
                p = f"{REND[m]}/k{k}/cam00/{f:04d}.png" if m in REND else ""
                r = cv2.imread(p)[sl] if os.path.exists(p) else np.zeros_like(gtc)
                ims.append(r)
                e = np.clip(np.abs(r.astype(int) - gtc.astype(int)).mean(2) * A.gain, 0, 255).astype("uint8")
                errs.append(cv2.applyColorMap(e, cv2.COLORMAP_JET))
            row = np.hstack(ims + errs)
            cv2.putText(row, f"k={k} frame {f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            for j, lab in enumerate(["GT"] + [f"{m}" for m in methods] + [f"err {m}" for m in methods]):
                cv2.putText(row, lab, (j * C + 8, C - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            rows.append(row)
        cv2.imwrite(f"{FIG}/crops_cam00_all.png", np.vstack(rows))
        print(f"crops -> {FIG}/crops_cam00_all.png | frames used per k: {used} | columns: GT | " +
              " | ".join(methods) + f" | error x{A.gain} per method")

    # arbitrary-t continuity: how much does a 1e-4 change of t move the render vs a 0.07 change?
    for m, root in REND.items():
        for k in KS:
            ad = f"{root}/k{k}/arbitrary_t"
            ps = {t: f"{ad}/cam00_t{t:.4f}.png" for t in (0.30, 0.37, 0.3701, 0.44)}
            if all(os.path.exists(p) for p in ps.values()):
                im = {t: cv2.imread(p).astype(np.float32) for t, p in ps.items()}
                d1 = np.abs(im[0.37] - im[0.3701]).mean()
                d2 = np.abs(im[0.37] - im[0.44]).mean()
                print(f"arbitrary-t {m} k={k}: mean|I(0.37)-I(0.3701)|={d1:.4f}  mean|I(0.37)-I(0.44)|={d2:.4f}  (0-255 scale)")

    # mask overlays / stats
    sp = f"{MASK}/mask_stats.csv"
    if os.path.exists(sp):
        s = pd.read_csv(sp)
        print("\nmask stats (dynamic fraction, mean|frame0-frame49|):")
        print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    if "mask" in A.stages:
        stage_mask()
    if "eval" in A.stages:
        stage_eval()
    if "report" in A.stages:
        stage_report()
    print("\nDONE. Outputs in", OUT)
