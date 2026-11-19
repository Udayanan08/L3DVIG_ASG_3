"""Visual result for one method, in the same format as the 4DGS team's crops_cam00_4dgs.png:
one row per k, columns GT | render | |error| x GAIN (JET colormap), on cam00, on the fast-moving crop.

Crop window and frames are computed exactly like run_all_eval.py / finish_4dgs.py (fastest-moving 320x320
window of cam00 from the RAFT flow magnitudes; per k, the test frame with most motion in the window),
so they match the 4DGS figure (x 488..808, y 415..735; frames k=2:1, k=4:2, k=8:4).

python crops_method.py --method stgs      # -> figures/crops_cam00_stgs.png
"""
import os, argparse
import numpy as np, cv2

T = os.path.expanduser("~/perception/tools")
ap = argparse.ArgumentParser()
ap.add_argument("--method", default="stgs")
ap.add_argument("--renders", default=None, help="dir with k2 k4 k8 (default <tools>/<m>/experiments/renders/<m>)")
ap.add_argument("--gt", default=f"{T}/4dgs/experiments/data/coffee_martini")
ap.add_argument("--flowmags", default=f"{T}/4dgs/experiments/masks/flowmags_cam00.npy")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"))
ap.add_argument("--crop", type=int, default=320)
ap.add_argument("--gain", type=int, default=4)
A = ap.parse_args()
R = A.renders or f"{T}/{A.method}/experiments/renders/{A.method}"
os.makedirs(A.out, exist_ok=True)
LABEL = {"4dgs": "4DGS", "stgs": "STGS", "ex4dgs": "Ex4DGS"}.get(A.method, A.method)

mags = np.load(A.flowmags).astype(np.float32)
C = A.crop
sm = cv2.blur(mags.max(0), (C, C))
sm[:C // 2] = 0; sm[-C // 2:] = 0; sm[:, :C // 2] = 0; sm[:, -C // 2:] = 0
cy, cx = np.unravel_index(sm.argmax(), sm.shape)
y0, x0 = cy - C // 2, cx - C // 2
sl = (slice(y0, y0 + C), slice(x0, x0 + C))
print(f"crop window cam00: x {x0}..{x0 + C}, y {y0}..{y0 + C}")

rows = []
for k in (2, 4, 8):
    tf = list(range(k // 2, (49 // k) * k, k))
    f = max(tf, key=lambda t: mags[t][sl].mean())
    gt = cv2.imread(f"{A.gt}/cam00/images/{f:04d}.png")
    rp = f"{R}/k{k}/cam00/{f:04d}.png"
    r = cv2.imread(rp)
    assert r is not None and r.shape == gt.shape, f"missing or wrong size: {rp}"
    gtc, rc = gt[sl], r[sl]
    err = np.abs(rc.astype(int) - gtc.astype(int)).mean(2)
    e = cv2.applyColorMap(np.clip(err * A.gain, 0, 255).astype("uint8"), cv2.COLORMAP_JET)
    row = np.hstack([gtc, rc, e])
    cv2.putText(row, f"k={k} frame {f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for j, lab in enumerate(["GT", LABEL, f"|error| x{A.gain}"]):
        cv2.putText(row, lab, (j * C + 8, C - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    psnr = 10 * np.log10(255.0 ** 2 / max(((rc.astype(float) - gtc) ** 2).mean(), 1e-9))
    print(f"k={k}: frame {f}, crop PSNR {psnr:.2f} dB, mean |err| {err.mean():.2f}/255")
    rows.append(row)
p = f"{A.out}/crops_cam00_{A.method}.png"
cv2.imwrite(p, np.vstack(rows))
print("->", p)
