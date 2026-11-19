import os, sys, cv2, numpy as np
k = int(sys.argv[1]); cams = sys.argv[2:] or ["cam00"]
R = os.path.expanduser("~/perception/tools/4dgs/experiments")
last_train = (49 // k) * k
test_frames = [f for f in range(k // 2, last_train, k)]
for cam in cams:
    out = f"{R}/figures/compare/k{k}/{cam}"; os.makedirs(out, exist_ok=True)
    for f in test_frames:
        r = cv2.imread(f"{R}/renders/4dgs/k{k}/{cam}/{f:04d}.png")
        g = cv2.imread(f"{R}/data/coffee_martini/{cam}/images/{f:04d}.png")
        if r is None or g is None: print("missing", cam, f); continue
        e = np.clip(np.abs(r.astype(int) - g.astype(int)).mean(2) * 4, 0, 255).astype("uint8")
        cv2.imwrite(f"{out}/{f:04d}.png", np.hstack([r, g, cv2.applyColorMap(e, cv2.COLORMAP_JET)]))
    print(f"k={k} {cam}: {len(test_frames)} test frames -> {out}")
