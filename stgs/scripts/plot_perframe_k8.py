"""Per-frame PSNR plots for k=8 with only the methods in the report (default 4DGS and STGS).
Same data and style as run_all_eval.py's report stage (circles = train frames, crosses = test frames),
read from final/eval/per_image_<method>.csv (4DGS per-camera masks, shared GT).

python plot_perframe_k8.py                      # -> figures/psnr_k8_traincams.png, figures/psnr_k8_cam00.png
python plot_perframe_k8.py --methods 4dgs stgs --k 8 --out figures
"""
import os, argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--methods", nargs="+", default=["4dgs", "stgs"])
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--eval", default=os.path.join(HERE, "final", "eval"))
ap.add_argument("--out", default=os.path.join(HERE, "figures"))
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)

LABEL = {"4dgs": "4DGS", "stgs": "STGS", "ex4dgs": "Ex4DGS"}
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
d = pd.concat(pd.read_csv(os.path.join(A.eval, f"per_image_{m}.csv")) for m in A.methods)

for name, sel, title in (("traincams", d.cam != "cam00", "training cameras"), ("cam00", d.cam == "cam00", "held-out cam00")):
    plt.figure(figsize=(6.2, 3.8))
    for i, m in enumerate(A.methods):
        s = d[(d.k == A.k) & sel & (d.method == m)].groupby(["frame", "kind"])["psnr"].mean().reset_index().sort_values("frame")
        assert not s.empty, f"no k={A.k} rows for {m}"
        c, lab = colors[i % 10], LABEL.get(m, m)
        plt.plot(s.frame, s.psnr, "-", color=c, lw=1, alpha=0.6)
        t = s[s.kind == "train"]
        plt.scatter(t.frame, t.psnr, marker="o", s=34, color=c, label=f"{lab} train frames", zorder=3)
        t = s[s.kind == "test"]
        plt.scatter(t.frame, t.psnr, marker="x", s=44, color=c, label=f"{lab} test frames", zorder=3)
        print(f"{name:9s} {lab:5s} " + " ".join(f"{int(f)}:{p:.2f}{'*' if k == 'test' else ''}" for f, k, p in s.values))
    plt.xlabel("frame")
    plt.ylabel("PSNR (dB)")
    plt.title(f"k={A.k}, {title}")
    plt.legend(fontsize=7)
    plt.tight_layout()
    p = os.path.join(A.out, f"psnr_k{A.k}_{name}.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print("->", p)
print("(* = test frame)")
