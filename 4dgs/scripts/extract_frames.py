import os, glob, argparse, cv2
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--datadir", default=os.path.expanduser("~/perception/tools/4dgs/experiments/data/coffee_martini"))
p.add_argument("--frames", type=int, default=50)
a = p.parse_args()
W, H = 1352, 1014
for v in sorted(glob.glob(os.path.join(a.datadir, "cam*.mp4"))):
    out = os.path.join(os.path.splitext(os.path.abspath(v))[0], "images")  # no realpath: stays in our folder
    if os.path.isdir(out) and len(glob.glob(out + "/*.png")) == a.frames:
        print("skip", v); continue
    os.makedirs(out, exist_ok=True)
    cap = cv2.VideoCapture(v)
    assert (cap.get(3), cap.get(4)) == (2704.0, 2028.0), v
    for i in range(a.frames):
        ok, f = cap.read(); assert ok, (v, i)
        img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).resize((W, H), Image.LANCZOS)
        img.save(os.path.join(out, "%04d.png" % i))
    cap.release(); print("done", v)
