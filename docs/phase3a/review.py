"""Group a run's detections into per-vehicle tracks (analysis only), measure
how stable the prediction is across a track, and draw contact sheets of one
representative (largest) crop per track for by-eye review.

usage: review.py NAME [--min-w PX]
"""
import argparse, json
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--min-w", type=int, default=0)
a = ap.parse_args()
run = HERE / "runs" / a.name
R = json.load(open(run / "results.json"))["rows"]


def iou(p, q):
    ix = max(0, min(p[2], q[2]) - max(p[0], q[0])); iy = max(0, min(p[3], q[3]) - max(p[1], q[1]))
    inter = ix * iy
    u = (p[2]-p[0])*(p[3]-p[1]) + (q[2]-q[0])*(q[3]-q[1]) - inter
    return inter / u if u else 0


# Greedy IoU linking between consecutive sampled frames.
frames = sorted({r["frame"] for r in R})
tracks, live = [], []
for f in frames:
    cur = [r for r in R if r["frame"] == f]
    new_live, used = [], set()
    for r in sorted(cur, key=lambda r: -r["w"] * r["h"]):
        best, bi = 0.3, None
        for ti in live:
            if ti in used: continue
            v = iou(tracks[ti][-1]["bbox"], r["bbox"])
            if v > best: best, bi = v, ti
        if bi is None:
            tracks.append([r]); bi = len(tracks) - 1
        else:
            tracks[bi].append(r)
        used.add(bi); new_live.append(bi)
    live = new_live

make = lambda r: r["top5"][0][0].split()[0].lower().replace("-benz", "")
model_ = lambda r: " ".join(r["top5"][0][0].split()[:-1]).lower()
summary = []
for ti, t in enumerate(tracks):
    rep = max(t, key=lambda r: r["w"] * r["h"])
    mk = Counter(make(r) for r in t); md = Counter(model_(r) for r in t)
    summary.append(dict(track=ti, n=len(t), rep=rep["k"], w=rep["w"], h=rep["h"], cls=rep["cls"],
                        top1=rep["top5"][0], top5=rep["top5"],
                        make_agree=mk.most_common(1)[0][1] / len(t),
                        model_agree=md.most_common(1)[0][1] / len(t),
                        makes=dict(mk.most_common(4))))
json.dump(summary, open(run / "tracks.json", "w"), indent=0)

multi = [s for s in summary if s["n"] >= 3]
print(f"{a.name}: {len(R)} detections, {len(tracks)} tracks, {len(multi)} with >=3 detections")
if multi:
    print(" mean top-1 make agreement within track (n>=3):", round(sum(s['make_agree'] for s in multi)/len(multi), 2),
          " model agreement:", round(sum(s['model_agree'] for s in multi)/len(multi), 2))

# Contact sheet: representative crops, largest first.
sel = sorted([s for s in summary if s["w"] >= a.min_w], key=lambda s: -s["w"] * s["h"])
font = ImageFont.load_default(size=13)
CW, CH, TH, COLS = 300, 190, 58, 4
for page in range(0, len(sel), 16):
    chunk = sel[page:page+16]
    rows = (len(chunk) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS*CW, rows*(CH+TH)), "white"); d = ImageDraw.Draw(sheet)
    for j, s in enumerate(chunk):
        im = Image.open(run / "crops" / f"{s['rep']:05d}.jpg"); im.thumbnail((CW-6, CH-6))
        x, y = (j % COLS)*CW, (j // COLS)*(CH+TH)
        sheet.paste(im, (x+3, y+3))
        d.text((x+3, y+CH), f"T{s['track']} {s['cls']} {s['w']}x{s['h']} n={s['n']}", fill="black", font=font)
        d.text((x+3, y+CH+15), f"{s['top1'][0]} {s['top1'][1]:.2f}", fill="blue", font=font)
        d.text((x+3, y+CH+30), f"#2 {s['top5'][1][0]} {s['top5'][1][1]:.2f}", fill="gray", font=font)
    sheet.save(run / f"sheet_{page//16:02d}.jpg", quality=88)
print(" sheets:", (len(sel)+15)//16, "reps with w >=", a.min_w, ":", len(sel))
