"""Stage 3a harness: every vehicle detection in a clip -> make/model top-5.

Uses the project's own Detector + sampling so the detections are exactly
the ones the pipeline would alert on (whole frame, vehicle classes).

usage: run_clip.py CLIP NAME [--sample-fps N] [--max-seconds N]
"""
import argparse, json, sys, time
from pathlib import Path

import cv2, numpy as np, torch, timm
from PIL import Image
from torchvision import transforms

sys.path.insert(0, "/Users/danemmanuel/Documents/Perimeter-anomaly/backend")
from app.services.detector import Detector
from app.services.clip_processor import VEHICLE_CLASSES, should_sample

HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("clip"); ap.add_argument("name")
ap.add_argument("--sample-fps", type=float, default=2.0)
ap.add_argument("--max-seconds", type=float, default=60.0)
a = ap.parse_args()

ck = torch.load(HERE / "vehicle_classifier.pth", map_location="cpu", weights_only=True)
names = ck["class_mapping"]
model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=len(names))
model.load_state_dict(ck["model_state"]); model.eval()
# Model card's preprocessing (squash to 380x380). Checkpoint config's eval is
# resize 400 + centre crop 380; compared separately.
tf = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

det = Detector(); det.load()
import os
if os.environ.get("CLS_THREADS"): torch.set_num_threads(int(os.environ["CLS_THREADS"]))
out = HERE / "runs" / a.name; (out / "crops").mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(a.clip); fps = cap.get(cv2.CAP_PROP_FPS)
rows, i = [], -1
with torch.inference_mode():
    model(tf(Image.new("RGB", (200, 200))).unsqueeze(0))  # warm-up
    while True:
        ok, frame = cap.read(); i += 1
        if not ok or i / fps > a.max_seconds: break
        if a.sample_fps > 0 and not should_sample(i, fps, a.sample_fps): continue
        for d in det.detect(frame):
            if d.class_name not in VEHICLE_CLASSES: continue
            x1, y1, x2, y2 = (int(round(v)) for v in d.bbox)
            crop = frame[max(y1, 0):y2, max(x1, 0):x2]
            if crop.size == 0: continue
            t0 = time.perf_counter()
            img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            p = torch.softmax(model(tf(img).unsqueeze(0)), 1)[0]
            ms = (time.perf_counter() - t0) * 1000
            top = torch.topk(p, 5)
            k = len(rows)
            cv2.imwrite(str(out / "crops" / f"{k:05d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows.append(dict(k=k, frame=i, t=round(i / fps, 2), cls=d.class_name,
                             conf=round(d.confidence, 3), bbox=[x1, y1, x2, y2],
                             w=x2 - x1, h=y2 - y1, ms=round(ms, 1),
                             top5=[(names[j.item()], round(q.item(), 4)) for q, j in zip(*top)]))
json.dump(dict(clip=a.clip, fps=fps, threads=torch.get_num_threads(), rows=rows),
          open(out / "results.json", "w"), indent=0)
ms = np.array([r["ms"] for r in rows])
print(a.name, "detections", len(rows), "ms/crop median", np.median(ms), "p90", np.percentile(ms, 90))
