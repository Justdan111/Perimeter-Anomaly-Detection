"""Make-level accuracy per preprocessing, on every judged rep whose true make
was identified by eye (first word of the verdict note, if it is a make)."""
import json, csv
from pathlib import Path
import torch, timm
from PIL import Image
from torchvision import transforms as T
H = Path(__file__).parent; torch.set_num_threads(4)
ck = torch.load(H / "vehicle_classifier.pth", map_location="cpu", weights_only=True)
names = ck["class_mapping"]; m = timm.create_model("efficientnet_b4", num_classes=len(names)); m.load_state_dict(ck["model_state"]); m.eval()
norm = lambda s: s.lower().replace("mercedes-benz", "mercedes").replace("-", "")
MAKES = {norm(n.split()[0]) for n in names.values()} - {"land", "am", "aston", "alfa", "can"}
alias = {"chevrolet": "chevrolet", "vw": "volkswagen", "bmw": "bmw", "toronto": None}
def pad(im):
    s = max(im.size); bg = Image.new("RGB", (s, s), (124, 116, 104)); bg.paste(im, ((s-im.width)//2, (s-im.height)//2)); return bg
nz = [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
P = {"squash": T.Compose([T.Resize((380, 380))] + nz), "centercrop": T.Compose([T.Resize(400), T.CenterCrop(380)] + nz),
     "pad": T.Compose([T.Lambda(pad), T.Resize((380, 380))] + nz)}
items = []
for c in ["sample", "toronto", "street720", "nyc", "nairobi", "hyderabad", "cuttack"]:
    tr = {s["track"]: s for s in json.load(open(H / "runs" / c / "tracks.json"))}
    for line in open(H / "verdicts" / f"{c}.txt"):
        if line.startswith("#") or not line.strip(): continue
        t, code, note = line.split(" ", 2)
        if code not in list("CMPUWB"): continue
        words = [w for w in note.replace("nonUS ", "").split() if w not in ("silver", "white", "black", "grey", "red", "blue", "dark", "dark-blue", "NYC", "taxi", "gray")]
        mk = alias.get(words[0].lower(), norm(words[0])) if words else None
        if mk in MAKES: items.append((c, tr[int(t)]["rep"], mk))
print(len(items), "reps with an identified make")
with torch.inference_mode():
    for name, tf in P.items():
        ok = 0
        for c, k, mk in items:
            im = Image.open(H / "runs" / c / "crops" / f"{k:05d}.jpg").convert("RGB")
            ok += norm(names[m(tf(im).unsqueeze(0)).argmax().item()].split()[0]) == mk
        print(f"{name:10} make correct {ok}/{len(items)} = {100*ok/len(items):.0f}%")
