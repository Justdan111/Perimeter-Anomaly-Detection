"""Tally by-eye verdicts per clip. Motorcycle reps not listed count as 'moto'."""
import json, re
from collections import Counter
from pathlib import Path
H = Path(__file__).parent
CL = ["sample", "toronto", "street720", "nyc", "nairobi", "hyderabad", "cuttack"]
rows = []
for c in CL:
    tr = {s["track"]: s for s in json.load(open(H / "runs" / c / "tracks.json"))}
    v = {}
    for line in open(H / "verdicts" / f"{c}.txt"):
        if line.startswith("#") or not line.strip(): continue
        t, code, note = line.split(" ", 2)
        v[int(t)] = (code, note.strip())
    minw = 0 if c == "sample" else 100
    for t, s in tr.items():
        if s["w"] < minw: continue
        code, note = v.get(t, ("moto" if s["cls"] == "motorcycle" else "MISSING", ""))
        rows.append(dict(clip=c, track=t, cls=s["cls"], w=s["w"], code=code, nonus="nonUS" in note,
                         p=s["top1"][1], n=s["n"], label=s["top1"][0]))
miss = [r for r in rows if r["code"] == "MISSING"]
print("missing verdicts:", [(r["clip"], r["track"], r["cls"]) for r in miss])
print(f"{'clip':10} reps  moto bus  X | judged  C   M   P   U   W   B | C%  C+M%  obvWrong%")
T = Counter()
for c in CL:
    k = Counter(r["code"] for r in rows if r["clip"] == c); T += k
    j = sum(k[x] for x in "CMPUWB")
    print(f"{c:10} {sum(k.values()):4} {k['moto']:4} {k['bus']:3} {k['X']:3} | {j:5} " + " ".join(f"{k[x]:3}" for x in "CMPUWB")
          + f" | {100*k['C']/j:3.0f} {100*(k['C']+k['M'])/j:4.0f} {100*k['B']/j:6.0f}")
j = sum(T[x] for x in "CMPUWB")
print(f"{'ALL':10} {sum(T.values()):4} {T['moto']:4} {T['bus']:3} {T['X']:3} | {j:5} " + " ".join(f"{T[x]:3}" for x in "CMPUWB")
      + f" | {100*T['C']/j:3.0f} {100*(T['C']+T['M'])/j:4.0f} {100*T['B']/j:6.0f}")
# Does confidence separate right from wrong?
J = [r for r in rows if r["code"] in "CMPUWB" and len(r["code"]) == 1]
for th in (0.0, 0.1, 0.2, 0.3, 0.5):
    s = [r for r in J if r["p"] >= th]; k = Counter(r["code"] for r in s)
    print(f"conf>={th:.1f}: {len(s):3} judged, correct {k['C']:3} ({100*k['C']/max(1,len(s)):.0f}%), W+B {k['W']+k['B']:3} ({100*(k['W']+k['B'])/max(1,len(s)):.0f}%)")
wrong_hi = sorted([r for r in rows if r["code"] in ("W", "B", "moto", "bus", "X") and r["p"] >= 0.4], key=lambda r: -r["p"])
print("confident nonsense (p>=0.4):", [(r["clip"], r["track"], r["code"], r["label"], round(r["p"], 2)) for r in wrong_hi])
nu = [r for r in J if r["nonus"]]; k = Counter(r["code"] for r in nu)
print("non-US-market vehicles judged:", len(nu), dict(k))
json.dump(rows, open(H / "verdict_rows.json", "w"), indent=0)
