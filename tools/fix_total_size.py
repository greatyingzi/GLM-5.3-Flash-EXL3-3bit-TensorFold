import json, os, sys

SRC = sys.argv[1]
p = os.path.join(SRC, "model.safetensors.index.json")
d = json.load(open(p))
files = sorted(set(d["weight_map"].values()))
tot = sum(os.path.getsize(os.path.join(SRC, f)) for f in files)
old = d.get("metadata", {}).get("total_size")
d.setdefault("metadata", {})["total_size"] = tot
json.dump(d, open(p, "w"))
print("分片数:", len(files))
print("真实合计:", tot)
print("原记录  :", old)
print("已写回 total_size =", tot)
