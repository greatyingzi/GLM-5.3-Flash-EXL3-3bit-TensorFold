#!/usr/bin/env python3
"""把 4bpw 包第 45 层专家的 EXL3 张量（3456 个）并入 3bpw 包 -> 新目录。
策略：原 17 个分片用硬链（零额外空间）；新增 1 个分片装第 45 层专家；重建索引。
不修改原包。需在容器内运行（要 torch + safetensors）。
"""
import json, os, shutil, sys

P3 = "/p3"
P4ROOT = "/p4root"
OUT = "/out"

# 定位 4bpw 的 snapshot 目录
p4 = None
if os.path.exists(os.path.join(P4ROOT, "model.safetensors.index.json")):
    p4 = P4ROOT
else:
    for d in sorted(os.listdir(P4ROOT)):
        c = os.path.join(P4ROOT, d)
        if os.path.exists(os.path.join(c, "model.safetensors.index.json")):
            p4 = c
            break
if p4 is None:
    print("找不到 4bpw index"); sys.exit(2)
print("4bpw:", p4)

idx3 = json.load(open(os.path.join(P3, "model.safetensors.index.json")))
wm3 = idx3["weight_map"]; meta3 = idx3.get("metadata", {})
idx4 = json.load(open(os.path.join(p4, "model.safetensors.index.json")))
wm4 = idx4["weight_map"]

keys45 = [k for k in wm4 if "layers.45.mlp.experts." in k]
print("待并入张量:", len(keys45))
if not keys45:
    print("4bpw 里没有第45层专家"); sys.exit(3)

os.makedirs(OUT, exist_ok=True)

# 1) 原分片硬链
shards3 = sorted(set(wm3.values()))
for s in shards3:
    src = os.path.join(P3, s); dst = os.path.join(OUT, s)
    if not os.path.exists(dst):
        try: os.link(src, dst)
        except Exception: shutil.copy2(src, dst)
print("分片已就位:", len(shards3))

# 2) 元数据（非 safetensors、非 index）先硬链/拷贝
for f in sorted(os.listdir(P3)):
    if f.endswith(".safetensors") or f == "model.safetensors.index.json":
        continue
    src = os.path.join(P3, f); dst = os.path.join(OUT, f)
    if not os.path.exists(dst):
        try: os.link(src, dst)
        except Exception: shutil.copy2(src, dst)

# 3) 读出第45层张量并写新分片
import torch
from safetensors import safe_open
from safetensors.torch import save_file

byfile = {}
for k in keys45:
    byfile.setdefault(wm4[k], []).append(k)

tensors = {}
for f, ks in byfile.items():
    with safe_open(os.path.join(p4, f), framework="pt", device="cpu") as fh:
        for k in ks:
            tensors[k] = fh.get_tensor(k).contiguous()
print("读出张量:", len(tensors))
sample = list(tensors)[:3]
for k in sample:
    print("   ", k, tuple(tensors[k].shape), tensors[k].dtype)

newshard = "model-00018-of-00018.safetensors"
save_file(tensors, os.path.join(OUT, newshard), metadata={"format": "pt"})
print("新分片写入:", newshard, os.path.getsize(os.path.join(OUT, newshard)), "bytes")

# 4) 重建索引
wm = dict(wm3)
for k in keys45:
    wm[k] = newshard
json.dump({"metadata": meta3, "weight_map": wm},
          open(os.path.join(OUT, "model.safetensors.index.json"), "w"), indent=1)
print("新索引张量数:", len(wm), "(原", len(wm3), ", +", len(keys45), ")")
print("完成 ->", OUT)
