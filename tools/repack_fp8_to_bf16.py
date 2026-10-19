#!/usr/bin/env python3
"""把 3bit 包里残留的 FP8 非专家张量还原成 BF16（引擎要求 "路由专家 4bit EXL3 + 其余 BF16"）。

背景
----
TensorFold 的 glm5_next 加载器只认 BF16（其余部分），而我们的包从 FP8 源 graft 而来，
`shared_experts.*`(129) + `self_attn.*`(48) + 密集 `mlp.*`(9) = 186 个张量仍是 F8_E4M3
⇒ 加载器 `DTYPE_BYTES['F8_E4M3']` KeyError，栈起不来。

做法
----
逐分片重写：`w_bf16 = (w_fp8.float() * scale_inv.float()).to(bfloat16)`，缩放按 128×128 块
重复到权重形状（断言 scale 形状 == ceil(H/128)×ceil(W/128)，块大小猜错立即报错）。
同时**丢弃**这些权重的 `*_scale_inv` 张量（BF16 不需要，且与能跑的 4bpw 包形态一致）。

只读输入；输出到新目录；原子落盘（.tmp → rename），可断点续跑。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

BLOCK = 128
SCALE_SUFFIX = ".weight_scale_inv"


def scale_name(n: str) -> str:
    """`…down_proj.weight` -> `…down_proj.weight_scale_inv`（替换掉尾部 .weight，不是追加）。"""
    base = n[: -len(".weight")] if n.endswith(".weight") else n
    return base + SCALE_SUFFIX


def load_index(root: str) -> dict[str, str]:
    idx = os.path.join(root, "model.safetensors.index.json")
    if os.path.isfile(idx):
        with open(idx) as fh:
            return json.load(fh)["weight_map"]
    out: dict[str, str] = {}
    for f in sorted(os.listdir(root)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(root, f), framework="pt") as sf:
                for k in sf.keys():
                    out[k] = f
    return out


def dtypes_of(root: str, names_by_shard: dict[str, list[str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for shard, names in names_by_shard.items():
        with safe_open(os.path.join(root, shard), framework="pt") as sf:
            for n in names:
                try:
                    out[n] = str(sf.get_slice(n).get_dtype())
                except Exception:  # noqa: BLE001
                    pass
    return out


def dequant(w_fp8: torch.Tensor, scale_inv: torch.Tensor, name: str) -> torch.Tensor:
    h, w = w_fp8.shape
    eh, ew = (h + BLOCK - 1) // BLOCK, (w + BLOCK - 1) // BLOCK
    if tuple(scale_inv.shape) != (eh, ew):
        raise SystemExit(
            f"FATAL: {name}: scale shape {tuple(scale_inv.shape)} != expected {(eh, ew)} "
            f"(weight {tuple(w_fp8.shape)}, block {BLOCK})"
        )
    s = (
        scale_inv.float()
        .repeat_interleave(BLOCK, dim=0)[:h]
        .repeat_interleave(BLOCK, dim=1)[:, :w]
    )
    return (w_fp8.float() * s).to(torch.bfloat16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="输入包目录（只读）")
    ap.add_argument("--source", required=True, help="FP8 源目录（提供 scale_inv）")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--only-layer", default=None, help="调试：只处理名字含该串的张量")
    ap.add_argument("--max-shards", type=int, default=None, help="小样验证：只处理前 N 个分片")
    args = ap.parse_args()

    p_map = load_index(args.pack)
    s_map = load_index(args.source)

    by_shard: dict[str, list[str]] = {}
    for n, sh in p_map.items():
        by_shard.setdefault(sh, []).append(n)
    dt = dtypes_of(args.pack, by_shard)
    fp8 = [n for n, d in dt.items() if d == "F8_E4M3"]
    if args.only_layer:
        fp8 = [n for n in fp8 if args.only_layer in n]
    print(f"FP8 张量: {len(fp8)} / 总 {len(dt)}")

    targets: dict[str, list[str]] = {}
    for n in fp8:
        targets.setdefault(p_map[n], []).append(n)
    scales_in_pack = {scale_name(n) for n in fp8} & set(p_map)
    print(f"涉及分片: {len(targets)}；包里将丢弃的 *_scale_inv: {len(scales_in_pack)}")

    os.makedirs(args.out, exist_ok=True)

    # 非 safetensors 的附属文件直接复制（index.json 之后再写）
    for f in sorted(os.listdir(args.pack)):
        if f.endswith(".safetensors") or f == "model.safetensors.index.json":
            continue
        src, dst = os.path.join(args.pack, f), os.path.join(args.out, f)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    t0 = time.time()
    done_shards = 0
    shard_list = sorted(set(p_map.values()))
    if args.max_shards:
        shard_list = shard_list[: args.max_shards]
    for shard in shard_list:
        dst = os.path.join(args.out, shard)
        if os.path.exists(dst):
            done_shards += 1
            continue
        if shard not in targets:
            # 无 FP8 张量的分片：直接复制（内容不变）
            shutil.copy2(os.path.join(args.pack, shard), dst)
            done_shards += 1
            continue

        t = load_file(os.path.join(args.pack, shard))
        n_done = 0
        for name in targets[shard]:
            w = t.get(name)
            if w is None:
                raise SystemExit(f"FATAL: {name} 不在分片 {shard} 里")
            if name not in s_map:
                raise SystemExit(f"FATAL: 源里没有 {name}")
            with safe_open(os.path.join(args.source, s_map[name]), framework="pt") as sf:
                wsrc = sf.get_tensor(name)
            sn = scale_name(name)
            if sn not in s_map:
                raise SystemExit(f"FATAL: 源里没有缩放 {sn}")
            with safe_open(os.path.join(args.source, s_map[sn]), framework="pt") as sfs:
                sc = sfs.get_tensor(sn)
            if wsrc.dtype != torch.float8_e4m3fn:
                raise SystemExit(f"FATAL: {name} 源 dtype {wsrc.dtype} 非 fp8")
            t[name] = dequant(wsrc, sc, name)
            n_done += 1
        for sname in list(scales_in_pack):
            if sname in t:
                del t[sname]
        tmp = dst + ".tmp"
        save_file(t, tmp, metadata={"format": "pt"})
        os.replace(tmp, dst)
        done_shards += 1
        print(
            f"  [{done_shards}/{len(set(p_map.values()))}] {shard}: 反量化 {n_done} 个 "
            f"({time.time()-t0:.0f}s)"
        )
        del t

    with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as fh:
        json.dump({"metadata": {"total_size": 0}, "weight_map": p_map}, fh, indent=1)

    print(f"完成: {args.out}  分片 {done_shards}  用时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
