[![Ko-fi](https://img.shields.io/badge/Ko--fi-Support%20yingzi-FF5E5B?logo=ko-fi&logoColor=white)](https://ko-fi.com/yingzi62662)

**Model weights (Hugging Face):** [androidli/GLM-5.3-Flash-Uncensored-EXL3-3.0bpw-mcg-TensorFold](https://huggingface.co/androidli/GLM-5.3-Flash-Uncensored-EXL3-3.0bpw-mcg-TensorFold) — this repo is its serving-side counterpart (patches / kernel / Dockerfile); the two projects depend on each other.

# GLM-5.3-Flash EXL3 3-bit on TensorFold (2× DGX Spark)

**Status: in production.** The 3.0bpw pack this project enables is published at
[androidli/GLM-5.3-Flash-Uncensored-EXL3-3.0bpw-mcg-TensorFold](https://huggingface.co/androidli/GLM-5.3-Flash-Uncensored-EXL3-3.0bpw-mcg-TensorFold)
(model card includes measured quality / KV-pool / speed vs Mia-AiLab's official 4bpw).

TensorFold 0.6.0's `glm5_next` CUDA family hard-rejects anything but 4-bit EXL3 routed experts.
This repo contains the **patch set + custom 3-bit expert kernel + reproducible Dockerfile** that make
3.0bpw/mcg EXL3 checkpoints serve correctly on a 2× DGX Spark (GB10) TP=2 pair — and measurably
**faster** than 4bpw (forward-pass −8 / −16 / −17.5% latency at spec-decode verify widths 1/4/8)
with a larger KV pool (3,891,200 tokens vs 2,582,528 under our tuned config).

## What's inside

| Path | Content |
|---|---|
| `patches/allow_3bit.py` | Relax the `EXL3_VARIANT` gate to `bits ∈ {3,4}` (codebook/scope still strict) |
| `patches/fix_v7_pick.py` | Clamp uninitialized garbage out of the pick buffer (harmless hardening) |
| `patches/fix_v12_smem.py` | Raise `group_kernel` dynamic shared memory to 96KB (must stay under the opt-in limit) |
| `patches/patch_k3.py` + `k3-kernel/` | **Custom 3-bit expert kernel** (`exl3k3.cu`, generated from the family's 4-bit kernel by `make_k3.py`): 24-word tiles, 3-bit lane-map decode, smem-resident prompt decode. `TF_GLM_EXL3_K3=0` falls back to the generic multi-width path |
| `Dockerfile` | Reproducible final image (build-time self-checks; no debug env) |
| `tools/repack_fp8_to_bf16.py` | Repack residual FP8 non-expert tensors to BF16 **preserving `weight_scale_inv` scales** — skip this and the model loads but outputs garbage |
| `tools/merge_l45_from_4bpw.py` | Graft layer 45 (MTP head) from a 4bpw pack instead of requantizing |
| `tools/fix_total_size.py` | Recompute `model.safetensors.index.json` `total_size` after surgery |

## Quick start

```bash
docker build -t tensorfold-glm53:3bit-v18-k3 .
# then serve (2x DGX Spark, rank1 over CX7):
tensorfold serve /path/to/androidli/GLM-5.3-Flash-Uncensored-EXL3-3.0bpw-mcg-TensorFold \
  --tp 2 --parallel 8 --context 1048576 --max-tokens 32768 --thinking --vision --spill-gib 300
```

Measured on 2× DGX Spark (GB10, CX7 direct): KV pool 3,891,200 tokens; EvalPlus 468/542 = 86.3%
(Mia 4bpw: 469/542 = 86.5%); GSM8K 98.0 / HumanEval 98.2. Full numbers and harness caveats in the
model card's `results/benchmarks.md`.

## License & attribution

Apache-2.0 (see `LICENSE` / `NOTICE`). Derived from [TensorFold](https://github.com/ashhart/TensorFold)
(Apache-2.0) as shipped in Mia's AI Lab `GLM-5.3-Flash-EXL3-2x-DGX-Sparks` recipe image; not affiliated
with MiaAI-Lab or the TensorFold author.

---

# 工程日志（保留全过程谱系）

## 为什么需要这个项目

## 为什么需要这个项目

我们在 `tf-glm-exl3-lowbit` 里把 GLM-5.3-Flash 的路由专家量化成了 **3.0bpw / 3.5bpw**
（`codebook=mcg`, `scope=glm53_routed_experts_only`）。包本身没问题，但 TensorFold 0.6.0 的
`glm5_next` **CUDA 家族拒绝加载**：

```
/usr/local/lib/python3.12/dist-packages/tensorfold/families/glm5_next/__init__.py:21
    EXL3_VARIANT = {"bits": 4, "codebook": "mcg", "scope": "glm53_routed_experts_only"}
...:50
    if {…int(v)…} != EXL3_VARIANT:
        raise ValueError("…CUDA engine reads EXL3 checkpoints with 4-bit mcg-codebook routed experts…")
```

现役能跑的 4bpw 包字段完全吻合（`bits=4/mcg/glm53_routed_experts_only`），我们的包是
`bits=3`（3.0）与 `bits=3.5`（3.5，`int(3.5)==3`）。**变体是逐字段严格相等**，所以被拒。

## 关键依据：参考解码路径本来就是 bit-width 参数化的

```
cuda/exl3.py:39   def states(trellis, bits: int = 4)
cuda/exl3.py:57   def unpack(trellis, bits: int = 4)     # 形状按 16*bits 推导
cuda/exl3.py:88   def dequantize(trellis, suh, svh, …, bits)
cuda/exl3.py:96   def forward(x, trellis, suh, …, bits)
```

⇒ 3-bit 的算法基础已具备（形状规则 `16×bits`：4bit=64，3bit=48）。**待办 = 放开变体校验 +
验证快速 CUDA 路径在 bits=3 下的数值正确性与几何**。

## 做法（分期，先跑起来再谈最优）

1. **派生镜像**：`FROM tensorfold-glm53:lowbit`，只把上面那条判据改成
   「codebook/scope 严格相等 + bits ∈ {3,4}」。**基镜像一字不改**，现役服务与契约不受影响。
   产物 tag：`tensorfold-glm53:3bit`。
2. **镜像内自检**（build 时断言 `EXL3_ALLOWED_BITS` 已生效，防止补丁静默失效）。
3. **起栈验证**：`MODEL_ID=local/tfport-30mcg MODEL_REVISION=rev-30mcg-20261015
   IMAGE=tensorfold-glm53:3bit`（两机缓存布局已按 hf 1.33 规范建好并验证过）。
4. **数值对拍**：同样张量上，引擎参考路径 `dequantize(..., bits=3)` 与量化侧
   `exllamav3` 的解码逐元素比对；不一致就把差异钉在具体的 tile/几何上。
5. **快速路径几何**：`cuda/exl3_mm.py` 的 tile/warp/K-split 是按 4bit（trellis 64 宽）调的，
   3bit 是 48 宽 ⇒ 需要重新标定；用 4 的对拍结果做验收。

## 当前进展与卡点（2026-10-15）

**已打通（每步都有验证）**

1. **变体门禁** → 派生镜像 `tensorfold-glm53:3bit`（同一 codebook/scope + `bits ∈ {3,4}`，build 期自检 + 新容器复验 ✓）
2. **数据形态** → 包内 186 个残留 FP8 非专家张量（shared_experts 129 + self_attn 48 + 密集 mlp 9）
   按 FP8 源 `*_weight_scale_inv`（128×128 块，F32）反量化为 BF16 → 新包
   `GLM53-UNCENSORED-EXL3-mcg-3.0bpw-bf16`。dtype 普查与能跑的 4bpw 参考**逐项一致**：
   `F16/I32/I16` + `BF16 1327` + `F32 291`，`F8_E4M3 = 0`。
   ⚠️ 缩放名是 `…x.weight_scale_inv`（**替换**尾部 `.weight`），不是追加；缩放张量分布在**别的分片**里。
3. **缓存布局** → hf ≥1.33 规范：`blobs/<git-blob-sha1>` + 相对 symlink 快照 + **无换行** `refs/*` +
   `trees/<commit>.json`（`_raise_if_incomplete_snapshot` 只校验文件名存在）。两机解析验证双路径 OK。
4. **启动环境** → 统一内存机器上大文件哈希会把**页缓存**吃满 ⇒ `MemFree` 掉到 2–3 GiB ⇒
   内存门禁静默拦死 ⇒ 起栈前两机 `drop_caches` + `MEM_GATE_DROP_CACHES=1`。

**结果：3bit 包已被引擎完整读入并完成初始化 ✓**

```
⋯ 138s elapsed, 80.1 of ~70.37 GiB on the GPU here, 79.0 on the worker
[tensorfold] drafter timings (ms, fastest of 7): v1 25.45 … v7 52.9
[tensorfold] verify windows up to 16 rows, CUDA graphs for 1..8, 16 rows
[tensorfold] vision tower ✓ · TF_GLM_KV=fp8 ✓ · spill ✓ · hyper-connections ✓
```

**最后卡点：引擎自检里第一次启动"专家分组"内核**

**关键定位（2026-10-15 深挖）**：走的是**我们自己移植进引擎的通用多宽路径** ✓（源码里带项目标记）：

```python
# tensorfold/families/glm5_next/cuda/exl3_mm.py:213
if ex.gen is not None:                     # [tf-glm-exl3-lowbit] 通用多宽/多码本路径
    from tensorfold.cuda.exl3 import experts as _generic
    ...
    _generic.routed(x, pick, None, ex.gen, gs, None, R, float(limit))
```
调用链：`decode.py:650 prefill_chunk_steps → forward.py:848/805 moe_block → exl3_mm.py:218 → cuda/exl3/experts.py:186 ext.group(...)`

`group()` 的 C++ 包装（`experts.cpp`）只做范围校验
（`TORCH_CHECK(pick.numel() >= R*slots)` / `uids.numel() >= min(R*slots, E)` / `members.size(0) >= uids.numel()`），
校验失败会抛**带消息的 Python 错**；实际报的是 `CUDA error: invalid argument` ⇒
**参数过了校验，是 CUDA 启动配置（grid/block/共享内存）本身非法**。
而 `group` 与 bit 宽无关 ⇒ 最大嫌疑是 **`E`（本 rank 专家数）或 `R`/`slots` 取到 0 或非法值**（grid 为 0）。

**根因定位（2026-10-15，已确诊 ✓）**

诊断补丁（`diag_group.py`，派生镜像 `tensorfold-glm53:3bit-diag`）在 `ext.group` 前打印实参，得到：

```
[DIAG-GROUP] R=1 slots=9 E=288 pick=144 ids=9 members=(9, 1) count=1
             ex.count=288 dims=4096 width=1024 k2_range=(6, 6)
```

⇒ **实参全部正常**（`E=288` 非 0；`k2_range=(6,6)` ⇒ trellis 被正确解析为 K2=6 = 3bit ✓✓，
即 graft 布局解析没问题）。先前"E/R 为 0"的猜测**被证伪**。

真正的错误（诊断打印自己把它演示了出来）：

```
[DIAG-GROUP] print failed: CUDA error: operation not permitted when stream is capturing
                            (cudaErrorStreamCaptureUnsupported)
```

⇒ **图捕获期间存在不允许的操作**。我们移植的通用路径在 `exl3_mm.routed` 里有**惰性分配**：

```python
# exl3_mm.py:~215
if gs is None or gs.y.shape[0] != P:
    gs = _generic.Scratch(ex.gen, rows, slots, device=x.device)   # 首次调用即分配
    s.gen = gs
```

而**首次调用恰好发生在 CUDA 图捕获内部**（`graphs.py:40 with torch.cuda.graph(g, pool=self.pool)`）
⇒ 捕获期建对象/分配 ⇒ `cudaErrorStreamCaptureUnsupported` → `…CaptureInvalidated`
⇒ 引擎初始化失败、栈起不来 ✓。（这也解释了 4bit 正式路径为何无事：它不在捕获期分配。）

**修法（根因，落点在我们自己的移植代码里）**

在图捕获**之前**把通用 scratch 预热出来（off-capture，用很小的 `R` 跑一次 `_generic.routed`），
使捕获路径里 `gs` 已存在、**零分配**：

- 位置：家族引擎初始化里、`decode.py:226 Graphs(...)` **之前**；条件是模型走通用路径
  （`m.experts.gen is not None`），且用家族那个 `Scratch`（`b.exl3` 同一个对象）
- 实现方式：仍走派生镜像叠加补丁（基镜像与现役服务不受影响），补丁要能断言"预热后
  `torch.cuda.is_current_stream_capturing()` 为 False 时 `s.gen is not None`"
- 注意：诊断打印本身不可留在生产镜像里（print 不是 capture-safe）


在派生镜像里临时给 `ext.group` 包一层，打印 `R, slots, E, pick.numel(), uids.numel(), members.shape`
再调用一次 ⇒ 直接看出哪个数是 0/非法；若 `E` 来自 `ex.gen.count`（通用专家对象）则回头查
`cuda/exl3/experts.py` 的 `Exl3RoutedExperts`/`scan` 对我们 graft 布局的解析（`k2_of` 读 trellis 末维
`16*K` 或 `16*K+8`，K2=6 ⇒ 48、K2=7 ⇒ 56）。


```
tensorfold/cuda/exl3/experts.py:186  ext.group(pick, ids, s.count, members, R, slots, E)
torch.AcceleratorError: CUDA error: invalid argument
```

**已排除**"内核不支持 3bit"：`K2_SUPPORTED = tuple(range(2,17))`（3bit=6、3.5bit=7 都在内）、
`decode.cuh` 明文提到 3.5 bits、`experts_grouped.cuh` 的 `template <int CB,int NT,int W,int PF,int LO,int HI>`
按 `[LO,HI]` 区间实例化。

**下一步（内核级）**

- 读 `experts_grouped.cuh` 的 `[LO,HI]` 实例表 + host 启动器（`experts.cpp` 只有 `m.def("group", &group)`，
  区间选择在 `.cuh`/`experts.cu` 侧）—— 看 K2=6 是否落在已编译区间内
- 最小复现：在容器里对 **K2=6 与 K2=8（4bit 对照组）** 各调一次 `ext.group(...)`，
  开 `CUDA_LAUNCH_BLOCKING=1`；注意 `_ext()` 用 `verbose=False` **会吞掉编译信息**，复现时必须打开
- 若确为实例缺失 ⇒ 在派生镜像里补编译覆盖 6/7 的实例（源码在镜像内，JIT 可重建）

**项目文件**

- `allow_3bit.py` — 门禁补丁（幂等、锚点失败即报错）
- `verify_gate.py` — 门禁自检（4/3/3.5 放行，2/6 仍拒）
- `repack_fp8_to_bf16.py` — FP8→BF16 重打包（块形状断言、原子落盘、可续跑、跳过无 fp8 分片直接复制）
- `Dockerfile` — 派生镜像（含 build 期自检）


- 栈目录：`~/dgx/stacks/glm53-tf`（`start.sh` / `prepare.sh` / `config/*.env`）
- 两机 `tensorfold-glm53:lowbit` 必须都在（prepare 会校验 "identical on both Sparks"）
- 大文件哈希/拷贝会把统一内存机器的**页缓存**灌满 ⇒ `MemFree` 掉到 2–3 GiB ⇒ 内存门禁静默拦死
  启动 ⇒ 起栈前两机 `drop_caches` + `MEM_GATE_DROP_CACHES=1`
- 本地包要满足 hf ≥1.33 缓存规范（`refs` 无换行 + `trees/<commit>.json` + blobs 用 git-blob sha1）

## 目录（本仓）

- `patches/` — 补丁脚本（幂等、锚点失败即报错）
- `k3-kernel/` — 自写 3-bit 专家算子（`make_k3.py` 生成 / `patch_k3.py` 接入 / `test_k3.py` 对拍）
- `tools/` — 包侧工具（FP8→BF16 重打包 / L45 嫁接 / index 修正）
- `Dockerfile` — 最终镜像（build 期自检）

## 2026-10-16 结案（Claude Code 接管）

3bit 服务已上线：`~/dgx/bin/start-glm53-3bit.sh`（DGX head），包 `androidli/GLM-5.3-Flash-Uncensored-EXL3-3.0bpw-mcg-TensorFold`，镜像 `tensorfold-glm53:3bit-v16-fast`，1235 实测正常。

真正的两个根因（此前"rank0 静默退出 / 两卡会合死等 / 3bit tp2 未适配"均为误判）：
1. **chat_template**：3bpw 包用了 uncensored 源的模板，没有官方那句 `<reminder>You are unable to process this ...`，
   TF `--vision` 在 `families/glm5_next/cuda/app.py:55` 抛 ValueError → rank0 exit 1（日志最后一行就写着，v14 起每次都是它）。
   修：两机两包都换成官方 4bpw 模板，原件留 `chat_template.jinja.uncensored.bak`。
2. **`-L45` 包权重错**：186 个非专家张量（shared_experts/self_attn/dense mlp）是 FP8 直接转 BF16、没乘 `weight_scale_inv`（absmean 30~77 vs 正确 0.01）→ 乱码。
   `-L45-bf16` 包是正确反量化的，用它。
3. 镜像 v15 写死 `CUDA_LAUNCH_BLOCKING=1`/`CUDA_LOG_FILE=stderr`；v16 去掉后 34→58 tok/s。

验证：专家 3bpw mcg 解码 vs FP8 源 cos=0.991 relerr=0.135（正常）。
对照（同题）：3bit 58.4 / 35.0 tok/s，KV 池 2.58M；4bpw 63.0 / 36.3 tok/s，KV 池 1.86M。
下一步：通用 EXL3 路径恢复 verify CUDA 图（fix_v5 关掉了）以追平/超过 4bpw 速度。
教训：先看 `docker logs` 完整尾部，不要只 grep error 关键词；分布式"死等"前先验数值。

### 2026-10-16 优化（Claude Code）
- v17 镜像 `tensorfold-glm53:3bit-v17-graphs`：撤掉 fix_v5，verify CUDA 图恢复（当初捕获失败是 CUDA_LAUNCH_BLOCKING=1 导致，不是 l2pf）。
- PARALLEL=8：KV 池仍 2.58M token；聚合 112 tok/s（4 并发 90），单路不变。启动脚本默认已是 v17 + PARALLEL=8。
- 实测单路：推理 60 / 中文散文 36 / 代码 64 / 英文散文 48 / 英文结构化 54 tok/s（与配方 README 4bpw 区间一致）。
- 微基准（L10 单层，rank 半宽，R=1）：3bit 通用 203us，4bit 通用 236us，4bit 专用 dec 237us ⇒ 3bit 专家省 ~14%；
  内核受 trellis 解码算力限制（186GB/s），tile 扫描最多 2-4%，未采用。专家只占单步 26ms 中 ~8.5ms。
- L45 借自 4bpw：与 uncensored FP8 源 cos 0.997（比 3bit 层还准），无需重量化。
- 工具：DGX `~/dgx/bin/bench-glm53.py [-p N]`；微基准脚本在 DGX `/tmp/ubench*.py`、`/tmp/sweep.py`。

### 2026-10-16 草稿模型对比（PARALLEL=1，tok/s；按请求 tf_policy 切换）
| 题型 | DFlash2 fnc7:0.3 | MTP (a) | auto:2:8:0.03 |
|---|---|---|---|
| 中文推理 | 61.0 | 58.8 | 59.7 |
| 中文散文 | 37.2 | 46.4 | 45.4 |
| 英文散文 | 49.0 | 46.6 | 45.0 |
| 代码 | 65.9 | 56.9 | 64.4 |
| 英文结构化 | 55.5 | 50.5 | 53.8 |
- 第 45 层就是 MTP 头（num_hidden_layers=45）；默认 TF_GLM_MTP=auto 不加载它。MTP/auto 只能 PARALLEL=1。
- 请求级切换：body 加 `"tf_policy": "a"` 或 model 写 `dgx-spark@a`（需 TF_GLM_MTP=1 PARALLEL=1 起栈）。
- 线上保持 PARALLEL=8 + DFlash2。接受率工具：DGX `~/dgx/bin/accept-glm53.py [policy]`。

### 2026-10-16 自写 3bit 专家算子 → v18（Claude Code）
- `k3-kernel/`：`make_k3.py` 由家族 4bit 内核 exl3.cu 生成 3bit 版 exl3k3.cu（tile 32→24 字、3bit LaneMap 解码、LD=nc 向量加载、
  prompt 内核直接从共享内存解码），调度/融合/epilogue 原样保留；`patch_k3.py` 接入引擎（trellis [...,48] 的 mcg 专家走它；
  `TF_GLM_EXL3_K3=0` 回退通用路径）；`test_k3.py` 对拍 + 测速。镜像 `tensorfold-glm53:3bit-v18-k3`（两机），启动脚本默认。
- 单层对拍：与通用模块 cos 0.999994；R=1 184us（通用 204、4bit 235）；prefill 2048 行 13.2ms（通用 66.6ms、4bit 10.6ms）。
- 端到端（v17→v18）：prefill 757→1757 tok/s；8 并发 112→167 tok/s；单路 推理 60→70.5、代码 64→67.6、结构化 54→62.2。
- 质量：greedy 11 题全对（含执行生成代码）；needle 30k/60k 均命中。
- 第 45 层（MTP）：量化时 `-mb 16` 有意存为 F16，并非漏量化；线上默认不加载。

### 2026-10-16 第二轮优化（Claude Code）
- 启动脚本新默认：`TF_GLM_PREFILL_ROWS=4096`、`TF_GLM_EXL3_LOADS=nc2`、`TF_GLM_EXL3_DEC_ORDER=2`（微基准扫描，输出逐位一致）。
- 同机公平对比（启动自测的单次前向，ms）：v1 4bpw 26.56 / 3bit 24.40（-8%）；v4 45.72 / 38.35（-16%）；v8 59.76 / 49.30（-17.5%）。
- prefill（4 万 token）：3bit 4096 行 1905~1924 tok/s，池 2.58M；4bpw 2048 行 1960（池 1.65M），4096 行 2077（池 1.39M）。
  3bit prompt 内核比 4bit 慢 ~18%：ncu 显示两者都是延迟受限（占用率 33%、发射槽 33%），3bit 多 14% 指令（非字节对齐的位提取）、
  short_scoreboard +73%；流水线深度/提前取字/PASS128 均 ≤2%，属 3bit 解码固有成本。
- 单路 decode 上限：专家仅占每步 ~7.4ms（已达带宽极限）；dense q4 ~10ms、通信+注意力 ~7ms 与专家位宽无关。

### 2026-10-16 EvalPlus（对齐 Mia 口径：greedy、思考开、上限 32768、EvalPlus 0.3.1 原版提示词）
- 3bit-v18：HumanEval+ 153/164，MBPP+ 315/378，合计 **468/542 = 86.3%**（Mia 4bpw 469/542 = 86.5%，TR3 468/542 = 86.3%）。
- 撞 32K 上限：HumanEval+ 5、MBPP+ 14（Mia MBPP+ 8、TR3 11）；MBPP+ 回答长度中位数 970 tok（Mia 报告 601）。
- 撞上限的题是 greedy 下的“长段自我怀疑循环”（如 HE/128 反复核对一条记错的测试用例），引擎 LoopGuard（≤16 token 周期）抓不到。
- 结果与脚本：DGX ~/dgx/var/evalplus/（evalplus_par.py 可断点续跑，he-32k-aligned*.json、run-mbpp-32k/）。

### 2026-10-16 decode 真实剖析（nsys，rank0，单路 prose，6 s ≈ 150 轮，~40 ms/轮）
- 专家 dec_kernel 16.0 ms（40%）| dense qmm 11.4 ms（28%）| RoCE all-gather 3.5 ms（9%，~104 次/轮）| 注意力 ~3.1 ms | lm_head FP8 1.6 ms | hc 等 ~1.5 ms；
  l2pf 4.9 ms 在旁路流并行。GPU 有 kernel 运行的时间占 96.6% ⇒ 无调度空转可挖。
- 参数扫描（均无收益，保持默认）：L2PF=0 慢 3-5%；L2PF_MB 12/16 持平；DRAFT_QUANT=mse 接受率略降；草稿策略 fnc7:0.3 已最优；KVB=fp8 +2% 但有损。
- 剩余杠杆：dense 改 3bit（预计每轮 -3.5 ms，有损，需质量门槛）；all-gather 合并（≤9%）；专家 L2 预取点（小）。
- 抓取方法：DRY_RUN=1 拿命令，rank0 加 --cap-add SYS_ADMIN 并用 `nsys launch --session-new=... --cuda-graph-trace=node` 包裹，`docker exec ... nsys start/stop` 录窗口。

### 2026-10-17 dense 3bit 质量模拟（DGX ~/dgx/var/evalplus/run-dense3sim）
- 包 androidli/GLM53-3bpw-dense3sim：326 个非专家矩阵换成 exllamav3 3bpw 的 BF16 解码值，DENSE=bf16 服务（head/kv_b 仍 BF16）。
- EvalPlus（同口径）：HumanEval+ 150/164（基线 153），MBPP+ 310/378（基线 315），合计 460/542 = 84.9%（基线 86.3%）。
  新做对 12、新做错 20，配对精确检验 p = 0.22（不显著，但方向为负）。撞 32K：HE 6、MBPP 9。
- 微基准：EXL3 3bpw dense 比 q4 快 1.2-1.6x，预计 decode 每轮 -3 ms（约 7-8%）。

### 2026-10-17 通信路径（两机同时 nsys + 两机微基准）
- decode：每轮 ~104 次 all-gather，每 rank 占总时间 11-12%；纯传输 ~4.7%，其余是两机 kernel 抖动造成的互等（TP 每层两次同步的固有代价）。
- 微基准（同步时）：RoCE 16 KiB 11 us（NCCL 43 us）；大/小核绑定无差别 ⇒ CPU 代理不是瓶颈。
- 第二个 200G 网口（CX7 的 PCIe 孪生口 roceP2p1s0f1）原先闲置：未配 IPv4。手动 `ip addr` 配上后 IPv4 GID 在 3 号，
  与第一口的 5 号不一致（第一口由 NM 管理多一个 IPv6 链路本地地址），配方 rails() 要求同号 ⇒ 静默跳过。
- 修复（持久）：两机 `nmcli con mod enP2p1s0f1np1` → 10.0.1.1/24 与 10.0.1.2/24、MTU 9000、ipv6 link-local、autoconnect；
  GID 对齐到 5 后配方自动识别 2 rails（NCCL+RoCE 双口），原版 v18 镜像即可。备份：~/dgx/archive/net-backup-20261017/。
- 效果：NCCL 大块 9.6→14.5 GiB/s；RoCE 256 KiB 35.9→26.7 us；prefill 4 万 token 1905→1982 tok/s（+3-4%）；decode 持平。
- 可提 PR：roce.py 两机也按 HCA 取各自 IPv4 GID（v19 补丁，GID 号不一致时仍能双口）；rails() 对“孪生口 up 但无 IPv4/GID 不同号”给出提示。

### 2026-10-18 KV 池 / 启动参数优化
- 发现：3bit 下池子被 KV_POOL_GIB=12.5 封顶（2,576,384 tok），引擎预算还剩 ~22 GiB；kept_prompts 卡在 32 上限而池子大量空闲。
- 实测峰值（memwatch 2 s 采样）：95 万 token prompt + 4 路生成，头节点仅多吃 ~1.1 GiB（配方假设 10 GiB 不适用于当前配置）。
- 逐档：18.5 GiB → 池 3.30M，填满时头节点最低 16.3 GiB；26.5 GiB → 池 3.89M（+51%），填满时最低 12.7 GiB（目标 ≥10 = 安全线 8 + 漂移 2），看门狗未触发。
- 新默认（start-glm53-3bit.sh）：KV_POOL_GIB=26.5、TF_GLM_CACHE_ENTRIES=48、TF_GLM_KEPT_BYTES_GIB=4.5；速度不变（v1 24.45 ms，prefill 4 万 1966 tok/s）。
- 未做：DISPLAY_KV_MIB=1792（+~27.7 万 tok，免费内存）需 nvidia_drm modeset=1 并重启两机。
- 工具：DGX ~/dgx/bin/memwatch.sh、memstress.py、poolfill.py。
