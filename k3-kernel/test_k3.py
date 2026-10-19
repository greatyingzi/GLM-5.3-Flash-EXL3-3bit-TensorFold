# tf-3bit-kernel: correctness vs the generic module + speed vs generic 3-bit and family 4-bit, one MoE layer, rank-0 half.
import json, os, sys, torch
from safetensors import safe_open
from tensorfold.cuda.build import load
from tensorfold.cuda.exl3 import experts as G
from tensorfold.cuda import experts as grouped
from tensorfold.families.glm5_next.cuda import exl3_mm as M
L, P3, P4 = int(sys.argv[1]), sys.argv[2], sys.argv[3]
E, D, I = 288, 4096, 2048; NI = I // 2; dev = "cuda"
K3 = load(name="tf_glm_exl3k3_dev", sources=["/k3/exl3k3.cpp", "/k3/exl3k3.cu"], extra_cuda_cflags=["-O3"], verbose=False)
EXT4 = M._ext()

def load_layer(d):
    m = json.load(open(d + "/model.safetensors.index.json"))["weight_map"]; fh = {}
    def get(n):
        f = m[n]
        if f not in fh: fh[f] = safe_open(d + "/" + f, "pt")
        return fh[f].get_tensor(n)
    P = f"model.language_model.layers.{L}.mlp.experts."
    out = []
    for name, split_n in (("gate_proj", True), ("up_proj", True), ("down_proj", False)):
        ts, us, vs = [], [], []
        for e in range(E):
            t, u, v = get(P + f"{e}.{name}.trellis"), get(P + f"{e}.{name}.suh"), get(P + f"{e}.{name}.svh")
            if split_n: t, v = t[:, : NI // 16], v[:NI]
            else: t, u = t[: NI // 16], u[:NI]
            ts.append(t.contiguous()); us.append(u); vs.append(v)
        out += [torch.stack(ts).to(dev), torch.stack(us).to(dev), torch.stack(vs).to(dev)]
    return out   # gt, sg, vg, ut, su, vu, dt, sd, vd

def family_experts(raw):
    gt, sg, vg, ut, su, vu, dt, sd, vd = raw
    w = lambda t: t.contiguous().view(torch.int32)
    return M.Exl3Experts(w(gt), w(ut), w(dt), sg, su, vg, vu, sd, vd, E, NI, D)

def picks(R, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    p = torch.full((R, 9), E, dtype=torch.int32)
    for r in range(R): p[r, :8] = torch.randperm(E, generator=g)[:8].int()
    return p.to(dev).contiguous()

def run_family(ext, ex, x, pick, R, prompt):
    s = M.Scratch(R, 9, D, NI, dev, prompt=prompt)
    plan = grouped.Plan(R, 9, E + 1, dev, prefill=prompt, tile=64 if prompt else None)
    y = torch.zeros((R * 9, D), dtype=torch.float32, device=dev)
    M._ext.cache_clear() if hasattr(M._ext, "cache_clear") else None
    orig = M._ext
    M._ext = lambda: ext
    try:
        def f():
            grouped.route(pick, plan, 64 if prompt else plan.tile)
            M.routed(x, pick, plan, ex, s, y, R, 7.0)
        return f, y
    finally:
        M._ext = orig

def timeit(f, iters):
    for _ in range(3): f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): f()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record()
    for _ in range(iters): g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1000

raw3 = load_layer(P3)
gen3 = G.prepare_stacked(raw3[0], raw3[3], raw3[6], raw3[1], raw3[4], raw3[2], raw3[5], raw3[7], raw3[8], "mcg")
fam3 = family_experts(raw3)
for loads in ("0", "nc"):
    os.environ["TF_GLM_EXL3_LOADS"] = loads
    print(f"==== TF_GLM_EXL3_LOADS={loads}")
    for R, prompt in ((1, False), (4, False), (16, False), (256, True), (2048, True)):
        if prompt and loads != "0": continue
        torch.manual_seed(R)
        x = (torch.randn(R, D, device=dev) * 0.5).to(torch.bfloat16)
        pick = picks(R)
        # reference: generic module
        gs = G.Scratch(gen3, R, 9)
        ref = G.routed(x, pick.reshape(-1).contiguous(), None, gen3, gs, None, R, 7.0).clone().view(R, 9, D)[:, :8]
        M._ext = (lambda e: (lambda: e))(K3)
        f, y = run_family(K3, fam3, x, pick, R, prompt)
        M._ext = (lambda: K3)
        f(); torch.cuda.synchronize()
        out = y.view(R, 9, D)[:, :8]
        err = float((out - ref).abs().max() / ref.abs().max())
        cos = float(torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0))
        it = 200 if R <= 16 else 20
        t_k3 = timeit(f, it)
        t_gen = timeit(lambda: G.routed(x, pick.reshape(-1).contiguous(), None, gen3, gs, None, R, 7.0), it)
        print(f"R={R:5d} {'prompt' if prompt else 'decode'}: k3 {t_k3:9.1f} us | generic {t_gen:9.1f} us | x{t_gen/t_k3:4.2f} | maxrel {err:.2e} cos {cos:.6f}", flush=True)
del fam3, gen3, raw3; torch.cuda.empty_cache()
raw4 = load_layer(P4); fam4 = family_experts(raw4)
M._ext = lambda: EXT4
os.environ["TF_GLM_EXL3_LOADS"] = "nc"
for R, prompt in ((1, False), (4, False), (16, False), (256, True), (2048, True)):
    torch.manual_seed(R); x = (torch.randn(R, D, device=dev) * 0.5).to(torch.bfloat16); pick = picks(R)
    f, y = run_family(EXT4, fam4, x, pick, R, prompt); M._ext = lambda: EXT4
    print(f"R={R:5d} {'prompt' if prompt else 'decode'}: family 4-bit {timeit(f, 200 if R <= 16 else 20):9.1f} us", flush=True)
