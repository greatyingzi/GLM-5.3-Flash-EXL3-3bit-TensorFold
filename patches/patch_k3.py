#!/usr/bin/env python3
"""[tf-3bit-kernel] route 3-bit mcg EXL3 experts (int16 trellis [..., 48]) through the family's tuned kernels built
for 24-word tiles (exl3k3.cu), instead of the generic multi-width module. TF_GLM_EXL3_K3=0: the generic module."""
import pathlib, shutil
C = pathlib.Path("/usr/local/lib/python3.12/dist-packages/tensorfold/families/glm5_next/cuda")
for f in ("exl3k3.cu", "exl3k3.cpp"):
    shutil.copy(f"/opt/tf-3bit/{f}", C / f)

def rep(path, old, new):
    p = C / path; s = p.read_text()
    assert s.count(old) == 1, (path, old[:60], s.count(old)); p.write_text(s.replace(old, new))

rep("exl3_mm.py", '''    if trellis.dtype != torch.int16 or trellis.shape[-1] != 64:
        raise ValueError("only 4-bit EXL3 trellises (int16 [..., 64]) are supported")
    return trellis.contiguous().view(torch.int32)
''', '''    if trellis.dtype != torch.int16 or trellis.shape[-1] not in (64, 48):
        raise ValueError("only 4-bit and 3-bit EXL3 trellises (int16 [..., 64] / [..., 48]) are supported")
    return trellis.contiguous().view(torch.int32)


@lru_cache(maxsize=1)
def _ext3():
    """[tf-3bit-kernel] the same kernels for 3-bit tiles (24 words): exl3k3.cu, generated from exl3.cu."""
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_glm_exl3k3_v1", sources=[str(here / "exl3k3.cpp"), str(here / "exl3k3.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


def k3_enabled() -> bool:
    """TF_GLM_EXL3_K3 (default on): 3-bit mcg experts through exl3k3 (0: the generic multi-width module)."""
    return os.environ.get("TF_GLM_EXL3_K3", "1") != "0"
''')
rep("exl3_mm.py", '''        return

    ext = _ext()
    slots, P = s.slots, s.rows * s.slots''', '''        return

    ext = _ext3() if ex.gt.shape[-1] == 24 else _ext()      # [tf-3bit-kernel] 24-word (3-bit) tiles
    slots, P = s.slots, s.rows * s.slots''')
rep("weights.py", '''        legacy = (cfg.bits == 4 and cb == "mcg")
        parts = {}''', '''        legacy = (cfg.bits == 4 and cb == "mcg")
        if not legacy and cb == "mcg":     # [tf-3bit-kernel] 3-bit mcg (trellis [..., 48]): the family kernels' k3 build
            from .exl3_mm import k3_enabled
            legacy = k3_enabled() and rd.get(PREFIX + p + "experts.0.gate_proj.trellis").shape[-1] == 48
        parts = {}''')
print("patched k3")
