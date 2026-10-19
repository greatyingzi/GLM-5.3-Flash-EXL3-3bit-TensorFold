#!/usr/bin/env python3
"""放宽 glm5_next 的 EXL3 变体门禁：允许 bits=3（3.5 经 int() 也是 3）。

上游要求与 EXL3_VARIANT 逐字段严格相等（bits=4/mcg/glm53_routed_experts_only）。
CUDA 参考解码（cuda/exl3.py: states/unpack/dequantize/forward）本身是 bit-width 参数化的
（形状按 16*bits 推导），因此 3bit 包只需要这里放行即可进入验证。

幂等：重复执行不叠加；锚点找不到直接报错退出（不静默放过）。
只在派生镜像里执行，基镜像不动。
"""
import pathlib
import sys

TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/tensorfold/families/glm5_next/__init__.py"
)

ANCHOR_VARIANT = 'EXL3_VARIANT = {"bits": 4, "codebook": "mcg", "scope": "glm53_routed_experts_only"}'
ADDITION = (
    "\n# 3-bit adaptation (tf-glm-exl3-3bit): same codebook/scope, lower bit width allowed\n"
    "EXL3_ALLOWED_BITS = (3, 4)"
)

ANCHOR_CMP = (
    '        if {k: (int(v) if k == "bits" and v is not None else v) '
    "for k, v in got.items()} != EXL3_VARIANT:"
)
REPLACEMENT_CMP = (
    '        _norm = {k: (int(v) if k == "bits" and v is not None else v) for k, v in got.items()}\n'
    '        _ok = (_norm.get("codebook") == EXL3_VARIANT["codebook"]\n'
    '               and _norm.get("scope") == EXL3_VARIANT["scope"]\n'
    '               and _norm.get("bits") in EXL3_ALLOWED_BITS)\n'
    "        if not _ok:"
)


def main() -> int:
    src = TARGET.read_text()
    if "EXL3_ALLOWED_BITS" in src and "if not _ok:" in src:
        print("already patched:", TARGET)
        return 0

    if ANCHOR_VARIANT not in src:
        print("FATAL: EXL3_VARIANT anchor not found in", TARGET, file=sys.stderr)
        return 2
    src = src.replace(ANCHOR_VARIANT, ANCHOR_VARIANT + ADDITION, 1)

    if ANCHOR_CMP not in src:
        print("FATAL: comparison anchor not found in", TARGET, file=sys.stderr)
        return 3
    src = src.replace(ANCHOR_CMP, REPLACEMENT_CMP, 1)

    TARGET.write_text(src)
    print("patched:", TARGET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
