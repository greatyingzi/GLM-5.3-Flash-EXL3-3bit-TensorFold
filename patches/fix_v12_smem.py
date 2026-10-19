#!/usr/bin/env python3
"""v12 = 最终修复：给通用 EXL3 的 group_kernel 提高动态共享内存上限。

铁证（v11 矩阵实测，CUDA 亲口）：
  Requested dynamic shared memory size of 73728 bytes exceeds maximum
  dynamic shared memory size of 49152 bytes  → cuLaunchKernel CUDA_ERROR_INVALID_VALUE
  73728 = 1024(R) × 18(slots) × 4(int)  ⇒ 大窗口(slots=18)那次调用必挂；
  49 KB 是 CUDA 默认上限，而 experts.cu 里 cudaFuncSetAttribute 出现次数 = 0（缺这一句）。
  ⇒ 这就是所有 `ext.group` "invalid argument" 的最终根因；
     小参数(1×9×4=36B)永远测不出来，所以此前逐条排除都"看起来"成立。

做法：在 exl3x_group_cuda 首次调用前设置 MaxDynamicSharedMemorySize。
      改的是 .cu 源文件；torch JIT 按源码哈希重建 ⇒ 需清 /cache/torch_extensions。
缩进无关正则定位。
"""
import pathlib
import re

TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/tensorfold/cuda/exl3/experts.cu"
)

PAT = re.compile(
    r"^(?P<i>[ \t]*)const size_t smem = \(size_t\)R \* slots \* sizeof\(int\);",
    re.M,
)

_ADD_LINES = [
    "// [tf-v12] 大窗口下 R*slots*4 会超过 CUDA 默认 48KB 动态共享内存上限 ⇒ 必须提权",
    "static bool _tf_smem_ok = false;",
    "if (!_tf_smem_ok) {",
    "    cudaFuncSetAttribute(group_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);",
    "    _tf_smem_ok = true;",
    "}",
]


def main() -> int:
    src = TARGET.read_text()
    if "[tf-v12]" in src:
        print("already patched")
        return 0
    m = PAT.search(src)
    if m is None:
        print("FATAL: regex anchor not found", flush=True)
        return 2
    i = m.group("i")
    _blk = "".join(i + _ln + "\n" for _ln in _ADD_LINES)
    src = src[: m.start()] + _blk + src[m.start():]
    TARGET.write_text(src)
    print(f"patched (indent={len(i)}):", TARGET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
