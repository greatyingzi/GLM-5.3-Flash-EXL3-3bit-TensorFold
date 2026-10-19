#!/usr/bin/env python3
"""v7：把传给通用 EXL3 内核的 pick 消毒（越界/垃圾值会被内核当真索引）。

依据（v6 契约实测）：
  [tf-v6] pick int32 shape=(16, 9) stride=(9,1) contig=True min=14 max=1014016334 ge_E=136
⇒ b.pick 是"最大窗口×slots"的常驻缓冲（16×9=144），本次只应使用 R*slots 个；
  其余含未初始化垃圾（max=1e9），且 144 个里有 136 个 ≥ E=288 ⇒ 内核据越界索引行事 ⇒ 崩。
  （同时实测 ids/ucount/members 契约全合法 ⇒ 不是它们）

做法：不动家族的缓冲区，复制一份 clamp 到 [0, E-1] 后传入，代价极微（int32 小张量）。
幂等；锚点取自实机原文（exl3_mm.py 的 generic 调用行）。
"""
import pathlib

TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/tensorfold/families/glm5_next/cuda/exl3_mm.py"
)

import re

_PAT = re.compile(
    r"^(?P<i>\s*)_generic\.routed\(x, pick, None, ex\.gen, gs, None, R, float\(limit\)\)",
    re.M,
)


def main() -> int:
    src = TARGET.read_text()
    if "[tf-v7]" in src:
        print("already patched")
        return 0
    m = _PAT.search(src)
    if m is None:
        print("FATAL: regex anchor not found", flush=True)
        return 2
    ind = m.group("i")
    repl = (
        f"{ind}_pick_clean = pick.reshape(-1).clamp(0, ex.count - 1).contiguous()  # [tf-v7]\n"
        f"{ind}_generic.routed(x, _pick_clean, None, ex.gen, gs, None, R, float(limit))"
    )
    TARGET.write_text(src[: m.start()] + repl + src[m.end():])
    print(f"patched (indent={len(ind)}):", TARGET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
