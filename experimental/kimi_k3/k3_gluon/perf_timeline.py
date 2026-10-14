"""Optional timestamps for diagnostic kernels, absent from serving launches."""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

ALL_CTAS = gl.constexpr(False)
PERSISTENT_CTAS = gl.constexpr(256)


@g.jit
def stamp(Timeline, stage: gl.constexpr, pid):
    if Timeline is not None:
        if ALL_CTAS or pid == 0:
            ticks = gl.inline_asm_elementwise(
                "s_memrealtime $0\n s_waitcnt lgkmcnt(0)",
                "=s,v,~{memory}",
                [pid],
                gl.uint64,
                False,
                1,
            )
            gl.store(Timeline + stage * PERSISTENT_CTAS + pid, ticks)
