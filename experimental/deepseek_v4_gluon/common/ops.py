# SPDX-License-Identifier: MIT
"""Device math and lane primitives of the FlyDSL mono kernels
(``vllm/models/deepseek_v41/amd/mono/common/ops.py``), in Gluon.

Values are one element a thread (``sync.T512``). The approximate hardware math
is the exact instruction the FlyDSL LLVM intrinsics lower to (``v_rsq_f32``,
``v_rcp_f32``, ``v_exp_f32``), so results match bit for bit. Lane exchanges go
through ``ds_bpermute``: the butterflies reduce with commutative ops (add, max),
where any exchange gives the same bits as FlyDSL's DPP / permlane swaps."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl



# ---------------------------------------------------------------- hw math
@gluon.jit
def hw_rsq(x):
    return gl.inline_asm_elementwise("v_rsq_f32 $0, $1", "=v,v", [x], gl.float32, True, 1)


@gluon.jit
def hw_rcp(x):
    return gl.inline_asm_elementwise("v_rcp_f32 $0, $1", "=v,v", [x], gl.float32, True, 1)


@gluon.jit
def hw_exp2(x):
    return gl.inline_asm_elementwise("v_exp_f32 $0, $1", "=v,v", [x], gl.float32, True, 1)


@gluon.jit
def fresh(v):
    """``v`` through an opaque move: lane math built on it stays at its use
    instead of being hoisted out of a task loop into long-lived registers."""
    return gl.inline_asm_elementwise(
        "v_mov_b32 $0, $1", "=v,v", [v.to(gl.int32)], gl.int32, False, 1
    )


# ---------------------------------------------------------------- bf16 words
@gluon.jit
def bf16_round(x):
    return x.to(gl.bfloat16).to(gl.float32)


@gluon.jit
def bf_lo(w):
    """f32 of a packed bf16 pair word's low half."""
    return (w.to(gl.uint32, bitcast=True) << 16).to(gl.float32, bitcast=True)


@gluon.jit
def bf_hi(w):
    return (w.to(gl.uint32, bitcast=True) & 0xFFFF0000).to(gl.float32, bitcast=True)


@gluon.jit
def bf16_pair(a, b):
    """Two f32 -> one u32 word (bf16(a) low, bf16(b) high)."""
    lo = a.to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint32)
    hi = b.to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint32)
    return lo | (hi << 16)


@gluon.jit
def bf16_bits_f32(bits):
    """A bf16's 16 bits (the low half of a u32) as f32."""
    return ((bits.to(gl.uint32, bitcast=True) & 0xFFFF) << 16).to(gl.float32, bitcast=True)


# ---------------------------------------------------------------- lanes
@gluon.jit
def lane_ids(like):
    """This thread's lane (0..63) in ``like``'s one-element-a-thread layout."""
    return gl.inline_asm_elementwise(
        "v_mbcnt_lo_u32_b32 $0, -1, 0\nv_mbcnt_hi_u32_b32 $0, -1, $0",
        "=v,v",
        [like.to(gl.int32)],
        gl.int32,
        True,
        1,
    )


@gluon.jit
def lane_gather(v, src_lane):
    """Lane ``src_lane``'s value of ``v`` (32-bit), a source lane a lane."""
    bits = v.to(gl.uint32, bitcast=True)
    got = gl.inline_asm_elementwise(
        "ds_bpermute_b32 $0, $1, $2\ns_waitcnt lgkmcnt(0)",
        "=v,v,v",
        [(src_lane * 4).to(gl.int32), bits],
        gl.uint32,
        True,
        1,
    )
    return got.to(v.dtype, bitcast=True)


@gluon.jit
def xshfl(v, lane, OFF: gl.constexpr):
    """Lane ``lane ^ OFF``'s value of ``v``."""
    return lane_gather(v, lane ^ OFF)


@gluon.jit
def xadd(v, lane, OFF: gl.constexpr):
    return v + xshfl(v, lane, OFF)


@gluon.jit
def xmax(v, lane, OFF: gl.constexpr):
    return gl.maximum(v, xshfl(v, lane, OFF))


@gluon.jit
def xmin(v, lane, OFF: gl.constexpr):
    return gl.minimum(v, xshfl(v, lane, OFF))


@gluon.jit
def sum_desc6(v, lane):
    """Butterfly sum over offsets 32, 16, 8, 4, 2, 1 (in that order)."""
    v = xadd(v, lane, 32)
    v = xadd(v, lane, 16)
    v = xadd(v, lane, 8)
    v = xadd(v, lane, 4)
    v = xadd(v, lane, 2)
    return xadd(v, lane, 1)


@gluon.jit
def sum_asc6(v, lane):
    """Butterfly sum over offsets 1, 2, 4, 8, 16, 32."""
    v = xadd(v, lane, 1)
    v = xadd(v, lane, 2)
    v = xadd(v, lane, 4)
    v = xadd(v, lane, 8)
    v = xadd(v, lane, 16)
    return xadd(v, lane, 32)


@gluon.jit
def sum_desc4(v, lane):
    """Butterfly sum over offsets 8, 4, 2, 1."""
    v = xadd(v, lane, 8)
    v = xadd(v, lane, 4)
    v = xadd(v, lane, 2)
    return xadd(v, lane, 1)


@gluon.jit
def max_desc6(v, lane):
    v = xmax(v, lane, 32)
    v = xmax(v, lane, 16)
    v = xmax(v, lane, 8)
    v = xmax(v, lane, 4)
    v = xmax(v, lane, 2)
    return xmax(v, lane, 1)


@gluon.jit
def min_desc6(v, lane):
    v = xmin(v, lane, 32)
    v = xmin(v, lane, 16)
    v = xmin(v, lane, 8)
    v = xmin(v, lane, 4)
    v = xmin(v, lane, 2)
    return xmin(v, lane, 1)


@gluon.jit
def ballot(pred):
    """The wave's lanes where ``pred`` holds, a bit a lane (u64, wave-uniform)."""
    return gl.inline_asm_elementwise(
        "v_cmp_ne_u32 $0, 0, $1", "=s,v", [pred.to(gl.uint32)], gl.uint64, True, 1
    )


@gluon.jit
def popcount64(x):
    lo = (x & 0xFFFFFFFF).to(gl.uint32)
    hi = (x >> 32).to(gl.uint32)
    return gl.inline_asm_elementwise(
        "v_bcnt_u32_b32 $0, $1, 0\nv_bcnt_u32_b32 $0, $2, $0",
        "=&v,v,v",
        [lo, hi],
        gl.uint32,
        True,
        1,
    ).to(gl.int32)


@gluon.jit
def popcount32(x):
    return gl.inline_asm_elementwise(
        "v_bcnt_u32_b32 $0, $1, 0", "=v,v", [x.to(gl.uint32, bitcast=True)], gl.uint32, True, 1
    ).to(gl.int32)


@gluon.jit
def lanes_below(bal):
    """How many lanes of ballot ``bal`` are below this lane (v_mbcnt)."""
    lo = (bal & 0xFFFFFFFF).to(gl.uint32)
    hi = (bal >> 32).to(gl.uint32)
    return gl.inline_asm_elementwise(
        "v_mbcnt_lo_u32_b32 $0, $1, 0\nv_mbcnt_hi_u32_b32 $0, $2, $0",
        "=&v,v,v",
        [lo, hi],
        gl.uint32,
        True,
        1,
    ).to(gl.int32)


@gluon.jit
def uniform(v):
    """``v`` read from the wave's first active lane (wave-uniform)."""
    return gl.inline_asm_elementwise(
        "v_readfirstlane_b32 $0, $1", "=s,v", [v.to(gl.int32)], gl.int32, True, 1
    )
