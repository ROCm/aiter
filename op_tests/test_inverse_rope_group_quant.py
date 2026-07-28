# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + benchmark for inverse_rope_group_quant, and its HIP-graph check.

The default run checks the fused op and an unfused two-kernel baseline against a
torch reference and prints the perf table. ``--graph`` additionally captures the
op in a HIP graph and replays it on fresh data: the kernel picks
THREAD_DATA_SIZE / K_PER_BLOCK from ``s`` on the *host*, so that choice has to
bake into the graph correctly.
"""

import argparse
import itertools
from collections import namedtuple

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.inverse_rope_group_quant import (
    inverse_rope_group_quant as inverse_rope_group_quant_cpp,
)
from aiter.ops.quant import dynamic_per_group_scaled_quant
from aiter.ops.triton.rope.rope import RotateStyle, _rope_cached_bwd
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)

torch.set_default_device("cuda")

# The HIP kernel widens its cross-lane amax reduction past a 16-lane DPP row with
# __builtin_amdgcn_permlane16_swap / permlane32_swap, which are gfx950+. Those
# instantiate whenever THREADS_PER_GROUP >= 32, i.e. the s <= 4 tier
# (THREAD_DATA_SIZE=2 -> 64 lanes per group), so the module does not build on
# gfx942 today.
SUPPORTED_GFX = ["gfx950"]

# Positions stay unique for every swept s, so cos/sin rows are not reused across
# tokens -- reuse would inflate the L2 hit rate versus a real decode batch spread
# over the context. 64Ki rows x rd/2 x 2B x (cos + sin) ~= 8MiB.
MAX_POS = 65536

# The kernel seeds each group's amax with this floor so an all-zero group cannot
# produce a zero scale (graph warmup, padded rows). Mirrors
# kFp8QuantAbsmaxFloorF32 in csrc/kernels/inverse_rope_group_quant.cu.
AMAX_FLOOR = 1e-8

# One row of the perf table: `once` is called for the correctness check, `bench`
# is the timed call, `ref` is the (dq, scale_byte) reference it is checked
# against, `scale_transposed` is the layout `once`'s scale should have, and
# `tol` / `scale_tol` are its (rtol, atol) pairs.
Cand = namedtuple("Cand", "once bench ref scale_transposed tol scale_tol")

# The fused op is a bit-for-bit match against the torch reference, so its scale
# bytes are compared exactly and its dequantized values only carry fp8 rounding.
FUSED_TOL, FUSED_SCALE_TOL = (1e-2, 1e-2), (0, 0)
# An unfused pair cannot be held to that: its rope leg is a different
# implementation, so a value one fp32 ulp from the reference's can round to the
# neighbouring bf16, and a group amax sitting on a power-of-two boundary then
# flips one e8m0 exponent -- which rescales that whole group by 2x. Measured at
# s=2048: 14 of 131072 scale bytes off by one, and every value delta is exactly
# one fp8 step (0.03125 at this amplitude, e4m3 having 3 mantissa bits). So allow
# one step on each. The check is still worth running -- a wrong rope convention
# or group mapping misses on ~100% of elements, not 0.03%.
UNFUSED_TOL, UNFUSED_SCALE_TOL = (5e-2, 5e-2), (0, 1)


def _e8m0_round_up(amax):
    """ceil_pow2(amax / fp8_max) -> (f32 dequant scale, e8m0 exponent byte).

    Bit-for-bit mirror of fp_f32_to_e8m0_scale<RoundUp, FP8_E4M3{,_FNUZ}> in
    csrc/include/mx_quant_utils.h so the bytes can be compared at rtol=atol=0.
    torch.finfo(dtypes.fp8).max picks the same max_pos the kernel compiles
    against (gfx942 e4m3fnuz = 240, gfx950 OCP e4m3fn = 448).
    """
    u32 = (amax * (1.0 / torch.finfo(dtypes.fp8).max)).contiguous().view(dtypes.i32)
    exponent = (u32 >> 23) & 0xFF
    bump = (exponent < 0xFF) & ((u32 & 0x7FFFFF) != 0)
    exponent = torch.where(bump, exponent + 1, exponent)
    return _e8m0_byte_to_scale(exponent), exponent.to(dtypes.u8)


def _e8m0_byte_to_scale(byte):
    """e8m0 exponent byte -> f32 dequant scale 2^(byte-127)."""
    return (byte.to(dtypes.i32) << 23).view(dtypes.fp32)


def _scale_bytes(scale):
    """Scale buffer -> uint8 view (both paths hand back fp8_e8m0 today)."""
    return scale if scale.dtype == dtypes.u8 else scale.view(dtypes.u8)


def _check_scale_layout(scale, s, transpose_scale, name):
    """Assert the scale buffer's strides really are the layout that was asked for.

    Logical shape is [s, g, ks] either way, so comparing values alone cannot tell
    the layouts apart -- only the strides can. Under transpose_scale the storage is
    group-major [g, ks, s], which makes each group's [s, ks] slice compact
    column-major, i.e. what a preshuffled-B blockscale GEMM reads.
    """
    if transpose_scale:
        assert scale.stride(0) == 1 and scale.stride(2) == s, (
            f"{name}: expected column-major scale (storage [g, ks, s]), "
            f"got strides {scale.stride()}"
        )
    else:
        assert (
            scale.stride(2) == 1
        ), f"{name}: expected row-major scale, got strides {scale.stride()}"


def _make_inputs(s, h, head_dim, rd, dtype, seed=0):
    """Build (o, positions, cos, sin) for one config.

    cos/sin are the 2D [max_pos, rd//2] the op takes. A model holding the
    singleton batch/head dims (atom deepseek_v4._build_cos_sin_cache does
    unsqueeze(-2) twice, landing on [max_pos, 1, 1, rd//2] -- aiter
    rope_cached_positions' layout, not [max_pos, rd//2, 1, 1]) reshapes at its
    own call site, the way run_inverse_rope_inplace does for the triton rope.
    Shared by the sweep and the graph check so the two cannot drift.
    """
    torch.manual_seed(seed)
    positions = torch.arange(s, dtype=dtypes.i64) % MAX_POS
    # /10 keeps a group's amax away from fp8 saturation, like a real
    # post-softmax attention output.
    o = torch.randn((s, h, head_dim), dtype=dtype) / 10
    theta = torch.randn((MAX_POS, rd // 2), dtype=dtypes.fp32)
    cos = torch.cos(theta).to(dtype).contiguous()
    sin = torch.sin(theta).to(dtype).contiguous()
    return o, positions, cos, sin


def _alloc_outputs(s, g, d, group_size, transpose_scale):
    """Pre-allocate (x_fp8, x_scale) the way the wrapper would.

    Mirrors aiter/ops/inverse_rope_group_quant.py: under transpose_scale the
    scale is stored group-major [g, ks, s] and viewed back as [s, g, ks].
    """
    from aiter.utility.dtypes import get_dtype_fp8

    x_fp8 = torch.empty((s, g, d), dtype=get_dtype_fp8())
    ks = d // group_size
    if transpose_scale:
        x_scale = torch.empty((g, ks, s), dtype=dtypes.fp8_e8m0).permute(2, 0, 1)
    else:
        x_scale = torch.empty((s, g, ks), dtype=dtypes.fp8_e8m0)
    return x_fp8, x_scale


def run_torch(
    o, positions, cos, sin, num_groups, quant_group_size, rd, roundtrip=False
):
    """Reference: inverse GPT-J RoPE on the rope tail, then e8m0 FP8 group quant.

    Returns ``(dq, scale_byte)`` -- the dequantized rows as fp32 ``[s, g, d]``
    and the e8m0 scale bytes as ``[s, g, ks]``. Reference only: not timed and not
    in the table.

    ``roundtrip`` casts the roped values back through the input dtype before
    quantizing, which is what any *unfused* pair of kernels is forced to do:
    the rope kernel has to land its result in a real bf16 buffer for the quant
    kernel to read. The fused op keeps it in fp32 registers, so the two want
    different references -- see run_unfused.
    """
    s, h, _ = o.shape
    ref = o.to(dtypes.fp32).clone()
    c = cos.index_select(0, positions).to(dtypes.fp32)
    sn = sin.index_select(0, positions).to(dtypes.fp32)
    pair = ref[..., -rd:].reshape(s, h, rd // 2, 2)
    even, odd = pair[..., 0], pair[..., 1]
    c, sn = c[:, None, :], sn[:, None, :]
    ref[..., -rd:] = torch.stack(
        (even * c + odd * sn, odd * c - even * sn), dim=-1
    ).reshape(s, h, rd)
    if roundtrip:
        # Only the rope tail moves: the nope part still holds the exact input
        # value, so casting it is a no-op.
        ref = ref.to(o.dtype).to(dtypes.fp32)

    # Flattening a contiguous [s, h, head_dim] to [s, g, d] is exactly the
    # kernel's row mapping: o index = s*h*head_dim + g*d + elem.
    groups = ref.reshape(s, num_groups, -1, quant_group_size)
    amax = groups.abs().amax(-1).clamp_min(AMAX_FLOOR)
    dq_scale, scale_byte = _e8m0_round_up(amax)
    # The kernel quantizes with * (1 / dq_scale); dq_scale is a power of two, so
    # the reciprocal is exact and this matches its rounding.
    q = (groups * (1.0 / dq_scale)[..., None]).to(dtypes.fp8)
    dq = q.to(dtypes.fp32) * dq_scale[..., None]
    return dq.reshape(s, num_groups, -1), scale_byte


def run_inverse_rope_inplace(x, positions, cos, sin, rd):
    """The rope leg on its own: triton inverse RoPE over ``x``'s rope tail.

    Same call atom's ``_V4RoPE.inverse`` makes. Shared with run_unfused so the
    "rope only" column is exactly that baseline's first kernel, not a lookalike.
    Overwrites ``x``. ``cos``/``sin`` come in as the 2D ``[max_pos, rd // 2]``
    cache and grow the singleton batch/head dims here, since taking 4 cos strides
    is this triton kernel's requirement rather than the cache's shape;
    ``positions`` is 2D ``[s, 1]``.
    """
    cos = cos.unsqueeze(-2).unsqueeze(-2)
    sin = sin.unsqueeze(-2).unsqueeze(-2)
    # The triton rope infers the rope width from cos and only handles
    # rotary_dim == d (no nope) or d // 2 -- never d // 8 == rd here -- so the
    # rope tail goes in as its own [s, b, h, rd] tensor instead of passing the
    # full head_dim with nope_first. That slice is strided, which is fine: the
    # wrapper forwards x.stride() to the kernel.
    tail = x[..., -rd:].unsqueeze(1)
    # _rope_cached_bwd rather than the public rope_cached_positions_bwd: only the
    # fwd wrappers have an inplace variant, and the public bwd allocates a
    # compact [s, b, h, rd] out that would need a second kernel to scatter back
    # under the nope part. out=x + inplace=True is what rope_cached_fwd_inplace
    # passes.
    _rope_cached_bwd(
        tail,
        tail,
        cos,
        sin,
        positions,
        None,
        RotateStyle.GPTJ,
        reuse_freqs_front_part=True,
        nope_first=False,
        inplace=True,
    )
    return x


def run_unfused(x, positions, cos, sin, num_groups, quant_group_size, rd, out):
    """Unfused baseline: triton inverse RoPE in place, then the HIP group quant.

    This is the two-kernel path the fused op replaces -- atom's
    ``_V4RoPE.inverse`` followed by a group quant. It emits the same format, not
    just a comparable time: ``dynamic_per_group_scaled_quant`` writes an e8m0
    exponent byte when handed an fp8_e8m0 scale buffer. It is checked against the
    roundtrip reference (see run_torch), being a two-kernel path.

    Rotates in place the way atom's inverse does, so it overwrites ``x``.
    """
    s = x.shape[0]
    run_inverse_rope_inplace(x, positions, cos, sin, rd)
    x_fp8, x_scale = out
    # shuffle_scale=False always: at group_size == 32 shuffle_scale means the MX
    # hardware swizzle rather than the plain transpose this op's transpose_scale
    # does, so a row-major scale is the one layout that stays comparable across
    # every swept group size. The scale is 1/group_size of the bytes written, so
    # its layout barely moves the baseline's time.
    dynamic_per_group_scaled_quant(
        x_fp8,
        x.view(s, num_groups, -1),
        x_scale,
        group_size=quant_group_size,
        shuffle_scale=False,
    )
    return x_fp8, x_scale


@benchmark()
def test_inverse_rope_group_quant(
    s, h, g, head_dim, rd, group_size, dtype, scale_layout
):
    d = h * head_dim // g
    scale_n = d // group_size
    transpose_scale = scale_layout == "col"

    o, positions, cos, sin = _make_inputs(s, h, head_dim, rd, dtype)

    ref = run_torch(o, positions, cos, sin, g, group_size, rd)
    ref_rt = run_torch(o, positions, cos, sin, g, group_size, rd, roundtrip=True)

    # The op is not handed output buffers in the model path
    # (atom/model_ops/v4_kernels/inverse_rope_group_quant.py forwards exactly
    # these args), so it allocates its own x_fp8 + x_scale. transpose_scale only
    # changes the scale's strides (column-major, group-major storage; the same
    # thing atom's per_1x128 transpose_scale does), not its logical [s, g, ks]
    # shape, so both layouts share one reference; _check_scale_layout pins strides.
    kwargs = {
        "num_groups": g,
        "quant_group_size": group_size,
        "transpose_scale": transpose_scale,
    }

    def fused():
        return inverse_rope_group_quant_cpp(o, positions, cos, sin, **kwargs)

    pos_r = positions.view(s, 1)
    # Timed on a dedicated scratch because the rope leg is in place. Re-applying
    # an inverse rotation across benchmark iterations does the same work and,
    # being norm-preserving, cannot push values out of range -- but it does leave
    # the buffer rotated n times, so correctness runs on a fresh copy instead.
    unfused_scratch = o.clone()
    unfused_out = _alloc_outputs(s, g, d, group_size, transpose_scale=False)

    def unfused_bench():
        return run_unfused(
            unfused_scratch, pos_r, cos, sin, g, group_size, rd, unfused_out
        )

    def unfused_once():
        return run_unfused(
            o.clone(),
            pos_r,
            cos,
            sin,
            g,
            group_size,
            rd,
            _alloc_outputs(s, g, d, group_size, transpose_scale=False),
        )

    # The unfused baseline always writes a row-major scale (see run_unfused), so
    # its cells repeat across both scale_layout rows.
    funcs = {
        "cpp": Cand(fused, fused, ref, transpose_scale, FUSED_TOL, FUSED_SCALE_TOL),
        "unfused": Cand(
            unfused_once, unfused_bench, ref_rt, False, UNFUSED_TOL, UNFUSED_SCALE_TOL
        ),
    }

    # inverse RoPE: 2 mul + 1 add per rope-tail element.
    # group quant: one |x| compare for the group amax + one scale multiply, per element.
    flops = s * h * rd * 3 + s * h * head_dim * 2
    # read o, plus one cos and one sin row per token (all heads of a token share
    # the row); write fp8 data at 1B/elem plus one e8m0 scale byte per group.
    # This is the fused op's traffic, so the unfused baseline's TB/s is an
    # effective figure over the same logical work -- it really moves more,
    # round-tripping the bf16 rows once between its two kernels.
    nbytes = (
        o.numel() * o.element_size()
        + s * (rd // 2) * 2 * cos.element_size()
        + s * g * d
        + s * g * scale_n
    )

    ret = {"gfx": get_gfx()}
    for name, cand in funcs.items():
        ref_dq, ref_scale = cand.ref
        x_fp8, x_scale = cand.once()
        _, us = run_perftest(cand.bench)
        _check_scale_layout(x_scale, s, cand.scale_transposed, name)
        scale_u8 = _scale_bytes(x_scale)
        dq = (
            x_fp8.to(dtypes.fp32).reshape(s, g, scale_n, group_size)
            * _e8m0_byte_to_scale(scale_u8)[..., None]
        ).reshape(s, g, d)
        # Dequantized values carry both the rope math and the scale, so a wrong
        # group scale shows up here as a whole-group error.
        err = checkAllclose(
            ref_dq,
            dq,
            rtol=cand.tol[0],
            atol=cand.tol[1],
            msg=f"{name}: inverse_rope_group_quant out",
        )
        # The e8m0 exponent byte feeds the GEMM's scale path, so the fused op is
        # held to it exactly (see FUSED_SCALE_TOL).
        scale_err = checkAllclose(
            ref_scale.to(dtypes.fp32),
            scale_u8.to(dtypes.fp32),
            rtol=cand.scale_tol[0],
            atol=cand.scale_tol[1],
            msg=f"{name}: inverse_rope_group_quant e8m0 scale",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
        ret[f"{name} scale err"] = scale_err

    return ret


def check_graph(s, h, g, head_dim, rd, group_size, dtype, scale_layout):
    """Capture the op in a HIP graph, replay on fresh data, compare against eager.

    Not part of the perf table: this is a pass/fail check that the host-side
    dispatch tier and the pre-allocated buffers survive capture/replay.
    """
    transpose_scale = scale_layout == "col"
    d = h * head_dim // g
    o, positions, cos, sin = _make_inputs(s, h, head_dim, rd, dtype)
    # Pre-allocate so the captured region owns no allocations and the buffers
    # replay in place.
    x_fp8, x_scale = _alloc_outputs(s, g, d, group_size, transpose_scale)
    kwargs = {
        "num_groups": g,
        "quant_group_size": group_size,
        "transpose_scale": transpose_scale,
        "x_fp8": x_fp8,
        "x_scale": x_scale,
    }

    # Warm up outside capture: the first call JIT-compiles / loads the module and
    # initialises the dispatch statics, neither of which may happen inside a
    # capture region.
    for _ in range(3):
        inverse_rope_group_quant_cpp(o, positions, cos, sin, **kwargs)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        inverse_rope_group_quant_cpp(o, positions, cos, sin, **kwargs)

    # Replay on new data, then compare against an eager run on the same data.
    o2, positions2, cos2, sin2 = _make_inputs(s, h, head_dim, rd, dtype, seed=7)
    o.copy_(o2)
    positions.copy_(positions2)
    cos.copy_(cos2)
    sin.copy_(sin2)
    graph.replay()
    torch.cuda.synchronize()
    graph_fp8, graph_scale = x_fp8.clone(), _scale_bytes(x_scale).clone()

    eager_fp8, eager_scale = inverse_rope_group_quant_cpp(
        o,
        positions,
        cos,
        sin,
        num_groups=g,
        quant_group_size=group_size,
        transpose_scale=transpose_scale,
    )
    torch.cuda.synchronize()

    fp8_match = torch.equal(graph_fp8.view(dtypes.u8), eager_fp8.view(dtypes.u8))
    scale_match = torch.equal(graph_scale, _scale_bytes(eager_scale))
    # Mirrors the host dispatch in csrc/kernels/inverse_rope_group_quant.cu.
    tds = 2 if s <= 4 else (4 if s <= 128 else 8)
    kpb = 1 if s <= 128 else (2 if s <= 512 else 4)
    aiter.logger.info(
        "graph s=%-6d h=%d g=%d gs=%-3d %s tier(TDS=%d,KPB=%d)  "
        "graph==eager: fp8=%s scale=%s",
        s,
        h,
        g,
        group_size,
        scale_layout,
        tds,
        kpb,
        fp8_match,
        scale_match,
    )
    assert fp8_match and scale_match, (
        f"graph replay diverged from eager at s={s} h={h} g={g} "
        f"group_size={group_size} scale_layout={scale_layout}"
    )


def main():
    # Whole-op arch gate lives here: @benchmark always returns the call-args
    # dict, so returning from inside the test fn would still emit a NaN row.
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "inverse_rope_group_quant unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp16"]],
        nargs="*",
        # The trailing comma is load-bearing: argparse runs `type` over a string
        # default, and str2Dtype only returns a tuple when it sees one -- plain
        # "bf16" would yield a bare dtype that the sweep below cannot iterate.
        default="bf16,",
        metavar="{bf16,fp16}",
        help="""Data type of o / cos / sin.
        e.g.: -d bf16""",
    )
    parser.add_argument(
        "-b",
        "--hg",
        type=dtypes.str2tuple,
        nargs="*",
        # (n_local_heads, n_local_groups) = (n_heads // tp, o_groups // tp).
        # deepseek_v4.ModelArgs has n_heads=128, head_dim=512, o_groups=16, so
        # d = n_local_heads*head_dim/n_local_groups = 4096 is tp-invariant and
        # every real config satisfies n_local_heads = 8 * n_local_groups:
        #   V4-Pro   (o_groups=16): tp8 (16,2), tp4 (32,4), tp2 (64,8), dp/tp1 (128,16)
        #   V4-Flash (o_groups=8) : tp8 (8,1),  tp2 (32,4)
        # Default to the two smallest so the sweep also covers the g=1 case
        # (degenerate row/g division) without allocating a 2GiB o at s=16384.
        default=[(16, 2), (8, 1)],
        help="""(n_local_heads, n_local_groups) of the attention output.
        e.g.: -b 16,2 64,8""",
    )
    parser.add_argument(
        "-s",
        "--tokens",
        type=int,
        nargs="*",
        # Spans all three dispatch tiers of the HIP kernel: s<=4 picks
        # THREAD_DATA_SIZE=2, s<=128 picks 4, above that 8; K_PER_BLOCK steps
        # 1 -> 2 -> 4 at s>128 and s>512.
        default=[1, 8, 32, 128, 512, 1024, 2048, 4096, 8192, 16384],
        help="""Number of tokens s.
        e.g.: -s 1 128 8192""",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        nargs="*",
        # The HIP template currently instantiates HEAD_DIM=512 only.
        default=[512],
        help="""Attention head dim.
        e.g.: --head-dim 512""",
    )
    parser.add_argument(
        "--rope-dim",
        type=int,
        nargs="*",
        # deepseek_v4 rope_head_dim; the HIP template instantiates RD=64 only.
        default=[64],
        help="""Rotary dim applied to each head's tail.
        e.g.: --rope-dim 64""",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        nargs="*",
        # The wo_a path uses 128; 32/64 exercise the kernel's other group tiers.
        default=[128],
        help="""Quant group size along d.
        e.g.: --group-size 32 64 128""",
    )
    parser.add_argument(
        "-l",
        "--scale-layout",
        type=str,
        choices=["row", "col"],
        nargs="*",
        default=["row", "col"],
        help="""e8m0 scale storage, both logically [s, g, ks]:
        row = contiguous [s, g, ks],
        col = column-major, storage [g, ks, s] (atom per_1x128 transpose_scale,
              the layout the preshuffled-B blockscale GEMM reads).
        e.g.: -l col""",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        # 300 / 700 sit just past the K_PER_BLOCK steps at s>128 and s>512, so
        # `-s 1 4 32 128 300 512 700 2048 --graph` covers every dispatch tier.
        help="""Also run the HIP-graph capture/replay check over the same sweep.
        e.g.: --graph -s 1 4 32 128 300 512 700 2048""",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        df = []
        for (h, g), s, head_dim, rd, group_size, scale_layout in itertools.product(
            args.hg,
            args.tokens,
            args.head_dim,
            args.rope_dim,
            args.group_size,
            args.scale_layout,
        ):
            ret = test_inverse_rope_group_quant(
                s, h, g, head_dim, rd, group_size, dtype, scale_layout
            )
            df.append(ret)
            if args.graph:
                check_graph(s, h, g, head_dim, rd, group_size, dtype, scale_layout)
        df = pd.DataFrame(df)
        aiter.logger.info(
            "inverse_rope_group_quant summary (markdown):\n%s",
            df.to_markdown(index=False),
        )
        if args.graph:
            aiter.logger.info("all graph capture/replay checks passed")


if __name__ == "__main__":
    main()
