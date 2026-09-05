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
import os
import sys
from collections import namedtuple

import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.inverse_rope_group_quant import (
    SCALE_LAYOUTS,
    scale_shape,
)
from aiter.ops.inverse_rope_group_quant import (
    inverse_rope_group_quant as inverse_rope_group_quant_cpp,
)
from aiter.ops.quant import dynamic_per_group_scaled_quant
from aiter.ops.triton.rope.rope import RotateStyle, _rope_cached_bwd
from aiter.test_common import (
    DATA_DISTS,
    benchmark,
    checkAllclose,
    fill,
    make_generator,
    print_json_table,
    run_perftest,
)

torch.set_default_device("cuda")

# The HIP kernel widens its cross-lane amax reduction past a 16-lane DPP row with
# __builtin_amdgcn_permlane16_swap / permlane32_swap, which are gfx950+. Those
# instantiate whenever THREADS_PER_GROUP >= 32, i.e. the s <= 4 tier
# (THREAD_DATA_SIZE=2 -> 64 lanes per group), so the module does not build on
# gfx942 today.
SUPPORTED_GFX = ["gfx950", "gfx1250", "gfx1201"]

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
# against, `scale_layout` is the layout `once`'s scale should have, and
# `tol` / `scale_tol` are its (rtol, atol) pairs.
Cand = namedtuple("Cand", "once bench ref scale_layout tol scale_tol")

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


def _unshuffle_mfma_scale(scale_shuffled, S, G, Ks, group_size):
    """Unshuffle mfma-layout scale [G, S_pad, Ks_pad] -> logical [S, G, Ks].

    Both chunk widths factorise into a single permute. The 256-byte [32_M, 8_K]
    tile splits as (k%4, s%16, (k/4)&1, (s/16)&1) at strides (64, 4, 2, 1); the
    64-byte [32_M, 2_K] chunk the kernel emits at group_size 128 splits as
    (s%16, k%2, (s/16)&1) at strides (4, 2, 1).
    """
    S_pad, Ks_pad = scale_shuffled.shape[1], scale_shuffled.shape[2]
    flat = _scale_bytes(scale_shuffled)
    if group_size == 128:
        chunks = flat.reshape(G, S_pad // 32, Ks_pad // 2, 16, 2, 2)
        out = chunks.permute(1, 5, 3, 0, 2, 4).reshape(S_pad, G, Ks_pad)
    else:
        tiles = flat.reshape(G, S_pad // 32, Ks_pad // 8, 4, 16, 2, 2)
        # -> (tile_m, (s/16)&1, s%16, G, tile_k, (k/4)&1, k%4), i.e. s and k
        # rebuilt most-significant first around the untouched G.
        out = tiles.permute(1, 6, 4, 0, 2, 5, 3).reshape(S_pad, G, Ks_pad)
    return out[:S, :, :Ks].contiguous()


def _unshuffle_n32k4_scale(scale_n32k4, S, G, Ks):
    """Unshuffle n32k4 scale [S_pad/32, G, Ks*32] -> logical [S, G, Ks].

    The last dim splits as (k//4, s%32, k%4), which is exactly the transpose
    ``aiter.ops.shuffle.shuffle_scale_n32k4`` applies to a weight scale -- so
    this reads back through the same permutation the consumer relies on.
    """
    n_super = scale_n32k4.shape[0]
    flat = _scale_bytes(scale_n32k4).view(n_super, G, Ks // 4, 32, 4)
    out = flat.permute(0, 3, 1, 2, 4).reshape(n_super * 32, G, Ks)
    return out[:S].contiguous()


def _unshuffle_scale(scale, s, g, ks, scale_layout, group_size):
    """Scale buffer in `scale_layout` -> logical [s, g, ks] uint8."""
    if scale_layout == "mfma_tile":
        return _unshuffle_mfma_scale(scale, s, g, ks, group_size)
    if scale_layout == "n32k4":
        return _unshuffle_n32k4_scale(scale, s, g, ks)
    return _scale_bytes(scale)


# --- cross-tree drift gate for the mfma_tile layout at group_size 128 ---------
#
# At group_size 128 the buffer this op emits *is* opus's `shuffle_scale_a(x, K,
# OPUS_SF_SHUF_SUB)`: one dword pairs two M subtiles `sub` rows apart crossed
# with two consecutive 128-blocks of K. `_unshuffle_mfma_scale` above is a
# hand-written inverse of that layout rather than a call into opus, because
# `shuffle_scale_a` does not exist in this tree -- so the two can drift silently
# and the failure mode is plausible wrong numbers, not an exception.
#
# The gate closes that by round-tripping a plain scale through opus's own
# forward and this file's inverse. It has to run opus in a **subprocess**: both
# trees ship a package named `aiter`, so a path insert here would resolve
# `aiter.ops.shuffle` out of whichever one is already in sys.modules -- this
# one, which has no shuffle_scale_a.
#
# `sub` is read from the opus traits header via its own single-source-of-truth
# accessor rather than hardcoded. Hardcoding it would make this gate a third
# copy of the constant it exists to police, and it is exactly the constant that
# moved once already (32 -> 16 when the producer's layout was made the shipped
# one).
_OPUS_REF_SNIPPET = r"""
import json, sys, torch
tree = sys.argv[1]
sys.path.insert(0, tree)
sys.path.insert(0, tree + "/csrc/opus_gemm")
from aiter.ops.shuffle import shuffle_scale_a
try:
    from opus_gemm_common import _opus_sf_shuf_sub
    sub = _opus_sf_shuf_sub()
except Exception as exc:                                  # header moved or renamed
    raise SystemExit(f"cannot read OPUS_SF_SHUF_SUB from {tree}: {exc}")
cases, out = json.loads(sys.argv[2]), {}
for g, s, ks in cases:
    torch.manual_seed(g * 1000 + s * 10 + ks)
    plain = torch.randint(0, 255, (g, s, ks), dtype=torch.uint8)
    out[f"{g},{s},{ks}"] = (plain, shuffle_scale_a(plain, ks * 128, sub),
                            shuffle_scale_a(plain, ks * 128, sub * 2))
torch.save({"sub": sub, "cases": out}, sys.argv[3])
"""


def check_opus_layout_identity(opus_tree, verbose=True):
    """mfma_tile @ group 128 == opus `shuffle_scale_a`; raises on drift.

    Returns the `sub` the opus tree is built with, so the caller can log which
    layout was actually checked.
    """
    import json
    import math
    import subprocess
    import tempfile

    # Ragged s on both sides of the 2*sub row block and an odd ks, because the
    # padding is where a layout disagreement hides: s=17 and s=129 pad, s=8 is a
    # whole block short, and ks=3 exercises the odd-K half-dword.
    cases = [(1, 32, 2), (2, 64, 8), (4, 17, 2), (1, 96, 6), (3, 8, 32), (2, 129, 4)]

    with tempfile.TemporaryDirectory() as td:
        ref_path = f"{td}/ref.pt"
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                _OPUS_REF_SNIPPET,
                opus_tree,
                json.dumps(cases),
                ref_path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"opus reference subprocess failed (rc={proc.returncode}):\n"
                f"{proc.stdout}\n{proc.stderr}"
            )
        blob = torch.load(ref_path, weights_only=False)

    sub, ran, neg_fired, neg_comparable = blob["sub"], 0, 0, 0
    for g, s, ks in cases:
        plain, ref, ref_wrong_sub = blob["cases"][f"{g},{s},{ks}"]
        s_pad, ks_pad = math.ceil(s / (2 * sub)) * (2 * sub), ((ks + 1) // 2) * 2
        expect = scale_shape(s, g, ks, "mfma_tile", 128)
        if (s_pad, ks_pad) != tuple(expect[1:]):
            raise AssertionError(
                f"g={g} s={s} ks={ks}: this tree pads to {tuple(expect[1:])}, "
                f"opus's sub={sub} layout wants ({s_pad}, {ks_pad})"
            )
        back = _unshuffle_mfma_scale(ref.view(g, s_pad, ks_pad), s, g, ks, 128)
        got = back.permute(1, 0, 2).contiguous()
        if not torch.equal(got, plain.to(got.device)):
            raise AssertionError(
                f"mfma_tile layout has drifted from opus shuffle_scale_a "
                f"(sub={sub}) at g={g} s={s} ks={ks}: "
                f"{(got != plain.to(got.device)).sum().item()} of {plain.numel()} "
                f"scales differ"
            )
        ran += 1

        # Built-in negative control. A gate that cannot fail proves nothing, and
        # this one is cheap: the same round trip against the *other* sub must
        # disagree, else the inverse is ignoring the byte the layout turns on.
        # Only the cases whose row count is a whole 2*(2*sub) block are
        # comparable -- elsewhere the two subs pad to different sizes, which the
        # s_pad assertion above already catches, so they are not controls.
        if ref_wrong_sub.numel() == ref.numel():
            neg_comparable += 1
            bad = (
                _unshuffle_mfma_scale(
                    ref_wrong_sub.view(g, s_pad, ks_pad), s, g, ks, 128
                )
                .permute(1, 0, 2)
                .contiguous()
            )
            neg_fired += not torch.equal(bad, plain.to(bad.device))

    assert ran == len(cases), f"gate was vacuous: ran {ran} of {len(cases)}"
    assert neg_fired == neg_comparable and neg_fired, (
        f"negative control: {neg_comparable - neg_fired} of {neg_comparable} "
        f"size-comparable cases failed to distinguish sub={sub} from "
        f"sub={2 * sub}, so the inverse is not reading the M-pairing byte"
    )
    if verbose:
        aiter.logger.info(
            "mfma_tile @ group 128 == opus shuffle_scale_a(sub=%d): %d/%d cases; "
            "negative control distinguished sub=%d in %d/%d size-comparable cases",
            sub,
            ran,
            len(cases),
            2 * sub,
            neg_fired,
            neg_comparable,
        )
    return sub


def _check_scale_layout(scale, s, g, ks, scale_layout, group_size, name):
    """Assert the scale buffer's shape matches the requested layout."""
    if scale_layout == "row":
        assert (
            scale.stride(2) == 1
        ), f"{name}: expected row-major scale, got strides {scale.stride()}"
        return
    assert scale.is_contiguous(), f"{name}: {scale_layout} scale must be contiguous"
    expect = scale_shape(s, g, ks, scale_layout, group_size)
    assert (
        tuple(scale.shape) == expect
    ), f"{name}: {scale_layout} scale should be {expect}, got {tuple(scale.shape)}"


def _make_inputs(s, h, head_dim, rd, dtype, data_init="norm", seed=0):
    """Build (o, positions, cos, sin) for one config.

    cos/sin are the 2D [max_pos, rd//2] the op takes. A model holding the
    singleton batch/head dims (atom deepseek_v4._build_cos_sin_cache does
    unsqueeze(-2) twice, landing on [max_pos, 1, 1, rd//2] -- aiter
    rope_cached_positions' layout, not [max_pos, rd//2, 1, 1]) reshapes at its
    own call site, the way run_inverse_rope_inplace does for the triton rope.
    Shared by the sweep and the graph check so the two cannot drift.
    """
    gen = make_generator(seed)
    positions = torch.arange(s, dtype=dtypes.i64) % MAX_POS
    # /10 keeps a group's amax away from fp8 saturation, like a real
    # post-softmax attention output.
    o = (
        fill((s * h, head_dim), data_init, gen, dtype=dtype)
        .view(s, h, head_dim)
        .div_(10)
    )
    theta = fill((MAX_POS, rd // 2), data_init, gen, dtype=dtypes.fp32)
    cos = torch.cos(theta).to(dtype).contiguous()
    sin = torch.sin(theta).to(dtype).contiguous()
    return o, positions, cos, sin


def _alloc_outputs(s, g, d, group_size, scale_layout="row"):
    """Pre-allocate (x_fp8, x_scale) the way the wrapper would."""
    from aiter.utility.dtypes import get_dtype_fp8

    x_fp8 = torch.empty((s, g, d), dtype=get_dtype_fp8())
    shape = scale_shape(s, g, d // group_size, scale_layout, group_size)
    # Unfilled like the wrapper, so the padded layouts leave their padding
    # undefined here too and any check that reads it shows up as flaky rather
    # than agreeing with itself by accident.
    x_scale = torch.empty(shape, dtype=dtypes.fp8_e8m0)
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
    s,
    h,
    g,
    head_dim,
    rd,
    group_size,
    dtype,
    scale_layout,
    data_init="norm",
    seed=0,
):
    d = h * head_dim // g
    scale_n = d // group_size

    o, positions, cos, sin = _make_inputs(
        s, h, head_dim, rd, dtype, data_init=data_init, seed=seed
    )

    ref = run_torch(o, positions, cos, sin, g, group_size, rd)
    ref_rt = run_torch(o, positions, cos, sin, g, group_size, rd, roundtrip=True)

    kwargs = {
        "num_groups": g,
        "quant_group_size": group_size,
        "scale_layout": scale_layout,
    }

    def fused():
        return inverse_rope_group_quant_cpp(o, positions, cos, sin, **kwargs)

    pos_r = positions.view(s, 1)
    # Timed on a dedicated scratch because the rope leg is in place. Re-applying
    # an inverse rotation across benchmark iterations does the same work and,
    # being norm-preserving, cannot push values out of range -- but it does leave
    # the buffer rotated n times, so correctness runs on a fresh copy instead.
    unfused_scratch = o.clone()
    unfused_out = _alloc_outputs(s, g, d, group_size)

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
            _alloc_outputs(s, g, d, group_size),
        )

    funcs = {
        "cpp": Cand(fused, fused, ref, scale_layout, FUSED_TOL, FUSED_SCALE_TOL),
        "unfused": Cand(
            unfused_once, unfused_bench, ref_rt, "row", UNFUSED_TOL, UNFUSED_SCALE_TOL
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
        _check_scale_layout(x_scale, s, g, scale_n, cand.scale_layout, group_size, name)
        scale_u8 = _unshuffle_scale(
            x_scale, s, g, scale_n, cand.scale_layout, group_size
        )
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


def check_graph(
    s,
    h,
    g,
    head_dim,
    rd,
    group_size,
    dtype,
    scale_layout,
    data_init="norm",
    seed=0,
):
    """Capture the op in a HIP graph, replay on fresh data, compare against eager.

    Not part of the perf table: this is a pass/fail check that the host-side
    dispatch tier and the pre-allocated buffers survive capture/replay.
    """
    d = h * head_dim // g
    o, positions, cos, sin = _make_inputs(
        s, h, head_dim, rd, dtype, data_init=data_init, seed=seed
    )
    x_fp8, x_scale = _alloc_outputs(s, g, d, group_size, scale_layout=scale_layout)
    kwargs = {
        "num_groups": g,
        "quant_group_size": group_size,
        "scale_layout": scale_layout,
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
    o2, positions2, cos2, sin2 = _make_inputs(
        s, h, head_dim, rd, dtype, data_init=data_init, seed=seed + 7
    )
    o.copy_(o2)
    positions.copy_(positions2)
    cos.copy_(cos2)
    sin.copy_(sin2)
    graph.replay()
    torch.cuda.synchronize()
    # Unshuffled, not raw bytes: the padded layouts round S up to 32 and the op
    # leaves the tail slots untouched, so for n32k4 -- whose consumer cannot see
    # them (md 20) -- the wrapper hands back an unfilled torch.empty and the two
    # allocations disagree there. Comparing the [s, g, Ks] view compares exactly
    # the bytes the op defines.
    scale_n_chk = d // group_size
    graph_fp8 = x_fp8.clone()
    graph_scale = _unshuffle_scale(x_scale, s, g, scale_n_chk, scale_layout, group_size)

    eager_fp8, eager_scale = inverse_rope_group_quant_cpp(
        o,
        positions,
        cos,
        sin,
        num_groups=g,
        quant_group_size=group_size,
        scale_layout=scale_layout,
    )
    torch.cuda.synchronize()

    fp8_match = torch.equal(graph_fp8.view(dtypes.u8), eager_fp8.view(dtypes.u8))
    scale_match = torch.equal(
        graph_scale,
        _unshuffle_scale(eager_scale, s, g, scale_n_chk, scale_layout, group_size),
    )
    # Mirrors the host dispatch in csrc/kernels/inverse_rope_group_quant.cu.
    tds = 2 if s <= 4 else (4 if s <= 128 else 8)
    wave_size = torch.cuda.get_device_properties(o.device).warp_size
    while group_size // tds > wave_size:
        tds *= 2
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
        # Spans all three dispatch tiers of the HIP kernel: s<=4 starts at
        # THREAD_DATA_SIZE=2, s<=128 at 4, above that 8. Wave32 targets raise
        # TDS as needed to keep a quant group within one hardware wave.
        # K_PER_BLOCK steps 1 -> 2 -> 4 at s>128 and s>512.
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
        # 32 is in the default because n32k4 exists only there, so dropping it
        # would silently leave that layout untested on a default run.
        default=[32, 128],
        help="""Quant group size along d. n32k4 is skipped unless 32 is swept.
        e.g.: --group-size 32 64 128""",
    )
    parser.add_argument(
        "-l",
        "--scale-layout",
        type=str,
        choices=list(SCALE_LAYOUTS),
        nargs="*",
        default=list(SCALE_LAYOUTS),
        help="""e8m0 scale storage:
        row = [s, g, ks],
        mfma_tile = [g, s_pad, ks_pad] for gfx950 V_MFMA_SCALE,
        n32k4 = [s_pad/32, g, ks*32] for gfx1250 WMMA scaleB
                (needs group size 32).
        e.g.: -l n32k4""",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        # 300 / 700 sit just past the K_PER_BLOCK steps at s>128 and s>512, so
        # `-s 1 4 32 128 300 512 700 2048 --graph` covers every dispatch tier.
        help="""Also run the HIP-graph capture/replay check over the same sweep.
        e.g.: --graph -s 1 4 32 128 300 512 700 2048""",
    )
    parser.add_argument(
        "--data-init",
        nargs="+",
        choices=list(DATA_DISTS),
        default=["norm"],
        help="DATA initialization distribution(s) (default: norm)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for o and the RoPE cache source (default: 0)",
    )
    parser.add_argument(
        "--opus-tree",
        default=os.environ.get("AITER_OPUS_TREE"),
        help="""Path to the opus aiter checkout. Round-trips this op's
        mfma_tile scale through opus's own shuffle_scale_a, which is the only
        thing keeping _unshuffle_mfma_scale from drifting away from the consumer
        (the two live in different trees). CPU only, ~2s. Also settable via
        AITER_OPUS_TREE.""",
    )
    args = parser.parse_args()

    if args.opus_tree:
        check_opus_layout_identity(args.opus_tree)

    for dtype in args.dtype:
        df = []
        for (
            (h, g),
            s,
            head_dim,
            rd,
            group_size,
            scale_layout,
            data_init,
        ) in itertools.product(
            args.hg,
            args.tokens,
            args.head_dim,
            args.rope_dim,
            args.group_size,
            args.scale_layout,
            args.data_init,
        ):
            # n32k4 only exists at group 32: its four packed k groups are one
            # WMMA-K=128 step, so 4 * group_size has to be 128. The op rejects
            # anything else, so sweeping it here would only collect failures.
            if scale_layout == "n32k4" and group_size != 32:
                continue
            ret = test_inverse_rope_group_quant(
                s,
                h,
                g,
                head_dim,
                rd,
                group_size,
                dtype,
                scale_layout,
                data_init=data_init,
                seed=args.seed,
            )
            df.append(ret)
            if args.graph:
                check_graph(
                    s,
                    h,
                    g,
                    head_dim,
                    rd,
                    group_size,
                    dtype,
                    scale_layout,
                    data_init=data_init,
                    seed=args.seed,
                )
        print_json_table("inverse_rope_group_quant summary", df)
        if args.graph:
            aiter.logger.info("all graph capture/replay checks passed")


if __name__ == "__main__":
    main()
