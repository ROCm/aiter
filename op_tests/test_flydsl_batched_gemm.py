# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + perf for flydsl strided-batched GEMMs.

DeepSeek-V4 grouped-output LoRA (``wo_a``) in ATOM is BF16 x BF16:

    o   : [M(tokens), B(groups), K]  bf16   (mbn physical)
    wo_a: [B, N(o_lora_rank), K]     bf16
    y   : [M, B, N] = einsum("mbk,bnk->mbn")

This test pins the a8w4 path the model will run:
  1. ``quant_act_mxfp8_mbn``  — fused MXFP8 quant + n32k4 scale layout,
  2. ``preshuffle_a8w4_weight_mbn`` — offline MXFP4 weight + n32k4 scale,
  3. ``flydsl_batched_gemm_a8w4_v2`` (layout='mbn', preallocated ``out=``),
and the a8w8 path ``batched_gemm_a8w8_mxscale_bpreshuffle`` with each e8m0
block the arch runs: 1x32 / 32x32 and 1x128 / 128x128 on gfx950 and gfx1250,
plus gfx1250's 1x32 (n32k4-shuffled) weight scale. gfx950 also runs 1x32 /
1x32 on FlyDSL's ``run_bmm_a8w8_mxfp8``, since the public entry runs it on
opus untuned.

On gfx1250 (the mxfp8_32 GEMM) and gfx950 (the bmm) this file also checks
the fused quant epilogue: ``y_scale_out`` must be byte-identical to
quantize_fp8 of the unfused bf16 output. On gfx1250 it runs on the heuristic
tile (M above 1024 is the compute-bound kernel) and on tuned tiles that pin
each epilogue branch (FUSED_QUANT_TILES: two LDS-staged tiles and seven
accumulator tiles, including the tuned DSv4.1 wo_a rows). gfx1250-strict runs that check and the compute
kernelName cases alone; gfx1250-strict also runs the gate_up check below.
gfx1250 also checks that the pipelined K loop (kernelName suffix ``_pl``)
writes the single-fence loop's output bit for bit (PIPELINED_CASES), and that
the gate_up GEMM with the fused SwiGLU + MXFP8 epilogue
(run_gate_up_silu_mxfp8_32_gfx1250 and the registered op
gate_up_silu_mxfp8_32) matches the bf16 GEMM + silu_and_mul + quantize_fp8
chain: scales exact, e4m3 codes at most one step apart and in each case at
most max(2, 1/10000 of its codes), for every tile bucket and swiglu_limit 10,
0 and 5.02.

Input magnitudes change every 32 K (and every 32 weight rows), so the e8m0
scales differ block to block and a scale applied to the wrong block shows up
in the cosine gate. 1x32 gets its own weight and reference, whose magnitudes
change every row; the other candidates share one.

Shape mapping (V4-Pro, ``config.json``):
  b = n_local_groups = o_groups // tp  (swept via ``-b``)
  m = num_tokens                     (swept via ``-s`` first dim)
  n = o_lora_rank = 1024
  k = n_heads * head_dim // o_groups = 4096

Run:
    python op_tests/test_flydsl_batched_gemm.py
    python op_tests/test_flydsl_batched_gemm.py -s 128,1024,4096 -b 2
"""

import argparse
import functools
import itertools
import sys

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.build_targets import base_gfx_arch
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.batched_gemm_op_a8w8 import batched_gemm_a8w8_mxscale_bpreshuffle
from aiter.ops.flydsl.batched_gemm_a8w8 import run_bmm_a8w8_mxfp8
from aiter.ops.flydsl.batched_gemm_mxfp4 import (
    flydsl_batched_gemm_a8w4_v2,
    preshuffle_a8w4_weight_mbn,
    quant_act_mxfp8_mbn,
)
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import assertAllclose, benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950", "gfx1250"]
A8W4_GFX = ["gfx1250"]  # flydsl_batched_gemm_a8w4_v2
SEED = 0
# a8w8 e8m0 blocks per arch as (rows, block): 1xBLOCK on the activation,
# ROWSxBLOCK on the weight.
A8W8_BLOCKS = {
    "gfx950": ((32, 32), (1, 32), (128, 128)),
    "gfx1250": ((32, 32), (128, 128)),
}
# Archs whose a8w8 GEMM also reads a 1x32 weight scale (n32k4-shuffled).
A8W8_1X32_GFX = ["gfx1250"]
# (rows, block) of the a8w8 weight scales whose GEMM runs a ragged last k-tile
# (K not a multiple of 128), per arch.
RAGGED_K_BLOCKS = {"gfx1250": ((32, 32),), "gfx950": ((32, 32), (1, 32))}
# gfx950 bmm on a ragged K (b, m, n, k, kernelName, None for the heuristic): a
# last MFMA step of 32 (K = 288), 64 and 96 K; split in K (_sk), B straight into
# registers (_bd) and per-stage scales (_sps).
RAGGED_K_CASES = (
    (1, 6, 5120, 288, None),
    (1, 2048, 5120, 288, None),
    (1, 64, 5120, 288, "flydsl_bmm_mxfp8_mfma_t16x32x128_w1x1_nb2_sk3"),
    (1, 64, 5120, 288, "flydsl_bmm_mxfp8_mfma_t128x128x128_w1x4_nb3_sk1_bd2"),
    (1, 64, 5120, 288, "flydsl_bmm_mxfp8_mfma_t64x64x128_w2x2_nb2_sk1_sps"),
    (2, 300, 1024, 576, "flydsl_bmm_mxfp8_mfma_t64x32x128_w4x1_nb2_sk5"),
    (2, 64, 1024, 352, "flydsl_bmm_mxfp8_mfma_t16x32x128_w1x1_nb2_sk3"),
)
# Archs whose a8w8 mxfp8_32 batched GEMM also runs the compute kernel.
COMPUTE_GFX = ["gfx1250", "gfx1250-strict"]
# (b, m, n, k, kernelName): ragged M, persistent N tiles with and without the
# four-buffer C relocation, a three-wide cluster, and a cluster_n = 1 config that
# falls back to a one-workgroup cluster at a prime M-tile count.
COMPUTE_CASES = (
    (
        8,
        333,
        1024,
        4096,
        "flydsl_mxfp8_32_bpreshuffle_compute_wmma_t256x256x128_mw2_nw2_nb2_sk1_cm4_cn2_ps2",
    ),
    (
        3,
        777,
        2048,
        2048,
        "flydsl_mxfp8_32_bpreshuffle_compute_wmma_t128x256x128_mw2_nw2_nb3_sk1_cm2_cn2",
    ),
    (
        2,
        333,
        2048,
        2048,
        "flydsl_mxfp8_32_bpreshuffle_compute_wmma_t256x256x128_mw2_nw2_nb4_sk1_cm2_cn2_ps2",
    ),
    (
        1,
        768,
        32768,
        1280,
        "flydsl_mxfp8_32_bpreshuffle_compute_wmma_t256x256x128_mw2_nw2_nb4_sk1_cm3_cn4",
    ),
    (
        2,
        1787,
        1792,
        5120,
        "flydsl_mxfp8_32_bpreshuffle_compute_wmma_t256x256x128_mw2_nw2_nb2_sk1_cm4_cn1",
    ),
    # 63 / 50 M-tiles: cluster_m 4 shrinks to 1 / 2 at launch
    (
        1,
        15997,
        1792,
        5120,
        "flydsl_mxfp8_32_bpreshuffle_compute_wmma_t256x256x128_mw2_nw2_nb2_sk1_cm4_cn1",
    ),
    (
        8,
        12716,
        1024,
        4096,
        "flydsl_mxfp8_32_bpreshuffle_compute_wmma_t256x256x128_mw2_nw2_nb2_sk1_cm4_cn2_ps2",
    ),
)

# (b, m, n, k, kernelName, other kernelName): wo_a on two compute kernels. A compute kernel must run as fast right
# after another compute kernel as it does back to back (it prefetches its own code, see prefetch_kernel_code).
SWITCH_CASES = (
    (
        8,
        1344,
        1024,
        4096,
        "flydsl_mxfp8_32_bpreshuffle_compute_wmma_t256x256x128_mw2_nw2_nb2_sk1_cm1_cn4",
        "flydsl_mxfp8_32_bpreshuffle_compute_wmma_t256x256x128_mw2_nw2_nb2_sk1_cm1_cn1",
    ),
    (
        8,
        1344,
        1024,
        4096,
        "flydsl_mxfp8_32_bpreshuffle_compute_wmma_t256x256x128_mw2_nw2_nb2_sk1_cm2_cn2",
        "flydsl_mxfp8_32_bpreshuffle_compute_wmma_t128x256x128_mw2_nw2_nb3_sk1_cm1_cn2",
    ),
)
# Alternating vs back to back: 1.03 measured with the prefetch, 1.4+ without.
SWITCH_RATIO_MAX = 1.10

# ATOM quantize_fp8's fp8 finite bound. The fused epilogue must match it.
_FP8_MAX = 448.0
# Tuned DSv4.1 tiles that pin both fused-quant epilogue branches: two
# LDS-staged tiles and seven accumulator tiles. One 16-column WMMA tile per
# warp (t16x64 nw4, t32x32 nw2) keeps the LDS-staged quant; an even count
# quantizes from the accumulators. t64x32 nw1 has one quant group per tile
# row; t160x128 has more rows than threads, so its scale rows take a second
# pass. The last entry runs t128x128 again with the pipelined K loop (_pl).
FUSED_QUANT_TILES = (
    "flydsl_bmm_mxfp8_32_wmma_t16x64x512_mw1_nw4_nb4_sk1_cm1_cn1",
    "flydsl_bmm_mxfp8_32_wmma_t32x64x512_mw2_nw2_nb4_sk1_cm1_cn1",
    "flydsl_bmm_mxfp8_32_wmma_t128x128x256_mw2_nw2_nb4_sk1_cm1_cn1",
    "flydsl_bmm_mxfp8_32_wmma_t64x32x512_mw4_nw1_nb4_sk1_cm1_cn1",
    "flydsl_bmm_mxfp8_32_wmma_t160x128x256_mw2_nw2_nb3_sk1_cm1_cn1",
    # DSv4.1 wo_a tuned rows the fused entry runs: t192x128 (more tile rows
    # than threads), t128x256 mw1 nw4 (eight groups per tile row), t96x128,
    # and t32x32 (LDS-staged, one group per tile row).
    "flydsl_bmm_mxfp8_32_wmma_t192x128x256_mw2_nw2_nb3_sk1_cm1_cn1",
    "flydsl_bmm_mxfp8_32_wmma_t128x256x256_mw1_nw4_nb3_sk1_cm1_cn1",
    "flydsl_bmm_mxfp8_32_wmma_t96x128x256_mw2_nw2_nb4_sk1_cm1_cn1",
    "flydsl_bmm_mxfp8_32_wmma_t32x32x512_mw2_nw2_nb4_sk1_cm1_cn1",
    "flydsl_bmm_mxfp8_32_wmma_t128x128x256_mw2_nw2_nb4_sk1_cm1_cn1_pl",
)
# M = 200 is ragged for every tile above; M = 48 is ragged for every tile but
# t16x64, of which it is a whole number. Both use the DSv4.1 wo_a N, K.
FUSED_QUANT_TILE_SHAPES = ((8, 48, 1024, 4096), (8, 200, 1024, 4096))
# The pipelined K loop (kernelName suffix _pl) must give the single-fence
# loop's output bit for bit: (b, m, n, k, kernelName without _pl) for a ragged
# K (k_tail), for a shipped DSv4.1 _pl row (b=1, as the 2-D table runs it), and
# for three- and four-buffer tiles on the DSv4.1 N=5120, K=8192 GEMM, where the
# tail fences wait on fewer outstanding tiles than the steady loop (with two
# buffers the tail fence count is always 0).
PIPELINED_CASES = (
    (2, 200, 512, 576, "flydsl_bmm_mxfp8_32_wmma_t64x128x256_mw2_nw2_nb2_sk1_cm1_cn1"),
    (
        1,
        576,
        32768,
        1280,
        "flydsl_bmm_mxfp8_32_wmma_t192x256x256_mw2_nw2_nb2_sk1_cm1_cn1",
    ),
    (
        1,
        672,
        5120,
        8192,
        "flydsl_bmm_mxfp8_32_wmma_t192x128x256_mw2_nw2_nb3_sk1_cm1_cn1",
    ),
    (
        1,
        576,
        5120,
        8192,
        "flydsl_bmm_mxfp8_32_wmma_t128x128x256_mw2_nw2_nb4_sk1_cm1_cn1",
    ),
)


def _to_e8m0_scale(scale):
    e = torch.ceil(torch.log2(scale.to(dtypes.fp32))).to(torch.int32) + 127
    e = torch.clamp(e, 0, 255).to(torch.uint8)
    return e, torch.exp2(e.to(dtypes.fp32) - 127.0)


def quant_act_e8m0(x_bf16, block):
    """[M,B,K] bf16 -> fp8 [M,B,K] + e8m0 [M,B,K/block]."""
    m, b, k = x_bf16.shape
    xb = x_bf16.to(dtypes.fp32).view(m, b, k // block, block)
    raw = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 448.0
    e8m0, scale = _to_e8m0_scale(raw)
    q = (xb / scale).clamp(-448.0, 448.0).to(dtypes.fp8)
    return q.view(m, b, k).contiguous(), e8m0.squeeze(-1)


def quant_weight_e8m0(w_bf16, rows, block):
    """[B,N,K] bf16 -> fp8 [B,N,K] + e8m0 [B,N/rows,K/block]."""
    b, n, k = w_bf16.shape
    wb = w_bf16.to(dtypes.fp32).view(b, n // rows, rows, k // block, block)
    raw = wb.abs().amax(dim=(2, 4), keepdim=True).clamp(min=1e-8) / 448.0
    e8m0, scale = _to_e8m0_scale(raw)
    q = (wb / scale).clamp(-448.0, 448.0).to(dtypes.fp8)
    return (
        q.view(b, n, k).contiguous(),
        e8m0.view(b, n // rows, k // block).contiguous(),
    )


def _ref_quant_n32k4_mbn(o_mbn: torch.Tensor):
    """Torch oracle for fused ``dynamic_mxfp8_quant_n32k4_mbn`` (bit-exact scales)."""
    from aiter.ops.shuffle import shuffle_scale_n32k4
    from aiter.ops.triton.quant import dynamic_mxfp8_quant

    M, B, K = o_mbn.shape
    a_fp8, a_scale = dynamic_mxfp8_quant(o_mbn.reshape(M * B, K))
    a_fp8 = a_fp8.view(M, B, K).contiguous()
    a_scale = a_scale.view(M, B, K // 32)
    M32 = ((M + 31) // 32) * 32
    if M32 != M:
        pad = torch.zeros(
            (M32 - M, B, K // 32), dtype=a_scale.dtype, device=a_scale.device
        )
        a_scale = torch.cat([a_scale, pad], dim=0)
    a_sh = shuffle_scale_n32k4(a_scale.transpose(0, 1).contiguous(), experts_cnt=B)
    a_sh = a_sh.transpose(0, 1).contiguous()  # [M32//32, B, (K//32)*32]
    return a_fp8, a_sh


def blockwise_randn(*shape, dtype, rows=32):
    """randn * 0.1 whose magnitude changes by 2**[-2, 2] every 32 elements along
    the last dim and every ``rows`` along the one before it (the rows of a weight)."""
    x = torch.randn(*shape) * 0.1
    lead, k = shape[:-1], shape[-1]
    mag_shape = (*lead[:-1], -(-lead[-1] // rows), k // 32)
    mag = torch.exp2(torch.randint(-2, 3, mag_shape).float())
    mag = mag.repeat_interleave(32, -1).repeat_interleave(rows, -2)[..., : lead[-1], :]
    return (x * mag).to(dtype)


def expand_32x32_to_n32k4(ws):
    """[B, N/32, K/32] 32x32 e8m0 -> [B, N, K/32] 1x32, n32k4-shuffled."""
    from aiter.ops.shuffle import shuffle_scale_n32k4

    b, nb, kb = ws.shape
    rows = ws.repeat_interleave(32, dim=1)
    return shuffle_scale_n32k4(rows, experts_cnt=b).view(b, nb * 32, kb)


def _a8w8(x8, w8_shuf, xs8, ws8, dtype):
    """The public a8w8 entry, as the transposed [b, m, n] view the test reads."""
    return batched_gemm_a8w8_mxscale_bpreshuffle(
        x8, w8_shuf, xs8, ws8, dtype=dtype
    ).transpose(0, 1)


def _a8w8_flydsl(x8, w8_shuf, xs8, ws8, dtype):
    """FlyDSL's kernel itself, for a block the public entry sends to opus."""
    m, b, _ = x8.shape
    out = torch.empty(m, b, w8_shuf.shape[1], dtype=dtype)
    return run_bmm_a8w8_mxfp8(x8, w8_shuf, xs8, ws8, out).transpose(0, 1)


def run_torch_wo_a(o_mbn: torch.Tensor, w_bnk: torch.Tensor, dtype=dtypes.bf16):
    """BF16 reference: y[m,b,n] = sum_k o[m,b,k] * w[b,n,k]. Not timed."""
    return torch.einsum(
        "mbk,bnk->mbn", o_mbn.to(dtypes.fp32), w_bnk.to(dtypes.fp32)
    ).to(dtype)


@benchmark()
def test_fused_quant_n32k4_mbn(b, m, k, dtype):
    """Fused MXFP8 quant + n32k4 scale must match the torch reshuffle path."""
    from aiter.ops.triton.quant import dynamic_mxfp8_quant_n32k4_mbn

    torch.manual_seed(SEED)
    o = torch.randn(m, b, k, dtype=dtype) * 0.1

    ref_fp8, ref_scale = _ref_quant_n32k4_mbn(o)

    candidates = {
        "fused": lambda: dynamic_mxfp8_quant_n32k4_mbn(o),
    }

    # Traffic: read bf16 activation, write fp8 payload + e8m0 scale buffer.
    nbytes = (
        m * b * k * o.element_size()
        + m * b * k  # fp8 payload
        + ((m + 31) // 32) * b * (k // 32) * 32  # n32k4 scale
    )

    ret = {"gfx": get_gfx_runtime()}
    for name, fn in candidates.items():
        (got_fp8, got_scale), us = run_perftest(fn)
        err_fp8 = checkAllclose(
            ref_fp8.view(torch.uint8).to(dtypes.fp32),
            got_fp8.view(torch.uint8).to(dtypes.fp32),
            rtol=0,
            atol=0,
            tol_err_ratio=0.0,
            msg=f"{name}: fp8 payload",
        )
        err_scale = checkAllclose(
            ref_scale.to(dtypes.fp32),
            got_scale.to(dtypes.fp32),
            rtol=0,
            atol=0,
            tol_err_ratio=0.0,
            msg=f"{name}: e8m0 scale",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} fp8 err"] = err_fp8
        ret[f"{name} scale err"] = err_scale
    return ret


@benchmark()
def test_batched_gemm(b, m, n, k, dtype, layout):
    """a8w4 and a8w8 strided-batched GEMM, each vs the BF16 einsum of its
    weight."""
    torch.manual_seed(SEED)
    # Model path: o is physically [m, b, k] (mbn); kernel out is a transposed
    # view of preallocated [m, b, n] — same as test_batched_gemm_bf16 mbn case.
    o_mbn = blockwise_randn(m, b, k, dtype=dtype)

    if layout == "mbn":
        y_phys = torch.empty(m, b, n, dtype=dtype)
        y_out = y_phys.transpose(0, 1)  # [b, m, n] view passed to kernel
    else:
        y_phys = torch.empty(b, m, n, dtype=dtype)
        y_out = y_phys

    candidates, nbytes, refs, weights = {}, {}, {}, {}

    def weight(rows):
        """The weight whose magnitudes change every ``rows`` rows, and its
        reference."""
        if rows not in weights:
            w = blockwise_randn(b, n, k, dtype=dtype, rows=rows)
            weights[rows] = w, run_torch_wo_a(o_mbn, w, dtype)
        return weights[rows]

    # A ragged K (not a multiple of 128) has no whole n32k4 scale group, so
    # a8w4, a8w8_128 and the n32k4 1x32 scale do not run it; the blocks in
    # RAGGED_K_BLOCKS do.
    ragged_k = k % 128 != 0
    if get_gfx_runtime() in A8W4_GFX and not ragged_k:
        w_bnk, refs["a8w4"] = weight(32)
        # Offline weight prep (not in the hot path); act quant mirrors every forward.
        w_codes, w_scales = preshuffle_a8w4_weight_mbn(w_bnk)
        a_fp8, a_scales = quant_act_mxfp8_mbn(o_mbn)

        def _launch():
            flydsl_batched_gemm_a8w4_v2(
                a_fp8,
                w_codes,
                a_scales,
                w_scales,
                N=n,
                dtype=dtype,
                layout=layout,
                out=y_phys,
            )
            return y_out

        candidates["a8w4"] = _launch
        # mxfp8 A + mxfp4 B + scale traffic + bf16 C
        nbytes["a8w4"] = (
            m * b * k  # fp8 A
            + b * n * (k // 2)  # mxfp4 B codes
            + ((m + 31) // 32) * b * (k // 32) * 32  # A scale
            + b * (n // 32) * (k // 32) * 32  # B scale
            + b * m * n * y_phys.element_size()  # bf16 out
        )

    a8w8_blocks = A8W8_BLOCKS[get_gfx_runtime()] if layout == "mbn" else ()
    for rows, block in a8w8_blocks:
        if ragged_k and (rows, block) not in RAGGED_K_BLOCKS.get(get_gfx_runtime(), ()):
            continue
        name = f"a8w8_{rows}x{block}"
        w_blk, refs[name] = weight(min(rows, 32))
        x8, xs8 = quant_act_e8m0(o_mbn, block)
        w8, ws8 = quant_weight_e8m0(w_blk, rows, block)
        run = _a8w8_flydsl if rows == 1 else _a8w8
        candidates[name] = functools.partial(
            run, x8, shuffle_weight(w8), xs8, ws8, dtype
        )
        nbytes[name] = (
            m * b * k  # fp8 A
            + b * n * k  # fp8 B (twice the fp4 codes)
            + m * b * (k // block)  # A scale (1 x block)
            + b * (n // rows) * (k // block)  # B scale (rows x block)
            + b * m * n * y_phys.element_size()
        )
        if rows == block == 32 and get_gfx_runtime() in A8W8_1X32_GFX and not ragged_k:
            candidates["a8w8_1x32"] = functools.partial(
                _a8w8, x8, shuffle_weight(w8), xs8, expand_32x32_to_n32k4(ws8), dtype
            )
            nbytes["a8w8_1x32"] = (
                nbytes[name] - b * (n // 32) * (k // 32) + b * n * (k // 32)
            )
            refs["a8w8_1x32"] = refs[name]

    flops = 2 * b * m * n * k

    ret = {"gfx": get_gfx_runtime()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        if layout == "mbn":
            out_mbn = out.transpose(0, 1).contiguous()
        else:
            out_mbn = out
        ref_f = refs[name].to(dtypes.fp32).flatten()
        out_f = out_mbn.to(dtypes.fp32).flatten()
        cos = torch.nn.functional.cosine_similarity(ref_f, out_f, dim=0).item()
        denom = ref_f.abs().mean().clamp_min(1e-6)
        mre = (ref_f - out_f).abs().mean().item() / denom.item()
        # MXFP8 x MXFP4 vs BF16: cosine is the primary gate (see original wo_a test).
        err = 0.0 if cos >= 0.99 else (1.0 - cos)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes[name] / us / 1e6
        ret[f"{name} cos"] = cos
        ret[f"{name} mre"] = mre
        ret[f"{name} err"] = err
    return ret


def bmm_dequant_reference(x8, xs8, w8, ws8):
    """FP32 [M, B, N] reference of the 32x32 batched GEMM from its fp8 operands and
    E8M0 scales."""
    x_scale = torch.exp2(xs8.to(dtypes.fp32) - 127).repeat_interleave(32, -1)
    w_scale = torch.exp2(ws8.to(dtypes.fp32) - 127).repeat_interleave(32, -2)
    w_scale = w_scale.repeat_interleave(32, -1)
    return torch.einsum(
        "mbk,bnk->mbn", x8.to(dtypes.fp32) * x_scale, w8.to(dtypes.fp32) * w_scale
    )


def check_batches(out, ref, msg):
    """Every batch of ``out`` must be finite and match its own slice of ``ref``;
    returns the worst mismatch ratio."""
    assert torch.isfinite(out).all(), f"{msg}: unwritten or non-finite output"
    return max(
        assertAllclose(
            ref[:, i],
            out[:, i].to(dtypes.fp32),
            tol_err_ratio=0.01,
            msg=f"{msg} batch {i}",
        )
        for i in range(ref.shape[1])
    )


@benchmark()
def test_batched_gemm_a8w8_compute(b, m, n, k, kernel_name, dtype=dtypes.bf16):
    """The 32x32 batched GEMM on an explicit kernelName (None: the heuristic's);
    the output starts as NaN and every batch is checked against its own
    dequantized reference. Runs gfx1250's COMPUTE_CASES and gfx950's
    RAGGED_K_CASES."""
    torch.manual_seed(SEED)
    x8, xs8 = quant_act_e8m0(blockwise_randn(m, b, k, dtype=dtype), 32)
    w = blockwise_randn(b, n, k, dtype=dtype)
    w = w * torch.exp2(torch.arange(b) % 4 - 1.0).to(dtype)[:, None, None]
    w8, ws8 = quant_weight_e8m0(w, 32, 32)
    out = torch.full((m, b, n), float("nan"), dtype=dtype)
    fn = functools.partial(
        run_bmm_a8w8_mxfp8,
        x8,
        shuffle_weight(w8),
        xs8,
        ws8,
        out,
        kernel_name=kernel_name,
    )
    _, us = run_perftest(fn)
    err = check_batches(
        out,
        bmm_dequant_reference(x8, xs8, w8, ws8),
        f"{kernel_name} B={b} M={m} N={n} K={k}",
    )
    nbytes = (
        m * b * k
        + b * n * k
        + m * b * (k // 32)
        + b * (n // 32) * (k // 32)
        + b * m * n * out.element_size()
    )
    return {
        "gfx": get_gfx_runtime(),
        "compute us": us,
        "compute TFLOPS": 2 * b * m * n * k / us / 1e6,
        "compute TB/s": nbytes / us / 1e6,
        "compute err": err,
    }


def test_compute_operand_guards():
    """The compute route refuses dense A / C at batch > 1, which its per-batch
    offsets would read and write out of bounds, and a fused quant on a kernel
    with no epilogue for it (persistent tiles here), before anything launches."""
    from aiter.ops.flydsl.mxfp8_bpreshuffle_gemm_gfx1250 import (
        run_gemm_a8w8_mxfp8_32_bpreshuffle_gfx1250,
    )

    b, m, n, k, kernel_name = COMPUTE_CASES[0]
    x8 = torch.empty((m, b, k), dtype=dtypes.fp8)
    w8 = torch.empty((b, n, k), dtype=dtypes.fp8)
    xs8 = torch.empty((m, b, k // 32), dtype=torch.uint8)
    ws8 = torch.empty((b, n // 32, k // 32), dtype=torch.uint8)
    out = torch.empty((m, b, n), dtype=dtypes.bf16)

    def run_dense(xq, o):
        return run_gemm_a8w8_mxfp8_32_bpreshuffle_gfx1250(
            xq,
            w8[0],
            xs8[:, 0].view(dtypes.fp8_e8m0),
            ws8[0].view(dtypes.fp8_e8m0),
            o,
            kernel_name,
            w_scale_32x32=True,
            batch=b,
        )

    with pytest.raises(RuntimeError, match="batch > 1 needs XQ strides"):
        run_dense(x8[:, 0].contiguous(), out[:, 0])
    with pytest.raises(RuntimeError, match="batch > 1 needs XQ strides"):
        run_dense(x8[:, 0], out[:, 0].contiguous())

    y8 = torch.empty((m, b, n), dtype=dtypes.fp8)
    y_scale = torch.empty(m * (b * n // 32), dtype=torch.uint8)
    with pytest.raises(ValueError, match="no fused quant epilogue"):
        run_bmm_a8w8_mxfp8(
            x8, w8, xs8, ws8, y8, kernel_name=kernel_name, y_scale_out=y_scale
        )


def atom_quantize_fp8(y_bf16: torch.Tensor):
    """Torch mirror of ATOM ``quantize_fp8``: group-32 over the flattened row,
    amax floored at 1e-10, code = ceil_pow2(amax/448). ``y_bf16`` is [rows, C]."""
    rows, c = y_bf16.shape
    yv = y_bf16.to(dtypes.fp32).view(rows, c // 32, 32)
    amax = yv.abs().amax(dim=-1).clamp(min=1e-10)
    raw = (amax * (1.0 / _FP8_MAX)).view(torch.int32)
    code = (raw >> 23) + ((raw & 0x7FFFFF) != 0).to(torch.int32)
    code = code.clamp(0, 255).to(torch.int32)
    scale = (code << 23).view(torch.float32)
    q = (yv / scale[..., None]).clamp(-_FP8_MAX, _FP8_MAX).to(dtypes.fp8)
    return q.view(rows, c).contiguous(), code.to(torch.uint8).contiguous()


def test_woa_fused_quant_epilogue(b, m, n, k, kernel_name=None):
    """``y_scale_out`` must match quantize_fp8 of the unfused bf16 GEMM, both
    run with ``kernel_name`` (the heuristic tile when None; gfx1250 names
    only, gfx950 always runs the tile its fused call picks).

    Returns (scale mismatches, fp8 mismatches). The n32k4 weight scale needs
    a whole 128-K tile, and the quant needs N and B*N to be whole groups of 32.
    Other shapes contribute nothing. gfx950 runs its bmm on the raw 32x32
    weight scale; it fuses one split, so the bf16 reference runs one too.
    """
    if n % 32 or k % _fused_quant_k() or (b * n) % 32:
        return 0, 0
    torch.manual_seed(SEED)
    o_mbn = blockwise_randn(m, b, k, dtype=dtypes.bf16)
    w_bnk = blockwise_randn(b, n, k, dtype=dtypes.bf16, rows=32)
    x8, xs8 = quant_act_e8m0(o_mbn, 32)
    w8, ws8 = quant_weight_e8m0(w_bnk, 32, 32)
    w_shuf = shuffle_weight(w8)
    if base_gfx_arch(get_gfx_runtime()) == "gfx950":
        from aiter.ops.flydsl import batched_gemm_a8w8_gfx950 as bmm

        w_scale = ws8
        scale_args = (32, 32, 32, False)
        # The unsplit tile the fused call picks for itself, for the reference.
        picked = (
            bmm._FUSED_LARGE_M_KERNEL
            if m > bmm._FUSED_LARGE_M
            else bmm.pick_bmm_kernel_name(b, m, n, k, *scale_args)
        )
        tile = bmm._fused_quant_name(picked, b, m, n, k, scale_args)
        run_ref = functools.partial(bmm.run_bmm_a8w8_mxfp8_gfx950, kernel_name=tile)
        run_fused = bmm.run_bmm_a8w8_mxfp8_gfx950
    else:
        from aiter.ops.flydsl.batched_gemm_a8w8_gfx1250 import (
            run_bmm_a8w8_mxfp8_32_gfx1250,
        )

        w_scale = expand_32x32_to_n32k4(ws8)
        run_ref = functools.partial(
            run_bmm_a8w8_mxfp8_32_gfx1250, kernel_name=kernel_name
        )
        run_fused = run_ref

    y_bf16 = torch.empty(m, b, n, dtype=dtypes.bf16)
    run_ref(x8, w_shuf, xs8, w_scale, y_bf16)
    y8_ref, yscale_ref = atom_quantize_fp8(y_bf16.view(m, b * n))

    y8 = torch.empty(m, b, n, dtype=dtypes.fp8)
    yscale = torch.empty(m * (b * n // 32), dtype=torch.uint8)
    run_fused(x8, w_shuf, xs8, w_scale, y8, y_scale_out=yscale)
    scale_mismatch = int((yscale.view(m, b * n // 32) != yscale_ref).sum())
    fp8_mismatch = int(
        (
            y8.view(m, b * n).view(torch.uint8).to(torch.int32)
            != y8_ref.view(torch.uint8).to(torch.int32)
        ).sum()
    )
    print(
        f"fused quant B={b} M={m} N={n} K={k} tile {kernel_name or 'heuristic'}: "
        f"scale mismatch {scale_mismatch}/{m * b * n // 32}, "
        f"fp8 mismatch {fp8_mismatch}/{m * b * n}"
    )
    return scale_mismatch, fp8_mismatch


def _fused_quant_k():
    """K granularity of the fused quant check: gfx1250's n32k4 weight scale
    needs whole 128-K groups, gfx950's bmm reads the raw 32x32 scale."""
    return 32 if base_gfx_arch(get_gfx_runtime()) == "gfx950" else 128


def test_pipelined_k_loop(b, m, n, k, kernel_name):
    """``kernel_name`` and ``kernel_name + "_pl"`` must write the same bf16
    output bit for bit (32x32 weight scale, which a ragged K needs). Returns
    the number of mismatched elements."""
    from aiter.ops.flydsl.batched_gemm_a8w8_gfx1250 import (
        run_bmm_a8w8_mxfp8_32_gfx1250,
    )

    torch.manual_seed(SEED)
    o_mbn = blockwise_randn(m, b, k, dtype=dtypes.bf16)
    w_bnk = blockwise_randn(b, n, k, dtype=dtypes.bf16, rows=32)
    x8, xs8 = quant_act_e8m0(o_mbn, 32)
    w8, ws8 = quant_weight_e8m0(w_bnk, 32, 32)
    w_shuf = shuffle_weight(w8)
    outs = []
    for name in (kernel_name, kernel_name + "_pl"):
        y = torch.empty(m, b, n, dtype=dtypes.bf16)
        run_bmm_a8w8_mxfp8_32_gfx1250(
            x8, w_shuf, xs8, ws8, y, kernel_name=name, w_scale_32x32=True
        )
        outs.append(y.view(torch.int16))
    mismatch = int((outs[0] != outs[1]).sum())
    print(
        f"pipelined K loop B={b} M={m} N={n} K={k} {kernel_name}: "
        f"mismatch {mismatch}/{m * b * n}"
    )
    return mismatch


# Fused gate_up + SwiGLU + MXFP8 quant at the DSv4.1 shared-expert shape
# (2*I = 4608, K = 5120): M covers every tile bucket of the picker's table,
# ragged M included, with swiglu_limit 10, 0 (no clamp, as silu_and_mul) and
# 5.02, which bf16 cannot hold (the clamped gate is rounded to bf16).
GATE_UP_SILU_MS = (1, 7, 384, 500, 640, 768, 960, 1152, 1345, 2048, 4096)
GATE_UP_SILU_SHAPE = (4608, 5120)
GATE_UP_SILU_LIMITS = (10.0, 0.0, 5.02)


def _gate_up_silu_table_tile(m):
    """The picker table's tile for ``m``, whatever the device: every bucket
    runs on any gfx1250, also one the picker declines for its CU count."""
    from aiter.ops.flydsl.batched_gemm_a8w8_gfx1250 import _GATE_UP_SILU_TILES

    return next(name for bound, name in _GATE_UP_SILU_TILES if m <= bound)


def test_gate_up_silu_fused(m, n2, k, limit, kernel_name=None):
    """``run_gate_up_silu_mxfp8_32_gfx1250`` against the unfused chain: the
    bf16 gate_up GEMM (same tile), aiter ``silu_and_mul`` and ATOM
    ``quantize_fp8``. The e8m0 scales must match exactly; an e4m3 code may
    differ by one step where the epilogue's exp/rcp and silu_and_mul's
    round to bf16 values on either side of an e4m3 rounding midpoint.
    Returns (scale mismatches, e4m3 mismatches, e4m3 codes off by > 1)."""
    from aiter.ops.batched_gemm_op_a8w8 import gate_up_silu_mxfp8_32
    from aiter.ops.flydsl.batched_gemm_a8w8_gfx1250 import (
        run_bmm_a8w8_mxfp8_32_gfx1250,
        run_gate_up_silu_mxfp8_32_gfx1250,
    )

    name = kernel_name or _gate_up_silu_table_tile(m)
    inter = n2 // 2
    torch.manual_seed(SEED)
    x = blockwise_randn(m, 1, k, dtype=dtypes.bf16) * 20
    # Rows past the clamp, tiny rows and a zero row, each only where it does
    # not land on another one (at M = 1 the one row carries the clamp).
    x[: max(m // 8, 1)] *= 8
    if m >= 4:  # below that m // 4 is row 0, the clamped row
        x[m // 4 : m // 4 + max(m // 8, 1)] *= 1e-3
    if m > 1:  # at M = 1 row 0 is already the clamped row
        x[m - 1] = 0
    w = blockwise_randn(1, n2, k, dtype=dtypes.bf16, rows=32)
    x8, xs8 = quant_act_e8m0(x.to(dtypes.bf16), 32)
    w8, ws8 = quant_weight_e8m0(w, 32, 32)
    w_shuf = shuffle_weight(w8)

    gu = torch.empty(m, 1, n2, dtype=dtypes.bf16)
    run_bmm_a8w8_mxfp8_32_gfx1250(
        x8, w_shuf, xs8, ws8, gu, kernel_name=name, w_scale_32x32=True
    )
    act = torch.empty(m, inter, dtype=dtypes.bf16)
    aiter.silu_and_mul(act, gu.view(m, n2), limit)
    q_ref, s_ref = atom_quantize_fp8(act)

    q = torch.empty(m, inter, dtype=dtypes.fp8)
    sc = torch.empty(m, inter // 32, dtype=torch.uint8)
    run_gate_up_silu_mxfp8_32_gfx1250(
        x8.view(m, k),
        w_shuf.view(n2, k),
        xs8.view(m, k // 32),
        ws8.view(n2 // 32, k // 32),
        q,
        sc,
        limit,
        kernel_name=name,
    )
    if kernel_name is None:
        # The registered op picks the same table tile from the runtime rows.
        q_op, sc_op = gate_up_silu_mxfp8_32(
            x8.view(m, k),
            w_shuf.view(n2, k),
            xs8.view(m, k // 32),
            ws8.view(n2 // 32, k // 32),
            limit,
        )
        if not (
            torch.equal(q_op.view(torch.uint8), q.view(torch.uint8))
            and torch.equal(sc_op.view(torch.uint8), sc)
        ):
            print(f"gate_up silu M={m}: registered op differs from the launch")
            return 1, 1, 1
    scale_bad = int((sc != s_ref.view(m, inter // 32)).sum())
    a = q.view(torch.uint8).to(torch.int32)
    r = q_ref.view(torch.uint8).to(torch.int32)
    code_bad = int((a != r).sum())
    # Same sign and adjacent magnitude codes (e4m3 bytes are sign-magnitude).
    sign_differs = ((a ^ r) & 0x80) != 0
    magnitude_far = ((a & 0x7F) - (r & 0x7F)).abs() > 1
    # 0x7F / 0xFF are NaN: next to 448 they are not a one-step code.
    nan_code = ((a & 0x7F) == 0x7F) | ((r & 0x7F) == 0x7F)
    far = int(((a != r) & (sign_differs | magnitude_far | nan_code)).sum())
    print(
        f"gate_up silu M={m} N2={n2} K={k} limit={limit} {name}: scale mismatch "
        f"{scale_bad}/{m * inter // 32}, e4m3 mismatch {code_bad}/{m * inter} "
        f"(off by more than one code: {far})"
    )
    return scale_bad, code_bad, far


def _run_gate_up_silu():
    from aiter.ops.flydsl.batched_gemm_a8w8_gfx1250 import (
        _gate_up_silu_device_measured,
        gate_up_silu_kernel_for,
    )

    n2, k = GATE_UP_SILU_SHAPE
    # On the part the tiles were measured on, the picker must serve every
    # bucket with the table's tile; elsewhere it declines (None) and the test
    # still runs the table's tiles.
    picks = [gate_up_silu_kernel_for(m, n2, k) for m in GATE_UP_SILU_MS]
    want = [_gate_up_silu_table_tile(m) for m in GATE_UP_SILU_MS]
    if _gate_up_silu_device_measured():
        bad_pick = picks != want
    else:
        print("gate_up silu: picker declines on this CU count; table tiles only")
        bad_pick = any(p is not None for p in picks)
    cases = list(itertools.product(GATE_UP_SILU_MS, GATE_UP_SILU_LIMITS))
    res = [test_gate_up_silu_fused(m, n2, k, limit) for m, limit in cases]
    scale_bad = sum(r[0] for r in res)
    far = sum(r[2] for r in res)
    # Isolated one-step e4m3 differences only (far below the MXFP8 noise),
    # allowed per case: at most max(2, 1/10000 of the case's codes).
    one_step_bad = [
        (m, limit, r[1])
        for (m, limit), r in zip(cases, res)
        if r[1] > max(2, m * (n2 // 2) // 10000)
    ]
    for m, limit, n in one_step_bad:
        print(f"gate_up silu M={m} limit={limit}: {n} one-step e4m3 codes, over budget")
    if bad_pick or not res or scale_bad or far or one_step_bad:
        print(f"gate_up silu fused epilogue mismatched (picker wrong: {bad_pick})")
        raise SystemExit(1)
    print(f"gate_up silu fused epilogue OK ({len(res)} cases)")


def _run_pipelined():
    bad = sum(bool(test_pipelined_k_loop(*case)) for case in PIPELINED_CASES)
    if bad:
        print(f"{bad} pipelined K-loop case(s) mismatched")
        raise SystemExit(1)
    print("pipelined K loop ALL BIT-EXACT")


def _fused_quant_shapes(args, include_ragged_m):
    shapes = [(b, m, n, k) for b in args.batch for (m, n, k) in args.mnk]
    if include_ragged_m:
        shapes.extend((b, 48, 1024, 4096) for b in args.batch)
    # One launch per shape. The ragged-M row is only on the default sweep.
    seen = set()
    unique = []
    for shape in shapes:
        if shape not in seen:
            seen.add(shape)
            unique.append(shape)
    return unique


def _run_fused_quant(shapes, pin_tiles=False):
    """Each shape on the heuristic tile; with ``pin_tiles`` also every
    FUSED_QUANT_TILES entry on FUSED_QUANT_TILE_SHAPES."""
    runs = [(shape, None) for shape in shapes]
    if pin_tiles:
        runs += [
            (shape, name)
            for name in FUSED_QUANT_TILES
            for shape in FUSED_QUANT_TILE_SHAPES
        ]
    bad = 0
    ran = 0
    for (b, m, n, k), kernel_name in runs:
        if n % 32 or k % _fused_quant_k() or (b * n) % 32:
            continue
        scale_mismatch, fp8_mismatch = test_woa_fused_quant_epilogue(
            b, m, n, k, kernel_name
        )
        ran += 1
        if scale_mismatch or fp8_mismatch:
            bad += 1
    if ran == 0:
        print("fused quant epilogue: no whole group-32 shape to check")
        raise SystemExit(1)
    if bad:
        print(f"{bad} fused-quant shape(s) mismatched")
        raise SystemExit(1)
    print("fused quant epilogue ALL BIT-EXACT")


@benchmark()
def test_compute_switch_penalty(
    b, m, n, k, kernel_name, other_kernel_name, dtype=dtypes.bf16
):
    """Two compute kernels run alternately must cost what they cost back to back,
    and each must still produce its own correct result."""
    torch.manual_seed(SEED)
    x8, xs8 = quant_act_e8m0(blockwise_randn(m, b, k, dtype=dtype), 32)
    w8, ws8 = quant_weight_e8m0(blockwise_randn(b, n, k, dtype=dtype), 32, 32)
    ref = bmm_dequant_reference(x8, xs8, w8, ws8)
    w_shuf = shuffle_weight(w8)
    out = torch.empty((m, b, n), dtype=dtype)
    names = (kernel_name, other_kernel_name)
    fns = [
        functools.partial(
            run_bmm_a8w8_mxfp8, x8, w_shuf, xs8, ws8, out, kernel_name=name
        )
        for name in names
    ]
    # The alternating run overwrites the output, so each kernel is checked on its own first.
    for name, fn in zip(names, fns):
        out.fill_(float("nan"))
        fn()
        check_batches(out, ref, f"{name} B={b} M={m} N={n} K={k}")

    def alternate():
        fns[0]()
        fns[1]()

    _, us_a = run_perftest(fns[0])
    _, us_b = run_perftest(fns[1])
    _, us_ab = run_perftest(alternate)
    assert us_ab <= SWITCH_RATIO_MAX * (us_a + us_b), (
        f"{kernel_name} then {other_kernel_name}: {us_ab:.1f} us alternating vs "
        f"{us_a:.1f} + {us_b:.1f} us back to back"
    )
    return {
        "gfx": get_gfx_runtime(),
        "compute us": us_a,
        "compute TFLOPS": 2 * b * m * n * k / us_a / 1e6,
        "other compute us": us_b,
        "alternating us": us_ab,
    }


def summarize(title: str, rows: list) -> None:
    df = pd.DataFrame(rows)
    aiter.logger.info("%s (markdown):\n%s", title, df.to_markdown(index=False))


def main():
    gfx = get_gfx_runtime()
    arch = base_gfx_arch(gfx)
    # gfx1250-strict shares the gfx1250 kernel but is not a full-suite arch.
    epilogue_only = arch == "gfx1250" and gfx not in SUPPORTED_GFX
    if gfx not in SUPPORTED_GFX and not epilogue_only:
        aiter.logger.warning("flydsl batched GEMMs unsupported on %s; skipping", gfx)
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.d_dtypes["bf16"]],
        nargs="*",
        default="bf16,",
        metavar="{bf16}",
        help="Data type.\n        e.g.: -d bf16",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[16, 8, 4, 2, 1],
        help="Batch size (n_local_groups).\n        e.g.: -b 2",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (1, 1024, 4096),
            (16, 1024, 4096),  # cudagraph decode bucket
            (17, 1024, 4096),  # M not a multiple of 32 -> super padding/OOB
            (32, 1024, 4096),
            (128, 1024, 4096),
            (1024, 1024, 4096),
            (4096, 1024, 4096),
            (16384, 1024, 4096),  # chunked-prefill budget
            # K not a multiple of 128: a ragged last k-tile (32x32 scale only).
            (33, 1024, 544),
            (4096, 512, 576),
        ],
        help="Shape m,n,k.\n        e.g.: -s 128,1024,4096",
    )
    parser.add_argument(
        "-l",
        "--layout",
        type=str,
        choices=["mbn"],
        nargs="*",
        default=["mbn"],
        help="Output layout (wo_a uses mbn only).\n        e.g.: -l mbn",
    )
    args = parser.parse_args()
    custom_shape = any(flag in sys.argv for flag in ("-s", "--mnk", "-b", "--batch"))
    if gfx in COMPUTE_GFX:
        test_compute_operand_guards()
        summarize(
            "batched_gemm a8w8 on explicit compute kernelNames",
            [test_batched_gemm_a8w8_compute(*case) for case in COMPUTE_CASES],
        )
        summarize(
            "compute kernels run alternately vs back to back",
            [test_compute_switch_penalty(*case) for case in SWITCH_CASES],
        )

    if epilogue_only:
        shapes = (
            _fused_quant_shapes(args, include_ragged_m=False)
            if custom_shape
            else [(2, m, 1024, 4096) for m in (16, 48, 128, 256, 1025, 2048)]
        )
        _run_fused_quant(shapes, pin_tiles=not custom_shape)
        if not custom_shape:
            _run_gate_up_silu()
        return

    for dtype in args.dtype:
        quant_rows = []
        for b, (m, n, k) in itertools.product(args.batch, args.mnk):
            if k % 128:  # n32k4 groups four 32-wide scales: whole 128-K groups only
                continue
            quant_rows.append(test_fused_quant_n32k4_mbn(b, m, k, dtype))
        summarize("fused_quant_n32k4_mbn", quant_rows)

        gemm_rows = []
        for layout, b, (m, n, k) in itertools.product(
            args.layout, args.batch, args.mnk
        ):
            gemm_rows.append(test_batched_gemm(b, m, n, k, dtype, layout))
        summarize("batched_gemm (a8w4 and a8w8, each vs its BF16 reference)", gemm_rows)

    if arch == "gfx950" and not custom_shape:
        summarize(
            "gfx950 bmm on a ragged K vs its dequantized reference",
            [test_batched_gemm_a8w8_compute(*case) for case in RAGGED_K_CASES],
        )

    if arch == "gfx950":
        _run_fused_quant(_fused_quant_shapes(args, include_ragged_m=not custom_shape))
    if arch == "gfx1250":
        _run_fused_quant(
            _fused_quant_shapes(args, include_ragged_m=not custom_shape),
            pin_tiles=not custom_shape,
        )
        if not custom_shape:
            _run_pipelined()
            _run_gate_up_silu()


if __name__ == "__main__":
    main()
