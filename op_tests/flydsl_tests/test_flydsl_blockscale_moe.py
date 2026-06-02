# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""End-to-end tests for the FlyDSL FP8 blockscale MoE GEMM adapter.

Exercises the full aiter dispatcher path:

    aiter.ops.flydsl.moe_kernels.compile_flydsl_moe_stage{1,2}
       -> aiter.ops.flydsl.kernels.blockscale_moe_gemm_2stage
       -> aiter.ops.flydsl.kernels._blockscale_moe_gemm_2stage_upstream
       -> ROCm/FlyDSL ``kernels/moe_blockscale_2stage.py``

Three groups:
  1. Adapter / dispatcher / Tier-C smoke tests (always run).
  2. Stage1 + stage2 + full pipeline correctness at small shape vs torch
     ref and vs CK 2-stage (always run; needs GPU).
  3. Production-shape perf benchmark (M=8192, DeepSeek-R1-0528 shape),
     gated behind ``AITER_RUN_PERF=1``. Compares FlyDSL stage1+stage2
     against ``aiter.ops.moe_op.ck_moe_stage{1,2}_fwd`` (CK 2-stage,
     today's TP=8 baseline) and ``aiter.fmoe_fp8_blockscale_g1u1`` (ASM
     1-stage fused fallback).

Run::

    pytest op_tests/flydsl_tests/test_flydsl_blockscale_moe.py -v -s
    AITER_RUN_PERF=1 pytest op_tests/flydsl_tests/test_flydsl_blockscale_moe.py -v -s
"""

from __future__ import annotations

import logging
import math
import os

import flydsl.compiler as flyc
import pytest
import torch
import torch.nn.functional as F

import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import moe_sorting as aiter_moe_sorting
from aiter.ops.flydsl.kernels.blockscale_moe_gemm_2stage import (
    SCALE_BLOCK_K_DEFAULT,
    SCALE_BLOCK_N_DEFAULT,
    compile_blockscale_moe_gemm1,
    pick_k_batch_for_blockscale_stage1,
)
from aiter.ops.flydsl.moe_kernels import (
    compile_flydsl_moe_stage1,
    compile_flydsl_moe_stage2,
)
from aiter.ops.moe_op import ck_moe_stage1_fwd, ck_moe_stage2_fwd
from aiter.ops.quant import per_group_quant_hip
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import checkAllclose, run_perftest

logging.getLogger("aiter").setLevel(logging.ERROR)

DEVICE = "cuda"
_RUN_PERF = os.environ.get("AITER_RUN_PERF", "0") in ("1", "true", "True", "yes", "YES")


def _fp8_dtype():
    if not torch.cuda.is_available():
        return torch.float8_e4m3fn
    arch = torch.cuda.get_device_properties(0).gcnArchName
    return torch.float8_e4m3fn if "gfx95" in arch else torch.float8_e4m3fnuz


DTYPE_FP8 = _fp8_dtype()


# ---------------------------------------------------------------------------
# Quantization helpers (kept in-test to avoid coupling to FlyDSL repo)
# ---------------------------------------------------------------------------
def _pertoken_quant_fp8(x: torch.Tensor):
    """Per-row max-abs FP8 quant with FP32 scale; returns (q, scale[..,1])."""
    finfo = torch.finfo(DTYPE_FP8)
    fmax = float(finfo.max)
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / fmax
    q = (x / scale).clamp(-fmax, fmax).to(DTYPE_FP8)
    return q, scale


def _block_quant_expert(w_fp32: torch.Tensor, blk_n: int, blk_k: int):
    """[N, K] fp32 weight -> ([N, K] fp8, [nbn*nbk] fp32 scales)."""
    n, k = w_fp32.shape
    nbn, nbk = n // blk_n, k // blk_k
    tmp = (
        w_fp32.float()
        .view(nbn, blk_n, nbk, blk_k)
        .permute(0, 2, 1, 3)
        .reshape(nbn * nbk, blk_n * blk_k)
    )
    q, sc = _pertoken_quant_fp8(tmp)
    q = q.view(nbn, nbk, blk_n, blk_k).permute(0, 2, 1, 3).reshape(n, k).contiguous()
    return q, sc.view(-1)


def _expand_blockscale(sc_flat, expert, nbn, nbk, blk_n, blk_k):
    return (
        sc_flat.view(-1, 1)
        .repeat(1, blk_n * blk_k)
        .view(expert, nbn, nbk, blk_n, blk_k)
        .permute(0, 1, 3, 2, 4)
        .reshape(expert, nbn * blk_n, nbk * blk_k)
    )


# ---------------------------------------------------------------------------
# Torch references
# ---------------------------------------------------------------------------
def _torch_stage1_ref(
    a_q, w1, topk_ids, a_scale, w1_scale, inter_dim, blk_n, blk_k, act: str = "silu"
):
    """[token, topk, inter_dim] FP32 reference of stage1 (gate+up + activation)."""
    tokens, model_dim = a_q.shape
    topk = topk_ids.shape[1]
    expert = w1.shape[0]

    a = a_q.float()
    if a_scale is not None:
        a = (a.view(tokens, -1, blk_k) * a_scale.unsqueeze(-1)).view(tokens, model_dim)

    w = w1.float()
    if w1_scale is not None:
        w = w * _expand_blockscale(
            w1_scale, expert, (2 * inter_dim) // blk_n, model_dim // blk_k, blk_n, blk_k
        )

    a_rep = a.unsqueeze(1).expand(-1, topk, -1)
    out = torch.zeros(tokens, topk, inter_dim, dtype=torch.float32, device=a.device)
    for e in range(expert):
        mask = topk_ids == e
        if not mask.any():
            continue
        gu = a_rep[mask] @ w[e].T
        gate, up = gu.split([inter_dim, inter_dim], dim=-1)
        if act == "gelu":
            out[mask] = F.gelu(gate, approximate="none") * up
        else:
            out[mask] = F.silu(gate) * up
    return out


def _torch_stage2_ref(
    a2_q,
    w2,
    topk_ids,
    topk_weights,
    a2_scale,
    w2_scale,
    tokens,
    model_dim,
    inter_dim,
    topk,
    blk_n,
    blk_k,
):
    """[token, model_dim] FP32 reference of stage2 (down + topk-weighted sum)."""
    expert = w2.shape[0]
    a = a2_q.float()
    if a2_scale is not None:
        a = (a.view(-1, inter_dim // blk_k, blk_k) * a2_scale.unsqueeze(-1)).view(
            -1, inter_dim
        )
    w = w2.float()
    if w2_scale is not None:
        w = w * _expand_blockscale(
            w2_scale, expert, model_dim // blk_n, inter_dim // blk_k, blk_n, blk_k
        )
    a_3d = a.view(tokens, topk, inter_dim)
    out = torch.zeros(tokens, topk, model_dim, dtype=torch.float32, device=a.device)
    for e in range(expert):
        mask = topk_ids == e
        if mask.any():
            out[mask] = a_3d[mask] @ w[e].T
    return (out * topk_weights.float().view(tokens, -1, 1)).sum(dim=1)


# ---------------------------------------------------------------------------
# Shared data prep — mirrors upstream FlyDSL test exactly
# ---------------------------------------------------------------------------
def _prepare_data(
    tokens,
    model_dim,
    inter_dim,
    experts,
    topk,
    *,
    seed=0,
    blk_n=SCALE_BLOCK_N_DEFAULT,
    blk_k=SCALE_BLOCK_K_DEFAULT,
):
    """Build a self-consistent FP8 blockscale fixture.

    Single block-quantization pass: ``per_group_quant_hip`` produces both
    the FP8 buffer the kernel consumes (``x_bq``) and the FP32 scale used
    by both the kernel and the FP32 reference. Removing the prior
    pertoken→block double-quant (which silently mismatched scales)
    aligns numerics so the small-shape stage1/stage2 tests can use a
    strict tolerance instead of being informational-only.
    """
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    s = 0.2
    x_fp32 = torch.randn(tokens, model_dim, device=DEVICE, generator=g) * s
    nblk_k_w1 = model_dim // blk_k

    # Block-quantize x once. ``x_bscale_fly`` is the [nblk_k, tokens] layout
    # the FlyDSL kernel expects directly; ``x_bscale_ref`` is the
    # [tokens, nblk_k] layout the FP32 reference + CK kernels expect.
    x_bq, x_bscale_fly = per_group_quant_hip(
        x_fp32.to(torch.bfloat16),
        quant_dtype=DTYPE_FP8,
        group_size=blk_k,
        transpose_scale=True,
    )
    x_bq = x_bq.view(tokens, model_dim)
    x_bscale_ref = x_bscale_fly.view(nblk_k_w1, tokens).t().contiguous()

    score = torch.rand(tokens, experts, device=DEVICE, generator=g)
    topk_vals, topk_ids = torch.topk(score, k=topk, dim=1)
    topk_weights = torch.softmax(topk_vals, dim=1).float()
    topk_ids = topk_ids.to(torch.int32)

    w1_bq = torch.empty(
        experts, 2 * inter_dim, model_dim, device=DEVICE, dtype=DTYPE_FP8
    )
    w1_bscale_flat = torch.empty(
        experts,
        ((2 * inter_dim) // blk_n) * (model_dim // blk_k),
        device=DEVICE,
        dtype=torch.float32,
    )
    w2_bq = torch.empty(experts, model_dim, inter_dim, device=DEVICE, dtype=DTYPE_FP8)
    w2_bscale_flat = torch.empty(
        experts,
        (model_dim // blk_n) * (inter_dim // blk_k),
        device=DEVICE,
        dtype=torch.float32,
    )
    for e in range(experts):
        w1e = torch.randn(2 * inter_dim, model_dim, device=DEVICE, generator=g) * s
        q, sc = _block_quant_expert(w1e, blk_n, blk_k)
        w1_bq[e], w1_bscale_flat[e] = q, sc
        w2e = torch.randn(
            model_dim,
            inter_dim,
            device=DEVICE,
            generator=g,
        ) * (s / math.sqrt(inter_dim))
        q, sc = _block_quant_expert(w2e, blk_n, blk_k)
        w2_bq[e], w2_bscale_flat[e] = q, sc
        del w1e, w2e
    torch.cuda.empty_cache()

    w1_shuf = shuffle_weight(w1_bq, layout=(16, 16))
    w2_shuf = shuffle_weight(w2_bq, layout=(16, 16))

    return dict(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        # Single block-quantized activation. x_bq is the kernel buffer,
        # x_bscale_fly is the [nblk_k, tokens] layout the kernel consumes,
        # x_bscale_ref is the [tokens, nblk_k] layout the FP32 reference
        # and CK/ASM call sites expect.
        x_bq=x_bq,
        x_bscale_fly=x_bscale_fly,
        x_bscale_ref=x_bscale_ref,
        # Back-compat aliases for CK/ASM call sites that read a1_bscale.
        a1_bq=x_bq,
        a1_bscale=x_bscale_ref,
        w1_bq=w1_bq,
        w1_bq_shuf=w1_shuf,
        w1_bscale_flat=w1_bscale_flat,
        w2_bq=w2_bq,
        w2_bq_shuf=w2_shuf,
        w2_bscale_flat=w2_bscale_flat,
    )


# ---------------------------------------------------------------------------
# Group 1: adapter / dispatcher / Tier-C smoke
# ---------------------------------------------------------------------------
def test_imports():
    from aiter.ops.flydsl.kernels import blockscale_moe_gemm_2stage as m

    assert hasattr(m, "compile_blockscale_moe_gemm1")
    assert hasattr(m, "compile_blockscale_moe_gemm2")
    assert m.SCALE_BLOCK_N_DEFAULT == 128
    assert m.SCALE_BLOCK_K_DEFAULT == 128


def test_dispatcher_routes_fp8_fp8_to_blockscale():
    exe = compile_flydsl_moe_stage1(
        model_dim=7168,
        inter_dim=512,
        experts=257,
        topk=9,
        tile_m=64,
        tile_n=128,
        tile_k=128,
        doweight_stage1=False,
        a_dtype="fp8",
        b_dtype="fp8",
        out_dtype="f16",
        act="silu",
        waves_per_eu=2,
    )
    assert exe is not None
    assert type(exe).__name__ == "JitFunction"


def test_dispatcher_routes_fp8_fp8_stage2():
    exe = compile_flydsl_moe_stage2(
        model_dim=7168,
        inter_dim=512,
        experts=257,
        topk=9,
        tile_m=64,
        tile_n=128,
        tile_k=128,
        doweight_stage2=True,
        a_dtype="fp8",
        b_dtype="fp8",
        out_dtype="f16",
    )
    assert exe is not None
    assert type(exe).__name__ == "JitFunction"


def test_dispatcher_routes_fp8_fp8_bf16_compiles():
    """Stage1 bf16 compiles via the fast cshuffle epilog (no f16-only guard)."""
    exe = compile_flydsl_moe_stage1(
        model_dim=7168,
        inter_dim=512,
        experts=257,
        topk=9,
        tile_m=64,
        tile_n=128,
        tile_k=128,
        doweight_stage1=False,
        a_dtype="fp8",
        b_dtype="fp8",
        out_dtype="bf16",
        act="silu",
        waves_per_eu=2,
    )
    assert exe is not None
    assert type(exe).__name__ == "JitFunction"


@pytest.mark.parametrize(
    "kwarg,bad_value,expected_match",
    [
        ("act", "relu", "act='relu'"),
        ("enable_bias", True, "enable_bias=True"),
        ("persist_m", 2, "persist_m=2"),
        ("xcd_swizzle", 4, "xcd_swizzle=4"),
        ("swiglu_limit", 7.0, "swiglu_limit=7"),
        ("model_dim_pad", 16, "model_dim_pad=16"),
        ("inter_dim_pad", 16, "inter_dim_pad=16"),
    ],
)
def test_tier_c_kwargs_raise(kwarg, bad_value, expected_match):
    base = dict(
        model_dim=7168,
        inter_dim=512,
        experts=257,
        topk=9,
        tile_m=64,
        tile_n=128,
        tile_k=128,
        doweight_stage1=False,
    )
    base[kwarg] = bad_value
    with pytest.raises(NotImplementedError, match=expected_match):
        compile_blockscale_moe_gemm1(**base)


# ---------------------------------------------------------------------------
# Split-K (k_batch) scaffolding tests
#
# Step 1 of the split-K plan lifts the adapter Tier-C gate but Steps 3-5
# (grid expansion, K-loop slicing, atomic gate/up epilogue) are still TODO
# in the upstream kernel. Until those land, k_batch>1 with otherwise-valid
# config compiles down to a NotImplementedError raised by upstream after
# adapter-side validation; bad K-slice still raises ValueError.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k_batch", [1, 2, 4])
def test_splitk_valid_config_compiles(k_batch):
    """k_batch>1 with a valid K-slice should compile to a JitFunction.
    Steps 3-5 (z-grid expansion, K-loop slicing, atomic gate/up epilogue)
    are wired; this guards against regressions in the compile path."""
    f = compile_blockscale_moe_gemm1(
        model_dim=7168,
        inter_dim=256,
        experts=8,
        topk=2,
        tile_m=64,
        tile_n=128,
        tile_k=128,
        doweight_stage1=False,
        a_dtype="fp8",
        b_dtype="fp8",
        out_dtype="bf16",
        act="silu",
        k_batch=k_batch,
    )
    assert f is not None


@pytest.mark.parametrize(
    "k_batch,model_dim,tile_k,expected_match",
    [
        # k_per_split = 1024/3 -> not integer
        (3, 1024, 128, "not divisible by k_batch"),
        # k_per_split = 256/4 = 64, < scale_block_k=128
        (4, 256, 64, "scale_block_k"),
    ],
)
def test_splitk_invalid_kslice_raises(k_batch, model_dim, tile_k, expected_match):
    with pytest.raises(ValueError, match=expected_match):
        compile_blockscale_moe_gemm1(
            model_dim=model_dim,
            inter_dim=256,
            experts=8,
            topk=2,
            tile_m=64,
            tile_n=128,
            tile_k=tile_k,
            doweight_stage1=False,
            a_dtype="fp8",
            b_dtype="fp8",
            out_dtype="bf16",
            act="silu",
            k_batch=k_batch,
        )


def test_invalid_dtype_combo_raises():
    with pytest.raises(ValueError, match="only a_dtype='fp8' and b_dtype='fp8'"):
        compile_blockscale_moe_gemm1(
            model_dim=7168,
            inter_dim=512,
            experts=257,
            topk=9,
            tile_m=64,
            tile_n=128,
            tile_k=128,
            doweight_stage1=False,
            a_dtype="fp4",
            b_dtype="fp4",
        )


# ---------------------------------------------------------------------------
# Group 2: functional correctness — stage1, stage2, full pipeline
# ---------------------------------------------------------------------------
def _launch_flydsl_stage1(
    data, *, tile_m, tile_n, tile_k, waves_per_eu, act: str = "silu"
):
    """Compile + run FlyDSL stage1 via the aiter dispatcher; return (out, us)."""
    tokens = data["tokens"]
    model_dim = data["model_dim"]
    inter_dim = data["inter_dim"]
    experts = data["experts"]
    topk = data["topk"]

    sorted_ids, sorted_w, sorted_e, num_valid, _ = aiter_moe_sorting(
        data["topk_ids"],
        data["topk_weights"],
        experts,
        model_dim,
        torch.bfloat16,
        tile_m,
    )
    size_expert_ids = sorted_e.numel()

    # Use the single block-quantized activation prepared upfront so the
    # FP32 reference and the kernel see identical (x_bq, x_bscale) inputs.
    a1_bq = data["x_bq"]
    a1_bscale_ref = data["x_bscale_ref"]
    a1_scale_fly = data["x_bscale_fly"].view(-1)
    w1_scale_fly = data["w1_bscale_flat"].view(-1)

    out1 = torch.zeros(tokens, topk, inter_dim, device=DEVICE, dtype=torch.float16)
    exe1 = compile_flydsl_moe_stage1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=False,
        a_dtype="fp8",
        b_dtype="fp8",
        out_dtype="f16",
        act=act,
        waves_per_eu=waves_per_eu,
    )
    stream = torch.cuda.current_stream()
    w1_shuf_flat = data["w1_bq_shuf"].view(-1)
    compiled = flyc.compile(
        exe1,
        out1.view(-1),
        a1_bq.view(-1),
        w1_shuf_flat,
        a1_scale_fly,
        w1_scale_fly,
        sorted_ids,
        sorted_e,
        sorted_w,
        num_valid,
        tokens,
        inter_dim,
        model_dim,
        size_expert_ids,
        stream,
    )

    def _run():
        compiled(
            out1.view(-1),
            a1_bq.view(-1),
            w1_shuf_flat,
            a1_scale_fly,
            w1_scale_fly,
            sorted_ids,
            sorted_e,
            sorted_w,
            num_valid,
            tokens,
            inter_dim,
            model_dim,
            size_expert_ids,
            stream,
        )

    _, us = run_perftest(_run, num_iters=10, num_warmup=3)
    torch.cuda.synchronize()
    return out1, a1_bq, a1_bscale_ref, sorted_ids, sorted_w, sorted_e, num_valid, us


def _launch_flydsl_stage2(
    data,
    out1,
    *,
    tile_m,
    tile_n,
    tile_k,
    waves_per_eu,
    sorted_ids,
    sorted_w,
    sorted_e,
    num_valid,
):
    tokens = data["tokens"]
    model_dim = data["model_dim"]
    inter_dim = data["inter_dim"]
    experts = data["experts"]
    topk = data["topk"]
    size_expert_ids = sorted_e.numel()

    a2_bq, a2_scale_fly = per_group_quant_hip(
        out1.to(torch.bfloat16).view(-1, inter_dim),
        quant_dtype=DTYPE_FP8,
        group_size=SCALE_BLOCK_K_DEFAULT,
        transpose_scale=True,
    )
    nblk_k_w2 = inter_dim // SCALE_BLOCK_K_DEFAULT
    a2_scale_2d = a2_scale_fly.view(nblk_k_w2, -1).t().contiguous()  # for ref
    a2_scale_fly = a2_scale_fly.view(-1)
    w2_scale_fly = data["w2_bscale_flat"].view(-1)

    out2 = torch.zeros(tokens, model_dim, device=DEVICE, dtype=torch.float16)
    exe2 = compile_flydsl_moe_stage2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage2=True,
        a_dtype="fp8",
        b_dtype="fp8",
        out_dtype="f16",
    )
    stream = torch.cuda.current_stream()
    w2_shuf_flat = data["w2_bq_shuf"].view(-1)
    compiled = flyc.compile(
        exe2,
        out2.view(-1),
        a2_bq.view(-1),
        w2_shuf_flat,
        a2_scale_fly,
        w2_scale_fly,
        sorted_ids,
        sorted_e,
        sorted_w,
        num_valid,
        tokens,
        model_dim,
        inter_dim,
        size_expert_ids,
        stream,
    )

    def _run():
        compiled(
            out2.view(-1),
            a2_bq.view(-1),
            w2_shuf_flat,
            a2_scale_fly,
            w2_scale_fly,
            sorted_ids,
            sorted_e,
            sorted_w,
            num_valid,
            tokens,
            model_dim,
            inter_dim,
            size_expert_ids,
            stream,
        )

    _, us = run_perftest(_run, num_iters=10, num_warmup=3)
    torch.cuda.synchronize()
    # Re-run clean so out2 reflects exactly one launch (atomic accumulation)
    out2.zero_()
    _run()
    torch.cuda.synchronize()
    return out2, a2_bq, a2_scale_2d, us


@pytest.mark.parametrize("act", ["silu", "gelu"])
@pytest.mark.parametrize(
    "tokens, model_dim, inter_dim, experts, topk",
    [
        pytest.param(64, 1024, 256, 8, 2, id="tiny"),
        pytest.param(256, 2048, 512, 16, 4, id="small"),
    ],
)
def test_blockscale_correctness_stage1_stage2_e2e(
    tokens,
    model_dim,
    inter_dim,
    experts,
    topk,
    act,
):
    """FlyDSL stage1 / stage2 / full pipeline within 10% rtol/atol of FP32 ref."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device required")

    data = _prepare_data(tokens, model_dim, inter_dim, experts, topk)
    tile_m, tile_n, tile_k = 64, 128, 128

    # ----- FlyDSL stage1 -----
    out1, a1_bq, a1_bscale_ref, sorted_ids, sorted_w, sorted_e, num_valid, us1 = (
        _launch_flydsl_stage1(
            data,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            waves_per_eu=2,
            act=act,
        )
    )
    out1_ref = _torch_stage1_ref(
        a1_bq,
        data["w1_bq"],
        data["topk_ids"],
        a1_bscale_ref,
        data["w1_bscale_flat"],
        inter_dim,
        SCALE_BLOCK_N_DEFAULT,
        SCALE_BLOCK_K_DEFAULT,
        act=act,
    )
    err_s1 = checkAllclose(
        out1_ref.to(out1.dtype),
        out1,
        rtol=0.1,
        atol=0.1,
        msg="flydsl-stage1 vs ref",
        printLog=False,
    )
    print(f"\n  [{tokens}t] stage1: {us1:.1f}us, err_ratio={err_s1:.4f}")
    # After the F3 single-block-quant fix, kernel and FP32 reference share
    # the same (x_bq, x_bscale) so small-shape numerics agree tightly.
    assert torch.isfinite(out1).all(), "stage1 produced non-finite values"
    assert err_s1 <= 0.02, f"stage1 numerics err_ratio={err_s1:.4f} > 0.02"

    # ----- FlyDSL stage2 -----
    out2, a2_bq, a2_scale_2d, us2 = _launch_flydsl_stage2(
        data,
        out1,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        waves_per_eu=2,
        sorted_ids=sorted_ids,
        sorted_w=sorted_w,
        sorted_e=sorted_e,
        num_valid=num_valid,
    )
    out2_ref_s2 = _torch_stage2_ref(
        a2_bq,
        data["w2_bq"],
        data["topk_ids"],
        data["topk_weights"],
        a2_scale_2d,
        data["w2_bscale_flat"],
        tokens,
        model_dim,
        inter_dim,
        topk,
        SCALE_BLOCK_N_DEFAULT,
        SCALE_BLOCK_K_DEFAULT,
    )
    n_nan = (~out2.isfinite()).sum().item()
    err_s2 = checkAllclose(
        out2_ref_s2.to(out2.dtype),
        out2,
        rtol=0.05,
        atol=0.05,
        msg="flydsl-stage2 vs ref",
        printLog=False,
    )
    print(
        f"  [{tokens}t] stage2: {us2:.1f}us, err_vs_ref={err_s2:.4f}, "
        f"non-finite={n_nan}/{out2.numel()}"
    )
    assert n_nan == 0, f"stage2 produced {n_nan} non-finite values"
    assert err_s2 <= 0.05, f"stage2 numerics err_ratio={err_s2:.4f} > 0.05"

    print(f"  [{tokens}t] e2e via aiter dispatcher: {us1 + us2:.1f}us total")


# ---------------------------------------------------------------------------
# Split-K heuristic table (DSR1 TP=8 calibration)
#
# Measured optima on gfx950 (see commit message of the split-K MR):
#   model_dim=7168, E=257, topk=9
#     M=1   idim=256/512 -> k=4 (1.65x / 1.52x vs CK)
#     M=8   idim=256     -> k=2 (1.09x vs CK)
#     M=8   idim=512     -> k=1 (0.98x, near parity; k=2 was 0.93x)
#     M>=16              -> k=1
# The selector is intentionally coarse-grained (picks from {1, 2, 4}).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "M, inter_dim, expected_k",
    [
        # DSR1 TP=8 routed-expert shapes (model_dim=7168, E=257, topk=9):
        (1, 256, 4),
        (1, 512, 4),
        (8, 256, 2),
        (8, 512, 1),
        (16, 256, 2),
        (16, 512, 1),
        (32, 256, 1),
        (64, 256, 1),
        (128, 256, 1),
        (256, 256, 1),
        (1024, 256, 1),
        (8192, 512, 1),
    ],
)
def test_pick_k_batch_dsr1_tp8(M, inter_dim, expected_k):
    """Pin the auto-selector to the values that matched the measured perf
    sweep on the DSR1 TP=8 shape (see commit message of the split-K MR).
    """
    k = pick_k_batch_for_blockscale_stage1(
        token_num=M,
        inter_dim=inter_dim,
        topk=9,
        model_dim=7168,
        tile_m=64,
        tile_n=128,
        tile_k=128,
    )
    assert k == expected_k, f"M={M} idim={inter_dim}: got k_batch={k}, expected {expected_k}"


def test_pick_k_batch_invalid_kslice_falls_back():
    """If desired k_batch violates the K-slice constraints (e.g. model_dim
    not divisible), selector snaps down to a valid value, ultimately 1.
    """
    # model_dim=1024, k=4 -> kps=256 (ok), tile_k=128 -> 2 tiles (even ok)
    # but choose a model_dim where k=4 fails: 1280 / 4 = 320; 320 / 128 = 2.5 -> fail.
    # Then k=2: 1280 / 2 = 640; 640 / 128 = 5 -> odd -> fail. Falls back to 1.
    k = pick_k_batch_for_blockscale_stage1(
        token_num=1,
        inter_dim=128,
        topk=9,
        model_dim=1280,
        tile_m=64,
        tile_n=128,
        tile_k=128,
    )
    assert k == 1


# ---------------------------------------------------------------------------
# Split-K stage1 correctness via flydsl_moe_stage1 dispatcher
# ---------------------------------------------------------------------------
def _launch_flydsl_stage1_splitk(data, *, tile_m, tile_n, tile_k, k_batch, act):
    """Compile + run FlyDSL stage1 with k_batch>1; return tmp_out (bf16,
    gate/up interleave gui_layout, shape (tokens, topk, inter_dim*2)).

    Mirrors _launch_flydsl_stage1 but: (a) passes k_batch + gate_mode to the
    compiler, (b) allocates a zero-init tmp_out buffer (the GEMM atomic-adds
    into it), and (c) writes to out_dtype='bf16'.
    """
    tokens = data["tokens"]
    model_dim = data["model_dim"]
    inter_dim = data["inter_dim"]
    experts = data["experts"]
    topk = data["topk"]

    sorted_ids, sorted_w, sorted_e, num_valid, _ = aiter_moe_sorting(
        data["topk_ids"],
        data["topk_weights"],
        experts,
        model_dim,
        torch.bfloat16,
        tile_m,
    )
    size_expert_ids = sorted_e.numel()

    a1_bq = data["x_bq"]
    a1_scale_fly = data["x_bscale_fly"].view(-1)
    w1_scale_fly = data["w1_bscale_flat"].view(-1)
    w1_shuf_flat = data["w1_bq_shuf"].view(-1)

    tmp_out = torch.zeros(
        tokens, topk, inter_dim * 2, device=DEVICE, dtype=torch.bfloat16
    )

    exe1 = compile_flydsl_moe_stage1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=False,
        a_dtype="fp8",
        b_dtype="fp8",
        out_dtype="bf16",
        act=act,
        waves_per_eu=2,
        k_batch=k_batch,
        gate_mode="interleave",
    )
    stream = torch.cuda.current_stream()
    compiled = flyc.compile(
        exe1,
        tmp_out.view(-1),
        a1_bq.view(-1),
        w1_shuf_flat,
        a1_scale_fly,
        w1_scale_fly,
        sorted_ids,
        sorted_e,
        sorted_w,
        num_valid,
        tokens,
        inter_dim,
        model_dim,
        size_expert_ids,
        stream,
    )
    # Atomic-add into a zeroed buffer: must clear before each launch.
    tmp_out.zero_()
    compiled(
        tmp_out.view(-1),
        a1_bq.view(-1),
        w1_shuf_flat,
        a1_scale_fly,
        w1_scale_fly,
        sorted_ids,
        sorted_e,
        sorted_w,
        num_valid,
        tokens,
        inter_dim,
        model_dim,
        size_expert_ids,
        stream,
    )
    torch.cuda.synchronize()
    return tmp_out


@pytest.mark.parametrize("act", ["silu"])
@pytest.mark.parametrize("k_batch", [2, 4])
@pytest.mark.parametrize(
    "tokens, model_dim, inter_dim, experts, topk",
    [
        pytest.param(8, 1024, 256, 8, 2, id="m8"),
        pytest.param(64, 1024, 256, 8, 2, id="m64"),
        pytest.param(256, 2048, 512, 16, 4, id="m256"),
    ],
)
def test_blockscale_splitk_stage1_e2e(
    tokens, model_dim, inter_dim, experts, topk, k_batch, act
):
    """End-to-end stage1 split-K: z-grid expansion + per-CTA K-slice + atomic
    gate/up store to ``tmp_out`` (interleave gui_layout). Validates the
    atomic gate+up partials reduce to the same result as the baseline
    fused-silu kernel within tolerance.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device required")
    from aiter.ops.activation import silu_and_mul

    data = _prepare_data(tokens, model_dim, inter_dim, experts, topk)
    tile_m, tile_n, tile_k = 64, 128, 128

    tmp_out = _launch_flydsl_stage1_splitk(
        data,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        k_batch=k_batch,
        act=act,
    )
    # gui_layout=True interleave: [gate0..15, up0..15, gate16..31, up16..31, ...]
    # silu_and_mul expects [..., 2*inter_dim] with gate/up split via 16-wide
    # blocks (this is the same layout the dispatcher's _gui_sk path consumes
    # via silu_and_mul_fq).
    out1 = torch.empty(tokens, topk, inter_dim, device=DEVICE, dtype=torch.bfloat16)
    # silu_and_mul over the concat-style layout: reinterpret tmp_out so the
    # gate-block of 16 and up-block of 16 line up. The dispatcher uses
    # silu_and_mul_fq with gui_layout=True; for an act-only correctness
    # check we permute gate/up into the standard [gate||up] split that
    # silu_and_mul consumes.
    t2 = tmp_out.view(tokens, topk, inter_dim // 16, 2, 16)
    gate_part = t2[..., 0, :].reshape(tokens, topk, inter_dim)
    up_part = t2[..., 1, :].reshape(tokens, topk, inter_dim)
    silu_input = torch.cat([gate_part, up_part], dim=-1)
    silu_and_mul(out1.view(-1, inter_dim), silu_input.view(-1, 2 * inter_dim))
    torch.cuda.synchronize()

    out1_ref = _torch_stage1_ref(
        data["x_bq"],
        data["w1_bq"],
        data["topk_ids"],
        data["x_bscale_ref"],
        data["w1_bscale_flat"],
        inter_dim,
        SCALE_BLOCK_N_DEFAULT,
        SCALE_BLOCK_K_DEFAULT,
        act=act,
    )

    assert torch.isfinite(out1).all(), "split-K stage1 produced non-finite values"
    err = checkAllclose(
        out1_ref.to(out1.dtype),
        out1,
        rtol=0.1,
        atol=0.1,
        msg=f"flydsl-stage1 splitk k_batch={k_batch}",
        printLog=False,
    )
    print(f"\n  [{tokens}t k_batch={k_batch}] split-K stage1 err_ratio={err:.4f}")
    assert err <= 0.05, f"split-K stage1 err_ratio={err:.4f} > 0.05"


# ---------------------------------------------------------------------------
# Group 3: production-shape perf (gated by AITER_RUN_PERF=1)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _RUN_PERF, reason="set AITER_RUN_PERF=1 to run perf benchmarks")
def test_perf_dsr1_e2e_vs_ck_and_asm():
    """Full perf table at DSR1 (M=8192, dim=7168, idim=512, E=257, topk=9).

    Expected on gfx950 / 256 CUs (prior standalone measurement):
      FlyDSL stage1+stage2   ~2112 us  (~768 TFLOPS)
      CK     stage1+stage2   ~2487 us  (~653 TFLOPS) -> FlyDSL 1.18x faster
      aiter fmoe ASM fused   ~3011 us               -> FlyDSL 1.43x faster
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device required")

    tokens, model_dim, inter_dim, experts, topk = 8192, 7168, 512, 257, 9
    tile_m, tile_n, tile_k = 64, 128, 128
    blk_n, blk_k = SCALE_BLOCK_N_DEFAULT, SCALE_BLOCK_K_DEFAULT

    data = _prepare_data(tokens, model_dim, inter_dim, experts, topk)

    # ===== FlyDSL stage1+stage2 (via aiter dispatcher) =====
    (
        out1,
        a1_bq,
        _a1_bscale_ref,
        sorted_ids,
        sorted_w,
        sorted_e,
        num_valid,
        us_fly_s1,
    ) = _launch_flydsl_stage1(
        data,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        waves_per_eu=2,
    )
    out2, a2_bq, a2_scale_2d, us_fly_s2 = _launch_flydsl_stage2(
        data,
        out1,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        waves_per_eu=2,
        sorted_ids=sorted_ids,
        sorted_w=sorted_w,
        sorted_e=sorted_e,
        num_valid=num_valid,
    )
    us_fly_total = us_fly_s1 + us_fly_s2

    # ===== CK 2-stage (today's TP=8 baseline) =====
    ck_block_m = 32
    sorted_ids_a, sorted_w_a, sorted_e_a, num_valid_a, _ = aiter_moe_sorting(
        data["topk_ids"],
        data["topk_weights"],
        experts,
        model_dim,
        torch.bfloat16,
        ck_block_m,
    )
    out_ck_s1 = torch.zeros(tokens, topk, inter_dim, device=DEVICE, dtype=torch.float16)
    nblk_n_w1 = (2 * inter_dim) // blk_n
    nblk_n_w2 = model_dim // blk_n
    nblk_k_w1 = model_dim // blk_k
    nblk_k_w2 = inter_dim // blk_k
    w1_scale_ck = (
        data["w1_bscale_flat"].view(experts, nblk_n_w1, nblk_k_w1).contiguous()
    )
    w2_scale_ck = (
        data["w2_bscale_flat"].view(experts, nblk_n_w2, nblk_k_w2).contiguous()
    )

    def _run_ck_s1():
        out_ck_s1.zero_()
        ck_moe_stage1_fwd(
            hidden_states=a1_bq,
            w1=data["w1_bq_shuf"],
            w2=data["w2_bq_shuf"],
            sorted_token_ids=sorted_ids_a,
            sorted_expert_ids=sorted_e_a,
            num_valid_ids=num_valid_a,
            out=out_ck_s1,
            topk=topk,
            kernelName="",
            w1_scale=w1_scale_ck,
            a1_scale=data["a1_bscale"],
            block_m=ck_block_m,
            sorted_weights=None,
            quant_type=QuantType.per_1x128,
            activation=ActivationType.Silu,
            dst_type=torch.float16,
        )

    _, us_ck_s1 = run_perftest(_run_ck_s1, num_iters=30, num_warmup=10)

    out_ck_s2 = torch.zeros(tokens, model_dim, device=DEVICE, dtype=torch.float16)
    a2_for_ck = a2_bq.view(tokens, topk, inter_dim)

    def _run_ck_s2():
        out_ck_s2.zero_()
        ck_moe_stage2_fwd(
            inter_states=a2_for_ck,
            w1=data["w1_bq_shuf"],
            w2=data["w2_bq_shuf"],
            sorted_token_ids=sorted_ids_a,
            sorted_expert_ids=sorted_e_a,
            num_valid_ids=num_valid_a,
            out=out_ck_s2,
            topk=topk,
            kernelName="",
            w2_scale=w2_scale_ck,
            a2_scale=a2_scale_2d,
            block_m=ck_block_m,
            sorted_weights=sorted_w_a,
            quant_type=QuantType.per_1x128,
            activation=ActivationType.Silu,
        )

    _, us_ck_s2 = run_perftest(_run_ck_s2, num_iters=30, num_warmup=10)
    us_ck_total = us_ck_s1 + us_ck_s2

    # ===== aiter ASM fused (for reference) =====
    us_asm = float("nan")
    try:
        nblk_k_w1 = model_dim // blk_k
        a1_scale_aiter = data["a1_bscale"].t().contiguous()  # [nblk_k_w1, token]
        out_asm = torch.zeros(tokens, model_dim, device=DEVICE, dtype=torch.bfloat16)

        def _run_asm():
            out_asm.zero_()
            aiter.fmoe_fp8_blockscale_g1u1(
                out_asm,
                a1_bq,
                data["w1_bq_shuf"],
                data["w2_bq_shuf"],
                sorted_ids_a,
                sorted_w_a,
                sorted_e_a,
                num_valid_a,
                topk,
                a1_scale_aiter,
                data["w1_bscale_flat"],
                data["w2_bscale_flat"],
                "",
                blk_n,
                blk_k,
                None,
            )

        _, us_asm = run_perftest(_run_asm, num_iters=30, num_warmup=10)
    except Exception as e:
        print(f"\n  (ASM fused unavailable: {type(e).__name__}: {str(e)[:80]})")

    # ===== Report =====
    flops_s1 = 2 * tokens * topk * (2 * inter_dim) * model_dim
    flops_s2 = 2 * tokens * topk * model_dim * inter_dim
    flops_total = flops_s1 + flops_s2

    def tflops(us):
        return flops_total / (us / 1e6) / 1e12 if us > 0 else 0.0

    print(
        f"\n  DSR1 e2e (M={tokens}, dim={model_dim}, idim={inter_dim}, E={experts}, k={topk}):"
    )
    print(f"    {'kernel':>20s} | {'us':>9s} | {'TFLOPS':>8s} | {'vs FlyDSL':>10s}")
    print(f"    {'-'*20}-+-{'-'*9}-+-{'-'*8}-+-{'-'*10}")
    print(
        f"    {'FlyDSL s1':>20s} | {us_fly_s1:9.1f} | {flops_s1/(us_fly_s1/1e6)/1e12:8.2f} | {'-':>10s}"
    )
    print(
        f"    {'FlyDSL s2':>20s} | {us_fly_s2:9.1f} | {flops_s2/(us_fly_s2/1e6)/1e12:8.2f} | {'-':>10s}"
    )
    print(
        f"    {'FlyDSL total':>20s} | {us_fly_total:9.1f} | {tflops(us_fly_total):8.2f} | {1.0:>9.2f}x"
    )
    print(
        f"    {'CK s1':>20s} | {us_ck_s1:9.1f} | {flops_s1/(us_ck_s1/1e6)/1e12:8.2f} | {us_ck_s1/us_fly_s1:>9.2f}x"
    )
    print(
        f"    {'CK s2':>20s} | {us_ck_s2:9.1f} | {flops_s2/(us_ck_s2/1e6)/1e12:8.2f} | {us_ck_s2/us_fly_s2:>9.2f}x"
    )
    print(
        f"    {'CK total':>20s} | {us_ck_total:9.1f} | {tflops(us_ck_total):8.2f} | {us_ck_total/us_fly_total:>9.2f}x"
    )
    if us_asm == us_asm:  # not NaN
        print(
            f"    {'ASM fused':>20s} | {us_asm:9.1f} | {tflops(us_asm):8.2f} | {us_asm/us_fly_total:>9.2f}x"
        )

    # Assert: FlyDSL total must beat CK total by at least 5% (we measured 1.18x).
    assert (
        us_fly_total < us_ck_total * 0.98
    ), f"FlyDSL total {us_fly_total:.1f}us not faster than CK {us_ck_total:.1f}us"


if __name__ == "__main__":
    import sys

    pytest.main([__file__, "-v", "-s"] + sys.argv[1:])
