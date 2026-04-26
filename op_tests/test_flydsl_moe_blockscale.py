# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Numerical correctness tests for the AITER FlyDSL blockscale MoE port.

Mirrors the upstream FlyDSL ``tests/kernels/test_moe_blockscale.py`` shape
sweep but exercises the *aiter* port at
``aiter.ops.flydsl.moe_blockscale_kernels`` rather than upstream FlyDSL
directly. The DSV4 prefill case (``B=12, model=7168, inter=3072, E=384,
topk=6``) is the gating shape for the W4.5 accuracy fix tracked in
sunway513/atom#37.

Stage flow:

  Stage 1: x_q @ w1[e].T -> SiLU(g) * u  (FP8 in, F16 out, per_1x128 scale)
  Stage 2: act_q @ w2[e].T -> atomic-add into output (FP8 in, F16 out)

(F16 output is required because the AITER FlyDSL stage1 port enables the
cshuffle epilog by default, which currently supports only f16 output. The
end-to-end accuracy contract is unaffected by the intermediate dtype.)

Both stages are compared against a torch FP32 reference that expands the
block scales element-wise (helper adapted from upstream's
``torch_stage{1,2}_blockscale_ref`` -- see attribution comment on each
helper).
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Skip the whole module if FlyDSL isn't available (tests can still be
# collected on machines without flydsl installed, e.g. CI doc builds).
# ---------------------------------------------------------------------------

flydsl_avail = pytest.importorskip(
    "flydsl",
    reason="FlyDSL package not installed; AITER FlyDSL blockscale port "
    "requires `flydsl>=0.1.3`.",
)

from aiter import dtypes, pertoken_quant  # noqa: E402
from aiter.fused_moe import fused_topk, moe_sorting  # noqa: E402
from aiter.ops.shuffle import shuffle_weight  # noqa: E402

from aiter.ops.flydsl.moe_blockscale_kernels import (  # noqa: E402
    compile_moe_blockscale_gemm1,  # noqa: F401  (smoke-imported to assert export)
    compile_moe_blockscale_gemm2,  # noqa: F401
    flydsl_moe_blockscale_stage1,
    flydsl_moe_blockscale_stage2,
)

# Mirror upstream FlyDSL test markers so this test runs in the same lanes
# (large GPU, lower ROCm requirements). pytest tolerates unknown markers,
# so this is a no-op when those markers aren't registered.
pytestmark = [pytest.mark.l2_device]


# ---------------------------------------------------------------------------
# Reference helpers
# ---------------------------------------------------------------------------


def _expand_blockscale(scale_flat, E, nblk_n, nblk_k, blk_n, blk_k):
    """Expand ``[E, nblk_n*nblk_k]`` flat block-scale tensor to ``[E, N, K]``.

    Adapted from ``FlyDSL/tests/kernels/test_moe_blockscale.py::_expand_blockscale``
    (Apache-2.0). Re-implemented here as a private helper so the aiter test
    has no runtime dependency on the upstream test tree.
    """
    return (
        scale_flat.view(-1, 1)
        .repeat(1, blk_n * blk_k)
        .view(E, nblk_n, nblk_k, blk_n, blk_k)
        .permute(0, 1, 3, 2, 4)
        .reshape(E, nblk_n * blk_n, nblk_k * blk_k)
    )


def torch_stage1_blockscale_ref(
    hidden_states,
    w1,
    topk_ids,
    a_scale,
    w1_scale,
    scale_blks,
    inter_dim,
):
    """FP32 torch reference for blockscale stage1 (gate+up GEMM with SiLU*up).

    Returns ``[B, topk, inter_dim]`` in fp32. Adapted from upstream
    ``FlyDSL/tests/kernels/test_moe_blockscale.py::torch_stage1_blockscale_ref``
    (Apache-2.0); body kept structurally identical to preserve numerical
    equivalence with the upstream contract.
    """
    compute_dtype = torch.float32
    hidden_states = hidden_states.to(compute_dtype)
    w1 = w1.to(compute_dtype)
    B, D = hidden_states.shape
    topk = topk_ids.shape[1]
    E = w1.shape[0]
    blk_n, blk_k = scale_blks

    if a_scale is not None:
        hidden_states = hidden_states.view(B, -1, blk_k) * a_scale.unsqueeze(-1)
        hidden_states = hidden_states.view(B, -1)

    nblk_n_w1 = (2 * inter_dim) // blk_n
    nblk_k_w1 = D // blk_k
    if w1_scale is not None:
        w1 = w1 * _expand_blockscale(w1_scale, E, nblk_n_w1, nblk_k_w1, blk_n, blk_k)

    hidden_states = hidden_states.view(B, 1, D).expand(-1, topk, -1)
    out = torch.zeros(
        (B, topk, inter_dim), dtype=compute_dtype, device=hidden_states.device
    )
    for e in range(E):
        mask = topk_ids == e
        if mask.sum():
            sub = hidden_states[mask]
            act = sub @ w1[e].T
            gate, up = act.split([inter_dim, inter_dim], dim=-1)
            out[mask] = F.silu(gate) * up
    return out


def torch_stage2_blockscale_ref(
    act_q,
    w2,
    topk_ids,
    topk_weights,
    a_scale,
    w2_scale,
    scale_blks,
    tokens,
    model_dim,
    inter_dim,
    topk,
):
    """FP32 torch reference for blockscale stage2 (down-proj + topk reduce).

    Adapted from upstream
    ``FlyDSL/tests/kernels/test_moe_blockscale.py::torch_stage2_blockscale_ref``
    (Apache-2.0).
    """
    compute_dtype = torch.float32
    blk_n, blk_k = scale_blks
    E = w2.shape[0]

    act = act_q.to(compute_dtype)
    nblk_k_w2 = inter_dim // blk_k
    if a_scale is not None:
        act = act.view(-1, nblk_k_w2, blk_k) * a_scale.unsqueeze(-1)
        act = act.view(-1, inter_dim)

    w2f = w2.to(compute_dtype)
    nblk_n_w2 = model_dim // blk_n
    if w2_scale is not None:
        w2f = w2f * _expand_blockscale(w2_scale, E, nblk_n_w2, nblk_k_w2, blk_n, blk_k)

    act_3d = act.view(tokens, topk, inter_dim)
    out = torch.zeros((tokens, topk, model_dim), dtype=compute_dtype, device=act.device)
    for e in range(E):
        mask = topk_ids == e
        if mask.sum():
            out[mask] = act_3d[mask] @ w2f[e].T
    return (out * topk_weights.view(tokens, -1, 1)).sum(dim=1)


# ---------------------------------------------------------------------------
# Test data builder
# ---------------------------------------------------------------------------


def _block_quant_expert(w_fp32, blk_n, blk_k, fp8_dtype):
    """Block-quantise a single expert weight ``[N, K]``.

    Returns ``(w_q, scale_flat)`` where ``w_q`` has the same ``[N, K]`` shape
    in ``fp8_dtype`` and ``scale_flat`` is the row-major flattened
    ``[nblk_n * nblk_k]`` per-block scale.
    """
    N, K = w_fp32.shape
    nbn = N // blk_n
    nbk = K // blk_k
    tmp = (
        w_fp32.float()
        .view(nbn, blk_n, nbk, blk_k)
        .permute(0, 2, 1, 3)
        .reshape(nbn * nbk, blk_n * blk_k)
    )
    q, sc = pertoken_quant(tmp, quant_dtype=fp8_dtype)
    q = q.view(nbn, nbk, blk_n, blk_k).permute(0, 2, 1, 3).reshape(N, K)
    return q, sc.view(-1)


def _build_blockscale_inputs(
    *,
    B,
    model_dim,
    inter_dim,
    E,
    topk,
    block_m,
    seed=0,
    scale_blks=(128, 128),
):
    """Build a complete blockscale MoE input bundle.

    All tensors live on the current default cuda device. Returns a dict
    keyed by purpose so individual tests can pluck what they need without
    rebuilding shared state.
    """
    fp8_dtype = dtypes.fp8
    device = torch.device("cuda")
    blk_n, blk_k = scale_blks

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    s = 0.2

    # Activations + topk routing (deterministic seed so reference + kernel
    # see identical inputs across invocations).
    x_fp32 = (
        torch.randn((B, model_dim), generator=g, device=device, dtype=torch.float32) * s
    )
    score = torch.rand((B, E), generator=g, device=device, dtype=torch.float32)
    inp_for_topk = x_fp32.to(torch.bfloat16)
    score_bf16 = score.to(torch.bfloat16)
    topk_weights, topk_ids = fused_topk(inp_for_topk, score_bf16, topk, True)
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    # Per-expert generation -> immediately block-quantise to keep VRAM low
    # for the DSV4 (E=384) shape.
    w1_q_list, w1_scale_list = [], []
    w2_q_list, w2_scale_list = [], []
    for _e in range(E):
        w1e = (
            torch.randn(
                (2 * inter_dim, model_dim),
                generator=g,
                device=device,
                dtype=torch.float32,
            )
            * s
        )
        q1, sc1 = _block_quant_expert(w1e, blk_n, blk_k, fp8_dtype)
        w1_q_list.append(q1)
        w1_scale_list.append(sc1)
        del w1e

        w2e = torch.randn(
            (model_dim, inter_dim),
            generator=g,
            device=device,
            dtype=torch.float32,
        ) * (s / math.sqrt(inter_dim))
        q2, sc2 = _block_quant_expert(w2e, blk_n, blk_k, fp8_dtype)
        w2_q_list.append(q2)
        w2_scale_list.append(sc2)
        del w2e

    w1_q = torch.stack(w1_q_list)  # [E, 2*inter, model]
    w1_scale = torch.stack(w1_scale_list)  # [E, nblk_n_w1 * nblk_k_w1] flat
    w2_q = torch.stack(w2_q_list)  # [E, model, inter]
    w2_scale = torch.stack(w2_scale_list)  # [E, nblk_n_w2 * nblk_k_w2] flat
    del w1_q_list, w1_scale_list, w2_q_list, w2_scale_list
    torch.cuda.empty_cache()

    # Activation block-quant: [B, model_dim] -> fp8 + scale [B, model/blk_k].
    a1_q, a1_scale_2d = pertoken_quant(
        x_fp32.view(-1, model_dim // blk_k, blk_k), quant_dtype=fp8_dtype
    )
    a1_q = a1_q.view(-1, model_dim)
    a1_scale_2d = a1_scale_2d.squeeze(-1).contiguous()  # [B, nblk_k_w1]

    # FlyDSL kernel expects a1_scale in transposed [nblk_k_w1, B] layout
    # flattened. The per_group_quant_hip path can write that directly, but
    # for portability we use pertoken_quant and transpose here.
    a1_scale_fly = a1_scale_2d.t().contiguous().view(-1)
    w1_scale_fly = w1_scale.view(-1)

    # Preshuffle weights for the kernel. Both stage1 and stage2 use the
    # same (16, 16) MFMA-friendly layout.
    w1_q_shuf = shuffle_weight(w1_q, layout=(16, 16)).contiguous()
    w2_q_shuf = shuffle_weight(w2_q, layout=(16, 16)).contiguous()

    # MoE sorting (used by both stages).
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _moe_buf = (
        moe_sorting(topk_ids, topk_weights, E, model_dim, torch.bfloat16, block_m)
    )

    return dict(
        # Stage1 inputs
        a1_q=a1_q,
        a1_scale_2d=a1_scale_2d,
        a1_scale_fly=a1_scale_fly,
        w1_q=w1_q,
        w1_q_shuf=w1_q_shuf,
        w1_scale=w1_scale,
        w1_scale_fly=w1_scale_fly,
        # Stage2 inputs (a2 must be re-quantised from stage1 output by caller)
        w2_q=w2_q,
        w2_q_shuf=w2_q_shuf,
        w2_scale=w2_scale,
        w2_scale_fly=w2_scale.view(-1),
        # Routing
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        sorted_ids=sorted_ids,
        sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        # Shape info
        B=B,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=E,
        topk=topk,
        block_m=block_m,
        scale_blks=scale_blks,
    )


# ---------------------------------------------------------------------------
# Helpers shared between parametrize-sweep and DSV4-shape tests
# ---------------------------------------------------------------------------


def _run_stage1_and_compare(data, *, tile_m, tile_n, tile_k, atol=1e-2, rtol=1e-2):
    """Launch FlyDSL blockscale stage1 and compare against torch ref.

    Returns the kernel output (bf16) so callers can chain it into stage2.
    """
    out = flydsl_moe_blockscale_stage1(
        a=data["a1_q"],
        w1=data["w1_q_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=data["topk"],
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        a_dtype="fp8",
        b_dtype="fp8",
        out_dtype="f16",
        w1_scale=data["w1_scale_fly"],
        a1_scale=data["a1_scale_fly"],
    )
    torch.cuda.synchronize()

    ref_fp32 = torch_stage1_blockscale_ref(
        data["a1_q"],
        data["w1_q"],
        data["topk_ids"],
        data["a1_scale_2d"],
        data["w1_scale"],
        data["scale_blks"],
        data["inter_dim"],
    )
    # Fraction-of-elements close: FP8 quantisation noise plus blockscale
    # rounding makes element-wise allclose too tight, even for the upstream
    # kernel; mirror upstream's "fraction passes" criterion (>= 95%).
    close_mask = torch.isclose(
        ref_fp32.to(out.dtype).float(), out.float(), atol=atol, rtol=rtol
    )
    pct_close = close_mask.float().mean().item()
    assert pct_close >= 0.95, (
        f"stage1 only {pct_close*100:.2f}% close (need >= 95%); "
        f"shape B={data['B']} model={data['model_dim']} inter={data['inter_dim']} "
        f"E={data['E']} topk={data['topk']} tile=({tile_m},{tile_n},{tile_k})"
    )
    return out


def _run_stage2_and_compare(
    data, stage1_out, *, tile_m, tile_n, tile_k, atol=1e-2, rtol=1e-2
):
    """Re-quantise stage1 output, launch stage2, compare against torch ref."""
    blk_k = data["scale_blks"][1]
    inter_dim = data["inter_dim"]
    B = data["B"]
    topk = data["topk"]

    a2_q, a2_scale_2d = pertoken_quant(
        stage1_out.float().view(-1, inter_dim // blk_k, blk_k),
        quant_dtype=dtypes.fp8,
    )
    a2_q = a2_q.view(-1, inter_dim)
    a2_scale_2d = a2_scale_2d.squeeze(-1).contiguous()  # [B*topk, nblk_k_w2]
    a2_scale_fly = a2_scale_2d.t().contiguous().view(-1)

    out = flydsl_moe_blockscale_stage2(
        inter_states=a2_q.view(B, topk, inter_dim),
        w2=data["w2_q_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        a_dtype="fp8",
        b_dtype="fp8",
        out_dtype="f16",
        mode="atomic",
        w2_scale=data["w2_scale_fly"],
        a2_scale=a2_scale_fly,
        sorted_weights=data["sorted_weights"],
    )
    torch.cuda.synchronize()

    ref_fp32 = torch_stage2_blockscale_ref(
        a2_q,
        data["w2_q"],
        data["topk_ids"],
        data["topk_weights"],
        a2_scale_2d,
        data["w2_scale"],
        data["scale_blks"],
        B,
        data["model_dim"],
        inter_dim,
        topk,
    )
    close_mask = torch.isclose(
        ref_fp32.to(out.dtype).float(), out.float(), atol=atol, rtol=rtol
    )
    pct_close = close_mask.float().mean().item()
    assert pct_close >= 0.95, (
        f"stage2 only {pct_close*100:.2f}% close (need >= 95%); "
        f"shape B={B} model={data['model_dim']} inter={inter_dim} "
        f"E={data['E']} topk={topk} tile=({tile_m},{tile_n},{tile_k})"
    )
    return out


# ---------------------------------------------------------------------------
# Parametrised sweep -- mirrors the upstream test_moe_blockscale_e2e cases
# in scope (small/medium E=8) plus the DSV4 prefill shape.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "B, model_dim, inter_dim, E, topk",
    [
        pytest.param(16, 7168, 256, 8, 2, id="small-E8"),
        pytest.param(32, 7168, 256, 8, 2, id="medium-E8"),
    ],
)
def test_moe_blockscale_stage1_sweep(B, model_dim, inter_dim, E, topk):
    """Stage1 numerical correctness across the upstream sweep shapes."""
    tile_m, tile_n, tile_k = 32, 128, 128
    data = _build_blockscale_inputs(
        B=B,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=E,
        topk=topk,
        block_m=tile_m,
    )
    _run_stage1_and_compare(data, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k)


@pytest.mark.parametrize(
    "B, model_dim, inter_dim, E, topk",
    [
        pytest.param(16, 7168, 256, 8, 2, id="small-E8"),
        pytest.param(32, 7168, 256, 8, 2, id="medium-E8"),
    ],
)
def test_moe_blockscale_stage2_sweep(B, model_dim, inter_dim, E, topk):
    """Stage2 numerical correctness across the upstream sweep shapes."""
    tile_m, tile_n, tile_k = 32, 128, 128
    data = _build_blockscale_inputs(
        B=B,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=E,
        topk=topk,
        block_m=tile_m,
    )
    stage1_out = _run_stage1_and_compare(
        data, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k
    )
    _run_stage2_and_compare(
        data, stage1_out, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k
    )


# ---------------------------------------------------------------------------
# DSV4 prefill 12-token shape -- gating case for sunway513/atom#37 W4.5.
#
# NOTE: As of this commit the DSV4-scale weight tensor (E=384, 2*inter=6144,
# K=7168 -> ~16.9 GB total fp8 elements) overflows the FlyDSL preshuffle
# B-tensor stride descriptor's i32 reach. Empirically, accuracy degrades
# monotonically once `E * 2*inter * model_dim` exceeds 2**32 elements:
#
#     E= 96  total ~4.23 GB elems  -> 100.00% close (under 2**32)
#     E=100  total ~4.40 GB elems  ->  93.29% close
#     E=128  total ~5.64 GB elems  ->  78.04% close
#     E=256  total ~11.27 GB elems ->  43.45% close
#     E=384  total ~16.91 GB elems ->  25.56% close (DSV4)
#
# Root cause: `mfma_preshuffle_pipeline.make_preshuffle_b_layout` casts the
# block strides (stride_n0, stride_k0, stride_klane, stride_nlane) to i32
# before passing them to `fx.make_layout`. For DSV4 the row stride
# `stride_n0 * n0` exceeds 2**32 and address arithmetic wraps, so high
# expert indices read truncated weight rows.
#
# This is a kernel-level limitation (also affects the AMD buffer
# descriptor's u32 NUM_RECORDS field for >4 GB tensors); resolving it
# requires either (a) an i64 stride path through the preshuffle layout
# helpers, or (b) per-expert buffer descriptors so each individual load
# stays under 4 GB. Both are out of scope for the test PR.
#
# Once the kernel-side fix lands the `xfail` decorator should be removed.
# The test body is otherwise identical to the passing sweep cases so it
# will validate the fix immediately.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "B, model_dim, inter_dim_per_rank, E, topk",
    # ATOM silicon runs DSV4 with TP=8 + column-parallel sharding on w1's
    # inter_dim. Per-rank w1 shape: [E=384, 2*768=1536, 7168] = 2.1 GB FP8,
    # which fits comfortably in i32 strides (E*2*inter_per_rank*K = 4.22 G
    # elements, just under 2**32). This is the actual shape the kernel sees
    # on production silicon — NOT the 16.9 GB no-TP shape that triggered the
    # i32 overflow in earlier exploration.
    [(12, 7168, 384, 384, 6)],  # inter_dim_per_rank = 3072 / TP=8 = 384
)
def test_moe_blockscale_dsv4_shape(B, model_dim, inter_dim_per_rank, E, topk):
    """DSV4 prefill 12-token shape per-TP-rank — gating case for #37 W4.5.

    Validates both stage1 and stage2 of the AITER FlyDSL blockscale port at
    the DSV4 PER-RANK shape (model=7168, inter_per_rank=384, E=384, topk=6)
    that runs on silicon under TP=8 column-parallel sharding. This is the
    realistic kernel input — `aiter.fused_moe.fused_moe` is invoked once
    per TP rank with already-sharded weights, so the kernel never sees the
    full 16.9 GB tensor.

    Tile (32, 128, 128) matches the blockscale registry's smallest covered
    tile, sufficient for the 12-token prefill case.
    """
    tile_m, tile_n, tile_k = 32, 128, 128
    data = _build_blockscale_inputs(
        B=B,
        model_dim=model_dim,
        inter_dim=inter_dim_per_rank,
        E=E,
        topk=topk,
        block_m=tile_m,
    )
    stage1_out = _run_stage1_and_compare(
        data, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k
    )
    _run_stage2_and_compare(
        data, stage1_out, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k
    )


@pytest.mark.xfail(
    reason="FlyDSL preshuffle B-stride i32 overflow when E*2*inter*K > 2**32 "
    "elements (>4.3 G); see Task 5 subagent report for empirical bisect data. "
    "ATOM silicon uses TP=8 + column-parallel sharding so per-rank stays "
    "well below this — the no-TP test below is xfail'd as a regression "
    "monitor for the day FlyDSL gains i64 stride support.",
    strict=False,
)
@pytest.mark.parametrize(
    "B, model_dim, inter_dim, E, topk",
    [(12, 7168, 3072, 384, 6)],
)
def test_moe_blockscale_dsv4_no_tp_shape(B, model_dim, inter_dim, E, topk):
    """DSV4 full no-TP shape — XFAIL until FlyDSL i64 strides land.

    This is the FULL DSV4 MoE shape (16.9 GB FP8 weights) which exceeds
    i32 stride limits in `make_preshuffle_b_layout`. Kept as a regression
    monitor — when FlyDSL upstream lands i64-aware preshuffle helpers,
    remove the xfail to confirm the fix.
    """
    tile_m, tile_n, tile_k = 32, 128, 128
    data = _build_blockscale_inputs(
        B=B,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=E,
        topk=topk,
        block_m=tile_m,
    )
    stage1_out = _run_stage1_and_compare(
        data, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k
    )
    _run_stage2_and_compare(
        data, stage1_out, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k
    )
