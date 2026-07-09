# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for FlyDSL MOE a16wfp4 (bf16 activation, mxfp4 weight, per_1x32).

Tests flydsl_moe_stage1/stage2 via mixed_moe_gemm_2stage a16w4 kernels:
  - Stage1: a_dtype="bf16", b_dtype="fp4"
  - Stage2: a_dtype="bf16", b_dtype="fp4"
  - End-to-end (stage1 -> stage2, bf16 activations throughout)

Reference: torch_moe_stage1 / torch_moe_stage2 with a1_scale=None, a2_scale=None.

Usage:
    pytest op_tests/flydsl_tests/test_flydsl_moe_a16wfp4.py -q
    python op_tests/flydsl_tests/test_flydsl_moe_a16wfp4.py
    python op_tests/flydsl_tests/test_flydsl_moe_a16wfp4.py --stage stage1 -t 16 64
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402
import aiter  # noqa: E402
from aiter import dtypes, QuantType, ActivationType  # noqa: E402
from aiter.fused_moe import (  # noqa: E402
    fused_topk,
    moe_sorting,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.jit.utils.chip_info import get_gfx  # noqa: E402
from aiter.ops.flydsl.utils import is_flydsl_available  # noqa: E402
from aiter.ops.flydsl.moe_kernels import (  # noqa: E402
    pick_flydsl_stage1_tile_n,
    pick_flydsl_stage2_tile_k,
)
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4  # noqa: E402
from aiter.utility.fp4_utils import e8m0_shuffle  # noqa: E402

_CUDA = torch.device("cuda")

_SKIP_GFX950_FLYDSL = pytest.mark.skipif(
    get_gfx() not in ("gfx950",) or not is_flydsl_available(),
    reason="gfx950 FlyDSL required",
)

Q_TYPE = QuantType.per_1x32
Q_DTYPE_W = dtypes.fp4x2


def _inter_pad(inter_dim: int) -> int:
    return ((inter_dim + 255) // 256 * 256) - inter_dim


def _stage1_tile_k(model_dim: int) -> int:
    return 512 if (model_dim % 512 == 0) else 256


def _generate_a16wfp4_data(
    token: int,
    model_dim: int,
    inter_dim: int,
    E: int,
    topk: int,
    block_m: int,
    dtype=torch.bfloat16,
    doweight_stage1: bool = False,
    seed: int = 0,
    activation: ActivationType = ActivationType.Silu,
    situ_beta: float = 1.0,
    situ_linear_beta: float = 1.0,
):
    """bf16 activations + mxfp4 weights; torch reference for stage1/stage2."""
    torch_quant = aiter.get_torch_quant(Q_TYPE)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    inter_pad = _inter_pad(inter_dim)

    inp = torch.randn((token, model_dim), dtype=dtype, device=_CUDA) / 10
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype, device=_CUDA) / 10
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype, device=_CUDA) / 10
    if inter_pad:
        w1[:, -inter_pad:, :] = 0
        w1[:, inter_dim - inter_pad : inter_dim, :] = 0
        w2[:, :, -inter_pad:] = 0
    score = torch.randn((token, E), dtype=dtype, device=_CUDA)
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)

    w1_qt, w1_scale = torch_quant(w1, quant_dtype=Q_DTYPE_W)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=Q_DTYPE_W)
    w1_qt = w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2)
    w2_qt = w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2)

    _s1_extra = {}
    if activation == ActivationType.Situv2:
        _s1_extra = dict(situ_beta=situ_beta, situ_linear_beta=situ_linear_beta)
    ref1 = torch_moe_stage1(
        inp,
        w1_qt,
        w2_qt,
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=activation,
        quant_type=Q_TYPE,
        a1_scale=None,
        w1_scale=w1_scale,
        doweight=doweight_stage1,
        **_s1_extra,
    )

    a2 = ref1

    ref2 = torch_moe_stage2(
        a2,
        w1_qt,
        w2_qt,
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=Q_TYPE,
        w2_scale=w2_scale,
        a2_scale=None,
        doweight=not doweight_stage1,
    )

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_m
    )

    if doweight_stage1:
        sorted_weights_s1 = sorted_weights
        sorted_weights_s2 = None
    else:
        sorted_weights_s1 = None
        sorted_weights_s2 = sorted_weights

    w1_qt_shuf = shuffle_weight_a16w4(w1_qt, 16, False)
    w1_scale_shuf = shuffle_scale_a16w4(w1_scale, E, False)
    w2_qt_shuf = shuffle_weight_a16w4(w2_qt, 16, False)
    w2_scale_shuf = e8m0_shuffle(w2_scale)

    return dict(
        inter_pad=inter_pad,
        ref_stage1=ref1,
        ref_stage2=ref2,
        inp=inp,
        a2=a2,
        w1_qt=w1_qt,
        w1_scale=w1_scale,
        topk_ids=topk_ids,
        w1_qt_shuf=w1_qt_shuf,
        w1_scale_shuf=w1_scale_shuf,
        w2_qt_shuf=w2_qt_shuf,
        w2_scale_shuf=w2_scale_shuf,
        sorted_ids=sorted_ids,
        sorted_weights_s1=sorted_weights_s1,
        sorted_weights_s2=sorted_weights_s2,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        dtype=dtype,
    )


def _check_result(ref_out, test_out, atol=1.0, rtol=0.05, pass_pct=95.0):
    r = ref_out.float().reshape(-1)
    t = test_out.float().reshape(-1)
    max_delta = (r - t).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(r, t, dim=0).item()
    rel_l2 = (t - r).norm().item() / (r.norm().item() + 1e-12)
    # Primary gate: cosine similarity (scale-invariant, catches the bugs atol=1.0 missed)
    cos_ok = cos > 0.999
    # Secondary: element-wise close (kept for backward compat, but using tighter atol)
    close_mask = torch.isclose(r, t, atol=atol, rtol=rtol)
    pct_close = close_mask.float().mean().item() * 100
    passed = cos_ok and pct_close > pass_pct
    print(
        f"  cos={cos:.5f} rel_l2={rel_l2:.4f} "
        f"max_delta={max_delta:.4f}, {pct_close:.1f}% close (atol={atol}, rtol={rtol})"
    )
    print(f"  ref  sample: {ref_out.reshape(-1)[:8]}")
    print(f"  test sample: {test_out.reshape(-1)[:8]}")
    print(f"  --> {'PASS' if passed else 'FAIL'}")
    return passed, max_delta, pct_close


@_SKIP_GFX950_FLYDSL
@pytest.mark.parametrize(
    "inter_dim,seed",
    [
        pytest.param(256, 101, id="i256"),
        pytest.param(384, 102, id="i384"),
        pytest.param(640, 0, id="i640_dsv4"),
    ],
)
def test_flydsl_stage1_a16wfp4_non256(inter_dim, seed):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1

    token, model_dim, E, topk, block_m = 16, 512, 8, 2, 32
    data = _generate_a16wfp4_data(
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=E,
        topk=topk,
        block_m=block_m,
        seed=seed,
    )
    out = flydsl_moe_stage1(
        a=data["inp"],
        w1=data["w1_qt_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=block_m,
        tile_n=256,
        tile_k=_stage1_tile_k(model_dim),
        a_dtype="bf16",
        b_dtype="fp4",
        out_dtype="bf16",
        w1_scale=data["w1_scale_shuf"],
        a1_scale=None,
        sorted_weights=data["sorted_weights_s1"],
        inter_dim_pad=data["inter_pad"],
    )
    torch.cuda.synchronize()
    passed, max_delta, pct_close = _check_result(data["ref_stage1"], out)
    assert passed, (
        f"stage1_a16wfp4_i{inter_dim} FAIL: max_delta={max_delta:.4f}, "
        f"{pct_close:.1f}% close"
    )


@_SKIP_GFX950_FLYDSL
@pytest.mark.parametrize(
    "inter_dim,seed",
    [
        pytest.param(256, 101, id="i256"),
        pytest.param(384, 102, id="i384"),
        pytest.param(640, 0, id="i640_dsv4"),
    ],
)
@pytest.mark.parametrize("mode", ["atomic"])
def test_flydsl_stage2_a16wfp4_non256(inter_dim, seed, mode):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2

    token, model_dim, E, topk, block_m = 16, 512, 8, 2, 32
    data = _generate_a16wfp4_data(
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=E,
        topk=topk,
        block_m=block_m,
        seed=seed,
    )
    out = flydsl_moe_stage2(
        inter_states=data["a2"],
        w2=data["w2_qt_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=block_m,
        tile_n=128,
        tile_k=256,
        a_dtype="bf16",
        b_dtype="fp4",
        out_dtype="bf16",
        mode=mode,
        w2_scale=data["w2_scale_shuf"],
        a2_scale=None,
        sorted_weights=data["sorted_weights_s2"],
        inter_dim_pad=data["inter_pad"],
    )
    torch.cuda.synchronize()
    passed, max_delta, pct_close = _check_result(data["ref_stage2"], out)
    assert passed, (
        f"stage2_a16wfp4_i{inter_dim}_{mode} FAIL: max_delta={max_delta:.4f}, "
        f"{pct_close:.1f}% close"
    )


@_SKIP_GFX950_FLYDSL
@pytest.mark.parametrize("inter_dim", [640])
def test_flydsl_e2e_a16wfp4_non256(inter_dim):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2

    token, model_dim, E, topk, block_m, seed = 16, 512, 8, 2, 32, 0
    data = _generate_a16wfp4_data(
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=E,
        topk=topk,
        block_m=block_m,
        seed=seed,
    )
    stage1_out = flydsl_moe_stage1(
        a=data["inp"],
        w1=data["w1_qt_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=block_m,
        tile_n=256,
        tile_k=_stage1_tile_k(model_dim),
        a_dtype="bf16",
        b_dtype="fp4",
        out_dtype="bf16",
        w1_scale=data["w1_scale_shuf"],
        a1_scale=None,
        sorted_weights=data["sorted_weights_s1"],
        inter_dim_pad=data["inter_pad"],
    )
    torch.cuda.synchronize()
    e2e_out = flydsl_moe_stage2(
        inter_states=stage1_out,
        w2=data["w2_qt_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=block_m,
        tile_n=128,
        tile_k=256,
        a_dtype="bf16",
        b_dtype="fp4",
        out_dtype="bf16",
        mode="atomic",
        w2_scale=data["w2_scale_shuf"],
        a2_scale=None,
        sorted_weights=data["sorted_weights_s2"],
        inter_dim_pad=data["inter_pad"],
    )
    torch.cuda.synchronize()
    passed, max_delta, pct_close = _check_result(
        data["ref_stage2"], e2e_out, pass_pct=90.0
    )
    assert passed, (
        f"e2e_a16wfp4_i{inter_dim} FAIL: max_delta={max_delta:.4f}, "
        f"{pct_close:.1f}% close"
    )


def test_pick_flydsl_stage2_tile_k_a16wfp4():
    assert pick_flydsl_stage2_tile_k(640) == 128
    assert pick_flydsl_stage2_tile_k(384) == 128
    assert pick_flydsl_stage2_tile_k(256) == 256


def test_pick_flydsl_stage1_tile_n_a16wfp4():
    assert pick_flydsl_stage1_tile_n(640) == 128
    assert pick_flydsl_stage1_tile_n(384) == 128
    assert pick_flydsl_stage1_tile_n(256) == 256


def _situv2_stage1_ref(
    inp, w1_qt, w1_scale, topk_ids, E, model_dim, inter_dim, beta, linear_beta
):
    """Inline SiTUv2 stage1 reference (mxfp4 dequant GEMM + situv2 activation).

    Mirrors the kernel: clamp gate<=7, up in [-7,7], then
    situ_g = beta*tanh(g/beta)*sigmoid(g); up_s = linear_beta*tanh(u/linear_beta).
    """
    from aiter.utility import fp4_utils

    N = inter_dim * 2
    w1_deq = fp4_utils.mxfp4_to_f32(w1_qt)
    w1_sc = fp4_utils.e8m0_to_f32(w1_scale)
    g = model_dim // 32
    w1_full = (w1_deq.view(E, N, g, 32) * w1_sc.view(E, N, g, 1)).view(E, N, model_dim)
    hidden = inp.float()
    token, topk = topk_ids.shape
    out = torch.zeros((token, topk, inter_dim), dtype=torch.float32, device=inp.device)
    for t in range(token):
        for k in range(topk):
            e = int(topk_ids[t, k])
            gu = hidden[t] @ w1_full[e].T
            gate = gu[:inter_dim].clamp(max=7.0)
            up = gu[inter_dim:].clamp(min=-7.0, max=7.0)
            situ_g = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
            up_s = linear_beta * torch.tanh(up / linear_beta)
            out[t, k] = situ_g * up_s
    return out.to(torch.bfloat16)


@_SKIP_GFX950_FLYDSL
@pytest.mark.parametrize("beta,linear_beta", [(1.0, 1.0), (0.5, 2.0), (1.5, 0.8)])
def test_flydsl_stage1_a16wfp4_situv2(
    beta,
    linear_beta,
    token: int = 128,
    model_dim: int = 3072,
    inter_dim: int = 256,
    E: int = 256,
    topk: int = 8,
    block_m: int = 32,
):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1

    print(f"\n[TEST] a16wfp4 stage1 SiTUv2 beta={beta} linear_beta={linear_beta}")
    data = _generate_a16wfp4_data(
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=E,
        topk=topk,
        block_m=block_m,
    )
    ref = _situv2_stage1_ref(
        data["inp"],
        data["w1_qt"],
        data["w1_scale"],
        data["topk_ids"],
        E,
        model_dim,
        inter_dim,
        beta,
        linear_beta,
    )
    out = flydsl_moe_stage1(
        a=data["inp"],
        w1=data["w1_qt_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=block_m,
        tile_n=256,
        tile_k=256,
        a_dtype="bf16",
        b_dtype="fp4",
        out_dtype="bf16",
        act="situv2",
        situ_beta=beta,
        situ_linear_beta=linear_beta,
        w1_scale=data["w1_scale_shuf"],
        a1_scale=None,
        sorted_weights=data["sorted_weights_s1"],
    )
    torch.cuda.synchronize()
    passed, max_delta, pct_close = _check_result(ref, out)
    assert passed, (
        f"stage1_a16wfp4_situv2 b{beta} lb{linear_beta} FAIL: "
        f"max_delta={max_delta:.4f}, {pct_close:.1f}% close"
    )


@_SKIP_GFX950_FLYDSL
def test_flydsl_stage1_a16wfp4(
    token: int = 16,
    model_dim: int = 512,
    inter_dim: int = 256,
    E: int = 64,
    topk: int = 4,
    block_m: int = 32,
    k_batch_intra_block: int = 1,
    atol: float = 1.0,
    rtol: float = 0.05,
):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1

    print(f"\n{'='*70}")
    print(
        f"[TEST] FlyDSL stage1 A16WFP4: token={token}, dim=({model_dim},{inter_dim}), "
        f"E={E}, topk={topk}, block_m={block_m}, k_batch={k_batch_intra_block}"
    )
    print(f"{'='*70}")

    data = _generate_a16wfp4_data(
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=E,
        topk=topk,
        block_m=block_m,
    )
    out_dtype_str = "bf16" if data["dtype"] == torch.bfloat16 else "f16"

    out = flydsl_moe_stage1(
        a=data["inp"],
        w1=data["w1_qt_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=block_m,
        tile_n=256,
        tile_k=256,
        a_dtype="bf16",
        b_dtype="fp4",
        out_dtype=out_dtype_str,
        w1_scale=data["w1_scale_shuf"],
        a1_scale=None,
        sorted_weights=data["sorted_weights_s1"],
        k_batch_intra_block=k_batch_intra_block,
    )
    torch.cuda.synchronize()

    passed, max_delta, pct_close = _check_result(
        data["ref_stage1"], out, atol=atol, rtol=rtol
    )
    assert (
        passed
    ), f"stage1_a16wfp4 FAIL: max_delta={max_delta:.4f}, {pct_close:.1f}% close"


@_SKIP_GFX950_FLYDSL
def test_flydsl_stage2_a16wfp4(
    token: int = 16,
    model_dim: int = 512,
    inter_dim: int = 256,
    E: int = 64,
    topk: int = 4,
    block_m: int = 32,
    mode: str = "atomic",
    atol: float = 1.0,
    rtol: float = 0.05,
):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2

    print(f"\n{'='*70}")
    print(
        f"[TEST] FlyDSL stage2 A16WFP4: token={token}, dim=({model_dim},{inter_dim}), "
        f"E={E}, topk={topk}, block_m={block_m}, mode={mode}"
    )
    print(f"{'='*70}")

    data = _generate_a16wfp4_data(
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=E,
        topk=topk,
        block_m=block_m,
    )
    out_dtype_str = "bf16" if data["dtype"] == torch.bfloat16 else "f16"

    out = flydsl_moe_stage2(
        inter_states=data["a2"],
        w2=data["w2_qt_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=block_m,
        tile_n=128,
        tile_k=256,
        a_dtype="bf16",
        b_dtype="fp4",
        out_dtype=out_dtype_str,
        mode=mode,
        w2_scale=data["w2_scale_shuf"],
        a2_scale=None,
        sorted_weights=data["sorted_weights_s2"],
    )
    torch.cuda.synchronize()

    passed, max_delta, pct_close = _check_result(
        data["ref_stage2"], out, atol=atol, rtol=rtol
    )
    assert (
        passed
    ), f"stage2_a16wfp4_{mode} FAIL: max_delta={max_delta:.4f}, {pct_close:.1f}% close"


@_SKIP_GFX950_FLYDSL
def test_flydsl_e2e_a16wfp4(
    token: int = 16,
    model_dim: int = 512,
    inter_dim: int = 256,
    E: int = 64,
    topk: int = 4,
    block_m: int = 32,
    mode: str = "atomic",
    atol: float = 1.0,
    rtol: float = 0.05,
):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2

    print(f"\n{'='*70}")
    print(
        f"[TEST] FlyDSL E2E A16WFP4: token={token}, dim=({model_dim},{inter_dim}), "
        f"E={E}, topk={topk}, block_m={block_m}, mode={mode}"
    )
    print(f"{'='*70}")

    data = _generate_a16wfp4_data(
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=E,
        topk=topk,
        block_m=block_m,
    )
    out_dtype_str = "bf16" if data["dtype"] == torch.bfloat16 else "f16"

    stage1_out = flydsl_moe_stage1(
        a=data["inp"],
        w1=data["w1_qt_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=block_m,
        tile_n=256,
        tile_k=256,
        a_dtype="bf16",
        b_dtype="fp4",
        out_dtype=out_dtype_str,
        w1_scale=data["w1_scale_shuf"],
        a1_scale=None,
        sorted_weights=data["sorted_weights_s1"],
    )
    torch.cuda.synchronize()

    e2e_out = flydsl_moe_stage2(
        inter_states=stage1_out,
        w2=data["w2_qt_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=block_m,
        tile_n=128,
        tile_k=256,
        a_dtype="bf16",
        b_dtype="fp4",
        out_dtype=out_dtype_str,
        mode=mode,
        w2_scale=data["w2_scale_shuf"],
        a2_scale=None,
        sorted_weights=data["sorted_weights_s2"],
    )
    torch.cuda.synchronize()

    passed, max_delta, pct_close = _check_result(
        data["ref_stage2"], e2e_out, atol=atol, rtol=rtol, pass_pct=90.0
    )
    assert (
        passed
    ), f"e2e_a16wfp4_{mode} FAIL: max_delta={max_delta:.4f}, {pct_close:.1f}% close"


@_SKIP_GFX950_FLYDSL
@pytest.mark.parametrize("beta,linear_beta", [(1.0, 1.0), (2.0, 1.5)])
def test_flydsl_a16wfp4_situv2_fused_moe(
    beta,
    linear_beta,
    token: int = 128,
    model_dim: int = 3072,
    inter_dim: int = 256,
    E: int = 32,
    topk: int = 8,
):
    """a16w4 SiTUv2 routed through the public fused_moe API (2-stage).

    Exercises the get_2stage_cfgs bf16xfp4 SiTUv2 routing: fused_moe must select
    the mixed_moe a16w4 kernel (a_type='bf16', separated gate) and keep the
    activation in bf16 (q_dtype_a inferred as bf16, not fp4). Weights use the
    a16w4 layout (shuffle_weight_a16w4 gate_up=False, w2 scale via e8m0_shuffle).
    """
    from aiter.fused_moe import fused_moe
    from aiter.ops.flydsl.moe_common import GateMode

    torch.set_default_device("cuda")
    torch_quant = aiter.get_torch_quant(Q_TYPE)
    torch.manual_seed(0)
    inp = torch.randn((token, model_dim), dtype=torch.bfloat16, device=_CUDA) / 10
    w1 = (
        torch.randn((E, inter_dim * 2, model_dim), dtype=torch.bfloat16, device=_CUDA)
        / 10
    )
    w2 = torch.randn((E, model_dim, inter_dim), dtype=torch.bfloat16, device=_CUDA) / 10
    score = torch.randn((token, E), dtype=torch.bfloat16, device=_CUDA)
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)

    w1_qt, w1_scale = torch_quant(w1, quant_dtype=Q_DTYPE_W)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=Q_DTYPE_W)
    w1_qt = w1_qt.view(E, inter_dim * 2, model_dim // 2)
    w2_qt = w2_qt.view(E, model_dim, inter_dim // 2)

    ref1 = torch_moe_stage1(
        inp,
        w1_qt,
        w2_qt,
        topk_weights,
        topk_ids,
        dtype=torch.bfloat16,
        activation=ActivationType.Situv2,
        quant_type=Q_TYPE,
        a1_scale=None,
        w1_scale=w1_scale,
        situ_beta=beta,
        situ_linear_beta=linear_beta,
    )
    ref2 = torch_moe_stage2(
        ref1,
        w1_qt,
        w2_qt,
        topk_weights,
        topk_ids,
        dtype=torch.bfloat16,
        quant_type=Q_TYPE,
        w2_scale=w2_scale,
        a2_scale=None,
        doweight=True,
    )

    out = fused_moe(
        inp,
        shuffle_weight_a16w4(w1_qt, 16, False),
        shuffle_weight_a16w4(w2_qt, 16, False),
        topk_weights,
        topk_ids,
        w1_scale=shuffle_scale_a16w4(w1_scale, E, False),
        w2_scale=e8m0_shuffle(w2_scale),
        quant_type=Q_TYPE,
        activation=ActivationType.Situv2,
        doweight_stage1=False,
        beta=beta,
        linear_beta=linear_beta,
        gate_mode=GateMode.SEPARATED.value,
    )
    torch.cuda.synchronize()
    passed, max_delta, pct_close = _check_result(ref2, out, pass_pct=90.0)
    assert passed, (
        f"a16wfp4_situv2_fused_moe b{beta} lb{linear_beta} FAIL: "
        f"max_delta={max_delta:.4f}, {pct_close:.1f}% close"
    )


_ACT_MAP = {"silu": ActivationType.Silu, "situv2": ActivationType.Situv2}


def _timing_kwargs(timing: str) -> dict:
    """Map a --timing choice to run_perftest kwargs.

    cuda_event : wall-clock mean per launch (includes host dispatch + launch
                 gaps + per-iter empty_cache) -- inflates tiny/small-token
                 kernels by a fixed host-overhead floor.
    device     : torch-profiler *device* time only (pure kernel), IQR-trimmed
                 when num_iters > 30. Matches "median-of-device-time" harnesses.
    graph      : CUDA graph replay + device time (lowest host overhead).
    """
    if timing == "cuda_event":
        return dict(use_cuda_event=True)
    if timing == "graph":
        return dict(use_cuda_event=False, testGraph=True)
    return dict(use_cuda_event=False)  # device


def _sweep_perf(args):
    """Correctness + perf sweep over (token x inter_dim), incl. non-256 cases."""
    from aiter.test_common import run_perftest
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2

    activation = _ACT_MAP[args.activation]
    timing_kw = _timing_kwargs(args.timing)
    print(
        f"\nSweep: model_dim={args.model_dim} E={args.experts} topk={args.topk} "
        f"act={args.activation} beta={args.beta} linear_beta={args.linear_beta} "
        f"block_m={args.block_m[0]} num_iters={args.num_iters}\n"
    )
    header = (
        f"{'tok':>5} {'inter':>6} {'s1_tn':>5} {'s2_tk':>5} "
        f"{'s1_us':>9} {'s1_TFLOPs':>10} {'s2_us':>9} {'s2_TFLOPs':>10} "
        f"{'s1':>4} {'s2':>4}"
    )
    print(header)
    print("-" * len(header))

    bm = args.block_m[0]
    all_ok = True
    for token in args.tokens:
        for inter_dim in args.inter_dims:
            try:
                data = _generate_a16wfp4_data(
                    token=token,
                    model_dim=args.model_dim,
                    inter_dim=inter_dim,
                    E=args.experts,
                    topk=args.topk,
                    block_m=bm,
                    seed=0,
                    activation=activation,
                    situ_beta=args.beta,
                    situ_linear_beta=args.linear_beta,
                )
                inter_pad = data["inter_pad"]
                s1_tn = pick_flydsl_stage1_tile_n(inter_dim)
                s2_tk = pick_flydsl_stage2_tile_k(inter_dim)

                s1_out, s1_us = run_perftest(
                    flydsl_moe_stage1,
                    a=data["inp"],
                    w1=data["w1_qt_shuf"],
                    sorted_token_ids=data["sorted_ids"],
                    sorted_expert_ids=data["sorted_expert_ids"],
                    num_valid_ids=data["num_valid_ids"],
                    topk=args.topk,
                    tile_m=bm,
                    tile_n=256,
                    tile_k=_stage1_tile_k(args.model_dim),
                    a_dtype="bf16",
                    b_dtype="fp4",
                    out_dtype="bf16",
                    act=args.activation,
                    situ_beta=args.beta,
                    situ_linear_beta=args.linear_beta,
                    w1_scale=data["w1_scale_shuf"],
                    a1_scale=None,
                    sorted_weights=data["sorted_weights_s1"],
                    inter_dim_pad=inter_pad,
                    num_iters=args.num_iters,
                    **timing_kw,
                )
                torch.cuda.synchronize()
                s1_pass, _, _ = _check_result(data["ref_stage1"], s1_out)

                s2_out, s2_us = run_perftest(
                    flydsl_moe_stage2,
                    inter_states=data["a2"],
                    w2=data["w2_qt_shuf"],
                    sorted_token_ids=data["sorted_ids"],
                    sorted_expert_ids=data["sorted_expert_ids"],
                    num_valid_ids=data["num_valid_ids"],
                    topk=args.topk,
                    tile_m=bm,
                    tile_n=128,
                    tile_k=256,
                    a_dtype="bf16",
                    b_dtype="fp4",
                    out_dtype="bf16",
                    mode="atomic",
                    w2_scale=data["w2_scale_shuf"],
                    a2_scale=None,
                    sorted_weights=data["sorted_weights_s2"],
                    inter_dim_pad=inter_pad,
                    num_iters=args.num_iters,
                    **timing_kw,
                )
                torch.cuda.synchronize()
                s2_pass, _, _ = _check_result(data["ref_stage2"], s2_out)

                s1_tflops = (
                    2.0 * token * args.topk * args.model_dim * (2 * inter_dim)
                ) / (s1_us * 1e6)
                s2_tflops = (2.0 * token * args.topk * inter_dim * args.model_dim) / (
                    s2_us * 1e6
                )
            except Exception as e:  # noqa: BLE001
                all_ok = False
                print(f"{token:>5} {inter_dim:>6}  ERROR: {type(e).__name__}: {e}")
                torch.cuda.empty_cache()
                continue

            all_ok = all_ok and s1_pass and s2_pass
            print(
                f"{token:>5} {inter_dim:>6} {s1_tn:>5} {s2_tk:>5} "
                f"{s1_us:>9.2f} {s1_tflops:>10.1f} {s2_us:>9.2f} {s2_tflops:>10.1f} "
                f"{'OK' if s1_pass else 'FAIL':>4} {'OK' if s2_pass else 'FAIL':>4}"
            )
            torch.cuda.empty_cache()

    print("\n" + ("ALL PASS" if all_ok else "SOME FAILED"))
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="FlyDSL MOE A16WFP4 unit tests")
    parser.add_argument("-t", "--tokens", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--inter-dim", type=int, default=256)
    parser.add_argument("-E", "--experts", type=int, default=64)
    parser.add_argument("-k", "--topk", type=int, default=4)
    parser.add_argument("--block-m", type=int, nargs="+", default=[32])
    parser.add_argument("--k-batch", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--mode", type=str, nargs="+", default=["atomic"], choices=["atomic", "reduce"]
    )
    parser.add_argument(
        "--stage",
        type=str,
        nargs="+",
        default=["stage1", "stage2", "e2e"],
        choices=["stage1", "stage2", "e2e"],
    )
    parser.add_argument("--atol", type=float, default=1.0)
    parser.add_argument("--rtol", type=float, default=0.05)
    # --- perf sweep mode (correctness + latency/TFLOPs over an inter_dim sweep) ---
    parser.add_argument(
        "--perf",
        action="store_true",
        help="Run the correctness+perf sweep (incl. non-256 inter_dim) instead "
        "of the unit tests.",
    )
    parser.add_argument(
        "--inter-dims",
        type=int,
        nargs="+",
        default=[256, 384, 512, 640, 768, 896, 1024],
        help="inter_dim sweep values for --perf mode.",
    )
    parser.add_argument(
        "--activation", type=str, default="situv2", choices=list(_ACT_MAP.keys())
    )
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--linear-beta", type=float, default=1.0)
    parser.add_argument("--num-iters", type=int, default=50)
    parser.add_argument( "--timing", type=str, default="cuda_event",
        choices=["cuda_event", "device", "graph"]
    )
    args = parser.parse_args()

    if not is_flydsl_available():
        print("[SKIP] FlyDSL is not available.")
        sys.exit(0)

    if args.perf:
        # perf mode uses realistic defaults (override with the flags above).
        if args.model_dim == 512:
            args.model_dim = 7168
        if args.experts == 64:
            args.experts = 256
        if args.topk == 4:
            args.topk = 8
        if args.tokens == [16, 64, 256]:
            args.tokens = [128, 1024]
        ok = _sweep_perf(args)
        sys.exit(0 if ok else 1)

    results = []
    for token in args.tokens:
        for bm in args.block_m:
            if "stage1" in args.stage:
                for kb in args.k_batch:
                    name = f"stage1_a16wfp4_t{token}_bm{bm}_kb{kb}"
                    try:
                        test_flydsl_stage1_a16wfp4(
                            token=token,
                            model_dim=args.model_dim,
                            inter_dim=args.inter_dim,
                            E=args.experts,
                            topk=args.topk,
                            block_m=bm,
                            k_batch_intra_block=kb,
                            atol=args.atol,
                            rtol=args.rtol,
                        )
                        results.append((name, "PASS"))
                    except Exception:
                        import traceback

                        traceback.print_exc()
                        results.append((name, "ERROR"))

            if "stage2" in args.stage:
                for mode in args.mode:
                    name = f"stage2_a16wfp4_t{token}_bm{bm}_{mode}"
                    try:
                        test_flydsl_stage2_a16wfp4(
                            token=token,
                            model_dim=args.model_dim,
                            inter_dim=args.inter_dim,
                            E=args.experts,
                            topk=args.topk,
                            block_m=bm,
                            mode=mode,
                            atol=args.atol,
                            rtol=args.rtol,
                        )
                        results.append((name, "PASS"))
                    except Exception:
                        import traceback

                        traceback.print_exc()
                        results.append((name, "ERROR"))

            if "e2e" in args.stage:
                for mode in args.mode:
                    name = f"e2e_a16wfp4_t{token}_bm{bm}_{mode}"
                    try:
                        test_flydsl_e2e_a16wfp4(
                            token=token,
                            model_dim=args.model_dim,
                            inter_dim=args.inter_dim,
                            E=args.experts,
                            topk=args.topk,
                            block_m=bm,
                            mode=mode,
                            atol=args.atol,
                            rtol=args.rtol,
                        )
                        results.append((name, "PASS"))
                    except Exception:
                        import traceback

                        traceback.print_exc()
                        results.append((name, "ERROR"))

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for name, status in results:
        print(f"  {status:>5s}  {name}")
    n_pass = sum(1 for _, s in results if s == "PASS")
    print(f"\n  {n_pass}/{len(results)} passed")
    if any(s == "ERROR" for _, s in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
