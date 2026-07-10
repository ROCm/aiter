# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MOE a8w4 (fp8 activation, fp4 weight, GUI shuffle) regression tests.

Covers stage2 tile_k auto-resolve for non-256-aligned inter_dim (e.g. DSV4
inter=640) and FlyDSL stage2 / E2E with GUI preshuffle on gfx950.

Usage:
    pytest op_tests/flydsl_tests/test_flydsl_moe_a8w4.py -q
    pytest op_tests/flydsl_tests/test_flydsl_moe_a8w4.py -k tile_k
"""

from __future__ import annotations

import os

import pytest
import torch

from aiter import dtypes, QuantType, ActivationType
from aiter.fused_moe import fused_topk, moe_sorting, torch_moe_stage1, torch_moe_stage2
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_kernels import (
    pick_flydsl_stage2_tile_k,
    resolve_flydsl_stage2_tile_k,
)
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.quant import (
    mxfp4_moe_sort_fwd,
    per_1x32_f4_quant,
    per_1x32_f8_scale_f8_quant,
)
from aiter.ops.shuffle import shuffle_weight, shuffle_weight_a16w4, shuffle_scale_a16w4
from aiter.utility.fp4_utils import e8m0_shuffle
from aiter.test_common import checkAllclose

Q_TYPE = QuantType.per_1x32

_SKIP_GFX950_FLYDSL = pytest.mark.skipif(
    get_gfx() not in ("gfx950",) or not is_flydsl_available(),
    reason="gfx950 FlyDSL required",
)


def _inter_pad(inter_dim: int) -> int:
    return ((inter_dim + 255) // 256 * 256) - inter_dim


def _stage1_tile_k(model_dim: int) -> int:
    return 512 if (model_dim % 512 == 0) else 256


def _check_close(ref, out, label, atol=1.0, rtol=0.05, max_err_ratio=0.05):
    assert not out.isnan().any(), f"{label}: output has NaN"
    assert not out.isinf().any(), f"{label}: output has Inf"
    err = checkAllclose(ref, out, msg=label, atol=atol, rtol=rtol)
    assert (
        err == 0 or err <= max_err_ratio
    ), f"{label}: checkAllclose failed (err={err}, max={max_err_ratio})"


def _generate_a8w4_gui_data(
    token: int,
    model_dim: int,
    inter_dim: int,
    E: int,
    topk: int,
    block_m: int,
    seed: int = 0,
    dtype=torch.bfloat16,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    inter_pad = _inter_pad(inter_dim)

    inp = torch.randn(token, model_dim, dtype=dtype, device="cuda") / 4
    w1 = torch.randn(E, inter_dim * 2, model_dim, dtype=dtype, device="cuda") / 4
    w2 = torch.randn(E, model_dim, inter_dim, dtype=dtype, device="cuda") / 4
    if inter_pad:
        w1[:, -inter_pad:, :] = 0
        w1[:, inter_dim - inter_pad : inter_dim, :] = 0
        w2[:, :, -inter_pad:] = 0

    score = torch.randn(token, E, dtype=dtype, device="cuda")
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_m
    )

    a_q, a_scale = per_1x32_f8_scale_f8_quant(
        inp, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    w1_q, w1_scale = per_1x32_f4_quant(w1, quant_dtype=dtypes.fp4x2)
    w2_q, w2_scale = per_1x32_f4_quant(w2, quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(E, inter_dim * 2, model_dim // 2)
    w2_q = w2_q.view(E, model_dim, inter_dim // 2)

    ref_stage1 = torch_moe_stage1(
        a_q,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Swiglu,
        quant_type=Q_TYPE,
        a1_scale=a_scale,
        w1_scale=w1_scale,
    )
    ref_stage2 = torch_moe_stage2(
        ref_stage1,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=Q_TYPE,
        w2_scale=w2_scale,
        a2_scale=None,
        doweight=True,
    )

    a2_q, a2_scale = per_1x32_f8_scale_f8_quant(
        ref_stage1, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    a2_q = a2_q.view(token, topk, inter_dim)

    a_scale_sort = mxfp4_moe_sort_fwd(
        a_scale,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token,
        cols=model_dim,
    )
    w1_shuf = shuffle_weight_a16w4(w1_q, 16, True)
    w1_scale_shuf = shuffle_scale_a16w4(w1_scale, E, True)
    w2_shuf = shuffle_weight_a16w4(w2_q, 16, False)
    w2_scale_shuf = shuffle_scale_a16w4(w2_scale, E, False)
    a2_scale_sort = mxfp4_moe_sort_fwd(
        a2_scale,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token,
        cols=inter_dim,
    )

    return dict(
        inter_pad=inter_pad,
        topk=topk,
        a_q=a_q,
        a_scale_sort=a_scale_sort,
        w1_shuf=w1_shuf,
        w1_scale_shuf=w1_scale_shuf,
        w2_shuf=w2_shuf,
        w2_scale_shuf=w2_scale_shuf,
        a2_q=a2_q,
        a2_scale_sort=a2_scale_sort,
        sorted_ids=sorted_ids,
        sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        ref_stage1=ref_stage1,
        ref_stage2=ref_stage2,
        token=token,
        inter_dim=inter_dim,
        model_dim=model_dim,
    )


@pytest.fixture(autouse=True)
def _a8w4_env():
    old_bound = os.environ.get("AITER_BF16_FP8_MOE_BOUND")
    old_aot = os.environ.get("FLYDSL_RUNTIME_RUN_ONLY")
    os.environ["AITER_BF16_FP8_MOE_BOUND"] = "0"
    os.environ.pop("FLYDSL_RUNTIME_RUN_ONLY", None)
    yield
    if old_bound is None:
        os.environ.pop("AITER_BF16_FP8_MOE_BOUND", None)
    else:
        os.environ["AITER_BF16_FP8_MOE_BOUND"] = old_bound
    if old_aot is None:
        os.environ.pop("FLYDSL_RUNTIME_RUN_ONLY", None)
    else:
        os.environ["FLYDSL_RUNTIME_RUN_ONLY"] = old_aot


def test_pick_flydsl_stage2_tile_k():
    assert pick_flydsl_stage2_tile_k(256) == 256
    assert pick_flydsl_stage2_tile_k(512) == 256
    assert pick_flydsl_stage2_tile_k(640) == 128
    assert pick_flydsl_stage2_tile_k(384) == 128
    assert pick_flydsl_stage2_tile_k(896) == 128
    assert pick_flydsl_stage2_tile_k(1024) == 256
    assert resolve_flydsl_stage2_tile_k(640, 256) == 128
    assert resolve_flydsl_stage2_tile_k(256, 256) == 256
    assert resolve_flydsl_stage2_tile_k(512, 128) == 128


@pytest.mark.parametrize(
    "inter_dim,seed",
    [
        pytest.param(256, 101, id="i256"),
        pytest.param(384, 102, id="i384"),
        pytest.param(640, 0, id="i640_dsv4"),
    ],
)
@_SKIP_GFX950_FLYDSL
def test_flydsl_stage2_a8w4_gui(inter_dim, seed):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2

    token, model_dim, E, topk, block_m = 16, 512, 8, 2, 32
    data = _generate_a8w4_gui_data(
        token, model_dim, inter_dim, E, topk, block_m, seed=seed
    )
    out = flydsl_moe_stage2(
        inter_states=data["a2_q"],
        w2=data["w2_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=32,
        tile_n=256,
        tile_k=256,
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="bf16",
        mode="atomic",
        w2_scale=data["w2_scale_shuf"],
        a2_scale=data["a2_scale_sort"],
        sorted_weights=data["sorted_weights"],
        inter_dim_pad=data["inter_pad"],
        model_dim_pad=0,
    )
    torch.cuda.synchronize()
    _check_close(data["ref_stage2"], out, f"stage2_a8w4_gui_i{inter_dim}")


@pytest.mark.parametrize("inter_dim", [256, 384, 640])
@_SKIP_GFX950_FLYDSL
def test_flydsl_e2e_a8w4_gui(inter_dim):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2

    token, model_dim, E, topk, block_m, seed = 16, 512, 8, 2, 32, 0
    data = _generate_a8w4_gui_data(
        token, model_dim, inter_dim, E, topk, block_m, seed=seed
    )
    stage1_out = flydsl_moe_stage1(
        a=data["a_q"],
        w1=data["w1_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=32,
        tile_n=256,
        tile_k=_stage1_tile_k(model_dim),
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="bf16",
        act="swiglu",
        gate_mode="interleave",
        w1_scale=data["w1_scale_shuf"],
        a1_scale=data["a_scale_sort"],
        inter_dim_pad=data["inter_pad"],
        model_dim_pad=0,
    )
    a2_q, a2_scale = per_1x32_f8_scale_f8_quant(
        stage1_out, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    a2_q = a2_q.view(token, topk, inter_dim)
    a2_scale_sort = mxfp4_moe_sort_fwd(
        a2_scale,
        sorted_ids=data["sorted_ids"],
        num_valid_ids=data["num_valid_ids"],
        token_num=token,
        cols=inter_dim,
    )
    out = flydsl_moe_stage2(
        inter_states=a2_q,
        w2=data["w2_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=32,
        tile_n=256,
        tile_k=256,
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="bf16",
        mode="atomic",
        w2_scale=data["w2_scale_shuf"],
        a2_scale=a2_scale_sort,
        sorted_weights=data["sorted_weights"],
        inter_dim_pad=data["inter_pad"],
        model_dim_pad=0,
    )
    torch.cuda.synchronize()
    _check_close(data["ref_stage2"], out, f"e2e_a8w4_gui_i{inter_dim}")


# ---------------------------------------------------------------------------
# SiTUv2 activation fused into the FlyDSL MXFP4 MoE stage1 (a8w4 + host ref).
# Migrated from the former test_flydsl_moe_situv2.py.
#
# SiTUv2 (fp32 intermediate, cast back at the end):
#     situ_g    = beta * tanh(gate / beta) * sigmoid(gate)
#     up_scaled = linear_beta * tanh(up / linear_beta)
#     out       = situ_g * up_scaled
# ---------------------------------------------------------------------------
SITUV2_BETA = 2.0
SITUV2_LINEAR_BETA = 1.5


def test_situv2_reference():
    """Verify aiter.fused_moe.situv2 matches the closed-form SiTUv2 in fp32.

    Host-only (no GPU / gfx950 required)."""
    from aiter.fused_moe import situv2

    torch.manual_seed(0)
    d = 512
    passed = True
    for beta in (0.5, 1.0, 2.0):
        for linear_beta in (0.5, 1.0, 2.0):
            gate = torch.randn(4, d) * 3.0
            up = torch.randn(4, d) * 3.0
            got = situv2(gate, up, beta=beta, linear_beta=linear_beta)
            g = gate.float()
            u = up.float()
            situ_g = beta * torch.tanh(g / beta) * torch.sigmoid(g)
            up_scaled = linear_beta * torch.tanh(u / linear_beta)
            expect = situ_g * up_scaled
            max_delta = (got.float() - expect).abs().max().item()
            ok = max_delta < 1e-5
            passed = passed and ok
    # Bounded intermediates property (mxfp4-friendly): |out| <= beta*linear_beta.
    beta, linear_beta = 1.5, 0.8
    gate = torch.randn(8, d) * 20.0
    up = torch.randn(8, d) * 20.0
    out = situv2(gate, up, beta=beta, linear_beta=linear_beta)
    bound = beta * linear_beta + 1e-4
    within = bool(out.abs().max().item() <= bound)
    assert passed and within, "situv2 reference mismatch or bound violated"


# (token, model_dim, inter_dim, E, topk, block_m, tile_m, tile_n, tile_k,
#  gate_mode, out_dtype, seed, situ_beta, situ_linear_beta)
A8W4_SITUV2_VEC4_CASES = [
    pytest.param(
        16,
        256,
        128,
        8,
        2,
        32,
        32,
        256,
        256,
        "separated",
        "bf16",
        1,
        SITUV2_BETA,
        SITUV2_LINEAR_BETA,
        id="t16_sep_bf16_default_beta",
    ),
    pytest.param(
        64,
        512,
        256,
        16,
        4,
        32,
        32,
        256,
        256,
        "separated",
        "bf16",
        2,
        SITUV2_BETA,
        SITUV2_LINEAR_BETA,
        id="t64_sep_bf16",
    ),
    pytest.param(
        16,
        256,
        128,
        8,
        2,
        64,
        64,
        128,
        256,
        "separated",
        "bf16",
        3,
        SITUV2_BETA,
        SITUV2_LINEAR_BETA,
        id="tile64_n128_sep_bf16",
    ),
    pytest.param(
        32,
        256,
        128,
        8,
        2,
        32,
        32,
        128,
        256,
        "separated",
        "f16",
        4,
        SITUV2_BETA,
        SITUV2_LINEAR_BETA,
        id="t32_sep_f16",
    ),
    pytest.param(
        16,
        256,
        128,
        8,
        2,
        32,
        32,
        256,
        256,
        "separated",
        "bf16",
        5,
        1.0,
        1.0,
        id="t16_sep_bf16_unit_beta",
    ),
    pytest.param(
        16,
        256,
        128,
        8,
        2,
        32,
        32,
        256,
        256,
        "interleave",
        "bf16",
        6,
        SITUV2_BETA,
        SITUV2_LINEAR_BETA,
        id="t16_interleave_bf16",
    ),
    pytest.param(
        # non-256-aligned inter_dim (DSV4 TP8); exercises fix-k K-tiling.
        64,
        512,
        640,
        16,
        4,
        32,
        32,
        256,
        256,
        "interleave",
        "bf16",
        7,
        SITUV2_BETA,
        SITUV2_LINEAR_BETA,
        id="t64_i640_interleave_bf16",
    ),
]


def _make_routes(hidden: torch.Tensor, experts: int, topk: int, block_m: int):
    score = torch.randn(
        (hidden.shape[0], experts), dtype=hidden.dtype, device=hidden.device
    )
    topk_weights, topk_ids = fused_topk(hidden, score, topk, True)
    sorted_ids, _, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, experts, hidden.shape[1], hidden.dtype, block_m
    )
    return topk_weights, topk_ids, sorted_ids, sorted_expert_ids, num_valid_ids


def _generate_a8w4_situv2_vec4_data(
    token: int,
    model_dim: int,
    inter_dim: int,
    E: int,
    topk: int,
    block_m: int,
    *,
    seed: int = 1,
    dtype=torch.bfloat16,
    situ_beta: float = SITUV2_BETA,
    situ_linear_beta: float = SITUV2_LINEAR_BETA,
    gate_mode: str = "separated",
):
    """a8w4 data for vec4 SiTUv2 epilogue (tile / gate_mode variants)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    inp = torch.randn((token, model_dim), dtype=dtype, device="cuda") / 4
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype, device="cuda") / 4
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype, device="cuda") / 4
    topk_weights, topk_ids, sorted_ids, sorted_expert_ids, num_valid_ids = _make_routes(
        inp, E, topk, block_m
    )

    a_q, a_scale = per_1x32_f8_scale_f8_quant(
        inp, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    w1_q, w1_scale = per_1x32_f4_quant(w1, quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(E, inter_dim * 2, model_dim // 2)
    w2_q, _w2_scale = per_1x32_f4_quant(w2, quant_dtype=dtypes.fp4x2)
    w2_q = w2_q.view(E, model_dim, inter_dim // 2)

    ref_stage1 = torch_moe_stage1(
        a_q,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Situv2,
        quant_type=Q_TYPE,
        a1_scale=a_scale,
        w1_scale=w1_scale,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    a_scale_sort = mxfp4_moe_sort_fwd(
        a_scale,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token,
        cols=model_dim,
    )

    w1_q_shuf = shuffle_weight(w1_q, (16, 16))
    if gate_mode == "interleave":
        w1_q_shuf = shuffle_weight(w1_q, (16, 16), is_guinterleave=True, gate_up=True)

    return dict(
        ref_stage1=ref_stage1,
        a_q=a_q,
        a_scale_sort=a_scale_sort,
        w1_q_shuf=w1_q_shuf,
        w1_scale_shuf=e8m0_shuffle(w1_scale),
        sorted_ids=sorted_ids,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        topk=topk,
    )


@pytest.mark.parametrize(
    "token,model_dim,inter_dim,E,topk,block_m,tile_m,tile_n,tile_k,"
    "gate_mode,out_dtype,seed,situ_beta,situ_linear_beta",
    A8W4_SITUV2_VEC4_CASES,
)
@_SKIP_GFX950_FLYDSL
def test_flydsl_situv2_a8w4_stage1_vec4(
    token,
    model_dim,
    inter_dim,
    E,
    topk,
    block_m,
    tile_m,
    tile_n,
    tile_k,
    gate_mode,
    out_dtype,
    seed,
    situ_beta,
    situ_linear_beta,
):
    """a8w4 SiTUv2 stage1 via mixed_moe_gemm_2stage vec4 activation path."""
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1

    torch.set_default_device("cuda")
    label = (
        f"a8w4_situv2_vec4 token={token} tile={tile_m}x{tile_n}x{tile_k} "
        f"gate={gate_mode} out={out_dtype} beta=({situ_beta},{situ_linear_beta})"
    )
    data = _generate_a8w4_situv2_vec4_data(
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=E,
        topk=topk,
        block_m=block_m,
        seed=seed,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        gate_mode=gate_mode,
    )
    out = flydsl_moe_stage1(
        a=data["a_q"],
        w1=data["w1_q_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=data["topk"],
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype=out_dtype,
        act="situv2",
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        w1_scale=data["w1_scale_shuf"],
        a1_scale=data["a_scale_sort"],
        gate_mode=gate_mode,
    )
    torch.cuda.synchronize()
    # ref is bf16 while out may be f16 (t32_sep_f16 case); compare in fp32.
    _check_close(data["ref_stage1"].float(), out.float(), label)


# ---------------------------------------------------------------------------
# Regression: a8w4 SiTUv2 end-to-end (stage1 -> stage2) numeric check, both
# gate_modes, over 128-multiple inter_dim.
#
# Guards the non-256 inter_dim bug where stage1 tile_n was not downgraded for
# a8w4 (fp8 x mxfp4) -> OOB on the N (gate/up) axis -> ~30% wrong final output
# or GPU memfault at inter=384/640 (separated). The previous a8w4 suite only
# exercised inter=640 in INTERLEAVE mode, which masked this. Here we compare the
# FULL E2E output against the torch reference (not just NaN) across aligned
# (256/512) and non-256 (128/384/640) inter_dim, in BOTH separated and
# interleave gate_modes, on the no-pad path (kernel resolves tile_n internally).
# ---------------------------------------------------------------------------
def _generate_a8w4_situv2_e2e_data(
    token,
    model_dim,
    inter_dim,
    E,
    topk,
    block_m,
    *,
    seed=11,
    gate_mode="separated",
    situ_beta=SITUV2_BETA,
    situ_linear_beta=SITUV2_LINEAR_BETA,
    dtype=torch.bfloat16,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    inp = torch.randn((token, model_dim), dtype=dtype, device="cuda") / 4
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype, device="cuda") / 4
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype, device="cuda") / 4

    topk_weights, topk_ids, sorted_ids, sorted_expert_ids, num_valid_ids = _make_routes(
        inp, E, topk, block_m
    )
    sorted_weights = moe_sorting(topk_ids, topk_weights, E, model_dim, dtype, block_m)[
        1
    ]

    a_q, a_scale = per_1x32_f8_scale_f8_quant(
        inp, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    w1_q, w1_scale = per_1x32_f4_quant(w1, quant_dtype=dtypes.fp4x2)
    w2_q, w2_scale = per_1x32_f4_quant(w2, quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(E, inter_dim * 2, model_dim // 2)
    w2_q = w2_q.view(E, model_dim, inter_dim // 2)

    ref1 = torch_moe_stage1(
        a_q,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Situv2,
        quant_type=Q_TYPE,
        a1_scale=a_scale,
        w1_scale=w1_scale,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    ref2 = torch_moe_stage2(
        ref1,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=Q_TYPE,
        w2_scale=w2_scale,
        a2_scale=None,
        doweight=True,
    )

    a_scale_sort = mxfp4_moe_sort_fwd(
        a_scale,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token,
        cols=model_dim,
    )
    # Interleave uses the a16w4-style GUI shuffle (shuffle_weight_a16w4 gate_up
    # + shuffle_scale_a16w4). The shuffle_weight(is_guinterleave=...) variant is
    # only approximately correct at stage1 and corrupts stage2 (~25% E2E error).
    if gate_mode == "interleave":
        w1_shuf = shuffle_weight_a16w4(w1_q, 16, True)
        w1_scale_shuf = shuffle_scale_a16w4(w1_scale, E, True)
    else:
        w1_shuf = shuffle_weight(w1_q, (16, 16))
        w1_scale_shuf = e8m0_shuffle(w1_scale)
    return dict(
        token=token,
        inter_dim=inter_dim,
        topk=topk,
        a_q=a_q,
        a_scale_sort=a_scale_sort,
        w1_shuf=w1_shuf,
        w1_scale_shuf=w1_scale_shuf,
        w2_shuf=shuffle_weight_a16w4(w2_q, 16, False),
        w2_scale_shuf=shuffle_scale_a16w4(w2_scale, E, False),
        sorted_ids=sorted_ids,
        sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        ref_stage1=ref1,
        ref_stage2=ref2,
    )


# Both gate_modes. Separated is the production/customer SiTUv2 path (fused_moe
# routes SiTUv2 -> separated). Interleave uses the a16w4-style GUI weight shuffle
# (see generator); the earlier shuffle_weight(is_guinterleave) recipe corrupted
# stage2, which is a test-recipe issue, not a kernel bug.
@pytest.mark.parametrize("gate_mode", ["separated", "interleave"])
@pytest.mark.parametrize(
    "inter_dim,seed",
    [
        pytest.param(128, 11, id="i128_non256"),
        pytest.param(256, 12, id="i256_aligned"),
        pytest.param(384, 13, id="i384_non256"),
        pytest.param(512, 14, id="i512_aligned"),
        pytest.param(640, 15, id="i640_non256"),
    ],
)
@_SKIP_GFX950_FLYDSL
def test_flydsl_e2e_a8w4_situv2(inter_dim, seed, gate_mode):
    """a8w4 SiTUv2 E2E (stage1->stage2), numeric vs torch ref.

    Caller passes tile_n=256; the kernel must internally downgrade for non-256
    inter_dim. Regression guard for the a8w4 separated non-256 OOB bug.
    """
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2

    torch.set_default_device("cuda")
    token, model_dim, E, topk, block_m = 1024, 512, 8, 2, 32
    d = _generate_a8w4_situv2_e2e_data(
        token, model_dim, inter_dim, E, topk, block_m, seed=seed, gate_mode=gate_mode
    )
    topk = d["topk"]

    s1 = flydsl_moe_stage1(
        a=d["a_q"],
        w1=d["w1_shuf"],
        sorted_token_ids=d["sorted_ids"],
        sorted_expert_ids=d["sorted_expert_ids"],
        num_valid_ids=d["num_valid_ids"],
        topk=topk,
        tile_m=32,
        tile_n=256,
        tile_k=256,
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="bf16",
        act="situv2",
        situ_beta=SITUV2_BETA,
        situ_linear_beta=SITUV2_LINEAR_BETA,
        w1_scale=d["w1_scale_shuf"],
        a1_scale=d["a_scale_sort"],
        gate_mode=gate_mode,
    )
    torch.cuda.synchronize()
    _check_close(
        d["ref_stage1"].float(), s1.float(), f"situv2_{gate_mode}_stage1_i{inter_dim}"
    )

    a2_q, a2_scale = per_1x32_f8_scale_f8_quant(
        s1, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    a2_q = a2_q.view(token, topk, inter_dim)
    a2_scale_sort = mxfp4_moe_sort_fwd(
        a2_scale,
        sorted_ids=d["sorted_ids"],
        num_valid_ids=d["num_valid_ids"],
        token_num=token,
        cols=inter_dim,
    )
    out = flydsl_moe_stage2(
        inter_states=a2_q,
        w2=d["w2_shuf"],
        sorted_token_ids=d["sorted_ids"],
        sorted_expert_ids=d["sorted_expert_ids"],
        num_valid_ids=d["num_valid_ids"],
        topk=topk,
        tile_m=32,
        tile_n=256,
        tile_k=256,
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="bf16",
        mode="atomic",
        w2_scale=d["w2_scale_shuf"],
        a2_scale=a2_scale_sort,
        sorted_weights=d["sorted_weights"],
        inter_dim_pad=0,
        model_dim_pad=0,
    )
    torch.cuda.synchronize()
    _check_close(
        d["ref_stage2"].float(), out.float(), f"situv2_{gate_mode}_e2e_i{inter_dim}"
    )
