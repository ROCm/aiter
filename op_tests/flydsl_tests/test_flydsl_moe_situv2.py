# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the SiTUv2 activation fused into the FlyDSL MXFP4 MoE stage1.

Covered variants (both use the shared fp32 activation epilogue in
``mixed_moe_gemm_2stage.py``):
  - a4w4: fp4 activation x fp4 weight (per_1x32 e8m0 microscale)
  - a8w4: fp8 activation x fp4 weight (per_1x32 e8m0 microscale)

SiTUv2 (fp32 intermediate, cast back at the end):
    situ_g    = beta * tanh(gate / beta) * sigmoid(gate)
    up_scaled = linear_beta * tanh(up / linear_beta)
    out       = situ_g * up_scaled

Usage:
    # Host-only reference check (no GPU required):
    python op_tests/flydsl_tests/test_flydsl_moe_situv2.py --ref-only

    # Full stage1 correctness on gfx950 (GPUs restricted to 6,7 on the box):
    HIP_VISIBLE_DEVICES=6,7 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
        python op_tests/flydsl_tests/test_flydsl_moe_situv2.py \
        -t 16 64 256 4096 --inter-dim 256 384 1536

    # a8w4 vec4 epilogue tile/gate_mode sweep (pytest, gfx950):
    pytest op_tests/flydsl_tests/test_flydsl_moe_situv2.py -k vec4 -q
"""

import argparse
import sys

import pytest
import torch

from aiter import dtypes, QuantType, ActivationType
from aiter.fused_moe import fused_topk, moe_sorting, torch_moe_stage1
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.quant import (
    mxfp4_moe_sort_fwd,
    per_1x32_f4_quant,
    per_1x32_f8_scale_f8_quant,
)
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.fp4_utils import e8m0_shuffle

Q_TYPE = QuantType.per_1x32
SITUV2_BETA = 2.0
SITUV2_LINEAR_BETA = 1.5

_SKIP_GFX950_FLYDSL = pytest.mark.skipif(
    get_gfx() not in ("gfx950",) or not is_flydsl_available(),
    reason="gfx950 FlyDSL required",
)

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
]


# ---------------------------------------------------------------------------
# Host-only reference test (no GPU): validate the torch SiTUv2 helper.
# ---------------------------------------------------------------------------


def test_situv2_reference():
    """Verify aiter.fused_moe.situv2 matches the closed-form SiTUv2 in fp32."""
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
            print(
                f"  beta={beta}, linear_beta={linear_beta}: "
                f"max_delta={max_delta:.2e} -> {'PASS' if ok else 'FAIL'}"
            )
    # Bounded intermediates property (mxfp4-friendly): |out| <= beta*linear_beta.
    beta, linear_beta = 1.5, 0.8
    gate = torch.randn(8, d) * 20.0
    up = torch.randn(8, d) * 20.0
    out = situv2(gate, up, beta=beta, linear_beta=linear_beta)
    bound = beta * linear_beta + 1e-4
    within = bool(out.abs().max().item() <= bound)
    print(f"  |out| bound <= {bound:.4f}: {'PASS' if within else 'FAIL'}")
    return passed and within


# ---------------------------------------------------------------------------
# GPU stage1 correctness (gfx950): flydsl_moe_stage1(SiTUv2) vs torch ref.
# ---------------------------------------------------------------------------


def _gen_stage1_data(
    token,
    model_dim,
    inter_dim,
    E,
    topk,
    block_m,
    a_dtype_str,
    beta,
    linear_beta,
    dtype=torch.bfloat16,
):
    import aiter
    from aiter import dtypes, QuantType, ActivationType
    from aiter.fused_moe import fused_topk, moe_sorting, torch_moe_stage1
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle, moe_mxfp4_sort

    q_type = QuantType.per_1x32
    q_dtype_a = dtypes.fp4x2 if a_dtype_str == "fp4" else dtypes.fp8
    q_dtype_w = dtypes.fp4x2
    torch_quant = aiter.get_torch_quant(q_type)

    torch.manual_seed(0)
    inp = torch.randn((token, model_dim), dtype=dtype) / 10
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype) / 10
    # w2 is only needed so torch_moe_stage1 can infer (model_dim, inter_dim)
    # from w2.shape via get_inter_dim; must be the real (E, model_dim, inter_dim).
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype) / 10
    score = torch.randn((token, E), dtype=dtype)
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)

    w1_qt, w1_scale = torch_quant(w1, quant_dtype=q_dtype_w)
    w1_qt = w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2)
    w2_qt, _w2_scale = torch_quant(w2, quant_dtype=q_dtype_w)
    w2_qt = w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2)

    if a_dtype_str == "fp8":
        # a8w4: fp8 activation (mxfp8, per-1x32 e8m0 microscale).  The fp4
        # torch_quant only handles fp4x2, so use the mxfp8 quantizer for the
        # kernel input.  torch_moe_stage1 unpacks the activation as mxfp4
        # whenever a1_scale is given, so the reference is instead fed the
        # dequantized fp8 activation (exactly what the kernel sees) with
        # a1_scale=None (the bf16-activation reference branch).
        from aiter.ops.quant import per_1x32_f8_scale_f8_quant
        from aiter.utility import fp4_utils

        a1_qt, a1_scale = per_1x32_f8_scale_f8_quant(
            inp, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )
        a1_scale_f32 = fp4_utils.e8m0_to_f32(a1_scale).view(token, model_dim // 32, 1)
        ref_hidden = (
            (a1_qt.float().view(token, model_dim // 32, 32) * a1_scale_f32)
            .view(token, model_dim)
            .to(dtype)
        )
        ref_a1_scale = None
    else:
        a1_qt, a1_scale = torch_quant(inp, quant_dtype=q_dtype_a)
        ref_hidden = a1_qt
        ref_a1_scale = a1_scale

    ref1 = torch_moe_stage1(
        ref_hidden,
        w1_qt,
        w2_qt,  # used only for (model_dim, inter_dim) inference
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Situv2,
        quant_type=q_type,
        a1_scale=ref_a1_scale,
        w1_scale=w1_scale,
        situ_beta=beta,
        situ_linear_beta=linear_beta,
    )

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_m
    )
    w1_qt_shuf = shuffle_weight(w1_qt, (16, 16))
    w1_scale_shuf = e8m0_shuffle(w1_scale)
    a1_scale_sort = moe_mxfp4_sort(
        a1_scale[:token, :].view(token, 1, -1),
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token,
        block_size=block_m,
    )
    return dict(
        ref_stage1=ref1,
        a1_qt=a1_qt,
        a1_scale_sort=a1_scale_sort,
        w1_qt_shuf=w1_qt_shuf,
        w1_scale_shuf=w1_scale_shuf,
        sorted_ids=sorted_ids,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        dtype=dtype,
    )


def _check(ref_out, test_out, atol=1.0, rtol=0.05, pass_pct=90.0):
    max_delta = (ref_out.float() - test_out.float()).abs().max().item()
    close = torch.isclose(ref_out.float(), test_out.float(), atol=atol, rtol=rtol)
    pct = close.float().mean().item() * 100
    ok = pct > pass_pct
    print(
        f"  max_delta={max_delta:.4f}, {pct:.1f}% close -> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def test_flydsl_stage1_situv2(
    token,
    model_dim,
    inter_dim,
    E,
    topk,
    block_m,
    a_dtype_str,
    beta=1.0,
    linear_beta=1.0,
):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1

    print(
        f"\n[TEST] SiTUv2 stage1 {a_dtype_str}w4: token={token}, "
        f"dim=({model_dim},{inter_dim}), E={E}, topk={topk}, "
        f"beta={beta}, linear_beta={linear_beta}"
    )

    data = _gen_stage1_data(
        token, model_dim, inter_dim, E, topk, block_m, a_dtype_str, beta, linear_beta
    )
    out_dtype_str = "bf16" if data["dtype"] == torch.bfloat16 else "f16"

    out = flydsl_moe_stage1(
        a=data["a1_qt"],
        w1=data["w1_qt_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=block_m,
        tile_n=256,
        tile_k=256,
        a_dtype=a_dtype_str,
        b_dtype="fp4",
        out_dtype=out_dtype_str,
        act="situv2",
        situ_beta=beta,
        situ_linear_beta=linear_beta,
        w1_scale=data["w1_scale_shuf"],
        a1_scale=data["a1_scale_sort"],
    )
    torch.cuda.synchronize()
    return _check(data["ref_stage1"], out)


def _on_gfx950():
    try:
        return get_gfx() == "gfx950"
    except Exception:
        return False


def _check_vec4_result(ref_out, test_out, label, atol=1.0, rtol=0.05, pass_pct=95.0):
    max_delta = (ref_out.float() - test_out.float()).abs().max().item()
    close_mask = torch.isclose(ref_out.float(), test_out.float(), atol=atol, rtol=rtol)
    pct_close = close_mask.float().mean().item() * 100
    passed = pct_close > pass_pct
    print(
        f"  [{label}] max_delta={max_delta:.4f}, {pct_close:.1f}% close "
        f"(atol={atol}, rtol={rtol}) -> {'PASS' if passed else 'FAIL'}"
    )
    return passed


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


def _run_a8w4_situv2_stage1_vec4(
    *,
    token: int,
    model_dim: int,
    inter_dim: int,
    E: int,
    topk: int,
    block_m: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    gate_mode: str,
    out_dtype: str,
    seed: int,
    situ_beta: float,
    situ_linear_beta: float,
    atol: float = 1.0,
    rtol: float = 0.05,
):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1

    label = (
        f"a8w4_situv2_vec4 token={token} tile={tile_m}x{tile_n}x{tile_k} "
        f"gate={gate_mode} out={out_dtype} beta=({situ_beta},{situ_linear_beta})"
    )
    print(f"\n{'=' * 70}\n[TEST] {label}\n{'=' * 70}")

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
    passed = _check_vec4_result(data["ref_stage1"], out, label, atol=atol, rtol=rtol)
    assert passed, label


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
    torch.set_default_device("cuda")
    _run_a8w4_situv2_stage1_vec4(
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=E,
        topk=topk,
        block_m=block_m,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        gate_mode=gate_mode,
        out_dtype=out_dtype,
        seed=seed,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref-only", action="store_true", help="host-only reference test")
    p.add_argument("-t", "--tokens", type=int, nargs="*", default=[16, 64, 256])
    p.add_argument("--inter-dim", type=int, nargs="*", default=[256, 384, 1536])
    p.add_argument("--a-dtype", nargs="*", default=["fp4", "fp8"])
    p.add_argument("--model-dim", type=int, default=3072)
    p.add_argument("--experts", type=int, default=256)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--block-m", type=int, default=32)
    args = p.parse_args()

    print("=" * 70)
    print("[SiTUv2] host-only reference test")
    print("=" * 70)
    ref_ok = test_situv2_reference()
    if args.ref_only:
        sys.exit(0 if ref_ok else 1)

    if not _on_gfx950():
        print("\nSKIP: GPU stage1 tests require gfx950 (MI35x). Ran reference only.")
        sys.exit(0 if ref_ok else 1)

    torch.set_default_device("cuda")
    all_ok = ref_ok
    for a_dtype_str in args.a_dtype:
        for inter_dim in args.inter_dim:
            for token in args.tokens:
                for beta, linear_beta in ((1.0, 1.0), (0.5, 2.0)):
                    ok = test_flydsl_stage1_situv2(
                        token,
                        args.model_dim,
                        inter_dim,
                        args.experts,
                        args.topk,
                        args.block_m,
                        a_dtype_str,
                        beta,
                        linear_beta,
                    )
                    all_ok = all_ok and ok
    print(f"\n{'ALL PASS' if all_ok else 'SOME FAILED'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
