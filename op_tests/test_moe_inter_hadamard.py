# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Block Hadamard of the SwiGLU output between the two MoE GEMMs.

An online-rotated MXFP4 checkpoint stores both expert GEMMs pre-rotated along
K. ``inter_hadamard`` rotates the stage-1 activation before stage 2 quantizes
or consumes it, so ``(y H)(w H)^T = y w^T``.

Run (the checks, then a latency table of the fused MoE with and without the
rotation):
    python3 op_tests/test_moe_inter_hadamard.py
    pytest op_tests/test_moe_inter_hadamard.py -q
"""

import argparse
import inspect

import pytest
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.ops.hadamard import block_hadamard_rotate, hadamard_matrix


def test_rotation_preserves_gemm():
    y = torch.randn(4, 256, dtype=torch.float32)
    w = torch.randn(64, 256, dtype=torch.float32)
    y_rot = block_hadamard_rotate(y, 128)
    w_rot = block_hadamard_rotate(w, 128)
    torch.testing.assert_close(y_rot @ w_rot.T, y @ w.T, atol=1e-4, rtol=1e-4)


def test_hadamard_is_orthonormal():
    h = hadamard_matrix(128, torch.device("cpu"))
    eye = torch.eye(128)
    torch.testing.assert_close(h.T @ h, eye, atol=1e-5, rtol=1e-5)


def test_rejects_non_power_of_two_and_indivisible_width():
    with pytest.raises(ValueError, match="power of two"):
        hadamard_matrix(96, torch.device("cpu"))
    x = torch.randn(2, 100)
    with pytest.raises(ValueError, match="not divisible"):
        block_hadamard_rotate(x, 128)


def test_inter_hadamard_is_in_every_signature():
    import aiter.fused_moe
    from aiter.moev2 import api

    for fn in (
        aiter.fused_moe.fused_moe,
        aiter.fused_moe.fused_moe_fake,
        aiter.fused_moe.fused_moe_,
        aiter.fused_moe._fused_moe_impl,
        aiter.fused_moe.fused_moe_2stages,
        api.fused_moe,
    ):
        assert "inter_hadamard" in inspect.signature(fn).parameters, fn.__name__


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("dtype", [dtypes.bf16, dtypes.fp16])
def test_low_precision_rotation_matches_fp32(dtype):
    """bf16/fp16 rotate in one GEMM in that dtype. The matrix entries are all
    +-1/sqrt(size), so rounding them only scales the rotation by ~1 - 1e-4."""
    torch.manual_seed(0)
    x = torch.randn(64, 768, device="cuda", dtype=dtype)
    got = block_hadamard_rotate(x, 128)
    h = hadamard_matrix(128, x.device)
    ref = (x.float().unflatten(-1, (-1, 128)) @ h).flatten(-2).to(dtype)
    assert got.dtype == dtype and got.shape == x.shape and got.is_contiguous()
    torch.testing.assert_close(got, ref, rtol=2**-7, atol=1e-3)


def _on_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return get_gfx_runtime() == "gfx950"


def _problem(M, model_dim, inter_dim, E, topk):
    """MXFP4 experts whose w2 was quantized after a block Hadamard along K, the
    layout of an online-rotated Quark checkpoint (w1 is rotated by the caller,
    who also rotates the hidden states, so it is left unrotated here).

    The weights are prepared as vLLM's AITER W4A4 backend does: (16, 16)
    shuffled and marked ``is_shuffled``, with ``e8m0_shuffle`` scales, so the
    tuned kernels a real model selects are the ones under test."""
    from aiter.fused_moe import fused_topk
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    torch.manual_seed(0)
    inp = torch.randn((M, model_dim), dtype=dtypes.bf16, device="cuda")
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtypes.bf16, device="cuda")
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtypes.bf16, device="cuda")
    w1 = w1 * model_dim**-0.5
    w2 = block_hadamard_rotate((w2 * inter_dim**-0.5).float(), 128).to(dtypes.bf16)
    score = torch.randn((M, E), dtype=dtypes.bf16, device="cuda")
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)
    tq = aiter.get_torch_quant(QuantType.per_1x32)
    w1_qt, w1_scale = tq(w1, quant_dtype=dtypes.fp4x2)
    w2_qt, w2_scale = tq(w2, quant_dtype=dtypes.fp4x2)
    w1_qt = w1_qt.view(E, inter_dim * 2, model_dim // 2)
    w2_qt = w2_qt.view(E, model_dim, inter_dim // 2)
    w1_k = shuffle_weight(w1_qt.view(dtypes.fp4x2), layout=(16, 16))
    w2_k = shuffle_weight(w2_qt.view(dtypes.fp4x2), layout=(16, 16))
    w1_k.is_shuffled = True
    w2_k.is_shuffled = True
    return {
        "dims": (M, model_dim, inter_dim, E, topk),
        "inp": inp,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "w1_qt": w1_qt,
        "w2_qt": w2_qt,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        "w1": w1_k,
        "w2": w2_k,
        "w1_scale_s": e8m0_shuffle(w1_scale.view(E * inter_dim * 2, -1)),
        "w2_scale_s": e8m0_shuffle(w2_scale.view(E * model_dim, -1)),
    }


_SWIGLU_LIMIT = 7.0


def _run(p, inter_hadamard):
    from aiter.fused_moe import fused_moe

    return fused_moe(
        p["inp"],
        p["w1"],
        p["w2"],
        p["topk_weights"],
        p["topk_ids"],
        w1_scale=p["w1_scale_s"],
        w2_scale=p["w2_scale_s"],
        quant_type=QuantType.per_1x32,
        activation=ActivationType.Swiglu,
        swiglu_limit=_SWIGLU_LIMIT,
        inter_hadamard=inter_hadamard,
    )


def _reference(p, rotate):
    """Unquantized-activation reference on the same quantized weights, with
    the SwiGLU output rotated (or not) before stage 2."""
    from aiter.fused_moe import torch_moe_stage1, torch_moe_stage2

    M, model_dim, inter_dim, E, topk = p["dims"]
    o1 = torch_moe_stage1(
        p["inp"],
        p["w1_qt"].view(dtypes.fp4x2),
        p["w2_qt"].view(dtypes.fp4x2),
        p["topk_weights"],
        p["topk_ids"],
        dtype=dtypes.bf16,
        activation=ActivationType.Swiglu,
        quant_type=QuantType.per_1x32,
        a1_scale=None,
        w1_scale=p["w1_scale"].view(E, inter_dim * 2, model_dim // 32),
        doweight=False,
        swiglu_limit=_SWIGLU_LIMIT,
    )
    o1 = o1.view(M, topk, inter_dim)
    if rotate:
        o1 = block_hadamard_rotate(o1.float(), 128).to(dtypes.bf16)
    return torch_moe_stage2(
        o1,
        p["w1_qt"].view(dtypes.fp4x2),
        p["w2_qt"].view(dtypes.fp4x2),
        p["topk_weights"],
        p["topk_ids"],
        dtype=dtypes.bf16,
        quant_type=QuantType.per_1x32,
        w2_scale=p["w2_scale"].view(E, model_dim, inter_dim // 32),
        a2_scale=None,
        doweight=True,
    )


def _rel(got, want):
    return float((got.float() - want.float()).norm() / want.float().norm())


# Below 256 tokens stage 2 consumes the bf16 SwiGLU output, rotated in torch.
# From 256 tokens stage 1 quantizes it in-kernel and rotates it first; for the
# MiniMax shape the tuned configs pick the MXMOE GEMM1 at 256, 1024 and 4096
# tokens and the FlyDSL moe1 kernel at 512.
@pytest.mark.skipif(not _on_gfx950(), reason="MXFP4 fused MoE needs gfx950")
@pytest.mark.parametrize("M", [1, 64, 255, 256, 512, 1024, 4096])
@pytest.mark.parametrize(
    "model_dim,inter_dim,E,topk",
    [
        (1024, 512, 32, 4),
        # MiniMax-M3.1 per GPU at TP4.
        (6144, 768, 128, 4),
    ],
)
def test_fused_moe_rotates_swiglu_output(M, model_dim, inter_dim, E, topk):
    p = _problem(M, model_dim, inter_dim, E, topk)
    got = _run(p, 128)
    assert not got.isnan().any().item()
    rotated = _reference(p, rotate=True)
    skipped = _reference(p, rotate=False)
    err = _rel(got, rotated)
    # ~0.005 with bf16 stage-2 activations, ~0.19 once they are quantized to
    # MXFP4; skipping the rotation is ~1.4, which is what this separates.
    assert err < 0.3, err
    assert err < 0.2 * _rel(got, skipped), (err, _rel(got, skipped))


def test_fused_quant_stage1_names_widen_to_hold_a_block():
    pytest.importorskip("flydsl")
    from aiter.fused_moe import (
        _flydsl_moe1_kname_with_hadamard,
        _mxmoe_g1_kname_with_hadamard,
    )

    assert (
        _mxmoe_g1_kname_with_hadamard(
            "flydsl_mxmoe_g1_a4w4_16x128x256_f16in_hpf_nt_swiglu_xcd2", 128
        )
        == "flydsl_mxmoe_g1_a4w4_16x256x256_f16in_hpf_nt_swiglu_xcd2_had128"
    )
    assert (
        _flydsl_moe1_kname_with_hadamard(
            "flydsl_moe1_afp4_wfp4_bf16_t32x64x256_w3_xcd4_kw2_fp4", 128
        )
        == "flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w3_xcd4_fp4"
    )
    name = "flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w4_fp4"
    assert _flydsl_moe1_kname_with_hadamard(name, 128) == name
    with pytest.raises(NotImplementedError, match="tile_n >= 256"):
        _flydsl_moe1_kname_with_hadamard(
            "flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w3_gui_fp8", 128
        )


def _bench(M, model_dim, inter_dim, E, topk):
    from aiter.test_common import run_perftest

    p = _problem(M, model_dim, inter_dim, E, topk)
    row = {"M": M, "shape": f"{model_dim}x{inter_dim} E{E} top{topk}"}
    for size in (0, 128):
        out, us = run_perftest(_run, p, size, num_iters=50, num_warmup=10)
        row[f"us inter_hadamard={size}"] = round(us, 1)
        if size:
            row["rel err vs rotated ref"] = round(_rel(out, _reference(p, True)), 4)
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--tokens", type=int, nargs="+", default=[1, 64, 256, 512, 1024, 4096]
    )
    parser.add_argument(
        "-s",
        "--shape",
        default="6144,768,128,4",
        help="model_dim,inter_dim,E,topk",
    )
    args = parser.parse_args()
    shape = tuple(int(v) for v in args.shape.split(","))
    rows = [_bench(M, *shape) for M in args.tokens]
    import pandas as pd

    print(pd.DataFrame(rows).to_markdown(index=False))
