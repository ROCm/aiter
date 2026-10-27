# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""fused_moe(deterministic=True) for MXFP4 MoE.

A token's output must be bitwise the same whatever the batch it is in and from
run to run, including across the token counts where the default changes its
kernels, tiles or activation dtype, and it must be as accurate as the
default, which also uses MXFP4 activations from 256 tokens.

Run:
    pytest op_tests/test_moe_mxfp4_deterministic.py -q
"""

import pytest
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk, torch_moe_stage1, torch_moe_stage2
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

_SKIP = pytest.mark.skipif(
    get_gfx() != "gfx950", reason="fused_moe(deterministic=True) requires gfx950"
)

# Token counts on both sides of where the default switches the activation dtype
# (256) and where the deterministic tile height grows (1,024 and 2,048).
_PREFIXES = (1, 31, 255, 256, 257, 1024, 1025, 2048, 2049)
_M = 2304


def _cos_diff(x, y):
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    return float(1 - 2 * (x * y).sum() / denom)


def _problem(M, model_dim, inter_dim, E, topk, bias=True):
    dtype = dtypes.bf16
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    inp = torch.randn((M, model_dim), dtype=dtype, device="cuda")
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype, device="cuda")
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype, device="cuda")
    score = torch.randn((M, E), dtype=dtype, device="cuda")
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)
    tq = aiter.get_torch_quant(QuantType.per_1x32)
    w1_qt, w1_scale = tq(w1 * model_dim**-0.5, quant_dtype=dtypes.fp4x2)
    w2_qt, w2_scale = tq(w2 * inter_dim**-0.5, quant_dtype=dtypes.fp4x2)
    w1_qt = w1_qt.view(E, inter_dim * 2, model_dim // 2)
    w2_qt = w2_qt.view(E, model_dim, inter_dim // 2)
    p = {
        "inp": inp,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "w1_qt": w1_qt,
        "w2_qt": w2_qt,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        "w1": shuffle_weight_a16w4(w1_qt, 16, False),
        "w2": shuffle_weight_a16w4(w2_qt, 16, False),
        "w1_scale_s": shuffle_scale_a16w4(w1_scale, E, False),
        "w2_scale_s": shuffle_scale_a16w4(w2_scale, E, False),
        "bias1": None,
        "bias2": None,
    }
    if bias:
        p["bias1"] = torch.randn((E, inter_dim * 2), dtype=torch.float32, device="cuda")
        p["bias2"] = torch.randn((E, model_dim), dtype=torch.float32, device="cuda")
    return p


def _run(p, rows, activation, deterministic, **kwargs):
    return fused_moe(
        p["inp"][rows],
        p["w1"],
        p["w2"],
        p["topk_weights"][rows],
        p["topk_ids"][rows],
        w1_scale=p["w1_scale_s"],
        w2_scale=p["w2_scale_s"],
        quant_type=QuantType.per_1x32,
        activation=activation,
        gate_mode=GateMode.SEPARATED.value,
        bias1=p["bias1"],
        bias2=p["bias2"],
        deterministic=deterministic,
        **kwargs,
    )


def _reference(p, M, model_dim, inter_dim, E, topk, activation, **act_kwargs):
    o1 = torch_moe_stage1(
        p["inp"],
        p["w1_qt"].view(dtypes.fp4x2),
        p["w2_qt"].view(dtypes.fp4x2),
        p["topk_weights"],
        p["topk_ids"],
        dtype=dtypes.bf16,
        activation=activation,
        quant_type=QuantType.per_1x32,
        a1_scale=None,
        w1_scale=p["w1_scale"].view(E, inter_dim * 2, model_dim // 32),
        w1_bias=p["bias1"],
        doweight=False,
        **act_kwargs,
    )
    return torch_moe_stage2(
        o1.view(M, topk, inter_dim),
        p["w1_qt"].view(dtypes.fp4x2),
        p["w2_qt"].view(dtypes.fp4x2),
        p["topk_weights"],
        p["topk_ids"],
        dtype=dtypes.bf16,
        quant_type=QuantType.per_1x32,
        w2_scale=p["w2_scale"].view(E, model_dim, inter_dim // 32),
        a2_scale=None,
        w2_bias=p["bias2"],
        doweight=True,
    )


def _check_batch_invariant(p, out, activation, **kwargs):
    for _ in range(2):
        assert torch.equal(
            _run(p, slice(None), activation, True, **kwargs), out
        ), "differs from run to run"
    for k in _PREFIXES:
        assert torch.equal(
            _run(p, slice(0, k), activation, True, **kwargs), out[:k]
        ), f"first {k} tokens differ from the same tokens in the full batch"
    assert torch.equal(
        _run(p, slice(1000, 1001), activation, True, **kwargs), out[1000:1001]
    ), "a token alone differs from the same token in the full batch"


@_SKIP
@pytest.mark.parametrize(
    "model_dim,inter_dim,E,topk",
    [
        (1024, 512, 32, 4),
        (3072, 3072, 32, 4),
        (3072, 3072, 128, 4),
        (6144, 768, 129, 6),
        (6144, 1536, 129, 5),
    ],
)
def test_deterministic_swiglu_bias(model_dim, inter_dim, E, topk):
    """SwiGLU with clamp and per-expert biases; gpt-oss shapes, and MiniMax-M3's per
    GPU at TP4 and TP2 (128 routed experts plus a fused shared expert)."""
    limit = 7.0
    p = _problem(_M, model_dim, inter_dim, E, topk)
    out = _run(p, slice(None), ActivationType.Swiglu, True, swiglu_limit=limit)
    assert not out.isnan().any().item(), "deterministic output contains NaN"
    ref = _reference(
        p, _M, model_dim, inter_dim, E, topk, ActivationType.Swiglu, swiglu_limit=limit
    )
    # The default also quantizes the activations to MXFP4 at this token count, so
    # it is the accuracy to match.
    default = _run(p, slice(None), ActivationType.Swiglu, False, swiglu_limit=limit)
    ld = _cos_diff(ref.float(), out.float())
    ld_default = _cos_diff(ref.float(), default.float())
    assert (
        ld <= ld_default * 1.05 + 1e-5
    ), f"cos/logits_diff vs the bf16 reference {ld:.3e}, default {ld_default:.3e}"
    _check_batch_invariant(p, out, ActivationType.Swiglu, swiglu_limit=limit)


@_SKIP
@pytest.mark.parametrize("capacity,local", [(2, 1), (8, 3), (300, 129)])
def test_deterministic_num_local_tokens(capacity, local):
    """With num_local_tokens below the batch capacity, the local tokens match the
    same tokens run alone."""
    p = _problem(capacity, 1024, 512, 32, 4)
    n_local = torch.tensor([local], dtype=dtypes.i32, device="cuda")
    out = _run(p, slice(None), ActivationType.Swiglu, True, num_local_tokens=n_local)
    alone = _run(p, slice(0, local), ActivationType.Swiglu, True)
    assert torch.equal(out[:local], alone), "local tokens differ"


@_SKIP
def test_deterministic_rejects_unsupported():
    """Configurations the deterministic path cannot run raise instead of running a
    path whose results would depend on the batch."""
    E = 32
    p = _problem(8, 1024, 512, E, 4, bias=False)

    def call(interleaved=False, in_dtype=dtypes.bf16, **kw):
        args = {"activation": ActivationType.Swiglu, "deterministic": True}
        args.update(kw)
        return fused_moe(
            p["inp"].to(in_dtype),
            shuffle_weight_a16w4(p["w1_qt"], 16, interleaved),
            p["w2"],
            p["topk_weights"],
            p["topk_ids"],
            w1_scale=shuffle_scale_a16w4(p["w1_scale"], E, interleaved),
            w2_scale=p["w2_scale_s"],
            quant_type=QuantType.per_1x32,
            gate_mode=GateMode.SEPARATED.value,
            **args,
        )

    with pytest.raises(NotImplementedError, match="SwiGLU only"):
        call(activation=ActivationType.Silu)
    with pytest.raises(NotImplementedError, match="SwiGLU only"):
        call(activation=ActivationType.Situv2)
    with pytest.raises(NotImplementedError, match="activation dtype"):
        call(quant_dtype_a=dtypes.bf16)
    with pytest.raises(NotImplementedError, match="output dtype"):
        call(in_dtype=dtypes.fp16)
    with pytest.raises(NotImplementedError, match="interleaved"):
        call(interleaved=True)
    with pytest.raises(NotImplementedError, match="block_size_M"):
        call(block_size_M=128)
    with pytest.raises(NotImplementedError, match="stage-2 activation dtype"):
        call(quant_dtype_a2=dtypes.fp8)
    with pytest.raises(NotImplementedError, match="hidden padding"):
        call(hidden_pad=256)

    q = _problem(8, 384, 512, E, 4, bias=False)
    with pytest.raises(NotImplementedError, match="hidden size 384"):
        fused_moe(
            q["inp"],
            q["w1"],
            q["w2"],
            q["topk_weights"],
            q["topk_ids"],
            w1_scale=q["w1_scale_s"],
            w2_scale=q["w2_scale_s"],
            quant_type=QuantType.per_1x32,
            activation=ActivationType.Swiglu,
            gate_mode=GateMode.SEPARATED.value,
            deterministic=True,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
