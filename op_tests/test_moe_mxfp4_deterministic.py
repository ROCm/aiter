# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""fused_moe(deterministic=True) for MXFP4 MoE.

A token's output must be bitwise the same whatever the batch it is in and from
run to run, including across the token counts where the default changes its
kernels, tiles or activation dtype, and it must be as accurate as the
default. The deterministic path takes gate/up-interleaved weights and MXFP8
activations, as the default does from 256 tokens.

Run (the checks, then a latency sweep of the default and the deterministic
path that ends in a markdown table):
    python3 op_tests/test_moe_mxfp4_deterministic.py
    python3 op_tests/test_moe_mxfp4_deterministic.py -t 1 256 -s 3072,3072,128,4
    pytest op_tests/test_moe_mxfp4_deterministic.py -q
"""

import argparse
import itertools
import sys

import pandas as pd
import pytest
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk, torch_moe_stage1, torch_moe_stage2
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.test_common import benchmark, checkAllclose, run_perftest

SUPPORTED_GFX = ["gfx950"]

_SKIP = pytest.mark.skipif(
    get_gfx_runtime() not in SUPPORTED_GFX,
    reason="fused_moe(deterministic=True) requires gfx950",
)

# Token counts on both sides of where the default switches the activation dtype
# (256) and where the deterministic tile height grows (2,048), plus 1,024.
_PREFIXES = (1, 31, 255, 256, 257, 1024, 1025, 2048, 2049)
_M = 2304
# checkAllclose against the bf16-activation torch reference. The MXFP8 activations
# (e4m3: 3 mantissa bits) put most elements a few percent off, so compare at 10%
# and allow at most 0.1% of the elements outside (measured on MI350X: <= 0.03%).
_RTOL = _ATOL = 0.1
_ERR_MAX = 1e-3


def _cos_diff(x, y):
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    return float(1 - 2 * (x * y).sum() / denom)


def _err(ref, out, msg):
    """checkAllclose mismatch ratio against the torch reference, in fp32."""
    return checkAllclose(
        ref.to(dtypes.fp32),
        out.to(dtypes.fp32),
        rtol=_RTOL,
        atol=_ATOL,
        tol_err_ratio=_ERR_MAX,
        msg=f"{msg}: ",
    )


def _problem(M, model_dim, inter_dim, E, topk, bias=True, interleaved=True):
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
        "w1": shuffle_weight_a16w4(w1_qt, 16, interleaved),
        "w2": shuffle_weight_a16w4(w2_qt, 16, False),
        "w1_scale_s": shuffle_scale_a16w4(w1_scale, E, interleaved),
        "w2_scale_s": shuffle_scale_a16w4(w2_scale, E, False),
        "bias1": None,
        "bias2": None,
        "gate_mode": GateMode.INTERLEAVE if interleaved else GateMode.SEPARATED,
    }
    if bias:
        p["bias1"] = torch.randn((E, inter_dim * 2), dtype=torch.float32, device="cuda")
        p["bias2"] = torch.randn((E, model_dim), dtype=torch.float32, device="cuda")
    return p


def _run(p, rows, activation, deterministic, **kwargs):
    if deterministic:
        # Deterministic mode takes the activation dtype from the caller.
        kwargs.setdefault("quant_dtype_a", dtypes.fp8)
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
        gate_mode=p["gate_mode"].value,
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
    "model_dim,inter_dim,E,topk,bias",
    [
        (1024, 512, 32, 4, True),
        (3072, 3072, 32, 4, True),
        (3072, 3072, 128, 4, True),
        (6144, 768, 129, 6, True),
        (6144, 1536, 129, 5, True),
        (6144, 768, 129, 6, False),
    ],
)
def test_deterministic_swiglu_bias(model_dim, inter_dim, E, topk, bias):
    """SwiGLU with clamp, with per-expert biases (gpt-oss shapes, and MiniMax-M3's
    per GPU at TP4 and TP2: 128 routed experts plus a fused shared expert) and
    without them. The stage-1 bias stays in [gate.., up..] order."""
    limit = 7.0
    p = _problem(_M, model_dim, inter_dim, E, topk, bias=bias)
    out = _run(p, slice(None), ActivationType.Swiglu, True, swiglu_limit=limit)
    assert not out.isnan().any().item(), "deterministic output contains NaN"
    ref = _reference(
        p, _M, model_dim, inter_dim, E, topk, ActivationType.Swiglu, swiglu_limit=limit
    )
    # The default also quantizes the activations to MXFP8 at this token count, so
    # it is the accuracy to match.
    default = _run(p, slice(None), ActivationType.Swiglu, False, swiglu_limit=limit)
    err = _err(ref, out, "deterministic")
    assert err <= _ERR_MAX, f"checkAllclose mismatch vs the bf16 reference {err:.3%}"
    ld = _cos_diff(ref.float(), out.float())
    ld_default = _cos_diff(ref.float(), default.float())
    assert (
        ld <= ld_default * 1.1 + 1e-5
    ), f"cos/logits_diff vs the bf16 reference {ld:.3e}, default {ld_default:.3e}"
    _check_batch_invariant(p, out, ActivationType.Swiglu, swiglu_limit=limit)


@_SKIP
@pytest.mark.parametrize("M", [1, 8, 64, 300])
def test_deterministic_zero_routing_weights(M):
    """Routes with zero weight must not change a token's result: half the tokens
    keep only their first route. Accuracy as with nonzero weights, and a token
    alone matches the same token in the batch. Dropped routes (topk_ids -1)
    give the same result as zero-weight ones, also when the token's kept route
    is expert 0."""
    shape = (3072, 3072, 128, 4)
    p = _problem(M, *shape)
    # Every token's first route goes to expert 0; the shift keeps a token's
    # experts distinct.
    p["topk_ids"] = (p["topk_ids"] - p["topk_ids"][:, :1]) % shape[2]
    p["topk_weights"][::2, 1:] = 0
    kw = {"swiglu_limit": 7.0}
    out = _run(p, slice(None), ActivationType.Swiglu, True, **kw)
    ref = _reference(p, M, *shape, ActivationType.Swiglu, **kw)
    err = _err(ref, out, f"deterministic, zero routing weights, M={M}")
    assert err <= _ERR_MAX, f"checkAllclose mismatch {err:.3%}"
    ld = _cos_diff(ref.float(), out.float())
    assert ld < 2e-2, f"cos/logits_diff with zero routing weights: {ld:.3e}"
    for i in range(min(M, 4)):
        alone = _run(p, slice(i, i + 1), ActivationType.Swiglu, True, **kw)
        assert torch.equal(alone, out[i : i + 1]), f"token {i} alone differs"
    p["topk_ids"][::2, 1:] = -1
    for _ in range(2):
        dropped = _run(p, slice(None), ActivationType.Swiglu, True, **kw)
        assert torch.equal(
            dropped, out
        ), "dropped routes differ from zero-weight routes"


@_SKIP
@pytest.mark.parametrize("use_mask", [False, True])
def test_reduction_skips_dropped_routes(use_mask):
    """The stage-2 reduction with skip_dropped leaves out slots whose topk_ids
    are -1, whose rows are never written (NaN here), also together with the
    expert-parallel mask."""
    from aiter.ops.flydsl.moe_kernels import _run_moe_reduction

    token, topk, model_dim, E = 64, 4, 512, 8
    gen = torch.Generator(device="cuda").manual_seed(0)
    topk_ids = torch.randint(0, E, (token, topk), generator=gen, device="cuda")
    topk_ids[::2, 1:] = -1
    topk_ids[1::4, 0] = -1
    topk_ids = topk_ids.to(dtypes.i32)
    # Small integers, so every sum is exact in bf16.
    target = torch.randint(
        1, 9, (token * topk, model_dim), generator=gen, device="cuda"
    ).to(dtypes.bf16)
    target[topk_ids.view(-1) < 0] = float("nan")
    expert_mask = torch.tensor([1, 0] * (E // 2), dtype=dtypes.i32, device="cuda")
    keep = topk_ids >= 0
    if use_mask:
        keep &= expert_mask[topk_ids.clamp(min=0).long()] != 0
    ref = torch.where(keep[..., None], target.view(token, topk, model_dim).float(), 0)
    out = torch.empty((token, model_dim), dtype=dtypes.bf16, device="cuda")
    _run_moe_reduction(
        target,
        out,
        token,
        topk,
        model_dim,
        expert_mask=expert_mask if use_mask else None,
        topk_ids=topk_ids,
        skip_dropped=True,
    )
    assert torch.equal(out, ref.sum(1).to(dtypes.bf16))


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

    def call(interleaved=True, in_dtype=dtypes.bf16, w2=None, **kw):
        args = {
            "activation": ActivationType.Swiglu,
            "deterministic": True,
            "quant_dtype_a": dtypes.fp8,
            "gate_mode": GateMode.INTERLEAVE.value,
        }
        args.update(kw)
        return fused_moe(
            p["inp"].to(in_dtype),
            shuffle_weight_a16w4(p["w1_qt"], 16, interleaved),
            p["w2"] if w2 is None else w2,
            p["topk_weights"],
            p["topk_ids"],
            w1_scale=shuffle_scale_a16w4(p["w1_scale"], E, interleaved),
            w2_scale=p["w2_scale_s"],
            quant_type=QuantType.per_1x32,
            **args,
        )

    with pytest.raises(ValueError, match="needs quant_dtype_a"):
        call(quant_dtype_a=None)
    with pytest.raises(NotImplementedError, match="SwiGLU only"):
        call(activation=ActivationType.Silu)
    with pytest.raises(NotImplementedError, match="SwiGLU only"):
        call(activation=ActivationType.Situv2)
    with pytest.raises(NotImplementedError, match="activation dtype"):
        call(quant_dtype_a=dtypes.bf16)
    with pytest.raises(NotImplementedError, match="MXFP8 only"):
        call(quant_dtype_a=dtypes.fp4x2)
    with pytest.raises(NotImplementedError, match="output dtype"):
        call(in_dtype=dtypes.fp16)
    with pytest.raises(NotImplementedError, match="gate/up-interleaved only"):
        call(interleaved=False, gate_mode=GateMode.SEPARATED.value)
    with pytest.raises(NotImplementedError, match="gate/up-interleaved only"):
        call(interleaved=False)
    with pytest.raises(NotImplementedError, match="block_size_M"):
        call(block_size_M=128)
    with pytest.raises(NotImplementedError, match="stage-2 activation dtype"):
        call(quant_dtype_a2=dtypes.fp4x2)
    with pytest.raises(NotImplementedError, match="hidden padding"):
        call(hidden_pad=256)
    with pytest.raises(NotImplementedError, match="intermediate padding"):
        call(
            intermediate_pad=256,
            bias2=torch.zeros((E, 1024), dtype=torch.float32, device="cuda"),
        )
    with pytest.raises(NotImplementedError, match="not both preshuffled"):
        call(w2=p["w2_qt"])

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
            gate_mode=GateMode.INTERLEAVE.value,
            deterministic=True,
            quant_dtype_a=dtypes.fp8,
        )

    # Widths the MXFP8 activation quantizer cannot take at small batches.
    for model_dim, inter_dim in ((8704, 512), (1024, 8704)):
        q = _problem(8, model_dim, inter_dim, 4, 2, bias=False)
        with pytest.raises(NotImplementedError, match="at most 8192"):
            _run(q, slice(None), ActivationType.Swiglu, True)


@benchmark()
def test_fused_moe_mxfp4(token, model_dim, inter_dim, E, topk):
    """Latency of the default and the deterministic path (SwiGLU, clamp 7, biases),
    each checked against the torch reference."""
    p = _problem(token, model_dim, inter_dim, E, topk)
    kw = {"swiglu_limit": 7.0}
    ref = _reference(
        p, token, model_dim, inter_dim, E, topk, ActivationType.Swiglu, **kw
    )
    candidates = {"default": False, "deterministic": True}

    # gate/up and down projections; MXFP4 weights (plus 1/32 E8M0 scales) of the
    # experts the tokens use, bf16 activations in and out.
    flops = 2 * token * topk * 3 * model_dim * inter_dim
    used = p["topk_ids"].unique().numel()
    nbytes = used * 3 * model_dim * inter_dim * (1 / 2 + 1 / 32) + 2 * (
        2 * token * model_dim
    )

    ret = {"gfx": get_gfx_runtime()}
    for name, deterministic in candidates.items():
        # One copy of the arguments: rotating copies of every expert's weights
        # would not fit, and the weights exceed the caches anyway.
        out, us = run_perftest(
            _run,
            p,
            slice(None),
            ActivationType.Swiglu,
            deterministic,
            num_rotate_args=1,
            **kw,
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = _err(ref, out, f"{name}: token={token}")
    return ret


test_fused_moe_mxfp4.__test__ = False


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "fused_moe(deterministic=True) unsupported on %s; skipping",
            get_gfx_runtime(),
        )
        return 0

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="fused_moe(deterministic=True) for MXFP4 MoE: checks and latency",
    )
    parser.add_argument(
        "-t",
        "--token",
        type=int,
        nargs="*",
        default=[1, 8, 32, 128, 256, 1024, 4096],
        help="""Tokens.
    e.g.: -t 1 256""",
    )
    parser.add_argument(
        "-s",
        "--shape",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(3072, 3072, 128, 4), (6144, 1536, 129, 5)],
        help="""model_dim, inter_dim, experts, topk (gpt-oss-120b; MiniMax-M3 at TP2).
    e.g.: -s 3072,3072,128,4""",
    )
    args = parser.parse_args()

    aiter.logger.info("fused_moe deterministic: running correctness checks...")
    status = pytest.main([__file__, "-q", "-p", "no:cacheprovider"])
    if status != pytest.ExitCode.OK:
        aiter.logger.error(
            "fused_moe deterministic: checks failed (pytest exit %s); skipping perf sweep",
            int(status),
        )
        return 1

    df = []
    for (model_dim, inter_dim, E, topk), token in itertools.product(
        args.shape, args.token
    ):
        df.append(test_fused_moe_mxfp4(token, model_dim, inter_dim, E, topk))
    df = pd.DataFrame(df)
    aiter.logger.info(
        "fused_moe deterministic summary (markdown):\n%s", df.to_markdown(index=False)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
