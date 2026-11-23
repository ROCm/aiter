# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CK-Tile MXFP4 MoE stage 1 must only split K into whole 256-wide K tiles.

bf16 activations with MXFP4 weights run stage 1 on the CK-Tile split-k kernel
(split_k >= 2 for SwiGLU). The kernel runs whole K tiles per split without
masking, so a split of model_dim that is not a multiple of 256 (model_dim=256,
768, 1280, ...) used to return silently wrong results or fault.

Run this file directly for the numerical/performance sweep and Markdown table;
use pytest for the numerical, selector and expected-error regression checks.
"""

import inspect
import itertools
from functools import partial
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

import aiter
import aiter.fused_moe as fused_moe_module
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    _cktile_gemm1_k_batch,
    cktile_moe_stage1,
    fused_moe,
    fused_topk,
    moe_sorting,
)
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.quant import per_1x32_f4_quant
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility import fp4_utils

# gfx942 also builds the bf16 CK-Tile instances, but like test_moe_2stage.py
# these MXFP4 MoE cases are only validated on gfx950 (a8w4 is gfx950-only).
_SKIP = pytest.mark.skipif(
    get_gfx() != "gfx950", reason="MXFP4 MoE cases are validated on gfx950 only"
)

INTER_DIM = 256
MAX_REL_L2 = 0.02
MAX_ERR_RATIO = 0.05


@pytest.mark.parametrize(
    ("K", "split_k", "want"),
    [
        (256, 2, 1),
        (512, 2, 2),
        (512, 4, 2),
        (768, 2, 1),
        (768, 3, 3),
        (1280, 2, 1),
        (6144, 2, 2),
        (6144, 16, 12),
        (384, 2, 1),
        (3072, 64, 12),
        (512, 1, 1),
        (128, 2, 1),
    ],
)
def test_cktile_gemm1_k_batch(K, split_k, want):
    assert _cktile_gemm1_k_batch(K, split_k) == want


@pytest.mark.parametrize(("split_k", "k_pad_zeros"), [(2, 128), (4, 256), (1, 128)])
def test_cktile_gemm1_k_batch_padded_k(split_k, k_pad_zeros):
    assert _cktile_gemm1_k_batch(3072, split_k, k_pad_zeros) == 1


def test_cktile_stage1_rejects_unaligned_model_dim(monkeypatch):
    monkeypatch.setattr(
        aiter,
        "moe_cktile2stages_gemm1",
        lambda *args, **kwargs: pytest.fail("unsafe CK-Tile kernel was called"),
    )
    hidden_states = SimpleNamespace(shape=(4, 384))
    w1 = SimpleNamespace(dtype=dtypes.fp4x2, shape=(4, 2 * INTER_DIM, 384 // 2))
    w2 = SimpleNamespace(shape=(4, 384, INTER_DIM // 2))
    with pytest.raises(NotImplementedError, match="model_dim.*multiple of 256"):
        cktile_moe_stage1(
            hidden_states, w1, w2, None, None, None, None, 2, 16, None, None
        )


def _quant(w):
    E, N, K = w.shape
    q, s = per_1x32_f4_quant(w.reshape(E * N, K), quant_dtype=dtypes.fp4x2)
    q = q.view(E, N, K // 2)
    s = s.view(torch.uint8).view(E * N, K // 32)
    deq = fp4_utils.mxfp4_to_f32(q) * fp4_utils.e8m0_to_f32(s).repeat_interleave(
        32, -1
    ).view(E, N, K)
    scale = fp4_utils.e8m0_shuffle(s).view(dtypes.fp8_e8m0)
    return shuffle_weight(q, (16, 16)), scale, deq.double()


def _make(model_dim, E, topk, token, hidden_pad=0, bias=False):
    torch.manual_seed(0)
    x = torch.randn(token, model_dim, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(E, 2 * INTER_DIM, model_dim, device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn(E, model_dim, INTER_DIM, device="cuda", dtype=torch.bfloat16)
    real_dim = model_dim - hidden_pad
    x[:, real_dim:] = 0
    w1[:, :, real_dim:] = 0
    w2[:, real_dim:] = 0
    score = torch.randn(token, E, device="cuda", dtype=torch.bfloat16)
    topk_weight, topk_ids = fused_topk(x, score, topk, True)
    w1q, w1_scale, w1_ref = _quant(w1 / model_dim**0.5)
    w2q, w2_scale, w2_ref = _quant(w2 / INTER_DIM**0.5)
    b1 = b2 = None
    if bias:
        b1 = torch.randn(E, 2 * INTER_DIM, device="cuda").clamp(-1.0, 1.0)
        b2 = torch.randn(E, model_dim, device="cuda").clamp(-1.0, 1.0)
        b2[:, real_dim:] = 0
    return (
        x,
        topk_weight,
        topk_ids,
        (w1q, w1_scale, w1_ref),
        (w2q, w2_scale, w2_ref),
        (b1, b2),
    )


def run_torch(x, topk_weight, topk_ids, w1_ref, w2_ref, activation, b1=None, b2=None):
    out = torch.zeros(x.shape, device=x.device, dtype=torch.float64)
    for m in range(x.shape[0]):
        for k in range(topk_ids.shape[1]):
            e = topk_ids[m, k]
            h = w1_ref[e] @ x[m].double()
            if b1 is not None:
                h = h + b1[e].double()
            if activation == ActivationType.Swiglu:
                gate, up = h.split(INTER_DIM)
                gate, up = gate.clamp(max=7.0), up.clamp(-7.0, 7.0)
                a2 = gate * torch.sigmoid(1.702 * gate) * (up + 1)
            else:
                # fp4 A/W Silu post-activation reads gate/up interleaved in
                # 16-row blocks (post_activation_layout="auto").
                gate, up = h.view(INTER_DIM // 16, 2, 16).unbind(1)
                a2 = torch.nn.functional.silu(gate.flatten()) * up.flatten()
            y = w2_ref[e] @ a2.to(torch.bfloat16).double()
            if b2 is not None:
                y = y + b2[e].double()
            out[m] += topk_weight[m, k].double() * y
    return out


@benchmark()
def benchmark_fused_moe(
    model_dim, E, topk, token, activation, hidden_pad=0, bias=False, split_k=0
):
    x, topk_weight, topk_ids, w1, w2, (b1, b2) = _make(
        model_dim, E, topk, token, hidden_pad, bias
    )
    candidates = {
        "cktile": partial(
            fused_moe,
            x,
            w1[0],
            w2[0],
            topk_weight,
            topk_ids,
            activation=activation,
            quant_type=QuantType.per_1x32,
            w1_scale=w1[1],
            w2_scale=w2[1],
            hidden_pad=hidden_pad,
            bias1=b1,
            bias2=b2,
        )
    }
    launched = []
    gemm1 = aiter.moe_cktile2stages_gemm1
    sig = inspect.signature(gemm1)

    def spy(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        launched.append(bound.arguments["split_k"])
        return gemm1(*args, **kwargs)

    fused_moe_module.get_ksplit.cache_clear()
    fused_moe_module.get_2stage_cfgs.cache_clear()
    try:
        with pytest.MonkeyPatch.context() as patch:
            # Keep these cases on CK-Tile even if FlyDSL learns the shapes.
            patch.setenv("AITER_FLYDSL_FORCE", "0")
            patch.setenv("AITER_KSPLIT", str(split_k))
            requested = max(
                fused_moe_module.get_ksplit(token, topk, E, INTER_DIM, model_dim), 2
            )
            want = _cktile_gemm1_k_batch(model_dim, requested, hidden_pad // 128 * 128)
            # Three GEMMs: gate, up and down. TB/s is a logical I/O lower
            # bound: read each routed expert's packed weights/scales/bias once,
            # plus activations, routing inputs and output (excludes scratch).
            flops = 6 * token * topk * model_dim * INTER_DIM
            active_experts = topk_ids.unique().numel()
            expert_tensors = [w1[0], w1[1], w2[0], w2[1]]
            if bias:
                expert_tensors.extend([b1, b2])
            nbytes = sum(t.numel() * t.element_size() for t in expert_tensors)
            nbytes = nbytes // E * active_experts
            nbytes += sum(
                t.numel() * t.element_size() for t in (x, x, topk_weight, topk_ids)
            )
            ret = {"gfx": get_gfx()}
            for name, fn in candidates.items():
                # Inspect one untimed call; signature binding/list appends must
                # not add host overhead to the measured candidate.
                launched.clear()
                with pytest.MonkeyPatch.context() as launch_patch:
                    launch_patch.setattr(aiter, "moe_cktile2stages_gemm1", spy)
                    fn()
                assert launched == [want], f"unexpected CK-Tile launches: {launched}"
                ref = run_torch(
                    x, topk_weight, topk_ids, w1[2], w2[2], activation, b1, b2
                )
                out, us = run_perftest(fn)
                err = checkAllclose(
                    ref.to(dtypes.fp32),
                    out.to(dtypes.fp32),
                    rtol=1e-2,
                    atol=1e-2,
                    tol_err_ratio=MAX_ERR_RATIO,
                    msg=f"{name}: bf16 MXFP4 MoE split-k",
                )
                # checkAllclose logs/returns the mismatch ratio; make a bad
                # result fail both pytest and the standalone sweep.
                assert err <= MAX_ERR_RATIO
                # Retain the original global-error guard as well, so a large
                # error in a few lanes cannot hide below the mismatch budget.
                rel_l2 = ((out.double() - ref).norm() / ref.norm()).item()
                assert rel_l2 < MAX_REL_L2
                ret[f"{name} us"] = us
                ret[f"{name} TFLOPS"] = flops / us / 1e6
                ret[f"{name} TB/s"] = nbytes / us / 1e6
                ret[f"{name} err"] = err
                ret["k_batch"] = launched[0]
            return ret
    finally:
        fused_moe_module.get_ksplit.cache_clear()
        fused_moe_module.get_2stage_cfgs.cache_clear()


@_SKIP
@pytest.mark.parametrize("model_dim", [256, 512, 768, 1024, 1280, 6144])
@pytest.mark.parametrize(("E", "topk", "token"), [(4, 2, 4), (128, 4, 16)])
def test_bf16_swiglu_cktile_stage1_split_alignment(model_dim, E, topk, token):
    benchmark_fused_moe(model_dim, E, topk, token, ActivationType.Swiglu)


@_SKIP
@pytest.mark.parametrize("hidden_pad", [192, 256])
def test_bf16_swiglu_cktile_stage1_padded_k_does_not_split(hidden_pad):
    # The CK-Tile pipeline skips the k_padded_zeros tail in every split.
    ret = benchmark_fused_moe(1024, 4, 2, 4, ActivationType.Swiglu, hidden_pad)
    assert ret["k_batch"] == 1


@_SKIP
@pytest.mark.parametrize("hidden_pad", [0, 192])
def test_bf16_swiglu_cktile_stage1_gptoss_bias(hidden_pad):
    # gpt-oss: hidden 2880 padded to 3072, with stage 1 and stage 2 biases.
    E, topk, token = 32, 4, 16
    ret = benchmark_fused_moe(
        3072,
        E,
        topk,
        token,
        ActivationType.Swiglu,
        hidden_pad,
        bias=True,
    )
    # Unpadded D=3072 keeps the forced split; the padding disables it.
    assert (ret["k_batch"] == 1) == bool(hidden_pad)


@_SKIP
def test_bf16_swiglu_cktile_stage1_rejects_partial_k_tile():
    with pytest.raises(NotImplementedError, match="model_dim.*multiple of 256"):
        benchmark_fused_moe(384, 4, 2, 4, ActivationType.Swiglu)


_GEMM1_NAME = "moe_cktile2stages_gemm1_256x16x128x256_1x4_16x16x32_1perCU_1x32_swiglu"
_N = 2 * INTER_DIM


def _direct_args(model_dim, n, x_dtype, block_m, E=4, topk=2, token=4):
    # The calls below must raise before any CK-Tile kernel runs, so random
    # MXFP4 bytes and unit scales are enough.
    torch.manual_seed(0)
    x = torch.randn(token, model_dim, device="cuda", dtype=torch.bfloat16)
    score = torch.randn(token, E, device="cuda", dtype=torch.bfloat16)
    topk_weight, topk_ids = fused_topk(x, score, topk, True)
    w = torch.randint(
        0, 256, (E, n, model_dim // 2), dtype=torch.uint8, device="cuda"
    ).view(dtypes.fp4x2)
    w_scale = torch.full(
        (E * n, model_dim // 32), 127, dtype=torch.uint8, device="cuda"
    ).view(dtypes.fp8_e8m0)
    sorted_ids, _, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weight, E, model_dim, torch.bfloat16, block_m
    )
    return x.to(x_dtype), w, w_scale, sorted_ids, sorted_expert_ids, num_valid_ids


_SWIGLU = ActivationType.Swiglu
_RELU2 = ActivationType.Relu2


@_SKIP
@pytest.mark.parametrize(
    ("model_dim", "n", "k_batch", "out_width", "act", "extra", "match"),
    [
        (768, _N, 2, _N, _SWIGLU, {}, "must be a multiple of k_batch"),
        (384, _N, 1, _N // 2, _SWIGLU, {}, "must be a multiple of k_batch"),
        (512, _N, 2, _N // 2, _SWIGLU, {}, "split-k workspace"),
        (512, _N, 1, _N // 4, _SWIGLU, {}, r"must be N \(split-k workspace\) or N/2"),
        (
            512,
            _N,
            1,
            _N // 2,
            _SWIGLU,
            {"kernel_name": _GEMM1_NAME + "_SplitK"},
            "does not match",
        ),
        (512, _N, 1, _N, _SWIGLU, {"kernel_name": _GEMM1_NAME}, "does not match"),
        (1024, _N, 2, _N, _SWIGLU, {"k_padded_zeros": 256}, "k_padded_zeros"),
        # Relu2 is gate-only: w1 has N = inter_dim rows and Y is N wide.
        (384, INTER_DIM, 1, INTER_DIM, _RELU2, {}, "must be a multiple of k_batch"),
        (
            512,
            INTER_DIM,
            2,
            INTER_DIM,
            _RELU2,
            {},
            r"Relu2 \(gate-only\) has no split-k",
        ),
        (512, INTER_DIM, 1, INTER_DIM // 2, _RELU2, {}, "must be N="),
    ],
)
def test_cktile_gemm1_rejects_bad_split(
    model_dim, n, k_batch, out_width, act, extra, match
):
    topk, token, block_m = 2, 4, 16
    x, w, w_scale, sorted_ids, sorted_expert_ids, num_valid_ids = _direct_args(
        model_dim, n, torch.bfloat16, block_m, topk=topk, token=token
    )
    out = torch.zeros(token * topk, out_width, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match=match):
        aiter.moe_cktile2stages_gemm1(
            x,
            w,
            out,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            topk,
            w_scale=w_scale,
            activation=act,
            block_m=block_m,
            split_k=k_batch,
            **extra,
        )


@_SKIP
def test_cktile_a8w4_gemm1_rejects_partial_k_tile():
    # a8w4 (fp8 activations) has no split-k instance; its fused gate/up
    # kernel needs whole K tiles too. Its smallest block_m is 32.
    topk, token, block_m = 2, 4, 32
    x, w, w_scale, sorted_ids, sorted_expert_ids, num_valid_ids = _direct_args(
        384, _N, dtypes.fp8, block_m, topk=topk, token=token
    )
    out = torch.zeros(token * topk, _N // 2, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="must be a multiple of k_batch"):
        aiter.moe_cktile2stages_gemm1(
            x,
            w,
            out,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            topk,
            w_scale=w_scale,
            activation=_SWIGLU,
            block_m=block_m,
            split_k=1,
        )


@_SKIP
def test_cktile_gemm2_rejects_split_k():
    E, topk, token, block_m, model_dim = 4, 2, 4, 16, 512
    _, w, w_scale, sorted_ids, sorted_expert_ids, num_valid_ids = _direct_args(
        INTER_DIM, model_dim, torch.bfloat16, block_m, E=E, topk=topk, token=token
    )
    a2 = torch.randn(token, topk, INTER_DIM, device="cuda", dtype=torch.bfloat16)
    out = torch.zeros(token, model_dim, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="split-k is not supported"):
        aiter.moe_cktile2stages_gemm2(
            a2,
            w,
            out,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            topk,
            w_scale=w_scale,
            activation=_SWIGLU,
            block_m=block_m,
            split_k=2,
        )


@_SKIP
@pytest.mark.parametrize(
    ("model_dim", "split_k", "k_batch"), [(512, 4, 2), (768, 2, 1), (1024, 4, 4)]
)
def test_silu_cktile_stage1_forced_split(model_dim, split_k, k_batch):
    # Silu reaches the same kernel through the ksplit > 1 branch.
    ret = benchmark_fused_moe(
        model_dim, 128, 4, 1, ActivationType.Silu, split_k=split_k
    )
    assert ret["k_batch"] == k_batch


def main():
    if get_gfx() != "gfx950":
        aiter.logger.warning(
            "MXFP4 MoE split-k sweep unsupported on %s; skipping", get_gfx()
        )
        return

    rows = []
    for model_dim, (E, topk, token) in itertools.product(
        [256, 512, 768, 1024, 1280, 6144], [(4, 2, 4), (128, 4, 16)]
    ):
        rows.append(
            benchmark_fused_moe(model_dim, E, topk, token, ActivationType.Swiglu)
        )
    for hidden_pad in [192, 256]:
        rows.append(
            benchmark_fused_moe(1024, 4, 2, 4, ActivationType.Swiglu, hidden_pad)
        )
    for hidden_pad in [0, 192]:
        rows.append(
            benchmark_fused_moe(
                3072, 32, 4, 16, ActivationType.Swiglu, hidden_pad, bias=True
            )
        )
    for model_dim, split_k in [(512, 4), (768, 2), (1024, 4)]:
        rows.append(
            benchmark_fused_moe(
                model_dim, 128, 4, 1, ActivationType.Silu, split_k=split_k
            )
        )
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "CK-Tile MXFP4 MoE split-k summary (markdown):\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
