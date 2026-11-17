# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Default-dispatch regressions for CK MoE and a16w16 swiglu_limit.

gfx942 A16W16 covers shards aligned to 64 but not 128 (for example 320).
gfx950 fp8 stage2 covers shuffled inter_dim 384 (K=128) and 512 (K=256),
rejects sizes that 128 does not divide, and rejects plain fp8 weights unless
both w1 and w2 are shuffled. That rejection follows the live GPU, not
GPU_ARCHS. gfx950 int8 uses that same stage2 K selection.

A finite swiglu_limit on BF16 QuantType.No reroutes to FlyDSL a16wmix a16w16
on gfx942/gfx950. Sentinels stay on CK.
"""

import inspect
import os
import sys
import tempfile

import pytest
import torch
import torch.nn.functional as F

import aiter.fused_moe as fused_moe_module
from aiter import ActivationType, QuantType
from aiter import dtypes as aiter_dtypes
from aiter.fused_moe import (
    _can_reroute_bf16_to_a16wmix,
    _flydsl_stage1_wrapper,
    ck_moe_stage1,
    fused_moe,
    get_2stage_cfgs,
    torch_moe,
)
from aiter.jit.utils.chip_info import get_gfx, get_gfx_runtime
from aiter.ops.moe_op import ck_moe_stage2_fwd
from aiter.ops.quant import pertoken_quant
from aiter.ops.shuffle import shuffle_weight

_ROCM = torch.cuda.is_available() and bool(torch.version.hip)
_ARCH = torch.cuda.get_device_properties(0).gcnArchName if _ROCM else ""
_GFX942 = _ROCM and "gfx942" in _ARCH
_GFX950 = _ROCM and "gfx950" in _ARCH


def _clear_dispatch_caches():
    # get_gfx() is cached from GPU_ARCHS. The torch op and this package can
    # each hold a copy of that cache, so clear every loaded chip_info module.
    # A test that points GPU_ARCHS at another chip must not leak.
    seen = set()
    for name in ("chip_info", "aiter.jit.utils.chip_info"):
        mod = sys.modules.get(name)
        if mod is None or id(mod) in seen:
            continue
        seen.add(id(mod))
        mod.get_gfx_custom_op_core.cache_clear()
        mod.get_gfx.cache_clear()
    get_2stage_cfgs.cache_clear()


@pytest.fixture(autouse=True)
def default_dispatch(monkeypatch):
    # Explicitly bypass CSV tuning: a tuned kernel must not hide this bug.
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    _clear_dispatch_caches()
    yield
    _clear_dispatch_caches()


def _check_moe(tokens, model_dim, inter_dim, experts, topk, block_m, dtype):
    torch.manual_seed(42)
    x = torch.randn(tokens, model_dim, device="cuda", dtype=dtype)
    w1 = (
        torch.randn(experts, 2 * inter_dim, model_dim, device="cuda", dtype=dtype)
        * 0.02
    )
    w2 = torch.randn(experts, model_dim, inter_dim, device="cuda", dtype=dtype) * 0.02
    weights, ids = torch.softmax(
        torch.randn(tokens, experts, device="cuda"), dim=-1
    ).topk(topk, dim=-1)
    ids = ids.to(torch.int32)
    actual = fused_moe(
        x, shuffle_weight(w1), shuffle_weight(w2), weights, ids, block_size_M=block_m
    )
    torch.cuda.synchronize()
    # Independent reference over ALL tokens and experts; accumulate in FP32.
    expected = torch.zeros(tokens, model_dim, device="cuda", dtype=torch.float32)
    for expert in range(experts):
        rows, slots = torch.where(ids == expert)
        if rows.numel() == 0:
            continue
        gate, up = F.linear(x[rows].float(), w1[expert].float()).chunk(2, dim=-1)
        hidden = (F.silu(gate) * up).to(dtype)
        y = F.linear(hidden.float(), w2[expert].float())
        expected.index_add_(0, rows, y * weights[rows, slots, None])
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.002)


@pytest.mark.skipif(
    not _GFX942, reason="Exercises the gfx942 A16W16 CK default dispatch"
)
@pytest.mark.parametrize("inter_dim", [192, 256, 320, 384, 448])
@pytest.mark.parametrize("block_m", [32, 64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_a16w16_ck_default_alignment(inter_dim, block_m, dtype):
    # 320/448 regress the missing alignment check; 192/256/384 retain old paths.
    _check_moe(33, 256, inter_dim, 8, 2, block_m, dtype)


@pytest.mark.skipif(
    not _GFX942, reason="Exercises the gfx942 A16W16 CK default dispatch"
)
@pytest.mark.parametrize("tokens", [4, 512, 8192])
def test_qwen_mtp_tp2_ep1_default_dispatch(tokens):
    # Qwen3.8-Flash-Next MTP TP=2/EP=1 shape, without a block-size override.
    _check_moe(tokens, 2560, 320, 512, 10, None, torch.bfloat16)


def _quantized_weight(weight, quant_type, quant_dtype):
    # per_Tensor is one scale per expert. per_Token is one scale per row.
    source = (
        weight.view(weight.shape[0], -1)
        if quant_type == QuantType.per_Tensor
        else weight
    )
    quantized, scale = pertoken_quant(source, quant_dtype=quant_dtype)
    return quantized.view(weight.shape), scale


def _rows(scale, rows):
    # torch_moe applies one scale per weight row.
    if scale.ndim == 2:
        scale = scale.unsqueeze(-1)
    if scale.shape[1] == rows:
        return scale
    return scale.expand(-1, rows, 1).contiguous()


def _gfx950_fp8_inputs(inter_dim, quant_type, quant_dtype=aiter_dtypes.fp8):
    torch.manual_seed(0)
    tokens, experts, topk, model_dim = 16, 8, 2, 896
    hidden = torch.randn(tokens, model_dim, device="cuda", dtype=torch.bfloat16)
    w1 = (
        torch.randn(
            experts, inter_dim * 2, model_dim, device="cuda", dtype=torch.bfloat16
        )
        * 0.02
    )
    w2 = (
        torch.randn(experts, model_dim, inter_dim, device="cuda", dtype=torch.bfloat16)
        * 0.02
    )
    weights, ids = torch.softmax(
        torch.randn(tokens, experts, device="cuda"), dim=-1
    ).topk(topk, dim=-1)
    ids = ids.to(torch.int32)
    w1q, w1_scale = _quantized_weight(w1, quant_type, quant_dtype)
    w2q, w2_scale = _quantized_weight(w2, quant_type, quant_dtype)
    return hidden, w1q, w2q, w1_scale, w2_scale, weights, ids


def _call_gfx950_fp8(inter_dim, block_m, quant_type, *, shuffle, a1_scale, inputs=None):
    hidden, w1q, w2q, w1_scale, w2_scale, weights, ids = (
        inputs if inputs is not None else _gfx950_fp8_inputs(inter_dim, quant_type)
    )
    if shuffle:
        w1q_in, w2q_in = shuffle_weight(w1q), shuffle_weight(w2q)
    else:
        w1q_in, w2q_in = w1q, w2q
    return fused_moe(
        hidden,
        w1q_in,
        w2q_in,
        weights,
        ids,
        quant_type=quant_type,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        block_size_M=block_m,
        dtype=torch.bfloat16,
    )


def _check_gfx950_fp8(inter_dim, block_m, quant_type, quant_dtype=aiter_dtypes.fp8):
    inputs = _gfx950_fp8_inputs(inter_dim, quant_type, quant_dtype)
    hidden, w1q, w2q, w1_scale, w2_scale, weights, ids = inputs
    # This kernel reads the shuffled weight layout. The reference keeps the
    # original layout and the same scales.
    actual = _call_gfx950_fp8(
        inter_dim,
        block_m,
        quant_type,
        shuffle=True,
        a1_scale=None,
        inputs=inputs,
    ).clone()
    expected = torch_moe(
        hidden,
        w1q,
        w2q,
        weights,
        ids,
        fc1_scale=_rows(w1_scale, w1q.shape[1]),
        fc2_scale=_rows(w2_scale, w2q.shape[1]),
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.05, atol=0.02)


@pytest.mark.skipif(not _GFX950, reason="Exercises the gfx950 fp8 CK default dispatch")
@pytest.mark.parametrize("inter_dim", [384, 512])
@pytest.mark.parametrize("block_m", [32, 64])
@pytest.mark.parametrize("quant_type", [QuantType.per_Tensor, QuantType.per_Token])
def test_gfx950_fp8_stage2_k_alignment(inter_dim, block_m, quant_type):
    # 384 can only complete on K=128. 512 is also correct on K=128, so which
    # tile launched is checked in test_gfx950_stage2_selects_k_tile.
    _check_gfx950_fp8(inter_dim, block_m, quant_type)


@pytest.mark.skipif(not _GFX950, reason="Exercises the gfx950 fp8 CK default dispatch")
@pytest.mark.parametrize("inter_dim", [192, 320])
def test_gfx950_fp8_stage2_rejects_non_128(inter_dim):
    with pytest.raises(RuntimeError, match="multiple of"):
        _check_gfx950_fp8(inter_dim, 32, QuantType.per_Tensor)


@pytest.mark.skipif(not _GFX950, reason="Exercises the gfx950 fp8 CK default dispatch")
@pytest.mark.parametrize("inter_dim", [384, 512])
@pytest.mark.parametrize(
    ("quant_type", "static_scale"),
    [
        (QuantType.per_Token, False),
        (QuantType.per_Tensor, True),
        (QuantType.per_Tensor, False),
    ],
)
def test_gfx950_fp8_rejects_unshuffled_weights(inter_dim, quant_type, static_scale):
    # Issue 6452 passes unshuffled weights: per-token, per-tensor with a static
    # activation scale, and per-tensor with a dynamic activation scale.
    a1_scale = (
        torch.tensor([1.0], dtype=torch.float32, device="cuda")
        if static_scale
        else None
    )
    with pytest.raises(RuntimeError, match="shuffle_weight"):
        _call_gfx950_fp8(inter_dim, None, quant_type, shuffle=False, a1_scale=a1_scale)


@pytest.mark.skipif(not _GFX950, reason="Exercises the gfx950 fp8 CK default dispatch")
@pytest.mark.parametrize("shuffle_w1", [True, False])
def test_gfx950_fp8_rejects_partially_shuffled_weights(shuffle_w1):
    # Either mixed pair must fail. is_shuffled on only one weight used to
    # satisfy the guard and let the other stage read the wrong layout.
    hidden, w1q, w2q, w1_scale, w2_scale, weights, ids = _gfx950_fp8_inputs(
        384, QuantType.per_Tensor
    )
    w1_in = shuffle_weight(w1q) if shuffle_w1 else w1q
    w2_in = w2q if shuffle_w1 else shuffle_weight(w2q)
    with pytest.raises(RuntimeError, match="both w1 and w2"):
        fused_moe(
            hidden,
            w1_in,
            w2_in,
            weights,
            ids,
            quant_type=QuantType.per_Tensor,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            block_size_M=32,
            dtype=torch.bfloat16,
        )


@pytest.mark.skipif(not _GFX950, reason="Exercises the gfx950 int8 CK default dispatch")
@pytest.mark.parametrize("inter_dim", [384, 512])
@pytest.mark.parametrize("block_m", [32, 64])
def test_gfx950_int8_stage2_k_alignment(inter_dim, block_m):
    # int8 shares a8w8_gfx950 with fp8. Tile identity is in
    # test_gfx950_stage2_selects_k_tile; this checks the numeric result.
    _check_gfx950_fp8(
        inter_dim, block_m, QuantType.per_Tensor, quant_dtype=aiter_dtypes.i8
    )


@pytest.mark.skipif(not _GFX950, reason="Exercises the gfx950 int8 CK default dispatch")
def test_gfx950_int8_stage2_rejects_non_128():
    # Int8 fused_moe stage 1 is the ASM kernel, and every gfx950 int8 stage-1
    # tile is 128 or 256, so width 192 fails there and never reaches stage 2.
    # The K check added in this change lives in the shared CK stage-2 dispatch.
    inter_dim, model_dim, tokens, experts = 192, 128, 4, 1
    inter = torch.empty((tokens, inter_dim), dtype=torch.int8, device="cuda")
    w1 = torch.empty(
        (experts, inter_dim * 2, model_dim), dtype=torch.int8, device="cuda"
    )
    w2 = torch.empty((experts, model_dim, inter_dim), dtype=torch.int8, device="cuda")
    sorted_ids = torch.zeros(32, dtype=torch.int32, device="cuda")
    sorted_expert_ids = torch.zeros(4, dtype=torch.int32, device="cuda")
    num_valid = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.empty((tokens, model_dim), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="multiple of"):
        ck_moe_stage2_fwd(
            inter,
            w1,
            w2,
            sorted_ids,
            sorted_expert_ids,
            num_valid,
            out,
            1,
            block_m=32,
            quant_type=QuantType.per_Tensor,
        )


@pytest.mark.skipif(not _GFX950, reason="Exercises the gfx950 fp8 CK default dispatch")
@pytest.mark.parametrize("gpu_archs", ["gfx942", "gfx950;gfx942"])
def test_gfx950_fp8_guard_ignores_gpu_archs(monkeypatch, gpu_archs):
    # get_gfx() returns the last GPU_ARCHS entry, so both of these name gfx942
    # on a gfx950 device. per_Tensor fp8 is not in the gfx942 1-stage table,
    # so the call still reaches the CK stage-2 guard. Reverting that guard to
    # get_gfx() would skip the error and return a tensor.
    monkeypatch.setenv("GPU_ARCHS", gpu_archs)
    _clear_dispatch_caches()
    try:
        assert get_gfx() == "gfx942"
        assert get_gfx_runtime() == "gfx950"
        with pytest.raises(RuntimeError, match="both w1 and w2"):
            _call_gfx950_fp8(
                384, 32, QuantType.per_Tensor, shuffle=False, a1_scale=None
            )
    finally:
        _clear_dispatch_caches()


@pytest.mark.skipif(not _GFX950, reason="Exercises the gfx950 fp8 CK default dispatch")
def test_gfx950_fp8_tuned_ck_rejects_mixed_shuffle(monkeypatch):
    # A matching tuned row skips the default heuristic. is_shuffled is the OR,
    # so one tagged weight used to miss the warning and launch the CK kernel.
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "0")
    key = (
        get_gfx_runtime(),
        fused_moe_module.get_cu_num(),
        16,
        896,
        384,
        8,
        2,
        str(ActivationType.Silu),
        str(torch.bfloat16),
        str(aiter_dtypes.fp8),
        str(aiter_dtypes.fp8),
        str(QuantType.per_Tensor),
        True,
        False,
    )
    monkeypatch.setattr(
        fused_moe_module,
        "cfg_2stages",
        (
            {
                key: {
                    "block_m": 32,
                    "ksplit": 0,
                    "kernelName1": "moe_ck2stages_gemm1_test",
                    "kernelName2": "moe_ck2stages_gemm2_test",
                    "run_1stage": False,
                }
            },
            {},
        ),
    )

    def dispatch(both_shuffled):
        return get_2stage_cfgs(
            16,
            896,
            384,
            8,
            2,
            torch.bfloat16,
            aiter_dtypes.fp8,
            aiter_dtypes.fp8,
            QuantType.per_Tensor,
            True,
            ActivationType.Silu,
            False,
            0,
            0,
            True,
            opus_weights_shuffled=both_shuffled,
        )

    _clear_dispatch_caches()
    # A missed row falls through to the default heuristic, whose stage-2
    # kernel name is empty. This has to be the tuned kernel or the raise
    # below is the default guard instead.
    selected = dispatch(True)
    assert selected.stage2.keywords["kernelName"] == "moe_ck2stages_gemm2_test"
    with pytest.raises(RuntimeError, match="both w1 and w2"):
        dispatch(False)


def _stage2_k(inter_dim, block_m):
    # block_m 32 and 64 use K=256 when 256 divides inter_dim, else K=128.
    if block_m in (32, 64) and inter_dim % 256 == 0:
        return 256
    return 128


def _chrome_trace(fn):
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn()
        torch.cuda.synchronize()
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        prof.export_chrome_trace(path)
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    finally:
        os.remove(path)


@pytest.mark.skipif(not _GFX950, reason="Exercises the gfx950 a8w8 CK default dispatch")
@pytest.mark.parametrize("inter_dim", [384, 512])
@pytest.mark.parametrize("block_m", [32, 64])
@pytest.mark.parametrize("quant_dtype", [aiter_dtypes.fp8, aiter_dtypes.i8])
def test_gfx950_stage2_selects_k_tile(inter_dim, block_m, quant_dtype):
    # A matching torch_moe result does not identify the tile: K=128 also
    # divides 512. The launched DeviceMoeGemm template carries
    # block size, MPerBlock, NPerBlock, KPerBlock. Stage 1 uses NPerBlock 64.
    expected = _stage2_k(inter_dim, block_m)
    marker = f"256, {block_m}, 128, {expected},"
    other = f"256, {block_m}, 128, {128 if expected == 256 else 256},"
    inputs = _gfx950_fp8_inputs(inter_dim, QuantType.per_Tensor, quant_dtype)

    def launch():
        _call_gfx950_fp8(
            inter_dim,
            block_m,
            QuantType.per_Tensor,
            shuffle=True,
            a1_scale=None,
            inputs=inputs,
        )

    launch()
    trace = _chrome_trace(launch)
    if marker not in trace:
        idx = trace.find("DeviceMoeGemm")
        snippet = trace[idx : idx + 500] if idx >= 0 else trace[:500]
        raise AssertionError(
            f"stage2 K={expected} marker {marker!r} missing from kernel trace: {snippet!r}"
        )
    assert other not in trace


# ---------------------------------------------------------------------------
# BF16 QuantType.No + finite swiglu_limit → FlyDSL a16wmix a16w16
# ---------------------------------------------------------------------------

_A16W16_S1 = "flydsl_moe1_abf16_wbf16_bf16_t16x128x128"
_A16W16_S2 = "flydsl_moe2_abf16_wbf16_bf16_t16x128x128_atomic"
_A16W16_ARCHS = ("gfx942", "gfx950")
_SERVER = {"model_dim": 4096, "inter_dim": 512, "expert": 128, "topk": 8}


def _patch_gfx(monkeypatch, gfx):
    monkeypatch.setattr(fused_moe_module, "get_gfx", lambda: gfx)
    monkeypatch.setattr(fused_moe_module, "get_gfx_runtime", lambda: gfx)


def _a16w16_cfg_kwargs(
    token=4,
    *,
    shuffled=True,
    limit=7.0,
    activation=ActivationType.Silu,
):
    return {
        "token": token,
        "model_dim": 256,
        "inter_dim": 128,
        "expert": 4,
        "topk": 2,
        "dtype": aiter_dtypes.bf16,
        "q_dtype_a": aiter_dtypes.bf16,
        "q_dtype_w": aiter_dtypes.bf16,
        "q_type": QuantType.No,
        "use_g1u1": True,
        "activation": activation,
        "doweight_stage1": False,
        "hidden_pad": 0,
        "intermediate_pad": 0,
        "is_shuffled": shuffled,
        "opus_weights_shuffled": shuffled,
        "swiglu_limit": limit,
    }


def _reroute_base(**overrides):
    base = {
        "gfx": "gfx942",
        "model_dim": 256,
        "inter_dim": 128,
        "dtype": aiter_dtypes.bf16,
        "q_dtype_a": aiter_dtypes.bf16,
        "q_dtype_w": aiter_dtypes.bf16,
        "q_type": QuantType.No,
        "activation": ActivationType.Silu,
        "is_shuffled": True,
        "use_g1u1": True,
        "doweight_stage1": False,
        "swiglu_limit": 7.0,
        "has_stage1_bias": False,
        "hidden_pad": 0,
        "intermediate_pad": 0,
    }
    base.update(overrides)
    return base


def _stage1(metadata):
    return getattr(metadata.stage1, "func", metadata.stage1)


def _silu_hidden(gate, up, *, limit=None, symmetric=False):
    if limit is not None:
        if symmetric:
            gate = gate.clamp(min=-limit, max=limit)
        else:
            gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


def _swiglu_hidden(gate, up, *, limit=None):
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(1.702 * gate) * (up + 1.0)


def _moe_ref(x, w1, w2, ids, weights, *, act, limit=None, symmetric=False):
    tokens, model_dim = x.shape
    expected = torch.zeros(tokens, model_dim, device=x.device, dtype=torch.float32)
    for expert in range(w1.shape[0]):
        rows, slots = torch.where(ids == expert)
        if rows.numel() == 0:
            continue
        gate, up = F.linear(x[rows].float(), w1[expert].float()).chunk(2, dim=-1)
        hidden = (
            _silu_hidden(gate, up, limit=limit, symmetric=symmetric)
            if act == "silu"
            else _swiglu_hidden(gate, up, limit=limit)
        )
        y = F.linear(hidden, w2[expert].float())
        expected.index_add_(0, rows, y * weights[rows, slots, None].float())
    return expected


def _run_fused(x, w1, w2, weights, ids, *, limit, activation):
    return fused_moe(
        x,
        shuffle_weight(w1),
        shuffle_weight(w2),
        weights,
        ids,
        activation=activation,
        quant_type=QuantType.No,
        swiglu_limit=limit,
    )


def _bind_inputs(tokens, *, seed=0, limit=1.0):
    torch.manual_seed(seed)
    experts, model_dim, inter_dim, topk = 4, 256, 128, 2
    x = torch.randn(tokens, model_dim, device="cuda", dtype=aiter_dtypes.bf16)
    w1 = torch.randn(
        experts, 2 * inter_dim, model_dim, device="cuda", dtype=aiter_dtypes.bf16
    )
    w2 = torch.randn(
        experts, model_dim, inter_dim, device="cuda", dtype=aiter_dtypes.bf16
    )
    with torch.no_grad():
        probe = F.linear(x[:1].float(), w1[0].float()).abs().max().item()
        w1.mul_(max(4.0 * limit / max(probe, 1e-3), 1.0))
    weights, ids = torch.softmax(
        torch.randn(tokens, experts, device="cuda"), dim=-1
    ).topk(topk, dim=-1)
    return x, w1, w2, weights, ids.to(torch.int32)


def test_a16w16_flydsl_names_are_bf16_only():
    from aiter.ops.flydsl.moe_kernels import (
        get_flydsl_kernel_params,
        get_flydsl_stage1_kernels_a16w16,
        get_flydsl_stage2_kernels_a16w16,
    )

    p1 = get_flydsl_kernel_params(_A16W16_S1)
    p2 = get_flydsl_kernel_params(_A16W16_S2)
    assert p1 is not None and p2 is not None
    assert p1["a_dtype"] == "bf16" and p1["b_dtype"] == "bf16"
    assert p2["mode"] == "atomic"
    assert get_flydsl_kernel_params("flydsl_moe1_abf16_wbf16_f16_t16x128x128") is None
    assert get_flydsl_stage1_kernels_a16w16("f16") == {}
    assert get_flydsl_stage2_kernels_a16w16("f16") == {}


def test_a16w16_force_reduce_stays_atomic(monkeypatch):
    import functools

    monkeypatch.setenv("AITER_FLYDSL_FORCE_REDUCE", "1")
    stage2 = functools.partial(
        fused_moe_module._flydsl_stage2_wrapper, kernelName=_A16W16_S2
    )
    assert not fused_moe_module._stage2_honors_runtime_reduce_override(stage2)
    assert not fused_moe_module._stage2_runtime_uses_route_reduce(stage2, 4, 8192)


@pytest.mark.parametrize("activation", [ActivationType.Silu, ActivationType.Swiglu])
def test_a16w16_reroute_gates(activation):
    base = _reroute_base(activation=activation)
    assert _can_reroute_bf16_to_a16wmix(**base)
    assert _can_reroute_bf16_to_a16wmix(**{**base, "gfx": "gfx950"})
    for limit in (None, 0.0, float("inf")):
        assert not _can_reroute_bf16_to_a16wmix(**{**base, "swiglu_limit": limit})
    assert not _can_reroute_bf16_to_a16wmix(**{**base, "is_shuffled": False})
    assert not _can_reroute_bf16_to_a16wmix(**{**base, "use_g1u1": False})
    assert not _can_reroute_bf16_to_a16wmix(**{**base, "gfx": "gfx90a"})
    assert not _can_reroute_bf16_to_a16wmix(**{**base, "gfx": "gfx1100"})


@pytest.mark.parametrize("activation", [ActivationType.Silu, ActivationType.Swiglu])
def test_a16w16_limit_selects_flydsl(monkeypatch, activation):
    _patch_gfx(monkeypatch, "gfx942")
    metadata = get_2stage_cfgs(**_a16w16_cfg_kwargs(activation=activation))
    assert _stage1(metadata) is _flydsl_stage1_wrapper
    assert metadata.stage1.keywords["kernelName"] == _A16W16_S1
    assert metadata.stage2.keywords["kernelName"] == _A16W16_S2


@pytest.mark.parametrize("limit", [None, 0.0, float("inf")])
def test_a16w16_sentinels_stay_on_ck(monkeypatch, limit):
    _patch_gfx(monkeypatch, "gfx942")
    metadata = get_2stage_cfgs(**_a16w16_cfg_kwargs(limit=limit))
    assert _stage1(metadata) is ck_moe_stage1
    assert not metadata.run_1stage


def test_a16w16_gfx90a_limit_stays_on_ck(monkeypatch):
    _patch_gfx(monkeypatch, "gfx90a")
    metadata = get_2stage_cfgs(**_a16w16_cfg_kwargs())
    assert _stage1(metadata) is ck_moe_stage1


def test_a16w16_aot_has_quanttype_no_recipe():
    from aiter.aot.flydsl import moe as aot_moe

    src = inspect.getsource(aot_moe._caller_inputs)
    assert "QuantType.No" in src
    assert "get_flydsl_stage1_kernels_a16w16" in src


def test_a16w16_untuned_tile_follows_tokens(monkeypatch):
    _patch_gfx(monkeypatch, "gfx942")
    md16 = get_2stage_cfgs(**_a16w16_cfg_kwargs(token=4))
    md32 = get_2stage_cfgs(**_a16w16_cfg_kwargs(token=2048))
    assert md16.block_m == 16 and "t16x" in md16.stage1.keywords["kernelName"]
    assert md32.block_m == 32 and "t32x" in md32.stage1.keywords["kernelName"]


@pytest.mark.skipif(not _GFX942 and not _GFX950, reason="a16wmix a16w16 gfx942/950")
@pytest.mark.parametrize(
    "activation,act_name",
    [(ActivationType.Silu, "silu"), (ActivationType.Swiglu, "swiglu")],
)
def test_a16w16_swiglu_limit_binds(activation, act_name):
    limit = 1.0
    x, w1, w2, weights, ids = _bind_inputs(4, limit=limit)
    metadata = get_2stage_cfgs(
        **_a16w16_cfg_kwargs(token=4, limit=limit, activation=activation)
    )
    assert _stage1(metadata) is _flydsl_stage1_wrapper
    actual = _run_fused(x, w1, w2, weights, ids, limit=limit, activation=activation)
    clamped = _moe_ref(x, w1, w2, ids, weights, act=act_name, limit=limit)
    unclamped = _moe_ref(x, w1, w2, ids, weights, act=act_name, limit=None)
    err_c = (actual.float() - clamped).abs().max().item()
    err_u = (actual.float() - unclamped).abs().max().item()
    assert err_c < err_u
    huge = _run_fused(x, w1, w2, weights, ids, limit=1e9, activation=activation)
    assert not torch.equal(actual, huge)


@pytest.mark.skipif(not _GFX942 and not _GFX950, reason="a16wmix a16w16 gfx942/950")
def test_a16w16_rejects_mismatched_block_size_m():
    x, w1, w2, weights, ids = _bind_inputs(4, limit=1.0)
    metadata = get_2stage_cfgs(
        **_a16w16_cfg_kwargs(token=4, limit=1.0, activation=ActivationType.Silu)
    )
    assert metadata.block_m == 16
    with pytest.raises(ValueError, match="block_size_M"):
        fused_moe(
            x,
            shuffle_weight(w1),
            shuffle_weight(w2),
            weights,
            ids,
            activation=ActivationType.Silu,
            quant_type=QuantType.No,
            swiglu_limit=1.0,
            block_size_M=32,
        )


@pytest.mark.skipif(not _GFX942 and not _GFX950, reason="a16wmix a16w16 gfx942/950")
def test_a16w16_server_shape_clamp_binds():
    limit = 7.0
    hidden, inter, experts, topk = (
        _SERVER[k] for k in ("model_dim", "inter_dim", "expert", "topk")
    )
    torch.manual_seed(20261016)
    tokens = 32
    x = torch.randn(tokens, hidden, device="cuda", dtype=aiter_dtypes.bf16)
    w1 = (torch.randn(experts, 2 * inter, hidden, device="cuda") * 0.75).to(
        aiter_dtypes.bf16
    )
    w2 = (torch.randn(experts, hidden, inter, device="cuda") / inter**0.5).to(
        aiter_dtypes.bf16
    )
    weights, ids = torch.softmax(
        torch.randn(tokens, experts, device="cuda"), dim=-1
    ).topk(topk, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    ids = ids.to(torch.int32)
    metadata = get_2stage_cfgs(
        **{**_a16w16_cfg_kwargs(token=tokens, limit=limit), **_SERVER}
    )
    assert metadata.stage1.keywords["kernelName"] == _A16W16_S1
    actual = _run_fused(
        x, w1, w2, weights, ids, limit=limit, activation=ActivationType.Silu
    ).float()
    loose = _run_fused(
        x, w1, w2, weights, ids, limit=1e9, activation=ActivationType.Silu
    ).float()
    clamped = _moe_ref(x, w1, w2, ids, weights, act="silu", limit=limit)
    unclamped = _moe_ref(x, w1, w2, ids, weights, act="silu", limit=None)
    scale_c = clamped.abs().max().item()
    err_c = (actual - clamped).abs().max().item()
    err_u = (actual - unclamped).abs().max().item()
    assert err_c <= 2e-2 * scale_c
    assert err_u >= 10 * scale_c
    assert not torch.equal(actual, loose)


@pytest.mark.skipif(not _GFX942 and not _GFX950, reason="a16wmix a16w16 gfx942/950")
def test_a16w16_server_shape_asymmetric_negative_gate():
    limit = 1.0
    hidden, inter, experts, topk = (
        _SERVER[k] for k in ("model_dim", "inter_dim", "expert", "topk")
    )
    tokens = 32
    w1 = torch.zeros(experts, 2 * inter, hidden, device="cuda", dtype=aiter_dtypes.bf16)
    w2 = torch.zeros(experts, hidden, inter, device="cuda", dtype=aiter_dtypes.bf16)
    w1[0, :inter, :] = -25.0 / hidden
    w1[0, inter:, :] = 3.0 / hidden
    w1[1, :inter, :] = 8.0 / hidden
    w1[1, inter:, :] = 3.0 / hidden
    w2[:2] = 1.0 / 64
    x = torch.ones(tokens, hidden, device="cuda", dtype=aiter_dtypes.bf16)
    ids = (
        torch.arange(topk, device="cuda", dtype=torch.int32)
        .unsqueeze(0)
        .repeat(tokens, 1)
    )
    weights = torch.full((tokens, topk), 0.5, device="cuda")
    actual = _run_fused(
        x, w1, w2, weights, ids, limit=limit, activation=ActivationType.Silu
    ).float()
    asym = _moe_ref(x, w1, w2, ids, weights, act="silu", limit=limit)
    sym = _moe_ref(x, w1, w2, ids, weights, act="silu", limit=limit, symmetric=True)
    err_as = (actual - asym).abs().max().item()
    err_sy = (actual - sym).abs().max().item()
    assert err_as < 0.05
    assert err_sy > 0.5
