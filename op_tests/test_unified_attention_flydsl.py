# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Routing of `unified_attention` to the FlyDSL fp8 gfx950 backend.

Three things are checked, and the first matters most: that routing HAPPENS.
A gate bug that declines everything would leave the numeric tests below passing
vacuously on Triton output, so the dispatch is asserted directly.

Numeric correctness of the FlyDSL kernel itself lives in
op_tests/flydsl_tests/test_flydsl_unified_attention.py.

There is no environment variable for this backend (deliberately -- see the
adapter's module docstring), so the A/B here patches the module-level arch
constant instead.
"""

import math
import os
import sys
from unittest import mock

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import aiter.ops.triton.attention.unified_attention as ua
from aiter.ops.flydsl.utils import is_flydsl_available
from op_tests.flydsl_tests.test_flydsl_unified_attention import _build
from op_tests.triton_tests.attention.test_unified_attention import (
    ref_paged_attn,
)

try:
    from aiter.jit.utils.chip_info import get_gfx_runtime

    _ARCH = get_gfx_runtime()
except Exception:  # noqa: BLE001
    _ARCH = None

pytestmark = [
    pytest.mark.skipif(_ARCH != "gfx950", reason=f"gfx950 only, got {_ARCH}"),
    pytest.mark.skipif(not is_flydsl_available(), reason="flydsl not available"),
]

PAGE = 64
HEAD_DIM = 128
SCALE = 1.0 / math.sqrt(HEAD_DIM)


def _call(query_lens, kv_lens, num_heads=64, num_kv_heads=4, seed=0, **over):
    """Invoke the public unified_attention on a supported-by-default config.

    `over` replaces any argument, which is how the decline cases below make a
    single aspect of the call unsupported.
    """
    q, k, v, out, cu_q, seqused_k, bt, q_ds, k_ds, v_ds = _build(
        query_lens, kv_lens, num_heads, num_kv_heads, seed
    )
    kwargs = {
        "q": q,
        "k": k,
        "v": v,
        "out": out,
        "cu_seqlens_q": cu_q,
        "max_seqlen_q": max(query_lens),
        "seqused_k": seqused_k,
        "max_seqlen_k": max(kv_lens),
        "softmax_scale": SCALE,
        "causal": True,
        "window_size": (-1, -1),
        "block_table": bt,
        "softcap": 0,
        "q_descale": q_ds,
        "k_descale": k_ds,
        "v_descale": v_ds,
    }
    kwargs.update(over)
    return ua.unified_attention(**kwargs)


def _triton_only(*args, **kwargs):
    """Run the same call with the FlyDSL backend disabled."""
    with mock.patch.object(ua, "_FLYDSL_UNIFIED_ATTN_ARCH", False):
        return _call(*args, **kwargs)


def _call_with_ref(query_lens, kv_lens, num_heads=64, num_kv_heads=4, seed=0, **over):
    """Like `_call`, but also returns the fp32 reference so a single build can
    check both routing and numeric agreement -- used by the positive-coverage
    tests below, which compare against `ref_paged_attn` rather than Triton."""
    q, k, v, out, cu_q, seqused_k, bt, q_ds, k_ds, v_ds = _build(
        query_lens, kv_lens, num_heads, num_kv_heads, seed
    )
    kwargs = {
        "q": q,
        "k": k,
        "v": v,
        "out": out,
        "cu_seqlens_q": cu_q,
        "max_seqlen_q": max(query_lens),
        "seqused_k": seqused_k,
        "max_seqlen_k": max(kv_lens),
        "softmax_scale": SCALE,
        "causal": True,
        "window_size": (-1, -1),
        "block_table": bt,
        "softcap": 0,
        "q_descale": q_ds,
        "k_descale": k_ds,
        "v_descale": v_ds,
    }
    kwargs.update(over)
    got = ua.unified_attention(**kwargs)
    want = ref_paged_attn(
        query=q,
        key_cache=k,
        value_cache=v,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=bt,
        scale=SCALE,
        out_dtype=kwargs["out"].dtype,
        q_descale=q_ds,
        k_descale=k_ds,
        v_descale=v_ds,
        sinks=kwargs.get("sinks"),
        causal=int(kwargs["causal"]),
    )
    return got.float(), want.float().reshape(got.shape)


def _assert_close(got, want, tol=1.5e-1):
    """Same scale-relative tolerance the parity checks above use -- these two
    are different fp8/fp32 kernels, not a bitwise comparison."""
    scale = want.abs().max().item()
    rel = (got - want).abs().max().item() / scale
    cos = torch.nn.functional.cosine_similarity(
        got.reshape(-1), want.reshape(-1), dim=0
    ).item()
    assert torch.isfinite(got).all()
    assert rel < tol, f"rel err {rel:.4g}"
    assert cos > 0.99, f"cosine {cos:.6f}"


# --- 1. routing actually happens ---------------------------------------------


def test_arch_constant_is_true_on_gfx950():
    """Without this the whole backend is dead code on its own target."""
    assert ua._FLYDSL_UNIFIED_ATTN_ARCH is True


def test_supported_config_routes_to_flydsl():
    """The load-bearing test in this file: a supported call must reach the
    adapter AND be served by it (not merely offered to it and declined)."""
    import aiter.ops.flydsl.unified_attention_kernels as uak

    real = uak.flydsl_unified_attention
    seen = {}

    def spy(*a, **kw):
        r = real(*a, **kw)
        seen["served"] = r is not None
        return r

    with mock.patch.object(uak, "flydsl_unified_attention", spy):
        _call([256], [256])

    assert seen.get("served") is True, (
        "unified_attention did not dispatch to the FlyDSL backend on a "
        "supported gfx950 fp8 paged config"
    )


@pytest.mark.parametrize(
    "query_lens,kv_lens",
    [
        ([256], [256]),  # single prefill
        ([1] * 8, [4096] * 8),  # pure decode
        ([4023] + [1] * 8, [4023] + [16384] * 8),  # mixed: the production shape
    ],
)
def test_routes_across_shape_regimes(query_lens, kv_lens):
    import aiter.ops.flydsl.unified_attention_kernels as uak

    real = uak.flydsl_unified_attention
    seen = {}

    def spy(*a, **kw):
        r = real(*a, **kw)
        seen["served"] = r is not None
        return r

    with mock.patch.object(uak, "flydsl_unified_attention", spy):
        _call(query_lens, kv_lens)
    assert seen.get("served") is True


# --- 2. parity with Triton ---------------------------------------------------


@pytest.mark.parametrize(
    "query_lens,kv_lens",
    [([256], [256]), ([1] * 8, [4096] * 8), ([4023] + [1] * 8, [4023] + [16384] * 8)],
)
def test_parity_with_triton(query_lens, kv_lens):
    """Both backends must agree to within fp8 noise.

    Deliberately NOT a bitwise comparison: these are two different fp8 kernels
    with different accumulation orders, so exact equality would be permanently
    red. Per-element correctness against an fp32 reference is covered in the
    direct-wrapper suite; what this checks is that routing does not change the
    answer beyond quantization noise.
    """
    got = _call(query_lens, kv_lens).float()
    want = _triton_only(query_lens, kv_lens).float()

    scale = want.abs().max().item()
    rel = (got - want).abs().max().item() / scale
    cos = torch.nn.functional.cosine_similarity(
        got.reshape(-1), want.reshape(-1), dim=0
    ).item()
    assert torch.isfinite(got).all()
    assert rel < 1.5e-1, f"rel err vs Triton {rel:.4g}"
    assert cos > 0.99, f"cosine vs Triton {cos:.6f}"


# --- 3. unsupported configs fall through, byte-identically -------------------


def _decline_cases():
    """Each entry makes exactly one aspect of an otherwise-supported call
    unsupported. Kept as a factory so each builds fresh tensors."""
    return {
        "softcap": {"softcap": 30.0},
        "sliding_window": {"window_size": (256, 0)},
        # block_table=None is covered in the predicate test instead: Triton's
        # own path dereferences it unconditionally (unified_attention.py:637),
        # so there is no fall-through target to compare against.
        "alibi_slopes": {
            "alibi_slopes": torch.ones(64, device="cuda", dtype=torch.float32)
        },
        "output_scale": {
            "output_scale": torch.ones(1, device="cuda", dtype=torch.float32)
        },
    }


@pytest.mark.parametrize("name", sorted(_decline_cases()))
def test_declined_configs_match_triton_bitwise(name):
    """A declined call must produce EXACTLY what Triton alone produces.

    Exact equality is legitimate here (same kernel, same inputs) and is what
    catches a leaking gate: if FlyDSL served a config it should have refused,
    the fp8 differences would show up immediately.
    """
    over = _decline_cases()[name]
    got = _call([256], [256], seed=7, **over)
    want = _triton_only([256], [256], seed=7, **over)
    assert torch.equal(got, want), (
        f"{name} was expected to fall through to Triton, but the result differs "
        "-- the support gate is letting an unsupported config through"
    )


@pytest.mark.parametrize("num_kv_heads", [64, 16, 4, 1])
def test_gqa_ratios_route_and_agree(num_kv_heads):
    got = _call([256], [256], num_kv_heads=num_kv_heads).float()
    want = _triton_only([256], [256], num_kv_heads=num_kv_heads).float()
    scale = want.abs().max().item()
    assert (got - want).abs().max().item() / scale < 1.5e-1


# --- positive coverage: configs that route to FlyDSL and must agree ----------


def test_non_causal_routes_and_agrees():
    """causal=False is accepted by `_supported` (item 3) and must actually reach
    the adapter through the public entry, not just be theoretically unsupported
    -- the Triton fallback below the dispatch is causal-only, so a non-causal
    call that FlyDSL declined would trip its `assert causal`."""
    import aiter.ops.flydsl.unified_attention_kernels as uak

    real = uak.flydsl_unified_attention
    seen = {}

    def spy(*a, **kw):
        r = real(*a, **kw)
        seen["served"] = r is not None
        return r

    with mock.patch.object(uak, "flydsl_unified_attention", spy):
        got, want = _call_with_ref([256], [256], causal=False)
    assert seen.get("served") is True, "non-causal call did not reach FlyDSL"
    _assert_close(got, want)


def test_fp16_output_routes_and_agrees():
    """`out.dtype == float16` is accepted now (was declined); check it both
    routes and agrees with the fp32 reference through the public entry."""
    import aiter.ops.flydsl.unified_attention_kernels as uak

    real = uak.flydsl_unified_attention
    seen = {}

    def spy(*a, **kw):
        r = real(*a, **kw)
        seen["served"] = r is not None
        return r

    q, k, v, out, cu_q, seqused_k, bt, q_ds, k_ds, v_ds = _build([256], [256], 64, 4)
    out = out.to(torch.float16)
    with mock.patch.object(uak, "flydsl_unified_attention", spy):
        got = ua.unified_attention(
            q=q,
            k=k,
            v=v,
            out=out,
            cu_seqlens_q=cu_q,
            max_seqlen_q=256,
            seqused_k=seqused_k,
            max_seqlen_k=256,
            softmax_scale=SCALE,
            causal=True,
            window_size=(-1, -1),
            block_table=bt,
            softcap=0,
            q_descale=q_ds,
            k_descale=k_ds,
            v_descale=v_ds,
        )
    assert seen.get("served") is True, "fp16-output call did not reach FlyDSL"
    want = ref_paged_attn(
        query=q,
        key_cache=k,
        value_cache=v,
        query_lens=[256],
        kv_lens=[256],
        block_tables=bt,
        scale=SCALE,
        out_dtype=torch.float16,
        q_descale=q_ds,
        k_descale=k_ds,
        v_descale=v_ds,
    )
    _assert_close(got.float(), want.float().reshape(got.shape))


def test_sinks_route_and_agree_and_stay_single_pass():
    """Per-head fp32 sinks are accepted now (item 4). Check the call routes,
    agrees with the fp32 reference, AND stays single-pass -- `_supported`'s
    sinks branch depends on the dispatch gate in `flydsl_unified_attention`
    never selecting split-K when sinks is not None (see its comment); this
    proves that invariant holds for a shape that would otherwise split."""
    import aiter.ops.flydsl.unified_attention_kernels as uak

    real_get_kernel = uak._get_kernel.__wrapped__
    seen = {}

    def spy(num_heads, num_kv_heads, causal, out_dtype_str, use_sinks, num_kv_splits=1):
        seen["num_kv_splits"] = num_kv_splits
        return real_get_kernel(
            num_heads, num_kv_heads, causal, out_dtype_str, use_sinks, num_kv_splits
        )

    # An all-decode shape that would otherwise take split-K (see
    # test_packed_splitk_decode.py), to prove sinks forces single-pass.
    query_lens, kv_lens = [1] * 8, [16384] * 8
    sinks = torch.randn(64, device="cuda", dtype=torch.float32)
    uak._get_kernel.cache_clear()
    with mock.patch.object(uak, "_get_kernel", spy):
        got, want = _call_with_ref(query_lens, kv_lens, sinks=sinks)
    assert seen.get("num_kv_splits") == 1, (
        f"sinks call used num_kv_splits={seen.get('num_kv_splits')}, expected 1 "
        "(single-pass) -- split-K + sinks double-counts exp(sink) in the "
        "combine denominator"
    )
    _assert_close(got, want)


def test_fp16_output_with_packed_splitk():
    """fp16 output crossed with packed split-K: the combine kernel's
    `part_dtype` store-packer logic is unique to fp16/bf16 output under
    split-K, and this combination is otherwise only exercised through fp16
    alone or split-K alone (item 13)."""
    import aiter.ops.flydsl.unified_attention_kernels as uak

    real_get_kernel = uak._get_kernel.__wrapped__
    seen = {}

    def spy(num_heads, num_kv_heads, causal, out_dtype_str, use_sinks, num_kv_splits=1):
        seen["num_kv_splits"] = num_kv_splits
        return real_get_kernel(
            num_heads, num_kv_heads, causal, out_dtype_str, use_sinks, num_kv_splits
        )

    # All-decode, deep context: the regime `flydsl_unified_attention` routes to
    # split-K (see its tier-selection comment).
    query_lens, kv_lens = [1] * 8, [16384] * 8
    q, k, v, out, cu_q, seqused_k, bt, q_ds, k_ds, v_ds = _build(
        query_lens, kv_lens, 64, 4
    )
    out = out.to(torch.float16)
    uak._get_kernel.cache_clear()
    with mock.patch.object(uak, "_get_kernel", spy):
        got = ua.unified_attention(
            q=q,
            k=k,
            v=v,
            out=out,
            cu_seqlens_q=cu_q,
            max_seqlen_q=max(query_lens),
            seqused_k=seqused_k,
            max_seqlen_k=max(kv_lens),
            softmax_scale=SCALE,
            causal=True,
            window_size=(-1, -1),
            block_table=bt,
            softcap=0,
            q_descale=q_ds,
            k_descale=k_ds,
            v_descale=v_ds,
        )
    assert seen.get("num_kv_splits", 1) > 1, (
        "expected this all-decode/16384-ctx shape to route through split-K; "
        "the cross-feature combination under test did not fire"
    )
    want = ref_paged_attn(
        query=q,
        key_cache=k,
        value_cache=v,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=bt,
        scale=SCALE,
        out_dtype=torch.float16,
        q_descale=q_ds,
        k_descale=k_ds,
        v_descale=v_ds,
    )
    _assert_close(got.float(), want.float().reshape(got.shape))


def test_padded_q_stride_declines_to_triton():
    """`_strides_ok` must decline a padded Q/O outer stride (e.g.
    ``q.stride(0) != num_query_heads * head_size``, even when ``q.stride(1)
    == head_size`` and ``q.stride(2) == 1``).

    `_run_compiled` launches on `q.reshape(-1)` / `out.reshape(-1)`, and those
    bake the reshaped tensor's full memref into the FlyDSL kernel cache
    signature. For a genuinely padded (non-flattenable) layout, `reshape(-1)`
    silently returns a COPY rather than a view: the kernel would write into
    that copy and the caller's real `out` buffer would stay untouched. So the
    only safe choice is to decline padded layouts to the Triton path, which
    handles them correctly. Build a Q/O pair with a few unused head slots per
    row (a padded, non-flattenable layout) and confirm it both fails
    `_supported` and routes to Triton instead of FlyDSL.

    Regression guard for the padded-layout silent-corruption fix: this must
    fail `_supported` and never reach the FlyDSL launch.
    """
    import aiter.ops.flydsl.unified_attention_kernels as uak
    from aiter.ops.flydsl.unified_attention_kernels import _supported

    query_lens, kv_lens = [256], [256]
    num_heads, num_kv_heads = 64, 4
    q, k, v, out, cu_q, seqused_k, bt, q_ds, k_ds, v_ds = _build(
        query_lens, kv_lens, num_heads, num_kv_heads
    )

    # A handful of unused head slots per row -- q.stride(0) is now larger than
    # num_heads * HEAD_DIM while stride(1)/stride(2) stay exactly what
    # _strides_ok's per-row checks require, so the slice is a real
    # (non-contiguous, non-flattenable) view.
    extra_heads = 3
    total_q = q.shape[0]
    q_buf = torch.zeros(
        total_q, num_heads + extra_heads, HEAD_DIM, device=q.device, dtype=q.dtype
    )
    q_buf[:, :num_heads, :] = q
    q_padded = q_buf[:, :num_heads, :]
    out_buf = torch.empty(
        total_q, num_heads + extra_heads, HEAD_DIM, device=out.device, dtype=out.dtype
    )
    out_padded = out_buf[:, :num_heads, :]

    assert q_padded.stride(0) > num_heads * HEAD_DIM, "not actually padded"
    assert q_padded.stride(0) == out_padded.stride(0), "Q/O must share the stride"
    with pytest.raises(RuntimeError):
        q_padded.view(-1)  # confirms this layout is genuinely non-flattenable

    assert not _supported(
        q=q_padded,
        k=k,
        v=v,
        out=out_padded,
        cu_seqlens_q=cu_q,
        seqused_k=seqused_k,
        max_seqlen_k=max(kv_lens),
        causal=True,
        window_size=(-1, -1),
        block_table=bt,
        softcap=0,
        q_descale=q_ds,
        k_descale=k_ds,
        v_descale=v_ds,
        num_kv_heads=num_kv_heads,
        block_size=PAGE,
        num_queries_per_kv=num_heads // num_kv_heads,
        num_seqs=len(query_lens),
        q_scales=None,
        alibi_slopes=None,
        output_scale=None,
        qq_bias=None,
        sinks=None,
        shuffled_kv_cache=False,
        skip_reduce=False,
    ), "a padded, non-flattenable outer stride must be declined by _strides_ok"

    real = uak.flydsl_unified_attention
    seen = {}

    def spy(*a, **kw):
        r = real(*a, **kw)
        seen["served"] = r is not None
        return r

    with mock.patch.object(uak, "flydsl_unified_attention", spy):
        got = ua.unified_attention(
            q=q_padded,
            k=k,
            v=v,
            out=out_padded,
            cu_seqlens_q=cu_q,
            max_seqlen_q=max(query_lens),
            seqused_k=seqused_k,
            max_seqlen_k=max(kv_lens),
            softmax_scale=SCALE,
            causal=True,
            window_size=(-1, -1),
            block_table=bt,
            softcap=0,
            q_descale=q_ds,
            k_descale=k_ds,
            v_descale=v_ds,
        )
    assert (
        seen.get("served") is False
    ), "a padded-stride Q/O pair must decline to Triton, not route to FlyDSL"
    want = ref_paged_attn(
        query=q_padded,
        key_cache=k,
        value_cache=v,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=bt,
        scale=SCALE,
        out_dtype=out.dtype,
        q_descale=q_ds,
        k_descale=k_ds,
        v_descale=v_ds,
    )
    _assert_close(got.float(), want.float().reshape(got.shape))


# --- 4. the gate predicate, without a GPU ------------------------------------


def test_predicate_declines_unsupported_geometry():
    """Cheap structural checks of _supported that need no device. These would
    otherwise only be covered by cases that are awkward to build for real."""
    from aiter.ops.flydsl.unified_attention_kernels import _supported

    q, k, v, out, cu_q, seqused_k, bt, q_ds, k_ds, v_ds = _build([256], [256], 64, 4)
    base = {
        "q": q,
        "k": k,
        "v": v,
        "out": out,
        "cu_seqlens_q": cu_q,
        "seqused_k": seqused_k,
        "max_seqlen_k": 256,
        "causal": True,
        "window_size": (-1, -1),
        "block_table": bt,
        "softcap": 0,
        "q_descale": q_ds,
        "k_descale": k_ds,
        "v_descale": v_ds,
        "num_kv_heads": 4,
        "block_size": PAGE,
        "num_queries_per_kv": 16,
        "num_seqs": 1,
        "q_scales": None,
        "alibi_slopes": None,
        "output_scale": None,
        "qq_bias": None,
        "sinks": None,
        "shuffled_kv_cache": False,
        "skip_reduce": False,
    }
    assert _supported(**base), "the baseline config should be supported"

    # Page size is structural (BLOCK_N), not a builder knob.
    for bs in (16, 32, 128):
        assert not _supported(**{**base, "block_size": bs})
    # The block-table LDS window caps KV at 2048 pages = 131072 tokens.
    assert not _supported(**{**base, "max_seqlen_k": 131072 + 1})
    assert _supported(**{**base, "max_seqlen_k": 131072})
    # A GQA group that does not divide block_m cannot pack into M.
    assert not _supported(**{**base, "num_queries_per_kv": 3})
    # Triton-only layouts and flags.
    assert not _supported(**{**base, "block_table": None})
    assert not _supported(**{**base, "shuffled_kv_cache": True})
    assert not _supported(**{**base, "skip_reduce": True})
    # causal is intentionally unconstrained here -- non-causal is served (see
    # test_non_causal_routes_and_agrees below); the assert above only checks
    # structural geometry, not the mask mode.
    assert _supported(**{**base, "causal": False})
    # Descales are mandatory on the fp8 path.
    assert not _supported(**{**base, "q_descale": None})


def test_predicate_declines_wrong_dtypes():
    from aiter.ops.flydsl.unified_attention_kernels import _supported

    q, k, v, out, cu_q, seqused_k, bt, q_ds, k_ds, v_ds = _build([256], [256], 64, 4)
    base = {
        "q": q,
        "k": k,
        "v": v,
        "out": out,
        "cu_seqlens_q": cu_q,
        "seqused_k": seqused_k,
        "max_seqlen_k": 256,
        "causal": True,
        "window_size": (-1, -1),
        "block_table": bt,
        "softcap": 0,
        "q_descale": q_ds,
        "k_descale": k_ds,
        "v_descale": v_ds,
        "num_kv_heads": 4,
        "block_size": PAGE,
        "num_queries_per_kv": 16,
        "num_seqs": 1,
        "q_scales": None,
        "alibi_slopes": None,
        "output_scale": None,
        "qq_bias": None,
        "sinks": None,
        "shuffled_kv_cache": False,
        "skip_reduce": False,
    }
    assert not _supported(**{**base, "q": q.to(torch.bfloat16)})
    # fp16 output is supported now (out.dtype in (bfloat16, float16)); see
    # test_fp16_output_routes_and_agrees below.
    assert _supported(**{**base, "out": out.to(torch.float16)})
    assert not _supported(**{**base, "block_table": bt.to(torch.int64)})
