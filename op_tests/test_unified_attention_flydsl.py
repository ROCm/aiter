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

import aiter.ops.triton.attention.unified_attention as ua  # noqa: E402
from aiter.ops.flydsl.utils import is_flydsl_available  # noqa: E402
from op_tests.flydsl_tests.test_flydsl_unified_attention import _build  # noqa: E402

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
    kwargs = dict(
        q=q, k=k, v=v, out=out,
        cu_seqlens_q=cu_q, max_seqlen_q=max(query_lens),
        seqused_k=seqused_k, max_seqlen_k=max(kv_lens),
        softmax_scale=SCALE, causal=True, window_size=(-1, -1),
        block_table=bt, softcap=0,
        q_descale=q_ds, k_descale=k_ds, v_descale=v_ds,
    )
    kwargs.update(over)
    return ua.unified_attention(**kwargs)


def _triton_only(*args, **kwargs):
    """Run the same call with the FlyDSL backend disabled."""
    with mock.patch.object(ua, "_FLYDSL_UNIFIED_ATTN_ARCH", False):
        return _call(*args, **kwargs)


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
        ([256], [256]),                      # single prefill
        ([1] * 8, [4096] * 8),               # pure decode
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
        "softcap": dict(softcap=30.0),
        "sliding_window": dict(window_size=(256, 0)),
        # block_table=None is covered in the predicate test instead: Triton's
        # own path dereferences it unconditionally (unified_attention.py:637),
        # so there is no fall-through target to compare against.
        "alibi_slopes": dict(alibi_slopes=torch.ones(64, device="cuda", dtype=torch.float32)),
        "sinks": dict(sinks=torch.zeros(64, device="cuda", dtype=torch.float32)),
        "output_scale": dict(output_scale=torch.ones(1, device="cuda", dtype=torch.float32)),
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


# --- 4. the gate predicate, without a GPU ------------------------------------


def test_predicate_declines_unsupported_geometry():
    """Cheap structural checks of _supported that need no device. These would
    otherwise only be covered by cases that are awkward to build for real."""
    from aiter.ops.flydsl.unified_attention_kernels import _supported

    q, k, v, out, cu_q, seqused_k, bt, q_ds, k_ds, v_ds = _build([256], [256], 64, 4)
    base = dict(
        q=q, k=k, v=v, out=out, cu_seqlens_q=cu_q, seqused_k=seqused_k,
        max_seqlen_k=256, causal=True, window_size=(-1, -1), block_table=bt,
        softcap=0, q_descale=q_ds, k_descale=k_ds, v_descale=v_ds,
        num_kv_heads=4, block_size=PAGE, num_queries_per_kv=16, num_seqs=1,
        q_scales=None, alibi_slopes=None, output_scale=None, qq_bias=None,
        sinks=None, shuffled_kv_cache=False, skip_reduce=False,
    )
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
    assert not _supported(**{**base, "causal": False})
    # Descales are mandatory on the fp8 path.
    assert not _supported(**{**base, "q_descale": None})


def test_predicate_declines_wrong_dtypes():
    from aiter.ops.flydsl.unified_attention_kernels import _supported

    q, k, v, out, cu_q, seqused_k, bt, q_ds, k_ds, v_ds = _build([256], [256], 64, 4)
    base = dict(
        q=q, k=k, v=v, out=out, cu_seqlens_q=cu_q, seqused_k=seqused_k,
        max_seqlen_k=256, causal=True, window_size=(-1, -1), block_table=bt,
        softcap=0, q_descale=q_ds, k_descale=k_ds, v_descale=v_ds,
        num_kv_heads=4, block_size=PAGE, num_queries_per_kv=16, num_seqs=1,
        q_scales=None, alibi_slopes=None, output_scale=None, qq_bias=None,
        sinks=None, shuffled_kv_cache=False, skip_reduce=False,
    )
    assert not _supported(**{**base, "q": q.to(torch.bfloat16)})
    assert not _supported(**{**base, "out": out.to(torch.float16)})
    assert not _supported(**{**base, "block_table": bt.to(torch.int64)})
