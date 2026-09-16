# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton  # noqa: F401  # isort: skip  # Must precede torch on this ROCm environment.

import math
from unittest import mock

import pytest
import torch

import aiter
from aiter.ops import mha as mha_ops
from aiter.ops.mha import (
    _fmha_v3_varlen_splitkv_fwd,
    flash_attn_varlen_func,
    fmha_v3_varlen_fwd,
)

_CUDA_AVAILABLE = torch.cuda.is_available()
_DEVICE_NAME = torch.cuda.get_device_name() if _CUDA_AVAILABLE else ""
pytestmark = pytest.mark.skipif(
    not _CUDA_AVAILABLE
    or aiter.get_gfx() != "gfx942"
    or not any(device in _DEVICE_NAME for device in ("MI300X", "MI325X")),
    reason="split-KV ASM is validated only on gfx942 MI300X/MI325X",
)


def _make_packed(sq: int, sk: int, h: int, seed: int = 0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(sq, h, 192, dtype=torch.bfloat16, generator=generator).cuda()
    k = torch.randn(sk, h, 192, dtype=torch.bfloat16, generator=generator).cuda()
    v = torch.randn(sk, h, 128, dtype=torch.bfloat16, generator=generator).cuda()
    cu_q = torch.tensor([0, sq], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, sk], dtype=torch.int32, device="cuda")
    return q, k, v, cu_q, cu_k


def _run_v3(q, k, v, cu_q, cu_k, scale, num_splits, *, return_lse=False):
    out, lse, _, _ = _fmha_v3_varlen_splitkv_fwd(
        q,
        k,
        v,
        cu_q,
        cu_k,
        q.shape[0],
        k.shape[0],
        scale,
        return_lse,
        num_splits=num_splits,
    )
    return (out, lse) if return_lse else out


def _production_asm(q, k, v, cu_q, cu_k, scale, *, return_lse=False):
    return _run_v3(q, k, v, cu_q, cu_k, scale, 1, return_lse=return_lse)


def _split_asm(q, k, v, cu_q, cu_k, scale, num_splits=3, *, return_lse=False):
    return _run_v3(q, k, v, cu_q, cu_k, scale, num_splits, return_lse=return_lse)


def _public_asm(q, k, v, cu_q, cu_k, scale, *, return_lse=False, out=None, plan=None):
    # plan=None is the no-CSV path: C++ num_splits=0 auto-select.
    with mock.patch.object(mha_ops, "_get_mha_fwd_tuned_plan", return_value=plan):
        result = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            q.shape[0],
            k.shape[0],
            softmax_scale=scale,
            causal=False,
            return_lse=return_lse,
            out=out,
        )
    return result


def _cosine_difference(reference: torch.Tensor, actual: torch.Tensor) -> float:
    ref = reference.double()
    got = actual.double()
    return 1.0 - 2.0 * (ref * got).sum().item() / max(
        (ref.square() + got.square()).sum().item(), 1e-12
    )


def _assert_close(reference: torch.Tensor, actual: torch.Tensor) -> None:
    assert _cosine_difference(reference, actual) < 1e-4
    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)


def test_splitkv_one_matches_unsplit_kernel():
    sq, sk, h = 129, 511, 4
    q, k, v, cu_q, cu_k = _make_packed(sq, sk, h, seed=17)
    scale = 1.0 / math.sqrt(192)
    reference, reference_lse, _, _ = fmha_v3_varlen_fwd(
        q,
        k,
        v,
        cu_q,
        cu_k,
        sq,
        sk,
        0,
        0.0,
        scale,
        0.0,
        False,
        False,
        -1,
        -1,
        True,
        False,
        1,
    )
    actual, actual_lse = _production_asm(q, k, v, cu_q, cu_k, scale, return_lse=True)
    assert torch.equal(actual, reference)
    assert torch.equal(actual_lse, reference_lse)


@pytest.mark.parametrize(
    "sq,sk",
    [
        (1, 96),
        (31, 129),
        (32, 160),
        (33, 191),
        (127, 192),
        (128, 193),
        (129, 224),
        (257, 511),
    ],
)
def test_splitkv_boundaries(sq, sk):
    q, k, v, cu_q, cu_k = _make_packed(sq, sk, 4, seed=sq + sk)
    scale = 1.0 / math.sqrt(192)
    reference = _production_asm(q, k, v, cu_q, cu_k, scale)
    actual = _split_asm(q, k, v, cu_q, cu_k, scale)
    assert torch.isfinite(actual).all()
    _assert_close(reference, actual)


@pytest.mark.parametrize("num_splits", range(2, 9))
def test_splitkv_counts(num_splits):
    sq, sk, h = 129, 2048, 4
    q, k, v, cu_q, cu_k = _make_packed(sq, sk, h, seed=100 + num_splits)
    scale = 1.0 / math.sqrt(192)
    reference, reference_lse = _production_asm(
        q, k, v, cu_q, cu_k, scale, return_lse=True
    )
    actual, actual_lse = _split_asm(
        q, k, v, cu_q, cu_k, scale, num_splits, return_lse=True
    )
    _assert_close(reference, actual)
    torch.testing.assert_close(actual_lse, reference_lse, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize(
    "sq,sk,h", [(4096, 8192, 12), (3969, 8192, 12), (4096, 131072, 12)]
)
def test_public_dispatch_uses_cpp_auto_split3_on_long_kv(sq, sk, h):
    q, k, v, cu_q, cu_k = _make_packed(sq, sk, h, seed=sk + sq)
    scale = 1.0 / math.sqrt(192)
    split1 = _production_asm(q, k, v, cu_q, cu_k, scale)
    split3, split3_lse = _split_asm(q, k, v, cu_q, cu_k, scale, 3, return_lse=True)
    actual, actual_lse = _public_asm(q, k, v, cu_q, cu_k, scale, return_lse=True)
    assert torch.equal(actual, split3)
    assert torch.equal(actual_lse, split3_lse)
    assert not torch.equal(actual, split1)
    _assert_close(split1, actual)


def test_public_dispatch_csv_override_uses_explicit_split():
    sq, sk, h = 4096, 8192, 12
    q, k, v, cu_q, cu_k = _make_packed(sq, sk, h, seed=42)
    scale = 1.0 / math.sqrt(192)
    split2 = _split_asm(q, k, v, cu_q, cu_k, scale, 2)
    split3 = _split_asm(q, k, v, cu_q, cu_k, scale, 3)
    actual = _public_asm(
        q,
        k,
        v,
        cu_q,
        cu_k,
        scale,
        plan={"backend": "asm_v3", "num_splits": 2, "backend_config": None},
    )
    assert torch.equal(actual, split2)
    assert not torch.equal(actual, split3)


def test_public_dispatch_keeps_unsplit_outside_heuristic():
    scale = 1.0 / math.sqrt(192)
    # Sk=8191 is the last length below 256 full KV tiles. High Q occupancy
    # (24 heads * 32 Q tiles > 2*304 CUs) also stays unsplit.
    cases = [
        (4096, 8191, 12),
        (4096, 8192, 24),
    ]
    for sq, sk, h in cases:
        q, k, v, cu_q, cu_k = _make_packed(sq, sk, h, seed=sq + sk + h)
        split1 = _production_asm(q, k, v, cu_q, cu_k, scale)
        actual = _public_asm(q, k, v, cu_q, cu_k, scale)
        assert torch.equal(actual, split1), (sq, sk, h)


def test_forced_splitkv_empty_k_matches_unsplit():
    sq, h = 129, 4
    q = torch.randn(sq, h, 192, dtype=torch.bfloat16, device="cuda")
    k = torch.empty(0, h, 192, dtype=torch.bfloat16, device="cuda")
    v = torch.empty(0, h, 128, dtype=torch.bfloat16, device="cuda")
    cu_q = torch.tensor([0, sq], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
    scale = 1.0 / math.sqrt(192)
    reference = _production_asm(q, k, v, cu_q, cu_k, scale)
    actual = _split_asm(q, k, v, cu_q, cu_k, scale, 3)
    assert torch.equal(actual, reference)


def test_public_splitkv_fullgraph_compile():
    sq, sk, h = 4096, 8192, 12
    q, k, v, cu_q, cu_k = _make_packed(sq, sk, h, seed=27)
    scale = 1.0 / math.sqrt(192)

    def call(q, k, v):
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            sq,
            sk,
            softmax_scale=scale,
            causal=False,
            return_lse=True,
        )

    # This shape has no CSV row, so public dispatch uses C++ auto-select.
    # Do not mock the lookup: torch.compile cannot trace unittest.mock.
    eager = call(q, k, v)
    compiled = torch.compile(call, fullgraph=True)(q, k, v)
    _assert_close(eager[0], compiled[0])
    torch.testing.assert_close(compiled[1], eager[1], rtol=2e-4, atol=2e-4)


def test_out_buffer_uses_public_dispatch():
    sq, sk, h = 4096, 8192, 12
    q, k, v, cu_q, cu_k = _make_packed(sq, sk, h, seed=28)
    scale = 1.0 / math.sqrt(192)
    out = torch.empty((sq, h, 128), dtype=torch.bfloat16, device="cuda")
    expected = _split_asm(q, k, v, cu_q, cu_k, scale, 3)
    actual = _public_asm(q, k, v, cu_q, cu_k, scale, out=out)
    assert actual.data_ptr() == out.data_ptr()
    assert torch.equal(actual, expected)


def test_splitkv_operator_torch_compile():
    sq, sk, h = 129, 2048, 4
    q, k, v, cu_q, cu_k = _make_packed(sq, sk, h, seed=25)
    scale = 1.0 / math.sqrt(192)

    def call(q, k, v):
        return _fmha_v3_varlen_splitkv_fwd(q, k, v, cu_q, cu_k, sq, sk, scale, True, 3)

    eager = call(q, k, v)
    compiled = torch.compile(call, fullgraph=True)(q, k, v)
    _assert_close(eager[0], compiled[0])
    torch.testing.assert_close(compiled[1], eager[1], rtol=2e-4, atol=2e-4)


def test_public_splitkv_cuda_graph_replay():
    sq, sk, h = 4096, 8192, 12
    q, k, v, cu_q, cu_k = _make_packed(sq, sk, h, seed=26)
    scale = 1.0 / math.sqrt(192)
    reference = _production_asm(q, k, v, cu_q, cu_k, scale)

    with mock.patch.object(mha_ops, "_get_mha_fwd_tuned_plan", return_value=None):
        for _ in range(3):
            flash_attn_varlen_func(
                q, k, v, cu_q, cu_k, sq, sk, softmax_scale=scale, causal=False
            )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = flash_attn_varlen_func(
                q, k, v, cu_q, cu_k, sq, sk, softmax_scale=scale, causal=False
            )
    graph.replay()
    first = captured.clone()
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(captured, first)
    _assert_close(reference, captured)


def test_splitkv_rejects_empty_final_partition():
    sq, sk, h = 129, 511, 4
    q, k, v, cu_q, cu_k = _make_packed(sq, sk, h, seed=21)
    with pytest.raises(RuntimeError, match="empty final KV partition"):
        _split_asm(q, k, v, cu_q, cu_k, 1.0 / math.sqrt(192), num_splits=5)


def test_splitkv_lse_and_determinism():
    sq, sk, h = 257, 511, 12
    q, k, v, cu_q, cu_k = _make_packed(sq, sk, h, seed=19)
    scale = 0.125
    reference, reference_lse = _production_asm(
        q, k, v, cu_q, cu_k, scale, return_lse=True
    )
    actual, actual_lse = _split_asm(q, k, v, cu_q, cu_k, scale, return_lse=True)
    _assert_close(reference, actual)
    torch.testing.assert_close(actual_lse, reference_lse, rtol=2e-4, atol=2e-4)
    for _ in range(100):
        repeat = _split_asm(q, k, v, cu_q, cu_k, scale)
        assert torch.equal(actual, repeat)
