# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The a16w16 ASM GEMM addresses A, B and the output with 32-bit byte offsets.

An operand of 2**32 bytes or more wrapped silently (ROCm/aiter#4743). The
Python entry now splits M so every launch stays in range. The CPU tests check
the split on meta tensors, so they need no GPU and no memory; the GPU test
compares a whole oversized output against torch and is opt-in because it needs
roughly 25 GiB.
"""

import os

import pytest
import torch

import aiter.ops.gemm_op_a16w16 as op
from aiter.ops.gemm_op_a16w16 import ASM_BUFFER_LIMIT_BYTES, asm_m_chunk_rows

LIMIT = ASM_BUFFER_LIMIT_BYTES
# Row padding of the kernel's tileM (all shipped values divide 3840 or are <= 256).
TILE_MS = (32, 48, 64, 80, 96, 128, 160, 256)


def _padded(rows, tile_m):
    return (rows + tile_m - 1) // tile_m * tile_m


def test_in_range_shapes_are_not_split():
    # K=12288 bf16: M=174000 is the largest size from the issue that was correct.
    assert asm_m_chunk_rows(174_000, 12288 * 2, 8192 * 2, 8192 * 12288 * 2) is None
    assert asm_m_chunk_rows(4096, 4096 * 2, 4096 * 2, 4096 * 4096 * 2) is None
    assert asm_m_chunk_rows(1, 64 * 2, 64 * 4, 64 * 64 * 2) is None


def test_boundary_includes_tile_padding():
    row = 12288 * 2
    # Unpadded M*row is below 2**32 but padding to a 256-row tile crosses it.
    m = LIMIT // row - 10
    assert m * row < LIMIT
    assert asm_m_chunk_rows(m, row, 64, 64 * 64 * 2) is not None


@pytest.mark.parametrize(
    "m,a_row,out_row,b_bytes",
    [
        (175_000, 12288 * 2, 8192 * 2, 8192 * 12288 * 2),  # A over the limit
        (360_360, 6144 * 2, 8192 * 4, 8192 * 6144 * 2),  # fp32 output over the limit
        (300_000, 1024 * 2, 16384 * 2, 16384 * 1024 * 2),  # only the output over
        (360_000, 12288 * 2, 8192 * 2, 8192 * 12288 * 2),
        (40_000_000, 64 * 2, 64 * 2, 64 * 64 * 2),  # tiny rows, huge M
    ],
)
def test_chunks_keep_every_launch_addressable(m, a_row, out_row, b_bytes):
    rows = asm_m_chunk_rows(m, a_row, out_row, b_bytes)
    assert rows is not None and 0 < rows < m
    assert rows % 3840 == 0
    for tile_m in TILE_MS:
        last = m % rows or rows
        for n_rows in (rows, last):
            padded = _padded(n_rows, tile_m)
            assert padded * a_row < LIMIT
            assert padded * out_row < LIMIT


def test_unsplittable_operands_raise():
    with pytest.raises(ValueError, match="B spans"):
        asm_m_chunk_rows(256, 64 * 2, 65536 * 2, LIMIT)
    with pytest.raises(ValueError, match="too large"):
        asm_m_chunk_rows(1 << 20, 2 * (LIMIT // 1024), 64, 64 * 64 * 2)


@pytest.fixture
def recorded_launches(monkeypatch):
    calls = []

    def fake(A, B, out, sema, bias, splitK, kernelName, bpreshuffle):
        calls.append((A, B, out, sema, bias, splitK, kernelName, bpreshuffle))

    monkeypatch.setattr(op, "_gemm_a16w16_asm", fake)
    monkeypatch.setattr(
        op, "get_semaphore_workspace", lambda device: torch.empty(0, device="meta")
    )
    return calls


def test_wrapper_covers_all_rows_once(recorded_launches):
    m, k, n = 360_000, 12288, 8192
    a = torch.empty(m, k, dtype=torch.bfloat16, device="meta")
    w = torch.empty(n, k, dtype=torch.bfloat16, device="meta")
    out = torch.empty(m, n, dtype=torch.bfloat16, device="meta")
    bias = torch.empty(n, dtype=torch.float32, device="meta")

    ret = op.gemm_a16w16_asm(a, w, out, bias, 1, "kernel", True)

    assert ret is out
    start = 0
    for ca, cb, co, _, cbias, split_k, name, preshuffle in recorded_launches:
        assert ca.shape[1] == k and co.shape[1] == n
        assert ca.shape[0] == co.shape[0]
        assert ca.storage_offset() == start * k
        assert co.storage_offset() == start * n
        assert cb is w and cbias is bias
        assert (split_k, name, preshuffle) == (1, "kernel", True)
        assert ca.shape[0] * k * 2 < LIMIT and ca.shape[0] * n * 2 < LIMIT
        start += ca.shape[0]
    assert start == m
    assert len(recorded_launches) > 1


def test_wrapper_keeps_in_range_call_unchanged(recorded_launches):
    a = torch.empty(4096, 4096, dtype=torch.bfloat16, device="meta")
    w = torch.empty(4096, 4096, dtype=torch.bfloat16, device="meta")
    out = torch.empty(4096, 4096, dtype=torch.bfloat16, device="meta")

    op.gemm_a16w16_asm(a, w, out)

    assert len(recorded_launches) == 1
    ca, cb, co = recorded_launches[0][:3]
    assert ca is a and cb is w and co is out


def test_wrapper_rejects_oversized_b(recorded_launches):
    n, k = 1 << 17, 1 << 15  # 2**32 elements, 2**33 bytes
    a = torch.empty(256, k, dtype=torch.bfloat16, device="meta")
    w = torch.empty(n, k, dtype=torch.bfloat16, device="meta")
    out = torch.empty(256, n, dtype=torch.bfloat16, device="meta")
    with pytest.raises(ValueError, match="B spans"):
        op.gemm_a16w16_asm(a, w, out)
    assert not recorded_launches


def _gpu_ready(min_free_gib):
    if not torch.cuda.is_available():
        return False
    if os.environ.get("AITER_RUN_LARGE_GEMM_TESTS") != "1":
        return False
    free, _ = torch.cuda.mem_get_info()
    return free >= min_free_gib * 2**30


def _check_whole_output(a, w, out):
    """Compare every row, in slices so the reference stays small."""
    step = 8192
    for i in range(0, a.shape[0], step):
        ref = torch.nn.functional.linear(a[i : i + step], w)
        torch.testing.assert_close(
            out[i : i + step].float(), ref.float(), rtol=3e-2, atol=3e-2
        )


@pytest.mark.skipif(
    not _gpu_ready(25),
    reason="needs a GPU with >=25 GiB free and AITER_RUN_LARGE_GEMM_TESTS=1",
)
@pytest.mark.parametrize(
    "m,k,n",
    [
        (175_000, 12288, 8192),  # A over 2**32 bytes (issue #4743)
        (300_000, 1024, 8192 * 2),  # only the output over 2**32 bytes
    ],
)
def test_gpu_oversized_operands_match_torch(m, k, n):
    from aiter import gemm_a16w16_asm

    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    gemm_a16w16_asm(a, w, out)
    torch.cuda.synchronize()
    _check_whole_output(a, w, out)


@pytest.mark.skipif(
    not _gpu_ready(12),
    reason="needs a GPU with >=12 GiB free and AITER_RUN_LARGE_GEMM_TESTS=1",
)
def test_gpu_direct_launcher_rejects_oversized_operands():
    # Bypass the Python split: the C++ launcher must fail instead of wrapping.
    m, k, n = 175_000, 12288, 64
    a = torch.empty(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.empty(n, k, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    sema = op.get_semaphore_workspace(out.device)
    with pytest.raises(RuntimeError, match="32-bit buffer addressing"):
        op._gemm_a16w16_asm(a, w, out, sema)
