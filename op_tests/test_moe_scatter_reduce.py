# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness and bounds safety for ``mxfp4_moe_scatter_reduce``.

MiniMax-M3 C40 (TP4, H=6144, TOPK=5, M=32768, BM=128) faulted in
``scatter_reduce_kernel_impl`` with ``MEMORY_APERTURE_VIOLATION`` when a
corrupt / uninitialized ``reverse_sorted`` index was nontemporal-loaded past
``flat_out``. The kernel must:

  * match a torch reference on legal reverse maps;
  * skip ``sorted_pos < 0`` (EP) and ``sorted_pos >= max_sorted`` (corrupt)
    without a GPU fault, including the BM>=128 NT path.

    pytest op_tests/test_moe_scatter_reduce.py -q
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available() or torch.version.hip is None:
    pytest.skip("ROCm GPU required", allow_module_level=True)

aiter = pytest.importorskip("aiter")

pytest.importorskip("aiter.ops.moe_mxfp4_aux", reason="mxfp4 MoE aux module")

# MiniMax-M3 TP4 shape from gen_instances.SHAPES / the C40 fault.
M3_H = 6144
M3_TOPK = 5
# Keep unit-test M small; coverage of the NT path is via MB>=128, not M.
M3_MS = (1, 3, 8, 64)


def _reference(flat_out, reverse_sorted, sorted_weights, M, topk, max_sorted):
    """fp32 gather-reduce matching the kernel's skip rules."""
    D = flat_out.shape[1]
    out = torch.zeros(M, D, dtype=torch.float32, device=flat_out.device)
    rs = reverse_sorted.view(M, topk)
    for t in range(M):
        acc = torch.zeros(D, dtype=torch.float32, device=flat_out.device)
        for k in range(topk):
            sp = int(rs[t, k].item())
            if sp < 0 or sp >= max_sorted:
                continue
            acc = acc + flat_out[sp].float() * float(sorted_weights[sp].item())
        out[t] = acc
    return out


def _assert_bf16_matches(out, ref, *, exact=False):
    assert torch.isfinite(out).all(), "output has non-finite values"
    # Compare at the output precision. Weighted FP32 FMA can cross a BF16
    # rounding midpoint relative to separate multiply/add, so allow one ULP.
    # Unit-weight sums use the same FP32 additions and must match exactly.
    torch.testing.assert_close(
        out,
        ref.to(out.dtype),
        rtol=0 if exact else torch.finfo(out.dtype).eps,
        atol=0 if exact else 1e-5,
    )


def _raise_pending_launch_error():
    try:
        torch.ones(1, device="cuda").add_(1)
    except RuntimeError as exc:
        pytest.fail(f"a kernel launch failed silently: {exc}")


def _run_scatter(flat_out, reverse_sorted, sorted_weights, out, topk, mb):
    aiter.mxfp4_moe_scatter_reduce(
        flat_out=flat_out,
        reverse_sorted=reverse_sorted,
        sorted_weights=sorted_weights,
        out=out,
        NE=129,  # dispatch key ignores NE; H/TOPK/NT matter
        TOPK=topk,
        D_HIDDEN=flat_out.shape[1],
        MB=mb,
    )
    _raise_pending_launch_error()
    torch.cuda.synchronize()


@pytest.mark.parametrize("mb", (16, 64, 128), ids=("nt0_bm16", "nt0_bm64", "nt1_bm128"))
@pytest.mark.parametrize("M", M3_MS)
def test_scatter_reduce_matches_reference(M, mb):
    torch.manual_seed(M * 17 + mb)
    D, topk = M3_H, M3_TOPK
    # Enough staging rows for a dense (token, topk) map with BM padding headroom.
    max_sorted = ((M * topk + mb - 1) // mb) * mb + mb
    flat = torch.randn(max_sorted, D, dtype=torch.bfloat16, device="cuda")
    weights = torch.rand(max_sorted, dtype=torch.float32, device="cuda")
    # Identity-ish reverse map: token t's k-th route -> row t*topk + k.
    rs = torch.arange(M * topk, dtype=torch.int32, device="cuda")
    out = torch.full((M, D), float("nan"), dtype=torch.bfloat16, device="cuda")

    _run_scatter(flat, rs, weights, out, topk, mb)
    ref = _reference(flat, rs, weights, M, topk, max_sorted)
    _assert_bf16_matches(out, ref)


@pytest.mark.parametrize("mb", (16, 128), ids=("nt0", "nt1"))
def test_scatter_reduce_skips_negative_and_oob_indices(mb):
    """Corrupt reverse map must not aperture-fault; skipped routes contribute 0."""
    torch.manual_seed(mb)
    M, D, topk = 4, M3_H, M3_TOPK
    max_sorted = 32
    flat = torch.randn(max_sorted, D, dtype=torch.bfloat16, device="cuda")
    weights = torch.ones(max_sorted, dtype=torch.float32, device="cuda")
    # Mix of legal, EP-skip (-1), and past-the-end indices.
    rs = torch.tensor(
        [
            0,
            -1,
            max_sorted,  # == max_sorted: must skip
            max_sorted + 7,  # well past flat_out
            1,
            -3,
            2**20,  # huge garbage
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
        ],
        dtype=torch.int32,
        device="cuda",
    )
    assert rs.numel() == M * topk
    out = torch.full((M, D), float("nan"), dtype=torch.bfloat16, device="cuda")

    _run_scatter(flat, rs, weights, out, topk, mb)
    ref = _reference(flat, rs, weights, M, topk, max_sorted)
    _assert_bf16_matches(out, ref, exact=True)


@pytest.mark.parametrize("mb", (128,), ids=("nt1_bm128",))
def test_scatter_reduce_m3_warmup_shape_smoke(mb):
    """M=256 stand-in for the 32K warmup: same H/TOPK/NT, tractable memory.

    Full M=32768 needs ~2 GB of staging alone; this covers the dispatch key
    and NT path the C40 fault used without OOM on a shared CI GPU.
    """
    torch.manual_seed(0)
    M, D, topk = 256, M3_H, M3_TOPK
    max_sorted = ((M * topk + mb - 1) // mb) * mb
    flat = torch.randn(max_sorted, D, dtype=torch.bfloat16, device="cuda") * 0.01
    weights = torch.rand(max_sorted, dtype=torch.float32, device="cuda")
    rs = torch.arange(M * topk, dtype=torch.int32, device="cuda") % max_sorted
    out = torch.empty(M, D, dtype=torch.bfloat16, device="cuda")
    _run_scatter(flat, rs, weights, out, topk, mb)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("mb", (16, 128), ids=("nt0", "nt1"))
def test_scatter_reduce_mxfp4_bounds_and_reference(mb):
    """Exercise packed FP4 data and varying E8M0 scales, including OOB skips."""
    torch.manual_seed(mb)
    M, rows, D, topk = 6, 32, M3_H, M3_TOPK
    packed = torch.randint(0, 256, (rows, D // 2), dtype=torch.uint8, device="cuda")
    scales = torch.randint(125, 130, (rows, D // 32), dtype=torch.uint8, device="cuda")
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
    )
    decoded = torch.stack((lut[(packed & 15).long()], lut[(packed >> 4).long()]), -1)
    decoded = decoded.reshape(rows, D) * torch.exp2(
        scales.float() - 127
    ).repeat_interleave(32, dim=1)
    weights = torch.rand(rows, device="cuda")
    rs = torch.arange(M * topk, dtype=torch.int32, device="cuda")
    rs[:5] = torch.tensor([-1, -7, rows, rows + 7, 2**20], device="cuda")
    rs[5] = rows - 1
    out = torch.full((M, D), float("nan"), dtype=torch.bfloat16, device="cuda")
    aiter.mxfp4_moe_scatter_reduce_q(
        flat_out_q=packed,
        flat_out_scale=scales,
        reverse_sorted=rs,
        sorted_weights=weights,
        out=out,
        NE=129,
        TOPK=topk,
        D_HIDDEN=D,
        MB=mb,
    )
    _raise_pending_launch_error()
    torch.cuda.synchronize()
    _assert_bf16_matches(out, _reference(decoded, rs, weights, M, topk, rows))
    assert torch.count_nonzero(out[0]) == 0


@pytest.mark.parametrize("mb", (16, 128), ids=("nt0", "nt1"))
def test_scatter_reduce_last_row_and_all_skipped(mb):
    rows, D, topk = 32, M3_H, M3_TOPK
    flat = torch.full((rows, D), float("nan"), dtype=torch.bfloat16, device="cuda")
    flat[-1] = 2
    weights = torch.ones(rows, device="cuda")
    rs = torch.tensor(
        [-1, -7, rows, rows + 7, 2**20, rows - 1, -1, rows, -1, -1],
        dtype=torch.int32,
        device="cuda",
    )
    out = torch.full((2, D), float("nan"), dtype=torch.bfloat16, device="cuda")
    _run_scatter(flat, rs, weights, out, topk, mb)
    torch.testing.assert_close(out[0], torch.zeros_like(out[0]), rtol=0, atol=0)
    torch.testing.assert_close(out[1], torch.full_like(out[1], 2), rtol=0, atol=0)
