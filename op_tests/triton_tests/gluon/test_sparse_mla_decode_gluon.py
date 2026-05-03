# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for `sparse_mla_decode_gluon` (DeepSeek-V4 sparse MLA decode).

Reference math mirrors vllm's `_ref_sparse_attn_decode`
(`vllm/vllm/model_executor/layers/deepseek_v4_attention.py`).

The sparse-gluon kernel is gfx950-only; tests skip elsewhere. Imports are
guarded by `pytest.importorskip` so the file is harmless on platforms
where gluon or aiter are unavailable.
"""

import pytest
import torch

pytest.importorskip("triton.experimental.gluon")
pytest.importorskip("aiter")

import triton  # noqa: E402

# The kernel uses gluon-language layouts (PaddedSharedLayout etc.) whose
# constructor signatures differ between community Triton (3.6.0) and the
# AMD-published amd-triton 3.7+ wheel. JIT-compilation is only validated
# against the AMD wheel; on community Triton the kernel will fail to
# compile. Skip the suite there to avoid noisy failures unrelated to the
# kernel logic.
_TRITON_VER = tuple(int(x) for x in triton.__version__.split("+")[0].split(".")[:2])
if _TRITON_VER < (3, 7):
    pytest.skip(
        f"sparse_mla_decode_gluon requires triton >= 3.7 (amd-triton wheel); "
        f"got triton {triton.__version__}. Install via "
        f"`pip install --extra-index-url "
        f"https://pypi.amd.com/triton/rocm-7.2.0/simple/ amd-triton`.",
        allow_module_level=True,
    )

import aiter.ops.triton.utils._triton.arch_info as arch_info  # noqa: E402

if arch_info.get_arch() != "gfx950":
    pytest.skip(f"sparse_mla_decode_gluon requires gfx950, got {arch_info.get_arch()}", allow_module_level=True)

from aiter.ops.triton.gluon.sparse_mla_decode_gluon import (  # noqa: E402
    sparse_mla_decode_gluon,
)


# Shape constants matching V4 (head_dim_ckv=512, K=V single latent, no rope path).
HEAD_DIM = 512


def _make_inputs(
    *,
    batch: int,
    topk_count: int,
    nhead: int,
    sink: bool,
    invalid_pattern: str,
    kv_pool_min: int = 4096,
    seed: int = 0,
):
    """Build (q, kv, topk_indices, attn_sink, sm_scale) for one test case.

    invalid_pattern:
      "all_valid"          - every topk slot is valid
      "partial_per_row"    - ~30% random -1 within each row
      "one_row_all_invalid"- one randomly chosen row has all -1
      "all_invalid"        - every row has all -1
    """
    torch.manual_seed(seed)
    device = "cuda"

    # Make the kv pool comfortably larger than topk_count so randperm has room.
    kv_pool = max(kv_pool_min, topk_count * 2)
    q = (
        torch.randn(batch, nhead, HEAD_DIM, dtype=torch.float32, device=device) * 0.5
    ).to(torch.bfloat16)
    kv = (
        torch.randn(kv_pool, 1, HEAD_DIM, dtype=torch.float32, device=device) * 0.5
    ).to(torch.bfloat16)

    topk = torch.full((batch, topk_count), -1, dtype=torch.int32, device=device)
    for b in range(batch):
        gen = torch.Generator(device=device).manual_seed(seed * 31 + b)
        sampled = torch.randperm(kv_pool, device=device, generator=gen)[:topk_count]
        topk[b] = sampled.int()

    if invalid_pattern == "all_valid":
        pass
    elif invalid_pattern == "partial_per_row":
        # randomly mask ~30% per row
        for b in range(batch):
            gen = torch.Generator(device=device).manual_seed(seed * 71 + b)
            n_invalid = max(1, int(topk_count * 0.3))
            idx = torch.randperm(topk_count, device=device, generator=gen)[:n_invalid]
            topk[b, idx] = -1
    elif invalid_pattern == "one_row_all_invalid":
        if batch >= 1:
            topk[batch // 2] = -1
    elif invalid_pattern == "all_invalid":
        topk[:] = -1
    else:
        raise ValueError(f"unknown invalid_pattern={invalid_pattern}")

    attn_sink = (
        (torch.randn(nhead, dtype=torch.float32, device=device) * 0.5)
        if sink else None
    )

    sm_scale = HEAD_DIM ** -0.5
    return q, kv, topk, attn_sink, sm_scale


def _ref_decode(q, kv, topk, attn_sink, sm_scale):
    """Mirrors vllm `_ref_sparse_attn_decode` semantics for a single pool.

    Returns: (out_bf16 [batch, nhead, HEAD_DIM], lse_sink_free [batch, nhead] fp32).
    """
    batch, nhead, d = q.shape
    n_total = kv.shape[0]
    topk_count = topk.shape[1]

    # Squeeze the M=1 dim of kv so we can index with [num_slots, d].
    kv_flat = kv.view(-1, d)
    invalid = topk == -1
    safe = topk.clone()
    safe[invalid] = 0
    gathered = (
        kv_flat.index_select(0, safe.flatten().long())
        .view(batch, topk_count, d)
        .float()
    )

    qf = q.float()
    attn = qf @ gathered.transpose(-1, -2)  # [batch, nhead, topk]
    attn *= sm_scale
    attn = attn.masked_fill(
        invalid.unsqueeze(1).expand(-1, nhead, -1), float("-inf")
    )

    lse = attn.logsumexp(-1)  # [batch, nhead], sink-FREE

    # Compute attention output. For lonely rows (lse=-inf) the divide is 0/0
    # which we explicitly zero out to match the kernel's lonely-row guard.
    safe_lse = torch.where(
        lse == float("-inf"), torch.full_like(lse, 0.0), lse
    )
    probs = torch.exp(attn - safe_lse.unsqueeze(-1))
    out = probs @ gathered  # [batch, nhead, d] fp32
    if attn_sink is not None:
        # output *= 1 / (1 + exp(sink - lse))
        sink_b = attn_sink.view(1, nhead).expand(batch, nhead)
        scale = 1.0 / (1.0 + torch.exp(sink_b - safe_lse))
        out = out * scale.unsqueeze(-1)
    lonely = (lse == float("-inf")).unsqueeze(-1).expand_as(out)
    out = out.masked_fill(lonely, 0.0)
    return out.to(torch.bfloat16), lse


def _run_kernel(q, kv, topk, attn_sink, sm_scale, return_lse: bool):
    out = torch.zeros_like(q)
    lse_out = (
        torch.empty((q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    _, lse_ret = sparse_mla_decode_gluon(
        q,
        kv,
        topk,
        out,
        sm_scale,
        attn_sink=attn_sink,
        return_lse=return_lse,
        lse_out=lse_out,
    )
    return out, (lse_ret if return_lse else None)


# -------- assertions --------


def _assert_ok(out, lse_kernel, out_ref, lse_ref, with_sink: bool, label: str):
    assert out.shape == out_ref.shape, f"[{label}] shape mismatch"
    assert out.dtype == torch.bfloat16, f"[{label}] dtype mismatch"
    assert torch.isfinite(out).all(), f"[{label}] kernel output contains NaN/Inf"

    atol = 2e-2 if with_sink else 1.5e-2
    rtol = 1e-2
    diff = (out.float() - out_ref.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    assert torch.allclose(
        out.float(), out_ref.float(), atol=atol, rtol=rtol
    ), f"[{label}] out mismatch: max_abs={max_abs:.4e} mean_abs={mean_abs:.4e}"

    if lse_kernel is not None:
        # Sink-FREE lse (kernel emits before sink fold).
        finite_mask = torch.isfinite(lse_ref)
        if finite_mask.any():
            diff_lse = (lse_kernel[finite_mask] - lse_ref[finite_mask]).abs().max().item()
            assert diff_lse < 5e-2, (
                f"[{label}] lse mismatch: max_abs(finite)={diff_lse:.4e}"
            )
        # Lonely rows must produce -inf in the kernel too.
        lonely = lse_ref == float("-inf")
        if lonely.any():
            kernel_lonely_inf = lse_kernel[lonely] == float("-inf")
            assert kernel_lonely_inf.all(), (
                f"[{label}] kernel lse not -inf on lonely rows"
            )


# -------- (1) topk × sink × invalid_pattern cross at batch=64, nhead=128 --------


@pytest.mark.parametrize("topk_count", [256, 512, 1024])
@pytest.mark.parametrize("with_sink", [False, True])
@pytest.mark.parametrize(
    "invalid_pattern",
    ["all_valid", "partial_per_row", "one_row_all_invalid", "all_invalid"],
)
def test_topk_sink_invalid_cross(topk_count, with_sink, invalid_pattern):
    if invalid_pattern == "all_invalid":
        # The all-invalid stress is covered by a dedicated test below.
        pytest.skip("covered by dedicated all-invalid test")
    q, kv, topk, sink, sm_scale = _make_inputs(
        batch=64, topk_count=topk_count, nhead=128,
        sink=with_sink, invalid_pattern=invalid_pattern,
    )
    out_ref, lse_ref = _ref_decode(q, kv, topk, sink, sm_scale)
    out, lse_kernel = _run_kernel(q, kv, topk, sink, sm_scale, return_lse=False)
    _assert_ok(out, None, out_ref, lse_ref, with_sink,
               f"topk={topk_count}/sink={with_sink}/{invalid_pattern}")


# -------- (2) batch × nhead sweep at topk=512 --------


@pytest.mark.parametrize("batch", [1, 2, 4, 64, 128, 256])
@pytest.mark.parametrize("nhead", [64, 128])
def test_batch_nhead_sweep(batch, nhead):
    q, kv, topk, sink, sm_scale = _make_inputs(
        batch=batch, topk_count=512, nhead=nhead,
        sink=False, invalid_pattern="partial_per_row",
    )
    out_ref, lse_ref = _ref_decode(q, kv, topk, sink, sm_scale)
    out, _ = _run_kernel(q, kv, topk, sink, sm_scale, return_lse=False)
    _assert_ok(out, None, out_ref, lse_ref, False,
               f"batch={batch}/nhead={nhead}")


# -------- (3) return_lse path --------


@pytest.mark.parametrize("batch", [1, 64, 256])
@pytest.mark.parametrize("topk_count", [256, 1024])
@pytest.mark.parametrize("with_sink", [False, True])
def test_return_lse(batch, topk_count, with_sink):
    q, kv, topk, sink, sm_scale = _make_inputs(
        batch=batch, topk_count=topk_count, nhead=128,
        sink=with_sink, invalid_pattern="partial_per_row",
    )
    # When return_lse=True the kernel forces NUM_KV_SPLITS=1 and fold-sink
    # is OFF (sink folded externally). To compare against ref we pass sink
    # to the ref but None to the kernel, then apply sink correction
    # externally below.
    out_ref, lse_ref = _ref_decode(q, kv, topk, sink, sm_scale)
    out, lse_kernel = _run_kernel(q, kv, topk, None, sm_scale, return_lse=True)

    # Kernel `out` lacks sink correction; apply it externally to compare
    # apples-to-apples (and also exercise the lse value).
    if with_sink:
        sink_b = sink.view(1, sink.shape[0]).expand(batch, sink.shape[0])
        safe_lse = torch.where(
            lse_kernel == float("-inf"),
            torch.full_like(lse_kernel, 0.0),
            lse_kernel,
        )
        scale = 1.0 / (1.0 + torch.exp(sink_b - safe_lse))
        out = (out.float() * scale.unsqueeze(-1)).to(torch.bfloat16)

    _assert_ok(out, lse_kernel, out_ref, lse_ref, with_sink,
               f"return_lse: batch={batch}/topk={topk_count}/sink={with_sink}")


# -------- (4) all-invalid stress (lonely-row HCA short-ctx case) --------


@pytest.mark.parametrize("with_sink", [False, True])
def test_all_invalid_lonely_rows(with_sink):
    q, kv, topk, sink, sm_scale = _make_inputs(
        batch=4, topk_count=256, nhead=128,
        sink=with_sink, invalid_pattern="all_invalid",
    )
    out, lse_kernel = _run_kernel(q, kv, topk, sink, sm_scale, return_lse=True)
    # When all-invalid + sink: kernel still folds sink? No — return_lse=True
    # forces in-kernel sink off. The output should be 0 (lonely-row guard),
    # and lse should be -inf.
    assert torch.isfinite(out).all(), "all-invalid output contains NaN/Inf"
    assert out.abs().max().item() == 0.0, (
        f"all-invalid output not all-zero: max_abs={out.abs().max().item():.4e}"
    )
    assert (lse_kernel == float("-inf")).all(), (
        "all-invalid lse not all -inf"
    )

    # Also test the in-kernel sink path (return_lse=False, sink in-kernel).
    if with_sink:
        out2, _ = _run_kernel(q, kv, topk, sink, sm_scale, return_lse=False)
        assert torch.isfinite(out2).all(), "all-invalid+kernel-sink: NaN/Inf"
        # When all topk are -1 AND sink is in-kernel, the lonely-row guard
        # short-circuits to 0 (we add sink_contrib only when e_max > -inf).
        assert out2.abs().max().item() == 0.0, (
            f"all-invalid+kernel-sink output not all-zero: max_abs={out2.abs().max().item():.4e}"
        )


# -------- (5) TOPK=128 transparent padding stress (V4 SWA layer) --------


def test_topk128_transparent_padding():
    # V4 SWA layer's window_size=128 → topk_count=128. The wrapper must
    # pad to 256 transparently and the result must equal the manually-padded
    # version.
    q, kv, topk_128, sink, sm_scale = _make_inputs(
        batch=64, topk_count=128, nhead=128,
        sink=False, invalid_pattern="partial_per_row",
    )

    out_ref, lse_ref = _ref_decode(q, kv, topk_128, sink, sm_scale)

    out, _ = _run_kernel(q, kv, topk_128, sink, sm_scale, return_lse=False)
    assert torch.isfinite(out).all(), "topk=128 padded output contains NaN/Inf"
    diff = (out.float() - out_ref.float()).abs()
    max_abs = diff.max().item()
    assert torch.allclose(
        out.float(), out_ref.float(), atol=1.5e-2, rtol=1e-2
    ), f"topk=128 padding mismatch vs ref: max_abs={max_abs:.4e}"
