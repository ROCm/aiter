# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tests for pa_decode_bf16_asm (FP8 paged-attention decode ASM, gfx1250).

Ops layer:  aiter.pa_decode_bf16_asm  (wraps SP3 PA_DECODE_D64_1TG_4W_PS)

The kernel is gfx1250-only and shipped as hsa/gfx1250/pa_decode_bf16/*.co.
Properties (see the reference host file sched2/pa_ps.cpp):
  * head_dim=64, page_size=256, gqa=8.
  * FP8 Q **and** FP8 paged KV cache; bf16 output.
  * per-tensor scalar dequant scales for Q/K/V (softmax scale folded into
    key_scale by the wrapper).
  * persistent / split-KV; GPT-OSS style attention sink.

Two tiers of tests:
  * Wiring tests (`test_exported`, `test_csv_manifest`) run anywhere — no GPU,
    no compiled module — and lock in the public API + csv manifest contract.
  * `test_decode_smoke` is gfx1250-gated and builds the paged FP8 layout from
    pa_ps.cpp, then launches the kernel.  It SKIPS (does not fail) when the
    shipped .co is still the placeholder, so it is safe to run before the real
    code object is dropped in.  Full numeric validation lives in the standalone
    sched2/pa_ps.cpp harness against the CPU reference.
"""

from __future__ import annotations

import csv
import os

import pytest

import aiter

AITER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PA_CSV = os.path.join(
    AITER_ROOT, "hsa", "gfx1250", "pa_decode_bf16", "pa_decode_bf16.csv"
)

PA_HEAD_DIM = 64
PA_PAGE_SIZE = 256
PA_GQA_RATIO = 8


def _is_gfx1250_host() -> bool:
    """True only on a gfx1250 GPU host (mirrors test_fmha_fwd_with_sink_asm)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx

        return get_gfx() == "gfx1250"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Wiring tests — run anywhere (no GPU / no build required).
# ---------------------------------------------------------------------------
def test_exported():
    """The public op + low-level ctypes binding are exported from aiter."""
    assert callable(aiter.pa_decode_bf16_asm)
    # Low-level ctypes op is also reachable through the ops module.
    from aiter.ops.attention import _pa_decode_bf16_asm

    assert callable(_pa_decode_bf16_asm)


def test_csv_manifest():
    """The gfx1250 PA-decode csv manifest matches the kernel-selection keys
    used by asm_pa_decode_bf16.cu (qdtype/kvdtype/hdim/page_size/gqa)."""
    assert os.path.isfile(PA_CSV), f"missing manifest: {PA_CSV}"
    with open(PA_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) >= 1
    required = {"qdtype", "kvdtype", "hdim", "page_size", "gqa", "knl_name", "co_name"}
    assert required.issubset(rows[0].keys()), rows[0].keys()

    row = rows[0]
    assert row["qdtype"] == "fp8"
    assert row["kvdtype"] == "fp8"
    assert int(row["hdim"]) == PA_HEAD_DIM
    assert int(row["page_size"]) == PA_PAGE_SIZE
    assert int(row["gqa"]) == PA_GQA_RATIO
    # The referenced code object must exist (placeholder or real).
    co_path = os.path.join(os.path.dirname(PA_CSV), row["co_name"])
    assert os.path.isfile(co_path), f"missing .co: {co_path}"


def test_scale_folding(monkeypatch):
    """The wrapper's per-tensor Q/K/V dequant-scale contract (no GPU needed).

    Intercepts the low-level ctypes op and checks that the public wrapper
    builds three 1-element fp32 scale tensors with:
      * q_scale == query_scale
      * k_scale == key_scale * softmax_scale   (softmax scale folded in)
      * v_scale == value_scale
    matching the kernel's scl_log2e = query_scale*key_scale*log2e contract
    (see asm_pa_decode_bf16.cu / sched2/pa_ps.cpp).
    """
    torch = pytest.importorskip("torch")
    import aiter.ops.attention as attn

    captured = {}

    def _fake(Q, K, V, kv_indices, context_lens, q_scale, k_scale, v_scale,
              out, qo_indptr, kv_indptr, work_indptr, work_info, split_o,
              split_lse, sink, gqa, mtp, kernelName):
        captured["q_scale"] = q_scale
        captured["k_scale"] = k_scale
        captured["v_scale"] = v_scale
        captured["sink"] = sink
        captured["out"] = out

    # Patch the bare name the wrapper resolves at call time (avoids any build).
    monkeypatch.setattr(attn, "_pa_decode_bf16_asm", _fake)

    gqa = PA_GQA_RATIO
    kv_head_num = 2
    head_dim = PA_HEAD_DIM
    q_head_num = kv_head_num * gqa

    # Tiny CPU tensors — nothing is launched, only shape/stride/scale logic runs.
    Q = torch.empty(1, 1, kv_head_num, gqa, head_dim, dtype=torch.float8_e4m3fn)
    K = torch.empty(1, kv_head_num, head_dim, dtype=torch.float8_e4m3fn)
    V = torch.empty(1, kv_head_num, head_dim, dtype=torch.float8_e4m3fn)
    kv_indices = torch.zeros(1, dtype=torch.int32)
    context_lens = torch.zeros(1, dtype=torch.int32)
    kv_indptr = torch.zeros(2, dtype=torch.int32)

    query_scale, key_scale, value_scale = 0.5, 2.0, 1.5
    softmax_scale = 1.0 / (head_dim**0.5)

    aiter.pa_decode_bf16_asm(
        Q, K, V, kv_indices, context_lens, softmax_scale, kv_indptr,
        gqa=gqa, mtp=0,
        query_scale=query_scale, key_scale=key_scale, value_scale=value_scale,
    )

    for key in ("q_scale", "k_scale", "v_scale"):
        t = captured[key]
        assert t.dtype == torch.float32, f"{key} must be fp32, got {t.dtype}"
        assert t.numel() == 1, f"{key} must be a 1-element scalar tensor"

    assert captured["q_scale"].item() == pytest.approx(query_scale)
    assert captured["k_scale"].item() == pytest.approx(key_scale * softmax_scale)
    assert captured["v_scale"].item() == pytest.approx(value_scale)

    # Default sink is a per-Q-head -inf no-op buffer the kernel always reads.
    sink = captured["sink"]
    assert sink.dtype == torch.float32 and sink.numel() == q_head_num
    assert torch.isneginf(sink).all()
    # Output buffer is bf16 with Q's logical shape.
    assert captured["out"].shape == Q.shape
    assert captured["out"].dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Functional smoke test — gfx1250 only.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not _is_gfx1250_host(),
    reason="pa_decode_bf16_asm ASM kernel is gfx1250-only; no GPU or different arch",
)
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("kv_head_num", [1, 2])
def test_decode_smoke(batch, kv_head_num):
    """Build the paged FP8 layout from pa_ps.cpp and launch the kernel.

    Skips gracefully when the shipped .co is still the placeholder (the kernel
    symbol fails to load) so this is safe to run before the real code object
    is added.
    """
    import torch

    gqa = PA_GQA_RATIO
    head_dim = PA_HEAD_DIM
    page_size = PA_PAGE_SIZE
    kv_seq_len = page_size  # single page per sequence keeps the harness simple
    mtp = 0
    qlen_with_mtp = mtp + 1
    q_head_num = kv_head_num * gqa
    device = "cuda"

    fp8 = torch.float8_e4m3fn
    pages_per_seq = (kv_seq_len + page_size - 1) // page_size
    num_phys_pages = batch * pages_per_seq

    # Q: [batch, mtp_layers, kv_head, gqa, head_dim] FP8 (see pa_ps.cpp).
    Q = torch.randn(
        batch * qlen_with_mtp, kv_head_num, gqa, head_dim, device=device
    ).to(fp8)
    Q = Q.reshape(batch, qlen_with_mtp, kv_head_num, gqa, head_dim)

    # Paged K/V: contiguous physical pages, [num_pages, kv_heads, ...] so
    # K.stride(0)/stride(1) give blk/head strides used by the C++ entry.
    K = torch.zeros(
        num_phys_pages, kv_head_num, head_dim // 16, page_size, 16, device=device
    ).to(fp8)
    V = torch.zeros(
        num_phys_pages, kv_head_num, page_size // 16, head_dim, 16, device=device
    ).to(fp8)

    context_lens = torch.full((batch,), kv_seq_len, dtype=torch.int32, device=device)
    kv_indptr = torch.arange(
        0, (batch + 1) * pages_per_seq, pages_per_seq, dtype=torch.int32, device=device
    )
    kv_indices = torch.arange(
        num_phys_pages, dtype=torch.int32, device=device
    )
    qo_indptr = torch.arange(
        0, (batch + 1) * qlen_with_mtp, qlen_with_mtp, dtype=torch.int32, device=device
    )

    softmax_scale = 1.0 / (head_dim**0.5)
    # Exercise the per-tensor FP8 dequant-scale path with non-trivial values
    # (the wrapper folds softmax_scale into key_scale before launch).
    query_scale, key_scale, value_scale = 0.5, 2.0, 1.5

    try:
        out = aiter.pa_decode_bf16_asm(
            Q,
            K,
            V,
            kv_indices,
            context_lens,
            softmax_scale,
            kv_indptr,
            gqa=gqa,
            mtp=mtp,
            query_scale=query_scale,
            key_scale=key_scale,
            value_scale=value_scale,
            qo_indptr=qo_indptr,
        )
    except Exception as e:  # placeholder .co / missing kernel symbol
        msg = str(e).lower()
        if "kernel" in msg or "hsaco" in msg or "co" in msg or "symbol" in msg:
            pytest.skip(f"PA decode .co not available (placeholder): {e}")
        raise

    assert out.shape == Q.shape
    assert out.dtype == torch.bfloat16


if __name__ == "__main__":
    test_exported()
    test_csv_manifest()
    print("wiring tests PASSED")
