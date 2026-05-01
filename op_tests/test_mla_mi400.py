# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Minimal gfx1250 MLA smoke test.

This exercises the single mi400 shader
`mla_a8w8_qh16_1tg_16mx4_64nx1_np_KERNEL_FUNC` through aiter's loader and
launcher. The shader comes from `test_kl_mla_tp_np_test` and has baked
masked/causal=1 semantics. This smoke only checks load/launch and output shape;
it does not validate non-causal API behavior or numerical correctness.
"""

from pathlib import Path

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx


def _require_gfx1250():
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA/HIP GPU")
    try:
        gfx = get_gfx()
    except Exception as exc:
        pytest.skip(f"cannot detect gfx arch: {exc}")
    if gfx != "gfx1250":
        pytest.skip(f"requires gfx1250, got {gfx}")


def test_mla_mi400_minimal_smoke(monkeypatch):
    _require_gfx1250()

    repo_hsa_dir = Path(__file__).resolve().parents[1] / "hsa"
    monkeypatch.setenv("AITER_ASM_DIR", str(repo_hsa_dir))

    device = torch.device("cuda")
    batch = 1
    q_seq_len = 4
    kv_seq_len = 578
    page_size = 64
    num_kv_splits = 1
    nhead = 16
    nhead_kv = 1
    qk_head_dim = 576
    v_head_dim = 512
    num_pages = (kv_seq_len + page_size - 1) // page_size

    q = torch.randn(
        (batch * q_seq_len, nhead, qk_head_dim), dtype=torch.bfloat16, device=device
    ).to(dtypes.fp8)
    kv_buffer = torch.randn(
        (num_pages, page_size, nhead_kv, qk_head_dim), dtype=torch.bfloat16, device=device
    ).to(dtypes.fp8)
    out = torch.empty((batch * q_seq_len, nhead, v_head_dim), dtype=torch.bfloat16, device=device)

    qo_indptr = torch.tensor([0, q_seq_len], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, kv_seq_len], dtype=torch.int32, device=device)
    kv_indices = torch.zeros(num_pages + 4, dtype=torch.int32, device=device)
    kv_indices[:num_pages] = torch.arange(num_pages, dtype=torch.int32, device=device)
    kv_last_page_lens = torch.tensor([kv_seq_len % page_size], dtype=torch.int32, device=device)
    num_kv_splits_indptr = torch.tensor([0, num_kv_splits], dtype=torch.int32, device=device)
    q_scale = torch.ones((batch,), dtype=torch.float32, device=device)
    kv_scale = torch.ones((batch,), dtype=torch.float32, device=device)

    attn_logits, attn_lse = aiter.mla.mla_decode_fwd(
        q,
        kv_buffer,
        out,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        q_seq_len,
        page_size,
        nhead_kv,
        1.0 / (qk_head_dim**0.5),
        num_kv_splits=num_kv_splits,
        num_kv_splits_indptr=num_kv_splits_indptr,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )

    assert out.shape == (batch * q_seq_len, nhead, v_head_dim)
    assert attn_logits.shape == out.shape
    assert attn_lse.shape == (batch * q_seq_len, num_kv_splits, nhead, 1)
