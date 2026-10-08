# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Row-width guards of ``mla_decode_stage1_asm_fwd`` on gfx942/gfx950.

The asm decode kernels are built for DeepSeek's MLA row (512 latent + 64 RoPE
elements, or 656 bytes per token for a packed byte KV) and are not told the row
width, so the op has to reject any other width before launching them. The
rejection has to reach Python as a RuntimeError through the ctypes error bridge
and leave the process able to decode valid rows.

CI runs this via ``python3 op_tests/test_mla_decode_row_width.py`` (also
pytest-collectable).
"""

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx

bf16 = dtypes.bf16
fp8 = dtypes.fp8

NHEAD = 16
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
SM_SCALE = 1.0 / (KV_LORA_RANK + QK_ROPE_HEAD_DIM) ** 0.5
BATCH = 4
CTX_LEN = 300
NUM_PAGE = 2048


def _has_asm_decode():
    try:
        return get_gfx() in ("gfx942", "gfx950")
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _has_asm_decode(), reason="checks the gfx942/gfx950 asm decode kernels"
)


def _inputs(q_dtype, kv_dtype, head_size):
    torch.manual_seed(0)
    q = torch.randn(BATCH, NHEAD, head_size, device="cuda").to(q_dtype)
    kv = torch.randn(NUM_PAGE, 1, 1, head_size, device="cuda").to(kv_dtype)
    return q, kv


def _mla_decode(q, kv_buffer, v_head_dim=KV_LORA_RANK):
    """One query token per sequence over CTX_LEN cached tokens, page_size 1."""
    qo_indptr = torch.arange(BATCH + 1, dtype=torch.int, device="cuda")
    kv_indptr = qo_indptr * CTX_LEN
    kv_indices = torch.randperm(NUM_PAGE, device="cuda")[: BATCH * CTX_LEN].int()
    kv_last_page_lens = torch.ones(BATCH, dtype=torch.int, device="cuda")
    one = torch.ones(1, dtype=torch.float, device="cuda")
    out = torch.empty(BATCH, NHEAD, v_head_dim, dtype=bf16, device="cuda")
    aiter.mla.mla_decode_fwd(
        q,
        kv_buffer,
        out,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        1,
        sm_scale=SM_SCALE,
        q_scale=one if q.dtype == fp8 else None,
        kv_scale=one if kv_buffer.dtype == fp8 else None,
    )
    return out, kv_indptr, kv_indices


def _reference(q, kv_buffer, kv_indptr, kv_indices):
    rows = kv_buffer.float().flatten(0, 2)
    out = []
    for b in range(BATCH):
        kv = rows[kv_indices[kv_indptr[b] : kv_indptr[b + 1]].long()]
        p = torch.softmax(q[b].float() @ kv.T * SM_SCALE, dim=-1)
        out.append(p @ kv[:, :KV_LORA_RANK])
    return torch.stack(out)


@pytest.mark.parametrize(
    "q_dtype,kv_dtype",
    [(bf16, bf16), (bf16, fp8), (fp8, fp8)],
    ids=["bf16-bf16", "bf16-fp8", "fp8-fp8"],
)
def test_rejects_rope_free_rows(q_dtype, kv_dtype):
    q, kv = _inputs(q_dtype, kv_dtype, KV_LORA_RANK)
    with pytest.raises(RuntimeError, match="576-element MLA rows"):
        _mla_decode(q, kv)


def test_rejects_v_head_dim_other_than_512():
    q, kv = _inputs(bf16, bf16, KV_LORA_RANK + QK_ROPE_HEAD_DIM)
    with pytest.raises(RuntimeError, match="v_head_dim 512"):
        _mla_decode(q, kv, v_head_dim=256)


def test_rejects_byte_kv_without_rope():
    q, _ = _inputs(bf16, bf16, KV_LORA_RANK + QK_ROPE_HEAD_DIM)
    # 512 fp8 latent + 4 fp32 scales per token, but no 64 bf16 RoPE elements.
    kv = torch.zeros(NUM_PAGE, 528, dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError, match="byte KV must be"):
        _mla_decode(q, kv)


@pytest.mark.parametrize(
    "dtype,max_rel_err", [(bf16, 1e-2), (fp8, 5e-2)], ids=["bf16", "fp8"]
)
def test_deepseek_rows_decode_after_a_rejection(dtype, max_rel_err):
    q, kv = _inputs(dtype, dtype, KV_LORA_RANK)
    with pytest.raises(RuntimeError, match="576-element MLA rows"):
        _mla_decode(q, kv)

    q, kv = _inputs(dtype, dtype, KV_LORA_RANK + QK_ROPE_HEAD_DIM)
    out, kv_indptr, kv_indices = _mla_decode(q, kv)
    ref = _reference(q, kv, kv_indptr, kv_indices)
    rel_err = ((out.float() - ref).norm() / ref.norm()).item()
    assert rel_err < max_rel_err, f"{rel_err=}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
