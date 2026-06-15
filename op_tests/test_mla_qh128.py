# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Focused mi400 (gfx1250) correctness sweep for qh128-q1-16mx4-64nx1-np.

Self-contained test (does not import op_tests/test_mla.py). Fixed knobs:
  - variant: qh128-q1-16mx4-64nx1-np (nhead=128, decode_qlen=1)
  - cartesian sweep (_RUN_POC_KL=False), not run.sh parity cases
  - _Q_PATTERN=3: 768-byte padded Q row stride (rope_split3)

Usage (ff_sp3 container, see poc_kl/mi400/mla/toaiter/README.md):

    docker exec ff_sp3 bash -lc 'cd /home/carhuang/feifei/aiter && \\
      rm -rf aiter/jit/build/module_mla_asm aiter/jit/module_mla_asm.so && \\
      env -u AITER_ASM_DEBUG -u AITER_MLA_DEBUG_SKIP_KERNEL \\
      ROCM_HOME=/opt/rocm ENABLE_CK=0 ENABLE_FLYDSL=0 \\
      GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 \\
      AITER_ASM_DIR=/home/carhuang/feifei/aiter/hsa \\
      python3 op_tests/test_mla_qh128.py'
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path

import pandas as pd
import torch

import aiter
import aiter.mla
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.attention import mla_decode_stage1_asm_fwd
from aiter.test_common import run_perftest

# In lean containers aiter.__init__ may skip bulk op exports; register explicitly.
aiter.mla_decode_stage1_asm_fwd = mla_decode_stage1_asm_fwd

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault("AITER_ASM_DIR", str(ROOT / "hsa"))

# qh128-q1-16mx4-64nx1-np
NHEAD = 128
DECODE_QLEN = 1
VARIANT = "qh128-q1-16mx4-64nx1-np"
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
V_HEAD_DIM = KV_LORA_RANK  # 512
NHEAD_KV = 1
PAGE_SIZE = 64
COS_THRESHOLD = 5e-2
PAGE_INDICES_OOB = 4
_RUN_POC_KL = False
_Q_PATTERN = 3

_CTX_LENS = [1, 3, 17, 1024, 1024+3, 1024+17]
_BATCH_SIZES = [1, 64]
_SPLIT_PER_BATCH = [1, 2]
_MASK_CHOICES = [1]

KV_MAX_SZ = 65536 * 32
# Poison slots beyond ctx_lens with NaN/Inf to detect kernel reads of invalid tokens.
POISON_INVALID_KV_SLOTS = True


def _poison_invalid_kv_slots(kv_pages_bf16: torch.Tensor, ctx_lens: int) -> None:
    """In-place: fill page token slots beyond ctx_lens with NaN / +Inf."""
    num_pages_per_batch = (ctx_lens + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = kv_pages_bf16.shape[0]
    for logical_page in range(total_pages):
        page_in_batch = logical_page % num_pages_per_batch
        tokens_before = page_in_batch * PAGE_SIZE
        if page_in_batch == num_pages_per_batch - 1:
            valid_in_page = ctx_lens - tokens_before
        else:
            valid_in_page = PAGE_SIZE
        if valid_in_page >= PAGE_SIZE:
            continue
        tail = kv_pages_bf16[logical_page, valid_in_page:]
        tail[..., ::2] = float("nan")
        tail[..., 1::2] = float("inf")


def _nonfinite_report(tensor: torch.Tensor, name: str) -> str:
    t = tensor.detach().float().cpu()
    nan_cnt = torch.isnan(t).sum().item()
    inf_cnt = torch.isinf(t).sum().item()
    if nan_cnt == 0 and inf_cnt == 0:
        return f"{name}: all finite"
    return f"{name}: nan={nan_cnt} inf={inf_cnt}"


def _pack_rope_split3_q_pages(tensor, nope_dim, rope_dim, padded_stride_bytes=768):
    shape = tensor.shape
    assert shape[-1] == nope_dim + rope_dim
    elem_size = tensor.element_size()
    padded_dim = padded_stride_bytes // elem_size
    rows = tensor.reshape(-1, shape[-1])
    padded = torch.zeros(
        (rows.shape[0], padded_dim),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded[:, : shape[-1]].copy_(rows)
    return torch.as_strided(
        padded,
        size=shape,
        stride=(
            shape[1] * shape[2] * padded_dim,
            shape[2] * padded_dim,
            padded_dim,
            1,
        ),
    )


def _pack_rope_split2_kv_pages(tensor, nope_dim, rope_dim):
    pages, page_size, nhead_kv, head_dim = tensor.shape
    assert nhead_kv == 1
    packed = torch.cat(
        (
            tensor[..., :nope_dim].reshape(pages, page_size * nope_dim),
            tensor[..., nope_dim:].reshape(pages, page_size * rope_dim),
        ),
        dim=-1,
    )
    return packed.reshape(pages, page_size, nhead_kv, head_dim).contiguous()


def _make_page_permutation(num_pages):
    if num_pages <= 1:
        return list(range(num_pages))
    for step in (7, 5, 3):
        if num_pages % step != 0:
            return [(i * step + 1) % num_pages for i in range(num_pages)]
    return list(reversed(range(num_pages)))


def _make_scales(batch, device):
    q_scale = torch.linspace(0.75, 1.25, 1, dtype=torch.float32, device=device)
    kv_scale = torch.linspace(1.20, 0.80, 1, dtype=torch.float32, device=device)
    return q_scale, kv_scale


def _make_mla_mi400_case(*, batch, ctx_lens, num_kv_splits):
    device = torch.device("cuda")
    torch.manual_seed(
        20260513
        + batch * 1009
        + ctx_lens
        + NHEAD * 7
        + DECODE_QLEN
        + num_kv_splits * 101
    )

    num_pages_per_batch = (ctx_lens + PAGE_SIZE - 1) // PAGE_SIZE
    last_page_len = ctx_lens % PAGE_SIZE or PAGE_SIZE
    kv_last_page_lens = torch.full(
        (batch,), last_page_len, dtype=torch.int32, device=device
    )
    num_kv_splits_indptr = (
        torch.arange(batch + 1, dtype=torch.int32, device=device) * num_kv_splits
    )
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(
        torch.full(
            (batch,), num_pages_per_batch, dtype=torch.int32, device=device
        ),
        dim=0,
    )
    q_scale, kv_scale = _make_scales(batch, device)
    return {
        "page_size": PAGE_SIZE,
        "num_kv_splits": num_kv_splits,
        "num_pages_per_batch": num_pages_per_batch,
        "kv_last_page_lens": kv_last_page_lens,
        "kv_indptr": kv_indptr,
        "num_kv_splits_indptr": num_kv_splits_indptr,
        "q_scale": q_scale,
        "kv_scale": kv_scale,
    }


def _make_mla_mi400_kv_case(*, kv_buffer_bf16, batch, ctx_lens):
    device = torch.device("cuda")
    num_pages_per_batch = (ctx_lens + PAGE_SIZE - 1) // PAGE_SIZE
    total_page_indices = batch * (num_pages_per_batch + PAGE_INDICES_OOB)
    total_pages = batch * num_pages_per_batch

    kv_buffer_source_bf16 = kv_buffer_bf16.view(
        -1, PAGE_SIZE, NHEAD_KV, QK_HEAD_DIM
    )
    available_pages = kv_buffer_source_bf16.size(0)
    if available_pages >= total_pages:
        kv_buffer_logical_bf16 = kv_buffer_source_bf16[:total_pages].contiguous()
    else:
        kv_buffer_logical_bf16 = torch.empty(
            (total_pages, PAGE_SIZE, NHEAD_KV, QK_HEAD_DIM),
            dtype=kv_buffer_source_bf16.dtype,
            device=kv_buffer_source_bf16.device,
        )
        kv_buffer_logical_bf16[:available_pages] = kv_buffer_source_bf16
        kv_buffer_logical_bf16[available_pages:] = torch.randn(
            (total_pages - available_pages, PAGE_SIZE, NHEAD_KV, QK_HEAD_DIM),
            dtype=kv_buffer_source_bf16.dtype,
            device=kv_buffer_source_bf16.device,
        )

    if POISON_INVALID_KV_SLOTS:
        _poison_invalid_kv_slots(kv_buffer_logical_bf16, ctx_lens)

    shuffled_page_indices = _make_page_permutation(total_pages)
    kv_buffer_scattered_bf16 = torch.empty_like(kv_buffer_logical_bf16)
    kv_indices = torch.zeros(total_page_indices, dtype=torch.int32, device=device)
    for logical_page, physical_page in enumerate(shuffled_page_indices):
        kv_buffer_scattered_bf16[physical_page] = kv_buffer_logical_bf16[logical_page]
        kv_indices[logical_page] = physical_page

    kv_buffer_ref = kv_buffer_scattered_bf16.to(dtypes.fp8)
    kv_buffer = _pack_rope_split2_kv_pages(
        kv_buffer_ref.view(total_pages, PAGE_SIZE, NHEAD_KV, QK_HEAD_DIM),
        V_HEAD_DIM,
        QK_HEAD_DIM - V_HEAD_DIM,
    )
    return kv_buffer, kv_buffer_ref, kv_indices


def _make_mla_mi400_q_case(*, q_fp8, batch, decode_qlen, nhead):
    assert _Q_PATTERN == 3
    q = q_fp8.view(batch, decode_qlen, nhead, QK_HEAD_DIM)
    q = _pack_rope_split3_q_pages(q, V_HEAD_DIM, QK_HEAD_DIM - V_HEAD_DIM)
    return torch.as_strided(
        q,
        size=(batch * decode_qlen, nhead, QK_HEAD_DIM),
        stride=(nhead * q.stride(2), q.stride(2), q.stride(3)),
    )


def _apply_causal_mask_(logits):
    _, s_q, s_k = logits.shape
    mask = torch.ones(s_q, s_k, dtype=torch.bool, device=logits.device).tril(
        diagonal=s_k - s_q
    )
    logits.masked_fill_(mask.logical_not().unsqueeze(0), float("-inf"))


def _ref_mla_mi400(case, q_ref, kv_buffer_ref, kv_indices, batch_size, ctx_lens, mask):
    outputs = []
    num_pages = case["num_pages_per_batch"]
    for b in range(batch_size):
        q_start = b * DECODE_QLEN
        q_end = q_start + DECODE_QLEN
        q_scale = case["q_scale"][0 if case["q_scale"].numel() == 1 else b]
        kv_scale = case["kv_scale"][0 if case["kv_scale"].numel() == 1 else b]
        q = q_ref[q_start:q_end].float() * q_scale
        page_indices = kv_indices[b * num_pages : (b + 1) * num_pages].long()
        kv = torch.index_select(kv_buffer_ref.float(), 0, page_indices) * kv_scale
        kv = kv.reshape(-1, NHEAD_KV, QK_HEAD_DIM)[:ctx_lens]
        key = kv
        value = kv[..., :V_HEAD_DIM]
        logits = torch.einsum("qhd,kmd->hqk", q, key) * (1.0 / (QK_HEAD_DIM**0.5))
        if mask:
            _apply_causal_mask_(logits)
        weights = torch.softmax(logits, dim=-1)
        outputs.append(
            torch.einsum("hqk,kmd->qhd", weights, value).to(torch.bfloat16)
        )
    return torch.cat(outputs, dim=0)


def _cosine_diff(actual, expected):
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    numerator = 2 * (actual.double() * expected.double()).sum()
    denominator = (
        (actual.double().square() + expected.double().square()).sum().clamp_min(1e-12)
    )
    return (1 - (numerator / denominator)).item()


def _make_base_buffers(batch_size, ctx_lens):
    num_page = (KV_MAX_SZ + PAGE_SIZE - 1) // PAGE_SIZE
    n_kv_idx = batch_size * ctx_lens + 10000
    kv_indices = torch.randint(0, num_page, (n_kv_idx,), dtype=torch.int)
    kv_buffer = torch.randn(
        (num_page * PAGE_SIZE, NHEAD_KV, QK_HEAD_DIM),
        dtype=torch.bfloat16,
    )
    total_q = batch_size * DECODE_QLEN
    qo_indptr = torch.arange(batch_size + 1, dtype=torch.int) * DECODE_QLEN
    q = torch.randn((total_q, NHEAD, QK_HEAD_DIM), dtype=torch.bfloat16)
    return kv_buffer, kv_indices, qo_indptr, q


def run_qh128_case(ctx_lens, batch_size, split_per_batch, mask):
    ret = {
        "ctx_lens": ctx_lens,
        "batch_size": batch_size,
        "split_per_batch": split_per_batch,
        "mask": mask,
        "variant": VARIANT,
        "passed": None,
        "finite": None,
        "out_finite": None,
        "logits_finite": None,
        "lse_finite": None,
        "cos_diff": None,
        "us": None,
        "TFLOPS": None,
        "TB/s": None,
    }

    kv_buffer, kv_indices, qo_indptr, q = _make_base_buffers(batch_size, ctx_lens)
    kv_buffer_mi400, kv_buffer_ref_mi400, kv_indices_mi400 = _make_mla_mi400_kv_case(
        kv_buffer_bf16=kv_buffer,
        batch=batch_size,
        ctx_lens=ctx_lens,
    )
    q_fp8_mi400 = q.to(dtypes.fp8)
    q_mi400 = _make_mla_mi400_q_case(
        q_fp8=q_fp8_mi400,
        batch=batch_size,
        decode_qlen=DECODE_QLEN,
        nhead=NHEAD,
    )
    case = _make_mla_mi400_case(
        batch=batch_size,
        ctx_lens=ctx_lens,
        num_kv_splits=split_per_batch,
    )

    out_mi400 = torch.zeros(
        (batch_size * DECODE_QLEN, NHEAD, V_HEAD_DIM),
        dtype=torch.bfloat16,
    )
    attn_logits, attn_lse = aiter.mla.mla_decode_fwd(
        q_mi400,
        kv_buffer_mi400,
        out_mi400,
        qo_indptr,
        case["kv_indptr"],
        kv_indices_mi400,
        case["kv_last_page_lens"],
        DECODE_QLEN,
        case["page_size"],
        NHEAD_KV,
        1.0 / (QK_HEAD_DIM**0.5),
        num_kv_splits=case["num_kv_splits"],
        num_kv_splits_indptr=case["num_kv_splits_indptr"],
        q_scale=case["q_scale"],
        kv_scale=case["kv_scale"],
        return_lse=True,
    )
    out_check = out_mi400.clone()

    logits_shape = (
        (batch_size * DECODE_QLEN, NHEAD, V_HEAD_DIM)
        if case["num_kv_splits"] == 1
        else (batch_size * DECODE_QLEN, case["num_kv_splits"], NHEAD, V_HEAD_DIM)
    )
    assert out_check.shape == (batch_size * DECODE_QLEN, NHEAD, V_HEAD_DIM)
    assert attn_logits.shape == logits_shape
    assert attn_lse.shape == (batch_size * DECODE_QLEN, NHEAD)

    out_finite = torch.isfinite(out_check.detach().float().cpu()).all().item()
    logits_finite = torch.isfinite(attn_logits.detach().float().cpu()).all().item()
    lse_finite = torch.isfinite(attn_lse.detach().float().cpu()).all().item()
    finite = out_finite and logits_finite and lse_finite
    if not finite:
        aiter.logger.warning(
            "non-finite outputs [%s | batch=%d ctx=%d splits=%d]: %s | %s | %s",
            VARIANT,
            batch_size,
            ctx_lens,
            case["num_kv_splits"],
            _nonfinite_report(out_check, "out"),
            _nonfinite_report(attn_logits, "attn_logits"),
            _nonfinite_report(attn_lse, "attn_lse"),
        )
    if finite:
        expected = _ref_mla_mi400(
            case,
            q_fp8_mi400,
            kv_buffer_ref_mi400,
            kv_indices_mi400,
            batch_size,
            ctx_lens,
            mask,
        )
        cos_diff = _cosine_diff(out_check, expected)
    else:
        cos_diff = float("inf")

    passed = finite and cos_diff < COS_THRESHOLD
    ret["finite"] = finite
    ret["out_finite"] = out_finite
    ret["logits_finite"] = logits_finite
    ret["lse_finite"] = lse_finite
    ret["cos_diff"] = cos_diff
    ret["passed"] = passed
    aiter.logger.info(
        "mla_decode-mi400 [%s | batch=%d ctx=%d splits=%d]: finite=%s cos_diff=%.3e %s",
        VARIANT,
        batch_size,
        ctx_lens,
        case["num_kv_splits"],
        finite,
        cos_diff,
        "passed" if passed else "FAILED",
    )

    if not _RUN_POC_KL:
        _, us_mi400 = run_perftest(
            aiter.mla.mla_decode_fwd,
            q_mi400,
            kv_buffer_mi400,
            out_mi400,
            qo_indptr,
            case["kv_indptr"],
            kv_indices_mi400,
            case["kv_last_page_lens"],
            DECODE_QLEN,
            case["page_size"],
            NHEAD_KV,
            1.0 / (QK_HEAD_DIM**0.5),
            num_kv_splits=case["num_kv_splits"],
            num_kv_splits_indptr=case["num_kv_splits_indptr"],
            q_scale=case["q_scale"],
            kv_scale=case["kv_scale"],
            return_lse=True,
        )
        total_q = batch_size * DECODE_QLEN
        total_kv = batch_size * ctx_lens
        mi_flops = (
            DECODE_QLEN * total_kv * NHEAD * (QK_HEAD_DIM + V_HEAD_DIM) * 2
        )
        mi_bytes = (
            total_kv * NHEAD_KV * QK_HEAD_DIM * (torch.finfo(dtypes.fp8).bits // 8)
            + total_q * NHEAD * QK_HEAD_DIM * (torch.finfo(dtypes.fp8).bits // 8)
            + total_q * NHEAD * V_HEAD_DIM * (torch.finfo(torch.bfloat16).bits // 8)
        )
        ret["us"] = us_mi400
        ret["TFLOPS"] = mi_flops / us_mi400 / 1e6
        ret["TB/s"] = mi_bytes / us_mi400 / 1e6
        aiter.logger.info(
            "mla_decode-mi400 [%s | batch=%d ctx=%d]: %8.2f us  %7.2f TFLOPS  %7.2f TB/s",
            VARIANT,
            batch_size,
            ctx_lens,
            us_mi400,
            ret["TFLOPS"],
            ret["TB/s"],
        )

    return ret


def main() -> None:
    if get_gfx() != "gfx1250":
        raise SystemExit(f"requires gfx1250 (mi400), got {get_gfx()}")

    assert _RUN_POC_KL is False
    assert _Q_PATTERN == 3

    aiter.logger.info(
        "test_mla_qh128: variant=%s _RUN_POC_KL=%s _Q_PATTERN=%s poison_invalid_kv=%s",
        VARIANT,
        _RUN_POC_KL,
        _Q_PATTERN,
        POISON_INVALID_KV_SLOTS,
    )
    aiter.logger.info(
        "sweep: ctx_lens=%s batch=%s splits=%s mask=%s",
        _CTX_LENS,
        _BATCH_SIZES,
        _SPLIT_PER_BATCH,
        _MASK_CHOICES,
    )

    failures: list[tuple] = []
    rows: list[dict] = []
    for ctx_len, batch_size, split_per_batch, mask in itertools.product(
        _CTX_LENS,
        _BATCH_SIZES,
        _SPLIT_PER_BATCH,
        _MASK_CHOICES,
    ):
        ret = run_qh128_case(ctx_len, batch_size, split_per_batch, mask)
        rows.append(ret)
        if not ret.get("passed", False):
            failures.append(
                (VARIANT, batch_size, ctx_len, split_per_batch, ret.get("cos_diff"))
            )

    df = pd.DataFrame(rows)
    aiter.logger.info("mla qh128 summary (markdown):\n%s", df.to_markdown(index=False))
    aiter.logger.info(
        "test_mla_qh128 done: passed=%d failed=%d total=%d",
        sum(1 for r in rows if r.get("passed")),
        len(failures),
        len(rows),
    )

    if failures:
        raise AssertionError(f"qh128 mi400 MLA numerics failed for: {failures}")


if __name__ == "__main__":
    main()
