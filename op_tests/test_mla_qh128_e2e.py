# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Focused mi400 (gfx1250) STAGE1-ONLY NaN reproducer for qh128-q1-16mx4-64nx1-np.

This variant deliberately runs ONLY the stage1 ASM kernel
(`mla_decode_stage1_asm_fwd`) and inspects its raw per-split partials
(splitData / splitLse). The Triton stage2 reduce is NOT used; splits are merged
with a pure-PyTorch online-softmax instead. A non-finite value in the stage1
partials is therefore decisively attributed to the ASM kernel (stage2
exonerated, no Triton dependency).

Two input sources:
  * synthetic sweep (default when no dump dir): rope_split3 Q + rope_split2 KV,
    optional POISON of >ctx_len token slots.
  * real ATOM dumps: set MLA_DECODE_DUMP_DIR (default /root/mla_decode_dump).
    Each .pt is replayed through stage1-only at num_kv_splits=1 and =2; the
    partials are checked for finiteness and (if finite) merged in torch and
    compared to the recorded server output.

The partial buffers are pre-seeded with a finite empty-split sentinel
(splitData=0, splitLse=-1e20) so any NaN/Inf afterwards was written by the ASM
kernel itself, not read uninitialized.

Usage (ff_mla container):

    docker exec ff_mla bash -lc 'cd /home/carhuang/feifei/aiter && \\
      ENABLE_CK=0 ENABLE_FLYDSL=0 \\
      AITER_ASM_DIR=/home/carhuang/feifei/aiter/hsa \\
      MLA_DECODE_DUMP_DIR=/root/mla_decode_dump \\
      python3 "op_tests/test_mla_qh128 copy.py"'
"""

from __future__ import annotations

import glob
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

# In lean containers aiter.__init__ may skip bulk op exports; register explicitly.
aiter.mla_decode_stage1_asm_fwd = mla_decode_stage1_asm_fwd

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault("AITER_ASM_DIR", str(ROOT / "hsa"))

# Real ATOM decode-MLA dumps (seg layout). Validated stage1-only when present.
DUMP_DIR = os.environ.get("MLA_DECODE_DUMP_DIR", "/root/mla_decode_dump")
# num_kv_splits values to replay for each real dump.
DUMP_SPLITS = [1, 2]

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

_CTX_LENS = list[int](range(63, 130))# [3, 7, 10, 60, 100, 150, 1024+3]
_BATCH_SIZES = [64]
_SPLIT_PER_BATCH = [1,2]
_MASK_CHOICES = [1]

KV_MAX_SZ = 65536 * 32
# Fill page token slots beyond ctx_lens: "nan" (NaN/Inf poison) or a scalar (e.g. 1.0).
POISON_INVALID_KV_SLOTS = True
INVALID_KV_SLOT_FILL = "nan"


def _poison_invalid_kv_slots(kv_pages_bf16: torch.Tensor, ctx_lens: int) -> None:
    """In-place: fill page token slots beyond ctx_lens."""
    if not POISON_INVALID_KV_SLOTS:
        return
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
        if INVALID_KV_SLOT_FILL == "nan":
            tail[..., ::2] = float("nan")
            tail[..., 1::2] = float("inf")
        else:
            tail.fill_(float(INVALID_KV_SLOT_FILL))


def _nonfinite_report(tensor: torch.Tensor, name: str) -> str:
    t = tensor.detach().float().cpu()
    nan_cnt = torch.isnan(t).sum().item()
    inf_cnt = torch.isinf(t).sum().item()
    if nan_cnt == 0 and inf_cnt == 0:
        return f"{name}: all finite"
    return f"{name}: nan={nan_cnt} inf={inf_cnt}"


def _stage1_only(
    *,
    q,
    kv_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    num_kv_splits_indptr,
    num_kv_splits,
    max_seqlen_q,
    page_size,
    sm_scale,
    nhead,
    v_head_dim,
    q_scale,
    kv_scale,
):
    """Launch ONLY the stage1 ASM kernel; return its raw per-split partials
    (splitData, splitLse). NO Triton stage2. Buffers are pre-seeded with a finite
    empty-split sentinel (data=0, lse=-1e20) so any NaN/Inf left behind was
    written by the ASM kernel."""
    device = q.device
    total_s = qo_indptr[-1].item() if qo_indptr.numel() else q.shape[0]
    o = torch.zeros((total_s, nhead, v_head_dim), dtype=torch.bfloat16, device=device)
    # mi400 dispatch requires splitData(R) to be bf16 for kv_split==1 and fp32 for
    # the multi-split path. Mirror mla.mla_decode_fwd exactly: for split==1 the
    # single-pass kernel writes the final bf16 output, so splitData IS a bf16 view
    # of o; for split>1 it is a separate fp32 partial buffer. splitLse is fp32.
    if num_kv_splits == 1:
        logits = o.view(total_s, num_kv_splits, nhead, v_head_dim)
    else:
        logits = torch.empty(
            (total_s, num_kv_splits, nhead, v_head_dim),
            dtype=torch.float32,
            device=device,
        )
        logits.fill_(0.0)
    attn_lse = torch.empty(
        (total_s, num_kv_splits, nhead, 1), dtype=torch.float32, device=device
    )
    attn_lse.fill_(-1.0e20)
    aiter.mla_decode_stage1_asm_fwd(
        q,
        kv_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        num_kv_splits_indptr,
        None,  # work_meta_data
        None,  # work_indptr
        None,  # work_info_set
        max_seqlen_q,
        page_size,
        NHEAD_KV,
        sm_scale,
        logits,
        attn_lse,
        o,
        None,  # final_lse
        q_scale,
        kv_scale,
    )
    torch.cuda.synchronize()
    return logits, attn_lse


def _torch_reduce(logits, attn_lse):
    """Pure-PyTorch replacement for the Triton stage2 online-softmax reduce.
    logits [S,K,H,D], attn_lse [S,K,H,1] -> final o [S,H,D]. Empty splits carry
    lse=-1e20 so their weight underflows to 0."""
    lse = attn_lse.float()
    m = lse.max(dim=1, keepdim=True).values
    w = torch.exp(lse - m)
    denom = w.sum(dim=1, keepdim=True).clamp_min(1e-30)
    return (w * logits.float()).sum(dim=1) / denom.squeeze(1)


def _stage1_diag(logits, attn_lse):
    """Per-buffer nan/inf counts + a kind= classification, plus which split
    indices carry the non-finite values."""
    lg = logits.detach().float()
    lse = attn_lse.detach().float()
    lg_fin = bool(torch.isfinite(lg).all().item())
    lse_fin = bool(torch.isfinite(lse).all().item())
    if lg_fin and lse_fin:
        return True, "all finite"
    lg_bad = (~torch.isfinite(lg).all(dim=3).all(dim=2).all(dim=0)).nonzero()
    lse_bad = (~torch.isfinite(lse).all(dim=3).all(dim=2).all(dim=0)).nonzero()
    lg_nan = int(torch.isnan(lg).sum().item())
    lg_pinf = int((lg == float("inf")).sum().item())
    lg_ninf = int((lg == float("-inf")).sum().item())
    lse_nan = int(torch.isnan(lse).sum().item())
    lse_pinf = int((lse == float("inf")).sum().item())
    lse_ninf = int((lse == float("-inf")).sum().item())
    if lse_nan > 0:
        kind = "L==NaN (denom NaN; R also NaN)"
    elif lse_ninf > 0 and lg_pinf == 0:
        kind = "L==0 exactly (0/0)"
    elif lg_pinf or lse_pinf:
        kind = "OVERFLOW(+inf -> bad running-max)"
    else:
        kind = "other"
    return False, (
        f"splitData_bad_splits={lg_bad.reshape(-1).tolist()} "
        f"splitLse_bad_splits={lse_bad.reshape(-1).tolist()} "
        f"splitData[+inf={lg_pinf} -inf={lg_ninf} nan={lg_nan}] "
        f"splitLse[+inf={lse_pinf} -inf={lse_ninf} nan={lse_nan}] kind={kind}"
    )


def _cos(a, b):
    a = a.detach().float().reshape(-1).cpu()
    b = b.detach().float().reshape(-1).cpu()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


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

    # STAGE1-ONLY: no Triton stage2. Inspect the raw per-split partials.
    logits, attn_lse = _stage1_only(
        q=q_mi400,
        kv_buffer=kv_buffer_mi400,
        qo_indptr=qo_indptr,
        kv_indptr=case["kv_indptr"],
        kv_indices=kv_indices_mi400,
        kv_last_page_lens=case["kv_last_page_lens"],
        num_kv_splits_indptr=case["num_kv_splits_indptr"],
        num_kv_splits=case["num_kv_splits"],
        max_seqlen_q=DECODE_QLEN,
        page_size=case["page_size"],
        sm_scale=1.0 / (QK_HEAD_DIM**0.5),
        nhead=NHEAD,
        v_head_dim=V_HEAD_DIM,
        q_scale=case["q_scale"],
        kv_scale=case["kv_scale"],
    )

    logits_finite = torch.isfinite(logits.detach().float().cpu()).all().item()
    lse_finite = torch.isfinite(attn_lse.detach().float().cpu()).all().item()
    finite = logits_finite and lse_finite
    _, diag = _stage1_diag(logits, attn_lse)
    if not finite:
        aiter.logger.warning(
            "stage1 NON-FINITE [%s | batch=%d ctx=%d splits=%d]: %s "
            "(Triton stage2 NOT run -> NaN is in the ASM stage1 kernel)",
            VARIANT,
            batch_size,
            ctx_lens,
            case["num_kv_splits"],
            diag,
        )

    # Triton-free split merge + cosine vs fp32 ref (only meaningful when finite).
    if finite:
        out_check = _torch_reduce(logits, attn_lse)
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
    ret["out_finite"] = finite
    ret["logits_finite"] = logits_finite
    ret["lse_finite"] = lse_finite
    ret["cos_diff"] = cos_diff
    ret["passed"] = passed
    ret["diag"] = diag
    aiter.logger.info(
        "stage1-only [%s | batch=%d ctx=%d splits=%d]: finite=%s cos_diff=%.3e %s",
        VARIANT,
        batch_size,
        ctx_lens,
        case["num_kv_splits"],
        finite,
        cos_diff,
        "passed" if passed else "FAILED",
    )
    return ret


def _normalize_dump(b):
    """Accept both ATOM dump schemas (dump_decode_mla / _dump_seg_decode_failure);
    both are the seg layout. Map alternate key names onto the canonical ones."""
    if "kv_compact" not in b and "seg_kv_compact" in b:
        b["kv_compact"] = b["seg_kv_compact"]
    if "kv_indices" not in b and "kv_indices_remapped" in b:
        b["kv_indices"] = b["kv_indices_remapped"]
    if "o_server" not in b and "o" in b:
        b["o_server"] = b["o"]
    b.setdefault("kv_layout", "seg")
    return b


def _print_dump_params(b, path):
    """Print all parameters carried by a dump (ctx_len, batch_size, split, dims,
    scales, page geometry) so a single-dump repro is fully described."""
    page_size = int(b["page_size"])
    kvi = b["kv_indptr"].to(torch.int64).cpu()
    klp = b["kv_last_page_lens"].to(torch.int64).cpu()
    bs = int(kvi.numel() - 1)
    o = b["o_server"]
    total_s, nhead, v_head_dim = o.shape
    q = b["q"]
    n_pages = (kvi[1:] - kvi[:-1])
    ctx = (n_pages - 1).clamp_min(0) * page_size + klp[:bs]
    ctx_list = ctx.tolist()
    uniq_ctx = sorted(set(ctx_list))

    def _scal(x):
        if x is None:
            return None
        return x.flatten()[0].item() if hasattr(x, "flatten") else x

    lines = [
        f"=== dump params: {os.path.basename(path)} ===",
        f"  batch_size      = {bs}",
        f"  total_s (rows)  = {int(total_s)}   (qo rows; decode_qlen per batch)",
        f"  nhead           = {int(nhead)}",
        f"  v_head_dim      = {int(v_head_dim)}",
        f"  kv_lora_rank    = {b.get('kv_lora_rank')}",
        f"  qk_rope_head_dim= {b.get('qk_rope_head_dim')}",
        f"  q.shape/dtype   = {tuple(q.shape)} / {q.dtype}",
        f"  kv.shape/dtype  = {tuple(b['kv_compact'].shape)} / {b['kv_compact'].dtype}",
        f"  page_size       = {page_size}",
        f"  max_q_len       = {b.get('max_q_len')}",
        f"  sm_scale        = {b.get('sm_scale')}",
        f"  q_scale         = {_scal(b.get('q_scale'))}",
        f"  kv_scale        = {_scal(b.get('kv_scale'))}",
        f"  layer_num       = {b.get('layer_num')}",
        f"  n_pages/batch   = min={int(n_pages.min())} max={int(n_pages.max())}",
        f"  ctx_len/batch   = min={min(ctx_list)} max={max(ctx_list)} "
        f"uniq={uniq_ctx if len(uniq_ctx) <= 16 else str(uniq_ctx[:16]) + '...'}",
        f"  splits tested   = {DUMP_SPLITS}",
    ]
    aiter.logger.info("\n".join(lines))


def run_dump_case(path, num_kv_splits):
    """STAGE1-ONLY replay of a real ATOM dump. Feeds the dumped (already
    seg-packed) q / kv directly to the stage1 ASM kernel and checks the raw
    per-split partials. No Triton, no re-packing."""
    dev = torch.device("cuda")
    b = _normalize_dump(torch.load(path, map_location="cpu"))
    name = os.path.basename(path)
    ret = {
        "name": name,
        "splits": num_kv_splits,
        "bs": int(b["kv_indptr"].numel() - 1),
        "finite": None,
        "logits_finite": None,
        "lse_finite": None,
        "cos_vs_server": None,
        "passed": None,
        "diag": "",
    }
    if b.get("kv_layout") != "seg":
        ret["diag"] = f"skip: kv_layout={b.get('kv_layout')} (need 'seg')"
        return ret

    q = b["q"].to(dev)
    seg_kv = b["kv_compact"].to(dev).contiguous()
    qo_indptr = b["qo_indptr"].to(dev).to(torch.int32)
    kv_indptr = b["kv_indptr"].to(dev).to(torch.int32)
    kv_indices = b["kv_indices"].to(dev).to(torch.int32)
    kv_last = b["kv_last_page_lens"].to(dev).to(torch.int32)
    q_scale = b["q_scale"].to(dev) if b.get("q_scale") is not None else None
    kv_scale = b["kv_scale"].to(dev) if b.get("kv_scale") is not None else None
    o_server = b["o_server"].to(dev)
    total_s, nhead, v_head_dim = o_server.shape
    bs = kv_indptr.numel() - 1
    num_kv_splits_indptr = torch.arange(
        0, (bs + 1) * num_kv_splits, num_kv_splits, dtype=torch.int32, device=dev
    )

    logits, attn_lse = _stage1_only(
        q=q,
        kv_buffer=seg_kv,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_last_page_lens=kv_last,
        num_kv_splits_indptr=num_kv_splits_indptr,
        num_kv_splits=num_kv_splits,
        max_seqlen_q=int(b["max_q_len"]),
        page_size=int(b["page_size"]),
        sm_scale=float(b["sm_scale"]),
        nhead=nhead,
        v_head_dim=v_head_dim,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )

    logits_finite = bool(torch.isfinite(logits.detach().float().cpu()).all().item())
    lse_finite = bool(torch.isfinite(attn_lse.detach().float().cpu()).all().item())
    finite = logits_finite and lse_finite
    _, diag = _stage1_diag(logits, attn_lse)
    cos_vs_server = float("nan")
    if finite:
        out = _torch_reduce(logits, attn_lse)  # [S, nhead, v_head_dim]
        cos_vs_server = _cos(out[..., :v_head_dim], o_server[..., :v_head_dim])
    ret["finite"] = finite
    ret["logits_finite"] = logits_finite
    ret["lse_finite"] = lse_finite
    ret["cos_vs_server"] = cos_vs_server
    ret["diag"] = diag
    # split=1 is the single-pass reference path: it must be finite. split>1 is
    # the path under test; finiteness there is the pass criterion.
    ret["passed"] = finite
    return ret


def run_dumps(all_dumps: bool = False) -> bool:
    paths = sorted(
        glob.glob(os.path.join(DUMP_DIR, "mla_decode_*.pt"))
        + glob.glob(os.path.join(DUMP_DIR, "seg_decode_*.pt"))
    )
    if not paths:
        aiter.logger.info("no dumps under %s", DUMP_DIR)
        return False
    if not all_dumps:
        paths = paths[:1]  # default: only the first dump
    aiter.logger.info(
        "replaying %d dump(s) from %s, stage1-only, splits=%s (mode=%s)",
        len(paths),
        DUMP_DIR,
        DUMP_SPLITS,
        "all" if all_dumps else "first",
    )
    rows: list[dict] = []
    failures: list[tuple] = []
    for p in paths:
        _print_dump_params(_normalize_dump(torch.load(p, map_location="cpu")), p)
        for nks in DUMP_SPLITS:
            r = run_dump_case(p, nks)
            rows.append(r)
            tag = "FINITE" if r["finite"] else "NaN/Inf"
            aiter.logger.info(
                "dump %-44s splits=%d finite=%s cos_vs_server=%s  %s",
                r["name"],
                nks,
                r["finite"],
                ("%.6f" % r["cos_vs_server"])
                if r["cos_vs_server"] == r["cos_vs_server"]
                else "nan",
                r["diag"] if not r["finite"] else "",
            )
            # Only count split>1 non-finite as a failure (split=1 is the control).
            if nks > 1 and not r["finite"]:
                failures.append((r["name"], nks, r["diag"]))

    df = pd.DataFrame(rows)
    aiter.logger.info("dump stage1-only summary:\n%s", df.to_markdown(index=False))
    n_split_gt1 = sum(1 for r in rows if r["splits"] > 1)
    n_nan_gt1 = sum(1 for r in rows if r["splits"] > 1 and not r["finite"])
    aiter.logger.info(
        "stage1-only dump replay: %d/%d split>1 dumps produced non-finite partials "
        "(=> ASM stage1 NaN; stage2/Triton not involved).",
        n_nan_gt1,
        n_split_gt1,
    )
    if failures:
        raise AssertionError(
            f"stage1 ASM produced non-finite partials (split>1) for: {failures}"
        )
    return True


def run_synth() -> bool:
    """Synthetic stage1-only sweep over _CTX_LENS x _BATCH_SIZES x _SPLIT_PER_BATCH."""
    assert _RUN_POC_KL is False
    assert _Q_PATTERN == 3
    aiter.logger.info(
        "test_mla_qh128 (stage1-only synth): variant=%s poison_invalid_kv=%s invalid_kv_fill=%s",
        VARIANT,
        POISON_INVALID_KV_SLOTS,
        INVALID_KV_SLOT_FILL,
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
    aiter.logger.info("mla qh128 synth summary (markdown):\n%s", df.to_markdown(index=False))
    aiter.logger.info(
        "test_mla_qh128 synth done: passed=%d failed=%d total=%d",
        sum(1 for r in rows if r.get("passed")),
        len(failures),
        len(rows),
    )
    if failures:
        raise AssertionError(f"qh128 mi400 MLA numerics failed for: {failures}")
    return True


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="stage1-only mi400 MLA NaN reproducer (qh128)."
    )
    parser.add_argument(
        "--mode",
        choices=["dump", "synth", "both"],
        # env fallback so the docker one-liner can pick the mode too.
        default=os.environ.get("MLA_QH128_MODE", "synth"),
        help=(
            "dump: replay real dumps from MLA_DECODE_DUMP_DIR (%s); "
            "synth: synthetic sweep over _CTX_LENS/_BATCH_SIZES/_SPLIT_PER_BATCH; "
            "both: dumps then synth. Default: synth." % DUMP_DIR
        ),
    )
    parser.add_argument(
        "--dumps",
        choices=["first", "all"],
        default=os.environ.get("MLA_QH128_DUMPS", "first"),
        help="first: only the first dump file (default); all: every dump file.",
    )
    args = parser.parse_args()

    if get_gfx() != "gfx1250":
        raise SystemExit(f"requires gfx1250 (mi400), got {get_gfx()}")

    aiter.logger.info("mode=%s dumps=%s", args.mode, args.dumps)
    if args.mode in ("dump", "both"):
        if not os.path.isdir(DUMP_DIR):
            raise SystemExit(
                f"--mode {args.mode} needs dumps but {DUMP_DIR} is not a directory "
                f"(set MLA_DECODE_DUMP_DIR)."
            )
        ran = run_dumps(all_dumps=(args.dumps == "all"))
        if args.mode == "dump":
            if not ran:
                raise SystemExit(f"no dumps found under {DUMP_DIR}")
            return

    run_synth()


if __name__ == "__main__":
    main()
