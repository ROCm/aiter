# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Triton-free reproducer / regression guard for the mi400 (gfx1250) decode-MLA
stage1 ASM kernel producing NaN under num_kv_splits>1.

Background
----------
With num_kv_splits>1 the per-split KV count is no longer a tile multiple, so the
multi-pass stage1 ASM kernel can leave stale garbage in score registers it does
not write for the short/partial split. The in-kernel online softmax then turns
that garbage into exp(inf-inf)=NaN, poisoning BOTH the denominator L and the
accumulator R -> the per-split partials (logits / attn_lse) come out non-finite.
num_kv_splits=1 uses the single-pass path and stays finite.

This test deliberately calls ONLY `aiter.mla_decode_stage1_asm_fwd` (the ASM
stage1) and inspects its raw per-split partials. It does NOT run the Triton
stage2 reduce, so a failure here is decisively attributed to the ASM kernel and
is reproducible without any Triton dependency.

The partial buffers are pre-seeded with a finite empty-split sentinel
(logits=0, attn_lse=-1e20); any non-finite value afterwards was written by the
ASM kernel itself (not an uninitialized read).

Semantics
---------
* Before the sp3 fix: the num_kv_splits=2 cases FAIL (NaN reproduced).
* After the sp3 fix: every case is finite -> the test PASSES and becomes a
  permanent regression guard.

The num_kv_splits=1 control cases must ALWAYS be finite; if they ever fail the
harness/inputs are wrong, not the split bug.

Usage:
    pytest -q op_tests/test_mla_decode_split_nan.py
    # more seed attempts per case (data-dependent trigger):
    MLA_SPLIT_NAN_SEEDS=8 pytest -q op_tests/test_mla_decode_split_nan.py
"""

import os

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.attention import mla_decode_stage1_asm_fwd

# In lean containers aiter.__init__ may skip bulk op exports; register explicitly
# (mirrors op_tests/test_mla.py).
aiter.mla_decode_stage1_asm_fwd = mla_decode_stage1_asm_fwd

# DeepSeek-MLA absorb dims used by the mi400 fp8 decode path.
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
V_HEAD_DIM = KV_LORA_RANK  # 512
PAGE_SIZE = 64
NHEAD_KV = 1

# How many distinct seeds to try per case. The NaN is data-dependent (same page
# geometry can be NaN or OK depending on the score values), so multiple draws
# raise the reproduction probability. Override via env for a heavier sweep.
_SEEDS = int(os.environ.get("MLA_SPLIT_NAN_SEEDS", "4"))

# (nhead, decode_qlen, batch, ctx_lens, num_kv_splits)
# The first two mirror poc_kl/mi400/mla/run.sh's known-failing split=2 cases
# (qh64 / qh128, ctx=962, passes=2). The rest stress short / non-tile-multiple
# contexts across multiple batches, which is where short splits appear.
_SPLIT_CASES = [
    (64, 1, 4, 962, 2),
    (128, 1, 4, 962, 2),
    (64, 1, 8, 65, 2),
    (64, 1, 8, 127, 2),
    (64, 1, 8, 129, 2),
    (64, 1, 8, 193, 2),
    (64, 1, 16, 577, 2),
    (128, 1, 8, 577, 2),
    (64, 1, 64, 962, 2),
]

# num_kv_splits=1 controls: must always be finite.
_CONTROL_CASES = [
    (64, 1, 4, 962, 1),
    (128, 1, 4, 962, 1),
]


def _require_mi400():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    gfx = get_gfx()
    if gfx != "gfx1250":
        pytest.skip(f"mi400 decode-MLA stage1 ASM kernel requires gfx1250, got {gfx}")


def _build_inputs(nhead, decode_qlen, batch, ctx_lens, num_kv_splits, seed):
    """Build a valid mi400 fp8 decode-MLA stage1 input set, reusing the already
    validated builders from op_tests.test_mla. Imported lazily so test
    collection never requires a CUDA device at import time."""
    from op_tests.test_mla import (
        _make_mla_mi400_case,
        _make_mla_mi400_kv_case,
        _make_mla_mi400_q_case,
    )

    device = torch.device("cuda")
    torch.manual_seed(seed)

    num_pages_per_batch = (ctx_lens + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = batch * num_pages_per_batch

    # Base bf16 KV pool and query, from which the builders derive the fp8 +
    # seg-packed (rope_split2) kernel layout.
    kv_buffer_bf16 = torch.randn(
        (total_pages * PAGE_SIZE, NHEAD_KV, QK_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    q_bf16 = torch.randn(
        (batch * decode_qlen, nhead, QK_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )

    kv_buffer, _kv_buffer_ref, kv_indices = _make_mla_mi400_kv_case(
        kv_buffer_bf16=kv_buffer_bf16,
        batch=batch,
        ctx_lens=ctx_lens,
        qk_head_dim=QK_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        page_indices_oob=4,
    )
    q = _make_mla_mi400_q_case(
        q_fp8=q_bf16.to(dtypes.fp8),
        batch=batch,
        decode_qlen=decode_qlen,
        nhead=nhead,
        qk_head_dim=QK_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
    )
    case = _make_mla_mi400_case(
        batch=batch,
        ctx_lens=ctx_lens,
        nhead=nhead,
        decode_qlen=decode_qlen,
        num_kv_splits=num_kv_splits,
    )

    qo_indptr = (
        torch.arange(batch + 1, dtype=torch.int32, device=device) * decode_qlen
    )
    return q, kv_buffer, kv_indices, case, qo_indptr


def _run_stage1_only(nhead, decode_qlen, batch, ctx_lens, num_kv_splits, seed):
    """Launch ONLY the ASM stage1 kernel and return its raw per-split partials
    (logits, attn_lse). Buffers are pre-seeded with a finite empty-split
    sentinel so any non-finite value is one the kernel wrote."""
    q, kv_buffer, kv_indices, case, qo_indptr = _build_inputs(
        nhead, decode_qlen, batch, ctx_lens, num_kv_splits, seed
    )
    device = torch.device("cuda")
    total_s = batch * decode_qlen

    o = torch.zeros((total_s, nhead, V_HEAD_DIM), dtype=torch.bfloat16, device=device)
    logits = torch.empty(
        (total_s, num_kv_splits, nhead, V_HEAD_DIM), dtype=torch.float32, device=device
    )
    attn_lse = torch.empty(
        (total_s, num_kv_splits, nhead, 1), dtype=torch.float32, device=device
    )
    # finite empty-split sentinel: data=0, lse=-1e20.
    logits.fill_(0.0)
    attn_lse.fill_(-1.0e20)

    aiter.mla_decode_stage1_asm_fwd(
        q,
        kv_buffer,
        qo_indptr,
        case["kv_indptr"],
        kv_indices,
        case["kv_last_page_lens"],
        case["num_kv_splits_indptr"],
        None,  # work_meta_data
        None,  # work_indptr
        None,  # work_info_set
        decode_qlen,  # max_seqlen_q
        case["page_size"],
        NHEAD_KV,
        1.0 / (QK_HEAD_DIM**0.5),
        logits,
        attn_lse,
        o,
        None,  # final_lse
        case["q_scale"],
        case["kv_scale"],
    )
    torch.cuda.synchronize()
    return logits, attn_lse


def _diagnose(logits, attn_lse):
    """Short human-readable summary of where/what the non-finite values are."""
    lg = logits.detach().float()
    lse = attn_lse.detach().float()
    lg_fin_per_split = torch.isfinite(lg).all(dim=3).all(dim=2).all(dim=0).cpu()
    lse_fin_per_split = torch.isfinite(lse).all(dim=3).all(dim=2).all(dim=0).cpu()
    bad_logit_splits = (~lg_fin_per_split).nonzero().reshape(-1).tolist()
    bad_lse_splits = (~lse_fin_per_split).nonzero().reshape(-1).tolist()
    lse_nan = int(torch.isnan(lse).sum().item())
    lse_posinf = int((lse == float("inf")).sum().item())
    lse_neginf = int((lse == float("-inf")).sum().item())
    lg_nan = int(torch.isnan(lg).sum().item())
    lg_posinf = int((lg == float("inf")).sum().item())
    if lse_nan > 0:
        kind = "L==NaN (denom NaN; R also NaN)"
    elif lse_neginf > 0 and lg_posinf == 0:
        kind = "L==0 exactly (0/0)"
    elif lg_posinf or lse_posinf:
        kind = "OVERFLOW(+inf -> bad running-max)"
    else:
        kind = "other"
    return (
        f"logits_bad_splits={bad_logit_splits} lse_bad_splits={bad_lse_splits} "
        f"logits[+inf={lg_posinf} nan={lg_nan}] "
        f"lse[+inf={lse_posinf} -inf={lse_neginf} nan={lse_nan}] kind={kind}"
    )


@pytest.mark.parametrize(
    "nhead,decode_qlen,batch,ctx_lens,num_kv_splits", _SPLIT_CASES
)
def test_stage1_split_partials_finite(
    nhead, decode_qlen, batch, ctx_lens, num_kv_splits
):
    """num_kv_splits>1 stage1 partials must be finite. Reproduces the NaN bug
    pre-fix; guards against regression post-fix."""
    _require_mi400()
    for s in range(_SEEDS):
        seed = 20260615 + s * 7919 + ctx_lens + nhead * 13 + batch
        logits, attn_lse = _run_stage1_only(
            nhead, decode_qlen, batch, ctx_lens, num_kv_splits, seed
        )
        finite = (
            torch.isfinite(logits).all().item()
            and torch.isfinite(attn_lse).all().item()
        )
        if not finite:
            diag = _diagnose(logits, attn_lse)
            pytest.fail(
                f"stage1 ASM produced non-finite partials for "
                f"(nhead={nhead}, decode_qlen={decode_qlen}, batch={batch}, "
                f"ctx_lens={ctx_lens}, num_kv_splits={num_kv_splits}, seed={seed}):\n"
                f"  {diag}\n"
                f"  (Triton stage2 NOT run -> NaN is in the ASM stage1 kernel.)"
            )


@pytest.mark.parametrize(
    "nhead,decode_qlen,batch,ctx_lens,num_kv_splits", _CONTROL_CASES
)
def test_stage1_single_split_partials_finite(
    nhead, decode_qlen, batch, ctx_lens, num_kv_splits
):
    """num_kv_splits=1 control: the single-pass path must always be finite. A
    failure here means the harness/inputs are wrong, not the split bug."""
    _require_mi400()
    for s in range(_SEEDS):
        seed = 20260615 + s * 7919 + ctx_lens + nhead * 13 + batch
        logits, attn_lse = _run_stage1_only(
            nhead, decode_qlen, batch, ctx_lens, num_kv_splits, seed
        )
        assert torch.isfinite(logits).all().item(), _diagnose(logits, attn_lse)
        assert torch.isfinite(attn_lse).all().item(), _diagnose(logits, attn_lse)


if __name__ == "__main__":
    # Allow running as a plain script (no pytest): prints a finiteness table.
    _require_mi400()
    print(
        f"{'nhead':>5} {'dq':>3} {'batch':>5} {'ctx':>6} {'splits':>6} "
        f"{'seed':>12} {'finite':>6}  diag"
    )
    for case in _CONTROL_CASES + _SPLIT_CASES:
        nhead, decode_qlen, batch, ctx_lens, num_kv_splits = case
        for s in range(_SEEDS):
            seed = 20260615 + s * 7919 + ctx_lens + nhead * 13 + batch
            logits, attn_lse = _run_stage1_only(
                nhead, decode_qlen, batch, ctx_lens, num_kv_splits, seed
            )
            finite = (
                torch.isfinite(logits).all().item()
                and torch.isfinite(attn_lse).all().item()
            )
            diag = "" if finite else _diagnose(logits, attn_lse)
            print(
                f"{nhead:>5} {decode_qlen:>3} {batch:>5} {ctx_lens:>6} "
                f"{num_kv_splits:>6} {seed:>12} {str(finite):>6}  {diag}"
            )
