#!/usr/bin/env python3
"""
Minimal reproducer for issue #4364: qh16 fp8 MLA decode is run-to-run
nondeterministic because the split-KV reduce is fp-non-associative and the
per-request split count is drawn from a *global* budget
(min(cu_num, max_split_per_batch * batch_size)) that varies with batch
composition. Two identical requests at temperature 0 can thus land on
different split counts on different steps and yield slightly different logits
-- observed as +/-1-2 NIAH needles at mid context on GLM-5.1-FP8 DSA.

What this shows (single MI300X, gfx942), nhead=16 fp8/fp8 persistent decode:

  * split=1 run twice            -> BIT-IDENTICAL (single split => reduce is a no-op)
  * split=1 vs split=N (N>1)     -> DIFFER          (split-KV changes the numerics)

The first line is the deterministic baseline; the second is the mechanism behind
the run-to-run variance (production picks N from the batch-dependent budget).

WORKAROUND (this branch): AITER_MLA_DECODE_DETERMINISTIC=1 (or
AITER_MLA_DECODE_MAX_SPLIT_PER_BATCH=1) clamps get_mla_metadata_v1 to a single
split per request, so a request's decode output no longer depends on batch
composition. With it set, the split=N path collapses onto the split=1 result
and becomes reproducible. Verified below by re-running the "N" case with the env
set and checking it matches the split=1 baseline.

Run:  python3 op_tests/test_mla_qh16_decode_determinism.py
"""
import os
import sys
import torch

import aiter
from aiter import dtypes, get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.mla import mla_decode_fwd
from aiter.jit.utils.chip_info import get_gfx

NHEAD = 16          # native qh16 (the kernel GLM gqa64 folds onto)
NHEAD_KV = 1
KV_LORA_RANK = 512
QK_ROPE = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE  # 576
V_HEAD_DIM = KV_LORA_RANK             # 512
PAGE_SIZE = 1
DECODE_QLEN = 1
BATCH = 1
CTX = 8192          # long enough that a multi-split budget engages
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)


def _run_decode(q, kv_buffer_fp8, max_split, device="cuda"):
    """One persistent qh16 decode with metadata built for `max_split` KV splits."""
    num_page = CTX  # page_size=1
    qo_indptr = torch.zeros(BATCH + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = DECODE_QLEN
    kv_indptr = torch.zeros(BATCH + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = num_page
    kv_indices = torch.arange(num_page, dtype=torch.int32, device=device)
    kv_last_page_lens = torch.ones(BATCH, dtype=torch.int32, device=device)

    (
        (wmd_sz, wmd_t), (wi_sz, wi_t), (ws_sz, ws_t),
        (ri_sz, ri_t), (rf_sz, rf_t), (rp_sz, rp_t),
    ) = get_mla_metadata_info_v1(
        BATCH, DECODE_QLEN, NHEAD, dtypes.fp8, dtypes.fp8,
        is_sparse=False, fast_mode=True,
        num_kv_splits=max_split, intra_batch_mode=False,
    )
    wmd = torch.empty(wmd_sz, dtype=wmd_t, device=device)
    wi = torch.empty(wi_sz, dtype=wi_t, device=device)
    ws = torch.empty(ws_sz, dtype=ws_t, device=device)
    ri = torch.empty(ri_sz, dtype=ri_t, device=device)
    rf = torch.empty(rf_sz, dtype=rf_t, device=device)
    rp = torch.empty(rp_sz, dtype=rp_t, device=device)

    get_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page_lens,
        NHEAD // NHEAD_KV, NHEAD_KV, False,
        wmd, ws, wi, ri, rf, rp,
        page_size=PAGE_SIZE, kv_granularity=max(PAGE_SIZE, 16),
        max_seqlen_qo=DECODE_QLEN, uni_seqlen_qo=DECODE_QLEN, fast_mode=True,
        max_split_per_batch=max_split, intra_batch_mode=False,
        dtype_q_nope=dtypes.fp8, dtype_kv_nope=dtypes.fp8,
    )

    out = torch.empty((BATCH * DECODE_QLEN, NHEAD, V_HEAD_DIM),
                      dtype=torch.bfloat16, device=device)
    mla_decode_fwd(
        q,
        kv_buffer_fp8.view(num_page, PAGE_SIZE, NHEAD_KV, QK_HEAD_DIM),
        out,
        qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
        DECODE_QLEN, PAGE_SIZE, NHEAD_KV, SM_SCALE,
        num_kv_splits=max_split,
        work_meta_data=wmd, work_indptr=wi, work_info_set=ws,
        reduce_indptr=ri, reduce_final_map=rf, reduce_partial_map=rp,
        q_scale=torch.ones(1, device=device),
        kv_scale=torch.ones(1, device=device),
    )
    torch.cuda.synchronize()
    return out.clone()


def _maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


def main():
    gfx = get_gfx()
    print(f"[det] gfx={gfx} nhead={NHEAD} fp8/fp8 page_size={PAGE_SIZE} "
          f"batch={BATCH} ctx={CTX}")
    torch.manual_seed(0)
    dev = "cuda"
    q = torch.randn(BATCH * DECODE_QLEN, NHEAD, QK_HEAD_DIM, device=dev).to(dtypes.fp8)
    kv = torch.randn(CTX, PAGE_SIZE, NHEAD_KV, QK_HEAD_DIM, device=dev).to(dtypes.fp8)

    N = 8
    det_env = os.getenv("AITER_MLA_DECODE_DETERMINISTIC", "0") not in ("0", "")

    o1a = _run_decode(q, kv, 1)
    o1b = _run_decode(q, kv, 1)
    oNa = _run_decode(q, kv, N)
    oNb = _run_decode(q, kv, N)

    d_s1 = _maxdiff(o1a, o1b)
    d_sN = _maxdiff(oNa, oNb)
    d_1vN = _maxdiff(o1a, oNa)

    print(f"[det] split=1 run-to-run  max|d| = {d_s1:.3e}   (expect 0: single-split reduce is a no-op)")
    print(f"[det] split={N} run-to-run  max|d| = {d_sN:.3e}")
    print(f"[det] split=1 vs split={N} max|d| = {d_1vN:.3e}   "
          f"({'DIFFER -> split-KV changes numerics (the bug mechanism)' if d_1vN > 0 else 'identical'})")

    if det_env:
        # With the workaround, get_mla_metadata_v1 clamps to a single split, so
        # even the "N" request collapses onto the split=1 baseline.
        print(f"[det] AITER_MLA_DECODE_DETERMINISTIC set: split={N} now matches split=1? "
              f"max|d| = {_maxdiff(oNa, o1a):.3e}")
        ok = d_s1 == 0 and _maxdiff(oNa, o1a) == 0
        print(f"[det] {'PASS' if ok else 'FAIL'}: deterministic single-split path.")
        return 0 if ok else 1

    ok_baseline = d_s1 == 0
    print(f"[det] baseline single-split deterministic: {'PASS' if ok_baseline else 'FAIL'}")
    print("[det] Re-run with AITER_MLA_DECODE_DETERMINISTIC=1 to confirm the "
          "workaround forces the deterministic single-split path for all requests.")
    return 0 if ok_baseline else 1


if __name__ == "__main__":
    sys.exit(main())
