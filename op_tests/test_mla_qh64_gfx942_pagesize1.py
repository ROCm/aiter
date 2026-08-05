#!/usr/bin/env python3
"""
Regression repro: native QH64 fp8 persistent MLA-decode kernel (PR #3188)
GPU-memory-access-faults on gfx942 (MI300X) at page_size=1.

BUG
---
PR #3188 ("Add native MLA QH64 fp8 persistent decode kernel for gfx942")
routes gfx942 + nhead==64 + fp8/fp8 + max_seqlen_q==1 to
    hsa/gfx942/mla/mla_a8w8_qh64_qseqlen1_gqaratio64_v3_ps.co
instead of the pre-#3188 qh16 fold. That kernel was validated ONLY at
page_size=64. At **page_size=1** (the KV layout vLLM MLA serves with,
block_size=1) it does an out-of-bounds GPU access and faults on the first
decode forward:

    [aiter] LoadKernel: _ZN5aiter39mla_a8w8_qh64_qseqlen1_gqaratio64_v3_psE ...
            .../gfx942/mla/mla_a8w8_qh64_qseqlen1_gqaratio64_v3_ps.co
    Memory access fault by GPU node-N (Agent handle: 0x...) on address 0x7f...000.
    Reason: Unknown.
    -> Aborted (exit 134)

Reproduced on multiple aiter builds carrying #3188 (v0.1.18, origin/main tip)
and on the GLM-5.1-FP8 (GlmMoeDsaForCausalLM) 1P1D disagg workload where the
per-rank head grouping is gqa_ratio=64 (DP8/TP1).

FIX (this branch)
-----------------
aiter/mla.py: gate the gfx942 native-qh64 dispatch clause to the validated
page_size==64. Other page sizes (page_size=1) fall through to the qh16 fold
(pre-#3188 path), which runs without fault. gfx950 behavior is unchanged.

With the fix applied this test PASSES (folds to
mla_a8w8_qh16_qseqlen1_gqaratio16_ps.co and completes). Without the fix
(stock #3188) it GPU-faults / aborts.

Run on a single MI300X (gfx942):
    python3 op_tests/test_mla_qh64_gfx942_pagesize1.py

NOTE: this is a *fault* regression test (the crash is deterministic and
unambiguous). It uses random fp8 inputs, so the printed 'finite=' flag is a
coarse liveness signal, not a numerical-correctness check; for golden-reference
numerics use test_mla_persistent.py at page_size=1, nhead=64, fp8/fp8.
"""
import sys
import torch

import aiter
from aiter import dtypes, get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.mla import mla_decode_fwd
from aiter.jit.utils.chip_info import get_gfx

# GLM-5.1-FP8 per-rank decode geometry (DP8/TP1 => gqa_ratio=64).
NUM_HEADS = 64
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
V_HEAD_DIM = KV_LORA_RANK                        # 512
PAGE_SIZE = 1        # <-- the case #3188 never validated; faults on gfx942
MAX_SEQLEN_Q = 1     # decode


def build_decode(batch, kv_len, device="cuda"):
    total_s = batch * MAX_SEQLEN_Q
    total_kv = batch * kv_len

    q = torch.randn(total_s, NUM_HEADS, QK_HEAD_DIM, device=device).to(dtypes.fp8)
    kv_buffer = torch.randn(
        total_kv, PAGE_SIZE, NUM_KV_HEADS, QK_HEAD_DIM, device=device
    ).to(dtypes.fp8)
    o = torch.empty(total_s, NUM_HEADS, V_HEAD_DIM, dtype=dtypes.bf16, device=device)

    qo_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device) * MAX_SEQLEN_Q
    kv_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device) * kv_len
    kv_indices = torch.arange(total_kv, dtype=torch.int32, device=device)
    kv_last_page_lens = torch.ones(batch, dtype=torch.int32, device=device)

    (
        (work_meta_data_size, work_meta_data_type),
        (work_indptr_size, work_indptr_type),
        (work_info_set_size, work_info_set_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = get_mla_metadata_info_v1(
        batch, MAX_SEQLEN_Q, NUM_HEADS, dtypes.fp8, dtypes.fp8,
        is_sparse=True, fast_mode=True,
    )
    work_meta_data = torch.empty(work_meta_data_size, dtype=work_meta_data_type, device=device)
    work_indptr = torch.empty(work_indptr_size, dtype=work_indptr_type, device=device)
    work_info_set = torch.empty(work_info_set_size, dtype=work_info_set_type, device=device)
    reduce_indptr = torch.empty(reduce_indptr_size, dtype=reduce_indptr_type, device=device)
    reduce_final_map = torch.empty(reduce_final_map_size, dtype=reduce_final_map_type, device=device)
    reduce_partial_map = torch.empty(reduce_partial_map_size, dtype=reduce_partial_map_type, device=device)

    get_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page_lens,
        NUM_HEADS, NUM_KV_HEADS, True,
        work_meta_data, work_info_set, work_indptr,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        page_size=PAGE_SIZE, kv_granularity=16,
        max_seqlen_qo=MAX_SEQLEN_Q, uni_seqlen_qo=MAX_SEQLEN_Q, fast_mode=True,
    )
    torch.cuda.synchronize()

    meta = dict(
        work_meta_data=work_meta_data, work_indptr=work_indptr,
        work_info_set=work_info_set, reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map, reduce_partial_map=reduce_partial_map,
    )
    return q, kv_buffer, o, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens, meta


def main():
    gfx = get_gfx()
    print(f"[test] gfx={gfx} nhead={NUM_HEADS} gqa_ratio={NUM_HEADS // NUM_KV_HEADS} "
          f"q/kv=fp8 page_size={PAGE_SIZE} max_seqlen_q={MAX_SEQLEN_Q}")
    if gfx != "gfx942":
        print(f"[test] SKIP: fault is gfx942-specific; running on {gfx}.")
        return 0

    sm_scale = 1.0 / (QK_HEAD_DIM ** 0.5)
    for batch, kv_len in [(1, 512), (4, 1024), (16, 2048)]:
        print(f"[test] mla_decode_fwd persistent batch={batch} kv_len={kv_len} ...", flush=True)
        (q, kv_buffer, o, qo_indptr, kv_indptr,
         kv_indices, kv_last_page_lens, meta) = build_decode(batch, kv_len)
        mla_decode_fwd(
            q, kv_buffer, o,
            qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
            MAX_SEQLEN_Q,
            page_size=PAGE_SIZE, nhead_kv=NUM_KV_HEADS, sm_scale=sm_scale,
            q_scale=torch.ones(1, device=q.device),
            kv_scale=torch.ones(1, device=q.device),
            **meta,
        )
        torch.cuda.synchronize()  # OOB fault (stock #3188) surfaces here
        print(f"[test]   returned batch={batch} kv_len={kv_len} out={tuple(o.shape)} "
              f"finite={bool(torch.isfinite(o.float()).all())}")

    print("[test] PASS: no GPU fault (page_size=1 folded to qh16; native qh64 gated "
          "to page_size=64 on gfx942).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
