#!/usr/bin/env python3
"""
Minimal MLA profiling script — no correctness checks, no Python overhead during kernel runs.

Usage:
    # Warm up JIT first (run once without profiler):
    MLA_FORCE_QH16_FOLD=0 python op_tests/profile_mla.py --batch 1 --ctx 8192
    MLA_FORCE_QH16_FOLD=1 python op_tests/profile_mla.py --batch 1 --ctx 8192

    # Then profile:
    MLA_FORCE_QH16_FOLD=0 rocprofv3 --stats -o qh32_b1.csv -- \
        python op_tests/profile_mla.py --batch 1 --ctx 8192 --iters 100
    MLA_FORCE_QH16_FOLD=1 rocprofv3 --stats -o fold_b1.csv -- \
        python op_tests/profile_mla.py --batch 1 --ctx 8192 --iters 100
"""

import argparse
import os
import torch
import aiter
import aiter.mla
from aiter import dtypes

torch.set_default_device("cuda")

NHEAD        = 32
NHEAD_KV     = 1
KV_LORA_RANK = 512
QK_ROPE_DIM  = 64
QK_HEAD_DIM  = KV_LORA_RANK + QK_ROPE_DIM   # 576
V_HEAD_DIM   = KV_LORA_RANK                  # 512
PAGE_SIZE    = 1
DECODE_QLEN  = 1
MAX_SPLIT    = 32
SM_SCALE     = 1.0 / (QK_HEAD_DIM ** 0.5)


def build_inputs(batch: int, ctx: int):
    dtype   = dtypes.fp8
    kvtype  = dtypes.fp8

    seq_lens_kv    = torch.full((batch,), ctx, dtype=torch.int32)
    kv_block_nums  = torch.full((batch,), (ctx + PAGE_SIZE - 1) // PAGE_SIZE, dtype=torch.int32)
    kv_last_page   = torch.full((batch,), ctx % PAGE_SIZE or PAGE_SIZE, dtype=torch.int32)
    kv_indptr      = torch.zeros(batch + 1, dtype=torch.int32)
    kv_indptr[1:]  = torch.cumsum(kv_block_nums, dim=0)
    num_page       = int(kv_indptr[-1].item())
    kv_indices     = torch.arange(num_page, dtype=torch.int32)

    qo_indptr      = torch.zeros(batch + 1, dtype=torch.int32)
    qo_indptr[1:]  = DECODE_QLEN  # each batch item has 1 query token

    q   = torch.randn(batch, NHEAD, QK_HEAD_DIM, dtype=torch.float16).to(dtype)
    kv  = torch.randn(num_page, PAGE_SIZE, NHEAD_KV, QK_HEAD_DIM, dtype=torch.float16).to(kvtype)
    out = torch.empty(batch, NHEAD, V_HEAD_DIM, dtype=torch.bfloat16)

    kv_scale = torch.ones([1], dtype=torch.float32)

    # Allocate metadata tensors
    (
        (wmd_sz, wmd_ty),
        (wi_sz,  wi_ty),
        (wis_sz, wis_ty),
        (ri_sz,  ri_ty),
        (rfm_sz, rfm_ty),
        (rpm_sz, rpm_ty),
    ) = aiter.get_mla_metadata_info_v1(
        batch, DECODE_QLEN, NHEAD, dtype, kvtype,
        is_sparse=False, fast_mode=True, num_kv_splits=MAX_SPLIT, intra_batch_mode=False,
    )

    work_meta_data   = torch.empty(wmd_sz, dtype=wmd_ty)
    work_indptr      = torch.empty(wi_sz,  dtype=wi_ty)
    work_info_set    = torch.empty(wis_sz, dtype=wis_ty)
    reduce_indptr    = torch.empty(ri_sz,  dtype=ri_ty)
    reduce_final_map = torch.empty(rfm_sz, dtype=rfm_ty)
    reduce_partial_map = torch.empty(rpm_sz, dtype=rpm_ty)

    aiter.get_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page,
        NHEAD // NHEAD_KV, NHEAD_KV, False,
        work_meta_data, work_info_set, work_indptr,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        page_size=PAGE_SIZE,
        kv_granularity=max(PAGE_SIZE, 16),
        max_seqlen_qo=int(DECODE_QLEN),
        uni_seqlen_qo=DECODE_QLEN,
        fast_mode=True,
        max_split_per_batch=MAX_SPLIT,
        intra_batch_mode=False,
        dtype_q=dtype,
        dtype_kv=kvtype,
    )

    return dict(
        q=q, kv=kv, out=out,
        qo_indptr=qo_indptr, kv_indptr=kv_indptr,
        kv_indices=kv_indices, kv_last_page=kv_last_page,
        kv_scale=kv_scale,
        work_meta_data=work_meta_data, work_indptr=work_indptr,
        work_info_set=work_info_set, reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map, reduce_partial_map=reduce_partial_map,
    )


def run(inputs: dict, iters: int):
    i = inputs
    for _ in range(iters):
        aiter.mla.mla_decode_fwd(
            i["q"], i["kv"], i["out"],
            i["qo_indptr"], i["kv_indptr"], i["kv_indices"], i["kv_last_page"],
            DECODE_QLEN, PAGE_SIZE, NHEAD_KV, SM_SCALE,
            num_kv_splits=MAX_SPLIT,
            work_meta_data=i["work_meta_data"],
            work_indptr=i["work_indptr"],
            work_info_set=i["work_info_set"],
            reduce_indptr=i["reduce_indptr"],
            reduce_final_map=i["reduce_final_map"],
            reduce_partial_map=i["reduce_partial_map"],
            intra_batch_mode=False,
            kv_scale=i["kv_scale"],
        )
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--ctx",   type=int, required=True)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    fold = os.environ.get("MLA_FORCE_QH16_FOLD", "0") == "1"
    kernel = "qh16-fold" if fold else "qh32-native"
    print(f"[profile_mla] kernel={kernel}  batch={args.batch}  ctx={args.ctx}  iters={args.iters}")

    inputs = build_inputs(args.batch, args.ctx)

    # Warmup — triggers JIT and fills GPU caches, not captured by profiler
    print(f"[profile_mla] warming up ({args.warmup} iters)...")
    run(inputs, args.warmup)

    print(f"[profile_mla] running ({args.iters} iters) — profiler should capture this...")
    run(inputs, args.iters)
    print("[profile_mla] done.")


if __name__ == "__main__":
    main()
