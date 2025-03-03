# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

# user interface

import torch
import aiter


def mla_decode_fwd(
    q,
    kv_buffer,
    o,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    sm_scale=None,  # 1.0 / (qk_head_dim**0.5)
    logit_cap=0.0,
    num_kv_splits=None,  # for experts only!!!
):
    device = q.device
    assert logit_cap <= 0, f"{logit_cap=} is not support yet"
    if sm_scale is None:
        sm_scale = 1.0 / (qk_head_dim**0.5)

    num_page, page_size, nhead_kv, qk_head_dim = kv_buffer.shape
    bs, nhead, v_head_dim = o.shape

    if num_kv_splits is None:
        device_properties = torch.cuda.get_device_properties(device)
        cu_num = device_properties.multi_processor_count
        num_kv_splits = max(1, cu_num//bs)

    logits = torch.empty(
        (bs, num_kv_splits, nhead, v_head_dim), dtype=torch.float, device=device)
    attn_lse = torch.empty(
        (bs, num_kv_splits, nhead,  1), dtype=torch.float, device=device)

    aiter.mla_stage1_asm_fwd(
        q,
        kv_buffer,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        sm_scale,
        logits,
        attn_lse,
    )

    from aiter.ops.triton import decode_mla
    import triton
    Lv = v_head_dim
    BLOCK_DV = triton.next_power_of_2(Lv)
    grid = (bs, nhead)
    extra_kargs = {"waves_per_eu": 4,
                   "matrix_instr_nonkdim": 16, "kpack": 2}
    decode_mla._fwd_kernel_stage2_asm[grid](
        logits,
        attn_lse,
        o,
        kv_indptr,
        attn_lse.stride(0),
        attn_lse.stride(2),
        attn_lse.stride(1),
        o.stride(0),
        o.stride(1),
        NUM_KV_SPLITS=num_kv_splits,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        mgc=64,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )
    return logits, attn_lse
