# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter.test_common import checkAllclose, benchmark, run_perftest
from aiter import dtypes
import random
import itertools
import argparse
import os
import numpy as np
import pandas as pd

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)

# current supported case in ps decode MLA: mtp == 0, 1, 2, 3 (decode_qlen = 1, 2, 3, 4)
# qdtype bf16, kdtype bf16: nhead16
# qdtype fp8, kdtype fp8: nhead16, nhead128
# qdtype fp8, kdtype fp8: nhead32, max_seqlen_qo=4
# qdtype fp8, kdtype bf16: nhead16


def check_support(dtype, kv_dtype, nhead):
    if dtype == dtypes.fp8 and kv_dtype == dtypes.bf16:
        return False
    if dtype == dtypes.bf16 and nhead == 32:
        return False
    return True


def init_3buffer_kv_cache(
    num_page: int,
    page_size: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    scale_dim: int,
) -> tuple:
    """
    Initialize KV cache for 3BUFFER layout with FP8 quantization.

    Generates random KV cache data and applies per-channel quantization to the nope buffer.

    Args:
        num_page: Number of pages
        page_size: Size of each page (block size)
        kv_lora_rank: Rank of KV LoRA (nope dimension)
        qk_rope_head_dim: Dimension of RoPE (rope dimension)
        scale_dim: Number of scale factors per nope buffer

    Returns:
        tuple containing:
            - kv_buffer: Concatenated buffer (BF16), shape (num_page, page_size, 1, kv_lora_rank + qk_rope_head_dim)
            - kv_nope_buffer_fp8: Quantized nope buffer (FP8), shape (num_page, page_size, 1, kv_lora_rank)
            - kv_nope_scale_factors_fp32: Scale factors (FP32), shape (num_page, page_size, 1, scale_dim)
            - kv_rope_buffer_bf16: Rope buffer (BF16), shape (num_page, page_size, 1, qk_rope_head_dim)
            - kv_nope_buffer_fp32: Original nope buffer (FP32), shape (num_page, page_size, 1, kv_lora_rank)
    """
    assert (
        kv_lora_rank % scale_dim == 0
    ), f"kv_lora_rank ({kv_lora_rank}) must be divisible by scale_dim ({scale_dim})"

    kv_nope_buffer_fp32 = torch.randn(
        (num_page, page_size, 1, kv_lora_rank), dtype=torch.float32
    )
    kv_rope_buffer_bf16 = torch.randn(
        (num_page, page_size, 1, qk_rope_head_dim),
        dtype=torch.bfloat16,
    )

    # Create full KV buffer (for golden reference without quantization)
    kv_buffer = torch.cat(
        [kv_nope_buffer_fp32.to(torch.bfloat16), kv_rope_buffer_bf16], dim=-1
    )

    # Generate random scale factors
    scale_values = [1.0, 2.0, 4.0, 8.0]
    # scale_values = [1.0, 1.0, 1.0, 1.0]
    scale_indices = torch.randint(
        0, len(scale_values), size=(num_page, page_size, 1, scale_dim)
    )
    kv_nope_scale_factors_fp32 = torch.tensor(
        [scale_values[idx] for idx in scale_indices.flatten()], dtype=torch.float32
    ).reshape(num_page, page_size, 1, scale_dim)

    # Apply per-channel scaling and quantize to FP8
    kv_nope_scaled_buffer = kv_nope_buffer_fp32.reshape(
        num_page, page_size, 1, scale_dim, kv_lora_rank // scale_dim
    ) / kv_nope_scale_factors_fp32.reshape(num_page, page_size, 1, scale_dim, 1)

    kv_nope_buffer_fp8 = kv_nope_scaled_buffer.reshape(
        num_page, page_size, 1, kv_lora_rank
    ).to(dtypes.fp8)

    return (
        kv_buffer,
        kv_nope_buffer_fp8,
        kv_nope_scale_factors_fp32,
        kv_rope_buffer_bf16,
        kv_nope_buffer_fp32,
    )


def split_3buffer_kv_cache(
    kv_buffer_bytes: torch.Tensor,
    page_size: int,
    nhead_kv: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    scale_dim: int,
) -> tuple:
    """
    Split concatenated KV cache buffer back into 3 separate buffers.

    This is the inverse operation of concatenating after flattening last 3 dimensions.

    Args:
        kv_buffer_bytes: Concatenated buffer (uint8), shape (num_page, page_size*656)
                        where 656 = 512(nope) + 16(scale) + 128(rope)
        page_size: Size of each page (block size)
        nhead_kv: Number of heads in the KV cache
        kv_lora_rank: Rank of KV LoRA (nope dimension)
        qk_rope_head_dim: Dimension of RoPE (rope dimension)
        scale_dim: Number of scale factors per nope buffer

    Returns:
        tuple containing:
            - kv_nope_buffer_fp8: Quantized nope buffer (FP8), shape (num_page, page_size, 1, kv_lora_rank)
            - kv_nope_scale_factors_fp32: Scale factors (FP32), shape (num_page, page_size, 1, scale_dim)
            - kv_rope_buffer_bf16: Rope buffer (BF16), shape (num_page, page_size, 1, qk_rope_head_dim)
    """
    num_page = kv_buffer_bytes.shape[0]

    nope_total_bytes = page_size * nhead_kv * kv_lora_rank * 1  # FP8: 1 byte/elem
    scale_total_bytes = page_size * nhead_kv * scale_dim * 4  # FP32: 4 bytes/elem
    rope_total_bytes = page_size * nhead_kv * qk_rope_head_dim * 2  # BF16: 2 bytes/elem

    nope_flat = kv_buffer_bytes[:, 0:nope_total_bytes]
    scale_flat = kv_buffer_bytes[
        :, nope_total_bytes : nope_total_bytes + scale_total_bytes
    ]
    rope_flat = kv_buffer_bytes[
        :,
        nope_total_bytes
        + scale_total_bytes : nope_total_bytes
        + scale_total_bytes
        + rope_total_bytes,
    ]

    nope_bytes = nope_flat.reshape(num_page, page_size, nhead_kv, kv_lora_rank * 1)
    scale_bytes = scale_flat.reshape(num_page, page_size, nhead_kv, scale_dim * 4)
    rope_bytes = rope_flat.reshape(num_page, page_size, nhead_kv, qk_rope_head_dim * 2)

    # Convert bytes back to original dtypes
    kv_nope_buffer_fp8 = (
        nope_bytes.contiguous()
        .view(dtypes.fp8)
        .reshape(num_page, page_size, nhead_kv, kv_lora_rank)
    )

    kv_nope_scale_factors_fp32 = (
        scale_bytes.contiguous()
        .view(torch.float32)
        .reshape(num_page, page_size, nhead_kv, scale_dim)
    )

    kv_rope_buffer_bf16 = (
        rope_bytes.contiguous()
        .view(torch.bfloat16)
        .reshape(num_page, page_size, nhead_kv, qk_rope_head_dim)
    )

    return kv_nope_buffer_fp8, kv_nope_scale_factors_fp32, kv_rope_buffer_bf16


def cal_diff(
    x: torch.Tensor, y: torch.Tensor, name: str, use_fp8: bool = False
) -> None:
    x, y = x.double(), y.double()
    # RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    # amax_diff = (x - y).abs().max().item()
    # print(f"{name}: {cos_diff=}, {RMSE=}, {amax_diff=}")
    if use_fp8:
        assert cos_diff < 3e-2
    else:
        assert cos_diff < 1e-5


def debug_per_batch_diff(
    out_ref, out_asm, qo_indptr, seq_lens_kv, batch_size, label="out"
):
    """Per-batch max abs delta analysis, prints which batch has nan/max error."""
    print(f"\n{'='*60}")
    print(f"[debug] Per-batch diff analysis for '{label}'")
    print(f"{'='*60}")
    worst_batch = -1
    worst_delta = -1.0
    for b in range(batch_size):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        ref_b = out_ref[q_start:q_end].double()
        asm_b = out_asm[q_start:q_end].double()
        diff_b = (ref_b - asm_b).abs()
        has_nan = torch.isnan(diff_b).any().item()
        has_inf = torch.isinf(diff_b).any().item()
        nan_count = torch.isnan(diff_b).sum().item()
        finite_diff = diff_b[torch.isfinite(diff_b)]
        max_delta = finite_diff.max().item() if finite_diff.numel() > 0 else float("nan")
        kv_sl = int(seq_lens_kv[b].item())

        flag = ""
        if has_nan or has_inf:
            flag = " *** NAN/INF ***"
        elif max_delta > 0.01:
            flag = " *** HIGH ***"

        print(
            f"  batch={b:3d}  kv_seqlen={kv_sl:6d}  "
            f"max_abs_delta={max_delta:.6f}  nan_count={nan_count}{flag}"
        )
        if max_delta > worst_delta and not (has_nan and finite_diff.numel() == 0):
            worst_delta = max_delta
            worst_batch = b
    if worst_batch >= 0:
        print(
            f"[debug] Worst finite batch={worst_batch}, "
            f"kv_seqlen={int(seq_lens_kv[worst_batch].item())}, "
            f"max_abs_delta={worst_delta:.6f}"
        )
    print(f"{'='*60}\n")


def dump_tensor_to_txt(tensor, filepath):
    """Dump tensor to txt: first line is shape/dtype, then flattened values."""
    if tensor is None:
        print(f"[debug] skip dump {filepath}  (tensor is None)")
        return
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    t = tensor.detach().float().cpu()
    with open(filepath, "w") as f:
        f.write(f"# shape={list(tensor.shape)} dtype={tensor.dtype}\n")
        flat = t.reshape(-1).numpy()
        for i, v in enumerate(flat):
            f.write(f"{i} {v:.8e}\n")
    print(f"[debug] dumped {filepath}  shape={list(tensor.shape)} dtype={tensor.dtype}")


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    dtype,
    is_causal=True,
    is_fp8_q=False,
    is_fp8_kvc=False,
    q_scale=None,
    kv_scale=None,
):
    if is_fp8_q and q_scale is not None:
        scale *= q_scale
    if is_fp8_kvc and kv_scale is not None:
        scale *= kv_scale
    attn_weights = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale

    if is_causal:
        s_q = query.shape[0]
        s_k = key.shape[0]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weights += attn_bias

    lse = attn_weights.logsumexp(dim=-1)
    m = attn_weights.max(-1).values
    attn_weights_exp = torch.exp(attn_weights - m.unsqueeze(-1))
    l = attn_weights_exp.sum(-1)  # noqa: E741
    if is_fp8_q:
        attn_weights_fp8 = attn_weights_exp.to(dtypes.fp8)
        attn_weights_exp = attn_weights_fp8.to(torch.float)

    out = torch.einsum("hqk,khd->qhd", attn_weights_exp.float(), value.float())
    out = out / l.transpose(0, 1).unsqueeze(-1)
    if is_fp8_kvc and kv_scale is not None:
        out *= kv_scale
    return out.to(dtype), lse


def torch_mla_extend_3buffer(
    q,  # [total_q, nheads, headdim_q]
    kvc_cache,  # [num_page, page_size*(nhead_kv*(kv_lora_rank+scale_dim+qk_rope_head_dim))]
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    page_size,
    nhead_kv,
    sm_scale,
    kv_lora_rank,
    qk_rope_head_dim,
    dtype,
    is_causal=True,
    q_scale=None,
    kv_scale=None,
    scale_dim=4,
):
    num_page = kvc_cache.shape[0]
    kv_nope_buffer_fp8, kv_nope_scale_factors_fp32, kv_rope_buffer_bf16 = (
        split_3buffer_kv_cache(
            kvc_cache, page_size, nhead_kv, kv_lora_rank, qk_rope_head_dim, scale_dim
        )
    )

    kv_nope_buffer_fp32 = kv_nope_buffer_fp8.to(torch.float32).reshape(
        num_page, page_size, nhead_kv, scale_dim, -1
    ) * kv_nope_scale_factors_fp32.reshape(num_page, page_size, nhead_kv, scale_dim, 1)
    kvc_cache_bf16 = torch.cat(
        [
            kv_nope_buffer_fp32.reshape(num_page, page_size, nhead_kv, kv_lora_rank).to(
                torch.bfloat16
            ),
            kv_rope_buffer_bf16,
        ],
        dim=-1,
    )

    return torch_mla_extend(
        q,
        kvc_cache_bf16,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        sm_scale,
        kv_lora_rank,
        qk_rope_head_dim,
        dtype,
        is_causal,
        q_scale,
        kv_scale,
    )


def torch_mla_extend(
    q,  # [total_q, nheads, headdim_q]
    kvc_cache,  # [num_page, page_size, nhead_kv, qk_head_dim]
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    sm_scale,
    kv_lora_rank,
    qk_rope_head_dim,
    dtype,
    is_causal=True,
    q_scale=None,
    kv_scale=None,
):
    num_page, page_size, nhead_kv, _ = kvc_cache.shape
    is_fp8_q = q.dtype == dtypes.fp8
    is_fp8_kvc = kvc_cache.dtype == dtypes.fp8

    if is_fp8_q:
        q = q.to(torch.float)

    if is_fp8_kvc:
        kvc_cache = kvc_cache.to(torch.float)

    qs = torch.tensor_split(q, qo_indptr.tolist()[1:])
    kvc = torch.index_select(kvc_cache, 0, kv_indices)
    kvs = torch.tensor_split(kvc, kv_indptr.tolist()[1:])
    bs = qo_indptr.shape[0] - 1

    os = []
    lses = []
    for i in range(bs):
        cur_num_page = kvs[i].shape[0]
        real_kv_seq_len = (cur_num_page - 1) * page_size + kv_last_page_lens.tolist()[i]
        kvc = kvs[i].flatten(0, 1)[:real_kv_seq_len,]
        q = qs[i]
        k = kvc
        v, _ = torch.split(kvc, [kv_lora_rank, qk_rope_head_dim], dim=-1)
        o, lse = ref_masked_attention(
            q,
            k,
            v,
            sm_scale,
            dtype,
            is_causal=is_causal,
            is_fp8_q=is_fp8_q,
            is_fp8_kvc=is_fp8_kvc,
            q_scale=q_scale,
            kv_scale=kv_scale,
        )
        os.append(o)
        lses.append(lse)
    o = torch.concat(os)
    lse = torch.concat(lses).transpose(0, 1)
    return o, lse


@benchmark()
def test_mla(
    ctx_lens,
    batch_size,
    nhead,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
    dtype,
    kvtype,
    page_size,
    varlen,
    decode_qlen,
    max_split_per_batch,
    non_persistent_mode,
    paged_layout,
    scale_dim,
):
    ret = {}

    out_dtype = torch.bfloat16
    kv_max_sz = (
        65536 * 32
    )  # calculated by rest of mem after weight loaded in frameworks
    num_page = (kv_max_sz + page_size - 1) // page_size

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    seq_lens_qo = torch.empty(batch_size, dtype=torch.int)
    seq_lens_kv = torch.empty(batch_size, dtype=torch.int)
    kv_block_nums = torch.empty(batch_size, dtype=torch.int)
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)

    a = [742, 1448, 2163, 2898, 3686, 4417, 5138, 5882, 6662,
     7396, 8137, 8876, 9614, 10361, 11089, 11816, 12585, 13315,
     14023, 14767, 15506, 16228, 16962, 17679, 18395, 19138,
     19871, 20612, 21356, 22085, 22800, 23541]
 
    b = [0, 742, 1448, 2163, 2898, 3686, 4417, 5138, 5882, 6662,
     7396, 8137, 8876, 9614, 10361, 11089, 11816, 12585,
     13315, 14023, 14767, 15506, 16228, 16962, 17679,
     18395, 19138, 19871, 20612, 21356, 22085, 22800]

    # [742, 706, 715, 735, 788, 731, 721, 744, 780, 734, 741, 739, 738, 747, 728, 727, 769, 730, 708, 744, 739, 722, 734, 717, 716, 743, 733, 741, 744, 729, 715, 741]
 
    seq_lens_kv = torch.tensor([x - y for x, y in zip(a, b)])
    seq_lens_qo.fill_(4)

    if varlen:
        for i in range(batch_size):
            # seq_lens_kv[i] = max(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens)
            # seq_lens_kv[i] = random.uniform(5, ctx_lens)
            # seq_lens_qo[i] = max(
            #     min(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens), 1
            # )
            kv_block_nums[i] = (seq_lens_kv[i] + page_size - 1) // page_size
            if seq_lens_kv[i] % page_size == 0:
                kv_last_page_lens[i] = page_size
            else:
                kv_last_page_lens[i] = seq_lens_kv[i] % page_size
    else:
        seq_lens_kv.fill_(ctx_lens)
        seq_lens_qo.fill_(ctx_lens)
        kv_block_nums.fill_((ctx_lens + page_size - 1) // page_size)
        if ctx_lens % page_size == 0:
            kv_last_page_lens.fill_(page_size)
        else:
            kv_last_page_lens.fill_(ctx_lens % page_size)

    kv_indptr[1 : batch_size + 1] = torch.cumsum(kv_block_nums, dim=0)
    num_page = kv_indptr[-1].item()
    kv_indices = torch.randperm(num_page, dtype=torch.int)
    qo_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
    max_seqlen_qo = seq_lens_qo.max().item()
    # max_seqlen_kv = seq_lens_kv.max().item()
    # total_qo = qo_indptr[-1].item()
    total_kv = seq_lens_kv.sum().item()

    kv_buffer = torch.randn(
        (num_page, page_size, 1, kv_lora_rank + qk_rope_head_dim),
        dtype=torch.bfloat16,
    )

    kv_nope_scale_factors_fp32 = None
    kv_nope_buffer_fp8 = None
    kv_rope_buffer_bf16 = None

    if paged_layout == "3BUFFER":
        (
            kv_buffer,
            kv_nope_buffer_fp8,
            kv_nope_scale_factors_fp32,
            kv_rope_buffer_bf16,
            _,
        ) = init_3buffer_kv_cache(
            num_page, page_size, kv_lora_rank, qk_rope_head_dim, scale_dim
        )

    # for none absorb (mha)
    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    sm_scale = 1.0 / (qk_head_dim**0.5)

    # us_asm = None
    # if batch_size * ctx_lens * nhead < 32 * 8192 * 16:
    #     us_asm = test_absorb_prefill()
    torch.cuda.empty_cache()
    nhead_kv = 1

    # ############################## absorb: decode
    # seq_lens_qo = torch.randint(1, 5, (batch_size,), dtype=torch.int)
    # if nhead == 16 and decode_qlen != 1:
    #     return
    seq_lens_qo.fill_(decode_qlen)

    max_seqlen_qo = seq_lens_qo.max().item()
    qo_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
    total_q = qo_indptr[-1].item()
    q = torch.randn((total_q, nhead, qk_head_dim), dtype=torch.bfloat16)

    # troch implementation
    out_ref, lse_ref = torch_mla_extend(
        q,
        kv_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        sm_scale,
        kv_lora_rank,
        qk_rope_head_dim,
        is_causal=True,
        dtype=out_dtype,
    )

    # It is necessary to limit the size of the tensor in the DP mode
    # so reduce the split_num in the DP mode.
    if nhead >= 128:
        gpu = torch.cuda.current_device()
        device_properties = torch.cuda.get_device_properties(gpu)
        cu_num = device_properties.multi_processor_count
        max_split_per_batch = min(
            (cu_num + batch_size - 1) // batch_size, max_split_per_batch
        )

    (
        (work_meta_data_size, work_meta_data_type),
        (work_indptr_size, work_indptr_type),
        (work_info_set_size, work_info_set_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = aiter.get_mla_metadata_info_v1(
        batch_size,
        max_seqlen_qo,
        nhead,
        dtype,
        kvtype,
        is_sparse=False,
        fast_mode=True if not non_persistent_mode else False,
        num_kv_splits=max_split_per_batch,
        intra_batch_mode=non_persistent_mode,
    )

    # aiter implementation
    # the tensor's meaning please refer aiter/ops/attention.py
    work_meta_data = torch.empty(
        work_meta_data_size, dtype=work_meta_data_type, device="cuda"
    )
    work_indptr = torch.empty(work_indptr_size, dtype=work_indptr_type, device="cuda")
    work_info_set = torch.empty(
        work_info_set_size,
        dtype=work_info_set_type,
        device="cuda",
    )
    reduce_indptr = torch.empty(
        reduce_indptr_size, dtype=reduce_indptr_type, device="cuda"
    )
    reduce_final_map = torch.empty(
        reduce_final_map_size, dtype=reduce_final_map_type, device="cuda"
    )
    reduce_partial_map = torch.empty(
        reduce_partial_map_size, dtype=reduce_partial_map_type, device="cuda"
    )

    aiter.get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        kv_last_page_lens,
        nhead // nhead_kv,
        nhead_kv,
        False,
        work_meta_data,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        kv_granularity=max(page_size, 16),  # for qh32 kv split is disabled
        max_seqlen_qo=int(max_seqlen_qo),
        uni_seqlen_qo=decode_qlen,
        fast_mode=True if not non_persistent_mode else False,
        max_split_per_batch=max_split_per_batch,
        intra_batch_mode=non_persistent_mode,
        dtype_q=dtype,
        dtype_kv=kvtype,
    )
    # ===== DEBUG: dump & verify metadata =====
    def debug_verify_metadata():
        wi_cpu = work_indptr.cpu().to(torch.int32)
        wis_cpu = work_info_set.cpu().to(torch.int32)
        ri_cpu = reduce_indptr.cpu().to(torch.int32)
        rfm_cpu = reduce_final_map.cpu().to(torch.int32)
        rpm_cpu = reduce_partial_map.cpu().to(torch.int32)

        num_cu = wi_cpu.shape[0] - 1
        total_works = int(wi_cpu[-1].item())
        print(f"\n[metadata] num_cu={num_cu}, total_works={total_works}")

        per_batch_works = {b: [] for b in range(batch_size)}
        for w in range(total_works):
            row = wis_cpu[w]
            info = {
                "batch_idx": int(row[0].item()),
                "partial_qo_loc": int(row[1].item()),
                "qo_start": int(row[2].item()),
                "qo_end": int(row[3].item()),
                "kv_start": int(row[4].item()),
                "kv_end": int(row[5].item()),
                "kv_offset": int(row[6].item()),
            }
            b = info["batch_idx"]
            if 0 <= b < batch_size:
                per_batch_works[b].append((w, info))

        for b in range(batch_size):
            works = per_batch_works[b]
            kv_sl = int(seq_lens_kv[b].item())
            kv_begin = int(kv_indptr[b].item())
            kv_end_expected = int(kv_indptr[b + 1].item())
            num_splits = len(works)

            if num_splits == 0:
                print(f"  batch={b:3d}  kv_seqlen={kv_sl:6d}  *** NO WORK ITEMS ***")
                continue

            kv_ranges = [(w["kv_start"], w["kv_end"]) for _, w in works]
            kv_ranges_sorted = sorted(kv_ranges, key=lambda x: x[0])
            first_start = kv_ranges_sorted[0][0]
            last_end = kv_ranges_sorted[-1][1]

            gaps = []
            for i in range(len(kv_ranges_sorted) - 1):
                if kv_ranges_sorted[i][1] < kv_ranges_sorted[i + 1][0]:
                    gaps.append(
                        (kv_ranges_sorted[i][1], kv_ranges_sorted[i + 1][0])
                    )

            flag = ""
            if first_start != kv_begin:
                flag += f" *** START_MISMATCH(expect={kv_begin},got={first_start}) ***"
            if last_end != kv_end_expected:
                flag += f" *** END_MISMATCH(expect={kv_end_expected},got={last_end}) ***"
            if gaps:
                flag += f" *** GAPS={gaps} ***"

            if flag or b == 16:
                print(
                    f"  batch={b:3d}  kv_seqlen={kv_sl:6d}  splits={num_splits}  "
                    f"kv_range=[{first_start},{last_end})  expect=[{kv_begin},{kv_end_expected}){flag}"
                )
                for w_idx, w in works:
                    print(
                        f"    work[{w_idx:4d}]: qo=[{w['qo_start']},{w['qo_end']})  "
                        f"kv=[{w['kv_start']},{w['kv_end']})  "
                        f"kv_offset={w['kv_offset']}  partial_qo_loc={w['partial_qo_loc']}"
                    )

        tot_qo_tiles = 0
        for b in range(batch_size):
            if per_batch_works[b]:
                tot_qo_tiles += 1
        print(f"[metadata] tot_qo_tiles={tot_qo_tiles}")
        print(f"[metadata] reduce_indptr (first {min(tot_qo_tiles+2, ri_cpu.shape[0])}): "
              f"{ri_cpu[:min(tot_qo_tiles+2, ri_cpu.shape[0])].tolist()}")

    debug_verify_metadata()
    print()

    def test_absorb_decode_bf16_fp8():
        out_asm = torch.empty((total_q, nhead, v_head_dim), dtype=out_dtype).fill_(-1)
        kv_buffer_fp8 = kv_buffer.to(kvtype)
        kv_scale = torch.ones([1], dtype=torch.float, device="cuda")

        out_ref_fp8, lse_ref_fp8 = torch_mla_extend(
            q,
            kv_buffer_fp8,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            sm_scale,
            kv_lora_rank,
            qk_rope_head_dim,
            dtype=out_dtype,
            is_causal=True,
            q_scale=None,
            kv_scale=kv_scale,
        )

        (attn_logits, attn_lse), us_asm_decode = run_perftest(
            aiter.mla.mla_decode_fwd,
            q,
            kv_buffer_fp8.view(num_page, page_size, nhead_kv, qk_head_dim),
            out_asm,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            max_seqlen_qo,
            page_size,
            nhead_kv,
            sm_scale,
            num_kv_splits=max_split_per_batch,
            work_meta_data=work_meta_data,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
            intra_batch_mode=non_persistent_mode,
            kv_scale=kv_scale,
        )

        err = checkAllclose(
            out_ref,
            out_asm,
            msg=f"mla_decode-absorb    [golden vs aiter_asm]: {us_asm_decode:>8.2f} us......",
        )
        checkAllclose(
            out_ref_fp8,
            out_asm,
            msg=f"mla_decode-absorb_fp8    [golden fp8 vs aiter_asm]: {us_asm_decode:>8.2f} us......",
        )
        return err, us_asm_decode

    def test_absorb_decode_bf16():
        kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
        out_asm = torch.empty((total_q, nhead, v_head_dim), dtype=out_dtype).fill_(-1)
        (attn_logits, attn_lse), us_asm_decode = run_perftest(
            aiter.mla.mla_decode_fwd,
            q,
            kv_buffer.view(num_page, page_size, nhead_kv, qk_head_dim),
            out_asm,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            max_seqlen_qo,
            page_size,
            nhead_kv,
            sm_scale,
            num_kv_splits=max_split_per_batch,
            work_meta_data=work_meta_data,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
            intra_batch_mode=non_persistent_mode,
        )

        # print(f"{out_ref.view(total_q, -1)=}")
        # print(f"{out_asm.view(total_q, -1)=}")
        # checkAllclose(logits_ref, attn_logits,
        #               msg=f'attn_logits [golden vs aiter_asm]')
        # checkAllclose(lse_ref, attn_lse, msg="attn_lse    [golden vs aiter_asm]")
        err = checkAllclose(
            out_ref,
            out_asm,
            msg=f"mla_decode-absorb    [golden vs aiter_asm]: {us_asm_decode:>8.2f} us......",
        )
        return err, us_asm_decode

    def test_absorb_decode_fp8():
        kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
        out_asm = torch.empty((total_q, nhead, v_head_dim), dtype=out_dtype).fill_(-1)

        q_fp8 = q.to(dtypes.fp8)
        q_scale = torch.ones([1], dtype=torch.float, device="cuda")

        kv_buffer_fp8 = kv_buffer.to(dtypes.fp8)
        kv_scale = torch.ones([1], dtype=torch.float, device="cuda")

        out_ref_fp8, lse_ref_fp8 = torch_mla_extend(
            q_fp8 if dtype == dtypes.fp8 else q,
            kv_buffer_fp8,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            sm_scale,
            kv_lora_rank,
            qk_rope_head_dim,
            dtype=out_dtype,
            is_causal=True,
            q_scale=None,
            kv_scale=kv_scale,
        )

        (attn_logits, attn_lse), us_asm_decode = run_perftest(
            aiter.mla.mla_decode_fwd,
            q_fp8 if dtype == dtypes.fp8 else q,
            kv_buffer_fp8.view(num_page, page_size, nhead_kv, qk_head_dim),
            out_asm,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            max_seqlen_qo,
            page_size,
            nhead_kv,
            sm_scale,
            num_kv_splits=max_split_per_batch,
            q_scale=q_scale,
            kv_scale=kv_scale,
            work_meta_data=work_meta_data,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
            intra_batch_mode=non_persistent_mode,
        )

        # ===== DEBUG: dump attn_logits and attn_lse to .pt (binary, much faster) =====
        dump_dir = "mla_debug_dump"
        os.makedirs(dump_dir, exist_ok=True)
        if attn_logits is not None:
            torch.save(attn_logits.cpu(), os.path.join(dump_dir, "fp8_attn_logits.pt"))
            print(f"[debug] saved attn_logits {list(attn_logits.shape)} {attn_logits.dtype}")
        if attn_lse is not None:
            torch.save(attn_lse.cpu(), os.path.join(dump_dir, "fp8_attn_lse.pt"))
            print(f"[debug] saved attn_lse {list(attn_lse.shape)} {attn_lse.dtype}")

        # ===== DEBUG: comprehensive analysis for a target batch =====
        def debug_batch_analysis(target_batch=16):
            wis_cpu = work_info_set.cpu().to(torch.int32)
            wi_cpu = work_indptr.cpu().to(torch.int32)
            total_works = int(wi_cpu[-1].item())
            logits_cpu = attn_logits.float().cpu() if attn_logits is not None else None
            lse_cpu = attn_lse.float().cpu() if attn_lse is not None else None
            kv_seqlen_b = int(seq_lens_kv[target_batch].item())

            # --- Part 1: partial output & LSE statistics ---
            print(f"\n{'='*70}")
            print(f"[debug] Part 1: Partial output stats for batch {target_batch} "
                  f"(kv_seqlen={kv_seqlen_b})")
            print(f"{'='*70}")
            splits_info = []
            for w in range(total_works):
                row = wis_cpu[w]
                b = int(row[0].item())
                if b != target_batch:
                    continue
                info = {
                    "w": w, "partial_loc": int(row[1].item()),
                    "qo_s": int(row[2].item()), "qo_e": int(row[3].item()),
                    "kv_s": int(row[4].item()), "kv_e": int(row[5].item()),
                    "kv_off": int(row[6].item()),
                }
                splits_info.append(info)
                kv_len = info["kv_e"] - info["kv_s"]
                msg = (f"  work[{w}]: qo=[{info['qo_s']},{info['qo_e']})  "
                       f"kv=[{info['kv_s']},{info['kv_e']})  kv_len={kv_len}  "
                       f"kv_offset={info['kv_off']}  partial_loc={info['partial_loc']}")
                if logits_cpu is not None and info["partial_loc"] >= 0:
                    for qi in range(max_seqlen_qo):
                        idx = info["partial_loc"] + qi
                        if idx < logits_cpu.shape[0]:
                            chunk = logits_cpu[idx]
                            has_nan = torch.isnan(chunk).any().item()
                            has_inf = torch.isinf(chunk).any().item()
                            fin = chunk[torch.isfinite(chunk)]
                            amax = fin.abs().max().item() if fin.numel() > 0 else float("nan")
                            msg += f"\n    q{qi}: nan={has_nan} inf={has_inf} abs_max={amax:.6f}"
                        if lse_cpu is not None and idx < lse_cpu.shape[0]:
                            lc = lse_cpu[idx]
                            lv = lc[torch.isfinite(lc)]
                            lr = f"[{lv.min().item():.4f},{lv.max().item():.4f}]" if lv.numel() > 0 else "N/A"
                            msg += f"  lse_range={lr}"
                print(msg)

            # --- Part 2: manual reduce → compare with ASM output ---
            print(f"\n{'='*70}")
            print(f"[debug] Part 2: Manual reduce vs ASM final output for batch {target_batch}")
            print(f"{'='*70}")
            if logits_cpu is not None and lse_cpu is not None and splits_info:
                n_splits = len(splits_info)
                s_q = max_seqlen_qo
                manual_out = torch.zeros(s_q, nhead, v_head_dim)
                for qi in range(s_q):
                    all_lse = []
                    all_out = []
                    for si in splits_info:
                        pl = si["partial_loc"]
                        if pl < 0:
                            continue
                        idx = pl + qi
                        part_out = logits_cpu[idx, 0, :, :v_head_dim]  # (nhead, v_head_dim)
                        part_lse = lse_cpu[idx, 0, :, 0]               # (nhead,)
                        all_lse.append(part_lse)
                        all_out.append(part_out)
                    if not all_lse:
                        continue
                    lse_stack = torch.stack(all_lse, dim=0)    # (n_splits, nhead)
                    out_stack = torch.stack(all_out, dim=0)    # (n_splits, nhead, v_head_dim)
                    max_lse = lse_stack.max(dim=0).values      # (nhead,)
                    scales = torch.exp(lse_stack - max_lse.unsqueeze(0))  # (n_splits, nhead)
                    sum_scales = scales.sum(dim=0)             # (nhead,)
                    weighted = (out_stack * scales.unsqueeze(-1)).sum(dim=0)  # (nhead, v_head_dim)
                    manual_out[qi] = weighted / sum_scales.unsqueeze(-1)

                asm_out_b = out_asm[qo_indptr[target_batch]:qo_indptr[target_batch+1]].float().cpu()
                golden_out_b = out_ref_fp8[qo_indptr[target_batch]:qo_indptr[target_batch+1]].float().cpu()

                diff_manual_vs_asm = (manual_out - asm_out_b[:, :, :v_head_dim]).abs()
                diff_manual_vs_golden = (manual_out - golden_out_b[:, :, :v_head_dim]).abs()
                diff_asm_vs_golden = (asm_out_b[:, :, :v_head_dim] - golden_out_b[:, :, :v_head_dim]).abs()

                print(f"  manual_reduce vs ASM_output:   max_abs_delta={diff_manual_vs_asm.max().item():.6f}")
                print(f"  manual_reduce vs golden_fp8:   max_abs_delta={diff_manual_vs_golden.max().item():.6f}")
                print(f"  ASM_output    vs golden_fp8:   max_abs_delta={diff_asm_vs_golden.max().item():.6f}")
                if diff_manual_vs_asm.max().item() < 0.01:
                    print("  >>> Manual reduce matches ASM output → reduce kernel is OK")
                    if diff_manual_vs_golden.max().item() > 0.1:
                        print("  >>> But manual reduce differs from golden → stage1 kernel produces wrong partial outputs!")
                elif diff_manual_vs_golden.max().item() < 0.01:
                    print("  >>> Manual reduce matches golden → stage1 kernel is OK, reduce kernel has a bug!")
                else:
                    print("  >>> Both manual reduce and ASM output differ from golden → stage1 kernel has a bug")

            # --- Part 3: per-split golden comparison ---
            print(f"\n{'='*70}")
            print(f"[debug] Part 3: Per-split golden comparison for batch {target_batch}")
            print(f"{'='*70}")
            kv_buffer_fp8_local = kv_buffer.to(dtypes.fp8)
            kv_indices_cpu = kv_indices.cpu()
            q_batch = (q_fp8 if dtype == dtypes.fp8 else q)[splits_info[0]["qo_s"]:splits_info[0]["qo_e"]]
            q_f = q_batch.float()
            s_q = q_f.shape[0]

            for si in splits_info:
                pl = si["partial_loc"]
                kv_s, kv_e, kv_off = si["kv_s"], si["kv_e"], si["kv_off"]
                kv_len = kv_e - kv_s
                if pl < 0:
                    print(f"  work[{si['w']}] kv=[{kv_s},{kv_e}) (direct output, skip)")
                    continue

                kv_page_idx = kv_indices_cpu[kv_s:kv_e].to(q.device)
                kv_data = kv_buffer_fp8_local[kv_page_idx, 0, 0, :].float()
                eff_scale = sm_scale
                if q_scale is not None:
                    eff_scale *= q_scale.item()
                if kv_scale is not None:
                    eff_scale *= kv_scale.item()
                attn = torch.einsum("qhd,kd->hqk", q_f, kv_data) * eff_scale
                if kv_off < s_q:
                    for qi in range(s_q):
                        visible = kv_len - max(0, (s_q - 1 - qi) - kv_off)
                        if visible < kv_len:
                            attn[:, qi, visible:] = float("-inf")
                m = attn.max(-1).values
                exp_attn = torch.exp(attn - m.unsqueeze(-1))
                l_sum = exp_attn.sum(-1)
                ref_lse = m + torch.log(l_sum)
                ref_out = torch.einsum("hqk,kd->qhd", exp_attn, kv_data[:, :v_head_dim])
                ref_out = ref_out / l_sum.transpose(0, 1).unsqueeze(-1)

                asm_outs = []
                asm_lses = []
                for qi in range(s_q):
                    idx = pl + qi
                    asm_outs.append(logits_cpu[idx, 0, :, :v_head_dim])
                    if lse_cpu is not None:
                        asm_lses.append(lse_cpu[idx, 0, :, 0])
                asm_out_s = torch.stack(asm_outs, dim=0).float()
                diff_out = (ref_out.cpu() - asm_out_s).abs()
                max_d = diff_out.max().item()
                nan_cnt = torch.isnan(asm_out_s).sum().item()
                lse_msg = ""
                if asm_lses:
                    asm_lse_s = torch.stack(asm_lses, dim=0).float()
                    ref_lse_t = ref_lse.transpose(0, 1).cpu()
                    lse_diff = (ref_lse_t - asm_lse_s).abs()
                    lse_msg = f" lse_max_delta={lse_diff.max().item():.6f}"
                print(f"  work[{si['w']}] kv=[{kv_s},{kv_e}) kv_off={kv_off}: "
                      f"out_max_delta={max_d:.6f} nan_cnt={nan_cnt}{lse_msg}")

            out_asm_b = out_asm[qo_indptr[target_batch]:qo_indptr[target_batch+1]]
            print(f"\n  final out_asm batch {target_batch}: "
                  f"has_nan={torch.isnan(out_asm_b).any().item()} "
                  f"nan_count={torch.isnan(out_asm_b).sum().item()} / {out_asm_b.numel()}")

        debug_batch_analysis(16)
        debug_batch_analysis(0)

        # ===== DEBUG: per-batch diff analysis =====
        debug_per_batch_diff(
            out_ref, out_asm, qo_indptr, seq_lens_kv, batch_size,
            label="golden vs aiter_asm (fp8)"
        )
        debug_per_batch_diff(
            out_ref_fp8, out_asm, qo_indptr, seq_lens_kv, batch_size,
            label="golden_fp8 vs aiter_asm (fp8)"
        )

        err = checkAllclose(
            out_ref,
            out_asm,
            msg=f"mla_decode-absorb_fp8    [golden vs aiter_asm]: {us_asm_decode:>8.2f} us......",
        )
        checkAllclose(
            out_ref_fp8,
            out_asm,
            msg=f"mla_decode-absorb_fp8    [golden fp8 vs aiter_asm]: {us_asm_decode:>8.2f} us......",
        )

        cal_diff(out_ref, out_asm, "out", True)
        return err, us_asm_decode

    def test_absorb_decode_3buffer():

        out_asm = torch.empty((total_q, nhead, v_head_dim), dtype=out_dtype).fill_(-1)

        # convert to bytes
        nope_bytes = kv_nope_buffer_fp8.view(torch.uint8)
        scale_bytes = kv_nope_scale_factors_fp32.view(torch.uint8)
        rope_bytes = kv_rope_buffer_bf16.view(torch.uint8)
        kv_buffer_bytes = torch.cat(
            [nope_bytes.flatten(1), scale_bytes.flatten(1), rope_bytes.flatten(1)],
            dim=-1,
        )

        out_ref_fp8, lse_ref_fp8 = torch_mla_extend_3buffer(
            q,
            kv_buffer_bytes,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            page_size,
            nhead_kv,
            sm_scale,
            kv_lora_rank,
            qk_rope_head_dim,
            dtype=out_dtype,
            is_causal=True,
            scale_dim=scale_dim,
        )

        checkAllclose(
            out_ref,
            out_ref_fp8,
            msg="mla_decode-absorb_fp8    [golden fp8 vs golden]:......",
        )

        (attn_logits, attn_lse), us_asm_decode = run_perftest(
            aiter.mla.mla_decode_fwd,
            q,
            kv_buffer_bytes,
            out_asm,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            max_seqlen_qo,
            page_size,
            nhead_kv,
            sm_scale,
            num_kv_splits=max_split_per_batch,
            work_meta_data=work_meta_data,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
            intra_batch_mode=non_persistent_mode,
        )

        err = checkAllclose(
            out_ref,
            out_asm,
            msg=f"mla_decode-absorb_fp8    [golden vs aiter_asm]: {us_asm_decode:>8.2f} us......",
        )
        checkAllclose(
            out_ref_fp8,
            out_asm,
            msg=f"mla_decode-absorb_fp8    [golden fp8 vs aiter_asm]: {us_asm_decode:>8.2f} us......",
        )
        cal_diff(out_ref, out_asm, "out", True)
        return err, us_asm_decode

    err = None
    us_asm_decode = 1e12

    if paged_layout == "3BUFFER" and not non_persistent_mode:
        err, us_asm_decode = test_absorb_decode_3buffer()
    elif dtype == torch.bfloat16 and kvtype == dtypes.fp8:
        err, us_asm_decode = test_absorb_decode_bf16_fp8()
    elif dtype == torch.bfloat16:
        err, us_asm_decode = test_absorb_decode_bf16()
    elif kvtype == dtypes.fp8:
        err, us_asm_decode = test_absorb_decode_fp8()

    ret["decode:err"] = err
    ret["decode:asm_576"] = us_asm_decode

    flops = decode_qlen * total_kv * nhead * (qk_head_dim + v_head_dim) * 2
    bytes = (
        total_kv * nhead_kv * qk_head_dim * (torch.finfo(kvtype).bits // 8)
        + total_q * nhead * qk_head_dim * (torch.finfo(dtype).bits // 8)
        + total_q * nhead * v_head_dim * (torch.finfo(out_dtype).bits // 8)
    )

    ret["decode:flops"] = flops
    ret["decode:bytes"] = bytes
    ret["decode:TFLOPS"] = flops / us_asm_decode / 1e6
    ret["decode:TB/s"] = bytes / us_asm_decode / 1e6

    return ret


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-k",
    "--kv_lora_rank",
    type=int,
    default=512,
    help="""kv lora rank.
    e.g.: -k 512""",
)
parser.add_argument(
    "-qn",
    "--qk_nope_head_dim",
    type=int,
    default=512,
    help="""qk nope head dim.
    e.g.: -qn 512""",
)
parser.add_argument(
    "-qr",
    "--qk_rope_head_dim",
    type=int,
    default=64,
    help="""qk rope head dim.
    e.g.: -qr 64""",
)
parser.add_argument(
    "-vh",
    "--v_head_dim",
    type=int,
    default=512,
    help="""v head dim.
    e.g.: -vh 512""",
)
parser.add_argument(
    "-blk",
    "--block_size",
    type=int,
    default=1,
    help="""Block size.
    e.g.: -blk 1""",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    choices=[dtypes.d_dtypes["fp8"]],
    nargs="*",
    default="bf16,fp8",
    metavar="{bf16, fp8}",
    help="""Data type of Q.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-kvd",
    "--kv_dtype",
    type=dtypes.str2Dtype,
    choices=[dtypes.d_dtypes["fp8"]],
    nargs="*",
    metavar="{bf16, fp8}",
    default="bf16,fp8",
    help="""Data type of KV.
    e.g.: -kvd bf16""",
)
parser.add_argument(
    "-c",
    "--ctxLen",
    type=int,
    nargs="*",
    default=[742, 706, 715, 735, 788, 731, 721, 744, 780, 734, 741, 739, 738, 747, 728, 727, 769, 730, 708, 744, 739, 722, 734, 717, 716, 743, 733, 741, 744, 729, 715, 741],
    help="""Context length.
    e.g.: -c 21""",
)
parser.add_argument(
    "-b",
    "--batchSize",
    type=int,
    nargs="*",
    default=[32],
    help="""Batch size.
    e.g.: -b 16""",
)
parser.add_argument(
    "-n",
    "--nhead",
    type=dtypes.str2tuple,
    nargs="*",
    const=None,
    default=[(32, 4)],
    help="""Number of heads.
    e.g.: -n 16,1""",
)
parser.add_argument(
    "-ms",
    "--max_split_per_batch",
    type=int,
    nargs="*",
    default=[32],
    help="""kv seqlens max split num for per batch.
    e.g.: -ms 32""",
)
parser.add_argument(
    "--varlen",
    action="store_true",
    help="""variable kv seqlens per batch. Default: False.
    --varlen # True""",
)
parser.add_argument(
    "-nps",
    "--non_persistent_mode",
    action="store_true",
    help="""variable kv seqlens per batch. Default: False.
    --varlen # True""",
)
parser.add_argument(
    "-pl",
    "--paged_layout",
    type=str,
    choices=["LEGACY", "3BUFFER"],
    default="LEGACY",
    help="""kv paged layout for persistent mode.
        LEGACY: kv buffer is common buffer with nope and rope parts.
        3BUFFER: kv buffer is 3-buffer with nope, kv_scale and rope parts.
        e.g.: -pl 3BUFFER""",
)
parser.add_argument(
    "-sd",
    "--scale_dim",
    type=int,
    default=4,
    help="""scale dim.
    e.g.: -sd 4""",
)

args = parser.parse_args()
for nhead, decode_qlen in args.nhead:
    df = []
    for dtype, kvtype, ctx_len, batch_size, max_split_per_batch in itertools.product(
        args.dtype, args.kv_dtype, args.ctxLen, args.batchSize, args.max_split_per_batch
    ):
        if check_support(dtype, kvtype, nhead):
            ret = test_mla(
                ctx_len,
                batch_size,
                nhead,
                args.kv_lora_rank,
                args.qk_nope_head_dim,
                args.qk_rope_head_dim,
                args.v_head_dim,
                dtype,
                kvtype,
                args.block_size,
                varlen=args.varlen,
                decode_qlen=decode_qlen,
                max_split_per_batch=max_split_per_batch,
                non_persistent_mode=args.non_persistent_mode,
                paged_layout=args.paged_layout,
                scale_dim=args.scale_dim,
            )
            df.append(ret)
    df = pd.DataFrame(df)
    # df.to_csv(f"mla_nhead{nhead}decode_qlen{decode_qlen}.csv")
    df_md = df.to_markdown(index=False)
    aiter.logger.info("mla_persistent summary (markdown):\n%s", df_md)
