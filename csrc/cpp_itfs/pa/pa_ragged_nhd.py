from jinja2 import Template
from csrc.cpp_itfs.utils import compile_template_op, AITER_CORE_DIR, str_to_bool
import ctypes
import math

MD_NAME = "pa_ragged_nhd"

with open(f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa/pa_ragged_nhd.cpp.jinja", "r") as f:
    src_template = Template(f.read())


def compile(
    gqa_ratio: int,
    num_kv_heads: int,
    head_size: int,
    npar_loops: int,
    dtype: str,
    kv_dtype: str,
    fp8_kv_dtype: str,
    out_dtype: str,
    block_size: int,
    alibi_enabled: bool = False,
    partition_size: int = 256,
    mtp: int = 1,
    logits_soft_cap_enabled: bool = False,
    func_name: str = None,
):
    import os

    version = os.getenv("QKV_VERSION", "GOLDEN")

    if head_size != 256 or kv_dtype != "__hip_bfloat16" or gqa_ratio > 16:
        raise ValueError(
            "pa_ragged_nhd requires head_size=256, kv_dtype=__hip_bfloat16, and gqa_ratio <= 16"
        )
    return compile_template_op(
        src_template,
        MD_NAME,
        [
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa/pa_ragged_nhd_impl.cpp",
        ],
        gqa_ratio=gqa_ratio,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        npar_loops=npar_loops,
        dtype=dtype,
        kv_dtype=kv_dtype,
        fp8_kv_dtype=fp8_kv_dtype,
        out_dtype=out_dtype,
        block_size=block_size,
        partition_size=partition_size,
        mtp=mtp,
        alibi_enabled=alibi_enabled,
        logits_soft_cap_enabled=logits_soft_cap_enabled,
        func_name=func_name,
        # choice: [GOLDEN, EXPERIMENTAL]. Classify original kernel and experimental kernel
        version=version,
    )


def paged_attention_ragged_nhd(
    out,  # [num_seqs, num_heads, head_size]
    exp_sums,  # float32, numel >= num_seqs*num_heads*max_num_partitions (layout: pool max_bs in SGLang)
    max_logits,  # float32, same numel as exp_sums
    tmp_out,  # bfloat16, numel >= num_seqs*num_heads*max_num_partitions*head_size
    query,  # [num_seqs, num_heads, head_size]
    key_cache,  # [num_blocks, num_heads, head_size/x, block_size, x]
    value_cache,  # [num_blocks, num_heads, head_size, block_size]
    scale,
    kv_indptr,
    kv_page_indices,  # 1D ragged: [kv_indptr[-1]) valid
    kv_last_page_lens,  # [num_seqs]
    block_size,
    max_num_partitions,
    alibi_slopes,
    kv_cache_dtype,
    kv_cache_layout,
    logits_soft_cap,
    k_scale,
    v_scale,
    fp8_out_scale=None,
    partition_size=256,
    mtp=1,
    q_scale=None,
):
    import os
    import torch
    from csrc.cpp_itfs.torch_utils import torch_to_c_types

    warpSize = torch.cuda.get_device_properties(out.device).warp_size
    if query.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise ValueError("paged_attention_ragged_nhd(pa-256) only supports bf16")
    if kv_cache_dtype != "auto":
        raise ValueError(
            "paged_attention_ragged_nhd(pa-256) only supports kv_cache_dtype='auto'"
        )
    if kv_cache_layout != "NHD":
        raise ValueError(
            "paged_attention_ragged_nhd(pa-256) only supports kv_cache_layout='NHD'"
        )
    if block_size != 1:
        raise ValueError("paged_attention_ragged_nhd(pa-256) only supports block_size=1")
    if query.size(2) != 256:
        raise ValueError("paged_attention_ragged_nhd(pa-256) only supports head_size=256")
    dtype = "__hip_bfloat16"
    kv_dtype = "__hip_bfloat16"
    out_dtype = "__hip_bfloat16"

    num_kv_heads = key_cache.size(1) if kv_cache_layout == "HND" else key_cache.size(2)
    num_seqs = query.size(0)
    num_heads = query.size(1)
    head_size = query.size(2)
    q_stride = query.stride(0)
    kv_block_stride = key_cache.stride(0)
    kv_head_stride = (
        key_cache.stride(1) if kv_cache_layout == "HND" else key_cache.stride(2)
    )
    kv_seq_stride = (
        key_cache.stride(2) if kv_cache_layout == "HND" else key_cache.stride(1)
    )
    if num_kv_heads < 1 or num_heads < 1 or num_seqs < 0:
        raise ValueError("invalid num_seqs/num_heads/num_kv_heads")
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads (GQA)")
    gqa_ratio = num_heads // num_kv_heads
    npar_loops = int(math.ceil(max_num_partitions / warpSize))
    n_blocks = int(key_cache.size(0))
    for t in (out, exp_sums, max_logits, tmp_out, query, key_cache, value_cache):
        assert t.is_contiguous(), (
            "paged_attention_ragged_nhd(pa-256) expects contiguous out/exp_sums/max_logits/tmp_out/query/key_cache/value_cache"
        )
    if exp_sums.dtype != torch.float32 or max_logits.dtype != torch.float32:
        raise ValueError("paged_attention_ragged_nhd: exp_sums and max_logits must be float32")
    if tmp_out.dtype != torch.bfloat16:
        raise ValueError("paged_attention_ragged_nhd: tmp_out must be bfloat16")
    _nhp_run = int(num_seqs) * int(num_heads) * int(max_num_partitions)
    _nhp_el = int(exp_sums.numel())
    if _nhp_el < _nhp_run or int(max_logits.numel()) < _nhp_run:
        raise ValueError(
            "paged_attention_ragged_nhd: exp_sums/max_logits numel must be >= "
            f"num_seqs*num_heads*max_num_partitions ({_nhp_run}), got exp_sums={_nhp_el}"
        )
    if int(tmp_out.numel()) < _nhp_run * int(head_size):
        raise ValueError(
            "paged_attention_ragged_nhd: tmp_out numel must be >= "
            f"num_seqs*num_heads*max_num_partitions*head_size"
        )
    if kv_cache_layout == "NHD":
        assert key_cache.size(1) == block_size and key_cache.size(2) == num_kv_heads, (
            "key_cache NHD shape [num_blocks, block_size, num_kv_heads, head_size]"
        )
        assert list(value_cache.shape[:3]) == [n_blocks, block_size, num_kv_heads], (
            "value_cache must match key_cache for [num_blocks, block_size, num_kv_heads, ...]"
        )
        assert key_cache.size(3) == head_size, "key_cache last dim must equal head_size"
        assert int(value_cache.size(3)) == int(head_size), "pa-256: value last dim must equal head_size (256)"
    #if num_seqs > 0:
        #_kv0 = int(kv_indptr[0].item())
        # if _kv0 != 0:
        #     print(
        #         "[paged_attention_ragged_nhd] warning: expected kv_indptr[0]==0 (1D ragged base), "
        #         f"got kv_indptr[0]={_kv0}; kernel uses ptr+kv_indptr[b] on kv_page_indices"
        #     )
    #     assert kv_indptr[0] == 0, "kv_indptr[0] must be 0" # 保留这个 先跳过 看看后面有没有别的雷
    #     assert int(kv_indptr.size(0)) == num_seqs + 1, "kv_indptr must be [num_seqs+1] (CSRS)"
    #     last_idx = int(kv_indptr[-1].item())
    #     assert last_idx >= 0
    #     if last_idx > 0:
    #         assert kv_page_indices.is_contiguous()
    #         assert int(kv_page_indices.numel()) >= last_idx, (
    #             "kv_page_indices is too small for kv_indptr[-1] (1D ragged buffer)"
    #         )
    #     lens = kv_indptr[1:] - kv_indptr[:-1]
    #     max_kv_len = int(lens.max().item()) if lens.numel() else 0
    #     if max_kv_len > 0:
    #         need_parts = (max_kv_len + int(partition_size) - 1) // int(partition_size)
    #         assert need_parts <= int(
    #             max_num_partitions
    #         ), (
    #             f"max_num_partitions too small: need ceil(max_kv_len/partition)={need_parts} "
    #             f"<= max_num_partitions, max_kv_len={max_kv_len}, partition_size={partition_size}"
    #         )
    #     for b in range(num_seqs):
    #         lo = int(kv_indptr[b].item())
    #         hi = int(kv_indptr[b + 1].item())
    #         assert 0 <= lo <= hi, "kv_indptr must be non-decreasing"
    #         if hi > lo:
    #             seg = kv_page_indices[lo:hi]
    #             assert bool((seg >= 0).all().item()) and bool((seg < n_blocks).all().item()), (
    #                 f"kv_page_ids must be in [0, num_blocks), num_blocks={n_blocks}, b={b}"
    #             )
    # _nhp = num_seqs * num_heads * int(max_num_partitions)
    # _meta = 2 * _nhp * 4
    # _tmp_base = (_meta + 1023) // 1024 * 1024
    # _need = _tmp_base + _nhp * int(head_size) * 2
    # assert int(workspace_buffer.numel()) >= int(_need), (
    #     f"workspace_buffer too small: need at least {_need} uint8, got {int(workspace_buffer.numel())} "
    #     "(see pa_ragged_nhd.cpp.jinja: exp_sums|max_logits, 1024-align tmp_out)"
    # )
    func = compile(
        gqa_ratio,
        num_kv_heads,
        head_size,
        npar_loops,
        dtype,
        kv_dtype,
        kv_cache_dtype,
        out_dtype,
        block_size,
        alibi_slopes is not None,
        partition_size,
        mtp,
        bool(logits_soft_cap),
    )

    alibi_slopes_ptr = (
        ctypes.cast(alibi_slopes.data_ptr(), ctypes.POINTER(ctypes.c_float))
        if alibi_slopes is not None
        else ctypes.POINTER(ctypes.c_int)()
    )
    kv_indptr_ptr = ctypes.cast(kv_indptr.data_ptr(), ctypes.POINTER(ctypes.c_int))
    kv_page_indices_ptr = ctypes.cast(
        kv_page_indices.data_ptr(), ctypes.POINTER(ctypes.c_int)
    )
    kv_last_page_lens_ptr = (
        ctypes.cast(kv_last_page_lens.data_ptr(), ctypes.POINTER(ctypes.c_int))
        if block_size > 1
        else ctypes.POINTER(ctypes.c_int)()
    )

    k_scale_ptr = ctypes.cast(k_scale.data_ptr(), ctypes.POINTER(ctypes.c_float))
    v_scale_ptr = ctypes.cast(v_scale.data_ptr(), ctypes.POINTER(ctypes.c_float))
    fp8_out_scale_ptr = (
        ctypes.cast(fp8_out_scale.data_ptr(), ctypes.POINTER(ctypes.c_float))
        if fp8_out_scale
        else ctypes.POINTER(ctypes.c_int)()
    )

    (
        out_ptr,
        exp_sums_ptr,
        max_logits_ptr,
        tmp_out_ptr,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        scale,
        logits_soft_cap,
        num_seqs,
        num_kv_heads,
        num_heads,
        max_num_partitions,
        q_stride,
        kv_block_stride,
        kv_head_stride,
        kv_seq_stride,
        stream,
    ) = torch_to_c_types(
        out,
        exp_sums,
        max_logits,
        tmp_out,
        query,
        key_cache,
        value_cache,
        scale,
        logits_soft_cap,
        num_seqs,
        num_kv_heads,
        num_heads,
        max_num_partitions,
        q_stride,
        kv_block_stride,
        kv_head_stride,
        kv_seq_stride,
        torch.cuda.current_stream(),
    )
    q_scale_ptr = (
        ctypes.cast(q_scale.data_ptr(), ctypes.POINTER(ctypes.c_float))
        if q_scale is not None
        else ctypes.POINTER(ctypes.c_float)()
    )
    func(
        out_ptr,
        exp_sums_ptr,
        max_logits_ptr,
        tmp_out_ptr,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        kv_indptr_ptr,
        kv_page_indices_ptr,
        kv_last_page_lens_ptr,
        alibi_slopes_ptr,
        q_scale_ptr,
        k_scale_ptr,
        v_scale_ptr,
        fp8_out_scale_ptr,
        scale,
        logits_soft_cap,
        num_seqs,
        num_kv_heads,
        num_heads,
        max_num_partitions,
        q_stride,
        kv_block_stride,
        kv_head_stride,
        kv_seq_stride,
        stream,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gqa_ratio", type=int, required=True)
    parser.add_argument("--num_kv_heads", type=int, required=True)
    parser.add_argument("--head_size", type=int, required=True)
    parser.add_argument("--npar_loops", type=int, required=True)
    parser.add_argument("--dtype", type=str, required=True)
    parser.add_argument("--kv_dtype", type=str, required=True)
    parser.add_argument("--fp8_kv_dtype", type=str, required=True)
    parser.add_argument("--out_dtype", type=str, required=True)
    parser.add_argument("--block_size", type=int, required=True)
    parser.add_argument("--alibi_enabled", type=str_to_bool, required=True)
    parser.add_argument("--func_name", type=str, default=None)
    args = parser.parse_args()
    compile(**vars(args))
