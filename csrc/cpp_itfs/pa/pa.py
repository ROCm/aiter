from jinja2 import Template
from cpp.utils import compile_template_op
from cpp.torch_utils import torch_to_c_types
import ctypes
import math


MD_NAME = "pa"
warpSize = 64
with open("pa.cpp.jinja", "r") as f:
    src_template = Template(f.read())


def compile(
    gqa_ratio: int,
    head_size: int,
    npar_loops: int,
    dtype: str,
    kv_dtype: str,
    fp8_kv_dtype: str,
    out_dtype: str,
    block_size: int,
    alibi_enabled: str,
    folder: str = None,
):
    return compile_template_op(
        src_template,
        MD_NAME,
        ["../utils.h", "pa.cuh", "../../csrc/include"],
        [],
        gqa_ratio=gqa_ratio,
        head_size=head_size,
        npar_loops=npar_loops,
        dtype=dtype,
        kv_dtype=kv_dtype,
        fp8_kv_dtype=fp8_kv_dtype,
        out_dtype=out_dtype,
        block_size=block_size,
        alibi_enabled=alibi_enabled,
        folder=folder,
    )


def paged_attention_rocm(
    out,
    exp_sums,
    max_logits,
    tmp_out,
    query,
    key_cache,
    value_cache,
    num_kv_heads,
    scale,
    block_tables,
    context_lens,
    block_size,
    max_context_len,
    alibi_slopes,
    kv_cache_dtype,
    k_scale,
    v_scale,
    fp8_out_scale,
):
    import torch

    if kv_cache_dtype == "auto":
        if query.dtype == torch.bfloat16:
            dtype = "__hip_bfloat16"
            kv_dtype = "__hip_bfloat16"
        elif query.dtype == torch.float16:
            dtype = "__half"
            kv_dtype = "__half"
        else:
            raise ValueError(f"Unsupported data type: {query.dtype}")
    elif kv_cache_dtype == "fp8" or kv_cache_dtype == "fp8_e4m3":
        if query.dtype == torch.bfloat16:
            dtype = "__hip_bfloat16"
            kv_dtype = "uint8_t"
        elif query.dtype == torch.float16:
            dtype = "__half"
            kv_dtype = "uint8_t"
        else:
            raise ValueError(f"Unsupported data type: {query.dtype}")
    else:
        raise ValueError(f"Unsupported kv_cache_dtype: {kv_cache_dtype}")

    if out.dtype == torch.bfloat16:
        out_dtype = "__hip_bfloat16"
    elif out.dtype == torch.float16:
        out_dtype = "__half"
    else:
        raise ValueError(f"Unsupported data type: {out.dtype}")

    num_seqs = query.size(0)
    num_heads = query.size(1)
    head_size = query.size(2)
    q_stride = query.stride(0)
    kv_block_stride = key_cache.stride(0)
    kv_head_stride = key_cache.stride(1)
    gqa_ratio = int(num_heads / num_kv_heads)
    npar_loops = int(math.ceil(max_num_partitions / warpSize))
    func = compile(
        gqa_ratio,
        head_size,
        npar_loops,
        dtype,
        kv_dtype,
        kv_cache_dtype,
        out_dtype,
        block_size,
        "true" if alibi_slopes else "false",
    )

    alibi_slopes_ptr = (
        ctypes.cast(alibi_slopes.data_ptr(), ctypes.POINTER(ctypes.c_float))
        if alibi_slopes
        else ctypes.POINTER(ctypes.c_int)()
    )

    context_lens_ptr = ctypes.cast(
        context_lens.data_ptr(), ctypes.POINTER(ctypes.c_int)
    )
    block_tables_ptr = ctypes.cast(
        block_tables.data_ptr(), ctypes.POINTER(ctypes.c_int)
    )

    fp8_out_scale_ptr = (
        ctypes.cast(fp8_out_scale.data_ptr(), ctypes.POINTER(ctypes.c_float))
        if fp8_out_scale
        else ctypes.POINTER(ctypes.c_int)()
    )

    (
        out_ptr,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        exp_sums_ptr,
        max_logits_ptr,
        tmp_out_ptr,
        scale,
        num_seqs,
        num_kv_heads,
        num_heads,
        max_num_partitions,
        max_context_len,
        q_stride,
        kv_block_stride,
        kv_head_stride,
        stream,
    ) = torch_to_c_types(
        out,
        query,
        key_cache,
        value_cache,
        exp_sums,
        max_logits,
        tmp_out,
        scale,
        num_seqs,
        num_kv_heads,
        num_heads,
        max_num_partitions,
        max_context_len,
        q_stride,
        kv_block_stride,
        kv_head_stride,
        torch.cuda.current_stream(),
    )

    func(
        out_ptr,
        exp_sums_ptr,
        max_logits_ptr,
        tmp_out_ptr,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        scale,
        block_tables_ptr,
        context_lens_ptr,
        max_context_len,
        num_seqs,
        num_kv_heads,
        num_heads,
        max_num_partitions,
        q_stride,
        kv_block_stride,
        kv_head_stride,
        alibi_slopes_ptr,
        k_scale,
        v_scale,
        fp8_out_scale_ptr,
        stream,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gqa_ratio", type=int, required=True)
    parser.add_argument("--head_size", type=int, required=True)
    parser.add_argument("--npar_loops", type=int, required=True)
    parser.add_argument("--dtype", type=str, required=True)
    parser.add_argument("--kv_dtype", type=str, required=True)
    parser.add_argument("--fp8_kv_dtype", type=str, required=True)
    parser.add_argument("--out_dtype", type=str, required=True)
    parser.add_argument("--block_size", type=int, required=True)
    parser.add_argument("--alibi_enabled", type=str, required=True)
    parser.add_argument("--folder", type=str, default=None)
    args = parser.parse_args()
    compile(**vars(args))
