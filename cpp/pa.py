from jinja2 import Template
from utils import compile_template_op
import ctypes


MD_NAME = "pa_ragged"
with open("pa.cpp.jinja", "r") as f:
    src_template = Template(f.read())


def compile(num_kv_heads, num_seqs, num_heads, head_size, max_num_partitions, dtype, kv_dtype, fp8_kv_dtype, out_dtype, block_size, alibi_enabled):
    return compile_template_op(src_template, MD_NAME, ["utils.h", "pa.cuh", "../csrc/include"], [], num_kv_heads=num_kv_heads, num_seqs=num_seqs, num_heads=num_heads, head_size=head_size, max_num_partitions=max_num_partitions, dtype=dtype, kv_dtype=kv_dtype, fp8_kv_dtype=fp8_kv_dtype, out_dtype=out_dtype, block_size=block_size, alibi_enabled=alibi_enabled)


def paged_attention_ragged(out,         # [num_seqs, num_heads, head_size]
                           workspace_buffer,    # [num_seqs, num_heads, max_num_partitions]
                           query,  # [num_seqs, num_heads, head_size]
                           key_cache,  # [num_blocks, num_heads, head_size/x, block_size, x]
                           value_cache,  # [num_blocks, num_heads, head_size, block_size]
                           scale,
                           kv_indptr,
                           kv_page_indices,  # [num_seqs, max_num_blocks_per_seq]
                           kv_last_page_lens,  # [num_seqs]
                           block_size, 
                           max_num_partitions,
                           alibi_slopes,
                           kv_cache_dtype,
                           kv_cache_layout,
                           logits_soft_cap,
                           k_scale, 
                           v_scale,
                           fp8_out_scale):
    import torch
    if kv_cache_dtype == "auto":
        if query.dtype == torch.bfloat16:
            dtype = "__hip_bfloat16"
            kv_dtype = "__hip_bfloat16"
        elif query.dtype == torch.float16:
            dtype = "_Float16"
            kv_dtype = "_Float16"
        else:
            raise ValueError(f"Unsupported data type: {query.dtype}")
    elif kv_cache_dtype == "fp8" or kv_cache_dtype == "fp8_e4m3":
        if query.dtype == torch.bfloat16:
            dtype = "__hip_bfloat16"
            kv_dtype = "uint8"
        elif query.dtype == torch.float16:
            dtype = "_Float16"
            kv_dtype = "uint8"
        else:
            raise ValueError(f"Unsupported data type: {query.dtype}")
    else:
        raise ValueError(f"Unsupported kv_cache_dtype: {kv_cache_dtype}")
    
    if out.dtype == torch.bfloat16:
        out_dtype = "__hip_bfloat16"
    elif out.dtype == torch.float16:
        out_dtype = "_Float16"
    else:
        raise ValueError(f"Unsupported data type: {out.dtype}")
    
    num_kv_heads = key_cache.size(1) if kv_cache_layout=="HND" else key_cache.size(2)
    num_seqs = query.size(0)
    num_heads = query.size(1)
    head_size = query.size(2)
    q_stride = query.stride(0)
    kv_block_stride = key_cache.stride(0)
    kv_head_stride  = key_cache.stride(1) if kv_cache_layout == "HND" else key_cache.stride(2)
    kv_seq_stride   = key_cache.stride(2) if kv_cache_layout == "HND" else key_cache.stride(1)

    func = compile(num_kv_heads, num_seqs, num_heads, head_size, max_num_partitions, dtype, kv_dtype, kv_cache_dtype, out_dtype, block_size, "true" if alibi_slopes else "false")

    out_ptr = ctypes.cast(out.data_ptr(), ctypes.c_void_p)
    alibi_slopes_ptr = ctypes.cast(alibi_slopes.data_ptr(), ctypes.POINTER(ctypes.c_float)) if alibi_slopes else ctypes.POINTER(ctypes.c_int)()

    query_ptr = ctypes.cast(query.data_ptr(), ctypes.c_void_p)
    key_cache_ptr = ctypes.cast(key_cache.data_ptr(), ctypes.c_void_p)
    value_cache_ptr = ctypes.cast(value_cache.data_ptr(), ctypes.c_void_p)
    kv_indptr_ptr = ctypes.cast(kv_indptr.data_ptr(), ctypes.POINTER(ctypes.c_int))
    kv_page_indices_ptr = ctypes.cast(kv_page_indices.data_ptr(), ctypes.POINTER(ctypes.c_int))
    kv_last_page_lens_ptr = ctypes.cast(kv_last_page_lens.data_ptr(), ctypes.POINTER(ctypes.c_int)) if block_size > 1 else ctypes.POINTER(ctypes.c_int)()

    k_scale_ptr = ctypes.cast(k_scale.data_ptr(), ctypes.POINTER(ctypes.c_float))
    v_scale_ptr = ctypes.cast(v_scale.data_ptr(), ctypes.POINTER(ctypes.c_float))
    fp8_out_scale_ptr = ctypes.cast(fp8_out_scale.data_ptr(), ctypes.POINTER(ctypes.c_float)) if fp8_out_scale else ctypes.POINTER(ctypes.c_int)()

    stream = ctypes.cast(torch.cuda.current_stream().cuda_stream, ctypes.c_void_p)
    workspace_buffer_ptr = ctypes.cast(workspace_buffer.data_ptr(), ctypes.c_void_p)
    scale = ctypes.c_float(scale)
    logits_soft_cap = ctypes.c_float(logits_soft_cap)
    block_size = ctypes.c_int(block_size)
    max_num_partitions = ctypes.c_int(max_num_partitions)

    func(out_ptr, workspace_buffer_ptr, query_ptr, key_cache_ptr, value_cache_ptr, scale, num_seqs, q_stride, kv_block_stride, kv_head_stride, kv_seq_stride, kv_indptr_ptr, kv_page_indices_ptr, kv_last_page_lens_ptr, alibi_slopes_ptr, logits_soft_cap, k_scale_ptr, v_scale_ptr, fp8_out_scale_ptr, stream)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_kv_heads", type=int, required=True)
    parser.add_argument("--num_seqs", type=int, required=True)
    parser.add_argument("--num_heads", type=int, required=True)
    parser.add_argument("--head_size", type=int, required=True)
    parser.add_argument("--max_num_partitions", type=int, required=True)
    parser.add_argument("--dtype", type=str, required=True)
    parser.add_argument("--kv_dtype", type=str, required=True)
    parser.add_argument("--fp8_kv_dtype", type=str, required=True)
    parser.add_argument("--out_dtype", type=str, required=True)
    parser.add_argument("--block_size", type=int, required=True)
    parser.add_argument("--alibi_enabled", type=str, required=True)
    args = parser.parse_args()
    compile(**vars(args))