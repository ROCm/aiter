from jinja2 import Template
from csrc.cpp_itfs.utils import compile_template_op, transfer_hsaco, AITER_CORE_DIR, get_default_func_name, not_built, run_lib
from aiter.aot.triton.compile import compile_kernel
import triton


MD_NAME = "asm_mla_decode_fwd"
warpSize = 64
with open(f"{AITER_CORE_DIR}/csrc/cpp_itfs/mla/asm_mla_decode_fwd.cpp.jinja", "r") as f:
    src_template = Template(f.read())


def compile(hsaco_path: str, page_size: int, q_dtype: str, kv_dtype: str, num_kv_splits:int, v_head_dim:int, func_name: str = None):
    if func_name is None:
        func_name = get_default_func_name(MD_NAME, (page_size, q_dtype, kv_dtype, num_kv_splits, v_head_dim))
        
    if not_built(func_name):
        bin_size, bin_data = transfer_hsaco(hsaco_path)
        triton_kernel, triton_header, triton_source = compile_kernel(f"{AITER_CORE_DIR}/aiter/mla.py", "_fwd_kernel_stage2_asm", f"*fp32:16,*fp32:16,*bf16:16,*i32:16,i32,i32,i32,i32,i32,i32,i32,{num_kv_splits},{triton.next_power_of_2(v_head_dim)},{v_head_dim},64", "bs,nheads,1", 4, 2, "decode_mla_stage2_asm", waves_per_eu=4, kpack=2, matrix_instr_nonkdim=16)

        return compile_template_op(src_template, MD_NAME, ["../utils.h", "../../include", triton_header], [triton_source], bin_size=bin_size, bin_data=bin_data, page_size=page_size, q_dtype=q_dtype, kv_dtype=kv_dtype, triton_header=triton_header, kernel_name="mla_stage1_a16w16_bf16", triton_kernel=triton_kernel, num_kv_splits=num_kv_splits, func_name=func_name)
    else:
        return run_lib(func_name)


def asm_mla_decode_fwd(q,         # [num_seqs, num_heads, head_size]
                       kv_buffer,    # [num_seqs, num_heads, max_num_partitions]
                       output,
                       kv_indptr, # [num_seqs, num_heads, head_size]
                       kv_page_indices,  # [num_blocks, num_heads, head_size/x, block_size, x]
                       kv_last_page_lens,  # [num_blocks, num_heads, head_size, block_size]
                       softmax_scale=None,
                       logit_cap=0.0,
                       num_kv_splits=None,  # for experts only!!!
                       logits=None,
                       attn_lse=None,
):
    import torch
    from csrc.cpp_itfs.torch_utils import torch_to_c_types
    
    if q.dtype != torch.bfloat16:
        raise ValueError(f"{asm_mla_decode_fwd.__name__}: only support dtype == torch.bfloat16 for now")
    
    num_kv_heads = kv_buffer.size(2)
    num_seqs = q.size(0)
    num_heads = q.size(1)
    head_size = q.size(2)
    page_size = kv_buffer.size(1)
    v_head_dim = output.size(2)

    if num_kv_heads != 1:
        raise ValueError(f"{asm_mla_decode_fwd.__name__}: only support num_kv_heads==1 for now")
    
    if head_size != kv_buffer.size(3):
        raise ValueError(f"{asm_mla_decode_fwd.__name__}: only support head_size == KV.size(3) for now")

    if logit_cap > 0:
        raise ValueError(f"{asm_mla_decode_fwd.__name__}: only support logit_cap==0 for now")
    
    if num_heads / num_kv_heads > 16:
        raise ValueError(f"{asm_mla_decode_fwd.__name__}: only support num_heads / num_kv_heads <= 16 for now")
    
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_size**0.5)

    if num_kv_splits is None:
        device_props = torch.cuda.get_device_properties(q.device)
        cu_num = device_props.multi_processor_count
        num_kv_splits = min(16, max(1, cu_num // num_seqs))

    if logits is None:
        logits = torch.empty(
            (num_seqs, num_kv_splits, num_heads, v_head_dim), dtype=torch.float, device=q.device)
    if attn_lse is None:
        attn_lse = torch.empty(
            (num_seqs, num_kv_splits, num_heads,  1), dtype=torch.float, device=q.device)
        
    if num_kv_splits != logits.size(1):
        raise ValueError(f"{asm_mla_decode_fwd.__name__}: num_kv_splits != logits.size(1)")

    func = compile(f"{AITER_CORE_DIR}/hsa/mla/mla_stage1_a16w16_bf16.co", page_size, "__hip_bfloat16", "__hip_bfloat16", num_kv_splits, v_head_dim)

    func(*torch_to_c_types(q, kv_buffer, kv_indptr, kv_page_indices, kv_last_page_lens, softmax_scale, logits, attn_lse, output, num_seqs, num_heads, num_kv_heads, q.stride(0), kv_buffer.stride(0), attn_lse.stride(0), attn_lse.stride(1), attn_lse.stride(2), output.stride(0), output.stride(1), torch.cuda.current_stream()))
    return logits, attn_lse 


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hsaco_path", type=str, required=True)
    parser.add_argument("--page_size", type=int, required=True)
    parser.add_argument("--q_dtype", type=str, required=True)
    parser.add_argument("--kv_dtype", type=str, required=True)
    parser.add_argument("--num_kv_splits", type=int, required=True)
    parser.add_argument("--v_head_dim", type=int, required=True)
    parser.add_argument("--func_name", type=str, default=None)
    args = parser.parse_args()
    compile(**vars(args))