from jinja2 import Template
from csrc.cpp_itfs.utils import (
    compile_template_op,
    transfer_hsaco,
    AITER_CORE_DIR,
    GPU_ARCH,
    get_default_func_name,
    not_built,
    run_lib,
)
from triton.tools.compile import compile_kernel, CompileArgs
import triton
import functools

MD_NAME = "asm_mla_decode_fwd"
warpSize = 64
with open(f"{AITER_CORE_DIR}/csrc/cpp_itfs/mla/asm_mla_decode_fwd.cpp.jinja", "r") as f:
    src_template = Template(f.read())

mgcs = {16: 64, 128: 16}


@functools.lru_cache()
def get_meta_param(num_kv_splits, device, bs):
    import torch

    if num_kv_splits is None:
        device_props = torch.cuda.get_device_properties(device)
        cu_num = device_props.multi_processor_count
        num_kv_splits = min(16, max(1, cu_num // bs))

    return num_kv_splits


def compile(
    gqa_ratio: int,
    page_size: int,
    q_dtype: str,
    kv_dtype: str,
    num_kv_splits: int,
    v_head_dim: int,
    func_name: str = None,
):
    if func_name is None:
        func_name = get_default_func_name(
            MD_NAME,
            (gqa_ratio, page_size, q_dtype, kv_dtype, num_kv_splits, v_head_dim),
        )

    if not_built(func_name):
        if gqa_ratio == 128:
            hsaco_path = f"{AITER_CORE_DIR}/hsa/{GPU_ARCH}/mla/mla_dec_stage1_bf16_a16w16_subQ128_mqa128.co"
            kernel_name = "_ZN5aiter41mla_dec_stage1_bf16_a16w16_subQ128_mqa128E"
        else:
            hsaco_path = f"{AITER_CORE_DIR}/hsa/{GPU_ARCH}/mla/mla_dec_stage1_bf16_a16w16_subQ16_mqa16.co"
            kernel_name = "_ZN5aiter39mla_dec_stage1_bf16_a16w16_subQ16_mqa16E"

        bin_size, bin_data = transfer_hsaco(hsaco_path)
        compile_args = CompileArgs(
            path=f"{AITER_CORE_DIR}/aiter/mla.py",
            kernel_name="_fwd_kernel_stage2_asm",
            signature=f"*fp32:16,*fp32:16,*bf16:16,*i32:16,*i32:16,i32,i32,i32,i32,i32,i32,i32,i32,{num_kv_splits},{triton.next_power_of_2(v_head_dim)},{v_head_dim},{mgcs[gqa_ratio]}",
            grid="bs,nheads,max_seqlen_q",
            num_warps=4,
            num_stages=2,
            out_name="decode_mla_stage2_asm",
        )
        triton_kernel, output_files = compile_kernel(compile_args)
        triton_header = None
        triton_source = None
        for output_file in output_files:
            if output_file.suffix == ".h":
                triton_header = output_file
            elif output_file.suffix == ".cpp":
                triton_source = output_file

        return compile_template_op(
            src_template,
            MD_NAME,
            ["../utils.h", "../../include", triton_header],
            [triton_source],
            bin_size=bin_size,
            bin_data=bin_data,
            page_size=page_size,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            triton_header=triton_header,
            kernel_name=kernel_name,
            triton_kernel=triton_kernel,
            num_kv_splits=num_kv_splits,
            func_name=func_name,
        )
    else:
        return run_lib(func_name)


def asm_mla_decode_fwd(
    q,  # [total_query_len, num_heads, head_size]
    kv_buffer,  # [num_page, page_size, num_kv_heads, kv_lora_rank + qk_rope_head_dim]
    output,  # [total_query_len, num_heads, head_size]
    qo_indptr,  # [num_seqs + 1]
    kv_indptr,  # [num_seqs + 1]
    kv_page_indices,  # [num_page_used]
    kv_last_page_lens,  # [batch_size]
    max_seqlen_q,
    softmax_scale=None,
    logit_cap=0.0,
    num_kv_splits=None,  # for experts only!!!
    logits=None,
    attn_lse=None,
):
    import torch
    from csrc.cpp_itfs.torch_utils import torch_to_c_types

    if q.dtype != torch.bfloat16:
        raise ValueError(
            f"{asm_mla_decode_fwd.__name__}: only support dtype == torch.bfloat16 for now"
        )

    num_kv_heads = kv_buffer.size(2)
    total_query_len = output.size(0)
    num_heads = output.size(1)
    head_size = q.size(2)
    page_size = kv_buffer.size(1)
    v_head_dim = output.size(2)
    num_seqs = qo_indptr.size(0) - 1

    if num_kv_heads != 1:
        raise ValueError(
            f"{asm_mla_decode_fwd.__name__}: only support num_kv_heads==1 for now"
        )

    if head_size != kv_buffer.size(3):
        raise ValueError(
            f"{asm_mla_decode_fwd.__name__}: only support head_size == KV.size(3) for now"
        )

    if logit_cap > 0:
        raise ValueError(
            f"{asm_mla_decode_fwd.__name__}: only support logit_cap==0 for now"
        )

    gqa_ratio = num_heads // num_kv_heads

    if softmax_scale is None:
        softmax_scale = 1.0 / (head_size**0.5)

    if num_kv_splits is None:
        num_kv_splits = get_meta_param(num_kv_splits, q.device, num_seqs)

    if logits is None:
        if num_heads == 16:
            logits = torch.empty(
                (total_query_len, num_kv_splits, num_heads, v_head_dim),
                dtype=torch.float,
                device=q.device,
            )
            if max_seqlen_q != 1:
                raise ValueError(
                    f"{asm_mla_decode_fwd.__name__}: only support max_seqlen_q==1 when n_head==16, but got {max_seqlen_q}"
                )
        elif num_heads == 128:
            logits = (
                output.view((total_query_len, num_kv_splits, num_heads, v_head_dim))
                if num_kv_splits == 1
                else torch.empty(
                    (total_query_len, num_kv_splits, num_heads, v_head_dim),
                    dtype=torch.float,
                    device=q.device,
                )
            )
        else:
            raise ValueError(
                f"{asm_mla_decode_fwd.__name__}: only support n_head==16 or n_head==128 for now"
            )
    if attn_lse is None:
        attn_lse = torch.empty(
            (total_query_len, num_kv_splits, num_heads, 1),
            dtype=torch.float,
            device=q.device,
        )

    if num_kv_splits != logits.size(1):
        raise ValueError(
            f"{asm_mla_decode_fwd.__name__}: num_kv_splits != logits.size(1)"
        )

    if gqa_ratio == 16 and max_seqlen_q != 1:
        raise ValueError(
            f"{asm_mla_decode_fwd.__name__}: only support max_seqlen_q==1 when gqa_ratio==16"
        )

    func = compile(
        gqa_ratio,
        page_size,
        "__hip_bfloat16",
        "__hip_bfloat16",
        num_kv_splits,
        v_head_dim,
    )

    func(
        *torch_to_c_types(
            q,
            kv_buffer,
            qo_indptr,
            kv_indptr,
            kv_page_indices,
            kv_last_page_lens,
            max_seqlen_q,
            softmax_scale,
            logits,
            attn_lse,
            output,
            num_seqs,
            num_heads,
            num_kv_heads,
            q.stride(0),
            kv_buffer.stride(0),
            attn_lse.stride(0),
            attn_lse.stride(1),
            attn_lse.stride(2),
            output.stride(0),
            output.stride(1),
            torch.cuda.current_stream(),
        )
    )
    if num_kv_splits == 1 and num_heads == 128:
        return logits.view(total_query_len, num_heads, v_head_dim), attn_lse
    return logits, attn_lse


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gqa_ratio", type=int, required=True)
    parser.add_argument("--page_size", type=int, required=True)
    parser.add_argument("--q_dtype", type=str, required=True)
    parser.add_argument("--kv_dtype", type=str, required=True)
    parser.add_argument("--num_kv_splits", type=int, required=True)
    parser.add_argument("--v_head_dim", type=int, required=True)
    parser.add_argument("--func_name", type=str, default=None)
    args = parser.parse_args()
    compile(**vars(args))
