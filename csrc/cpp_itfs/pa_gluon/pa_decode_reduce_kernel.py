import os
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


MD_NAME = "pa_decode_reduce_kernel"
warpSize = 64
with open(f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa_gluon/pa_decode_reduce_kernel.cpp.jinja", "r") as f:
    src_template = Template(f.read())


def compile(
    head_size: int,
    query_group_size: int,
    sequence_partition_size: int,
    max_num_seq_partitions: int,
    func_name: str = None,
):
    if func_name is None:
        func_name = get_default_func_name(
            MD_NAME,
            (head_size, query_group_size, sequence_partition_size, max_num_seq_partitions),
        )

    if not_built(func_name):
        head_size_pow2 = triton.next_power_of_2(head_size)
        query_group_size_pow2 = triton.next_power_of_2(query_group_size)
        max_num_seq_partitions_pow2 = triton.next_power_of_2(max_num_seq_partitions)

        # Build signature based on kernel parameters
        signature_parts = [
            "*fp32:16",  # output_ptr
            "*fp32:16",  # exp_sums_ptr
            "*fp32:16",  # max_logits_ptr
            "*fp32:16",  # logits_ptr
            "*i32:16",   # sequence_lengths_ptr
            "i32",       # stride_output_seq
            "i32",       # stride_output_head
            "i32",       # stride_exp_sums_seq
            "i32",       # stride_exp_sums_head
            "i32",       # stride_exp_sums_part
            "i32",       # stride_logits_seq
            "i32",       # stride_logits_head
            "i32",       # stride_logits_part
            "i32",       # stride_logits_group
            "i32",       # num_seqs
            "i32",       # num_kv_heads
            f"{head_size}",
            f"{head_size_pow2}",
            f"{query_group_size}",
            f"{query_group_size_pow2}",
            f"{sequence_partition_size}",
            f"{max_num_seq_partitions}",
            f"{max_num_seq_partitions_pow2}",
        ]

        signature = ",".join(signature_parts)

        compile_args = CompileArgs(
            path=f"{AITER_CORE_DIR}/aiter/ops/triton/gluon/pa_decode_triton_gluon_fp8.py",
            kernel_name="paged_attention_decode_v2_reduce_kernel",
            signature=signature,
            grid="num_seqs,num_kv_heads,1",
            num_warps=4,
            num_stages=2,
            out_name="pa_decode_reduce_kernel",
        )
        triton_kernel, output_files = compile_kernel(compile_args)
        triton_header = None
        triton_source = None
        for output_file in output_files:
            if output_file.suffix == ".h":
                triton_header = output_file
            elif output_file.suffix == ".cpp":
                triton_source = output_file

        kernel_name = "paged_attention_decode_v2_reduce_kernel"
        current_dir = os.path.dirname(__file__)
        return compile_template_op(
            src_template,
            MD_NAME,
            [current_dir + "/../utils.h", current_dir + "/../../include", triton_header],
            [triton_source],
            triton_header=triton_header,
            kernel_name=kernel_name,
            triton_kernel=triton_kernel,
            func_name=func_name,
        )
    else:
        return run_lib(func_name)


def pa_decode_reduce_kernel(
    output,             # [num_seqs, num_kv_heads, query_group_size, head_size]
    exp_sums,           # [num_seqs, num_kv_heads, max_parts, query_group_size]
    max_logits,         # [num_seqs, num_kv_heads, max_parts, query_group_size]
    logits,             # [num_seqs, num_kv_heads, max_parts, query_group_size, head_size]
    sequence_lengths,   # [num_seqs]
    head_size: int,
    query_group_size: int,
    sequence_partition_size: int = 256,
    max_num_seq_partitions: int = None,
):
    import torch
    from csrc.cpp_itfs.torch_utils import torch_to_c_types

    # Extract dimensions
    num_seqs = output.shape[0]
    num_kv_heads = output.shape[1]

    # Calculate power-of-2 values
    head_size_pow2 = triton.next_power_of_2(head_size)
    query_group_size_pow2 = triton.next_power_of_2(query_group_size)

    # Calculate max_num_seq_partitions if not provided
    if max_num_seq_partitions is None:
        max_sequence_length = sequence_lengths.max().item()
        max_num_seq_partitions = (max_sequence_length + sequence_partition_size - 1) // sequence_partition_size

    max_num_seq_partitions_pow2 = triton.next_power_of_2(max_num_seq_partitions)

    func = compile(
        head_size=head_size,
        query_group_size=query_group_size,
        sequence_partition_size=sequence_partition_size,
        max_num_seq_partitions=max_num_seq_partitions,
    )

    func(
        *torch_to_c_types(
            output,
            exp_sums,
            max_logits,
            logits,
            sequence_lengths,
            output.stride(0),
            output.stride(1),
            exp_sums.stride(0),
            exp_sums.stride(1),
            exp_sums.stride(2),
            logits.stride(0),
            logits.stride(1),
            logits.stride(2),
            logits.stride(3),
            num_seqs,
            num_kv_heads,
            torch.cuda.current_stream(output.device),
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--head_size", type=int, required=True)
    parser.add_argument("--query_group_size", type=int, required=True)
    parser.add_argument("--sequence_partition_size", type=int, required=True)
    parser.add_argument("--max_num_seq_partitions", type=int, required=True)
    parser.add_argument("--func_name", type=str, default=None)
    args = parser.parse_args()
    compile(**vars(args))