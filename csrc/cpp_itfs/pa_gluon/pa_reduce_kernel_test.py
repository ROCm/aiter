import os
import sys
import hashlib
import triton
import functools
import aiter
import torch
import numpy as np
import argparse

from triton.tools.compile import compile_kernel, CompileArgs
from jinja2 import Template
from aiter.test_common import perftest, run_perftest
from aiter.ops.triton.gluon.pa_decode_triton_gluon_fp8 import paged_attention_decode_v2_reduce_kernel
from csrc.cpp_itfs.torch_utils import torch_to_c_types
from csrc.cpp_itfs.gluon_aot_tools.compile_gluon import compile_gluon_kernel, CompileGluonArgs
from csrc.cpp_itfs.utils import (
    compile_template_op,
    AITER_CORE_DIR,
    get_default_func_name,
    not_built,
    run_lib,
)


def setup_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def tensor_to_hash(tensor: torch.Tensor, algorithm: str = 'md5') -> str:
    """
    Convert a PyTorch tensor to a hash value using the specified algorithm.
    
    Args:
        tensor (torch.Tensor): Input tensor
        algorithm (str): Hash algorithm, defaults to 'md5',
                        options: 'md5', 'sha1', 'sha256', etc.
        
    Returns:
        str: Hexadecimal string representation of the hash value
    """
    hash_func = getattr(hashlib, algorithm)()
    
    # Process tensor data
    tensor_data = (
        tensor.contiguous()
        .view(torch.uint8)
        .detach()
        .cpu()
        .numpy()
        .tobytes()
    )
    
    hash_func.update(tensor_data)
    return hash_func.hexdigest()


def compile_reduce_kernel(
    head_size: int,
    query_group_size: int,
    sequence_partition_size: int,
    max_num_seq_partitions: int,
    md_name: str,
    func_name: str = None,
):
    """Compile the reduce kernel for paged attention decode."""
    if func_name is None:
        func_name = get_default_func_name(
            md_name,
            (head_size, query_group_size, sequence_partition_size, max_num_seq_partitions),
        )

    # if not_built(func_name):
    if True:
        # Build signature based on kernel parameters
        signature_parts = [
            "*bf16:16",  # output_ptr
            "*fp32:16",  # exp_sums_ptr
            "*fp32:16",  # max_logits_ptr
            "*bf16:16",  # logits_ptr
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
            f"{query_group_size}",
            f"{sequence_partition_size}",
            f"{max_num_seq_partitions}",
        ]
        signature = ",".join(signature_parts)

        compile_args = CompileArgs(
            path=f"{AITER_CORE_DIR}/aiter/ops/triton/gluon/pa_decode_triton_gluon_fp8.py",
            kernel_name="paged_attention_decode_v2_reduce_kernel",
            signature=signature,
            grid="num_seqs,num_kv_heads,1",
            # grid="num_seqs,1,1",
            num_warps=4,
            num_stages=2,
            out_name=md_name,
        )
        triton_kernel, output_files = compile_kernel(compile_args)
        triton_header = None
        triton_source = None
        for output_file in output_files:
            if output_file.suffix == ".h":
                triton_header = output_file
            elif output_file.suffix == ".cpp":
                triton_source = output_file

        with open(f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa_gluon/pa_decode_reduce_kernel.cpp.jinja", "r") as f:
            src_template = Template(f.read())

        kernel_name = "paged_attention_decode_v2_reduce_kernel"
        return compile_template_op(
            src_template,
            md_name,
            [triton_header],
            [triton_source],
            triton_header=triton_header,
            kernel_name=kernel_name,
            triton_kernel=triton_kernel,
            func_name=func_name,
        )
    else:
        return run_lib(func_name)


@perftest()
def run_compiled_kernel(
    output: torch.Tensor,
    exp_sums: torch.Tensor,
    max_logits: torch.Tensor,
    temporary_output: torch.Tensor,
    sequence_lengths: torch.Tensor,
    num_sequences: int,
    num_kv_heads: int,
    head_size: int,
    query_group_size: int,
    sequence_partition_size: int,
    max_num_seq_partitions: int,
    md_name: str,
    func_name: str = None,
):
    """
    Compile and run the compiled kernel with perftest timing
    """
    reduce_func = compile_reduce_kernel(
        head_size=head_size,
        query_group_size=query_group_size,
        sequence_partition_size=sequence_partition_size,
        max_num_seq_partitions=max_num_seq_partitions,
        md_name=md_name,
    )

    reduce_func(
        *torch_to_c_types(
            output,
            exp_sums,
            max_logits,
            temporary_output,
            sequence_lengths,
            output.stride(0),
            output.stride(1),
            exp_sums.stride(0),
            exp_sums.stride(1),
            exp_sums.stride(2),
            temporary_output.stride(0),
            temporary_output.stride(1),
            temporary_output.stride(2),
            temporary_output.stride(3),
            num_sequences,
            num_kv_heads,
            torch.cuda.current_stream(output.device),
        )
    )


@perftest()
def run_direct_kernel(
    output: torch.Tensor,
    exp_sums: torch.Tensor,
    max_logits: torch.Tensor,
    temporary_output: torch.Tensor,
    sequence_lengths: torch.Tensor,
    head_size: int,
    query_group_size: int,
    sequence_partition_size: int,
    max_num_seq_partitions: int,
):
    """
    Directly call the paged_attention_decode_v2_reduce_kernel with perftest timing
    """
    num_seqs = output.shape[0]
    num_kv_heads = exp_sums.shape[1]

    # Configure grid
    grid = (num_seqs, num_kv_heads, 1)

    # Launch the kernel directly
    paged_attention_decode_v2_reduce_kernel[grid](
        output,
        exp_sums,
        max_logits,
        temporary_output,
        sequence_lengths,
        output.stride(0),
        output.stride(1),
        exp_sums.stride(0),
        exp_sums.stride(1),
        exp_sums.stride(2),
        temporary_output.stride(0),
        temporary_output.stride(1),
        temporary_output.stride(2),
        temporary_output.stride(3),
        num_seqs,
        num_kv_heads,
        HEAD_SIZE=head_size,
        QUERY_GROUP_SIZE=query_group_size,
        SEQUENCE_PARTITION_SIZE=sequence_partition_size,
        MAX_NUM_SEQ_PARTITIONS=max_num_seq_partitions,
    )


def torch_reduce_reference(
    output: torch.Tensor,  # [num_seqs, num_q_heads_total, head_size]
    exp_sums: torch.Tensor,  # [num_seqs, num_kv_heads, max_num_partitions, query_group_size]
    max_logits: torch.Tensor,  # [num_seqs, num_kv_heads, max_num_partitions, query_group_size]
    temporary_output: torch.Tensor,  # [num_seqs, num_kv_heads, max_num_partitions, query_group_size, head_size]
    sequence_lengths: torch.Tensor,  # [num_seqs]
    sequence_partition_size: int = 256,
) -> torch.Tensor:
    """
    Reference implementation of the reduce kernel.
    This mimics the reduce stage from torch_mha_extend_flashattn_style function.
    """
    num_seqs = output.shape[0]
    num_q_heads_total = output.shape[1]
    head_size = output.shape[2]
    final_output = torch.zeros_like(output)
    # final_output = torch.empty_like(output)

    for seq_idx in range(num_seqs):
        seq_len = sequence_lengths[seq_idx].item()
        num_parts = (seq_len + sequence_partition_size - 1) // sequence_partition_size

        # Global max across partitions
        global_max = (
            max_logits[seq_idx, :, :num_parts, :].max(dim=1).values
        )  # [num_kv_heads, query_group_size]

        # Rescale exp_sums
        exp_sums_local = exp_sums[
            seq_idx, :, :num_parts, :
        ]  # [num_kv_heads, num_parts, query_group_size]
        max_local = max_logits[
            seq_idx, :, :num_parts, :
        ]  # [num_kv_heads, num_parts, query_group_size]
        exp_sums_rescaled = exp_sums_local * torch.exp(
            max_local - global_max.unsqueeze(1)
        )  # [num_kv_heads, num_parts, query_group_size]
        global_exp_sum = exp_sums_rescaled.sum(
            dim=1
        )  # [num_kv_heads, query_group_size]

        # Avoid division by zero
        global_exp_sum = torch.clamp(global_exp_sum, min=1e-12)

        # Weighted sum of partial outputs
        weights = exp_sums_rescaled / global_exp_sum.unsqueeze(
            1
        )  # [num_kv_heads, num_parts, query_group_size]
        partial_seq = temporary_output[
            seq_idx, :, :num_parts, :, :
        ]  # [num_kv_heads, num_parts, query_group_size, head]
        weighted = (partial_seq * weights.unsqueeze(-1)).sum(
            dim=1
        )  # [num_kv_heads, query_group_size, head]

        final_output[seq_idx] = weighted.view(num_q_heads_total, head_size)

    return final_output


def test_reduce_kernel(kernel_type: str = "compiled"):
    """Test the reduce kernel with provided parameters.

    Args:
        kernel_type: Type of kernel to test - "compiled" or "direct"
    """
    print(f"\n=== Testing Reduce Kernel (Type: {kernel_type}) ===")
    setup_seed(123)

    # Parameters from the provided debug information
    head_size = 128
    query_group_size = 8
    sequence_partition_size = 256
    max_num_seq_partitions = 16
    num_sequences = 128
    num_kv_heads = 2
    # num_kv_heads = 1

    # print("\n=== Reduce Kernel Compile Parameters ===")
    # print(f"  head_size: {head_size}")
    # print(f"  query_group_size: {query_group_size}")
    # print(f"  sequence_partition_size: {sequence_partition_size}")
    # print(f"  max_num_seq_partitions: {max_num_seq_partitions}")

    # Create test tensors with the provided shapes
    # output = torch.zeros((num_sequences, num_kv_heads * query_group_size, head_size),
    output = torch.empty((num_sequences, num_kv_heads * query_group_size, head_size),
                        dtype=torch.bfloat16, device='cuda')
    exp_sums = torch.zeros((num_sequences, num_kv_heads, max_num_seq_partitions, query_group_size),
                          dtype=torch.float32, device='cuda')
    max_logits = torch.zeros((num_sequences, num_kv_heads, max_num_seq_partitions, query_group_size),
                            dtype=torch.float32, device='cuda')
    temporary_output = torch.zeros((num_sequences, num_kv_heads, max_num_seq_partitions, query_group_size, head_size),
                                  dtype=torch.bfloat16, device='cuda')
    sequence_lengths = torch.randint(1, sequence_partition_size * max_num_seq_partitions,
                                    (num_sequences,), dtype=torch.int32, device='cuda')

    # Initialize with random data for testing
    torch.manual_seed(42)
    exp_sums.uniform_(0.1, 1.0)
    max_logits.uniform_(0.1, 5.0)
    temporary_output.uniform_(-1.0, 1.0)

    # print("\n=== Reduce Kernel Execution Parameters ===")
    # print("Tensor parameters:")
    # print(f"  output: shape={output.shape}, dtype={output.dtype}")
    # print(f"  exp_sums: shape={exp_sums.shape}, dtype={exp_sums.dtype}")
    # print(f"  max_logits: shape={max_logits.shape}, dtype={max_logits.dtype}")
    # print(f"  temporary_output: shape={temporary_output.shape}, dtype={temporary_output.dtype}")
    # print(f"  sequence_lengths: shape={sequence_lengths.shape}, dtype={sequence_lengths.dtype}")

    # print("\nScalar parameters:")
    # print(f"  output.stride(0): {output.stride(0)}")
    # print(f"  output.stride(1): {output.stride(1)}")
    # print(f"  exp_sums.stride(0): {exp_sums.stride(0)}")
    # print(f"  exp_sums.stride(1): {exp_sums.stride(1)}")
    # print(f"  exp_sums.stride(2): {exp_sums.stride(2)}")
    # print(f"  temporary_output.stride(0): {temporary_output.stride(0)}")
    # print(f"  temporary_output.stride(1): {temporary_output.stride(1)}")
    # print(f"  temporary_output.stride(2): {temporary_output.stride(2)}")
    # print(f"  temporary_output.stride(3): {temporary_output.stride(3)}")
    # print(f"  num_sequences: {num_sequences}")
    # print(f"  num_kv_heads: {num_kv_heads}")

    # Run reference implementation
    reference_output = torch_reduce_reference(
        output.clone(),
        exp_sums,
        max_logits,
        temporary_output,
        sequence_lengths,
        sequence_partition_size
    )

    # Execute kernel based on selected type
    if kernel_type == "compiled":
        # Compile and run the compiled kernel
        print("\n=== Running Compiled Kernel ===")
        _, compiled_time = run_compiled_kernel(
            output,
            exp_sums,
            max_logits,
            temporary_output,
            sequence_lengths,
            num_sequences,
            num_kv_heads,
            head_size=head_size,
            query_group_size=query_group_size,
            sequence_partition_size=sequence_partition_size,
            max_num_seq_partitions=max_num_seq_partitions,
            md_name="pa_decode_reduce_kernel",
        )
        print(f"Compiled kernel execution time: {compiled_time:.2f} us/iter")
    elif kernel_type == "direct":
        # Directly call the kernel from pa_decode_triton_gluon_fp8.py
        print("\n=== Running Direct Kernel ===")
        _, direct_time = run_direct_kernel(
            output,
            exp_sums,
            max_logits,
            temporary_output,
            sequence_lengths,
            head_size,
            query_group_size,
            sequence_partition_size,
            max_num_seq_partitions,
        )
        print(f"Direct kernel execution time: {direct_time:.2f} us/iter")
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    # Compare results
    print("\n=== Comparing Results ===")
    diff = (output - reference_output).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")

    # Check for NaN values
    reference_nan_cnt = torch.isnan(reference_output).sum().item()
    output_nan_cnt = torch.isnan(output).sum().item()
    print(f"Reference NaN count: {reference_nan_cnt}")
    print(f"Output NaN count: {output_nan_cnt}")

    # MD5 hashes for verification
    reference_md5 = tensor_to_hash(reference_output)
    output_md5 = tensor_to_hash(output)
    print(f"Reference MD5: {reference_md5}")
    print(f"Output MD5: {output_md5}")

    # Detailed error analysis
    if max_diff > 1e-4:
        print("\n=== Detailed Error Analysis ===")
        # Find top 5 differences
        flat_diff = diff.flatten()
        top_k = 5
        top_k_indices = torch.topk(flat_diff, top_k).indices

        print(f"Top {top_k} differences:")
        for i in range(top_k):
            idx = top_k_indices[i]
            orig_idx = np.unravel_index(idx.cpu().numpy(), output.shape)
            print(f"  Position {orig_idx}: kernel={output[orig_idx].item():.6f}, ref={reference_output[orig_idx].item():.6f}, diff={flat_diff[idx].item():.6e}")

    # Test result
    # tolerance = 1e-4
    tolerance = 5e-3
    if max_diff < tolerance:
        print(f"\n✅ TEST PASSED: Max difference ({max_diff:.6e}) < tolerance ({tolerance})")
    else:
        print(f"\n❌ TEST FAILED: Max difference ({max_diff:.6e}) >= tolerance ({tolerance})")

    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "output_nan_cnt": output_nan_cnt,
        "reference_nan_cnt": reference_nan_cnt,
        "output_md5": output_md5,
        "reference_md5": reference_md5,
        "passed": max_diff < tolerance
    }


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Test paged attention reduce kernel")
    parser.add_argument(
        "--kernel-type",
        type=str,
        choices=["compiled", "direct"],
        default="compiled",
        help="Type of kernel to test: 'compiled' (default) or 'direct'"
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=101,
        help="Number of iterations for performance testing"
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=2,
        help="Number of warmup iterations"
    )

    args = parser.parse_args()

    result = test_reduce_kernel(kernel_type=args.kernel_type)
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
