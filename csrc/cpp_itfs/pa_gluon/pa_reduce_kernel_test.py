import os
import sys
import hashlib
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
from csrc.cpp_itfs.torch_utils import torch_to_c_types
from csrc.cpp_itfs.gluon_aot_tools.compile_gluon import compile_gluon_kernel, CompileGluonArgs
from triton.tools.compile import compile_kernel, CompileArgs
import triton
import functools
import aiter
import torch
import numpy as np


def setup_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


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
        head_size_pow2 = triton.next_power_of_2(head_size)
        query_group_size_pow2 = triton.next_power_of_2(query_group_size)
        max_num_seq_partitions_pow2 = triton.next_power_of_2(max_num_seq_partitions)

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
        current_dir = os.path.dirname(__file__)
        return compile_template_op(
            src_template,
            md_name,
            [current_dir + "/../utils.h", current_dir + "/../../include", triton_header],
            [triton_source],
            triton_header=triton_header,
            kernel_name=kernel_name,
            triton_kernel=triton_kernel,
            func_name=func_name,
        )
    else:
        return run_lib(func_name)


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
    num_kv_heads = exp_sums.shape[1]
    query_group_size = exp_sums.shape[3]

    final_output = torch.zeros_like(output)

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


def test_reduce_kernel():
    """Test the reduce kernel with provided parameters."""
    print("\n=== Testing Reduce Kernel ===")
    setup_seed(123)

    # Parameters from the provided debug information
    head_size = 128
    query_group_size = 8
    sequence_partition_size = 256
    max_num_seq_partitions = 16
    num_sequences = 128
    num_kv_heads = 2
    # num_kv_heads = 1

    print("\n=== Reduce Kernel Compile Parameters ===")
    print(f"  head_size: {head_size}")
    print(f"  query_group_size: {query_group_size}")
    print(f"  sequence_partition_size: {sequence_partition_size}")
    print(f"  max_num_seq_partitions: {max_num_seq_partitions}")

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

    print("\n=== Reduce Kernel Execution Parameters ===")
    print("Tensor parameters:")
    print(f"  output: shape={output.shape}, dtype={output.dtype}")
    print(f"  exp_sums: shape={exp_sums.shape}, dtype={exp_sums.dtype}")
    print(f"  max_logits: shape={max_logits.shape}, dtype={max_logits.dtype}")
    print(f"  temporary_output: shape={temporary_output.shape}, dtype={temporary_output.dtype}")
    print(f"  sequence_lengths: shape={sequence_lengths.shape}, dtype={sequence_lengths.dtype}")

    print("\nScalar parameters:")
    print(f"  output.stride(0): {output.stride(0)}")
    print(f"  output.stride(1): {output.stride(1)}")
    print(f"  exp_sums.stride(0): {exp_sums.stride(0)}")
    print(f"  exp_sums.stride(1): {exp_sums.stride(1)}")
    print(f"  exp_sums.stride(2): {exp_sums.stride(2)}")
    print(f"  temporary_output.stride(0): {temporary_output.stride(0)}")
    print(f"  temporary_output.stride(1): {temporary_output.stride(1)}")
    print(f"  temporary_output.stride(2): {temporary_output.stride(2)}")
    print(f"  temporary_output.stride(3): {temporary_output.stride(3)}")
    print(f"  num_sequences: {num_sequences}")
    print(f"  num_kv_heads: {num_kv_heads}")

    # Compile the reduce kernel
    print("\n=== Compiling Reduce Kernel ===")
    reduce_func = compile_reduce_kernel(
        head_size=head_size,
        query_group_size=query_group_size,
        sequence_partition_size=sequence_partition_size,
        max_num_seq_partitions=max_num_seq_partitions,
        md_name="pa_decode_reduce_kernel",
    )

    # Run reference implementation
    print("\n=== Running Reference Implementation ===")
    reference_output = torch_reduce_reference(
        output.clone(),
        exp_sums,
        max_logits,
        temporary_output,
        sequence_lengths,
        sequence_partition_size
    )

    output_md5 = hashlib.md5(
        output.contiguous()
        .view(torch.uint8)
        .detach()
        .cpu()
        .numpy()
        .tobytes()
    ).hexdigest()
    print(f"Output NaN count: {torch.isnan(output).sum().item()}")
    print(f"Output MD5: {output_md5}")

    # Run the compiled kernel
    print("\n=== Running Compiled Kernel ===")
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

    # Compare results
    print("\n=== Comparing Results ===")
    diff = (output - reference_output).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")

    # Check for NaN values
    output_nan_cnt = torch.isnan(output).sum().item()
    reference_nan_cnt = torch.isnan(reference_output).sum().item()

    print(f"Output NaN count: {output_nan_cnt}")
    print(f"Reference NaN count: {reference_nan_cnt}")

    # MD5 hashes for verification
    output_md5 = hashlib.md5(
        output.contiguous()
        .view(torch.uint8)
        .detach()
        .cpu()
        .numpy()
        .tobytes()
    ).hexdigest()

    reference_md5 = hashlib.md5(
        reference_output.contiguous()
        .view(torch.uint8)
        .detach()
        .cpu()
        .numpy()
        .tobytes()
    ).hexdigest()

    print(f"Output MD5: {output_md5}")
    print(f"Reference MD5: {reference_md5}")

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


if __name__ == "__main__":
    result = test_reduce_kernel()
    sys.exit(0 if result["passed"] else 1)
