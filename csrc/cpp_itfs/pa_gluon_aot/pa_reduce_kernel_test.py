import sys
import hashlib
import triton
import triton.language as tl
import aiter
import torch
import numpy as np
import argparse

from triton.tools.compile import compile_kernel, CompileArgs
from jinja2 import Template
from aiter.test_common import perftest, run_perftest
from aiter.ops.triton.gluon.pa_decode_gluon import (
    paged_attention_decode_v2_reduce_kernel,
)
from csrc.cpp_itfs.torch_utils import torch_to_c_types
from csrc.cpp_itfs.utils import (
    compile_template_op,
    AITER_CORE_DIR,
    get_default_func_name,
    not_built,
    run_lib,
)
from op_tests.triton_tests.test_pa_decode_gluon import (
    torch_reduce_compute,
)


# os.environ['TRITON_CACHE_DIR'] = '/mnt/raid0/heyanguang/code/fa_triton/aiter/triton_cache'
compile_reduce_kernel_count = 0
TORCH_TO_TL_DTYPE = {
    aiter.dtypes.fp8: tl.float8e4b8,
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
}


def parse_version(version_str):
    """Parse version string into comparable tuple format, handling possible development version suffixes"""
    # Remove potential suffixes like .dev, +git etc.
    version_str = version_str.split("+")[0].split("-")[0]

    # Split version number and convert to integers
    parts = []
    for part in version_str.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            break

    return tuple(parts)


TRITON_VERSION = parse_version(triton.__version__)


def setup_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def tensor_to_hash(tensor: torch.Tensor, algorithm: str = "md5") -> str:
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
    tensor_data = tensor.contiguous().view(torch.uint8).detach().cpu().numpy().tobytes()

    hash_func.update(tensor_data)
    return hash_func.hexdigest()


def compile_reduce_kernel(
    compute_type: tl.dtype,
    query_group_size: int,
    head_size: int,
    max_context_partition_num: int,
    context_partition_size: int,
    md_name: str,
    func_name: str = None,
):
    """Compile the reduce kernel for paged attention decode."""
    head_size_pow2 = triton.next_power_of_2(head_size)

    if query_group_size < 16:
        query_group_size_pow2 = 16
    else:
        query_group_size_pow2 = triton.next_power_of_2(query_group_size)

    if func_name is None:
        func_name = get_default_func_name(
            md_name,
            (
                query_group_size_pow2,
                head_size_pow2,
                context_partition_size,
            ),
        )

    global compile_reduce_kernel_count
    compile_reduce_kernel_count += 1

    if compute_type == tl.bfloat16:
        logits_sig = "*bf16:16"
        output_sig = "*bf16:16"
    elif compute_type == tl.float16:
        logits_sig = "*fp16:16"
        output_sig = "*fp16:16"
    else:
        raise ValueError(f"Unsupported compute type: {compute_type}")

    # if not_built(func_name):
    if compile_reduce_kernel_count == 1:
        # Build signature based on kernel parameters
        signature_parts = [
            output_sig,  # output_ptr
            "*fp32:16",  # exp_sums_ptr
            "*fp32:16",  # max_logits_ptr
            logits_sig,  # logits_ptr
            "*i32:16",  # context_lengths_ptr
            "i32:16",  # stride_output_seq
            "i32:16",  # stride_output_head
            "i32:16",  # stride_exp_sums_seq
            "i32:16",  # stride_exp_sums_head
            "i32:16",  # stride_exp_sums_part
            "i32:16",  # stride_logits_seq
            "i32:16",  # stride_logits_head
            "i32:16",  # stride_logits_part
            "i32:16",  # stride_logits_group
            "i32:16",  # query_group_size
            "i32:16",  # head_size
            "i32:16",  # num_seqs
            "i32:16",  # num_kv_heads
            f"{query_group_size_pow2}",
            f"{head_size_pow2}",
            f"{context_partition_size}",
        ]
        signature = ",".join(signature_parts)

        reduce_kernel_name = "paged_attention_decode_v2_reduce_kernel_triton34"
        if TRITON_VERSION > (3, 4, 0):
            reduce_kernel_name = "paged_attention_decode_v2_reduce_kernel"

        compile_args = CompileArgs(
            path=f"{AITER_CORE_DIR}/aiter/ops/triton/gluon/pa_decode_gluon.py",
            kernel_name=reduce_kernel_name,
            signature=signature,
            grid="num_seqs,num_kv_heads,1",
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

        with open(
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa_gluon_aot/pa_decode_reduce_kernel.cpp.jinja",
            "r",
        ) as f:
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
    context_lengths: torch.Tensor,
    num_sequences: int,
    num_kv_heads: int,
    compute_type: tl.dtype,
    query_group_size: int,
    head_size: int,
    max_context_partition_num: int,
    context_partition_size: int,
    md_name: str,
    func_name: str = None,
):
    """
    Compile and run the compiled kernel with perftest timing
    """
    reduce_func = compile_reduce_kernel(
        compute_type=compute_type,
        query_group_size=query_group_size,
        head_size=head_size,
        max_context_partition_num=max_context_partition_num,
        context_partition_size=context_partition_size,
        md_name=md_name,
        func_name=func_name,
    )

    reduce_func(
        *torch_to_c_types(
            output,
            exp_sums,
            max_logits,
            temporary_output,
            context_lengths,
            output.stride(0),
            output.stride(1),
            exp_sums.stride(0),
            exp_sums.stride(1),
            exp_sums.stride(2),
            temporary_output.stride(0),
            temporary_output.stride(1),
            temporary_output.stride(2),
            temporary_output.stride(3),
            query_group_size,
            head_size,
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
    context_lengths: torch.Tensor,
    query_group_size: int,
    head_size: int,
    max_context_partition_num: int,
    context_partition_size: int,
):
    """
    Directly call the paged_attention_decode_v2_reduce_kernel with perftest timing
    """
    num_seqs = output.shape[0]
    num_kv_heads = exp_sums.shape[1]
    # Configure grid
    grid = (num_seqs, num_kv_heads, 1)

    if query_group_size < 16:
        QUERY_GROUP_SIZE_POW2 = 16
    else:
        QUERY_GROUP_SIZE_POW2 = triton.next_power_of_2(query_group_size)

    kernel = paged_attention_decode_v2_reduce_kernel_triton34
    if TRITON_VERSION > (3, 4, 0):
        kernel = paged_attention_decode_v2_reduce_kernel

    # Launch the kernel directly
    kernel[grid](
        output,
        exp_sums,
        max_logits,
        temporary_output,
        context_lengths,
        output.stride(0),
        output.stride(1),
        exp_sums.stride(0),
        exp_sums.stride(1),
        exp_sums.stride(2),
        temporary_output.stride(0),
        temporary_output.stride(1),
        temporary_output.stride(2),
        temporary_output.stride(3),
        query_group_size,
        head_size,
        num_seqs,
        num_kv_heads,
        QUERY_GROUP_SIZE_POW2=QUERY_GROUP_SIZE_POW2,
        HEAD_SIZE_POW2=triton.next_power_of_2(head_size),
        CONTEXT_PARTITION_SIZE=context_partition_size,
    )


def test_reduce_kernel(kernel_type: str = "compiled"):
    """Test the reduce kernel with provided parameters.

    Args:
        kernel_type: Type of kernel to test - "compiled" or "direct"
    """
    print(f"\n=== Testing Reduce Kernel (Type: {kernel_type}) ===")
    setup_seed(123)

    compute_type = aiter.dtypes.bf16
    # compute_type = aiter.dtypes.fp16
    data_type = compute_type
    compute_type = TORCH_TO_TL_DTYPE[compute_type]

    query_sequence_length = 1
    query_group_size = 8
    head_size = 128
    max_context_partition_num = 8
    context_partition_size = 256
    num_sequences = 128
    num_kv_heads = 8
    # num_kv_heads = 1

    equivalent_query_group_size = query_sequence_length * query_group_size

    # Create test tensors with the provided shapes
    # output = torch.zeros((num_sequences, num_kv_heads * equivalent_query_group_size, head_size),
    output = torch.empty(
        (num_sequences, num_kv_heads * equivalent_query_group_size, head_size),
        dtype=data_type,
        device="cuda",
    )
    exp_sums = torch.zeros(
        (
            num_sequences,
            num_kv_heads,
            max_context_partition_num,
            equivalent_query_group_size,
        ),
        dtype=torch.float32,
        device="cuda",
    )
    max_logits = torch.zeros(
        (
            num_sequences,
            num_kv_heads,
            max_context_partition_num,
            equivalent_query_group_size,
        ),
        dtype=torch.float32,
        device="cuda",
    )
    temporary_output = torch.zeros(
        (
            num_sequences,
            num_kv_heads,
            max_context_partition_num,
            equivalent_query_group_size,
            head_size,
        ),
        dtype=data_type,
        device="cuda",
    )
    context_lengths = torch.randint(
        1,
        context_partition_size * max_context_partition_num,
        (num_sequences,),
        dtype=torch.int32,
        device="cuda",
    )

    # Initialize with random data for testing
    torch.manual_seed(42)
    exp_sums.uniform_(0.1, 1.0)
    max_logits.uniform_(0.1, 5.0)
    temporary_output.uniform_(-1.0, 1.0)

    # Run reference implementation
    reference_output = torch_reduce_compute(
        output.clone(),
        exp_sums,
        max_logits,
        temporary_output,
        context_lengths,
        context_partition_size,
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
            context_lengths,
            num_sequences,
            num_kv_heads,
            compute_type=compute_type,
            query_group_size=equivalent_query_group_size,
            head_size=head_size,
            max_context_partition_num=max_context_partition_num,
            context_partition_size=context_partition_size,
            md_name="pa_decode_reduce_kernel",
        )
        print(f"Compiled kernel execution time: {compiled_time:.2f} us/iter")
    elif kernel_type == "direct":
        # Directly call the kernel from pa_decode_gluon.py
        print("\n=== Running Direct Kernel ===")
        _, direct_time = run_direct_kernel(
            output,
            exp_sums,
            max_logits,
            temporary_output,
            context_lengths,
            query_group_size,
            head_size,
            max_context_partition_num,
            context_partition_size,
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
            print(
                f"  Position {orig_idx}: kernel={output[orig_idx].item():.6f}, ref={reference_output[orig_idx].item():.6f}, diff={flat_diff[idx].item():.6e}"
            )

    # Test result
    # tolerance = 1e-4
    tolerance = 5e-3
    if max_diff < tolerance:
        print(
            f"\n✅ TEST PASSED: Max difference ({max_diff:.6e}) < tolerance ({tolerance})"
        )
    else:
        print(
            f"\n❌ TEST FAILED: Max difference ({max_diff:.6e}) >= tolerance ({tolerance})"
        )

    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "output_nan_cnt": output_nan_cnt,
        "reference_nan_cnt": reference_nan_cnt,
        "output_md5": output_md5,
        "reference_md5": reference_md5,
        "passed": max_diff < tolerance,
    }


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Test paged attention reduce kernel")
    parser.add_argument(
        "--kernel-type",
        type=str,
        choices=["compiled", "direct"],
        default="compiled",
        help="Type of kernel to test: 'compiled' (default) or 'direct'",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=101,
        help="Number of iterations for performance testing",
    )
    parser.add_argument(
        "--num-warmup", type=int, default=2, help="Number of warmup iterations"
    )

    args = parser.parse_args()

    result = test_reduce_kernel(kernel_type=args.kernel_type)
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
