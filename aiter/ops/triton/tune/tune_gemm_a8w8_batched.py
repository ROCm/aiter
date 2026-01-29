import time
from typing import Any, Callable, Dict, Tuple, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from aiter.ops.triton.batched_gemm_a8w8 import batched_gemm_a8w8
from aiter.ops.triton.tune.base import (
    save_configs_to_json,
    tune_kernel,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH


def run_torch_reference(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Run reference implementation using PyTorch for batched GEMM A8W8.

    Based on test_batched_gemm_a8w8.py pattern.

    Args:
        x: Input tensor of shape (B, M, K)
        w: Weight tensor of shape (B, N, K)
        x_scale: Scale tensor for x with shape (B, M, 1)
        w_scale: Scale tensor for w with shape (B, 1, N)
        bias: Bias tensor with shape (B, 1, N) or None
        dtype: Output dtype

    Returns:
        Reference output tensor
    """
    B = x.size(0)
    M = x.size(1)
    N = w.size(1)
    out = torch.empty(B, M, N, dtype=torch.bfloat16, device="cuda")

    for b in range(B):
        b_x = F.linear(x[b, :, :].to(torch.float32), w[b, :, :].to(torch.float32))
        b_scale = torch.matmul(x_scale[b, :, :], w_scale[b, :, :])
        b_out = torch.mul(b_x, b_scale)
        if bias is not None:
            b_out = b_out.to(bias[b, :, :]) + bias[b, :, :]
        out[b, :, :] = b_out

    return out.to(dtype)


def input_helper(
    B: int,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype = torch.bfloat16,
    layout: str = "TN",
):
    """
    Generate input tensors for batched GEMM A8W8 kernel.

    Args:
        B: Batch size
        M, N, K: Matrix dimensions
        dtype: Output data type
        layout: Memory layout (default "TN" - matches test)

    Returns:
        Tuple of (x, w, x_scale, w_scale, bias, y, config)
    """
    # Generate INT8 tensors with range -20 to 20 (matching test file)
    x = torch.randint(-20, 20, (B, M, K), dtype=torch.int8, device="cuda")
    w = torch.randint(-20, 20, (B, N, K), dtype=torch.int8, device="cuda")

    # Generate positive scale tensors (avoid zeros)
    x_scale = torch.rand([B, M, 1], dtype=torch.float32, device="cuda") + 1e-6
    w_scale = torch.rand([B, 1, N], dtype=torch.float32, device="cuda") + 1e-6

    # No bias tensor (simpler tuning)
    bias = None

    # Pre-allocated output tensor
    y = torch.empty((B, M, N), dtype=dtype, device="cuda")

    # Base config that will be modified during tuning
    config = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
        "waves_per_eu": 2,
        "kpack": 2,
        "matrix_instr_nonkdim": 16,
    }

    return x, w, x_scale, w_scale, bias, y, config


def make_run_and_gt_fn_factory(
    B: int,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Factory function to create run and ground truth functions for tuning.

    Args:
        B: Batch size
        M, N, K: Matrix dimensions
        dtype: Output data type

    Returns:
        Function that takes a config and returns (run_fn, ground_truth)
    """
    def make_run_and_gt(config: Dict[str, Any]) -> Tuple[Callable[[], Any], Any]:
        x, w, x_scale, w_scale, bias, y, _ = input_helper(B, M, N, K, dtype)

        def run() -> torch.Tensor:
            return batched_gemm_a8w8(x, w, x_scale, w_scale, bias, dtype, YQ=y, config=config)

        ground_truth = run_torch_reference(x, w, x_scale, w_scale, bias, dtype)
        return run, ground_truth

    return make_run_and_gt


def get_configs_compute_bound() -> List[Dict[str, Any]]:
    """
    Generate configuration space for tuning the batched gemm_a8w8 kernel.

    Full comprehensive search space for batched operations.

    Returns:
        List of configuration dictionaries
    """
    configs = []

    for num_stages in [1, 2, 3, 4]:
        for block_m in [32, 64, 128, 256]:
            for block_n in [32, 64, 128, 256]:
                for block_k in [64, 128, 256]:
                    for group_size_m in [1, 8]:
                        for num_warps in [2, 4, 8]:
                            for waves_per_eu in [2, 4]:
                                for kpack in [2]:
                                    configs.append({
                                        "BLOCK_SIZE_M": block_m,
                                        "BLOCK_SIZE_N": block_n,
                                        "BLOCK_SIZE_K": block_k,
                                        "GROUP_SIZE_M": group_size_m,
                                        "num_warps": num_warps,
                                        "num_stages": num_stages,
                                        "waves_per_eu": waves_per_eu,
                                        "kpack": kpack,
                                        "matrix_instr_nonkdim": 16,
                                    })

    return configs


def tune_gemm_a8w8_batched(
    B: int,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    search_space: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Tune the batched GEMM A8W8 kernel for specific dimensions.

    Args:
        B: Batch size
        M, N, K: Matrix dimensions
        dtype: Output data type
        search_space: List of configuration dictionaries to test

    Returns:
        Best configuration dictionary
    """
    make_run_and_gt = make_run_and_gt_fn_factory(B, M, N, K, dtype)

    best_config = tune_kernel(
        search_space=search_space,
        make_run_and_gt_fn=make_run_and_gt,
    )
    return best_config


def tune_and_save_configs(
    batch_sizes: List[int],
    M_values: List[int],
    N: int,
    K: int,
    dtype: torch.dtype,
    search_space: List[Dict[str, Any]],
    save_path: str,
    device_name: str,
    tag: str,
):
    """
    Tune configurations for multiple M values and save results.

    Note: Batch size (B) is fixed at a representative value (16) since the kernel
    processes each batch element independently. Config selection is primarily based
    on M dimension (context length) using standard M_LEQ_x format.

    Args:
        batch_sizes: List of batch sizes to test (will use median value)
        M_values: List of M dimensions to test
        N, K: Matrix dimensions
        dtype: Output data type
        search_space: List of configuration dictionaries to test
        save_path: Path to save configuration files
        device_name: Device/architecture name for file naming
        tag: Tag for configuration file naming
    """
    start = time.time()
    benchmark_results = []

    # Use median batch size for tuning (kernel processes batches independently)
    # The batch dimension doesn't significantly affect optimal config parameters
    batch_size = batch_sizes[len(batch_sizes) // 2]  # Use middle value (e.g., 16 from [4,8,16,32,64])

    # Test different M values (context lengths) - this is what matters for config selection
    for M in M_values:
        config = tune_gemm_a8w8_batched(batch_size, M, N, K, dtype, search_space)
        benchmark_results.append((M, config))

    # Categorize configs by M dimension using standard M_LEQ_x format
    # This matches the config selection logic in gemm_config_utils.py
    # Maps each tested M value to its corresponding M_LEQ_x key
    best_configs = {}
    for M, config in benchmark_results:
        # Map M to standard M_LEQ_x keys (follows power-of-2 boundaries)
        # The get_gemm_config function searches in order: M_LEQ_4, M_LEQ_8, M_LEQ_16, etc.
        # So M=16 gets mapped to M_LEQ_16, M=32 to M_LEQ_32, etc.
        if M <= 4:
            best_configs["M_LEQ_4"] = config
        elif M <= 8:
            best_configs["M_LEQ_8"] = config
        elif M <= 16:
            best_configs["M_LEQ_16"] = config
        elif M <= 32:
            best_configs["M_LEQ_32"] = config
        elif M <= 64:
            best_configs["M_LEQ_64"] = config
        elif M <= 128:
            best_configs["M_LEQ_128"] = config
        elif M <= 256:
            best_configs["M_LEQ_256"] = config
        elif M <= 512:
            best_configs["M_LEQ_512"] = config
        elif M <= 1024:
            best_configs["M_LEQ_1024"] = config
        elif M <= 2048:
            best_configs["M_LEQ_2048"] = config
        elif M <= 4096:
            best_configs["M_LEQ_4096"] = config
        else:
            best_configs["M_LEQ_8192"] = config

    # Add fallback "any" config (use the largest M tested)
    if benchmark_results:
        best_configs["any"] = benchmark_results[-1][1]

    json_file_name = f"{device_name}-BATCHED_GEMM-A8W8-{tag}.json"
    save_configs_to_json(json_file_name, save_path, best_configs)

    end = time.time()
    print(f"Tuning for {tag} took {end - start:.2f} seconds")


def main():
    """Main tuning entry point."""
    dev = arch_info.get_arch()
    torch.cuda.init()

    # Test configurations based on common LLM dimensions
    # Format: (batch_sizes, M_values, N, K)
    # M_values align with power-of-2 boundaries following gemm_config_utils.STANDARD_M_BOUNDS
    test_configs = [
        ([4, 8, 16, 32, 64], [16, 32, 64, 128, 256, 512, 1024, 2048, 4096], 1024, 1024),  # Standard attention
        ([4, 8, 16, 32, 64], [16, 32, 64, 128, 256, 512, 1024, 2048, 4096], 4096, 1024),  # FFN intermediate
        ([4, 8, 16, 32, 64], [16, 32, 64, 128, 256, 512, 1024, 2048, 4096], 1024, 2048),  # Wider input
        ([4, 8, 16, 32, 64], [16, 32, 64, 128, 256, 512, 1024, 2048, 4096], 6144, 1024),  # Larger FFN
        ([4, 8, 16, 32, 64], [16, 32, 64, 128, 256, 512, 1024, 2048, 4096], 1024, 3072),  # Deeper input
    ]

    search_space = get_configs_compute_bound()
    save_path = AITER_TRITON_CONFIGS_PATH + "/gemm/"

    print(f"Architecture: {dev}")
    print(f"Search space size: {len(search_space)} configurations")
    print(f"Total test configurations: {len(test_configs)}")
    print()

    # Tune for each configuration
    for batch_sizes, M_values, N, K in test_configs:
        tag = f"BATCHED-N={N}-K={K}"
        print(f"Running {tag}...")
        tune_and_save_configs(
            batch_sizes=batch_sizes,
            M_values=M_values,
            N=N,
            K=K,
            dtype=torch.bfloat16,
            search_space=search_space,
            save_path=save_path,
            device_name=dev,
            tag=tag,
        )


if __name__ == "__main__":
    main()
