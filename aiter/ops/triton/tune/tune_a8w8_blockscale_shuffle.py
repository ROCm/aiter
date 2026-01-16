import os
import time
from typing import Any, Callable, Dict, Tuple, List

import torch
import triton
import torch.nn.functional as F
from tqdm import tqdm

from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
    gemm_a8w8_blockscale_preshuffle,
)
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.tune.base import (
    get_search_space,
    save_configs_to_json,
    tune_kernel,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils.types import get_fp8_dtypes


# Get FP8 data types
e5m2_type, e4m3_type = get_fp8_dtypes()


def run_torch_reference(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    block_shape: Tuple[int, int],
    dtype=torch.bfloat16,
) -> torch.Tensor:
    """
    Run reference implementation using PyTorch for blockscale kernel.

    Args:
        x: Input tensor of shape (M, K)
        w: Weight tensor of shape (N, K)
        x_scale: Scale tensor for x with shape (M, scale_k)
        w_scale: Scale tensor for w with shape (scale_n, scale_k)
        block_shape: Tuple of (block_shape_n, block_shape_k)
        dtype: Output dtype

    Returns:
        Reference output tensor
    """
    block_shape_n, block_shape_k = block_shape
    m, k = x.shape
    n = w.shape[0]

    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k

    # Expand scales to match the block sizes
    x_scale_expanded = x_scale.repeat_interleave(block_shape_k, dim=1)
    x_dequant = x.to(x_scale_expanded.dtype) * x_scale_expanded[:m, :k]

    w_scale_expanded = w_scale.repeat_interleave(block_shape_n, dim=0)
    w_scale_expanded = w_scale_expanded.repeat_interleave(block_shape_k, dim=1)
    w_scale_expanded = w_scale_expanded[:n, :k]
    weight_dequant = w.to(w_scale_expanded.dtype) * w_scale_expanded

    out = F.linear(x_dequant.to(torch.float32), weight_dequant.to(torch.float32))

    return out.to(dtype)


def input_helper(
    M: int,
    N: int,
    K: int,
    block_shape_n: int = 128,
    block_shape_k: int = 128,
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Generate input tensors for GEMM A8W8 blockscale kernel with shuffled weights.

    Args:
        M: Batch/sequence size
        N: Output feature dimension
        K: Input feature dimension
        block_shape_n: Block size for N dimension quantization
        block_shape_k: Block size for K dimension quantization
        dtype: Output data type

    Returns:
        Tuple of (x, w_shuffled, y, x_scale_shuffled, w_scale, config)
    """
    scale_n = (N + block_shape_n - 1) // block_shape_n
    scale_k = (K + block_shape_k - 1) // block_shape_k

    # Generate input tensors in FP8 (E4M3 format)
    x = (torch.rand((M, K), dtype=torch.float16, device="cuda") / 10).to(e4m3_type)
    w = (torch.rand((N, K), dtype=torch.float16, device="cuda") / 10).to(e4m3_type)

    # Generate scale tensors
    x_scale = torch.rand([M, scale_k], dtype=torch.float32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=torch.float32, device="cuda")

    # Shuffle weight for preshuffle kernel
    weight_shuffle_layout = (16, 16)
    w_shuffled = shuffle_weight(w, weight_shuffle_layout).reshape(
        w.shape[0] // weight_shuffle_layout[0],
        w.shape[1] * weight_shuffle_layout[0],
    )

    # Transpose x_scale for preshuffle kernel
    x_scale_shuffled = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)

    # Output tensor
    y = torch.empty((M, N), dtype=dtype, device="cuda")

    # Base config that will be modified during tuning
    config = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
        "NUM_KSPLIT": 1,
        "waves_per_eu": 2,
        "kpack": 2,
        "matrix_instr_nonkdim": 16,
        "cache_modifier": "",
    }

    return x, w_shuffled, y, x_scale_shuffled, w_scale, config


def make_run_and_gt_fn_factory(
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Factory function to create run and ground truth functions for tuning.

    Args:
        M: Batch/sequence size
        N: Output feature dimension
        K: Input feature dimension
        dtype: Output data type

    Returns:
        Function that takes a config and returns (run_fn, ground_truth)
    """

    def make_run_and_gt(config: Dict[str, Any]) -> Tuple[Callable[[], Any], Any]:
        # Generate inputs including original and shuffled weights
        scale_n = (N + 128 - 1) // 128
        scale_k = (K + 128 - 1) // 128

        # Generate input tensors in FP8 (E4M3 format)
        x = (torch.rand((M, K), dtype=torch.float16, device="cuda") / 10).to(
            e4m3_type
        )
        w = (torch.rand((N, K), dtype=torch.float16, device="cuda") / 10).to(e4m3_type)

        # Generate scale tensors
        x_scale = torch.rand([M, scale_k], dtype=torch.float32, device="cuda")
        w_scale = torch.rand([scale_n, scale_k], dtype=torch.float32, device="cuda")

        # Shuffle weight for preshuffle kernel
        weight_shuffle_layout = (16, 16)
        w_shuffled = shuffle_weight(w, weight_shuffle_layout).reshape(
            w.shape[0] // weight_shuffle_layout[0],
            w.shape[1] * weight_shuffle_layout[0],
        )

        # Transpose x_scale for preshuffle kernel
        x_scale_shuffled = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)

        # Output tensor
        y = torch.empty((M, N), dtype=dtype, device="cuda")

        def run() -> torch.Tensor:
            return gemm_a8w8_blockscale_preshuffle(
                x,
                w_shuffled,
                x_scale_shuffled,
                w_scale,
                dtype,
                y,
                config,
                skip_reduce=False,
                is_x_scale_tranposed=True,
            )

        # Generate ground truth using PyTorch reference with original (non-shuffled) weights
        ground_truth = run_torch_reference(x, w, x_scale, w_scale, (128, 128), dtype)

        return run, ground_truth

    return make_run_and_gt


def tune_gemm_a8w8_blockscale_shuffle(
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    search_space: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Tune the GEMM A8W8 blockscale preshuffle kernel for specific dimensions.

    Args:
        M: Batch/sequence size
        N: Output feature dimension
        K: Input feature dimension
        dtype: Output data type
        search_space: List of configuration dictionaries to test

    Returns:
        Best configuration dictionary
    """
    make_run_and_gt = make_run_and_gt_fn_factory(M, N, K, dtype)

    best_config = tune_kernel(
        search_space=search_space,
        make_run_and_gt_fn=make_run_and_gt,
    )
    return best_config


def get_configs_compute_bound() -> List:
    """
    Generate configuration space for tuning the gemm_a8w8_blockscale_preshuffle kernel.

    For blockscale kernel with (128, 128) quantization blocks:
    - GROUP_K will be computed as 128 for K=1024
    - BLOCK_SIZE_K must equal GROUP_K, so BLOCK_SIZE_K must be 128

    Returns:
        List of configuration dictionaries
    """
    configs: List[Dict[str, Any]] = []

    # Fixed BLOCK_SIZE_K to match computed GROUP_K for (128, 128) quantization blocks
    block_k = 128

    # Explore optimized parameter space for blockscale kernel
    for num_stages in [1, 2, 3, 4]:
        for block_m in [32, 64, 128]:
            for block_n in [32, 64, 128]:
                for group_size_m in [1, 8, 16]:
                    for num_warps in [2, 4, 8]:
                        for num_ksplit in [1, 2, 4]:
                            for waves_per_eu in [2, 4, 8]:
                                for kpack in [2]:
                                    for cache_modifier in ["", ".cg"]:
                                        configs.append(
                                            {
                                                "BLOCK_SIZE_M": block_m,
                                                "BLOCK_SIZE_N": block_n,
                                                "BLOCK_SIZE_K": block_k,
                                                "GROUP_SIZE_M": group_size_m,
                                                "num_warps": num_warps,
                                                "num_stages": num_stages,
                                                "NUM_KSPLIT": num_ksplit,
                                                "waves_per_eu": waves_per_eu,
                                                "kpack": kpack,
                                                "matrix_instr_nonkdim": 16,
                                                "cache_modifier": cache_modifier,
                                            }
                                        )
    return configs


def tune_and_save_configs(
    batch_sizes: List[int],
    N: int,
    K: int,
    dtype: torch.dtype,
    search_space: List,
    save_path: str,
    device_name: str,
    tag: str,
):
    """
    Tune configurations for multiple batch sizes and save results.

    Args:
        batch_sizes: List of M dimensions (batch sizes) to test
        N: Output feature dimension
        K: Input feature dimension
        dtype: Output data type
        search_space: List of configuration dictionaries to test
        save_path: Path to save configuration files
        device_name: Device/architecture name for file naming
        tag: Tag for configuration file naming
    """
    start = time.time()
    benchmark_results = [
        tune_gemm_a8w8_blockscale_shuffle(
            batch_size,
            N,
            K,
            dtype=dtype,
            search_space=search_space,
        )
        for batch_size in tqdm(batch_sizes, desc=f"Tuning {tag}")
    ]

    # Categorize configs by batch size ranges following the sample pattern
    best_configs = {}
    for i, (M, config) in enumerate(zip(batch_sizes, benchmark_results)):
        if i == len(batch_sizes) - 1:
            best_configs["any"] = config
        elif M < 32:
            best_configs["small"] = config
        elif M <= 128:
            BLK_M = triton.next_power_of_2(M)
            if BLK_M == 32:
                best_configs["medium_M32"] = config
            elif BLK_M == 64:
                best_configs["medium_M64"] = config
            elif BLK_M == 128:
                best_configs["medium_M128"] = config
        elif M <= 256:
            best_configs["large"] = config
        else:
            best_configs["xlarge"] = config

    json_file_name = f"{device_name}-GEMM-{tag}.json"
    save_configs_to_json(json_file_name, save_path, best_configs)

    end = time.time()
    print(f"Tuning for {tag} took {end - start:.2f} seconds")

    return best_configs


def create_default_config(
    all_configs: List,
    save_path: str,
    device_name: str,
    tag: str = "A8W8_BLOCKSCALE_PRESHUFFLED",
) -> dict:
    """
    Create a default config by selecting the most common config across all shapes.
    
    This function loads existing configs from save_dir that match the tag pattern,
    combining them with the current run's configs to create a comprehensive default.

    Args:
        all_configs: List of best_configs dictionaries from different (N, K) shapes
        save_path: Path to the directory where config files are saved
        device_name: Device/architecture name for file matching
        tag: Tag prefix to match config files (e.g., "A8W8_BLOCKSCALE_PRESHUFFLED")

    Returns:
        Default configuration dictionary with most common configs for each category
    """
    import json
    import glob
    from collections import Counter

    # Collect all configs for each category following the sample pattern
    category_configs = {
        "small": [],
        "medium_M32": [],
        "medium_M64": [],
        "medium_M128": [],
        "large": [],
        "xlarge": [],
        "any": [],
    }

    # Load existing configs from save_dir that match the tag pattern
    # Pattern: {device_name}-GEMM-{tag}*.json
    pattern = os.path.join(save_path, f"{device_name}-GEMM-{tag}*.json")
    existing_config_files = glob.glob(pattern)
    
    print(f"Found {len(existing_config_files)} existing config files matching pattern: {pattern}")
    
    for config_file in existing_config_files:
        try:
            with open(config_file, 'r') as f:
                existing_config = json.load(f)
                # Add existing configs to category_configs
                for category, params in existing_config.items():
                    if category in category_configs:
                        config_tuple = tuple(sorted(params.items()))
                        category_configs[category].append(config_tuple)
                        print(f"  Loaded {category} config from {os.path.basename(config_file)}")
        except Exception as e:
            print(f"Warning: Failed to load config from {config_file}: {e}")

    # Add configs from current run
    for config in all_configs:
        for category, params in config.items():
            if category in category_configs:
                # Convert config to a hashable tuple for counting
                config_tuple = tuple(sorted(params.items()))
                category_configs[category].append(config_tuple)

    # Find the most common config for each category
    default_config = {}
    for category, configs in category_configs.items():
        if configs:
            most_common = Counter(configs).most_common(1)[0][0]
            default_config[category] = dict(most_common)
            print(f"Selected most common {category} config from {len(configs)} total configs")

    return default_config


def save_default_config(
    config: dict,
    save_path: str,
    device_name: str,
    tag: str = "A8W8_BLOCKSCALE_PRESHUFFLED",
) -> None:
    """
    Save the default config file (without N,K parameters).

    Args:
        config: Default configuration dictionary
        save_path: Path to save configuration files
        device_name: Device/architecture name for file naming
        tag: Tag for configuration file naming
    """
    import json

    os.makedirs(save_path, exist_ok=True)
    json_file_name = f"{device_name}-GEMM-{tag}.json"

    config_file_path = os.path.join(save_path, json_file_name)
    print(f"Writing default config to {config_file_path}...")

    with open(config_file_path, "w") as f:
        json.dump(config, f, indent=4)
        f.write("\n")


def main():
    """Main tuning entry point."""
    dev = arch_info.get_arch()

    torch.cuda.init()

    # Test configurations based on common LLM dimensions
    # Note: For preshuffle kernel, N must be multiple of 16 and K must be multiple of 32
    test_configs = [
        # (batch_sizes, N, K)
        # (
        #     [
        #         16,
        #         32,
        #         64,
        #         128,
        #         256,
        #         512,
        #         2048,
        #         4096,
        #     ],
        #     1024,
        #     1024,
        # ),  # Standard attention head dims
        # (
        #     [
        #         16,
        #         32,
        #         64,
        #         128,
        #         256,
        #         512,
        #         2048,
        #         4096,
        #     ],
        #     4096,
        #     1024,
        # ),  # FFN intermediate
        # (
        #     [
        #         16,
        #         32,
        #         64,
        #         128,
        #         256,
        #         512,
        #         2048,
        #         4096,
        #     ],
        #     1024,
        #     2048,
        # ),  # Wider input
        # (
        #     [
        #         16,
        #         32,
        #         64,
        #         128,
        #         256,
        #         512,
        #         2048,
        #         4096,
        #     ],
        #     6144,
        #     1024,
        # ),  # Larger FFN
        (
            [
                16,
                32,
                64,
                128,
                256,
                512,
                2048,
                4096,
            ],
            1024,
            3072,
        ),  # Deeper input
    ]

    search_space = get_configs_compute_bound()
    save_path = AITER_TRITON_CONFIGS_PATH + "/gemm/"

    print(f"Architecture: {dev}")
    print(f"Search space size: {len(search_space)} configurations")
    print(f"Total test configurations: {len(test_configs)}")
    print()

    # Collect all configs to determine best overall config
    all_configs: List[Dict[str, Any]] = []

    # Tune for each configuration
    for batch_sizes, N, K in test_configs:
        tag = f"A8W8_BLOCKSCALE_PRESHUFFLED-N={N}-K={K}"
        print(f"Running {tag}...")
        best_configs = tune_and_save_configs(
            batch_sizes=batch_sizes,
            N=N,
            K=K,
            dtype=torch.bfloat16,
            search_space=search_space,
            save_path=save_path,
            device_name=dev,
            tag=tag,
        )
        # Store configs for later analysis
        all_configs.append(best_configs)

    # Create a default config file (without N,K parameters) by selecting most common config
    default_config = create_default_config(all_configs, save_path, dev, "A8W8_BLOCKSCALE_PRESHUFFLED")
    save_default_config(
        default_config, save_path, dev, "A8W8_BLOCKSCALE_PRESHUFFLED"
    )


if __name__ == "__main__":
    main()
