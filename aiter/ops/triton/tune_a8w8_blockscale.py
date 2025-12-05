import argparse
import json
import multiprocessing as mp
import os
import time
import triton
from datetime import datetime
from typing import List, Dict, Union, Tuple, Optional
import torch
from tqdm import tqdm


from gemm_a8w8_blockscale import gemm_a8w8_blockscale  # type: ignore
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH  # type: ignore
from aiter.ops.triton.utils.types import get_fp8_dtypes

mp.set_start_method("spawn", force=True)

# Get FP8 data types
e5m2_type, e4m3_type = get_fp8_dtypes()


def generate_gemm_a8w8_blockscale_inputs(M, N, K, dtype, block_shape=(128, 128), layout="TN", output=True, bias=False):
    """
    Generate inputs for gemm_a8w8_blockscale kernel.
    
    Args:
        M, N, K: Matrix dimensions
        dtype: Output data type
        block_shape: Tuple of (block_shape_n, block_shape_k) for block scaling
        layout: Input matrix layout
        output: Whether to generate output tensor
        bias: Whether to generate bias tensor
        
    Returns:
        Tuple of (x, w, x_scale, w_scale, bias, y)
    """
    block_shape_n, block_shape_k = block_shape
    scale_n = (N + block_shape_n - 1) // block_shape_n
    scale_k = (K + block_shape_k - 1) // block_shape_k

    # Generate input matrix x (M, K)
    if layout[0] == "T":
        x = (torch.rand((M, K), dtype=torch.float16, device="cuda") / 10).to(e4m3_type)
    else:
        x = ((torch.rand((K, M), dtype=torch.float16, device="cuda") / 10).to(e4m3_type)).T

    # Generate weight matrix w (N, K)
    if layout[1] == "N":
        w = (torch.rand((N, K), dtype=torch.float16, device="cuda") / 10).to(e4m3_type)
    else:
        w = ((torch.rand((K, N), dtype=torch.float16, device="cuda") / 10).to(e4m3_type)).T

    # Generate scale tensors
    x_scale = torch.rand([M, scale_k], dtype=torch.float32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=torch.float32, device="cuda")

    # Generate bias tensor if needed
    bias_tensor = None
    if bias:
        bias_tensor = torch.empty((N), dtype=dtype, device="cuda")

    # Generate output tensor if needed
    y = None
    if output:
        y = torch.empty((M, N), dtype=dtype, device="cuda")

    return x, w, x_scale, w_scale, bias_tensor, y


def get_configs_compute_bound() -> List[Dict[str, int | str]]:
    """
    Generate configuration space for tuning the gemm_a8w8_blockscale kernel.
    Based on the sample config file, we'll tune around those values.
    Note: GROUP_K must equal BLOCK_SIZE_K as required by the kernel.
    """
    configs = []
    # Based on the sample config from MI300X-GEMM-A8W8_BLOCKSCALE.json
    # We'll explore a reasonable range around these values
    for num_stages in [1, 2]:  # Sample config uses 2
        for block_m in [64, 128, 256]:  # Sample config uses 128
            for block_k in [64, 128, 256]:  # Sample config uses 128
                for block_n in [64, 128, 256]:  # Sample config uses 128
                    for group_size in [1, 8]:  # Sample config uses 1
                        for num_warps in [4, 8]:  # Sample config uses 4
                            for num_ksplit in [1, 2, 4]:  # Sample config uses 1
                                for waves_per_eu in [1, 2, 4]:  # Sample config uses 2
                                    for kpack in [1,2 ]:  # Sample config uses 2
                                        for cache_modifier in ["", ".cg"]:  # Sample config uses ".cg"
                                            configs.append(
                                                {
                                                    "BLOCK_SIZE_M": block_m,
                                                    "BLOCK_SIZE_N": block_n,
                                                    "BLOCK_SIZE_K": block_k,
                                                    "GROUP_SIZE_M": group_size,
                                                    "num_warps": num_warps,
                                                    "num_stages": num_stages,
                                                    "waves_per_eu": waves_per_eu,
                                                    "matrix_instr_nonkdim": 16,  # Fixed value used in kernel
                                                    "cache_modifier": cache_modifier,
                                                    "NUM_KSPLIT": num_ksplit,
                                                    "kpack": kpack,
                                                    # "SPLITK_BLOCK_SIZE": 1,  # Will be set dynamically
                                                }
                                            )
    return configs


def get_weight_shapes(tp_size: int) -> List[Tuple[int, int]]:
    """Get weight shapes to test during tuning."""
    total = [
        (1024, 1024),
        (4096, 1024),
        (1024, 2048),
        (6144, 1024),
        (1024, 3072),
    ]

    weight_shapes: List[Tuple[int, int]] = []
    for t in total:
        weight_shapes.append(t)

    return weight_shapes


def run_torch_reference(x, w, x_scale, w_scale, bias, dtype=torch.bfloat16, block_shape=(128, 128)):
    """
    Run reference implementation using PyTorch.
    This is used for correctness verification.
    Based on the test file implementation.
    """
    block_shape_n, block_shape_k = block_shape
    m, k = x.shape
    n = w.shape[0]
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k
    
    # Expand scales to match the full matrix dimensions
    x_scale_expanded = x_scale.repeat_interleave(block_shape_k, dim=1)
    x_scaled = x.to(x_scale_expanded.dtype) * x_scale_expanded[:m, :k]
    x_scaled = x_scaled.view(m, k)
    
    w_scale_expanded = w_scale.repeat_interleave(block_shape_n, dim=0)
    w_scale_expanded = w_scale_expanded.repeat_interleave(block_shape_k, dim=1)
    w_scale_expanded = w_scale_expanded[:n, :k]
    w_scaled = w.to(w_scale_expanded.dtype) * w_scale_expanded

    # Compute the matrix multiplication with bias if provided
    # Convert bias to float32 if it's not None to match the other tensors
    bias_float32 = bias.to(torch.float32) if bias is not None else None
    out = torch.nn.functional.linear(x_scaled.to(torch.float32), w_scaled.to(torch.float32), bias=bias_float32)
    
    return out.to(dtype)


def benchmark_config(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    dtype: torch.dtype,
    config: Dict[str, Union[str, int]],
    y: Optional[torch.Tensor] = None,
    num_iters=10,
) -> float:
    """
    Benchmark the performance of a GEMM operation with a specific configuration.
    
    This function measures the execution time of the gemm_a8w8_blockscale kernel by running
    it multiple times with synchronization points to ensure accurate timing. It performs
    warmup runs before the actual benchmarking to account for JIT compilation overhead.
    
    Args:
        x (torch.Tensor): Input tensor of shape (M, K) representing the first matrix operand.
        w (torch.Tensor): Weight tensor of shape (N, K) representing the second matrix operand.
        x_scale (torch.Tensor): Scale tensor for x with shape (M, scale_k).
        w_scale (torch.Tensor): Scale tensor for w with shape (scale_n, scale_k).
        dtype (torch.dtype): Data type for the computation (e.g., torch.bfloat16).
        config (Dict[str, Union[str, int]]): Configuration dictionary containing kernel
            parameters such as block sizes, number of warps, etc.
        y (Optional[torch.Tensor], optional): Output tensor to store the result. If None,
            a new tensor will be allocated. Defaults to None.
        num_iters (int, optional): Number of benchmark iterations to run. Defaults to 10.
    
    Returns:
        float: Average execution time in microseconds (us) per iteration.
    
    Note:
        The function performs 5 warmup iterations before benchmarking to account for
        JIT compilation and GPU warmup effects. The timing is measured using CUDA events
        for accurate GPU kernel timing.
    """
    # Calculate GROUP_K based on the input dimensions and w_scale shape
    # This must match BLOCK_SIZE_K to satisfy the kernel assertion
    M, K = x.shape
    N, _ = w.shape
    w_scale_T = w_scale.T  # Transpose to match kernel's expectation
    group_k = triton.next_power_of_2(triton.cdiv(K, w_scale_T.shape[0]))
    
    # Create a copy of config to modify
    modified_config = config.copy()
    # Set BLOCK_SIZE_K to match GROUP_K to satisfy kernel assertion
    modified_config["BLOCK_SIZE_K"] = group_k
    
    # Get reference output for correctness verification
    torch_out = run_torch_reference(x, w, x_scale, w_scale, bias, dtype)

    # Run kernel
    def run():
        # Pass the modified config to the kernel
        return gemm_a8w8_blockscale(x, w, x_scale, w_scale, dtype, y, modified_config, skip_reduce=False)

    torch.cuda.synchronize()
    # JIT compilation & warmup
    for _ in range(5):
        run()
    torch.cuda.synchronize()

    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)

    latencies: list[float] = []
    for i in range(num_iters):
        torch.cuda.synchronize()
        start_event.record()
        triton_out_raw = run()
        # Convert to the same dtype as the reference for comparison
        # Handle the case where triton_out_raw might be None
        if triton_out_raw is not None:
            triton_out = triton_out_raw.to(torch_out.dtype)
        else:
            triton_out = torch_out  # Fallback to reference output
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
        torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)
    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    return avg


def tune(
    M: int, N: int, K: int, search_space: List[Dict[str, int | str]], input_type: str
):
    """Tune the kernel for specific matrix dimensions."""
    if input_type == "bfloat16":
        # Use the same input generation as test file
        x, w, x_scale, w_scale, bias, y = generate_gemm_a8w8_blockscale_inputs(
            M, N, K, torch.bfloat16, bias=True
        )
    else:
        raise RuntimeError("Currently, only support tune a8w8 blockscale kernel with bfloat16 output.")

    best_config = None
    best_time = float("inf")
    for config in tqdm(search_space):
        try:
            kernel_time = benchmark_config(
                x=x,
                w=w,
                x_scale=x_scale,
                w_scale=w_scale,
                bias=bias,
                dtype=torch.bfloat16,
                y=None,
                config=config,
                num_iters=10,
            )
        except triton.runtime.autotuner.OutOfResources as e:
            # Some configurations may be invalid and fail to compile.
            continue
        except AssertionError as e:
            print("Assert error:", e)
            continue

        if kernel_time < best_time:
            best_time = kernel_time
            best_config = config
    now = datetime.now()
    print(f"{now.ctime()}] Completed tuning for batch_size={M}")
    assert best_config is not None
    return best_config


def save_configs(
    N,
    K,
    configs,
    save_path,
) -> None:
    """Save the best configurations to a JSON file."""
    os.makedirs(save_path, exist_ok=True)
    device_name = "MI300X"  # TODO: Hardcoded, make it dynamic
    json_file_name = f"{device_name}-GEMM-A8W8_BLOCKSCALE-N={N}-K={K}.json"

    config_file_path = os.path.join(save_path, json_file_name)
    print(f"Writing best config to {config_file_path}...")

    with open(config_file_path, "w") as f:
        json.dump(configs, f, indent=4)
        f.write("\n")


def tune_on_gpu(
    gpu_id: int,
    batch_sizes: List[int],
    weight_shapes: List[Tuple[int, int]],
    input_type: str,
) -> None:
    """Run tuning on a specific GPU."""
    torch.cuda.set_device(gpu_id)
    print(f"Starting tuning on GPU {gpu_id} with batch sizes {batch_sizes}")

    save_path = AITER_TRITON_CONFIGS_PATH + "/gemm/"

    search_space = get_configs_compute_bound()

    start = time.time()

    # Collect all configs to determine the best overall config
    all_configs: List[Dict[str, Dict[str, int | str]]] = []

    for shape in tqdm(weight_shapes, desc=f"GPU {gpu_id} - Shapes"):
        N, K = shape[0], shape[1]
        print(f"[GPU {gpu_id}] Tune for weight shape of `N: {N}, K: {K}`")
        benchmark_results = [
            tune(
                batch_size,
                N,
                K,
                search_space,
                input_type,
            )
            for batch_size in tqdm(batch_sizes, desc=f"GPU {gpu_id} - Batch sizes")
        ]
        best_configs: Dict[str, Dict[str, int | str]] = {}
        # Create configs for different M size categories as expected by the kernel
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
        # Store configs for later analysis
        all_configs.append(best_configs)
        save_configs(N, K, best_configs, save_path)

    # Create a default config file (without N,K parameters) by selecting the most common config
    default_config = create_default_config(all_configs)
    save_default_config(default_config, save_path)

    end = time.time()
    print(f"Tuning on GPU {gpu_id} took {end - start:.2f} seconds")


def create_default_config(
    all_configs: List[Dict[str, Dict[str, Union[int, str]]]],
) -> Dict[str, Dict[str, Union[int, str]]]:
    """Create a default config by selecting the most common config across all shapes."""
    from collections import Counter

    # Collect all configs for each category
    category_configs = {
        "small": [],
        "medium_M32": [],
        "medium_M64": [],
        "medium_M128": [],
        "large": [],
        "xlarge": [],
        "any": [],
    }

    for config in all_configs:
        for category, params in config.items():
            if category in category_configs:
                # Convert config to a hashable tuple for counting
                config_tuple = tuple(sorted(params.items()))
                category_configs[category].append(config_tuple)

    # Find the most common config for each category
    default_config: Dict[str, Dict[str, Union[int, str]]] = {}
    for category, configs in category_configs.items():
        if configs:
            most_common = Counter(configs).most_common(1)[0][0]
            default_config[category] = dict(most_common)

    return default_config


def save_default_config(
    config: Dict[str, Dict[str, Union[int, str]]], save_path: str
) -> None:
    """Save the default config file (without N,K parameters)."""
    os.makedirs(save_path, exist_ok=True)
    device_name = "MI300X"  # TODO: Hardcoded, make it dynamic
    json_file_name = f"{device_name}-GEMM-A8W8_BLOCKSCALE.json"

    config_file_path = os.path.join(save_path, json_file_name)
    print(f"Writing default config to {config_file_path}...")

    with open(config_file_path, "w") as f:
        json.dump(config, f, indent=4)
        f.write("\n")


def distribute_batch_sizes(batch_sizes: List[int], num_gpus: int) -> List[List[int]]:
    """Distribute batch sizes across available GPUs."""
    batches_per_gpu: List[List[int]] = []
    for i in range(num_gpus):
        start_idx = i * len(batch_sizes) // num_gpus
        end_idx = (i + 1) * len(batch_sizes) // num_gpus
        batches_per_gpu.append(batch_sizes[start_idx:end_idx])
    return batches_per_gpu


def main(args):
    print(args)
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPU available for tuning")
    print(f"Found {num_gpus} GPUs for parallel tuning")

    torch.cuda.init()

    if args.batch_size is None:
        batch_sizes = [
            16,  # For small config
            32,  # For medium_M32 config
            64,  # For medium_M64 config
            128,  # For medium_M128 config
            256,  # For large config
            512,  # For large config
            2048,  # For xlarge config
            4096,  # For xlarge config
        ]
    else:
        batch_sizes = [args.batch_size]
        num_gpus = 1  # If only one batch size, use only one GPU

    weight_shapes = get_weight_shapes(args.tp_size)

    batches_per_gpu = distribute_batch_sizes(batch_sizes, 1)

    # Prepare arguments for each GPU process
    process_args = []
    for gpu_id in range(1):
        process_args.append(
            (
                gpu_id,
                batches_per_gpu[gpu_id],
                weight_shapes,  # Each GPU processes all weight shapes
                args.input_type,
            )
        )

    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:
        pool.starmap(tune_on_gpu, process_args)

    print("Multi-GPU tuning completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--tp-size", "-tp", type=int, default=1)
    parser.add_argument(
        "--input-type", type=str, choices=["bfloat16"], default="bfloat16"
    )
    parser.add_argument(
        "--out-dtype",
        type=str,
        choices=["float32", "float16", "bfloat16", "half"],
        default="bfloat16",
    )
    parser.add_argument("--batch-size", type=int, required=False)
    args = parser.parse_args()

    main(args)
