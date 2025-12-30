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


from gemm_a16w16_atomic import gemm_a16w16_atomic  # type: ignore
from utils.core import AITER_TRITON_CONFIGS_PATH  # type: ignore

mp.set_start_method("spawn", force=True)


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "half": torch.half,
    "bfloat16": torch.bfloat16,
}


def generate_gemm_a16w16_inputs(M, N, K, dtype, layout="TN", output=True, bias=False):
    if isinstance(dtype, str):
        dtype = DTYPE_MAP[dtype]

    # TN is default layout
    if layout[0] == "T":
        x = torch.randn((M, K), dtype=dtype, device="cuda")
    else:
        x = torch.randn((K, M), dtype=dtype, device="cuda").T

    if layout[1] == "T":
        weight = torch.randn((K, N), dtype=dtype, device="cuda").T
    else:
        weight = torch.randn((N, K), dtype=dtype, device="cuda")

    bias_tensor = None
    if bias:
        bias_tensor = torch.empty((N), dtype=dtype, device="cuda")

    y = None
    if output:
        y = torch.empty((M, N), dtype=dtype, device="cuda")
        out_dtype = (None,)
    else:
        out_dtype = dtype

    return x, weight, bias_tensor, out_dtype, y


def get_configs_compute_bound() -> List[Dict[str, int | str]]:
    configs = []
    # Optimize parameters based on kernel analysis
    # Only test parameters that are actually used in the kernel
    # Based on the generated configs, we'll use the optimal values
    for num_stages in [2, 3, 4, 5]:  # Only 1 stage is used
        for block_m in [16, 32, 64, 128, 256]:  # Fixed to 64 as in current configs
            for block_n in [32, 64, 128, 256]:  # Fixed to 32 as in current configs
                for block_k in [64, 128]:  # Fixed to 64 as in current configs
                    for group_size in [1, 8, 16, 32, 64]:
                        for num_warps in [4, 8]:
                            for num_ksplit in [
                                1,
                            ]:  # Only test 1 since higher values may cause issues
                                for waves_per_eu in [
                                    2,
                                    4,
                                    8,
                                ]:  # Fixed to 3 as in current configs
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
                                            "cache_modifier": "",  # Empty string for atomic kernel
                                            "NUM_KSPLIT": num_ksplit,  # Atomic kernel specific
                                            # "kpack": 1,  # Fixed value used in kernel
                                            # "SPLITK_BLOCK_SIZE": 1,  # Will be set dynamically
                                        }
                                    )
    return configs


def get_weight_shapes(tp_size: int) -> List[Tuple[int, int]]:
    total = [
        # (1024, 1024),
        # (4096, 1024),
        # (1024, 2048),
        (6144, 1024),
        (1024, 3072),
    ]

    weight_shapes: List[Tuple[int, int]] = []
    for t in total:
        weight_shapes.append(t)

    return weight_shapes


def benchmark_config(
    x: torch.Tensor,
    w: torch.Tensor,
    dtype: torch.dtype,
    config: Dict[str, Union[str, int]],
    y: Optional[torch.Tensor] = None,
    num_iters=10,
) -> float:
    """
    Benchmark the performance of a GEMM operation with a specific configuration.

    This function measures the execution time of the gemm_a16w16_atomic kernel by running
    it multiple times with synchronization points to ensure accurate timing. It performs
    warmup runs before the actual benchmarking to account for JIT compilation overhead.

    Args:
        x (torch.Tensor): Input tensor of shape (M, K) representing the first matrix operand.
        w (torch.Tensor): Weight tensor of shape (N, K) representing the second matrix operand.
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
    torch_out = torch.nn.functional.linear(x, w, bias=None)

    # run the kernel
    def run():
        return gemm_a16w16_atomic(x, w, dtype, y, config)

    torch.cuda.synchronize()
    # JIT complication & warmup
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
        triton_out = triton_out_raw.to(torch_out.dtype)
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
        torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)
    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    return avg


def tune(
    M: int, N: int, K: int, search_space: List[Dict[str, int | str]], input_type: str
):
    if input_type == "bfloat16":
        # Use the same input generation as test file
        x, weight, x_scale, w_scale, y = generate_gemm_a16w16_inputs(
            M, N, K, torch.bfloat16, output=True
        )
    else:
        raise RuntimeError("Currently, only support tune w16a16 block fp16 kernel.")

    best_config = None
    best_time = float("inf")
    for config in tqdm(search_space):
        try:
            kernel_time = benchmark_config(
                x=x,
                w=weight,
                dtype=torch.float32,
                y=None,
                config=config,
                num_iters=10,
            )
        except triton.runtime.autotuner.OutOfResources as e:
            # print("config failed!", config)
            # print("error: ", e)
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
    os.makedirs(save_path, exist_ok=True)
    device_name = "R9700"  # TODO: Hardcoded, make it dynamic
    json_file_name = f"{device_name}-GEMM-A16W16-ATOMIC-N={N}-K={K}.json"

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

    # Collect all configs to determine best overall config
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
        # Create configs for different M size categories as expected by the atomic kernel
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

    # Create a default config file (without N,K parameters) by selecting most common config
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
    device_name = "R9700"  # TODO: Hardcoded, make it dynamic
    json_file_name = f"{device_name}-GEMM-A16W16-ATOMIC.json"

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
#
