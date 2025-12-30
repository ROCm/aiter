import argparse
import json
import os
import time
import triton
from datetime import datetime

import torch
from tqdm import tqdm

from aiter.ops.triton.gemm_a16w16 import gemm_a16w16

from op_tests.triton_tests.utils.types import str_to_torch_dtype

from utils.core import AITER_TRITON_CONFIGS_PATH  # type: ignore


def generate_gemm_a16w16_inputs(M, N, K, dtype, layout="TN", output=True, bias=False):
    if isinstance(dtype, str):
        dtype = str_to_torch_dtype[dtype]

    # TN is default layout
    if layout[0] == "T":
        print(M, K)
        print(dtype)
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


def get_configs_compute_bound():
    configs = []
    for num_stages in [2, 3, 4, 5]:
        for block_m in [16, 32, 64, 128, 256]:
            for block_k in [64, 128]:
                for block_n in [32, 64, 128, 256]:
                    for num_warps in [4, 8]:
                        for group_size in [1, 16, 32, 64]:
                            for waves_per_eu in [2, 3, 4]:
                                configs.append(
                                    {
                                        "BLOCK_SIZE_M": block_m,
                                        "BLOCK_SIZE_N": block_n,
                                        "BLOCK_SIZE_K": block_k,
                                        "GROUP_SIZE_M": group_size,
                                        "num_warps": num_warps,
                                        "num_stages": num_stages,
                                        "waves_per_eu": waves_per_eu,
                                        "matrix_instr_nonkdim": 16,
                                        "NUM_KSPLIT": 1,
                                        "kpack": 1,
                                        "cache_modifier": None, 
                                    }
                                )
    return configs


def get_weight_shapes():
    total = [
        (1024, 1024),
        (4096, 1024),
        (1024, 2048),
        (6144, 1024),
        (1024, 3072),
    ]

    weight_shapes = []
    for t in total:
        weight_shapes.append(t)

    return weight_shapes


def benchmark_config(x, w, bias, dtype, y, config, activation, num_iters=10):
    torch_out = torch.nn.functional.linear(x, w, bias=bias) # Ground truth

    def run():
        return gemm_a16w16(
            x, w, bias=bias, dtype=dtype, y=y, config=config, activation=None
        )

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
        triton_out = run()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
        torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)
    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    return avg


def tune(M, N, K, dtype, search_space):
    x, w, bias, out_dtype, y = generate_gemm_a16w16_inputs(
        M, N, K, dtype, output=True, bias=True
    )

    best_config = None
    best_time = float("inf")
    for config in tqdm(search_space):
        config["SPLITK_BLOCK_SIZE"]= triton.cdiv(K, config["NUM_KSPLIT"])
        try:
            kernel_time = benchmark_config(
                x=x,
                w=w,
                bias=bias,
                dtype=out_dtype,
                y=y,
                config=config,
                activation=None,
                num_iters=10,
            )
        except triton.runtime.autotuner.OutOfResources:
            print("OutOfResources encountered during tuning.")
            # Some configurations may be invalid and fail to compile.
            continue
        except AssertionError:
            print("AssertionError encountered during tuning.")
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
    json_file_name = f"{device_name}-GEMM-A16W16-N={N}-K={K}.json"

    config_file_path = os.path.join(save_path, json_file_name)
    print(f"Writing best config to {config_file_path}...")

    with open(config_file_path, "w") as f:
        json.dump(configs, f, indent=4)
        f.write("\n")


def tune_on_gpu(args_dict):
    """Run tuning on a specific GPU."""
    gpu_id = args_dict["gpu_id"]
    batch_sizes = args_dict["batch_sizes"]
    weight_shapes = args_dict["weight_shapes"]
    args = args_dict["args"]

    torch.cuda.set_device(gpu_id)
    print(f"Starting tuning on GPU {gpu_id} with batch sizes {batch_sizes}")

    dtype = str_to_torch_dtype[args.dtype]
    save_path = AITER_TRITON_CONFIGS_PATH + "/gemm/"

    search_space = get_configs_compute_bound()

    start = time.time()
    for shape in tqdm(weight_shapes, desc=f"GPU {gpu_id} - Shapes"):
        N, K = shape[0], shape[1]
        print(f"[GPU {gpu_id}] Tune for weight shape of `N: {N}, K: {K}`")
        benchmark_results = [
            tune(
                batch_size,
                N,
                K,
                dtype,
                search_space,
            )
            for batch_size in tqdm(batch_sizes, desc=f"GPU {gpu_id} - Batch sizes")
        ]
        best_configs = {
            ("any" if i == len(batch_sizes) - 1 else f"M_LEQ_{M}"): config
            for i, (M, config) in enumerate(zip(batch_sizes, benchmark_results))
        }
        save_configs(N, K, best_configs, save_path)

    end = time.time()
    print(f"Tuning on GPU {gpu_id} took {end - start:.2f} seconds")


def main(args):
    print(args)
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPU available for tuning")
    print(f"Found {num_gpus} GPUs for tuning")

    torch.cuda.init()

    batch_sizes = [
        64,
        128,
        256,
        512,
        2048,
        4096,
    ]

    weight_shapes = get_weight_shapes()

    # Run tuning sequentially on GPU 0
    tune_on_gpu(
        {
            "gpu_id": 0,
            "batch_sizes": batch_sizes,
            "weight_shapes": weight_shapes,
            "args": args,
        }
    )

    print("Tuning completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16", "half"],
        default="bfloat16",
    )
    args = parser.parse_args()

    main(args)
