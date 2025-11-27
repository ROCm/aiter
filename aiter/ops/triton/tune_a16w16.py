import argparse
import json
import multiprocessing as mp
import os
import time
import triton
from datetime import datetime

import torch
from tqdm import tqdm


from gemm_a16w16 import gemm_a16w16  # type: ignore
from utils.core import AITER_TRITON_CONFIGS_PATH  # type: ignore

mp.set_start_method("spawn", force=True)


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "half": torch.half,
    "bfloat16": torch.bfloat16,
}


def get_configs_compute_bound():
    configs = []
    for num_stages in [2]:
        for block_m in [16]:
            for block_k in [64]:
                for block_n in [32]:
                    for num_warps in [4]:
                        for group_size in [1]:
                            for waves_per_eu in [3]:
                                configs.append(
                                    {
                                        "BLOCK_SIZE_M": block_m,
                                        "BLOCK_SIZE_N": block_n,
                                        "BLOCK_SIZE_K": block_k,
                                        "GROUP_SIZE_M": group_size,
                                        "num_warps": num_warps,
                                        "num_stages": num_stages,
                                        "waves_per_eu": waves_per_eu,  # TODO check if compatible
                                        "matrix_instr_nonkdim": 16,  # TODO
                                        "cache_modifier": None,  # TODO
                                        "NUM_KSPLIT": 1,  # TODO
                                        "kpack": 1,  # TODO
                                        "SPLITK_BLOCK_SIZE": 1,
                                    }
                                )
    return configs


# def get_configs_compute_bound():
#     configs = []
#     for num_stages in [2, 3, 4, 5]:
#         for block_m in [16, 32, 64, 128, 256]:
#             for block_k in [64, 128]:
#                 for block_n in [32, 64, 128, 256]:
#                     for num_warps in [4, 8]:
#                         for group_size in [1, 16, 32, 64]:
#                             for waves_per_eu in [1,2,3,4]:
#                                 configs.append(
#                                     {
#                                         "BLOCK_SIZE_M": block_m,
#                                         "BLOCK_SIZE_N": block_n,
#                                         "BLOCK_SIZE_K": block_k,
#                                         "GROUP_SIZE_M": group_size,
#                                         "num_warps": num_warps,
#                                         "num_stages": num_stages,
#                                         "waves_per_eu": waves_per_eu, # TODO check if compatible
#                                         "matrix_instr_nonkdim": 16, # TODO
#                                         "cache_modifier": None, # TODO
#                                         "NUM_KSPLIT": 1, # TODO
#                                         "kpack": 1, # TODO
#                                         "SPLITK_BLOCK_SIZE":1,
#                                     }
#                                 )
#     return configs


def get_weight_shapes(tp_size):
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
    def run():
        gemm_a16w16(x, w, bias, dtype, y, config, activation)

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
        run()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    return avg


def tune(M, N, K, out_dtype, search_space, input_type):
    if input_type == "bfloat16":
        fp16_info = torch.finfo(torch.bfloat16)
        fp16_max, fp16_min = fp16_info.max, fp16_info.min

        x_fp32 = (
            (torch.rand(M, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp16_max
        )
        x = x_fp32.clamp(min=fp16_min, max=fp16_max).to(torch.bfloat16)

        w_fp32 = (
            (torch.rand(N, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp16_max
        )
        w = w_fp32.clamp(min=fp16_min, max=fp16_max).to(torch.bfloat16)
    else:
        raise RuntimeError("Currently, only support tune w16a16 block fp16 kernel.")

    best_config = None
    best_time = float("inf")
    for config in tqdm(search_space):
        try:
            kernel_time = benchmark_config(
                x=x,
                w=w,
                bias=None,
                dtype=torch.bfloat16,
                y=None,
                config=config,
                activation=None,
                num_iters=10,
            )
        except triton.runtime.autotuner.OutOfResources:
            # Some configurations may be invalid and fail to compile.
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

    out_dtype = DTYPE_MAP[args.out_dtype]
    save_path = AITER_TRITON_CONFIGS_PATH + "/gemm/"
    input_type = args.input_type

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
                out_dtype,
                search_space,
                input_type,
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


def distribute_batch_sizes(batch_sizes, num_gpus):
    """Distribute batch sizes across available GPUs."""
    batches_per_gpu = []
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
            64,
            128,
            256,
            512,
            2048,
            4096,
        ]
    else:
        batch_sizes = [args.batch_size]
        num_gpus = 1  # If only one batch size, use only one GPU

    weight_shapes = get_weight_shapes(args.tp_size)

    batches_per_gpu = distribute_batch_sizes(batch_sizes, 1)

    process_args = []
    for gpu_id in range(1):
        process_args.append(
            {
                "gpu_id": gpu_id,
                "batch_sizes": batches_per_gpu[gpu_id],
                "weight_shapes": weight_shapes,  # Each GPU processes all weight shapes
                "args": args,
            }
        )

    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:
        pool.map(tune_on_gpu, process_args)

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
