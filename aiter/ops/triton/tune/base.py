import json
import os
import triton
import torch
import tqdm

def tune_kernel(
    search_space,
    make_run_and_gt_fn,
    config_callback=None,
    num_iters=10,
    atol=1e-2,
    rtol=1e-2,
):
    """
    Args:
        search_space: List of config dicts.
        make_run_and_gt_fn: Callable(config) -> (run_fn, ground_truth)
        config_callback: Optional function to modify config before use.
        num_iters: Number of iterations for benchmarking.
        atol, rtol: Tolerances for output comparison.
    Returns:
        The best config dict.
    """
    best_config = None
    best_time = float("inf")
    for config in tqdm.tqdm(search_space):
        if config_callback:
            config_callback(config)
        run, ground_truth = make_run_and_gt_fn(config)
        try:
            kernel_time = benchmark_config(
                run, ground_truth, num_iters=num_iters, atol=atol, rtol=rtol
            )
        except triton.runtime.autotuner.OutOfResources:
            # print("OutOfResources encountered during tuning.")
            # Some configurations may be invalid and fail to compile.
            continue
        except AssertionError as e:
            print(f"AssertionError encountered during tuning: {e}")
            continue
        except Exception as e:
            print(f"Config failed: {e}")
            continue
        if kernel_time < best_time:
            best_time = kernel_time
            best_config = config
    assert best_config is not None
    return best_config

def benchmark_config(run, ground_truth, num_iters=10, atol=1e-1, rtol=1e-1):
    """
    Args:
        run: Callable that returns the kernel output when called (no arguments).
        ground_truth: The expected output tensor to compare against.
        num_iters: Number of iterations to benchmark.
        atol: Absolute tolerance for comparison.
        rtol: Relative tolerance for comparison.
    Returns:
        Average latency in microseconds.
    """
    torch.cuda.synchronize()
    # JIT compilation & warmup
    for _ in range(5):
        run()
    torch.cuda.synchronize()

    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)

    latencies: list[float] = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        start_event.record()
        kernel_out = run()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
        torch.testing.assert_close(kernel_out, ground_truth, atol=atol, rtol=rtol)
    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    return avg


def get_search_space(small: bool = False):
    """
    Returns the search space for tuning.
    Args:
        small (bool): If True, returns a small search space for testing. If False, returns the full search space.
    """
    configs = []
    if small:
        num_stages_list = [2, 3]
        block_m_list = [16, 32]
        block_k_list = [64]
        block_n_list = [32, 64]
        num_warps_list = [4]
        group_size_list = [1]
        waves_per_eu_list = [3]
    else:
        num_stages_list = [2, 3, 4, 5]
        block_m_list = [16, 32, 64, 128, 256]
        block_k_list = [64, 128]
        block_n_list = [32, 64, 128, 256]
        num_warps_list = [4, 8]
        group_size_list = [1, 16, 32, 64]
        waves_per_eu_list = [2, 3, 4]

    for num_stages in num_stages_list:
        for block_m in block_m_list:
            for block_k in block_k_list:
                for block_n in block_n_list:
                    for num_warps in num_warps_list:
                        for group_size in group_size_list:
                            for waves_per_eu in waves_per_eu_list:
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
                                    }
                                )
    return configs

def save_configs_to_json(
    json_file_name: str,
    save_path: str,
    configs: dict | list[dict],
) -> None:
    os.makedirs(save_path, exist_ok=True)
    config_file_path = os.path.join(save_path, json_file_name)

    with open(config_file_path, "w") as f:
        json.dump(configs, f, indent=4)
        f.write("\n")