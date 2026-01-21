import torch
import time

from tqdm import tqdm

from aiter.ops.triton.tune.base import (
    tune_kernel,
    save_configs_to_json,
)
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.moe_routing_sigmoid_top1_fused import routing_sigmoid_top1


def torch_routing_sigmoid_top1(
    x, w, topk, fused_shared_experts=False, dummy_ids=None, dummy_weights=None
):
    scores = torch.matmul(x, w)  # [M, N]

    scores = torch.sigmoid(scores.to(torch.float32))  # [M, N]

    assert topk == 1

    topk_weights, topk_ids = torch.topk(scores, topk, dim=1)  # [M, topk]

    topk_ids = topk_ids.to(torch.int32)
    topk_weights = topk_weights.to(torch.float32)

    if fused_shared_experts:
        topk_ids = torch.cat(
            [
                topk_ids,
                dummy_ids,
            ],
            dim=1,
        )
        topk_weights = torch.cat(
            [topk_weights, dummy_weights],
            dim=1,
        )

    return topk_ids, topk_weights


def input_helper(
    M: int,
    N: int,
    K: int,
    fused_shared_experts: bool,
    dtype,
):
    """
    Generate input tensors for tuning.
    """
    torch.manual_seed(7)
    device = "cuda"

    x = torch.randint(-2, 3, (M, K), device=device).to(dtype)
    w = torch.randint(-2, 3, (K, N), device=device).to(dtype)

    dummy_ids = torch.ones((M, 1), dtype=torch.int32, device="cuda") * N
    dummy_weights = torch.ones((M, 1), dtype=torch.float32, device="cuda")

    return x, w, dummy_ids, dummy_weights


def make_run_and_gt_fn_factory(
    M, N, K, topk, fused_shared_experts, dtype
):
    """
    Factory function to create run and ground truth functions for a given config.
    """
    def make_run_and_gt(config):
        x, w, dummy_ids, dummy_weights = input_helper(
            M, N, K, fused_shared_experts, dtype
        )

        def run():
            topk_ids, topk_weights = routing_sigmoid_top1(
                x, w, topk, config=config
            )
            return topk_ids, topk_weights

        gt_topk_ids, gt_topk_weights = torch_routing_sigmoid_top1(
            x, w, topk, fused_shared_experts=fused_shared_experts,
            dummy_ids=dummy_ids, dummy_weights=dummy_weights
        )

        return run, (gt_topk_ids, gt_topk_weights)

    return make_run_and_gt


def tune_routing_sigmoid_top1(
    M,
    N,
    K,
    topk,
    fused_shared_experts,
    dtype,
    search_space,
):
    """
    Tune the routing_sigmoid_top1 kernel for a specific configuration.
    """
    make_run_and_gt = make_run_and_gt_fn_factory(
        M, N, K, topk, fused_shared_experts, dtype
    )

    best_config = tune_kernel(
        search_space=search_space,
        make_run_and_gt_fn=make_run_and_gt,
        num_iters=100,
    )
    return best_config


def tune_and_save_configs(
    batch_sizes,
    N_values,
    K,
    topk,
    fused_shared_experts,
    dtype,
    search_space,
    save_path,
    device_name,
):
    """
    Tune for multiple batch sizes and N values, then save configs.
    """
    start = time.time()
    best_configs = {}

    for N in tqdm(N_values, desc="N values"):
        n_key = "N16" if N <= 16 else "N128"
        n_configs = {}

        for M in tqdm(batch_sizes, desc=f"N={N} - M", leave=False):
            best_config = tune_routing_sigmoid_top1(
                M=M,
                N=N,
                K=K,
                topk=topk,
                fused_shared_experts=fused_shared_experts,
                dtype=dtype,
                search_space=search_space,
            )

            # Determine M key
            m_key = (
                "xlarge"
                if M >= 8192
                else "large" if M >= 4096 else "medium" if M >= 2048 else "small"
            )
            n_configs[m_key] = best_config

        best_configs[n_key] = n_configs

    json_file_name = f"{device_name}-MOE_ROUTING_SIGMOID_TOPK1.json"
    save_configs_to_json(json_file_name, save_path, best_configs)
    end = time.time()
    print(f"Tuning took {end - start:.2f} seconds")

def get_routing_search_space(small: bool = False):
    """
    Returns the search space for tuning the routing kernel.
    Args:
        small (bool): If True, returns a small search space for testing. If False, returns the full search space.
    """
    configs = []
    if small:
        block_m_list = [32, 64]
        block_k_list = [32, 64]
        num_warps_list = [4]
        num_stages_list = [2, 3]
        waves_per_eu_list = [1]
        kpack_list = [1]
    else:
        block_m_list = [16, 32, 64]
        block_k_list = [16, 32, 64, 128, 256]
        num_warps_list = [2, 4, 8]
        num_stages_list = [1, 2, 3, 4]
        waves_per_eu_list = [0, 1, 2, 4, 6, 8]
        kpack_list = [1, 2]

    for block_m in block_m_list:
        for block_k in block_k_list:
            for num_warps in num_warps_list:
                for num_stages in num_stages_list:
                    for waves_per_eu in waves_per_eu_list:
                        for kpack in kpack_list:
                            configs.append(
                                {
                                    "BLOCK_M": block_m,
                                    "BLOCK_K": block_k,
                                    "num_warps": num_warps,
                                    "num_stages": num_stages,
                                    "waves_per_eu": waves_per_eu,
                                    "kpack": kpack
                        }
                    )
    return configs

def main():
    dev = arch_info.get_arch()

    torch.cuda.init()

    # Test parameters
    batch_sizes = [1024]  # M values
    N_values = [128]
    K = 5120
    topk = 1
    fused_shared_experts = False
    dtype = torch.bfloat16

    # Use routing-specific search space
    search_space = get_routing_search_space()
    save_path = AITER_TRITON_CONFIGS_PATH + "/moe/"

    # Tune for default (bfloat16 with fused_shared_experts)
    tune_and_save_configs(
        batch_sizes=batch_sizes,
        N_values=N_values,
        K=K,
        topk=topk,
        fused_shared_experts=fused_shared_experts,
        dtype=dtype,
        search_space=search_space,
        save_path=save_path,
        device_name=dev,
    )


if __name__ == "__main__":
    main()
