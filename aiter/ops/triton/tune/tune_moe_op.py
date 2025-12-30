import torch
import time

from tqdm import tqdm

from aiter.ops.triton.tune.base import (
    tune_kernel,
    get_search_space,
    save_configs_to_json,
)
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils.types import torch_to_triton_dtype
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.moe_op import fused_moe

from op_tests.triton_tests.moe.test_moe import (
    quantize_fp8,
    quantize_int8,
    torch_moe_ref,
    torch_moe_align_block_size_ref,
)


def input_helper(
    M: int,
    N: int,
    K: int,
    top_k: int,
    E: int,
    routed_weight: bool,
    dtype,
    fp8_w8a8: bool,
    int8_w8a16: bool,
    config: dict,
):
    assert not (fp8_w8a8 and int8_w8a16)

    a = torch.randn((M, K), dtype=dtype, device="cuda")
    b = torch.rand((E, N, K), dtype=dtype, device="cuda")
    a_scale = None
    b_scale = None

    if fp8_w8a8:
        b, _, b_scale = quantize_fp8(b, dim=(0,))

    if int8_w8a16:
        b, _, b_scale = quantize_int8(b, dim=(0,))

    b_zp = False

    c = torch.zeros((M, top_k, N), dtype=dtype, device="cuda")
    c_silu = torch.zeros((M * top_k, N // 2), dtype=dtype, device="cuda")

    values = torch.randn(M, E, dtype=dtype, device="cuda")

    softmax_vals = torch.softmax(values, dim=1)
    topk_weights, topk_ids = torch.topk(softmax_vals, k=top_k, dim=1)

    sorted_token_ids, expert_ids, num_tokens_post_padded = (
        torch_moe_align_block_size_ref(topk_ids, config["BLOCK_SIZE_M"], E)
    )

    return (
        a,
        b,
        c,
        c_silu,
        b_zp,
        a_scale,
        b_scale,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        config,
    )


def make_run_and_gt_fn_factory(
    M, N, K, top_k, E, routed_weight, dtype, fp8_w8a8, int8_w8a16, int4_w4a16
):
    def make_run_and_gt(config):
        (
            a,
            b,
            triton_out,
            triton_out_silu,
            b_zp,
            a_scale,
            b_scale,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            _,
        ) = input_helper(
            M,
            N,
            K,
            top_k,
            E,
            routed_weight=routed_weight,
            dtype=dtype,
            fp8_w8a8=fp8_w8a8,
            int8_w8a16=int8_w8a16,
            config=config,
        )

        def run():
            torch.cuda.empty_cache()
            fused_moe(
                a,
                b,
                triton_out,
                a_scale,
                b_scale,
                b_zp,
                topk_weights,
                topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                routed_weight,
                top_k,
                torch_to_triton_dtype[dtype],
                fp8_w8a8,
                int8_w8a16,
                False,
                config=config,
            )
            return triton_out

        torch_out = torch.empty_like(triton_out)
        ground_truth = torch_moe_ref(
            a,
            b,
            torch_out,
            a_scale,
            b_scale,
            None,
            0,
            topk_ids,
            topk_weights,
            routed_weight,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            dtype,
            fp8_w8a8,
            int8_w8a16,
            False,
        )
        return run, ground_truth

    return make_run_and_gt


def tune_fused_moe(
    M,
    N,
    K,
    top_k,
    E,
    routed_weight,
    dtype,
    fp8_w8a8,
    int8_w8a16,
    int4_w4a16,
    search_space,
):
    make_run_and_gt = make_run_and_gt_fn_factory(
        M, N, K, top_k, E, routed_weight, dtype, fp8_w8a8, int8_w8a16, int4_w4a16
    )

    best_config = tune_kernel(
        search_space=search_space,
        make_run_and_gt_fn=make_run_and_gt,
    )
    return best_config


def tune_and_save_configs(
    batch_sizes,
    N,
    K,
    topk,
    E,
    routed_weight,
    dtype,
    fp8_w8a8,
    int8_w8a16,
    int4_w4a16,
    search_space,
    save_path,
    device_name,
    tag,
):
    start = time.time()
    benchmark_results = [
        tune_fused_moe(
            batch_size,
            N,
            K,
            topk,
            E,
            routed_weight=routed_weight,
            dtype=dtype,
            fp8_w8a8=fp8_w8a8,
            int8_w8a16=int8_w8a16,
            int4_w4a16=int4_w4a16,
            search_space=search_space,
        )
        for batch_size in tqdm(batch_sizes)
    ]
    best_configs = {
        (
            "small_M"
            if batch_size <= 256
            else "medium_M"
            if batch_size <= 2048
            else "large_M"
        ): config
        for batch_size, config in zip(batch_sizes, benchmark_results)
    }
    json_file_name = f"{device_name}-MOE-{tag}.json"
    save_configs_to_json(json_file_name, save_path, best_configs)
    end = time.time()
    print(f"Tuning for {tag} took {end - start:.2f} seconds")


def main():
    dev = arch_info.get_arch()

    torch.cuda.init()

    batch_sizes = [256, 2048, 4096]  # M
    N = 384
    K = 768
    topk = 8
    E = 128
    search_space = get_search_space()
    save_path = AITER_TRITON_CONFIGS_PATH + "/moe/"

    # Tune for default (float16)
    # tune_and_save_configs(
    #     batch_sizes=batch_sizes,
    #     N=N,
    #     K=K,
    #     topk=topk,
    #     E=E,
    #     routed_weight=False,
    #     dtype=torch.float16,
    #     fp8_w8a8=False,
    #     int8_w8a16=False,
    #     int4_w4a16=False,
    #     search_space=search_space,
    #     save_path=save_path,
    #     device_name=dev,
    #     tag="DEFAULT",
    # )
    # tune_and_save_configs(
    #     batch_sizes=batch_sizes,
    #     N=N,
    #     K=K,
    #     topk=topk,
    #     E=E,
    #     routed_weight=False,
    #     dtype=torch.float16,
    #     fp8_w8a8=True,
    #     int8_w8a16=False,
    #     int4_w4a16=False,
    #     search_space=search_space,
    #     save_path=save_path,
    #     device_name=dev,
    #     tag="FP8_W8A8",
    # )
    tune_and_save_configs(
        batch_sizes=batch_sizes,
        N=N,
        K=K,
        topk=topk,
        E=E,
        routed_weight=False,
        dtype=torch.float16,
        fp8_w8a8=False,
        int8_w8a16=True,
        int4_w4a16=False,
        search_space=search_space,
        save_path=save_path,
        device_name=dev,
        tag="INT8_W8A16",
    )
    tune_and_save_configs(
        batch_sizes=batch_sizes,
        N=N,
        K=K,
        topk=topk,
        E=E,
        routed_weight=False,
        dtype=torch.float16,
        fp8_w8a8=False,
        int8_w8a16=False,
        int4_w4a16=True,
        search_space=search_space,
        save_path=save_path,
        device_name=dev,
        tag="INT4_W4A16",
    )


if __name__ == "__main__":
    main()
