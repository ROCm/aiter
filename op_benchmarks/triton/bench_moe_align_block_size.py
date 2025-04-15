import triton
import triton.language as tl
from utils.benchmark_utils import get_model_configs, get_available_models
from op_tests.triton.test_moe_align_block_size import input_helper
import torch
import argparse
from aiter.ops.triton.moe_align_block_size import moe_align_block_size_triton
import sys

def model_benchmark_configs(args):
    config_file = args.model_configs
    configs = get_model_configs(config_path=config_file, models="mistral")
    moe_configs = []
    M = args.M if args.M else 4096  # check size
    # M, K, N, E, top_k

    for model_name, config in configs.items():
        N1 = config["intermediate_size"]
        K1 = config["hidden_size"]

        N2 = config["hidden_size"]
        K2 = config["intermediate_size"] // 2

        E = 8
        top_k = 2

        moe_configs.append((model_name, M, N1, K1, E, top_k))
        moe_configs.append((model_name, M, N2, K2, E, top_k))

    return moe_configs

def fused_moe_align_block_size(M: int, E: int, top_k: int, block_size: int):
    topk_ids = input_helper(M, E, top_k)

    max_num_tokens_padded = topk_ids.numel() + E * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded, ),
                                dtype=torch.int32,
                                device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty((max_num_m_blocks, ),
                                dtype=torch.int32,
                                device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1),
                                        dtype=torch.int32,
                                        device=topk_ids.device)

    return lambda: moe_align_block_size_triton(topk_ids,
    E,
    block_size,
    sorted_ids,
    expert_ids,
    num_tokens_post_pad)


def run_benchmark(custom, args):
    block_size = args.block_size
    x_names = ['M', 'N', 'K', 'E', 'top_k']

    x_vals_list = model_benchmark_configs(args)
    x_names = ['model', 'M', 'N', 'K', 'E', 'top_k']


    line_names = ['Time (ms)', 'Bandwidth (GB/s)']
    line_vals = ['time', 'bandwidth']

    benchmark = triton.testing.Benchmark(
        x_names=x_names, x_vals=x_vals_list, line_arg='metric', line_vals=line_vals, line_names=line_names,
        styles=[('red', '-'), ('blue', '-')], ylabel='ms / GB/s', plot_name='moe-align-block-size', args={})

    @triton.testing.perf_report([benchmark])
    def bench_moe_align_block_size(M, N, K, E, top_k, metric, model=None):
        max_num_tokens_padded = M * K + E * (block_size - 1)
        max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)

        # topk_ids, int64
        mem_read = (M * K) * 8

        mem_write = (max_num_tokens_padded + max_num_m_blocks + 1) * 4
        mem = mem_read + mem_write

        fn = fused_moe_align_block_size(M, E, top_k, block_size)

        ms = triton.testing.do_bench(fn, warmup=25, rep=100)

        bandwidth = mem / (ms * 1e-3) * 1e-9  # GB/s
        # tflops = flops / ms * 1e-9

        # Return exactly one scalar depending on which metric is active
        if metric == 'time':
            return ms
        # elif metric == 'tflops':
        #     return tflops
        elif metric == 'bandwidth':
            return bandwidth
        else:
            raise ValueError("Unknown metric: " + metric)

    bench_moe_align_block_size.run(save_path=".", print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark MoE align block size",
        allow_abbrev=False,
    )
    parser.add_argument('-model_configs', type=str, default="utils/model_configs.json", help="Model config json file.")
    available_models = get_available_models()  # Dynamically load model names
    model_help = ("Model name to benchmark. Select from: [" + ", ".join(available_models) +
                  "]. Use 'all' to benchmark all models or leave blank for the default benchmark script.")
    parser.add_argument('-model', type=str, default=None, help=model_help)
    parser.add_argument("-M", type=int, default=0, help="M dimension")
    parser.add_argument("-block_size", type=int, default=128)
    args = parser.parse_args()
    return args


arg_to_torch_dtype = {
    'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32, "e5m2fnuz": torch.float8_e5m2fnuz, "e4m3fnuz":
    torch.float8_e4m3fnuz
}


def main():
    args = parse_args()
    custom_config = False
    # If user provides all M,K,N,E,top_k we consider it custom
    if args.M and args.K and args.N and args.E and args.top_k:
        custom_config = True
    run_benchmark(custom_config, args)


if __name__ == '__main__':
    sys.exit(main())
