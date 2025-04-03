import triton
import triton.language as tl
from utils.benchmark_utils import get_model_configs, get_available_models, get_dtype_bytes, torch_to_tl_dtype
from op_tests.triton.test_pa_decode import input_helper
import torch
import argparse
from aiter.ops.triton.pa_decode import paged_attention_decode
import sys

def model_benchmark_configs(args):
    config_file = args.model_configs
    configs = get_model_configs(config_path=config_file, models="llama3" if args.model == None else args.model)
    fa_configs = []
    batch_size = args.b if args.b else 1024

    for model_name, config in configs.items():
        HQ = config["num_attention_heads"]
        HK = HQ if config["num_key_value_heads"] is None else config["num_key_value_heads"]
        SEQ_LEN = args.sq if args.sq else 8192
        HEAD_DIM = config["hidden_size"] // HQ
        fa_configs.append((model_name, batch_size, HQ, HK, SEQ_LEN, HEAD_DIM))

    return fa_configs

def paged_attn(B, H_Q, H_KV, D, KV_BLK_SZ, SEQ_LEN, num_blocks, dtype, kv_cache_dtype, compute_type, output_type):
    query, triton_output, _, _, key_cache_tri, value_cache_tri, context_lens, block_tables, max_context_len = input_helper(B, H_Q, H_KV, D, KV_BLK_SZ, SEQ_LEN, dtype, kv_cache_dtype, output_type, num_blocks)
    attn_scale = 1.0 / (D**0.5)

    return lambda: paged_attention_decode(
        triton_output,
        query,
        key_cache_tri,
        value_cache_tri,
        context_lens,
        block_tables,
        attn_scale,
        max_context_len,
        compute_type,
    )

def run_benchmark(args):
    dtype = arg_to_torch_dtype[args.dtype]
    kv_cache_dtype = arg_to_torch_dtype[args.kv_cache_dtype]
    compute_type = torch_to_tl_dtype[arg_to_torch_dtype[args.compute_type]]
    output_type = arg_to_torch_dtype[args.output_type]

    x_vals_list = model_benchmark_configs(args)
    x_names = ['model', 'batch_size', 'HQ', 'HK', 'SEQ_LEN', "HEAD_DIM"]

    model_name = "paged-attn-decode"

    line_names = ['Time (ms)', 'TFLOPS', 'Bandwidth (GB/s)']
    line_vals = ['time', 'tflops', 'bandwidth']

    benchmark = triton.testing.Benchmark(
        x_names=x_names, x_vals=x_vals_list, line_arg='metric', line_vals=line_vals, line_names=line_names,
        styles=[('red', '-'), ('blue', '-'),
                ('yellow', '-')], ylabel='ms / TFLOPS / GB/s', plot_name=f'{model_name}-benchmark', args={})

    @triton.testing.perf_report([benchmark])
    def bench_paged_attn_decode(batch_size, HQ, HK, SEQ_LEN, HEAD_DIM, metric, model=None):
        # TODO tune this
        KV_BLK_SZ = 128
        num_blocks = 4
        fn = paged_attn(batch_size, HQ, HK, HEAD_DIM, KV_BLK_SZ, SEQ_LEN, num_blocks, dtype, kv_cache_dtype, compute_type, output_type)

        ms = triton.testing.do_bench(fn, warmup=25, rep=100)

        # query and output
        mem = (batch_size * HQ * HEAD_DIM) * (get_dtype_bytes(dtype) + get_dtype_bytes(output_type))
        # kv_cache
        mem += (num_blocks * HK * KV_BLK_SZ * HEAD_DIM * get_dtype_bytes(kv_cache_dtype) * 2)
        # block_tables int32
        mem += batch_size * ((SEQ_LEN + KV_BLK_SZ - 1) // KV_BLK_SZ) * 4
        # context_lens fp32
        mem += batch_size * 4

        # bhd bhsd => bhs bhsd => bhs, 2 for multiplication and accumulation. and there are 2 gemms
        flops = (2.0 * batch_size * HQ * SEQ_LEN * HEAD_DIM) * 2

        bandwidth = mem / (ms * 1e-3) * 1e-9  # GB/s
        # bandwidth = mem / (ms * 1e-3) * 1e-9  # GB/s
        tflops = flops / ms * 1e-9

        # Return exactly one scalar depending on which metric is active
        if metric == 'time':
            return ms
        elif metric == 'tflops':
            return tflops
        elif metric == 'bandwidth':
            return bandwidth
        else:
            raise ValueError("Unknown metric: " + metric)

    bench_paged_attn_decode.run(save_path=".", print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark Paged Attention decode",
        allow_abbrev=False,
    )
    parser.add_argument('-model_configs', type=str, default="utils/model_configs.json", help="Model config json file.")
    available_models = get_available_models()  # Dynamically load model names
    model_help = ("Model name to benchmark. Select from: [" + ", ".join(available_models) +
                  "]. Use 'all' to benchmark all models or leave blank for the default benchmark script.")
    parser.add_argument('-model', type=str, default=None, help=model_help)
    parser.add_argument("-b", type=int, default=0)
    parser.add_argument("-hq", type=int, default=0)
    parser.add_argument("-hk", type=int, default=0)
    parser.add_argument("-sq", type=int, default=0)
    parser.add_argument("-dtype", default='fp16')
    parser.add_argument("-kv_cache_dtype", default='fp16')
    parser.add_argument("-compute_type", default='fp16')
    parser.add_argument("-output_type", default='fp16')
    args = parser.parse_args()
    return args


arg_to_torch_dtype = {
    'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32, "e5m2fnuz": torch.float8_e5m2fnuz, "e4m3fnuz":
    torch.float8_e4m3fnuz
}


def main():
    args = parse_args()
    run_benchmark(args)


if __name__ == '__main__':
    sys.exit(main())
