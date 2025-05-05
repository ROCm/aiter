
import argparse
import sys
import torch
import triton
from aiter.ops.triton.rope import RotateStyle, rope_fwd, rope_cached_thd_positions_2c_fwd
# from utils.benchmark_utils import get_model_configs, get_available_models

#TODO: move to aiter/op_tests/triton_tests/test_rope_triton.py
def generate_rope_inputs(B: int, S: int, H: int, D: int, cached: bool, reuse_freqs_front_part: bool, nope: bool, pos: bool, offs: bool, two_inputs: bool, layout: str, dtype: torch.dtype):
    torch.manual_seed(20)
    
    device = "cuda"
    if layout == "thd": # T == S
        assert B == 1, "B should always be 1 in THD layout"
        input_shape = (S, H, D)
        pos_offs_shape = (S, )
    elif layout == "sbhd":
        input_shape = (S, B, H, D)
        pos_offs_shape = (S, B)
    else:
        raise NotImplementedError(f"layout '{layout}' not supported")

    x = torch.randn(input_shape, dtype=dtype, device="cuda")
    y = torch.randn(input_shape, dtype=dtype, device="cuda") if two_inputs else None

    freqs_D = D
    if nope:
        freqs_D = freqs_D // 2
    if reuse_freqs_front_part:
        freqs_D = freqs_D // 2
    
    freqs = torch.randn((S, 1, 1, freqs_D), dtype=dtype, device="cuda")
    positions = torch.randint(int(S * 0.25) if offs else 0, int(S * 0.75) if offs else S, pos_offs_shape, device=device) if pos else None
    offsets  = torch.randint(int(S * -0.25), int(S * 0.25), pos_offs_shape, device=device) if offs else None
    # ref_freqs = freqs[positions if offsets is None else torch.add(positions, offsets)].squeeze(-2) if pos else freqs
    
    cos = torch.cos(freqs) if cached else None
    sin = torch.sin(freqs) if cached else None

    if cached and layout == "thd":
        cos = cos.reshape(S, freqs_D)
        sin = sin.reshape(S, freqs_D)

    return x, y, freqs, positions, offsets, cos, sin

def get_x_vals():
    """
        this get_x_vals is for DeepSeekV2 (thd)
        https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/config.json
        H = num_attention_heads = 128
        D = qk_rope_head_dim = 64
    """
    B = 1
    H = 128
    D = 64

    cached = True
    rotate_style = RotateStyle.GPTJ
    reuse_freqs_front_part = True
    nope = False
    nope_first = False
    pos = True
    offs = False
    two_inputs = True
    layout = "thd"
    inplace = False
    dtype = torch.bfloat16

    # x_vals = [(B, 2**i, H, D, cached, rotate_style, reuse_freqs_front_part, nope, nope_first, pos, offs, two_inputs, layout, inplace, dtype) 
    #           for i in range(0, 13)]
    x_vals = [(B, 1, H, D, cached, rotate_style, reuse_freqs_front_part, nope, nope_first, pos, offs, two_inputs, layout, inplace, dtype)]
    return x_vals

def run_benchmark(args):
    assert not(args.shape and args.model) or not(args.shape and args.M), \
        "User can specify --shape or --model MODEL -M VAL exclusively"

    x_names = ['B', 'S', 'H', 'D', "cached", "rotate_style", "reuse_freqs_front_part", "nope", "nope_first", "pos", "offs", "two_inputs", "layout", "inplace", "dtype"]
    if args.shape:
        x_vals_list = [args.shape]
    else:
        x_vals_list = get_x_vals()

    if args.metric == 'time':
        ylabel = 'Time (ms)'
    elif args.metric == 'throughput':
        ylabel = 'Throughput (TFLOPS)'
    elif args.metric == 'bandwidth':
        ylabel = 'Bandwidth (GB/s)'
    else:
        raise NotImplementedError(f"{args.metric} is not supported")

    line_names = ["Triton"]
    line_vals = ['triton']
    benchmark = triton.testing.Benchmark(
        x_names=x_names, x_vals=x_vals_list,
        line_arg='provider', line_vals=line_vals, line_names=line_names,
        styles=[('green', '-')],
        ylabel=ylabel, plot_name=f'RoPE Benchmark', args={"metric": args.metric})

    @triton.testing.perf_report([benchmark])
    def bench_rope(B: int, S: int, H: int, D: int, cached: bool, rotate_style: int, reuse_freqs_front_part: bool, nope: bool, nope_first: bool, pos: bool, offs: bool, two_inputs: bool, layout: str, inplace: bool, dtype: torch.dtype, metric, provider):
        x, y, freqs, positions, offsets, cos, sin = generate_rope_inputs(B, S, H, D, cached, reuse_freqs_front_part, nope, pos, offs, two_inputs, layout, dtype)
        # flops
        flops = B * S * H * (D/2.0) * 3.0 * 2.0 * (2.0 if two_inputs else 1.0)
        
        # memory transfer (B = 1, T = S for thd layout, positions and offsets are always int)
        mem_read = B * S * H * D * ((2.0 * x        .element_size()) if two_inputs else 1.0) + \
                       S *     D * ((2.0 * freqs    .element_size()) if cached     else 1.0) + \
                   B * S *         ((1.0 * positions.element_size()) if pos        else 0.0) + \
                   B * S *         ((1.0 * offsets  .element_size()) if offs       else 0.0)
                   
        mem_write = B * S * H * D * (2.0 if two_inputs else 1.0) * x.element_size()
        mem = mem_read + mem_write
        
        fn = None
        if cached and two_inputs and pos and not offs and layout == "thd" and not inplace:
            fn = lambda: rope_cached_thd_positions_2c_fwd(x, y, cos, sin, positions, rotate_style, reuse_freqs_front_part, nope_first, transpose_output = False)
        # elif not cached and not two_inputs and not pos and not offs:
        #     if layout == "sbhd":
        #         if inplace:
        #             fn = lambda: rope_fwd_inplace(x, freqs, rotate_style, reuse_freqs_front_part, nope_first, transpose_output = False)
        #         else:
        #             fn = lambda: rope_fwd(x, freqs, rotate_style, reuse_freqs_front_part, nope_first, transpose_output = False)
        #     elif layout == "thd":
        #         seqlens = [0, S]
        #         cu_seqlens = torch.Tensor(seqlens).to(torch.int).to(freqs.device)
        #         if inplace:
        #             fn = lambda: rope_fwd_thd_inplace(x, cu_seqlens, freqs, rotate_style, reuse_freqs_front_part, nope_first, transpose_output = False)
        #         else:
        #             fn = lambda: rope_fwd_thd(x, cu_seqlens, freqs, rotate_style, reuse_freqs_front_part, nope_first, transpose_output = False)        
        else:
            raise NotImplementedError(f"No API with option: [layout='{layout}', cached={cached}, two_inputs={two_inputs}, pos={pos}, offs={offs}, inplace={inplace}].")
        
        from triton.testing import runtime
        di = runtime.driver.active.get_device_interface()
        cache = runtime.driver.active.get_empty_cache_for_benchmark()
        for i in range(1000):
            cache.zero_()
            di.synchronize()
            fn()
            di.synchronize()
        ms = 0
        # ms = triton.testing.do_bench(fn, warmup=25, rep=100)

        # Return exactly one scalar depending on which metric is active
        if metric == 'time':
            return ms
        elif metric == 'throughput':
            tflops = flops / ms * 1e-9
            return tflops
        elif metric == 'bandwidth':
            bandwidth = mem / (ms * 1e-3) * 1e-9  # GB/s
            # print(S, ms*1000, mem/1e9)
            return bandwidth
        else:
            raise ValueError("Unknown metric: " + metric)


    bench_rope.run(save_path=".", print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark RoPE",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # available_models = get_available_models()  # Dynamically load model names
    # model_help = ("Model name to benchmark. Select from: [" + ", ".join(available_models) +
    #         "]. Use 'all' to benchmark all models or leave blank for the default benchmark script.")
    # parser.add_argument('--model-configs', type=str, default="utils/model_configs.json",
    #         help="Model config json file.")
    # parser.add_argument('--model', type=str, help=model_help)
    parser.add_argument("--shape", type=int, nargs=4, metavar=("B", "S", "H", "D"),
            help="user-defined shape to benchmark")
    parser.add_argument("--metric", type=str, choices=["time", "throughput", "ban-dwidth"],
            default="bandwidth", help="metric to plot")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    run_benchmark(args)


if __name__ == '__main__':
    sys.exit(main())
