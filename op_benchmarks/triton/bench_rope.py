import argparse
import sys
import torch
import triton
from aiter.ops.triton.rope import (
    RotateStyle,
    rope_cached_thd_positions_2c_fwd,
)

# from utils.benchmark_utils import get_model_configs, get_available_models


# TODO: move to aiter/op_tests/triton_tests/test_rope_triton.py
def generate_rope_inputs(
    B: int,
    S: int,
    H: int,
    D: int,
    cached: bool,
    reuse_freqs_front_part: bool,
    nope: bool,
    pos: bool,
    offs: bool,
    two_inputs: bool,
    layout: str,
    dtype: torch.dtype,
):
    torch.manual_seed(20)

    device = "cuda"
    if layout == "thd":  # T == S
        assert B == 1, "B should always be 1 in THD layout"
        input_shape = (S, H, D)
        pos_offs_shape = (S,)
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
    positions = (
        torch.randint(
            int(S * 0.25) if offs else 0,
            int(S * 0.75) if offs else S,
            pos_offs_shape,
            device=device,
        )
        if pos
        else None
    )
    offsets = (
        torch.randint(int(S * -0.25), int(S * 0.25), pos_offs_shape, device=device)
        if offs
        else None
    )
    # ref_freqs = freqs[positions if offsets is None else torch.add(positions, offsets)].squeeze(-2) if pos else freqs

    cos = torch.cos(freqs) if cached else None
    sin = torch.sin(freqs) if cached else None

    if cached and layout == "thd":
        cos = cos.reshape(S, freqs_D)
        sin = sin.reshape(S, freqs_D)

    return x, y, freqs, positions, offsets, cos, sin


def str_to_bool(v, vstr):
    if v.lower() in ["true", "yes"]:
        return True
    elif v.lower() in ["false", "no"]:
        return False
    else:
        raise NotImplementedError(f"invalid {vstr}: {v}")


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
    x_vals = [
        (
            B,
            1,
            H,
            D,
            cached,
            rotate_style,
            reuse_freqs_front_part,
            nope,
            nope_first,
            pos,
            offs,
            two_inputs,
            layout,
            inplace,
            dtype,
        )
    ]
    return x_vals


def run_benchmark(args):
    (
        B,
        S,
        H,
        D,
        cached,
        rotate_style,
        reuse_freqs_front_part,
        nope,
        nope_first,
        pos,
        offs,
        two_inputs,
        layout,
        inplace,
        dtype,
    ) = (
        args.B,
        args.S,
        args.H,
        args.D,
        args.cached,
        args.rotate_style,
        args.reuse_freqs_front_part,
        args.nope,
        args.nope_first,
        args.pos,
        args.offs,
        args.two_inputs,
        args.l,
        args.inplace,
        args.dtype,
    )

    cached = str_to_bool(cached, "cached")
    reuse_freqs_front_part = str_to_bool(
        reuse_freqs_front_part, "reuse_freqs_front_part"
    )
    nope = str_to_bool(nope, "nope")
    nope_first = str_to_bool(nope_first, "nope_first")
    pos = str_to_bool(pos, "pos")
    offs = str_to_bool(offs, "offs")
    two_inputs = str_to_bool(two_inputs, "two_inputs")
    inplace = str_to_bool(inplace, "inplace")

    rep = args.repeat

    if dtype == "fp16":
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    else:
        raise NotImplementedError(f"dtype {dtype} not supported")

    if rotate_style not in ["gptj", "neox"]:
        raise NotImplementedError(f"rotate_style {rotate_style} not supported")

    x_names = [
        "B",
        "S",
        "H",
        "D",
        "cached",
        "rotate_style",
        "reuse_freqs_front_part",
        "nope",
        "nope_first",
        "pos",
        "offs",
        "two_inputs",
        "layout",
        "inplace",
        "dtype",
    ]
    x_vals_list = [
        (
            B,
            S,
            H,
            D,
            cached,
            rotate_style,
            reuse_freqs_front_part,
            nope,
            nope_first,
            pos,
            offs,
            two_inputs,
            layout,
            inplace,
            dtype,
        )
    ]
    # x_vals_list = get_x_vals()

    # @triton.testing.perf_report([benchmark])
    def bench_rope(
        B: int,
        S: int,
        H: int,
        D: int,
        cached: bool,
        rotate_style: int,
        reuse_freqs_front_part: bool,
        nope: bool,
        nope_first: bool,
        pos: bool,
        offs: bool,
        two_inputs: bool,
        layout: str,
        inplace: bool,
        dtype: torch.dtype,
        metric,
        provider,
    ):
        x, y, freqs, positions, offsets, cos, sin = generate_rope_inputs(
            B,
            S,
            H,
            D,
            cached,
            reuse_freqs_front_part,
            nope,
            pos,
            offs,
            two_inputs,
            layout,
            dtype,
        )
        if rotate_style == "gptj":
            rotate_style = RotateStyle.GPTJ
        elif rotate_style == "neox":
            rotate_style = RotateStyle.NEOX
        # flops
        flops = B * S * H * (D / 2.0) * 3.0 * 2.0 * (2.0 if two_inputs else 1.0)

        # memory transfer (B = 1, T = S for thd layout, positions and offsets are always int)
        mem_read = (
            B * S * H * D * ((2.0 * x.element_size()) if two_inputs else 1.0)
            + S * D * ((2.0 * freqs.element_size()) if cached else 1.0)
            + B * S * ((1.0 * positions.element_size()) if pos else 0.0)
            + B * S * ((1.0 * offsets.element_size()) if offs else 0.0)
        )

        mem_write = B * S * H * D * (2.0 if two_inputs else 1.0) * x.element_size()
        mem = mem_read + mem_write

        fn = None
        if (
            cached
            and two_inputs
            and pos
            and not offs
            and layout == "thd"
            and not inplace
        ):
            fn = lambda: rope_cached_thd_positions_2c_fwd(  # noqa: E731
                x,
                y,
                cos,
                sin,
                positions,
                rotate_style,
                reuse_freqs_front_part,
                nope_first,
                transpose_output=False,
            )
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
            raise NotImplementedError(
                f"No API with option: [layout='{layout}', cached={cached}, two_inputs={two_inputs}, pos={pos}, offs={offs}, inplace={inplace}]."
            )

        from triton.testing import runtime

        di = runtime.driver.active.get_device_interface()
        cache = runtime.driver.active.get_empty_cache_for_benchmark()
        for i in range(rep):
            cache.zero_()
            di.synchronize()
            fn()
            di.synchronize()

        return flops, mem

        # ms = triton.testing.do_bench(fn, warmup=25, rep=100)
        # bandwidth = mem / (ms * 1e-3) * 1e-9  # GB/s
        # return bandwidth

    for x_vals in x_vals_list:
        print("Running input config:")
        for i in range(len(x_names)):
            print(f"    {x_names[i]:23s}= {x_vals_list[0][i]}")
        print(f"Number of repitition = {rep}")
        flops, mem = bench_rope(*x_vals, None, None)
        print(f"Total flops  = {flops/1e12 : .6e} (TFLOPS)")
        print(f"Total memory = {mem/1e9 : .6e} (GB)")
    # bench_rope.run(save_path=".", print_data=False)
    print("")
    print(
        "This script will not print out runtime as short running kernels cannot be measured accuratly throught triton.testing.do_bench function, please use rocprof to measure accurate runtime, use -h/--help for more information"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark RoPE",
        description="This script will not print out runtime as short running kernels cannot be measured accuratly throught triton.testing.do_bench function, please use rocprof to measure accurate runtime (the DurationNs column) in results.csv or results.stats.csv. For instance, try \"rocprof --stats python bench_rope.py -l 'thd' -T 1 -H 128 -D 64 --two_inputs=true\"",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-l", type=str, help="'thd' or 'sbhd' the layout of the input.", default="thd"
    )
    parser.add_argument(
        "-B",
        type=int,
        help="the batch size, B (this argument will be ignored if you set -l to 'thd').",
        default=1,
    )
    parser.add_argument(
        "-S",
        "-T",
        type=int,
        help="the sequence length, S, or the number of tokens, T",
        default=4096,
    )
    parser.add_argument("-H", type=int, help="the number of heads, H", default=128)
    parser.add_argument("-D", type=int, help="the head dimension, D", default=64)
    parser.add_argument("--cached", type=str, help="cached sin/cos", default="true")
    parser.add_argument("--rotate_style", type=str, help="gptj or neox", default="gptj")
    parser.add_argument(
        "--reuse_freqs_front_part",
        type=str,
        help="turn on reuse_freqs_front_part",
        default="true",
    )
    parser.add_argument("--nope", type=str, help="turn on nope", default="true")
    parser.add_argument(
        "--nope_first", type=str, help="turn on nope_fist", default="true"
    )
    parser.add_argument("--pos", type=str, help="input positions", default="true")
    parser.add_argument("--offs", type=str, help="input offsets", default="false")
    parser.add_argument(
        "--two_inputs", type=str, help="input both K and Q", default="true"
    )
    parser.add_argument(
        "--inplace", type=str, help="inplace operation", default="false"
    )
    parser.add_argument("--dtype", type=str, help="data type", default="bf16")
    parser.add_argument(
        "--repeat", type=int, help="number of repetition for benchmark", default=1000
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
