import argparse
import sys
import torch
from triton.testing import runtime
from op_tests.triton_tests.test_rope import generate_rope_inputs
from aiter.ops.triton.rope import RotateStyle
from aiter.ops.triton.rope import (
    rope_fwd,
    rope_fwd_inplace,
    rope_bwd,
    rope_thd_fwd,
    rope_thd_fwd_inplace,
    rope_thd_bwd,
    rope_cached_fwd,
    rope_cached_fwd_inplace,
    rope_cached_positions_fwd,
    rope_cached_positions_fwd_inplace,
    rope_cached_positions_offsets_fwd,
    rope_cached_positions_offsets_fwd_inplace,
    rope_cached_bwd,
    rope_cached_positions_bwd,
    rope_cached_positions_offsets_bwd,
    rope_cached_thd_positions_2c_fwd,
    rope_cached_thd_positions_2c_fwd_inplace,
    rope_cached_thd_positions_offsets_2c_fwd,
    rope_cached_thd_positions_offsets_2c_fwd_inplace,
    rope_cached_thd_positions_2c_bwd,
    rope_cached_thd_positions_offsets_2c_bwd,
    # rope_fwd_2d,
    # rope_fwd_2d_inplace,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_configs,
    get_available_models,
)


def str_to_bool(v, vstr):
    if v.lower() in ["true", "yes"]:
        return True
    elif v.lower() in ["false", "no"]:
        return False
    else:
        raise NotImplementedError(f"invalid {vstr}: {v}")


def run_benchmark(args):
    (
        B,
        S,
        H,
        Q,
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
        bwd,
    ) = (
        args.B,
        args.S,
        args.H,
        args.Q,
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
        args.bwd,
    )
    if args.model:
        config_file = args.model_configs
        configs = get_model_configs(config_path=config_file, models=args.model)
        config = configs[args.model]
        num_q_heads = config["num_attention_heads"]
        num_kv_heads = config["num_key_value_heads"]
        H = num_kv_heads
        Q = num_q_heads // num_kv_heads  # num Q heads per K head
        D = (
            config["hidden_size"] // num_q_heads
        )  # head_dimension = hidden_size / num_heads

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
    bwd = str_to_bool(bwd, "inplace")

    Q = Q if two_inputs == True else 1  # noqa: E712
    is_mha = True if Q == 1 else False

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
        "HK",
        "HQ_per_HK",
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
        "bwd",
    ]
    x_vals_list = [
        (
            B,
            S,
            H,
            Q,
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
            bwd,
        )
    ]

    def bench_rope(
        B: int,
        S: int,
        H: int,
        Q: int,
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
        bwd: bool,
    ):
        x, y, gx, gy, freqs, positions, offsets, cos, sin = generate_rope_inputs(
            B,
            S,
            H,
            Q,
            D,
            cached,
            reuse_freqs_front_part,
            nope,
            pos,
            offs,
            two_inputs,
            layout,
            dtype,
            bwd,
        )
        if rotate_style == "gptj":
            rotate_style = RotateStyle.GPTJ
        elif rotate_style == "neox":
            rotate_style = RotateStyle.NEOX

        # flops
        flops = (
            B
            * S
            * H
            * D
            * (0.5 if nope else 1.0)
            * 3.0
            * 2.0
            * ((Q + 1) if two_inputs else 1.0)
        )

        # memory transfer (B = 1, T = S for thd layout, positions and offsets are always int)

        mem_freqs = (
            freqs.element_size()
            * S
            * D
            * (0.5 if reuse_freqs_front_part else 1.0)
            * (2.0 if cached else 1.0)
        )
        mem_pos = (positions.element_size() * B * S) if pos else 0.0
        mem_offs = (offsets.element_size() * B * S) if offs else 0.0
        mem_x = (
            x.element_size()
            * B
            * S
            * H
            * D
            * ((Q + 1) if two_inputs else 1.0)
            * (0.5 if nope and inplace and not bwd else 1.0)
        )
        mem = 2.0 * mem_x + mem_freqs + mem_pos + mem_offs

        transpose_output = False
        fn = None

        if two_inputs and cached and pos and layout == "thd":
            if offs:
                if bwd:
                    fn = lambda: rope_cached_thd_positions_offsets_2c_bwd(  # noqa: E731
                        gx,
                        gy,
                        cos,
                        sin,
                        positions,
                        offsets,
                        rotate_style,
                        reuse_freqs_front_part,
                        nope_first,
                        transpose_output,
                    )
                elif inplace:
                    fn = lambda: rope_cached_thd_positions_offsets_2c_fwd_inplace(  # noqa: E731
                        x,
                        y,
                        cos,
                        sin,
                        positions,
                        offsets,
                        rotate_style,
                        reuse_freqs_front_part,
                        nope_first,
                        transpose_output,
                    )
                else:
                    fn = lambda: rope_cached_thd_positions_offsets_2c_fwd(  # noqa: E731
                        x,
                        y,
                        cos,
                        sin,
                        positions,
                        offsets,
                        rotate_style,
                        reuse_freqs_front_part,
                        nope_first,
                        transpose_output,
                    )
            else:
                if bwd:
                    fn = lambda: rope_cached_thd_positions_2c_bwd(  # noqa: E731
                        gx,
                        gy,
                        cos,
                        sin,
                        positions,
                        rotate_style,
                        reuse_freqs_front_part,
                        nope_first,
                        transpose_output,
                    )
                elif inplace:
                    fn = lambda: rope_cached_thd_positions_2c_fwd_inplace(  # noqa: E731
                        x,
                        y,
                        cos,
                        sin,
                        positions,
                        rotate_style,
                        reuse_freqs_front_part,
                        nope_first,
                        transpose_output,
                    )
                else:
                    fn = lambda: rope_cached_thd_positions_2c_fwd(  # noqa: E731
                        x,
                        y,
                        cos,
                        sin,
                        positions,
                        rotate_style,
                        reuse_freqs_front_part,
                        nope_first,
                        transpose_output,
                    )

        if not two_inputs and cached and pos and offs:
            if layout == "sbhd":
                if bwd:
                    fn = lambda: rope_cached_positions_offsets_bwd(  # noqa: E731
                        gx,
                        cos,
                        sin,
                        positions,
                        offsets,
                        rotate_style,
                        reuse_freqs_front_part,
                        nope_first,
                        transpose_output,
                    )
                elif inplace:
                    fn = (  # noqa: E731
                        lambda: rope_cached_positions_offsets_fwd_inplace(  # noqa: E731
                            x,
                            cos,
                            sin,
                            positions,
                            offsets,
                            rotate_style,
                            reuse_freqs_front_part,
                            nope_first,
                            transpose_output,
                        )
                    )
                else:
                    fn = lambda: rope_cached_positions_offsets_fwd(  # noqa: E731
                        x,
                        cos,
                        sin,
                        positions,
                        offsets,
                        rotate_style,
                        reuse_freqs_front_part,
                        nope_first,
                        transpose_output,
                    )
            elif layout == "thd":
                pass

        if not two_inputs and cached and pos and not offs:
            if layout == "sbhd":
                if bwd:
                    fn = lambda: rope_cached_positions_bwd(  # noqa: E731
                        gx,
                        cos,
                        sin,
                        positions,
                        rotate_style,
                        reuse_freqs_front_part,
                        nope_first,
                        transpose_output,
                    )
                elif inplace:
                    fn = lambda: rope_cached_positions_fwd_inplace(  # noqa: E731
                        x,
                        cos,
                        sin,
                        positions,
                        rotate_style,
                        reuse_freqs_front_part,
                        nope_first,
                        transpose_output,
                    )
                else:
                    fn = lambda: rope_cached_positions_fwd(  # noqa: E731
                        x,
                        cos,
                        sin,
                        positions,
                        rotate_style,
                        reuse_freqs_front_part,
                        nope_first,
                        transpose_output,
                    )
            else:
                pass

        if not two_inputs and cached and not pos and not offs:
            if layout == "sbhd":
                if bwd:
                    fn = lambda: rope_cached_bwd(  # noqa: E731
                        gx,
                        cos,
                        sin,
                        positions,
                        rotate_style,
                        reuse_freqs_front_part,
                        nope_first,
                        transpose_output,
                    )
                elif inplace:
                    fn = lambda: rope_cached_fwd_inplace(  # noqa: E731
                        x,
                        cos,
                        sin,
                        rotate_style,
                        reuse_freqs_front_part,
                        nope_first,
                        transpose_output,
                    )
                else:
                    fn = lambda: rope_cached_fwd(  # noqa: E731
                        x,
                        cos,
                        sin,
                        positions,
                        rotate_style,
                        reuse_freqs_front_part,
                        nope_first,
                        transpose_output,
                    )
            else:
                pass

        if not two_inputs and not cached and not pos and not offs:
            if layout == "sbhd":
                if bwd:
                    fn = lambda: rope_bwd(  # noqa: E731
                        gx,
                        freqs,
                        rotate_style,
                        reuse_freqs_front_part,
                        nope_first,
                        transpose_output,
                    )
                elif inplace:
                    fn = lambda: rope_fwd_inplace(  # noqa: E731
                        x,
                        freqs,
                        rotate_style,
                        reuse_freqs_front_part,
                        nope_first,
                        transpose_output,
                    )
                else:
                    fn = lambda: rope_fwd(  # noqa: E731
                        x,
                        freqs,
                        rotate_style,
                        reuse_freqs_front_part,
                        nope_first,
                        transpose_output,
                    )
            elif layout == "thd":
                seqlens = [0, S]
                cu_seqlens = torch.Tensor(seqlens).to(torch.int).to(freqs.device)
                if bwd:
                    fn = lambda: rope_thd_bwd(  # noqa: E731
                        gx,
                        cu_seqlens,
                        freqs,
                        rotate_style,
                        reuse_freqs_front_part,
                        nope_first,
                        transpose_output,
                    )
                elif inplace:
                    fn = lambda: rope_thd_fwd_inplace(  # noqa: E731
                        x,
                        cu_seqlens,
                        freqs,
                        rotate_style,
                        reuse_freqs_front_part,
                        nope_first,
                        transpose_output,
                    )
                else:
                    fn = lambda: rope_thd_fwd(  # noqa: E731
                        x,
                        cu_seqlens,
                        freqs,
                        rotate_style,
                        reuse_freqs_front_part,
                        nope_first,
                        transpose_output,
                    )

        # TODO enable rope_fwd_2d and rope_fwd_2d_inplace

        if fn is None:
            raise NotImplementedError(
                f"No API with option: [layout='{layout}', cached={cached}, two_inputs={two_inputs}, pos={pos}, offs={offs}, inplace={inplace}, (HQ_per_HK > 1)={is_mha}]."
            )

        di = runtime.driver.active.get_device_interface()
        cache = runtime.driver.active.get_empty_cache_for_benchmark()
        for i in range(rep):
            cache.zero_()
            di.synchronize()
            fn()
            di.synchronize()

        return flops, mem

    for x_vals in x_vals_list:
        print("Running input config:")
        for i in range(len(x_names)):
            print(f"    {x_names[i]:23s}= {x_vals_list[0][i]}")
        print(f"Number of repitition = {rep}")
        flops, mem = bench_rope(*x_vals)
        print(f"Total flops  = {flops/1e12 : .6e} (TFLOPS)")
        print(f"Total memory = {mem/1e9 : .6e} (GB)")

    print("")
    print(
        "This script will not print out runtime as short running kernels cannot be measured accurately through triton.testing.do_bench function, please use rocprof to measure accurate runtime, use -h/--help for more information"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark RoPE",
        description="This script will not print out runtime as short running kernels cannot be measured accurately through triton.testing.do_bench function, please use rocprof to measure accurate runtime. For instance, try \"rocprofv2 --kernel-trace python bench_rope.py -l 'thd' -T 1 -H 128 -D 64 --two_inputs=true\"",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    available_models = get_available_models()  # Dynamically load model names
    model_help = (
        "Model name to benchmark. Select from: ["
        + ", ".join(available_models)
        + "]. Use 'all' to benchmark all models or leave blank for the default benchmark script."
    )
    parser.add_argument(
        "--model-configs",
        type=str,
        default="utils/model_configs.json",
        help="Model config json file.",
    )
    parser.add_argument("--model", type=str, help=model_help)
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
    parser.add_argument("-H", type=int, help="the number of K heads, H", default=128)
    parser.add_argument(
        "-Q",
        type=int,
        help="the number of Q heads per K heads, Q (default is 1, MHA implementation, this argument will be ignored if --two_inputs=false)",
        default=1,
    )
    parser.add_argument("-D", type=int, help="the head dimension, D", default=64)
    parser.add_argument("--cached", type=str, help="cached sin/cos", default="true")
    parser.add_argument("--rotate_style", type=str, help="gptj or neox", default="gptj")
    parser.add_argument(
        "--reuse_freqs_front_part",
        type=str,
        help="turn on reuse_freqs_front_part",
        default="true",
    )
    parser.add_argument("--nope", type=str, help="turn on nope", default="false")
    parser.add_argument(
        "--nope_first", type=str, help="turn on nope_fist", default="false"
    )
    parser.add_argument("--pos", type=str, help="input positions", default="true")
    parser.add_argument("--offs", type=str, help="input offsets", default="false")
    parser.add_argument(
        "--two_inputs", type=str, help="input both K and Q", default="true"
    )
    parser.add_argument(
        "--inplace",
        type=str,
        help="inplace operation (this argument will be ignored if bwd=true)",
        default="false",
    )
    parser.add_argument("--bwd", type=str, help="backward operation", default="false")
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
