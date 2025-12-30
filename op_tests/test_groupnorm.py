import random
import numpy as np
import torch
import argparse
from aiter.ops.groupnorm import GroupNorm

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


class GroupNormTimer:

    def __init__(self, num_groups, num_channels, device, dtype):
        self.norm = GroupNorm(
            num_groups, num_channels, eps=1e-6, affine=True, device=device, dtype=dtype
        )
        self.norm.weight = torch.nn.Parameter(
            torch.randn((num_channels,), dtype=dtype, device=device)
        )
        self.norm.bias = torch.nn.Parameter(
            torch.randn((num_channels,), dtype=dtype, device=device)
        )
        self.num_channels = num_channels
        self.device = device
        self.dtype = dtype

    @torch.inference_mode()
    def run_and_get_time(self, input_dims: list, print_tensors: bool = False):
        num_warmups = 5
        num_iters = 25

        assert len(input_dims) >= 3
        assert input_dims[1] == self.num_channels

        x = torch.randn(tuple(input_dims), dtype=self.dtype, device=self.device)
        if print_tensors:
            print("x :")
            print(x)

        with torch.no_grad():
            for _ in range(num_warmups):
                y = self.norm(x, use_torch=True)
        e_start = torch.cuda.Event(enable_timing=True)
        e_end = torch.cuda.Event(enable_timing=True)
        e_start.record()
        with torch.no_grad():
            for _ in range(num_iters):
                y = self.norm(x, use_torch=True)
        e_end.record()
        e_end.synchronize()
        time_elapsed_torch = e_start.elapsed_time(e_end) / num_iters
        if print_tensors:
            print("y :")
            print(y)

        for _ in range(num_warmups):
            z = self.norm(x, use_torch=False)
        e_start = torch.cuda.Event(enable_timing=True)
        e_end = torch.cuda.Event(enable_timing=True)
        e_start.record()
        for _ in range(num_iters):
            z = self.norm(x, use_torch=False)
        e_end.record()
        e_end.synchronize()
        time_elapsed_opt = e_start.elapsed_time(e_end) / num_iters
        if print_tensors:
            print("z :")
            print(z)

        is_equal = torch.allclose(y, z, rtol=1e-3, atol=1e-2)
        return (time_elapsed_torch, time_elapsed_opt, is_equal)


def str2tuple(s):
    return tuple(map(int, s.split(",")))


def main(args):
    torch.set_printoptions(precision=6)

    device = torch.device(args.device)
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    bench_shapes = args.bench_shapes

    speedups = []
    for shape in bench_shapes:
        norm_timer = GroupNormTimer(shape[0], shape[2], device, dtype)
        torch_time, opt_time, is_equal = norm_timer.run_and_get_time(
            shape[1:], print_tensors=args.print_tensors
        )
        speedup = torch_time / opt_time if opt_time > 0 else float("inf")
        speedups.append(speedup)

        print(
            "shape={} torch_time={:.4f} ms, opt_time={:.4f} ms, speedup={:.4f} is_equal={}".format(
                shape, torch_time, opt_time, speedup, is_equal
            ),
            flush=True,
        )

    print("\n=== Performance Summary ===")
    print("Speedups with all shapes, including batch_size > 1 and odd hw values:")
    print(f"Average speedup: {np.mean(speedups):.4f}", flush=True)
    print(f"Median speedup: {np.median(speedups):.4f}", flush=True)

    if len(speedups) > 6:
        print("\nSpeedups with batch_size == 1 only")
        speedups_batch1 = speedups[:-6]
        print(f"Average speedup: {np.mean(speedups_batch1):.4f}", flush=True)
        print(f"Median speedup: {np.median(speedups_batch1):.4f}", flush=True)


if __name__ == "__main__":
    default_bench_shapes = [
        [1, 1, 1, 2],
        [4, 1, 4, 4],
        [8, 1, 512, 1728],
        [16, 1, 128, 9, 144, 256],
        [32, 1, 512, 1728],
        [32, 1, 512, 5120],
        [32, 1, 128, 9, 144, 256],
        [32, 1, 128, 17, 256, 128],
        [32, 1, 128, 17, 256, 256],
        [32, 1, 256, 9, 128, 128],
        [32, 1, 256, 9, 128, 256],
        [32, 1, 256, 17, 144, 256],
        [32, 1, 256, 17, 256, 256],
        [32, 1, 512, 3, 18, 32],
        [32, 1, 512, 3, 64, 64],
        [32, 1, 512, 5, 32, 32],
        [32, 1, 512, 5, 64, 64],
        [32, 1, 512, 9, 128, 128],
        [32, 4, 256, 17, 144, 256],
        [32, 7, 512, 3, 18, 32],
        [32, 3, 512, 5, 64, 64],
        [16, 3, 256, 5, 7, 11],
        [16, 1, 32, 15, 17, 11],
        [16, 5, 32, 2, 5, 3],
    ]

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="GroupNorm Performance Benchmark",
    )

    parser.add_argument(
        "-b",
        "--bench_shapes",
        nargs="*",
        type=str2tuple,
        default=default_bench_shapes,
        help="""Benchmark shapes in format [num_groups, batch_size, num_channels, ...]
        e.g.: -b 1,1,1,2 4,1,4,4 8,1,512,1728""",
    )

    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run benchmark on (default: cuda)",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Data type to use (default: float16)",
    )

    parser.add_argument(
        "--print_tensors",
        action="store_true",
        help="Print input and output tensors for debugging",
    )

    parser.add_argument(
        "--num_warmups",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)",
    )

    parser.add_argument(
        "--num_iters",
        type=int,
        default=25,
        help="Number of measurement iterations (default: 25)",
    )

    args = parser.parse_args()

    print("=== GroupNorm Performance Benchmark ===", flush=True)
    print(f"Device: {args.device}", flush=True)
    print(f"Data type: {args.dtype}", flush=True)
    print(f"Number of warmups: {args.num_warmups}", flush=True)
    print(f"Number of iterations: {args.num_iters}", flush=True)
    print(f"Number of benchmark shapes: {len(args.bench_shapes)}", flush=True)
    print("=" * 50, flush=True)

    main(args)

    print("\n=== Benchmark Complete ===", flush=True)
