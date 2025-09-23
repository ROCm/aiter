import sys
import random
import numpy as np

import torch
from torch.profiler import ProfilerActivity

from aiter.ops.groupnorm import GroupNorm
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["TORCH_CPP_EXTENSION_WARNING_LEVEL"] = "2"

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

DEVICE = torch.device("cuda")
DTYPE = torch.float16


class GroupNormTimer:

    def __init__(self, num_groups, num_channels):
        self.norm = GroupNorm(
            num_groups, num_channels, eps=1e-6, affine=True, device=DEVICE, dtype=DTYPE
        )
        self.norm.weight = torch.nn.Parameter(
            torch.randn((num_channels,), dtype=DTYPE, device=DEVICE)
        )
        self.norm.bias = torch.nn.Parameter(
            torch.randn((num_channels,), dtype=DTYPE, device=DEVICE)
        )
        self.num_channels = num_channels

    @torch.inference_mode()
    def run_and_get_time(self, input_dims: list, print_tensors: bool = False):
        num_warmups = 5
        num_iters = 25

        assert len(input_dims) >= 3
        assert input_dims[1] == self.num_channels

        x = torch.randn(tuple(input_dims), dtype=DTYPE, device=DEVICE)
        if print_tensors:
            print("x :")
            print(x)

        # torch
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

        # opt
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


def main():
    torch.set_printoptions(precision=6)
    bench_shapes = [
        [1, 1, 1, 2],  # [num_groups, n, c, ...]
        [4, 1, 4, 4],
        [8, 1, 512, 1728],
        [16, 1, 128, 9, 144, 256],
        [32, 1, 512, 1728],
        [32, 1, 512, 5120],
        [32, 1, 128, 9, 144, 256],
        [32, 1, 128, 17, 256, 128],
        [32, 1, 128, 17, 256, 256],
        [32, 1, 256, 9, 128, 128],
        [32, 1, 256, 9, 144, 256],
        [32, 1, 256, 17, 144, 256],
        [32, 1, 256, 17, 256, 256],
        [32, 1, 512, 3, 18, 32],
        [32, 1, 512, 3, 64, 64],
        [32, 1, 512, 5, 32, 32],
        [32, 1, 512, 5, 64, 64],
        [32, 1, 512, 9, 128, 128],
        # add extra cases with batch_size>1
        [32, 4, 256, 17, 144, 256],
        [32, 7, 512, 3, 18, 32],
        [32, 3, 512, 5, 64, 64],
        # add extra cases with prime or odd hw value
        [16, 3, 256, 5, 7, 11],
        [16, 1, 32, 15, 17, 11],
        [16, 5, 32, 2, 5, 3],
    ]

    speedups = []
    for shape in bench_shapes:
        norm_timer = GroupNormTimer(shape[0], shape[2])
        torch_time, opt_time, is_equal = norm_timer.run_and_get_time(
            shape[1:], print_tensors=False
        )
        speedup = torch_time / opt_time if opt_time > 0 else float("inf")
        speedups.append(speedup)

        print(
            "shape={} torch_time={:.4f} ms, opt_time={:.4f} ms, speedup={:.4f} is_equal={}".format(
                shape, torch_time, opt_time, speedup, is_equal
            ),
            flush=True,
        )

    print("Speedups with all shapes, including batch_size > 1 and odd hw values:")
    print("Average speedup: {:.4f}".format(np.mean(speedups)), flush=True)
    print("Median speedup: {:.4f}".format(np.median(speedups)), flush=True)

    print("Speedups with batch_size == 1 only")
    speedups = speedups[:-6]
    print("Average speedup: {:.4f}".format(np.mean(speedups)), flush=True)
    print("Median speedup: {:.4f}".format(np.median(speedups)), flush=True)


if __name__ == "__main__":
    print("main start", flush=True)
    main()
    print("main end", flush=True)
