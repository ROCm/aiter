# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Check that FP8 kernels do not prevent supported FP16/BF16 custom GEMMs."""

import argparse

import torch

import aiter
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.test_common import checkAllclose, perftest


@perftest(num_iters=50, num_rotate_args=1, use_cuda_event=True)
def benchmark_custom(
    weight: torch.Tensor, x: torch.Tensor, out: torch.Tensor, cu_num: int
) -> torch.Tensor:
    aiter.wv_splitk_small_fp16_bf16(weight, x, out, x.shape[0], cu_num)
    return out


def test_custom_gemm(dtype: torch.dtype, m: int, n: int, k: int, timing: bool) -> None:
    x = torch.randint(-1, 2, (m, k), device="cuda").to(dtype)
    weight = torch.randint(-1, 2, (n, k), device="cuda").to(dtype)
    out = torch.full((m, n), float("nan"), device="cuda", dtype=dtype)
    ref = (x.float() @ weight.float().t()).to(dtype)
    # All entry points share module_custom, including the unrelated FP8 kernels.
    # Use the supported 16-bit entry point directly, without tuned CSV dispatch.
    aiter.wv_splitk_small_fp16_bf16(weight, x, out, m, get_cu_num())
    torch.cuda.synchronize()
    error = checkAllclose(ref, out, rtol=0, atol=0, printLog=False)
    assert error == 0, f"Incorrect custom GEMM: {dtype}, {(m, n, k)}, {error=}"
    print(f"PASS {dtype} {(m, n, k)}", flush=True)
    if timing:
        _, latency = benchmark_custom(weight, x, out, get_cu_num())
        print(f"BENCH {dtype} {(m, n, k)}: {latency:.3f} us", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    gfx = get_gfx_runtime()
    if gfx not in ("gfx90a", "gfx942", "gfx950"):
        print(f"SKIP: custom 16-bit GEMM kernels are not supported on {gfx}")
    else:
        torch.manual_seed(42)
        for dtype in (torch.float16, torch.bfloat16):
            for shape in ((16, 96, 256), (1, 128, 256), (4, 96, 512), (8, 96, 5120)):
                test_custom_gemm(dtype, *shape, timing=args.benchmark)
        print(f"custom GEMM architecture regression: PASS ({gfx})", flush=True)
