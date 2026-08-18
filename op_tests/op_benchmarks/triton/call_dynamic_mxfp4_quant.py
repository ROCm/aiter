import argparse
import sys

import torch

from aiter.ops.triton.quant import dynamic_mxfp4_quant


def get_default_shapes() -> list[list[int]]:
    return [
        # [8, 1024]
        # [8, 3072]
        # [8, 7168]
        # [32, 1024]
        # [32, 3072]
        # [32, 7168]
        # [256, 1024]
        # [256, 3072]
        # [256, 7168]
        # [2048, 1024]
        # [2048, 3072]
         [2048, 7168]
        # [8192, 1024]
        # [8192, 3072]
        # [8192, 7168]
        # [16384, 1024]
        # [16384, 3072]
        # [16384, 7168]
    ]


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        prog="Call dynamic_mxfp4_quant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--M", type=int, default=128, help="Rows of input tensor")
    parser.add_argument("--N", type=int, default=64, help="Cols of input tensor")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16"],
        default="bf16",
        help="Input dtype",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible input",
    )
    return parser.parse_args(args=args)


def get_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def main(args=None):
    parsed = parse_args(args)
    torch.manual_seed(parsed.seed)
    dtype = get_dtype(parsed.dtype)

    print("dynamic_mxfp4_quant calls:")
    for m, n in get_default_shapes():
        if n % 2 != 0:
            raise ValueError(f"N must be even because FP4 packs 2 values per byte, got N={n}")

        x = torch.randn((m, n), dtype=dtype, device="cuda")
        x_fp4, x_scale = dynamic_mxfp4_quant(x)

        print(f"shape=({m}, {n})")
        print(f"  input  : {tuple(x.shape)}, {x.dtype}, {x.device}")
        print(f"  x_fp4  : {tuple(x_fp4.shape)}, {x_fp4.dtype}, {x_fp4.device}")
        print(f"  x_scale: {tuple(x_scale.shape)}, {x_scale.dtype}, {x_scale.device}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
