import sys
import os
import argparse
# !!!!!!!!!!!!!!!! never import aiter
# from aiter.jit import core
this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{this_dir}/../../../aiter/")
from jit.core import compile_ops

@compile_ops("libmha_fwd", fc_name="compile_mha_fwd")
def compile_mha_fwd(): ...


@compile_ops("libmha_bwd", fc_name="compile_mha_bwd")
def compile_mha_bwd(): ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="compile",
        description="compile C++ instance with torch excluded",
    )
    parser.add_argument(
        "--api",
        default="",
        required=False,
        help="supply API(s) to generate (default: fwd). separated by comma."
    )

    args = parser.parse_args()

    if args.api == "fwd":
        print(";;;")
        compile_mha_fwd()
    elif args.api == "bwd":
        compile_mha_bwd()
    elif args.api == "":
        compile_mha_fwd()
        compile_mha_bwd()
    else:
        raise ValueError("Invalid input value: only support 'fwd', 'bwd' or default to be ''")


