# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse

from aiter import logger
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.attention.test_kda_one_token import (
    make_one_token_inputs,
    run_one_token,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compare public KDA decode paths on gfx1250."
    )
    parser.add_argument("--tokens", type=int, default=72)
    parser.add_argument("--pool-size", type=int, nargs="+", default=[1, 16])
    args = parser.parse_args()
    for size in args.pool_size:
        bencher = RotatingBufferBench.from_dict(
            {"min_pool_len": size, "max_pool_len": size, "kernels_per_graph": 64}
        )
        bencher.build_pool(
            [lambda: (make_one_token_inputs(args.tokens, padded=True),)], [["inputs"]]
        )
        generic_ms = bencher.bench(
            lambda inputs: run_one_token(inputs, config={"use_one_token_decode": False})
        )
        one_token_ms = bencher.bench(lambda inputs: run_one_token(inputs))
        logger.info(
            "tokens=%d pool=%d generic_us=%.4f one_token_us=%.4f speedup=%.4f",
            args.tokens,
            size,
            generic_ms * 1000,
            one_token_ms * 1000,
            generic_ms / one_token_ms,
        )


if __name__ == "__main__":
    main()
