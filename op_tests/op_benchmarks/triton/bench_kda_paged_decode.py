# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse

from aiter import logger
from aiter.ops.triton.attention.kda import fused_recurrent_kda
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.attention.test_kda_paged_decode import (
    make_paged_decode_inputs,
)


def main():
    parser = argparse.ArgumentParser(description="Full single-token paged KDA decode")
    parser.add_argument("--num-seqs", type=int, default=72)
    parser.add_argument("--rotation", type=int, choices=(1, 16), default=1)
    args = parser.parse_args()
    bencher = RotatingBufferBench.from_dict(
        {
            "min_pool_len": args.rotation,
            "max_pool_len": args.rotation,
            "kernels_per_graph": 64,
            "num_replays": 9,
        }
    )
    bencher.build_pool(
        inp_gen_rot_fns=[
            lambda: (make_paged_decode_inputs(args.num_seqs, negative_slot=False),)
        ],
        inp_rot_keys=[["inputs"]],
    )
    latency_ms = bencher.bench(lambda inputs: fused_recurrent_kda(**inputs))
    logger.info(
        "num_seqs=%s rotation=%s graph_us_per_call=%s",
        args.num_seqs,
        args.rotation,
        latency_ms * 1000,
    )


if __name__ == "__main__":
    main()
