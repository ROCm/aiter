# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compare ATOM's gating entry against the single-index KDA paths on gfx1250.

Modes:
  one_token   existing single-index one-token kernel (baseline)
  generic     packed kernel, one-token dispatch disabled
  atom_same   fused_sigmoid_gating_delta_rule_update, same tensor for read/write
  atom_split  same slot values, but read and write are distinct tensors
  atom_fork   read slots and write slots are disjoint (state fork)
"""

import argparse

import torch

from aiter.ops.triton.attention.kda_atom import fused_sigmoid_gating_delta_rule_update
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.attention.test_kda_one_token import (
    make_one_token_inputs,
    run_one_token,
)

H, K = 12, 128


def make_inputs(tokens, mode):
    base = make_one_token_inputs(tokens, padded=False)
    write = torch.arange(tokens, dtype=torch.int32, device="cuda")
    if mode == "fork":
        read = write + tokens
    elif mode == "atom_split":
        read = write.clone()
    else:
        read = write
    base["initial_state"] = torch.randn(
        (2 * tokens, H, K, K), dtype=torch.float32, device="cuda"
    )
    base["ssm_state_indices"] = write
    if mode not in ("one_token", "generic"):
        # The single-index baselines take no read table.
        base["ssm_state_indices_in"] = read
    return base


def run_atom(inputs):
    return fused_sigmoid_gating_delta_rule_update(
        A_log=inputs["A_log"],
        a=inputs["g"],
        b=inputs["beta"],
        dt_bias=inputs["dt_bias"],
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        o=inputs["out"],
        initial_state=inputs["initial_state"],
        cu_seqlens=inputs["cu_seqlens"],
        ssm_state_indices=inputs["ssm_state_indices"],
        ssm_state_indices_in=inputs["ssm_state_indices_in"],
        use_qk_l2norm_in_kernel=True,
        is_kda=True,
        lower_bound=-5.0,
    )


def run_mode(inputs, mode):
    if mode == "one_token":
        return run_one_token(inputs)
    if mode == "generic":
        return run_one_token(inputs, config={"use_one_token_decode": False})
    return run_atom(inputs)


def main():
    parser = argparse.ArgumentParser(
        description="Compare KDA decode entries on gfx1250 (graph replay, rotating state)."
    )
    parser.add_argument("--tokens", type=int, default=72)
    parser.add_argument("--pool-size", type=int, nargs="+", default=[1, 16])
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["one_token", "generic", "atom_same", "atom_split", "atom_fork"],
    )
    args = parser.parse_args()
    for size in args.pool_size:
        results = {}
        for mode in args.modes:
            bencher = RotatingBufferBench.from_dict(
                {"min_pool_len": size, "max_pool_len": size, "kernels_per_graph": 64}
            )
            bencher.build_pool(
                [lambda mode=mode: (make_inputs(args.tokens, mode),)], [["inputs"]]
            )
            results[mode] = (
                bencher.bench(lambda inputs, mode=mode: run_mode(inputs, mode)) * 1000
            )
            torch.cuda.empty_cache()
        base = results.get("one_token")
        for mode in args.modes:
            ratio = "" if base is None else f" vs_one_token={base / results[mode]:.3f}x"
            print(
                f"tokens={args.tokens} pool={size} mode={mode:<11} "
                f"us={results[mode]:.3f}{ratio}",
                flush=True,
            )


if __name__ == "__main__":
    main()
