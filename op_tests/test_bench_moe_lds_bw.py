# SPDX-License-Identifier: Apache-2.0

"""Quick smoke-test for bench_moe_lds_bw: verify TDM global→LDS loads run without error.

Usage:
    python op_tests/test_bench_moe_lds_bw.py [--dtype f16] [--experts 2] [--inter_dim 128] \
        [--model_dim 64] [--tile_n 64] [--tile_k 32] [--num_stages 2]

    # Run all built-in configs:
    python op_tests/test_bench_moe_lds_bw.py --all
"""

import argparse
import sys
import os
import traceback

import torch

torch.set_default_device("cuda")

_kernel_dir = os.path.join(os.path.dirname(__file__), "..", "aiter", "ops", "flydsl")
if _kernel_dir not in sys.path:
    sys.path.insert(0, _kernel_dir)

from kernels.bench_moe_lds_bw import _compile_lds_bw_kernel, _DTYPE_CFG


_ALL_CONFIGS = [
    # (experts, inter_dim, model_dim, tile_n, tile_k, num_stages, dtype)
    (1, 64, 64,   64, 64, 1, "f16"),
    (2, 128, 64,  64, 32, 1, "f16"),
    (2, 128, 64,  64, 32, 2, "f16"),
    (2, 128, 64,  64, 32, 2, "bf16"),
    (2, 128, 64,  64, 32, 2, "fp8"),
    (2, 128, 64,  64, 32, 2, "mxfp4"),
    (4, 128, 128, 64, 64, 2, "f16"),
    (2, 192, 192, 64, 64, 3, "f16"),
]


def run_one(experts, inter_dim, model_dim, tile_n, tile_k, num_stages, dtype):
    tag = f"E{experts}_I{inter_dim}_K{model_dim}_tn{tile_n}_tk{tile_k}_s{num_stages}_{dtype}"
    elem_bytes, pack_factor, torch_dtype = _DTYPE_CFG[dtype]
    packed_k = model_dim // pack_factor

    w = torch.ones(experts, inter_dim, packed_k, dtype=torch.uint8).cuda()
    sink = torch.zeros(4 * 32, dtype=torch.float16)

    exe = _compile_lds_bw_kernel(
        model_dim=model_dim, inter_dim=inter_dim, experts=experts,
        tile_n=tile_n, tile_k=tile_k,
        num_stages=num_stages, dtype=dtype, waves_per_eu=None,
    )
    exe(w, sink, torch.cuda.current_stream())
    torch.cuda.synchronize()

    non_zero = (sink != 0).sum().item()
    nan_count = torch.isnan(sink).sum().item()
    print(f"  [{tag}] non_zero={non_zero}/{sink.numel()}  nan={nan_count}  OK")
    return True


def main():
    p = argparse.ArgumentParser(description="Smoke-test for bench_moe_lds_bw")
    p.add_argument("--all", action="store_true", help="Run all built-in configs")
    p.add_argument("--experts", type=int, default=2)
    p.add_argument("--inter_dim", type=int, default=128)
    p.add_argument("--model_dim", type=int, default=64)
    p.add_argument("--tile_n", type=int, default=64)
    p.add_argument("--tile_k", type=int, default=32)
    p.add_argument("--num_stages", type=int, default=2)
    p.add_argument("--dtype", type=str, default="f16", choices=list(_DTYPE_CFG.keys()))
    args = p.parse_args()

    if args.all:
        configs = _ALL_CONFIGS
    else:
        configs = [(args.experts, args.inter_dim, args.model_dim,
                     args.tile_n, args.tile_k, args.num_stages, args.dtype)]

    passed, failed = 0, 0
    for cfg in configs:
        try:
            run_one(*cfg)
            passed += 1
        except Exception:
            tag = f"E{cfg[0]}_I{cfg[1]}_K{cfg[2]}_tn{cfg[3]}_tk{cfg[4]}_s{cfg[5]}_{cfg[6]}"
            print(f"  [{tag}] FAIL")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(configs)} configs")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
