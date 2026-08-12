# SPDX-License-Identifier: MIT

"""Fresh-process gfx950 cross-compile probe for unified decode policies."""

from __future__ import annotations

import argparse

from aiter.aot.flydsl.gemm import compile_one_config
from aiter.ops.flydsl.kernels.gemm_decode_config import (
    ActivationSource,
    BlockMfmaDecodeConfig,
    ContractionMode,
    WaveDecodeConfig,
    gemm_decode_kernel_name,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        choices=("wave", "block", "block_boundary", "persistent_block"),
        required=True,
    )
    args = parser.parse_args()

    if args.policy == "wave":
        m, n, k = 3, 2304, 1536
        config = WaveDecodeConfig(
            m_per_wave=m,
            n_per_wave=4,
            kvec=8,
            prefetch_depth=2,
            waves_per_eu=2,
            contraction=ContractionMode.DOT2_BF16,
        )
    elif args.policy == "block":
        m, n, k = 3, 6288, 7168
        config = BlockMfmaDecodeConfig(
            waves_per_workgroup=8,
            columns_per_wave=2,
            b_load_width=8,
            k_unroll=2,
            prefetch_stages=2,
            waves_per_eu=2,
        )
    elif args.policy == "block_boundary":
        m, n, k = 4, 6288, 7168
        config = BlockMfmaDecodeConfig(
            waves_per_workgroup=8,
            columns_per_wave=4,
            b_load_width=8,
            k_unroll=2,
            prefetch_stages=1,
            waves_per_eu=2,
        )
    else:
        m, n, k = 3, 2304, 1536
        config = BlockMfmaDecodeConfig(
            waves_per_workgroup=8,
            columns_per_wave=1,
            activation_source=ActivationSource.FULL_LDS,
            b_load_width=8,
            k_unroll=2,
            persistent_n=True,
            workgroups_per_cu=1,
        )

    result = compile_one_config(
        gemm_decode_kernel_name("gfx950", m, n, k, config),
        "decode",
        m,
        n,
        k,
        cu_num=256,
        arch="gfx950",
        config=config,
    )
    assert result["compile_time"] is not None
    print(f"GFX950_DECODE_COMPILE_OK policy={args.policy}")


if __name__ == "__main__":
    main()
