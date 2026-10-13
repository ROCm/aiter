# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Profile a selected kernel; only its case imports GPU and operator modules."""

import argparse
from functools import partial
from pathlib import Path

KERNEL_CONFIG_NAMES = {
    "batched_gemm_bf16": "BATCHED_GEMM-A16W16",
    "gemm_a16w16": "GEMM-A16W16",
    "gemm_a16w16_atomic": "GEMM-A16W16-ATOMIC",
    "gemm_a16w16_gated": "GEMM-A16W16-gated",
    "gemm_a16w8_blockscale": "GEMM-A16W8_BLOCKSCALE",
    "gemm_a16w8_blockscale_preshuffle": "GEMM-A16W8_BLOCKSCALE_PRESHUFFLED",
    "gemm_a16wfp4": "GEMM-A16WFP4",
    "gemm_a8w8": "GEMM-A8W8",
    "gemm_a8w8_blockscale": "GEMM-A8W8_BLOCKSCALE",
    "gemm_a8w8_blockscale_preshuffle": "GEMM-A8W8_BLOCKSCALE_PRESHUFFLED",
    "gemm_a8w8_per_token_scale": "GEMM-A8W8_PER_TOKEN_SCALE",
    "gemm_a8wfp4": "GEMM-A8WFP4",
    "gemm_afp4wfp4": "GEMM-AFP4WFP4",
    "gemm_afp4wfp4_pre_quant_atomic": "GEMM-A16WFP4",
    "gemm_afp4wfp4_preshuffle": "GEMM-AFP4WFP4_PRESHUFFLED",
    "gemm_afp8wfp8_preshuffle": "GEMM-AFP8WFP8_PRESHUFFLED",
}


def kernel_name(value):
    """Accept kernel names and legacy harness filenames in the tuning CLIs."""
    name = Path(value).stem.removeprefix("harness_")
    if name not in KERNEL_CONFIG_NAMES:
        raise argparse.ArgumentTypeError(
            f"Unknown kernel {value!r}; choose from {list(KERNEL_CONFIG_NAMES)}"
        )
    return name


def get_kernel_runner(kernel, input_shape):
    """Create inputs once and bind them to the selected kernel's config argument."""
    M, N, K = input_shape

    match kernel:
        case "batched_gemm_bf16":
            import torch

            from aiter.ops.triton.gemm.batched.batched_gemm_bf16 import (
                batched_gemm_bf16,
            )
            from op_tests.triton_tests.gemm.batched.test_batched_gemm_bf16 import (
                generate_batched_gemm_a16w16_inputs,
            )

            dtype = torch.bfloat16
            B = 8 if K == 4096 else 16
            x, w, bias, y = generate_batched_gemm_a16w16_inputs(
                B, M, N, K, dtype, output=True
            )
            return partial(batched_gemm_bf16, x, w, bias, dtype, YQ=y)

        case "gemm_a16w16":
            import torch

            from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
            from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import (
                generate_gemm_a16w16_inputs,
            )

            dtype = torch.bfloat16
            x, w, bias, _, y = generate_gemm_a16w16_inputs(
                M, N, K, dtype, output=True, bias=True
            )
            return partial(gemm_a16w16, x, w, bias, dtype, y)

        case "gemm_a16w16_atomic":
            import torch

            from aiter.ops.triton.gemm.basic.gemm_a16w16_atomic import (
                gemm_a16w16_atomic,
            )
            from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import (
                generate_gemm_a16w16_inputs,
            )

            dtype = torch.bfloat16
            x, w, _, _, y = generate_gemm_a16w16_inputs(M, N, K, dtype, output=True)

            def run(config):
                y.zero_()
                gemm_a16w16_atomic(x, w, dtype, y, config=config)

            return run

        case "gemm_a16w16_gated":
            import torch

            from aiter.ops.triton.gemm.basic.gemm_a16w16_gated import gemm_a16w16_gated
            from op_tests.triton_tests.gemm.basic.test_gemm_a16w16_gated import (
                generate_gemm_a16w16_gated_inputs,
            )

            dtype = torch.bfloat16
            x, w, _, y = generate_gemm_a16w16_gated_inputs(M, N, K, dtype, output=True)
            return partial(gemm_a16w16_gated, x, w, dtype, y)

        case "gemm_a16w8_blockscale" | "gemm_a16w8_blockscale_preshuffle":
            import torch

            from aiter.ops.triton.gemm.basic.gemm_a16w8_blockscale import (
                gemm_a16w8_blockscale,
                gemm_a16w8_blockscale_preshuffle,
            )
            from op_tests.triton_tests.gemm.basic.test_gemm_a16w8_blockscale import (
                generate_gemm_a16w8_blockscale_inputs,
            )

            dtype = torch.bfloat16
            shuffle = kernel == "gemm_a16w8_blockscale_preshuffle"
            gemm = (
                gemm_a16w8_blockscale_preshuffle if shuffle else gemm_a16w8_blockscale
            )
            x, _, w, w_scale, y = generate_gemm_a16w8_blockscale_inputs(
                M, N, K, 128, 128, dtype=dtype, output=True, shuffle=shuffle
            )
            return partial(gemm, x, w, w_scale, dtype, y, prequant=False)

        case "gemm_a16wfp4":
            import torch

            from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
            from op_tests.triton_tests.gemm.basic.test_gemm_a16wfp4 import (
                generate_gemm_a16wfp4_inputs,
            )

            dtype = torch.bfloat16
            x, w, _, _, w_scales, _, y = generate_gemm_a16wfp4_inputs(
                M,
                N,
                K,
                output=True,
                atomic_add=False,
                dtype=dtype,
                layout="TN",
                shuffle=False,
            )
            return partial(gemm_a16wfp4, x, w, w_scales, False, dtype, y)

        case "gemm_a8w8":
            import torch

            from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8
            from aiter.ops.triton.utils.types import get_fp8_dtypes
            from op_tests.triton_tests.gemm.basic.test_gemm_a8w8 import (
                generate_gemm_a8w8_inputs,
            )

            _, fp8_dtype = get_fp8_dtypes()
            dtype = torch.bfloat16
            x, _, w, x_scale, w_scale, _, y = generate_gemm_a8w8_inputs(
                M, N, K, in_dtype=fp8_dtype, out_dtype=dtype, layout="TN", output=True
            )
            return partial(gemm_a8w8, x, w, x_scale, w_scale, None, dtype, y)

        case "gemm_a8w8_blockscale" | "gemm_a8w8_blockscale_preshuffle":
            import torch

            from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
                gemm_a8w8_blockscale,
                gemm_a8w8_blockscale_preshuffle,
            )
            from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_blockscale import (
                generate_gemm_a8w8_blockscale_inputs,
            )

            dtype = torch.bfloat16
            shuffle = kernel == "gemm_a8w8_blockscale_preshuffle"
            gemm = gemm_a8w8_blockscale_preshuffle if shuffle else gemm_a8w8_blockscale
            x, _, w, _, x_scale, w_scale, y = generate_gemm_a8w8_blockscale_inputs(
                M,
                N,
                K,
                128,
                128,
                dtype=dtype,
                layout="TN",
                output=True,
                shuffle=shuffle,
            )
            return partial(gemm, x, w, x_scale, w_scale, dtype, y)

        case "gemm_a8w8_per_token_scale":
            import torch

            from aiter.ops.triton.gemm.basic.gemm_a8w8_per_token_scale import (
                gemm_a8w8_per_token_scale,
            )
            from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_per_token_scale import (
                generate_gemm_a8w8_per_token_scale_inputs,
            )

            dtype = torch.bfloat16
            x, w, x_scale, w_scale, y = generate_gemm_a8w8_per_token_scale_inputs(
                M, N, K, dtype=dtype, layout="TN", output=True
            )
            return partial(gemm_a8w8_per_token_scale, x, w, x_scale, w_scale, dtype, y)

        case "gemm_a8wfp4":
            import torch

            from aiter.ops.triton.gemm.basic.gemm_a8wfp4 import gemm_a8wfp4
            from aiter.ops.triton.utils.types import get_fp8_dtypes
            from op_tests.triton_tests.gemm.basic.test_gemm_a8wfp4 import (
                generate_gemm_a8wfp4_inputs,
            )

            _, fp8_dtype = get_fp8_dtypes()
            dtype = torch.float16
            x, w, x_scales, w_scales, _, _, y = generate_gemm_a8wfp4_inputs(
                M, N, K, fp8_dtype, dtype, layout="TN", output=True
            )
            return partial(gemm_a8wfp4, x, w, y, x_scales, w_scales, dtype)

        case "gemm_afp4wfp4" | "gemm_afp4wfp4_preshuffle":
            import torch

            from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import (
                gemm_afp4wfp4,
                gemm_afp4wfp4_preshuffle,
            )
            from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import (
                generate_gemm_afp4wfp4_inputs,
            )

            dtype = torch.bfloat16
            shuffle = kernel == "gemm_afp4wfp4_preshuffle"
            gemm = gemm_afp4wfp4_preshuffle if shuffle else gemm_afp4wfp4
            x, _, w, _, _, x_scales, w_scales, _, y = generate_gemm_afp4wfp4_inputs(
                M,
                N,
                K,
                dtype,
                output=True,
                shuffle_scales_fg=shuffle,
                shuffle_weight_fg=shuffle,
            )
            return partial(gemm, x, w, x_scales, w_scales, dtype, y)

        case "gemm_afp4wfp4_pre_quant_atomic":
            import torch

            from aiter.ops.triton.gemm.basic.gemm_afp4wfp4_pre_quant_atomic import (
                gemm_afp4wfp4_pre_quant,
            )
            from op_tests.triton_tests.gemm.basic.test_gemm_a16wfp4 import (
                generate_gemm_a16wfp4_inputs,
            )

            dtype = torch.float32
            x, w, _, _, w_scales, _, y = generate_gemm_a16wfp4_inputs(
                M,
                N,
                K,
                output=True,
                atomic_add=True,
                dtype=dtype,
                layout="TN",
                shuffle=False,
            )
            return partial(gemm_afp4wfp4_pre_quant, x, w, w_scales, dtype, y)

        case "gemm_afp8wfp8_preshuffle":
            import torch

            from aiter.ops.triton.gemm.basic.gemm_afp8wfp8 import (
                gemm_afp8wfp8_preshuffle,
            )
            from aiter.ops.triton.utils.types import get_fp8_dtypes
            from op_tests.triton_tests.gemm.basic.test_gemm_afp8wfp8 import (
                generate_inputs,
            )

            get_fp8_dtypes()
            dtype = torch.bfloat16
            x, _, w, x_scales, w_scales = generate_inputs(M, N, K, shuffle=True)
            return partial(
                gemm_afp8wfp8_preshuffle, x, w, x_scales, w_scales, dtype=dtype
            )

        case _:
            raise ValueError(f"Unknown kernel: {kernel}")


def _prepare_config(kernel, K, config):
    """Apply kernel-specific config adjustments before the timed call."""
    if config is None:
        return None

    match kernel:
        case "batched_gemm_bf16" | "gemm_a16w16" | "gemm_a16w16_atomic" | "gemm_a8wfp4":
            import triton

            config = config.copy()
            config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])
        case "gemm_a16w16_gated":
            config = config.copy()
            config.pop("NUM_KSPLIT", None)
            config.pop("SPLITK_BLOCK_SIZE", None)
        case "gemm_a8w8" | "gemm_afp8wfp8_preshuffle":
            from aiter.ops.triton.utils.gemm_config_utils import compute_splitk_params

            compute_splitk_params(config, K)
        case (
            "gemm_a16w8_blockscale"
            | "gemm_a16w8_blockscale_preshuffle"
            | "gemm_a8w8_blockscale"
            | "gemm_a8w8_blockscale_preshuffle"
        ):
            assert config["BLOCK_SIZE_K"] == 128

    return config


def get_profile_functions(kernel, input_shape, config_list):
    """Reuse one set of inputs across all configs, preparing each outside profiling."""
    run = get_kernel_runner(kernel, input_shape)
    for config in config_list:
        yield partial(run, config=_prepare_config(kernel, input_shape[2], config))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", type=kernel_name, choices=KERNEL_CONFIG_NAMES)
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument(
        "configs", nargs="*", help="Ten values per candidate, in config_parms_key order"
    )
    args = parser.parse_args(argv)

    from _utils import get_config_list, run_profile

    for fn in get_profile_functions(
        args.kernel, [args.M, args.N, args.K], get_config_list(args.configs)
    ):
        run_profile(fn)


if __name__ == "__main__":
    main()
