# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Profile a selected kernel; only its case imports GPU and operator modules."""

import argparse
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


def get_profile_functions(kernel, input_shape, config_list):
    """Generate inputs once and yield one profiling callable per config."""
    match kernel:
        case "batched_gemm_bf16":
            import torch
            import triton

            from aiter.ops.triton.gemm.batched.batched_gemm_bf16 import (
                batched_gemm_bf16,
            )
            from op_tests.triton_tests.gemm.batched.test_batched_gemm_bf16 import (
                generate_batched_gemm_a16w16_inputs,
            )

            dtype = torch.bfloat16

            M, N, K = input_shape

            B = 8 if K == 4096 else 16

            x, weight, bias, y = generate_batched_gemm_a16w16_inputs(
                B,
                M,
                N,
                K,
                dtype,
                output=True,
            )

            for config in config_list:
                if config is not None:
                    config = config.copy()
                    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(
                        input_shape[2], config["NUM_KSPLIT"]
                    )

                def fn(config=config):
                    batched_gemm_bf16(x, weight, bias, dtype, YQ=y, config=config)

                yield fn

        case "gemm_a16w16":
            import torch
            import triton

            from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
            from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import (
                generate_gemm_a16w16_inputs,
            )

            dtype = torch.bfloat16

            x, w, bias, _, y = generate_gemm_a16w16_inputs(
                *input_shape,
                dtype,
                output=True,
                bias=True,
            )

            for config in config_list:
                if config is not None:
                    config = config.copy()
                    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(
                        input_shape[2], config["NUM_KSPLIT"]
                    )

                def fn(config=config):
                    gemm_a16w16(x, w, bias, dtype, y, config=config)

                yield fn

        case "gemm_a16w16_atomic":
            import torch
            import triton

            from aiter.ops.triton.gemm.basic.gemm_a16w16_atomic import (
                gemm_a16w16_atomic,
            )
            from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import (
                generate_gemm_a16w16_inputs,
            )

            dtype = torch.bfloat16

            x, w, _, _, y = generate_gemm_a16w16_inputs(
                *input_shape,
                dtype,
                output=True,
            )

            for config in config_list:
                if config is not None:
                    config = config.copy()
                    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(
                        input_shape[2], config["NUM_KSPLIT"]
                    )

                def fn(config=config):
                    y.zero_()
                    gemm_a16w16_atomic(x, w, dtype, y, config=config)

                yield fn

        case "gemm_a16w16_gated":
            import torch

            from aiter.ops.triton.gemm.basic.gemm_a16w16_gated import gemm_a16w16_gated
            from op_tests.triton_tests.gemm.basic.test_gemm_a16w16_gated import (
                generate_gemm_a16w16_gated_inputs,
            )

            M, N, K = input_shape

            dtype = torch.bfloat16

            x, w, _, y = generate_gemm_a16w16_gated_inputs(
                M,
                N,
                K,
                dtype,
                output=True,
            )

            for config in config_list:
                if config is not None:
                    config = config.copy()
                    # Gated kernel doesn't support split-K, remove those keys
                    config.pop("NUM_KSPLIT", None)
                    config.pop("SPLITK_BLOCK_SIZE", None)

                def fn(config=config):
                    gemm_a16w16_gated(x, w, dtype, y, config=config)

                yield fn

        case "gemm_a16w8_blockscale":
            import torch

            from aiter.ops.triton.gemm.basic.gemm_a16w8_blockscale import (
                gemm_a16w8_blockscale,
            )
            from op_tests.triton_tests.gemm.basic.test_gemm_a16w8_blockscale import (
                generate_gemm_a16w8_blockscale_inputs,
            )

            dtype = torch.bfloat16

            shuffle = False

            block_shape_n, block_shape_k = 128, 128

            x, weight, weight_triton, w_scale, y = (
                generate_gemm_a16w8_blockscale_inputs(
                    *input_shape,
                    block_shape_n,
                    block_shape_k,
                    dtype=dtype,
                    output=True,
                    shuffle=shuffle,
                )
            )

            for config in config_list:
                assert config is None or config["BLOCK_SIZE_K"] == 128

                def fn(config=config):
                    gemm_a16w8_blockscale(
                        x,
                        weight_triton,
                        w_scale,
                        dtype,
                        y,
                        prequant=False,
                        config=config,
                    )

                yield fn

        case "gemm_a16w8_blockscale_preshuffle":
            import torch

            from aiter.ops.triton.gemm.basic.gemm_a16w8_blockscale import (
                gemm_a16w8_blockscale_preshuffle,
            )
            from op_tests.triton_tests.gemm.basic.test_gemm_a16w8_blockscale import (
                generate_gemm_a16w8_blockscale_inputs,
            )

            dtype = torch.bfloat16

            shuffle = True

            block_shape_n, block_shape_k = 128, 128

            x, weight, weight_triton, w_scale, y = (
                generate_gemm_a16w8_blockscale_inputs(
                    *input_shape,
                    block_shape_n,
                    block_shape_k,
                    dtype=dtype,
                    output=True,
                    shuffle=shuffle,
                )
            )

            for config in config_list:
                assert config is None or config["BLOCK_SIZE_K"] == 128

                def fn(config=config):
                    gemm_a16w8_blockscale_preshuffle(
                        x,
                        weight_triton,
                        w_scale,
                        dtype,
                        y,
                        prequant=False,
                        config=config,
                    )

                yield fn

        case "gemm_a16wfp4":
            import torch

            from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
            from op_tests.triton_tests.gemm.basic.test_gemm_a16wfp4 import (
                generate_gemm_a16wfp4_inputs,
            )

            M, N, K = input_shape

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

            for config in config_list:

                def fn(config=config):
                    # Signature: gemm_a16wfp4(x, w, w_scales, atomic_add, dtype, y, config)
                    gemm_a16wfp4(x, w, w_scales, False, dtype, y, config=config)

                yield fn

        case "gemm_a8w8":
            import torch

            from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8
            from aiter.ops.triton.utils.gemm_config_utils import compute_splitk_params
            from aiter.ops.triton.utils.types import get_fp8_dtypes
            from op_tests.triton_tests.gemm.basic.test_gemm_a8w8 import (
                generate_gemm_a8w8_inputs,
            )

            M, N, K = input_shape

            _, e4m3_type = get_fp8_dtypes()

            dtype = torch.bfloat16

            x, weight, weight_triton, x_scale, w_scale, bias, y = (
                generate_gemm_a8w8_inputs(
                    *input_shape,
                    in_dtype=e4m3_type,
                    out_dtype=dtype,
                    layout="TN",
                    output=True,
                )
            )

            for config in config_list:
                if config is not None:
                    compute_splitk_params(config, K)

                def fn(config=config):
                    gemm_a8w8(
                        x,
                        weight_triton,
                        x_scale,
                        w_scale,
                        None,
                        dtype,
                        y,
                        config=config,
                    )

                yield fn

        case "gemm_a8w8_blockscale":
            import torch

            from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
                gemm_a8w8_blockscale,
            )
            from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_blockscale import (
                generate_gemm_a8w8_blockscale_inputs,
            )

            dtype = torch.bfloat16

            shuffle = False

            block_shape_n, block_shape_k = 128, 128

            x, weight, weight_triton, x_scale, x_scale_shuffled, w_scale, y = (
                generate_gemm_a8w8_blockscale_inputs(
                    *input_shape,
                    block_shape_n,
                    block_shape_k,
                    dtype=dtype,
                    layout="TN",
                    output=True,
                    shuffle=shuffle,
                )
            )

            for config in config_list:
                assert config is None or config["BLOCK_SIZE_K"] == 128

                def fn(config=config):
                    gemm_a8w8_blockscale(
                        x,
                        weight_triton,
                        x_scale_shuffled,
                        w_scale,
                        dtype,
                        y,
                        config=config,
                    )

                yield fn

        case "gemm_a8w8_blockscale_preshuffle":
            import torch

            from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
                gemm_a8w8_blockscale_preshuffle,
            )
            from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_blockscale import (
                generate_gemm_a8w8_blockscale_inputs,
            )

            dtype = torch.bfloat16

            shuffle = True

            block_shape_n, block_shape_k = 128, 128

            x, weight, weight_triton, x_scale, x_scale_shuffled, w_scale, y = (
                generate_gemm_a8w8_blockscale_inputs(
                    *input_shape,
                    block_shape_n,
                    block_shape_k,
                    dtype=dtype,
                    layout="TN",
                    output=True,
                    shuffle=shuffle,
                )
            )

            for config in config_list:
                assert config is None or config["BLOCK_SIZE_K"] == 128

                def fn(config=config):
                    gemm_a8w8_blockscale_preshuffle(
                        x,
                        weight_triton,
                        x_scale_shuffled,
                        w_scale,
                        dtype,
                        y,
                        config=config,
                    )

                yield fn

        case "gemm_a8w8_per_token_scale":
            import torch

            from aiter.ops.triton.gemm.basic.gemm_a8w8_per_token_scale import (
                gemm_a8w8_per_token_scale,
            )
            from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_per_token_scale import (
                generate_gemm_a8w8_per_token_scale_inputs,
            )

            dtype = torch.bfloat16

            x, weight, x_scale, w_scale, y = generate_gemm_a8w8_per_token_scale_inputs(
                *input_shape,
                dtype=dtype,
                layout="TN",
                output=True,
            )

            for config in config_list:

                def fn(config=config):
                    gemm_a8w8_per_token_scale(
                        x, weight, x_scale, w_scale, dtype, y, config=config
                    )

                yield fn

        case "gemm_a8wfp4":
            import torch
            import triton

            from aiter.ops.triton.gemm.basic.gemm_a8wfp4 import gemm_a8wfp4
            from aiter.ops.triton.utils.types import get_fp8_dtypes
            from op_tests.triton_tests.gemm.basic.test_gemm_a8wfp4 import (
                generate_gemm_a8wfp4_inputs,
            )

            M, N, K = input_shape

            _, e4m3_type = get_fp8_dtypes()

            dtype = torch.float16

            x, w, x_scales, w_scales, _, _, y = generate_gemm_a8wfp4_inputs(
                M,
                N,
                K,
                e4m3_type,
                dtype,
                layout="TN",
                output=True,
            )

            for config in config_list:
                if config is not None:
                    config = config.copy()
                    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])

                def fn(config=config):
                    gemm_a8wfp4(x, w, y, x_scales, w_scales, dtype, config=config)

                yield fn

        case "gemm_afp4wfp4":
            import torch

            from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
            from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import (
                generate_gemm_afp4wfp4_inputs,
            )

            dtype = torch.bfloat16

            shuffle = False

            (
                x,
                w,
                w_triton,
                x_scales,
                w_scales,
                x_scales_triton,
                w_scales_triton,
                _out_dtype,
                y,
            ) = generate_gemm_afp4wfp4_inputs(
                *input_shape,
                dtype,
                output=True,
                shuffle_scales_fg=shuffle,
                shuffle_weight_fg=shuffle,
            )

            for config in config_list:

                def fn(config=config):
                    gemm_afp4wfp4(
                        x,
                        w_triton,
                        x_scales_triton,
                        w_scales_triton,
                        dtype,
                        y,
                        config=config,
                    )

                yield fn

        case "gemm_afp4wfp4_pre_quant_atomic":
            import torch

            from aiter.ops.triton.gemm.basic.gemm_afp4wfp4_pre_quant_atomic import (
                gemm_afp4wfp4_pre_quant,
            )
            from op_tests.triton_tests.gemm.basic.test_gemm_a16wfp4 import (
                generate_gemm_a16wfp4_inputs,
            )

            M, N, K = input_shape

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

            for config in config_list:

                def fn(config=config):
                    gemm_afp4wfp4_pre_quant(x, w, w_scales, dtype, y, config=config)

                yield fn

        case "gemm_afp4wfp4_preshuffle":
            import torch

            from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import (
                gemm_afp4wfp4_preshuffle,
            )
            from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import (
                generate_gemm_afp4wfp4_inputs,
            )

            dtype = torch.bfloat16

            shuffle = True

            (
                x,
                w,
                w_triton,
                x_scales,
                w_scales,
                x_scales_triton,
                w_scales_triton,
                _out_dtype,
                y,
            ) = generate_gemm_afp4wfp4_inputs(
                *input_shape,
                dtype,
                output=True,
                shuffle_scales_fg=shuffle,
                shuffle_weight_fg=shuffle,
            )

            for config in config_list:

                def fn(config=config):
                    gemm_afp4wfp4_preshuffle(
                        x,
                        w_triton,
                        x_scales_triton,
                        w_scales_triton,
                        dtype,
                        y,
                        config=config,
                    )

                yield fn

        case "gemm_afp8wfp8_preshuffle":
            import torch

            from aiter.ops.triton.gemm.basic.gemm_afp8wfp8 import (
                gemm_afp8wfp8_preshuffle,
            )
            from aiter.ops.triton.utils.gemm_config_utils import compute_splitk_params
            from aiter.ops.triton.utils.types import get_fp8_dtypes
            from op_tests.triton_tests.gemm.basic.test_gemm_afp8wfp8 import (
                generate_inputs,
            )

            M, N, K = input_shape

            _, e4m3_type = get_fp8_dtypes()

            dtype = torch.bfloat16

            x_fp8, _w_fp8, w_kernel, x_scales, w_scales = generate_inputs(
                *input_shape,
                shuffle=True,
            )

            for config in config_list:
                if config is not None:
                    compute_splitk_params(config, K)

                def fn(config=config):
                    gemm_afp8wfp8_preshuffle(
                        x_fp8, w_kernel, x_scales, w_scales, dtype=dtype, config=config
                    )

                yield fn

        case _:
            raise ValueError(f"Unknown kernel: {kernel}")


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
