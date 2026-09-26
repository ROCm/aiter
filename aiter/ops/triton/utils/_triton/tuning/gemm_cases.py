# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""One tuning case per GEMM: how to build its inputs and run it once.

A case is a function named after the wrapper it runs. It takes the op's shape
dims (M, N, K, ...), builds the inputs the way the unit test does, and returns
a callable that runs the op. It never passes config=: tune_gemm.py answers the
op's own get_gemm_config() lookup, which is how it learns the config family,
the backend and where the tuned file goes. So every arch and backend the
wrapper supports is tuned through the same case.

Take `backend=None` and pass `**backend_kwarg(backend)` when the wrapper has a
backend argument, so --backend can pick one. `space=` on the decorator pins
keys the kernel constrains (blockscale kernels need BLOCK_SIZE_K=128), so the
sweep does not try values that can only fail.
`kernels=` lists name substrings for rocprofv3 to time, including reducers.
The default "gemm" excludes cache clears, resets and unrelated GPU work.

Imports stay inside each case, so listing the cases imports no kernels.
"""

CASES = {}


def gemm_case(space=None, kernels=("gemm",)):
    def register(case):
        case.space = dict(space or {})
        case.kernels = kernels
        CASES[case.__name__] = case
        return case

    return register


def backend_kwarg(backend):
    """Pass backend only when asked for, so the wrapper keeps its own default."""
    return {} if backend is None else {"backend": backend}


# Blockscale kernels read one scale per 128-wide block of K.
BLOCKSCALE = {"BLOCK_SIZE_K": [128]}


@gemm_case()
def gemm_a16w16(M, N, K, backend=None):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16 as op
    from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import (
        generate_gemm_a16w16_inputs,
    )

    dtype = torch.bfloat16
    x, w, bias, _, y = generate_gemm_a16w16_inputs(
        M, N, K, dtype, output=True, bias=True
    )
    return lambda: op(x, w, bias, dtype, y, **backend_kwarg(backend))


@gemm_case()
def gemm_a16w16_atomic(M, N, K):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_a16w16_atomic import (
        gemm_a16w16_atomic as op,
    )
    from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import (
        generate_gemm_a16w16_inputs,
    )

    dtype = torch.bfloat16
    x, w, _, _, y = generate_gemm_a16w16_inputs(M, N, K, dtype, output=True)

    def fn():
        y.zero_()  # the kernel accumulates into y
        return op(x, w, dtype, y)

    return fn


@gemm_case()
def gemm_a16w16_gated(M, N, K):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_a16w16_gated import (
        gemm_a16w16_gated as op,
    )
    from op_tests.triton_tests.gemm.basic.test_gemm_a16w16_gated import (
        generate_gemm_a16w16_gated_inputs,
    )

    dtype = torch.bfloat16
    x, w, _, y = generate_gemm_a16w16_gated_inputs(M, N, K, dtype, output=True)
    return lambda: op(x, w, dtype, y)


@gemm_case(space=BLOCKSCALE)
def gemm_a16w8_blockscale(M, N, K):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_a16w8_blockscale import (
        gemm_a16w8_blockscale as op,
    )
    from op_tests.triton_tests.gemm.basic.test_gemm_a16w8_blockscale import (
        generate_gemm_a16w8_blockscale_inputs,
    )

    dtype = torch.bfloat16
    x, _, w, w_scale, y = generate_gemm_a16w8_blockscale_inputs(
        M, N, K, 128, 128, dtype=dtype, output=True, shuffle=False
    )
    return lambda: op(x, w, w_scale, dtype, y, prequant=False)


@gemm_case(space=BLOCKSCALE)
def gemm_a16w8_blockscale_preshuffle(M, N, K):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_a16w8_blockscale import (
        gemm_a16w8_blockscale_preshuffle as op,
    )
    from op_tests.triton_tests.gemm.basic.test_gemm_a16w8_blockscale import (
        generate_gemm_a16w8_blockscale_inputs,
    )

    dtype = torch.bfloat16
    x, _, w, w_scale, y = generate_gemm_a16w8_blockscale_inputs(
        M, N, K, 128, 128, dtype=dtype, output=True, shuffle=True
    )
    return lambda: op(x, w, w_scale, dtype, y, prequant=False)


@gemm_case()
def gemm_a16wfp4(M, N, K):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4 as op
    from op_tests.triton_tests.gemm.basic.test_gemm_a16wfp4 import (
        generate_gemm_a16wfp4_inputs,
    )

    dtype = torch.bfloat16
    x, w, _, _, w_scales, _, y = generate_gemm_a16wfp4_inputs(
        M, N, K, output=True, atomic_add=False, dtype=dtype, layout="TN", shuffle=False
    )
    return lambda: op(x, w, w_scales, False, dtype, y)


@gemm_case()
def gemm_a8w8(M, N, K, backend=None):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8 as op
    from aiter.ops.triton.utils.types import get_fp8_dtypes
    from op_tests.triton_tests.gemm.basic.test_gemm_a8w8 import (
        generate_gemm_a8w8_inputs,
    )

    _, e4m3_type = get_fp8_dtypes()
    dtype = torch.bfloat16
    x, _, w, x_scale, w_scale, _, y = generate_gemm_a8w8_inputs(
        M, N, K, in_dtype=e4m3_type, out_dtype=dtype, layout="TN", output=True
    )
    return lambda: op(x, w, x_scale, w_scale, None, dtype, y, **backend_kwarg(backend))


@gemm_case(space=BLOCKSCALE)
def gemm_a8w8_blockscale(M, N, K, backend=None):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
        gemm_a8w8_blockscale as op,
    )
    from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_blockscale import (
        generate_gemm_a8w8_blockscale_inputs,
    )

    dtype = torch.bfloat16
    x, _, w, _, x_scale, w_scale, y = generate_gemm_a8w8_blockscale_inputs(
        M, N, K, 128, 128, dtype=dtype, layout="TN", output=True, shuffle=False
    )
    return lambda: op(x, w, x_scale, w_scale, dtype, y, **backend_kwarg(backend))


@gemm_case(space=BLOCKSCALE)
def gemm_a8w8_blockscale_preshuffle(M, N, K, backend=None):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
        gemm_a8w8_blockscale_preshuffle as op,
    )
    from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_blockscale import (
        generate_gemm_a8w8_blockscale_inputs,
    )

    dtype = torch.bfloat16
    x, _, w, _, x_scale, w_scale, y = generate_gemm_a8w8_blockscale_inputs(
        M, N, K, 128, 128, dtype=dtype, layout="TN", output=True, shuffle=True
    )
    return lambda: op(x, w, x_scale, w_scale, dtype, y, **backend_kwarg(backend))


@gemm_case()
def gemm_a8w8_per_token_scale(M, N, K):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_a8w8_per_token_scale import (
        gemm_a8w8_per_token_scale as op,
    )
    from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_per_token_scale import (
        generate_gemm_a8w8_per_token_scale_inputs,
    )

    dtype = torch.bfloat16
    x, w, x_scale, w_scale, y = generate_gemm_a8w8_per_token_scale_inputs(
        M, N, K, dtype=dtype, layout="TN", output=True
    )
    return lambda: op(x, w, x_scale, w_scale, dtype, y)


@gemm_case()
def gemm_a8wfp4(M, N, K):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_a8wfp4 import gemm_a8wfp4 as op
    from aiter.ops.triton.utils.types import get_fp8_dtypes
    from op_tests.triton_tests.gemm.basic.test_gemm_a8wfp4 import (
        generate_gemm_a8wfp4_inputs,
    )

    _, e4m3_type = get_fp8_dtypes()
    dtype = torch.float16
    x, w, x_scales, w_scales, _, _, y = generate_gemm_a8wfp4_inputs(
        M, N, K, e4m3_type, dtype, layout="TN", output=True
    )
    return lambda: op(x, w, y, x_scales, w_scales, dtype)


@gemm_case()
def gemm_afp4wfp4(M, N, K, backend=None):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4 as op
    from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import (
        generate_gemm_afp4wfp4_inputs,
    )

    dtype = torch.bfloat16
    x, _, w, _, _, x_scales, w_scales, _, y = generate_gemm_afp4wfp4_inputs(
        M, N, K, dtype, output=True, shuffle_scales_fg=False, shuffle_weight_fg=False
    )
    return lambda: op(x, w, x_scales, w_scales, dtype, y, **backend_kwarg(backend))


@gemm_case()
def gemm_afp4wfp4_preshuffle(M, N, K, backend=None):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import (
        gemm_afp4wfp4_preshuffle as op,
    )
    from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import (
        generate_gemm_afp4wfp4_inputs,
    )

    dtype = torch.bfloat16
    x, _, w, _, _, x_scales, w_scales, _, y = generate_gemm_afp4wfp4_inputs(
        M, N, K, dtype, output=True, shuffle_scales_fg=True, shuffle_weight_fg=True
    )
    return lambda: op(x, w, x_scales, w_scales, dtype, y, **backend_kwarg(backend))


@gemm_case()
def gemm_afp4wfp4_pre_quant(M, N, K):
    """Reads the GEMM-A16WFP4 family: it is gemm_a16wfp4 with atomic_add=True."""
    import torch

    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4_pre_quant_atomic import (
        gemm_afp4wfp4_pre_quant as op,
    )
    from op_tests.triton_tests.gemm.basic.test_gemm_a16wfp4 import (
        generate_gemm_a16wfp4_inputs,
    )

    dtype = torch.float32
    x, w, _, _, w_scales, _, y = generate_gemm_a16wfp4_inputs(
        M, N, K, output=True, atomic_add=True, dtype=dtype, layout="TN", shuffle=False
    )

    def fn():
        y.zero_()  # the kernel accumulates into y
        return op(x, w, w_scales, dtype, y)

    return fn


@gemm_case()
def gemm_afp8wfp8_preshuffle(M, N, K, backend=None):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_afp8wfp8 import (
        gemm_afp8wfp8_preshuffle as op,
    )
    from op_tests.triton_tests.gemm.basic.test_gemm_afp8wfp8 import generate_inputs

    dtype = torch.bfloat16
    x, _, w, _, x_scales, w_scales = generate_inputs(M, N, K, shuffle=True)
    return lambda: op(x, w, x_scales, w_scales, dtype=dtype, **backend_kwarg(backend))


@gemm_case()
def batched_gemm_bf16(M, N, K, B, backend=None):
    import torch

    from aiter.ops.triton.gemm.batched.batched_gemm_bf16 import (
        batched_gemm_bf16 as op,
    )
    from op_tests.triton_tests.gemm.batched.test_batched_gemm_bf16 import (
        generate_batched_gemm_a16w16_inputs,
    )

    dtype = torch.bfloat16
    x, w, bias, y = generate_batched_gemm_a16w16_inputs(B, M, N, K, dtype, output=True)
    return lambda: op(x, w, bias, dtype, YQ=y, **backend_kwarg(backend))


# Additional basic variants use the same lookup mechanism as the cases above.


@gemm_case(space={"NUM_KSPLIT": [1]})
def gemm_a16w16_persistent(M, N, K, backend=None):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16 as op
    from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import (
        generate_gemm_a16w16_inputs,
    )

    dtype = torch.bfloat16
    x, w, bias, _, y = generate_gemm_a16w16_inputs(
        M, N, K, dtype, output=True, bias=True
    )
    return lambda: op(x, w, bias, dtype, y, persistent=True, **backend_kwarg(backend))


@gemm_case()
def gemm_a16wfp4_preshuffle(M, N, K):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import (
        gemm_a16wfp4_preshuffle as op,
    )
    from op_tests.triton_tests.gemm.basic.test_gemm_a16wfp4 import (
        generate_gemm_a16wfp4_inputs,
    )

    dtype = torch.bfloat16
    x, _, w, _, _, w_scales, y = generate_gemm_a16wfp4_inputs(
        M, N, K, output=True, atomic_add=False, dtype=dtype, shuffle=True
    )
    return lambda: op(x, w, w_scales, dtype=dtype, y=y)


@gemm_case()
def gemm_a8w8_preshuffle(M, N, K):
    """This wrapper has only a Gluon implementation (currently gfx950)."""
    import torch

    from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8_preshuffle as op
    from aiter.ops.triton.utils.types import get_fp8_dtypes
    from op_tests.triton_tests.gemm.basic.test_gemm_a8w8 import (
        generate_gemm_a8w8_inputs,
    )

    _, e4m3_type = get_fp8_dtypes()
    dtype = torch.bfloat16
    x, _, w, xs, ws, bias, y = generate_gemm_a8w8_inputs(
        M, N, K, e4m3_type, dtype, output=True, shuffle=True
    )
    return lambda: op(x, w, xs, ws, bias, dtype, y)


@gemm_case()
def gemm_a8w8_blockscale_group32(M, N, K):
    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale_group32 import (
        gemm_a8w8_blockscale_group32 as op,
    )
    from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_blockscale_group32 import (
        generate_inputs,
    )

    x, w, xs, ws = generate_inputs(M, N, K)
    return lambda: op(x, w, xs, ws)


@gemm_case()
def gemm_afp4wfp4_preshuffled_scales(M, N, K):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import (
        gemm_afp4wfp4_preshuffled_scales as op,
    )
    from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import (
        generate_gemm_afp4wfp4_inputs,
    )

    dtype = torch.bfloat16
    x, _, w, _, _, xs, ws, _, y = generate_gemm_afp4wfp4_inputs(
        M, N, K, dtype, output=True, shuffle_scales_fg=True, shuffle_weight_fg=False
    )
    return lambda: op(x, w, xs, ws, dtype, y)


@gemm_case()
def gemm_afp8wfp8(M, N, K):
    from aiter.ops.triton.gemm.basic.gemm_afp8wfp8 import gemm_afp8wfp8 as op
    from op_tests.triton_tests.gemm.basic.test_gemm_afp8wfp8 import generate_inputs

    x, _, w, _, xs, ws = generate_inputs(M, N, K, shuffle=False)
    return lambda: op(x, w, xs, ws)


@gemm_case()
def batched_gemm_a8w8(M, N, K, B):
    import torch

    from aiter.ops.triton.gemm.batched.batched_gemm_a8w8 import (
        batched_gemm_a8w8 as op,
    )
    from op_tests.triton_tests.gemm.batched.test_batched_gemm_a8w8 import (
        generate_batched_gemm_a8w8_inputs,
    )

    dtype = torch.bfloat16
    x, w, xs, ws, bias, y = generate_batched_gemm_a8w8_inputs(
        B, M, N, K, dtype, output=True
    )
    return lambda: op(x, w, xs, ws, bias, dtype, YQ=y)


@gemm_case(space=BLOCKSCALE)
def batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(M, N, K, B):
    import torch

    from aiter.ops.triton.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as op,
    )
    from op_tests.triton_tests.gemm.batched.test_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
        generate_batched_gemm_a16w8_inputs,
    )

    dtype = torch.bfloat16
    x, w, ws, bias, y = generate_batched_gemm_a16w8_inputs(
        B, M, N, K, dtype, has_bias=True, output=True
    )
    return lambda: op(x, w, ws, group_size=128, bias=bias, dtype=dtype, YQ=y)


@gemm_case()
def batched_gemm_a16wfp4(M, N, K, B):
    import torch

    from aiter.ops.triton.gemm.batched.batched_gemm_a16wfp4 import (
        batched_gemm_a16wfp4 as op,
    )
    from op_tests.triton_tests.gemm.batched.test_batched_gemm_a16wfp4 import (
        generate_batched_gemm_a16wfp4_inputs,
    )

    dtype = torch.bfloat16
    x, w, _, ws, y = generate_batched_gemm_a16wfp4_inputs(
        B, M, N, K, dtype, output=True
    )
    return lambda: op(x, w, ws, dtype=dtype, y=y)


@gemm_case()
def batched_gemm_afp4wfp4(M, N, K, B):
    import torch

    from aiter.ops.triton.gemm.batched.batched_gemm_afp4wfp4 import (
        batched_gemm_afp4wfp4 as op,
    )
    from op_tests.triton_tests.gemm.batched.test_batched_gemm_afp4wfp4 import (
        generate_batched_gemm_afp4wfp4_inputs,
    )

    dtype = torch.bfloat16
    x, w, xs, ws, y = generate_batched_gemm_afp4wfp4_inputs(
        B, M, N, K, dtype, output=True
    )
    return lambda: op(x, w, xs, ws, dtype=dtype, y=y)


@gemm_case()
def batched_gemm_afp4wfp4_pre_quant(M, N, K, B):
    """Deprecated API with identical semantics to batched_gemm_a16wfp4."""
    return batched_gemm_a16wfp4(M, N, K, B)


@gemm_case()
def fused_gemm_a16w16_quant_x(M, N, K):
    import torch

    from aiter.ops.triton.gemm.fused.fused_gemm_a16w16_quant_x import (
        fused_gemm_a16w16_quant_x as op,
    )
    from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import (
        generate_gemm_a16w16_inputs,
    )

    dtype = torch.bfloat16
    x, w, _, _, y = generate_gemm_a16w16_inputs(M, N, K, dtype, output=True)
    return lambda: op(x, w, dtype=dtype, y=y)


@gemm_case(space=BLOCKSCALE)
def fused_gemm_a8w8_blockscale_a16w16(M, N1, N2, K):
    """N1 is the FP8 projection width; N2 is the BF16 projection width."""
    import torch

    from aiter.ops.triton.gemm.fused.fused_gemm_a8w8_blockscale_a16w16 import (
        fused_gemm_a8w8_blockscale_a16w16 as op,
    )
    from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_blockscale import (
        generate_gemm_a8w8_blockscale_inputs,
    )
    from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import (
        generate_gemm_a16w16_inputs,
    )

    dtype = torch.bfloat16
    x8, w8, _, xs, _, ws, y8 = generate_gemm_a8w8_blockscale_inputs(
        M, N1, K, 128, 128, dtype, output=True
    )
    x16, w16, _, _, y16 = generate_gemm_a16w16_inputs(M, N2, K, dtype, output=True)
    return lambda: op(x8, w8, xs, ws, x16, w16, dtype=dtype, y_fp8=y8, y_bf16=y16)


def _mixed_fp4_case(M, N1, N2, K, shuffle):
    import torch

    from aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_a16w16 import (
        fused_gemm_afp4wfp4_a16w16 as op,
    )
    from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import (
        generate_gemm_a16w16_inputs,
    )
    from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import (
        generate_gemm_afp4wfp4_inputs,
    )

    dtype = torch.bfloat16
    x4, _, w4, _, _, xs, ws, _, y4 = generate_gemm_afp4wfp4_inputs(
        M,
        N1,
        K,
        dtype,
        output=True,
        shuffle_scales_fg=shuffle,
        shuffle_weight_fg=shuffle,
    )
    x16, w16, _, _, y16 = generate_gemm_a16w16_inputs(M, N2, K, dtype, output=True)
    return lambda: op(
        x4,
        w4,
        xs,
        ws,
        x16,
        w16,
        is_fp4_preshuffled=shuffle,
        dtype=dtype,
        y_fp4=y4,
        y_bf16=y16,
        use_aot=False,
    )


@gemm_case()
def fused_gemm_afp4wfp4_a16w16(M, N1, N2, K):
    return _mixed_fp4_case(M, N1, N2, K, shuffle=False)


@gemm_case()
def fused_gemm_afp4wfp4_preshuffle_a16w16(M, N1, N2, K):
    return _mixed_fp4_case(M, N1, N2, K, shuffle=True)


@gemm_case(space=BLOCKSCALE)
def fused_gemm_a8w8_blockscale_mul_add(M, N, K):
    import torch

    from aiter.ops.triton.gemm.fused.fused_gemm_a8w8_blockscale_mul_add import (
        fused_gemm_a8w8_blockscale_mul_add as op,
    )
    from op_tests.triton_tests.fusions.test_fused_mul_add import (
        generate_fused_mul_add_inputs,
    )
    from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_blockscale import (
        generate_gemm_a8w8_blockscale_inputs,
    )

    dtype = torch.bfloat16
    x, w, _, xs, _, ws, y = generate_gemm_a8w8_blockscale_inputs(
        M, N, K, 128, 128, dtype, output=True
    )
    _, a, b = generate_fused_mul_add_inputs([M, N], False, False, dtype)
    return lambda: op(x, w, xs, ws, a, b, dtype=dtype, y=y)


def _fp4_mul_add_case(M, N, K, shuffle):
    import torch

    from aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_mul_add import (
        fused_gemm_afp4wfp4_mul_add,
        fused_gemm_afp4wfp4_preshuffle_add_mul,
    )
    from op_tests.triton_tests.fusions.test_fused_mul_add import (
        generate_fused_mul_add_inputs,
    )
    from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import (
        generate_gemm_afp4wfp4_inputs,
    )

    dtype = torch.bfloat16
    x, _, w, _, _, xs, ws, _, y = generate_gemm_afp4wfp4_inputs(
        M,
        N,
        K,
        dtype,
        output=True,
        shuffle_scales_fg=shuffle,
        shuffle_weight_fg=shuffle,
    )
    _, a, b = generate_fused_mul_add_inputs([M, N], False, False, dtype)
    if shuffle:
        return lambda: fused_gemm_afp4wfp4_preshuffle_add_mul(
            x, w, xs, ws, a, b, dtype=dtype, y=y, use_aot=False
        )
    return lambda: fused_gemm_afp4wfp4_mul_add(x, w, xs, ws, a, b, dtype=dtype, y=y)


@gemm_case()
def fused_gemm_afp4wfp4_mul_add(M, N, K):
    return _fp4_mul_add_case(M, N, K, shuffle=False)


@gemm_case()
def fused_gemm_afp4wfp4_preshuffle_add_mul(M, N, K):
    return _fp4_mul_add_case(M, N, K, shuffle=True)


def _split_cat_sizes(N, D):
    if D <= 0 or N % D:
        raise ValueError("split-cat requires N divisible by D")
    S = N // D
    return S // 2, S - S // 2


def _fp8_split_cat_case(M, N, K, D, S3, shuffle):
    import torch

    from aiter.ops.triton.gemm.fused.fused_gemm_a8w8_blockscale_split_cat import (
        fused_gemm_a8w8_blockscale_preshuffle_split_cat,
        fused_gemm_a8w8_blockscale_split_cat,
    )
    from op_tests.triton_tests.gemm.fused.test_fused_gemm_a8w8_blockscale_split_cat import (
        generate_fused_gemm_a8w8_blockscale_split_cat_inputs,
    )

    dtype = torch.bfloat16
    S1, S2 = _split_cat_sizes(N, D)
    x, _, w, y, _, xs, ws = generate_fused_gemm_a8w8_blockscale_split_cat_inputs(
        M, N, K, S3, 128, 128, dtype, shuffle=shuffle
    )
    op = (
        fused_gemm_a8w8_blockscale_preshuffle_split_cat
        if shuffle
        else fused_gemm_a8w8_blockscale_split_cat
    )
    y = y.expand(M, D, S3)
    return lambda: op(x, w, y, xs, ws, S1, S2, dtype)


@gemm_case(space=BLOCKSCALE)
def fused_gemm_a8w8_blockscale_split_cat(M, N, K, D, S3):
    return _fp8_split_cat_case(M, N, K, D, S3, shuffle=False)


@gemm_case(space=BLOCKSCALE)
def fused_gemm_a8w8_blockscale_preshuffle_split_cat(M, N, K, D, S3):
    return _fp8_split_cat_case(M, N, K, D, S3, shuffle=True)


def _fp4_split_cat_case(M, N, K, D, S3, shuffle):
    import torch

    from aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_split_cat import (
        fused_gemm_afp4wfp4_preshuffle_split_cat,
        fused_gemm_afp4wfp4_split_cat,
    )
    from op_tests.triton_tests.gemm.fused.test_fused_gemm_afp4wfp4_split_cat import (
        generate_fused_gemm_afp4wfp4_split_cat_inputs,
    )

    dtype = torch.bfloat16
    S1, S2 = _split_cat_sizes(N, D)
    x, _, w, y, _, _, xs, ws = generate_fused_gemm_afp4wfp4_split_cat_inputs(
        M, N, K, S3, dtype, shuffle=shuffle
    )
    op = (
        fused_gemm_afp4wfp4_preshuffle_split_cat
        if shuffle
        else fused_gemm_afp4wfp4_split_cat
    )
    y = y.expand(M, D, S3)
    return lambda: op(x, w, y, xs, ws, S1, S2, dtype)


@gemm_case()
def fused_gemm_afp4wfp4_split_cat(M, N, K, D, S3):
    return _fp4_split_cat_case(M, N, K, D, S3, shuffle=False)


@gemm_case()
def fused_gemm_afp4wfp4_preshuffle_split_cat(M, N, K, D, S3):
    return _fp4_split_cat_case(M, N, K, D, S3, shuffle=True)


def _fused_ff_case(M, N, K, gated):
    import torch

    from aiter.ops.triton.gemm.feed_forward.ff_a16w16_fused_gated import (
        ff_a16w16_fused_gated,
    )
    from aiter.ops.triton.gemm.feed_forward.ff_a16w16_fused_ungated import (
        ff_a16w16_fused_ungated,
    )
    from op_tests.triton_tests.gemm.feed_forward.ff_test_utils import generate_ff_inputs

    if gated and N % 2:
        raise ValueError("gated feed-forward requires even N (the full up projection)")
    dtype = torch.bfloat16
    x, w1, w2, _, _, y = generate_ff_inputs(
        M,
        K,
        N // 2 if gated else N,
        dtype,
        gating=gated,
        output=True,
        y_init="zeros",
    )
    op = ff_a16w16_fused_gated if gated else ff_a16w16_fused_ungated

    def fn():
        y.zero_()  # both fused feed-forward kernels accumulate atomically
        return op(x, w1, w2, dtype=dtype, y=y, activation="silu")

    return fn


@gemm_case(kernels=("_ff_",))
def ff_a16w16_fused_gated(M, N, K):
    return _fused_ff_case(M, N, K, gated=True)


@gemm_case(kernels=("_ff_",))
def ff_a16w16_fused_ungated(M, N, K):
    return _fused_ff_case(M, N, K, gated=False)
