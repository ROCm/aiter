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

Imports stay inside each case, so listing the cases imports no kernels.
"""

CASES = {}


def gemm_case(space=None):
    def register(case):
        case.space = dict(space or {})
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
def gemm_afp4wfp4_preshuffle(M, N, K):
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
    return lambda: op(x, w, x_scales, w_scales, dtype, y)


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
