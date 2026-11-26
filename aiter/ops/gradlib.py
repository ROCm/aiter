# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from enum import IntEnum

import torch

from csrc.cpp_itfs.torch_utils import direct_register_custom_op

from ..jit.core import compile_ops
from ..jit.utils.torch_guard import torch_compile_guard


@compile_ops("module_hipbsolgemm")
def hipb_create_extension() -> None:
    """Initialize hipBLASLt resources for the current device.

    Repeated calls on that device are no-ops. To change devices, synchronize
    outstanding work and destroy the existing extension before initializing
    again. Only one device's resources can be active in the module at a time.
    """


@compile_ops("module_hipbsolgemm")
def hipb_destroy_extension() -> None:
    """Release hipBLASLt resources, allowing a later initialization.

    Synchronize outstanding GEMMs before calling. Destruction uses the owning
    device and restores the caller's current device. Without an active
    extension this is a no-op, including after a previous destruction.
    """


class HipblasLtMatmulScaleMode(IntEnum):
    SCALAR_32F = 0
    VEC32_UE8M0 = 2
    OUTER_VEC_32F = 3
    BLK32_UE8M0_32_8_EXT = 1001


def gen_hipb_mm_fake_tensor(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    solution_index: int,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    scaleA: torch.Tensor | None = None,
    scaleB: torch.Tensor | None = None,
    scaleOut: torch.Tensor | None = None,
    bpreshuffle: bool | None = None,
    use_gelu: bool | None = None,
    out: torch.Tensor | None = None,
    scaleA_mode: int | None = None,
    scaleB_mode: int | None = None,
):
    return torch.empty(
        (mat1.size(0), mat2.size(1)),
        dtype=_hipb_mm_out_dtype(mat1, out_dtype, out),
        device=mat1.device,
    )


def _hipb_mm_out_dtype(
    mat1: torch.Tensor, out_dtype: torch.dtype | None, out: torch.Tensor | None
) -> torch.dtype:
    """The result dtype: ``out``'s when given, else ``out_dtype``, else the input's."""
    if out is None:
        return out_dtype if out_dtype is not None else mat1.dtype
    if out_dtype is not None and out_dtype != out.dtype:
        raise ValueError(f"hipb_mm: out_dtype={out_dtype} but out is {out.dtype}")
    return out.dtype


# torch-free kernel entry: writes into caller-allocated `result` (the de-torched C++
# TU can no longer torch::empty). outDtype is read from result.dtype() on the C++ side.
@compile_ops(
    "module_hipbsolgemm",
    fc_name="hipb_mm",
    develop=True,
    tensor_cache_args=("mat2",),
)
def _hipb_mm(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    solution_index: int,
    result: torch.Tensor,
    bias: torch.Tensor | None = None,
    scaleA: torch.Tensor | None = None,
    scaleB: torch.Tensor | None = None,
    scaleOut: torch.Tensor | None = None,
    bpreshuffle: bool | None = None,
    use_gelu: bool | None = None,
    scaleA_mode: int | None = None,
    scaleB_mode: int | None = None,
) -> None: ...


def hipb_mm(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    solution_index: int,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    scaleA: torch.Tensor | None = None,
    scaleB: torch.Tensor | None = None,
    scaleOut: torch.Tensor | None = None,
    bpreshuffle: bool | None = None,
    use_gelu: bool | None = None,
    out: torch.Tensor | None = None,
    scaleA_mode: int | None = None,
    scaleB_mode: int | None = None,
) -> torch.Tensor:
    """``out``: the [M, N] tensor to write instead of allocating; returned."""
    dtype = _hipb_mm_out_dtype(mat1, out_dtype, out)
    result = (
        torch.empty((mat1.size(0), mat2.size(1)), dtype=dtype, device=mat1.device)
        if out is None
        else out
    )
    _hipb_mm(
        mat1,
        mat2,
        solution_index,
        result,
        bias,
        scaleA,
        scaleB,
        scaleOut,
        bpreshuffle,
        use_gelu,
        scaleA_mode,
        scaleB_mode,
    )
    return result


direct_register_custom_op(
    "hipb_mm",
    hipb_mm,
    ["out"],
    fake_impl=gen_hipb_mm_fake_tensor,
)


@compile_ops("module_hipbsolgemm", fc_name="hipb_findallsols", develop=True)
def _hipb_findallsols(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    result: torch.Tensor,
    bias: torch.Tensor | None = None,
    scaleA: torch.Tensor | None = None,
    scaleB: torch.Tensor | None = None,
    scaleC: torch.Tensor | None = None,
    bpreshuffle: bool = False,
    use_gelu: bool = False,
    scaleA_mode: int | None = None,
    scaleB_mode: int | None = None,
) -> list[int]: ...


def hipb_findallsols(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    scaleA: torch.Tensor | None = None,
    scaleB: torch.Tensor | None = None,
    scaleC: torch.Tensor | None = None,
    bpreshuffle: bool = False,
    use_gelu: bool = False,
    scaleA_mode: int | None = None,
    scaleB_mode: int | None = None,
) -> list[int]:
    out_dtype = out_dtype if out_dtype is not None else mat1.dtype
    result = torch.empty(
        (mat1.size(0), mat2.size(1)), dtype=out_dtype, device=mat1.device
    )
    return _hipb_findallsols(
        mat1,
        mat2,
        result,
        bias,
        scaleA,
        scaleB,
        scaleC,
        bpreshuffle,
        use_gelu,
        scaleA_mode,
        scaleB_mode,
    )


@compile_ops("module_hipbsolgemm")
def getHipblasltKernelName() -> None: ...


class HipblasLtInt4Encoding(IntEnum):
    """``hipblasLtInt4Encoding_t``: how the int4 weights of ``hipb_w4a16_mm`` are stored.

    Both keep the K order, element ``2j`` of a row in the low nibble of byte ``j``.
    SIGNED holds two's-complement values in [-8, 7]; UNSIGNED_BIAS8 holds [0, 15]
    with an implicit zero-point of 8 when there are no zero-points (the GPTQ and
    compressed-tensors ``qweight`` layout).
    """

    SIGNED = 0
    UNSIGNED_BIAS8 = 1


# hipBLASLt places the zero-points at the next 256-byte boundary after the scales.
HIPB_W4A16_ZERO_POINT_ALIGNMENT = 256


@compile_ops("module_hipbsolgemm", fc_name="hipb_w4a16_is_supported")
def _hipb_w4a16_is_supported(
    bf16: bool,
    n: int,
    k: int,
    lda: int,
    ldb: int,
    ldd: int,
    group_size: int,
    has_zp: bool,
    int4_encoding: int,
    batch_sizes: list[int],
) -> bool: ...


def hipb_w4a16_is_supported(
    dtype: torch.dtype,
    n: int,
    k: int,
    group_size: int,
    has_zp: bool,
    lda: int | None = None,
    int4_encoding: int = int(HipblasLtInt4Encoding.UNSIGNED_BIAS8),
    batch_sizes: tuple[int, ...] = (1, 8192),
    ldb: int | None = None,
    ldd: int | None = None,
) -> bool:
    """Whether ``hipb_w4a16_mm`` can run this layer on the current device.

    Asks hipBLASLt's heuristic for a kernel at each of ``batch_sizes``, so the
    answer reflects the hipBLASLt library actually loaded for this GPU: only
    architectures that ship w4a16 kernels say yes. Requires
    ``hipb_create_extension()`` and the extension's device to be current.
    ``lda`` is the weight row stride in int4
    elements, ``k`` when the rows are not padded; ``ldb`` and ``ldd`` are the
    row strides of ``x`` and of the output in elements, ``k`` and ``n`` by
    default.
    """
    if dtype not in (torch.float16, torch.bfloat16):
        return False
    return _hipb_w4a16_is_supported(
        dtype == torch.bfloat16,
        n,
        k,
        k if lda is None else lda,
        k if ldb is None else ldb,
        n if ldd is None else ldd,
        group_size,
        has_zp,
        int(int4_encoding),
        list(batch_sizes),
    )


def hipb_w4a16_scale_buffer(
    w_scales: torch.Tensor, w_zeros: torch.Tensor | None = None
) -> torch.Tensor:
    """The ``scale`` argument of ``hipb_w4a16_mm``.

    ``w_scales`` is ``[N, K // group_size]`` float16/bfloat16, the activation
    dtype. Without zero-points it is returned as a dense tensor. With ``w_zeros``, ``[ceil(N / 8),
    K // group_size]`` int32 words holding the zero-point of row ``n`` in bits
    ``4 * (n % 8)`` to ``4 * (n % 8) + 3``, returns one uint8 buffer with the
    scales followed, at the next 256-byte boundary, by the zero-points.
    """
    # hipBLASLt reads 16-bit scales; once packed as bytes their dtype is gone.
    if w_scales.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"w_scales must be float16 or bfloat16, got {w_scales.dtype}")
    # A contiguous tensor with a size-1 dim may keep an arbitrary stride there
    # (e.g. a transposed [1, N] scale); flattening gives a unit-stride view, which
    # dtype reinterpretation and the kernel's dense indexing both require.
    shape = w_scales.shape
    w_scales = w_scales.contiguous().flatten()
    if w_zeros is None:
        return w_scales.reshape(shape)
    n, num_groups = shape
    if w_zeros.shape != ((n + 7) // 8, num_groups) or w_zeros.dtype != torch.int32:
        raise ValueError(
            f"w_zeros must be int32 [{(n + 7) // 8}, {num_groups}], got "
            f"{w_zeros.dtype} {list(w_zeros.shape)}"
        )
    scale_bytes = w_scales.numel() * w_scales.element_size()
    zp_offset = -(-scale_bytes // HIPB_W4A16_ZERO_POINT_ALIGNMENT) * (
        HIPB_W4A16_ZERO_POINT_ALIGNMENT
    )
    zp_bytes = w_zeros.contiguous().flatten().view(torch.uint8)
    buf = torch.zeros(
        zp_offset + zp_bytes.numel(), dtype=torch.uint8, device=w_scales.device
    )
    buf[:scale_bytes].copy_(w_scales.view(torch.uint8))
    buf[zp_offset:].copy_(zp_bytes)
    return buf


@compile_ops("module_hipbsolgemm", fc_name="hipb_w4a16_mm", develop=True)
def _hipb_w4a16_mm(
    x: torch.Tensor,
    w_q: torch.Tensor,
    scale: torch.Tensor,
    result: torch.Tensor,
    workspace: torch.Tensor | None,
    bias: torch.Tensor | None,
    group_size: int,
    has_zp: bool,
    int4_encoding: int,
) -> bool: ...


@compile_ops("module_hipbsolgemm", fc_name="hipb_w4a16_workspace_size", develop=True)
def _hipb_w4a16_workspace_size(
    x: torch.Tensor,
    w_q: torch.Tensor,
    result: torch.Tensor,
    group_size: int,
    has_zp: bool,
    int4_encoding: int,
    has_bias: bool,
) -> int: ...


def _hipb_w4a16_mm_out(
    x: torch.Tensor,
    w_q: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    has_zp: bool,
    int4_encoding: int,
) -> None:
    # Stream-K kernels keep partial sums in the workspace. Taking it from torch's
    # allocator per call ties it to the current stream, so launches on different
    # streams never share scratch memory.
    nbytes = _hipb_w4a16_workspace_size(
        x, w_q, out, group_size, has_zp, int4_encoding, bias is not None
    )
    workspace = (
        torch.empty(nbytes, dtype=torch.uint8, device=x.device) if nbytes else None
    )
    # The bias goes to hipBLASLt's epilogue when it has a kernel with one.
    fused = _hipb_w4a16_mm(
        x, w_q, scale, out, workspace, bias, group_size, has_zp, int4_encoding
    )
    if bias is not None and not fused:
        out.add_(bias)


def _hipb_w4a16_mm_out_fake(
    x: torch.Tensor,
    w_q: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    has_zp: bool,
    int4_encoding: int,
) -> None:
    return None


# Writes into `out` and returns nothing, so the op never aliases an input in its
# output; `hipb_w4a16_mm` below allocates and returns the buffer.
direct_register_custom_op(
    "hipb_w4a16_mm_out",
    _hipb_w4a16_mm_out,
    ["out"],
    fake_impl=_hipb_w4a16_mm_out_fake,
)


def hipb_w4a16_mm(
    x: torch.Tensor,
    w_q: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    has_zp: bool = False,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    int4_encoding: int = int(HipblasLtInt4Encoding.UNSIGNED_BIAS8),
) -> torch.Tensor:
    """``x @ dequant(w_q).T`` on hipBLASLt, with int4 weights and fp16/bf16 activations.

    ``x`` is ``[M, K]`` fp16/bf16. ``w_q`` is ``[N, K // 2]`` uint8/int8 in K
    order (see ``HipblasLtInt4Encoding``, default UNSIGNED_BIAS8). The rows of
    ``x``, ``w_q`` and ``out`` may be padded (row stride above the row size).
    ``scale`` comes from ``hipb_w4a16_scale_buffer``, and ``has_zp`` says
    whether it carries zero-points. The dequantized weight is
    ``(q - zero) * scale``. ``bias``: an ``[N]`` tensor in ``x``'s dtype, added
    by hipBLASLt's epilogue when it has a kernel with one. ``out``: the
    ``[M, N]`` tensor to write instead of allocating; returned. Check ``hipb_w4a16_is_supported`` first. Call with
    ``x``'s device current: the launch stream is that device's current stream.
    """
    result = (
        torch.empty((x.size(0), w_q.size(0)), dtype=x.dtype, device=x.device)
        if out is None
        else out
    )
    torch.ops.aiter.hipb_w4a16_mm_out(
        x, w_q, scale, result, bias, group_size, has_zp, int(int4_encoding)
    )
    return result


def gen_hipb_grouped_mm_fake_tensor(
    a: torch.Tensor,
    b: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor,
    a_is_transposed: bool = False,
    bias: torch.Tensor | None = None,
    solution_index: int = -1,
) -> None:
    return None


@compile_ops(
    "module_hipb_groupedgemm",
    fc_name="hipb_grouped_mm",
    gen_fake=gen_hipb_grouped_mm_fake_tensor,
)
def _hipb_grouped_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor,
    a_is_transposed: bool = False,
    bias: torch.Tensor | None = None,
    solution_index: int = -1,
) -> None: ...


@torch_compile_guard(
    device="cuda",
    mutates_args=["out"],
    gen_fake=gen_hipb_grouped_mm_fake_tensor,
)
def hipb_grouped_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor,
    a_is_transposed: bool = False,
    bias: torch.Tensor | None = None,
    solution_index: int = -1,
) -> None:
    _hipb_grouped_mm(a, b, cu_seqlens, out, a_is_transposed, bias, solution_index)


def gen_hipb_multistream_mm_fake_tensor(
    a: torch.Tensor,
    b: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor,
    a_is_transposed: bool = False,
    bias: torch.Tensor | None = None,
    b_is_transposed: bool = False,
) -> None:
    return None


@compile_ops(
    "module_hipb_groupedgemm",
    fc_name="hipb_multistream_mm",
    gen_fake=gen_hipb_multistream_mm_fake_tensor,
)
def _hipb_multistream_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor,
    a_is_transposed: bool = False,
    bias: torch.Tensor | None = None,
    b_is_transposed: bool = False,
) -> None: ...


@torch_compile_guard(
    device="cuda",
    mutates_args=["out"],
    gen_fake=gen_hipb_multistream_mm_fake_tensor,
)
def hipb_multistream_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor,
    a_is_transposed: bool = False,
    bias: torch.Tensor | None = None,
    b_is_transposed: bool = False,
) -> None:
    _hipb_multistream_mm(
        a,
        b,
        cu_seqlens,
        out,
        a_is_transposed,
        bias,
        b_is_transposed,
    )


@compile_ops("module_rocsolgemm")
def rocb_create_extension() -> None: ...


@compile_ops("module_rocsolgemm")
def rocb_destroy_extension() -> None: ...


def gen_rocb_mm_fake_tensor(
    arg0: torch.Tensor, arg1: torch.Tensor, arg2: int
) -> torch.Tensor:
    # gemm out = (M, N) = (arg0.size(0), arg1.size(1)).
    return torch.empty(
        (arg0.size(0), arg1.size(1)), dtype=arg0.dtype, device=arg0.device
    )


# torch-free kernel entry: writes into caller-allocated `result`.
@compile_ops("module_rocsolgemm", fc_name="rocb_mm", develop=True)
def _rocb_mm(
    arg0: torch.Tensor, arg1: torch.Tensor, result: torch.Tensor, arg2: int
) -> None: ...


def rocb_mm(arg0: torch.Tensor, arg1: torch.Tensor, arg2: int) -> torch.Tensor:
    result = torch.empty(
        (arg0.size(0), arg1.size(1)), dtype=arg0.dtype, device=arg0.device
    )
    _rocb_mm(arg0, arg1, result, arg2)
    return result


direct_register_custom_op(
    "rocb_mm",
    rocb_mm,
    [],
    fake_impl=gen_rocb_mm_fake_tensor,
)


@compile_ops("module_rocsolgemm", fc_name="rocb_findallsols", develop=True)
def _rocb_findallsols(
    arg0: torch.Tensor, arg1: torch.Tensor, result: torch.Tensor
) -> list[int]: ...


def rocb_findallsols(arg0: torch.Tensor, arg1: torch.Tensor) -> list[int]:
    result = torch.empty(
        (arg0.size(0), arg1.size(1)), dtype=arg0.dtype, device=arg0.device
    )
    return _rocb_findallsols(arg0, arg1, result)
