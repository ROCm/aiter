# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
import triton
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16w16 import (
    _gemm_a16_w16_kernel,
    _get_config as _get_triton_config,
)
from aiter.ops.triton._triton_kernels.common.splitk_reduce import (
    _gemm_splitk_reduce_kernel,
)
from aiter.ops.triton._triton_kernels.activation import _get_activation_from_str
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.common_utils import serialize_dict, deserialize_str
from aiter.jit.utils.torch_guard import torch_compile_guard

_LOGGER = AiterTritonLogger()

_GLUON_SUPPORTED_ARCHS = ("gfx1250",)


def _is_gluon_available():
    """Check if the gluon backend is available for the current GPU architecture."""
    try:
        return any(supported in get_arch() for supported in _GLUON_SUPPORTED_ARCHS)
    except Exception:
        return False


def gemm_a16w16_fake_tensor(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[str] = None,
    activation: Optional[str] = None,
    skip_reduce: Optional[bool] = False,
    kernel_type: str = "bandwidth_bound",
    backend: Optional[str] = None,
) -> torch.Tensor:
    M, K = x.shape
    N, _ = w.shape
    # [triton only] split-K with skip_reduce returns the unreduced partials.
    if skip_reduce:
        cfg = deserialize_str(config) if config else _get_triton_config(M, N, K)[0]
        num_ksplit = cfg.get("NUM_KSPLIT", 1)
        if num_ksplit > 1:
            return torch.empty((num_ksplit, M, N), dtype=torch.float32, device=x.device)
    if y is not None:
        return y
    return torch.empty((M, N), dtype=dtype, device=x.device)


@torch_compile_guard(gen_fake=gemm_a16w16_fake_tensor)
def gemm_a16w16_(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[str] = None,
    activation: Optional[str] = None,
    skip_reduce: Optional[bool] = False,
    kernel_type: str = "bandwidth_bound",
    backend: Optional[str] = None,
) -> torch.Tensor:
    """
    Computes 16 bit matrix multiplication Y = X @ W^T

    Uses the gluon backend automatically on supported architectures (gfx1250)
    and the triton backend everywhere else. Pass ``backend`` to force a choice.

    Args:
        x (torch.Tensor): Input matrix with shape (M, K).
        w (torch.Tensor): Weight matrix with shape (N, K), internally transposed.
        bias (Optional[torch.Tensor]): Bias vector with shape (N,).
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N).
        config (Optional[str]): Serialized kernel tuning parameters.
        activation (Optional[str]): Activation function ("gelu", "gelu_tanh", "silu",
            "silu_exp2", "relu").
        skip_reduce (Optional[bool]): [triton only] Skip reduction of split-K partial
            results. Returns shape (NUM_KSPLIT, M, N) instead of (M, N).
        kernel_type (str): [gluon only] Kernel variant ("bandwidth_bound", "compute_bound").
        backend (Optional[str]): "triton", "gluon", or None (auto-detect).

    Returns:
        torch.Tensor: Output with shape (M, N) or (NUM_KSPLIT, M, N) if skip_reduce=True.
    """
    config = deserialize_str(config) if config is not None else None

    if backend is None:
        backend = "gluon" if _is_gluon_available() else "triton"
    backend = backend.lower()
    assert backend in (
        "triton",
        "gluon",
    ), f"Unknown backend '{backend}', must be 'triton' or 'gluon'"

    if backend == "gluon":
        assert (
            _is_gluon_available()
        ), f"Gluon backend requires one of {_GLUON_SUPPORTED_ARCHS}, got '{get_arch()}'"
        from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16 import (
            _KERNEL_MAP,
            create_shared_layouts,
            create_wmma_layouts,
        )

        assert (
            kernel_type in _KERNEL_MAP
        ), f"Unknown kernel_type '{kernel_type}', must be one of {list(_KERNEL_MAP.keys())}"
        _LOGGER.info(
            f"GEMM_A16W16 [gluon/gfx1250]: x={tuple(x.shape)} w={tuple(w.shape)} "
            f"kernel={kernel_type}"
        )
        assert x.dtype in (
            torch.float16,
            torch.bfloat16,
        ), f"Activations (x) must be fp16 or bf16, got {x.dtype}"
        assert w.dtype in (
            torch.float16,
            torch.bfloat16,
        ), f"Weights (w) must be fp16 or bf16, got {w.dtype}"
        assert x.shape[1] == w.shape[1], "Incompatible matrix shapes."

        M, _ = x.shape
        K = x.stride(0)
        N, _ = w.shape

        if config is None:
            config, _ = get_gemm_config("GEMM-A16W16", M, N, K)

        BLOCK_M = config["BLOCK_M"]
        BLOCK_N = config["BLOCK_N"]
        BLOCK_K = config["BLOCK_K"]
        NUM_BUFFERS = config.get("NUM_BUFFERS", 2)
        num_warps = config["num_warps"]

        # The kernels walk K with update_tensor_descriptor(add_offsets=...),
        # which advances the load position without shrinking the descriptor's
        # OOB bound, so a partial last K-tile would read past the end of K.
        # Require K to be a multiple of BLOCK_K rather than padding here. (M and
        # N may be unaligned: their descriptor bounds + store mask zero-fill the
        # partial tiles.)
        assert (
            K % BLOCK_K == 0
        ), f"K ({K}) must be a multiple of BLOCK_K ({BLOCK_K}) for the gluon a16w16 GEMM"

        # Clamp the software-pipeline depth to the number of K-tiles.
        #
        # The prologue/epilogue walk a fixed number of K-tiles determined by
        # NUM_BUFFERS, independent of how many real tiles exist. If NUM_BUFFERS
        # exceeds that count the pipeline loop counts go negative, so cap the
        # depth at the real tile count. Variants differ in reach and in the
        # minimum depth they require:
        #   bandwidth_bound : reaches num_k_tiles -> cap = num_k_tiles
        #   compute_bound : preloads one tile ahead (needs num_k_tiles >= NB + 1)
        #                   -> cap = num_k_tiles - 1
        num_k_tiles = triton.cdiv(K, BLOCK_K)
        _MIN_BUFFERS = {"bandwidth_bound": 1, "compute_bound": 2}
        _DEPTH_SLACK = {"compute_bound": 1}

        # Fall back to the bandwidth_bound kernel when the requested variant cannot
        # satisfy its minimum pipeline depth for this K. The bandwidth_bound kernel
        # has no such floor (min depth 1) and is valid for every K, so we downgrade
        # rather than error.
        depth_cap = num_k_tiles - _DEPTH_SLACK.get(kernel_type, 0)
        if depth_cap < _MIN_BUFFERS[kernel_type]:
            needed = _MIN_BUFFERS[kernel_type] + _DEPTH_SLACK.get(kernel_type, 0)
            _LOGGER.info(
                f"GEMM_A16W16 [gluon/gfx1250]: kernel_type='{kernel_type}' needs "
                f"num_k_tiles>={needed} but num_k_tiles={num_k_tiles} "
                f"(K={K}, BLOCK_K={BLOCK_K}); falling back to kernel_type='bandwidth_bound'."
            )
            kernel_type = "bandwidth_bound"
            depth_cap = num_k_tiles  # bandwidth_bound: depth slack 0, min depth 1

        NUM_BUFFERS = min(NUM_BUFFERS, depth_cap)

        w = w.T

        # Operand layout in BLAS TT/TN/NT/NN form: 'T' (row-major, trailing dim
        # contiguous) or 'N' (column-major, leading dim contiguous). First char
        # is x (A), second is w (B, after the internal transpose above).
        if x.stride(1) == 1:
            layout = "T"
        elif x.stride(0) == 1:
            layout = "N"
        else:
            raise ValueError(
                f"x must be contiguous in at least one dimension, got strides {x.stride()}"
            )

        if w.stride(1) == 1:
            layout += "T"
        elif w.stride(0) == 1:
            layout += "N"
        else:
            raise ValueError(
                f"w must be contiguous in at least one dimension, got strides {w.stride()}"
            )

        if y is None:
            y = torch.empty((M, N), dtype=dtype, device=x.device)

        wmma_layout, operand_a, operand_b = create_wmma_layouts(num_warps)
        shared_a, shared_b = create_shared_layouts(BLOCK_M, BLOCK_N, BLOCK_K, layout)

        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)

        _KERNEL_MAP[kernel_type][grid](
            x,
            w,
            y,
            bias,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            w.stride(0),
            w.stride(1),
            y.stride(0),
            y.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            NUM_BUFFERS=NUM_BUFFERS,
            LAYOUT=layout,
            SHARED_LAYOUT_A=shared_a,
            SHARED_LAYOUT_B=shared_b,
            WMMA_LAYOUT=wmma_layout,
            OPERAND_LAYOUT_A=operand_a,
            OPERAND_LAYOUT_B=operand_b,
            activation=_get_activation_from_str(activation) if activation else None,
            USE_ACTIVATION=activation is not None,
            ADD_BIAS=(bias is not None),
            num_warps=num_warps,
        )

        return y

    _LOGGER.info(f"GEMM_A16W16 [triton]: x={tuple(x.shape)} w={tuple(w.shape)}")

    assert x.shape[1] == w.shape[1], "Incompatible matrix shapes."

    M, K = x.shape
    N, K = w.shape
    w = w.T

    if config is None:
        config, _ = _get_triton_config(M, N, K)

    if y is None and (config["NUM_KSPLIT"] == 1 or not skip_reduce):
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    if config["NUM_KSPLIT"] > 1:
        y_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N),
            dtype=torch.float32,
            device=y.device if y is not None else x.device,
        )
    else:
        y_pp = None

    grid = lambda META: (  # noqa: E731
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )
    _gemm_a16_w16_kernel[grid](
        x,
        w,
        bias,
        y if config["NUM_KSPLIT"] == 1 else y_pp,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
        y.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
        y.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
        activation=_get_activation_from_str(activation) if activation else "",
        use_activation=activation is not None,
        ADD_BIAS=(bias is not None),
        SKIP_REDUCE=skip_reduce,
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        if skip_reduce:
            return y_pp

        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _gemm_splitk_reduce_kernel[grid_reduce](
            y_pp,
            y,
            bias,
            M,
            N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            y.stride(0),
            y.stride(1),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(config["NUM_KSPLIT"]),
            ADD_BIAS=(bias is not None),
            activation=_get_activation_from_str(activation) if activation else "",
            use_activation=activation is not None,
            KERNEL_NAME="_gemm_a16w16_reduce_kernel",
        )

    return y


def gemm_a16w16(
    x,
    w,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    activation: Optional[str] = None,
    skip_reduce: Optional[bool] = False,
    kernel_type: str = "bandwidth_bound",
    backend: Optional[str] = None,
):
    """
    Computes 16 bit matrix multiplication Y = X @ W^T

    Uses the gluon backend automatically on supported architectures (gfx1250)
    and the triton backend everywhere else. Pass ``backend`` to force a choice.
    See ``gemm_a16w16_`` for the full argument description; ``config`` is a dict
    here and is serialized before dispatch so the op is torch.compile-traceable.
    """
    # dtype must be a torch.dtype at the custom-op boundary (callers sometimes
    # pass a placeholder when a preallocated y already fixes the output dtype).
    if not isinstance(dtype, torch.dtype):
        dtype = torch.bfloat16
    config_hashable = serialize_dict(config) if config else None
    return gemm_a16w16_(
        x,
        w,
        bias,
        dtype,
        y,
        config_hashable,
        activation,
        skip_reduce,
        kernel_type,
        backend,
    )
