# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.triton._triton_kernels.activation import _get_activation_from_str
from aiter.ops.triton._triton_kernels.common.splitk_reduce import (
    _gemm_splitk_reduce_kernel,
)
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16w16 import (
    _gemm_a16_w16_kernel,
)
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16w16 import (
    _get_config as _get_triton_config,
)
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16w16_persistent import (
    gemm_a16w16_persistent_kernel_ as _triton_persistent_kernel,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.common_utils import deserialize_str, serialize_dict
from aiter.ops.triton.utils.gemm_config_utils import (
    compute_splitk_params,
    get_gemm_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

_GLUON_SUPPORTED_ARCHS = ("gfx1250",)


def _is_gluon_available():
    """Check if the gluon backend is available for the current GPU architecture."""
    try:
        return any(supported in get_arch() for supported in _GLUON_SUPPORTED_ARCHS)
    except Exception:  # noqa: BLE001
        return False


def gemm_a16w16_fake_tensor(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None = None,
    dtype: torch.dtype | None = torch.bfloat16,
    y: torch.Tensor | None = None,
    config: str | None = None,
    activation: str | None = None,
    skip_reduce: bool | None = False,
    kernel_type: str = "bandwidth_bound",
    backend: str | None = None,
    persistent: bool = False,
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
    bias: torch.Tensor | None = None,
    dtype: torch.dtype | None = torch.bfloat16,
    y: torch.Tensor | None = None,
    config: str | None = None,
    activation: str | None = None,
    skip_reduce: bool | None = False,
    kernel_type: str = "bandwidth_bound",
    backend: str | None = None,
    persistent: bool = False,
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
            Ignored when persistent=True; the persistent kernels have one variant.
        backend (Optional[str]): "triton", "gluon", or None (auto-detect).
        persistent (bool): Use the persistent kernel, which launches one workgroup
            per CU and walks a strided subset of the output tiles, instead of one
            workgroup per tile. Reads the GEMM-A16W16-PERSISTENT config family and
            does not support split-K (so it is incompatible with skip_reduce).

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

    if persistent:
        assert x.shape[1] == w.shape[1], "Incompatible matrix shapes."
        assert not skip_reduce, (
            "persistent=True does not support skip_reduce; the persistent kernels "
            "have no split-K path to leave unreduced"
        )
        M, K = x.shape
        N, _ = w.shape

        if config is None:
            config, _ = get_gemm_config("GEMM-A16W16-PERSISTENT", M, N, K)
            if backend == "triton":
                config = compute_splitk_params(config, K)

        assert config.get("NUM_KSPLIT", 1) == 1, (
            f"persistent=True does not support split-K yet (got NUM_KSPLIT="
            f"{config.get('NUM_KSPLIT')}); call without persistent=True instead"
        )

        if backend == "gluon":
            assert (
                _is_gluon_available()
            ), f"Gluon backend requires one of {_GLUON_SUPPORTED_ARCHS}, got '{get_arch()}'"
            import triton.experimental.gluon.language as gl

            from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16_persistent import (
                gemm_a16w16_persistent_kernel_ as _gluon_persistent_kernel,
            )

            _LOGGER.info(
                f"GEMM_A16W16 [gluon/gfx1250, persistent]: x={tuple(x.shape)} "
                f"w={tuple(w.shape)}"
            )
            assert x.dtype in (
                torch.float16,
                torch.bfloat16,
            ), f"Activations (x) must be fp16 or bf16, got {x.dtype}"
            assert w.dtype in (
                torch.float16,
                torch.bfloat16,
            ), f"Weights (w) must be fp16 or bf16, got {w.dtype}"

            BLOCK_M = config["BLOCK_M"]
            BLOCK_N = config["BLOCK_N"]
            BLOCK_K = config["BLOCK_K"]
            NUM_BUFFERS = config.get("NUM_BUFFERS", 2)
            GROUP_SIZE_M = config.get("GROUP_SIZE_M", 1)
            num_warps = config["num_warps"]

            w = w.T

            # Clamp the pipeline depth
            num_k_tiles = triton.cdiv(K, BLOCK_K)
            NUM_BUFFERS = max(2, min(NUM_BUFFERS, num_k_tiles + 1))

            if y is None:
                y = torch.empty((M, N), dtype=dtype, device=x.device)

            assert x.stride(1) == 1, (
                f"gluon persistent gemm requires x row-major (M, K), got strides "
                f"{x.stride()}"
            )
            
            if w.stride(1) == 1:
                TRANSPOSE = True
            elif w.stride(0) == 1:
                TRANSPOSE = False
            else:
                raise ValueError(
                    f"w must be contiguous in at least one dimension, got strides "
                    f"{w.stride()}"
                )

            shared_a = gl.PaddedSharedLayout.with_identity_for(
                [[BLOCK_K, 8]], [BLOCK_M, BLOCK_K], [1, 0]
            )
            if TRANSPOSE:
                shared_b = gl.PaddedSharedLayout.with_identity_for(
                    [[BLOCK_N, 16]], [BLOCK_K, BLOCK_N], [1, 0]
                )
            else:
                shared_b = gl.PaddedSharedLayout.with_identity_for(
                    [[BLOCK_K, 8]], [BLOCK_N, BLOCK_K], [1, 0]
                )

            warp_bases = tuple(
                (0, 1) if i == 0 else (1 << (i - 1), 0)
                for i in range(num_warps.bit_length() - 1)
            )

            # Persistent, NUM_WGS processes num_tiles
            _LOGGER.info(
                f"GEMM_A16W16 [gluon, persistent]: x={tuple(x.shape)} w={tuple(w.shape)}"
            )
            NUM_WGS = torch.cuda.get_device_properties(x.device).multi_processor_count
            num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)

            _gluon_persistent_kernel[(min(NUM_WGS, num_tiles),)](
                x,
                w,
                bias,
                y,
                M,
                N,
                K,
                num_tiles,
                x.stride(0),
                x.stride(1),
                w.stride(0),
                w.stride(1),
                y.stride(0),
                y.stride(1),
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_K=BLOCK_K,
                GROUP_SIZE_M=GROUP_SIZE_M,
                NUM_BUFFERS=NUM_BUFFERS,
                SHARED_LAYOUT_A=shared_a,
                SHARED_LAYOUT_B=shared_b,
                WARP_BASES=warp_bases,
                TRANSPOSE=TRANSPOSE,
                activation=_get_activation_from_str(activation) if activation else None,
                USE_ACTIVATION=activation is not None,
                ADD_BIAS=(bias is not None),
                NUM_WGS=NUM_WGS,
                num_warps=num_warps,
            )

            return y

        _LOGGER.info(
            f"GEMM_A16W16 [triton, persistent]: x={tuple(x.shape)} w={tuple(w.shape)}"
        )

        w = w.T

        if y is None:
            y = torch.empty((M, N), dtype=dtype, device=x.device)

        # Persistent, one WG per CU
        NUM_WGS = torch.cuda.get_device_properties(x.device).multi_processor_count
        num_tiles = triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(
            N, config["BLOCK_SIZE_N"]
        )
        _triton_persistent_kernel[(min(NUM_WGS, num_tiles),)](
            x,
            w,
            bias,
            y,
            M,
            N,
            K,
            num_tiles,
            x.stride(0),
            x.stride(1),
            w.stride(0),
            w.stride(1),
            0,  # stride_ck
            y.stride(0),
            y.stride(1),
            activation=_get_activation_from_str(activation) if activation else "",
            use_activation=activation is not None,
            ADD_BIAS=(bias is not None),
            SKIP_REDUCE=False,
            NUM_WGS=NUM_WGS,
            **config,
        )

        return y

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

        M, K = x.shape
        N, _ = w.shape

        if config is None:
            config, _ = get_gemm_config("GEMM-A16W16", M, N, K)

        kernel_type_from_config = config.pop("kernel_type", None)
        if kernel_type_from_config is not None:
            kernel_type = kernel_type_from_config

        BLOCK_M = config["BLOCK_M"]
        BLOCK_N = config["BLOCK_N"]
        BLOCK_K = config["BLOCK_K"]
        NUM_BUFFERS = config.get("NUM_BUFFERS", 2)
        num_warps = config["num_warps"]

        num_k_tiles = triton.cdiv(K, BLOCK_K)
        _MIN_BUFFERS = {"bandwidth_bound": 1, "compute_bound": 2}
        _DEPTH_SLACK = {"compute_bound": 2}

        if kernel_type_from_config is None:
            depth_cap = num_k_tiles - _DEPTH_SLACK.get(kernel_type, 0)
            if depth_cap < _MIN_BUFFERS[kernel_type]:
                needed = _MIN_BUFFERS[kernel_type] + _DEPTH_SLACK.get(kernel_type, 0)
                _LOGGER.warning(
                    f"GEMM_A16W16 [gluon/gfx1250]: kernel_type='{kernel_type}' needs "
                    f"num_k_tiles>={needed} but num_k_tiles={num_k_tiles} "
                    f"(K={K}, BLOCK_K={BLOCK_K}); falling back to kernel_type='bandwidth_bound'."
                )
                kernel_type = "bandwidth_bound"
                depth_cap = num_k_tiles
        else:
            depth_cap = num_k_tiles - _DEPTH_SLACK.get(kernel_type, 0)

        NUM_BUFFERS = min(NUM_BUFFERS, depth_cap)

        w = w.T

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

        _LOGGER.info(
            f"GEMM_A16W16 [gluon, non-persistent]: x={tuple(x.shape)} w={tuple(w.shape)}"
        )

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

    grid = lambda META: (
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
    bias: torch.Tensor | None = None,
    dtype: torch.dtype | None = torch.bfloat16,
    y: torch.Tensor | None = None,
    config: dict | None = None,
    activation: str | None = None,
    skip_reduce: bool | None = False,
    kernel_type: str = "bandwidth_bound",
    backend: str | None = None,
    persistent: bool = False,
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
        persistent,
    )
