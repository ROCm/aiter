# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os

import torch
import triton

from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.triton._triton_kernels.common.activation import _get_activation_from_str
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
    clamp_num_buffers,
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


_GLUON_MIN_BUFFERS = {"bandwidth_bound": 1, "compute_bound": 2}
_GLUON_DEPTH_SLACK = {"compute_bound": 2}


def _gluon_plan(
    M,
    N,
    K,
    x_strides,
    w_strides,
    elem_bits,
    kernel_type,
    config,
    create_shared_layouts,
    create_wmma_layouts,
):
    """Kernel, grid and launch parameters of the non-persistent gluon GEMM.

    ``w_strides`` are those of the (K, N) view the kernel reads, i.e. ``w.T``.
    Returns ``(kernel, grid, constexprs, num_warps, waves_per_eu)``; the
    constexprs exclude ``activation``/``USE_ACTIVATION``/``ADD_BIAS``.
    """
    from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16 import (
        _KERNEL_MAP,
    )

    if config is None:
        config, _ = get_gemm_config("GEMM-A16W16", M, N, K, backend="gluon")

    kernel_type_from_config = config.pop("kernel_type", None)
    if kernel_type_from_config is not None:
        kernel_type = kernel_type_from_config

    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]
    BLOCK_K = config["BLOCK_K"]
    NUM_BUFFERS = config.get("NUM_BUFFERS", 2)
    num_warps = config["num_warps"]
    waves_per_eu = config.get("waves_per_eu", 0)

    num_k_tiles = triton.cdiv(K, BLOCK_K)

    if kernel_type_from_config is None:
        depth_cap = num_k_tiles - _GLUON_DEPTH_SLACK.get(kernel_type, 0)
        if depth_cap < _GLUON_MIN_BUFFERS[kernel_type]:
            needed = _GLUON_MIN_BUFFERS[kernel_type] + _GLUON_DEPTH_SLACK.get(
                kernel_type, 0
            )
            _LOGGER.warning(
                "GEMM_A16W16 [gluon/gfx1250]: kernel_type='%s' needs "
                "num_k_tiles>=%s but num_k_tiles=%s (K=%s, BLOCK_K=%s); "
                "defaults to kernel_type='bandwidth_bound'.",
                kernel_type,
                needed,
                num_k_tiles,
                K,
                BLOCK_K,
            )
            kernel_type = "bandwidth_bound"
            depth_cap = num_k_tiles
    else:
        depth_cap = num_k_tiles - _GLUON_DEPTH_SLACK.get(kernel_type, 0)

    NUM_BUFFERS = min(NUM_BUFFERS, depth_cap)

    if x_strides[1] == 1:
        layout = "T"
    elif x_strides[0] == 1:
        layout = "N"
    else:
        raise ValueError(
            f"x must be contiguous in at least one dimension, got strides {tuple(x_strides)}"
        )

    if w_strides[1] == 1:
        layout += "T"
    elif w_strides[0] == 1:
        layout += "N"
    else:
        raise ValueError(
            f"w must be contiguous in at least one dimension, got strides {tuple(w_strides)}"
        )

    wmma_layout, operand_a, operand_b = create_wmma_layouts(num_warps)
    shared_a, shared_b = create_shared_layouts(
        BLOCK_M, BLOCK_N, BLOCK_K, layout, elem_bits
    )

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)

    meta = {
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "NUM_BUFFERS": NUM_BUFFERS,
        "LAYOUT": layout,
        "SHARED_LAYOUT_A": shared_a,
        "SHARED_LAYOUT_B": shared_b,
        "WMMA_LAYOUT": wmma_layout,
        "OPERAND_LAYOUT_A": operand_a,
        "OPERAND_LAYOUT_B": operand_b,
    }
    return _KERNEL_MAP[kernel_type], grid, meta, num_warps, waves_per_eu


# ---------------------------------------------------------------------------
# AITER_GEMM_FAST_LAUNCH=1: host fast path for the eager gfx1250 gluon GEMM.
#
# The default path pays, per call, two torch custom-op dispatches, a JSON
# round trip of the config, the config lookup + copy, a rebuild of five gluon
# layout objects (~35 us: PaddedSharedLayout.with_identity_for is slow), and
# JITFunction.run (env-backed knob reads, binder, hashing a specialization
# that holds those layout objects, str(options), ...).
#
# The fast path computes the plan once per (shape, strides, dtypes, kernel
# type, bias) and binds the compiled kernel once per Triton specialization.
# The specialization is still computed every call by Triton's own binder, so
# the kernel picked is exactly the one JITFunction.run would pick; only the
# constexpr part (fixed by the plan) is left out of the lookup key.  Anything
# the fast path does not cover (config/activation/skip_reduce/persistent/
# explicit backend, non-2D, torch.compile tracing, Triton pre-run or
# stage-inspection hooks) takes the default path.
# ---------------------------------------------------------------------------
_FAST_LAUNCH = os.environ.get("AITER_GEMM_FAST_LAUNCH", "0") == "1"

# Leading runtime (non-constexpr) parameters of both gluon a16w16 kernels.
_GLUON_RUNTIME_ARGS = (
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_ak",
    "stride_bk",
    "stride_bn",
    "stride_cm",
    "stride_cn",
)
_N_RT = len(_GLUON_RUNTIME_ARGS)

# pkey -> (fn, grid, constexpr tail, num_warps, waves_per_eu, binder, device)
_gluon_fast_plans = {}
# (pkey, runtime specialization) -> CompiledKernel
_gluon_fast_kernels = {}

_fast_rt = None


def _fast_runtime():
    global _fast_rt
    if _fast_rt is None:
        from triton import knobs
        from triton.runtime.driver import driver

        _fast_rt = (knobs, driver.active)
    return _fast_rt


def _build_gluon_fast_plan(x, w, M, N, K, has_bias, kernel_type):
    from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16 import (
        _KERNEL_MAP,
        create_shared_layouts,
        create_wmma_layouts,
    )

    # The same checks, in the same order, as the default path.
    assert _is_gluon_available()
    assert x.shape[1] == w.shape[1], (
        f"x is (M, K)={tuple(x.shape)} and w is "
        f"(N, K)={tuple(w.shape)}, K must match"
    )
    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"Activations (x) must be fp16 or bf16, got {x.dtype}"
    assert w.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"Weights (w) must be fp16 or bf16, got {w.dtype}"
    assert (
        kernel_type in _KERNEL_MAP
    ), f"Unknown kernel_type '{kernel_type}', must be one of {list(_KERNEL_MAP.keys())}"

    wt = w.T
    fn, grid, meta, num_warps, waves_per_eu = _gluon_plan(
        M,
        N,
        K,
        x.stride(),
        wt.stride(),
        x.element_size() * 8,
        kernel_type,
        None,
        create_shared_layouts,
        create_wmma_layouts,
    )
    meta["activation"] = None
    meta["USE_ACTIVATION"] = False
    meta["ADD_BIAS"] = has_bias
    names = tuple(fn.arg_names)
    if names[:_N_RT] != _GLUON_RUNTIME_ARGS or set(names[_N_RT:]) != set(meta):
        return None
    tail = tuple(meta[n] for n in names[_N_RT:])
    _knobs, drv = _fast_runtime()
    device = drv.get_current_device()
    binder = fn.device_caches[device][4]
    return (fn, grid, tail, num_warps, waves_per_eu, binder, device)


def _gemm_a16w16_gluon_fast(x, w, bias, dtype, y, kernel_type):
    """Default-path-equivalent gluon launch; None means "use the default path"."""
    if x.dim() != 2 or w.dim() != 2:
        return None
    M, K = x.shape
    N = w.shape[0]
    xs = x.stride()
    ws = w.stride()
    pkey = (M, N, K, w.shape[1], xs, ws, x.dtype, w.dtype, kernel_type, bias is None)
    plan = _gluon_fast_plans.get(pkey)
    if plan is None:
        plan = _build_gluon_fast_plan(x, w, M, N, K, bias is not None, kernel_type)
        if plan is None:
            return None
        _gluon_fast_plans[pkey] = plan
    fn, grid, tail, num_warps, waves_per_eu, binder, device = plan

    knobs, drv = _fast_runtime()
    rt = knobs.runtime
    if fn.pre_run_hooks or rt.add_stages_inspection_hook is not None:
        return None
    if drv.get_current_device() != device:
        return None

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)
    ys = y.stride()
    # w is passed for its w.T view: the kernel only sees the data pointer (and
    # the storage, for buffer ops), which the two share; strides are w.T's.
    args = (x, w, y, bias, M, N, K, xs[0], xs[1], ws[1], ws[0], ys[0], ys[1]) + tail

    # Same options JITFunction.run passes to the binder.
    debug = fn.debug or rt.debug
    instrumentation_mode = knobs.compilation.instrumentation_mode
    fpsan = knobs.compilation.fpsan_homomorphic_casts
    _, specialization, _ = binder(
        *args,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
        debug=debug,
        instrumentation_mode=instrumentation_mode,
        fpsan_homomorphic_casts=fpsan,
    )
    kkey = (pkey, tuple(specialization[:_N_RT]), debug, instrumentation_mode, fpsan)
    kernel = _gluon_fast_kernels.get(kkey)
    if kernel is None:
        # First sighting: the regular launch, which compiles or finds the
        # kernel in Triton's own cache, and hands it back for reuse.
        kernel = fn[grid](*args, num_warps=num_warps, waves_per_eu=waves_per_eu)
        if kernel is not None:
            _gluon_fast_kernels[kkey] = kernel
        return y

    not_present = object()
    for (name, _), (val, globals_dict) in fn.used_global_vals.items():
        if globals_dict.get(name, not_present) != val:
            raise RuntimeError(
                f"Global variable {name} has changed since we compiled this kernel"
            )
    stream = drv.get_current_stream(device)
    kernel.run(
        grid[0],
        grid[1],
        1,
        stream,
        kernel.function,
        kernel.packed_metadata,
        kernel.launch_metadata(grid, stream, *args),
        rt.launch_enter_hook,
        rt.launch_exit_hook,
        *args,
    )
    return y


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
        backend (Optional[str]): "triton", "gluon", or None (auto-detect).
        persistent (bool): Use the persistent kernel, NUM_SMS workgroups

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

    assert x.shape[1] == w.shape[1], (
        f"x is (M, K)={tuple(x.shape)} and w is "
        f"(N, K)={tuple(w.shape)}, K must match"
    )

    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"Activations (x) must be fp16 or bf16, got {x.dtype}"
    assert w.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"Weights (w) must be fp16 or bf16, got {w.dtype}"

    if persistent:
        assert not skip_reduce, (
            "persistent=True does not support skip_reduce; the persistent kernels "
            "have no split-K path to leave unreduced"
        )
        M, K = x.shape
        N, _ = w.shape

        NUM_WGS = torch.cuda.get_device_properties(x.device).multi_processor_count

        if config is None:
            config, _ = get_gemm_config(
                "GEMM-A16W16-PERSISTENT", M, N, K, backend=backend
            )
        if backend == "triton":
            config["NUM_KSPLIT"] = 1
            config = compute_splitk_params(config, K)

        if backend == "gluon":
            assert (
                _is_gluon_available()
            ), f"Gluon backend requires one of {_GLUON_SUPPORTED_ARCHS}, got '{get_arch()}'"
            from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16_persistent import (
                _KERNEL_MAP as _GLUON_PERSISTENT_KERNEL_MAP,
            )

            _LOGGER.info(
                "GEMM_A16W16 [gluon/gfx1250, persistent]: x=%s w=%s",
                x.shape,
                w.shape,
            )

            kernel_type_from_config = config.pop("kernel_type", None)
            if kernel_type_from_config is not None:
                kernel_type = kernel_type_from_config
            assert kernel_type in _GLUON_PERSISTENT_KERNEL_MAP, (
                f"Unknown kernel_type '{kernel_type}', must be one of "
                f"{list(_GLUON_PERSISTENT_KERNEL_MAP.keys())}"
            )

            BLOCK_M = config["BLOCK_M"]
            BLOCK_N = config["BLOCK_N"]
            BLOCK_K = config["BLOCK_K"]
            NUM_BUFFERS = config.get("NUM_BUFFERS", 2)
            GROUP_SIZE_M = config.get("GROUP_SIZE_M", 1)
            num_warps = config["num_warps"]
            num_stages = config.get("num_stages", 0)
            waves_per_eu = config.get("waves_per_eu", 0)
            NUM_KSPLIT = config.get("NUM_KSPLIT", 1)

            # Compute split-K parameters
            SPLITK_BLOCK_SIZE = triton.cdiv(K, NUM_KSPLIT)
            while NUM_KSPLIT > 1 and BLOCK_K > SPLITK_BLOCK_SIZE:
                NUM_KSPLIT = max(NUM_KSPLIT // 2, 1)
                SPLITK_BLOCK_SIZE = triton.cdiv(K, NUM_KSPLIT)
            if NUM_KSPLIT > 1 and SPLITK_BLOCK_SIZE % BLOCK_K != 0:
                SPLITK_BLOCK_SIZE = triton.cdiv(SPLITK_BLOCK_SIZE, BLOCK_K) * BLOCK_K
                NUM_KSPLIT = triton.cdiv(K, SPLITK_BLOCK_SIZE)

            w = w.T

            # Clamp the pipeline depth to the K tiles of a split; at least 2 buffers
            num_k_tiles = triton.cdiv(SPLITK_BLOCK_SIZE, BLOCK_K)
            NUM_BUFFERS = max(2, clamp_num_buffers(NUM_BUFFERS, num_k_tiles))

            if y is None:
                y = torch.empty((M, N), dtype=dtype, device=x.device)

            if NUM_KSPLIT > 1:
                y_pp = torch.empty(
                    (NUM_KSPLIT, M, N), dtype=torch.float32, device=x.device
                )
            else:
                y_pp = None

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

            warp_bases = tuple(
                (0, 1) if i == 0 else (1 << (i - 1), 0)
                for i in range(num_warps.bit_length() - 1)
            )

            _LOGGER.info(
                "GEMM_A16W16 [gluon, persistent]: x=%s w=%s",
                x.shape,
                w.shape,
            )

            # sgpr spills to 0
            num_pid_n = triton.cdiv(N, BLOCK_N)
            num_mn_tiles = triton.cdiv(M, BLOCK_M) * num_pid_n
            num_tiles = num_mn_tiles * NUM_KSPLIT

            out_ptr = y if NUM_KSPLIT == 1 else y_pp

            _GLUON_PERSISTENT_KERNEL_MAP[kernel_type][(NUM_WGS,)](
                x,
                w,
                bias,
                out_ptr,
                M,
                N,
                K,
                num_tiles,
                x.stride(0),
                x.stride(1),
                w.stride(0),
                w.stride(1),
                0 if NUM_KSPLIT == 1 else y_pp.stride(0),
                out_ptr.stride(-2),
                out_ptr.stride(-1),
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_K=BLOCK_K,
                GROUP_SIZE_M=GROUP_SIZE_M,
                NUM_BUFFERS=NUM_BUFFERS,
                NUM_KSPLIT=NUM_KSPLIT,
                SPLITK_BLOCK_SIZE=SPLITK_BLOCK_SIZE,
                WARP_BASES=warp_bases,
                TRANSPOSE=TRANSPOSE,
                activation=_get_activation_from_str(activation) if activation else None,
                USE_ACTIVATION=activation is not None,
                ADD_BIAS=(bias is not None),
                SKIP_REDUCE=bool(skip_reduce),
                NUM_SMS=NUM_WGS,
                NUM_PID_N=num_pid_n,
                num_warps=num_warps,
                num_stages=num_stages,
                waves_per_eu=waves_per_eu,
            )

            if NUM_KSPLIT > 1:
                if skip_reduce:
                    return y_pp

                REDUCE_BLOCK_SIZE_M = 32
                REDUCE_BLOCK_SIZE_N = 32
                ACTUAL_KSPLIT = triton.cdiv(K, SPLITK_BLOCK_SIZE)

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
                    triton.next_power_of_2(NUM_KSPLIT),
                    ADD_BIAS=(bias is not None),
                    activation=(
                        _get_activation_from_str(activation) if activation else ""
                    ),
                    use_activation=activation is not None,
                    KERNEL_NAME="_gemm_a16w16_persistent_reduce_kernel",
                )

            return y

        _LOGGER.info(
            "GEMM_A16W16 [triton, persistent]: x=%s w=%s",
            x.shape,
            w.shape,
        )

        w = w.T

        if y is None:
            y = torch.empty((M, N), dtype=dtype, device=x.device)

        # Persistent, one WG per CU
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
            "GEMM_A16W16 [gluon/gfx1250]: x=%s w=%s kernel=%s",
            x.shape,
            w.shape,
            kernel_type,
        )

        M, K = x.shape
        N, _ = w.shape

        w = w.T

        kernel, grid, meta, num_warps, waves_per_eu = _gluon_plan(
            M,
            N,
            K,
            x.stride(),
            w.stride(),
            x.element_size() * 8,
            kernel_type,
            config,
            create_shared_layouts,
            create_wmma_layouts,
        )

        if y is None:
            y = torch.empty((M, N), dtype=dtype, device=x.device)

        _LOGGER.info(
            "GEMM_A16W16 [gluon, non-persistent]: x=%s w=%s",
            x.shape,
            w.shape,
        )

        kernel[grid](
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
            **meta,
            activation=_get_activation_from_str(activation) if activation else None,
            USE_ACTIVATION=activation is not None,
            ADD_BIAS=(bias is not None),
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
        )

        return y

    _LOGGER.info("GEMM_A16W16 [triton]: x=%s w=%s", x.shape, w.shape)

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
    if (
        _FAST_LAUNCH
        and not config
        and activation is None
        and not skip_reduce
        and backend is None
        and not persistent
        and _is_gluon_available()
        and not torch.compiler.is_compiling()
    ):
        y_fast = _gemm_a16w16_gluon_fast(x, w, bias, dtype, y, kernel_type)
        if y_fast is not None:
            return y_fast
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
