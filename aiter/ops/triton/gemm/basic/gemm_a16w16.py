# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

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
    pick_gemm_tile,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.utility.graph_alloc import persistent_alloc

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


# The host side of the gfx950 gluon Stream-K kernel (kernel_type="streamk"); the
# kernel, its config and plan are in _gluon_kernels/gfx950/gemm/basic/
# gemm_a16w16_streamk.py.
#
# Workspace. Flags are zeroed once per (device, stream) and every launch hands them
# back zeroed: owners clear their peers' locks, the last reduce-scatter arrival
# clears the count, and L2 readers clear their own rows. Generation words only
# increment, so the flag layout is fixed for every configuration. Buffers come from
# AITER's persistent pool so they outlive CUDA graph captures; if the flags are first
# created while capturing, their zero fill is recorded into the graph and runs on
# every replay, which one eager launch on the capture stream beforehand avoids.
#
# Bounded and fixed-size like AITER's FlyDSL split-K buffers, with the same risk:
# past 128 (device, stream) keys an evicted buffer returns to the pool while a
# captured CUDA graph may still point at it.
@functools.lru_cache(maxsize=128)
def _streamk_flags(device, stream, words):
    with persistent_alloc(device):
        return torch.zeros(words, dtype=torch.int32, device=device)


@functools.lru_cache(maxsize=128)
def _streamk_persistent_partials(device, stream, elems):
    with persistent_alloc(device):
        return torch.empty(elems, dtype=torch.float32, device=device)


def _streamk_workspace(sk, device, num_programs, num_xcds, tile, launch, persistent):
    """``(partials, locks, sync)`` views for one launch of the Stream-K module ``sk``.

    ``sync`` is the reduce-scatter (count, generation) pairs followed by one
    programs-per-XCD L2 flag row per program slot (128-byte aligned rows); one
    owner lock per program follows it. Partial slots are indexed by Stream-K
    program id, which spans the whole resident grid for the L2 exchange. Partials
    need no zeroing: every slot is written before it is read.
    """
    device = torch.device(device)
    if device.index is None:
        device = torch.device(device.type, torch.cuda.current_device())
    stream = torch.cuda.current_stream(device)
    sync_words = sk.sync_words(num_programs, num_xcds)
    flags = _streamk_flags(device, stream, sync_words + num_programs)
    tile_elems = tile.block_m * tile.block_n
    if persistent:
        partials = _streamk_persistent_partials(
            device, stream, num_programs * sk.MAX_TILE_ELEMS
        )
        partials = partials[: num_programs * tile_elems]
    else:
        slots = num_programs if launch["L2"] else launch["STREAMK_PROGRAMS"]
        partials = torch.empty(slots * tile_elems, dtype=torch.float32, device=device)
    streamk_programs = launch["STREAMK_PROGRAMS"]
    return (
        partials,
        flags[sync_words : sync_words + streamk_programs],
        flags[:sync_words],
    )


def _validate_streamk_tensors(a, b, c, bias, tile):
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must be rank-2 tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError("incompatible GEMM dimensions")
    if c.shape != (a.shape[0], b.shape[1]):
        raise ValueError(f"y must have shape {(a.shape[0], b.shape[1])}")
    if a.device != b.device or c.device != a.device:
        raise ValueError("A, B and C must be on the same device")
    if a.dtype != b.dtype:
        raise ValueError("A and B must have the same dtype")
    if c.dtype != a.dtype:
        raise ValueError(
            f"gfx950 gluon a16w16 writes x's dtype {a.dtype}, not {c.dtype}"
        )
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("gfx950 gluon a16w16 supports fp16 and bf16 inputs")
    if a.stride(1) != 1:
        raise ValueError("A must be row-major/K-contiguous")
    if b.stride(0) != 1:
        raise ValueError("B must be column-major/K-contiguous")
    if c.stride(1) != 1:
        raise ValueError("C must be row-major")
    if a.stride(0) % 8 or b.stride(1) % 8 or c.stride(0) % 8:
        raise ValueError("A, B and C leading strides must be multiples of 8 elements")
    if any(tensor.data_ptr() % 16 for tensor in (a, b, c)):
        raise ValueError("A, B and C data pointers must be 16-byte aligned")
    if bias is not None:
        if bias.device != a.device or bias.dtype != a.dtype:
            raise ValueError("bias must match A's device and dtype")
        if bias.shape != (b.shape[1],) or not bias.is_contiguous():
            raise ValueError("bias must be contiguous with shape (N,)")
    strides = (*a.stride(), *b.stride(), *c.stride())
    if any(stride < 0 for stride in strides):
        raise ValueError("negative strides are unsupported")

    def fits(*spans, tensor):
        return sum(spans) * tensor.element_size() < 2**31 - 2

    if not fits(tile.block_m * a.stride(0), 2 * tile.block_k * a.stride(1), tensor=a):
        raise ValueError("A's tile-local 32-bit byte offsets overflow")
    if not fits(2 * tile.block_k * b.stride(0), tile.block_n * b.stride(1), tensor=b):
        raise ValueError("B's tile-local 32-bit byte offsets overflow")
    if not fits(tile.block_m * c.stride(0), c.shape[1] * c.stride(1), tensor=c):
        raise ValueError("C's tile-local 32-bit byte offsets overflow")


def _gemm_a16w16_gfx950_streamk(x, w, y, bias, config, persistent_partials=False):
    """``y = x @ w.T (+ bias)`` with the gfx950 gluon Stream-K kernel, x (M, K) and
    w (N, K) both K-contiguous, into y (M, N) of x's dtype. ``config`` names the
    tile, and the plan too when tuned; without the plan the default plan
    (streamk_plan) runs, and without a tile DEFAULT_TILE does.

    ``persistent_partials`` keeps one full-grid fp32 partial buffer sized for the
    largest tile per (device, stream) instead of allocating per call.
    """
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic import (
        gemm_a16w16_streamk as sk,
    )

    M, K = x.shape
    N = w.shape[0]
    num_programs, num_xcds = sk.streamk_topology()
    reason = sk.unsupported_reason(M, N, K)
    if reason is None:
        try:
            config, launch = sk._launch_constants(
                M, N, K, sk.with_default_tile(config), num_programs, num_xcds
            )
        except (TypeError, ValueError) as error:
            reason = str(error)
    if reason is not None:
        raise ValueError(f"gfx950 gluon a16w16: {reason}")
    tile = config.tile
    b = w.T  # (K, N) view, K contiguous: the layout the kernel loads
    _validate_streamk_tensors(x, b, y, bias, tile)

    # Each ragged dimension specializes its own clamped loads and masked stores.
    even_m = M % tile.block_m == 0
    even_n = N % tile.block_n == 0
    even_k = K == sk.k_pad(K, tile.block_k)
    partials, locks, sync = _streamk_workspace(
        sk, x.device, num_programs, num_xcds, tile, launch, persistent_partials
    )
    owner_qpb = min(4, 16 // max(1, tile.block_m * tile.block_n // 4096))
    total_full_tiles = launch["TOTAL_TILES"] - launch["STREAMK_TILES"]
    has_full = total_full_tiles > 0
    # The persistent prologue loads program_id's first tile unclamped.
    assert not has_full or launch["GRID"] <= total_full_tiles
    sk._KERNEL_MAP["streamk"][(launch["GRID"], 1)](
        x,
        b,
        y,
        bias,
        partials,
        locks,
        sync,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        b.stride(0),
        b.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_M=tile.block_m,
        BLOCK_N=tile.block_n,
        BLOCK_K=tile.block_k,
        NUM_PROGRAMS=num_programs,
        NUM_XCDS=num_xcds,
        GROUP_SIZE_M=config.group_size_m,
        ADD_BIAS=bias is not None,
        EVEN_M=even_m,
        EVEN_N=even_n,
        EVEN_K=even_k,
        STREAMK_TILES=launch["STREAMK_TILES"],
        SPLIT_N=launch["RESOLVED_SPLIT_N"],
        REDUCE_SCATTER=config.fixup == "rs",
        RS_NB=launch["RS_NB"],
        RS_U=launch["RS_U"],
        RS_R=launch["RS_R"],
        END_TO_END=launch["END_TO_END"],
        L2=launch["L2"],
        HAS_FULL=has_full,
        GRID=launch["GRID"] if launch["COMPACT"] else 0,
        OWNER_QPB=owner_qpb,
        RS_E=launch["RS_E"],
        BAR_STRIDE=launch["BAR_STRIDE"],
        num_warps=sk.NUM_WARPS,
        schedule_hint=sk.schedule_hint(),
    )
    return y


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
    On gfx950, ``backend="gluon"`` with ``kernel_type="compute_bound"`` runs the
    gfx950 gluon compute-bound kernel (scheduled by Triton's MFMA scheduler); it
    runs the tile its gluon config picks for the shape (sides 256, 128 or 64),
    handles ragged M, N and K (N and K multiples of 8, K at least 256).
    ``kernel_type="streamk"`` runs the gfx950 gluon Stream-K kernel with the
    whole ``config``: the tile (BLOCK_M, BLOCK_N, BLOCK_K; 256x256x64 when absent)
    and optionally GROUP_SIZE_M and the Stream-K plan (STREAMK_POLICY,
    STREAMK_FIXUP, SPLIT_N, STREAMK_EXCHANGE; the kernel's default plan when
    absent). It writes x's dtype; N a multiple of 8 and K of 16.

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
        kernel_type (str): [gluon only] Kernel variant ("bandwidth_bound", "compute_bound",
            "streamk" on gfx950).
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

    if backend == "gluon" and not persistent and "gfx950" in get_arch():
        from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a16w16 import (
            _KERNEL_MAP as _GFX950_KERNEL_MAP,
        )

        # Stream-K is launched here, not through the compute-bound kernel's map.
        kernel_types = [*_GFX950_KERNEL_MAP, "streamk"]
        assert kernel_type in kernel_types, (
            f"gfx950 gluon a16w16 supports kernel_type in {kernel_types}, "
            f"got '{kernel_type}'"
        )
        assert not skip_reduce, "gfx950 gluon a16w16 has no split-K (skip_reduce)"
        if activation:
            raise ValueError("gfx950 gluon a16w16: no activation support yet")
        M, K = x.shape
        N, _ = w.shape
        if y is None:
            y = torch.empty((M, N), dtype=dtype, device=x.device)
        if kernel_type == "streamk":
            # The whole config goes through: BLOCK_K and the Stream-K plan too.
            _LOGGER.info(
                "GEMM_A16W16 [gluon/gfx950]: x=%s w=%s kernel=streamk config=%s",
                x.shape,
                w.shape,
                config,
            )
            return _gemm_a16w16_gfx950_streamk(x, w, y, bias, config)
        # The tile comes from the config: a tuned one names BLOCK_M/BLOCK_N, the
        # default one carries the candidates and the rule (see pick_gemm_tile).
        if config is None or "BLOCK_M" not in config:
            config, _ = get_gemm_config("GEMM-A16W16", M, N, K, backend="gluon")
        tile = pick_gemm_tile(
            config,
            M,
            N,
            torch.cuda.get_device_properties(x.device).multi_processor_count,
        )
        _LOGGER.info(
            "GEMM_A16W16 [gluon/gfx950]: x=%s w=%s kernel=%s tile=%s",
            x.shape,
            w.shape,
            kernel_type,
            tile,
        )
        return _GFX950_KERNEL_MAP[kernel_type](x, w, y, bias, tile=tile)

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

            # Clamp the pipeline depth (per-partition k tiles)
            num_k_tiles = triton.cdiv(SPLITK_BLOCK_SIZE, BLOCK_K)
            NUM_BUFFERS = max(2, min(NUM_BUFFERS, num_k_tiles + 1))

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
        _MIN_BUFFERS = {"bandwidth_bound": 1, "compute_bound": 2}
        _DEPTH_SLACK = {"compute_bound": 2}

        if kernel_type_from_config is None:
            depth_cap = num_k_tiles - _DEPTH_SLACK.get(kernel_type, 0)
            if depth_cap < _MIN_BUFFERS[kernel_type]:
                needed = _MIN_BUFFERS[kernel_type] + _DEPTH_SLACK.get(kernel_type, 0)
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
        shared_a, shared_b = create_shared_layouts(
            BLOCK_M, BLOCK_N, BLOCK_K, layout, x.element_size() * 8
        )

        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)

        _LOGGER.info(
            "GEMM_A16W16 [gluon, non-persistent]: x=%s w=%s",
            x.shape,
            w.shape,
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
    On gfx950, ``backend="gluon"`` with ``kernel_type="compute_bound"`` runs the
    gfx950 gluon compute-bound kernel (scheduled by Triton's MFMA scheduler); it
    runs the tile its gluon config picks for the shape (sides 256, 128 or 64),
    handles ragged M, N and K (N and K multiples of 8, K at least 256).
    ``kernel_type="streamk"`` runs the gfx950 gluon Stream-K kernel with the
    whole ``config``: the tile (BLOCK_M, BLOCK_N, BLOCK_K; 256x256x64 when absent)
    and optionally GROUP_SIZE_M and the Stream-K plan (STREAMK_POLICY,
    STREAMK_FIXUP, SPLIT_N, STREAMK_EXCHANGE; the kernel's default plan when
    absent). It writes x's dtype; N a multiple of 8 and K of 16.
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
