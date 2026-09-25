# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import copy
import math

import torch
import triton

from aiter.ops.triton._triton_kernels.common.splitk_reduce import (
    _gemm_splitk_reduce_kernel,
)
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp8wfp8 import (
    _gemm_afp8wfp8_kernel,
    _gemm_afp8wfp8_packed_kernel,
    _gemm_afp8wfp8_preshuffle_kernel,
    _get_config,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.logger import AiterTritonLogger

from aiter.utility.graph_alloc import ROUTES_INSIDE_CAPTURE, persistent_alloc

# One arrival counter per output tile of a fused split-K launch.
_SPLIT_COUNTERS = 1 << 16
_split_counters_by_stream: dict[tuple[torch.device, int], torch.Tensor] = {}


def _split_counters(device: torch.device) -> torch.Tensor | None:
    """Zeroed arrival counters per (device, stream), as for the a16w16 split-K
    semaphores; kernels re-zero what they use. None when a stream's first use
    falls inside a capture that cannot allocate outside the graph pool."""
    key = (device, torch.cuda.current_stream(device).cuda_stream)
    counters = _split_counters_by_stream.get(key)
    if counters is None:
        if not ROUTES_INSIDE_CAPTURE and torch.cuda.is_current_stream_capturing():
            return None
        with persistent_alloc(device):
            counters = torch.zeros(_SPLIT_COUNTERS, dtype=torch.int32, device=device)
        _split_counters_by_stream[key] = counters
    return counters


def _split_k_partition(k: int, splits: int, step: int) -> tuple[int, int]:
    """Partition size rounded up to whole steps, and the non-empty count."""
    part = -(-k // splits)
    size = -(-part // step) * step
    return size, -(-k // size)


def _fused_workspace(y, tiles, splits, block_m, block_n):
    """(grid, partials, counters) for fused split-K, or None to run unfused."""
    assert tiles <= _SPLIT_COUNTERS, "Too many output tiles for fused split-K"
    counters = _split_counters(y.device)
    if counters is None:
        return None
    partials = torch.empty(
        tiles * splits * block_m * block_n, dtype=torch.float32, device=y.device
    )
    return (-(-tiles // 8) * 8 * splits,), partials, counters


_LOGGER = AiterTritonLogger()

_GLUON_SUPPORTED_ARCHS = ("gfx1250",)


def _is_gluon_available():
    """Check if the gluon backend is available for the current GPU architecture."""
    try:
        return any(supported in get_arch() for supported in _GLUON_SUPPORTED_ARCHS)
    except Exception:  # noqa: BLE001
        return False


def _resolve_x_scale_strides(
    x_scales: torch.Tensor,
    M: int,
    K: int,
    x_scale_group_size: int,
    is_x_scale_transposed: bool,
) -> tuple[int, int]:
    """Validate the activation-scale buffer and return its (row, group) strides.

    ``x_scale_group_size`` is how many K elements share one e8m0 byte: 32 for MX
    activations, 128 for blockscale activations (what aiter's per-group quant
    emits for ATOM's per_1x128 path).

    ``is_x_scale_transposed`` means the buffer still reads ``(M, K // group)``
    through ``.shape`` but its bytes are laid out column-major -- logically
    ``(K // group, M)``. That is what ``per_group_quant_hip(transpose_scale=True)``
    produces for group sizes other than 32. Folding it into strides here keeps
    both the triton and gluon kernels layout-agnostic.
    """
    assert x_scale_group_size in (
        32,
        128,
    ), f"x_scale_group_size must be 32 (MX) or 128 (blockscale), got {x_scale_group_size}"
    assert (
        K % x_scale_group_size == 0
    ), f"K={K} must be divisible by x_scale_group_size={x_scale_group_size}"

    expected = (M, K // x_scale_group_size)
    assert tuple(x_scales.shape[-2:]) == expected, (
        f"x_scales must have shape {expected} for x_scale_group_size="
        f"{x_scale_group_size}, got {tuple(x_scales.shape)}"
    )

    if is_x_scale_transposed:
        return x_scales.stride(1), x_scales.numel() // x_scales.stride(0)
    return x_scales.stride(0), x_scales.stride(1)


def gemm_afp8wfp8(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    dtype: torch.dtype | None = torch.bfloat16,
    y: torch.Tensor | None = None,
    config: dict | None = None,
    skip_reduce: bool | None = False,
    x_scale_group_size: int = 128,
    is_x_scale_transposed: bool = False,
    w_scale_group_size: tuple[int, int] = (128, 128),
    split_k: int | None = None,
) -> torch.Tensor:
    """Compute X @ W.T with E4M3 operands and compact E8M0 scales.

    Activation scales cover 1x32 or 1x128 elements. ``w_scale_group_size``
    specifies (N, K) elements per weight scale: (1, 32), (32, 32), or
    (128, 128). Scale tensors have shapes (M, K / activation group) and
    (ceil(N / weight N group), K / weight K group). Byte views are accepted.

    Config files select ordinary or ``packed`` K-panel execution, N_FIRST
    traversal, NUM_KSPLIT, and FUSED_SPLITK. Each scale layout has a separate
    config family. An explicit ``split_k`` overrides the configured count;
    partitions are aligned to whole kernel steps. ``skip_reduce`` disables
    fused reduction and returns FP32 partials when more than one split remains.
    Packed execution and fused reduction require gfx950.
    """
    assert x.ndim == w.ndim == 2, "Expected matrix operands"
    M, K = x.shape
    N, K_w = w.shape
    assert K == K_w and K > 0 and N > 0, "Expected matching positive K and positive N"
    w_scale_group_size = tuple(w_scale_group_size)
    assert w_scale_group_size in ((1, 32), (32, 32), (128, 128))
    group_n, group_k = w_scale_group_size
    assert K % group_k == 0, "K must be divisible by the weight scale K group"
    stride_asm, stride_ask = _resolve_x_scale_strides(
        x_scales, M, K, x_scale_group_size, is_x_scale_transposed
    )
    assert w_scales.ndim == 2 and tuple(w_scales.shape) == (
        triton.cdiv(N, group_n),
        K // group_k,
    ), "Invalid weight scale shape"
    assert x_scales.ndim == 2, "Expected matrix activation scales"
    assert all(
        t.dtype in (torch.uint8, torch.float8_e4m3fn) for t in (x, w)
    ), "Operands must be E4M3 or byte views"
    assert all(
        t.dtype in (torch.uint8, torch.float8_e8m0fnu) for t in (x_scales, w_scales)
    ), "Scales must be E8M0 or byte views"
    assert all(
        t.is_cuda and t.device == x.device for t in (x, w, x_scales, w_scales)
    ), "Operands must share a GPU"
    assert all(
        all(s > 0 for s in t.stride()) for t in (x, w, x_scales, w_scales)
    ), "Expected positive strides"
    assert dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert split_k is None or (type(split_k) is int and split_k > 0)
    if y is not None:
        assert y.shape == (M, N) and y.dtype == dtype and y.device == x.device
        assert all(s > 0 for s in y.stride())
    if M == 0:
        return y if y is not None else torch.empty((M, N), dtype=dtype, device=x.device)
    if config is None:
        config, _ = _get_config(
            M,
            N,
            K,
            x_scale_group_size=x_scale_group_size,
            w_scale_group_size=w_scale_group_size,
        )
    # Never mutate a caller's config or a nested packed config.
    config = copy.deepcopy(config)
    packed = config.get("packed")
    launch = config if packed is None else packed
    block_m, block_n, block_k = (
        launch[k] for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K")
    )
    assert block_k >= 64 and block_k % 32 == 0
    k_pack = 1 if packed is None else launch["K_PACK"]
    if packed is not None:
        assert k_pack in (1, 2, 4) and block_m * k_pack >= 16 and block_k >= 128
    requested_splits = launch["NUM_KSPLIT"] if split_k is None else split_k
    assert type(requested_splits) is int and requested_splits > 0
    split_size, num_splits = _split_k_partition(K, requested_splits, block_k * k_pack)
    fused = (
        launch.get("FUSED_SPLITK", config["FUSED_SPLITK"])
        and not skip_reduce
        and num_splits > 1
    )
    if packed is not None or fused:
        assert get_arch() == "gfx950", "Packed-K and fused split-K require gfx950"
    if y is None and (num_splits == 1 or not skip_reduce):
        y = torch.empty((M, N), dtype=dtype, device=x.device)
    grid_m, grid_n = triton.cdiv(M, block_m), triton.cdiv(N, block_n)
    workspace = None
    if fused:
        workspace = _fused_workspace(y, grid_m * grid_n, num_splits, block_m, block_n)
    if workspace is None:
        out = (
            y
            if num_splits == 1
            else torch.empty((num_splits, M, N), dtype=torch.float32, device=x.device)
        )
        partials = counters = out
        fused_splits = 1
        grid = (
            (grid_m, grid_n, num_splits)
            if packed is not None
            else (grid_m * grid_n * num_splits,)
        )
    else:
        grid, partials, counters = workspace
        out, fused_splits = y, num_splits
    stride_ck = out.stride(0) if workspace is None and num_splits > 1 else 0
    stride_cm, stride_cn = out.stride()[-2:]
    # Native FP8 pointers preserve the compiler's efficient scaled-dot layouts.
    # Byte pointers can inflate LDS allocation despite identical E4M3 bits.
    x, w = x.view(torch.float8_e4m3fn), w.view(torch.float8_e4m3fn)
    x_scales, w_scales = x_scales.view(torch.uint8), w_scales.view(torch.uint8)
    launch_options = {
        k: launch[k]
        for k in ("num_warps", "num_stages", "waves_per_eu", "matrix_instr_nonkdim")
    }
    scales = dict(
        A_SCALE_K_GROUP=x_scale_group_size,
        B_SCALE_N_GROUP=group_n,
        B_SCALE_K_GROUP=group_k,
    )
    if packed is not None:
        # Native E4M3 pointers avoid the extra LDS conversion generated for
        # byte operands in packed dot_scaled, while accepting public byte views.
        _gemm_afp8wfp8_packed_kernel[grid](
            x.view(torch.float8_e4m3fn),
            w.view(torch.float8_e4m3fn),
            x_scales,
            w_scales,
            out,
            partials,
            counters,
            M,
            N,
            K,
            block_m,
            block_n,
            block_k,
            k_pack,
            LAUNCH_OPTIONS=tuple(launch_options.items()),
            SPLITK_BLOCK_SIZE=split_size,
            NUM_KSPLIT=num_splits,
            FUSED_SPLITS=fused_splits,
            B_CACHE_MODIFIER=launch["cache_modifier"],
            stride_am=x.stride(0),
            stride_ak=x.stride(1),
            stride_bn=w.stride(0),
            stride_bk=w.stride(1),
            stride_asm=stride_asm,
            stride_ask=stride_ask,
            stride_bsn=w_scales.stride(0),
            stride_bsk=w_scales.stride(1),
            stride_cm=stride_cm,
            stride_cn=stride_cn,
            stride_ck=stride_ck,
            **scales,
            **launch_options,
        )
    else:
        _gemm_afp8wfp8_kernel[grid](
            x,
            w.T,
            out,
            x_scales,
            w_scales,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            w.stride(1),
            w.stride(0),
            stride_ck,
            stride_cm,
            stride_cn,
            stride_asm,
            stride_ask,
            w_scales.stride(0),
            w_scales.stride(1),
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            GROUP_SIZE_M=config["GROUP_SIZE_M"],
            NUM_KSPLIT=num_splits,
            SPLITK_BLOCK_SIZE=split_size,
            N_FIRST=config["N_FIRST"],
            FUSED_SPLITS=fused_splits,
            ws_ptr=partials,
            cnt_ptr=counters,
            cache_modifier=launch["cache_modifier"],
            **scales,
            **launch_options,
        )
    if workspace is None and num_splits > 1:
        if skip_reduce:
            return out
        reduce_m, reduce_n = (
            config["REDUCE_BLOCK_SIZE_M"],
            config["REDUCE_BLOCK_SIZE_N"],
        )
        _gemm_splitk_reduce_kernel[
            (triton.cdiv(M, reduce_m), triton.cdiv(N, reduce_n))
        ](
            out,
            y,
            None,
            M,
            N,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            y.stride(0),
            y.stride(1),
            BLOCK_SIZE_M=reduce_m,
            BLOCK_SIZE_N=reduce_n,
            ACTUAL_KSPLIT=num_splits,
            MAX_KSPLIT=triton.next_power_of_2(num_splits),
            ADD_BIAS=False,
            activation=None,
            use_activation=False,
            KERNEL_NAME="_gemm_afp8wfp8_reduce_kernel",
        )
    return y


def gemm_afp8wfp8_preshuffle(
    x: torch.Tensor,
    w_shuffled: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    dtype: torch.dtype | None = torch.bfloat16,
    y: torch.Tensor | None = None,
    config: dict | None = None,
    skip_reduce: bool | None = False,
    x_scale_group_size: int = 128,
    is_x_scale_transposed: bool = False,
    kernel_type: str = "bandwidth_bound",
    backend: str | None = None,
) -> torch.Tensor:
    """
    Preshuffle variant of gemm_afp8wfp8. The weight tensor has already been
    permuted via aiter.ops.shuffle.shuffle_weight(..., layout=(16, 16)). Scales
    are left unshuffled in the compact 128x128 layout.

    Uses the gluon backend automatically on supported architectures (gfx1250)
    and the triton backend everywhere else. Pass ``backend`` to force a choice.

    Args:
        x: FP8 e4m3 activations with shape (M, K).
        w_shuffled: FP8 e4m3 weights, shuffled in place to (N, K) storage
            (same total bytes; bytes rearranged for the kernel's read pattern).
        x_scales: e8m0 (uint8) per-group scale with shape
            (M, K // x_scale_group_size).
        w_scales: e8m0 (uint8) per-block weight scale with shape (N // 128, K // 128).
        dtype: Output dtype.
        y: Optional pre-allocated output (M, N).
        config: Optional kernel-tuning dict.
        x_scale_group_size: K elements per activation scale — 128 for blockscale
            activations (default), 32 for MX activations.
        is_x_scale_transposed: x_scales bytes are column-major, i.e. logically
            (K // group, M). Default False (row-major).
        kernel_type: [gluon only] Kernel variant. Only "bandwidth_bound" exists
            so far.
        backend: "triton", "gluon", or None (auto-detect).

    Returns:
        torch.Tensor: Output with shape (M, N).
    """
    M, K = x.shape
    N, K_w = w_shuffled.shape
    assert K == K_w, f"K mismatch: x={K}, w={K_w}"
    assert N % 16 == 0, f"N must be divisible by 16 for preshuffle, got {N}"
    stride_asm, stride_ask = _resolve_x_scale_strides(
        x_scales, M, K, x_scale_group_size, is_x_scale_transposed
    )

    # The kernel expects to address the shuffled tensor as (N//16, K*16).
    w_view = w_shuffled.view(N // 16, K * 16)

    if x.dtype != torch.uint8:
        x = x.view(torch.uint8)
    if w_view.dtype != torch.uint8:
        w_view = w_view.view(torch.uint8)

    # Resolve the backend up-front so the config is loaded from the backend's
    # config dir (<arch>/<backend>/gemm/) -- the two kernels take different keys.
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

    if config is None:
        config, _ = _get_config(M, N, K, shuffle=True, backend=backend)

    # CTA-cluster (CGA) multicast, gluon only. CTAS_M x CTAS_N CTAs form one
    # cluster, and each operand fetch is multicast to every CTA in the cluster
    # that wants it.
    #
    # NOTE the convention flip that happens right here, because the two sides
    # disagree on what BLOCK_SIZE_M / BLOCK_SIZE_N mean:
    #
    #   in the JSON   -> the PER-CTA tile. A tuned config keeps its meaning when
    #                    CTAS changes, and the LDS/register budget stays
    #                    readable straight off the file.
    #   in the kernel -> the CLUSTER tile. gl.arange over BLOCK_SIZE_N then
    #                    shards across the cluster on its own, and the grid
    #                    lambda below counts clusters rather than CTAs (triton
    #                    multiplies the grid by num_ctas at launch).
    #
    # So `"BLOCK_SIZE_N": 256, "CTAS_N": 2` is 256 columns per CTA and a
    # 512-column cluster tile -- per-CTA LDS is unchanged from 1x1, which is the
    # whole point. Verified: LDS/CTA is 311536 B at 1x1 and 311792 B at both 1x2
    # and 2x2, rather than halving or quartering.
    if backend == "gluon":
        ctas_m, ctas_n = config["CTAS_M"], config["CTAS_N"]
        num_ctas = ctas_m * ctas_n
        config["BLOCK_SIZE_M"] *= ctas_m
        config["BLOCK_SIZE_N"] *= ctas_n
    else:
        ctas_m, ctas_n, num_ctas = 1, 1, 1

    if y is None and (config["NUM_KSPLIT"] == 1 or not skip_reduce):
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(
        K, config["NUM_KSPLIT"]
    )  # How big each split_k partition is
    if config["NUM_KSPLIT"] > 1:
        y_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N),
            dtype=torch.float32,
            device=x.device,
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
    if backend == "gluon":
        from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_mxfp8 import (
            _PRESHUFFLE_KERNEL_MAP,
        )

        kernel_type = config.pop("kernel_type", kernel_type)
        assert kernel_type in _PRESHUFFLE_KERNEL_MAP, (
            f"Unknown kernel_type '{kernel_type}', must be one of "
            f"{list(_PRESHUFFLE_KERNEL_MAP.keys())}"
        )
        assert (
            num_ctas == 1 or kernel_type == "bandwidth_bound"
        ), f"CGA multicast is only wired into bandwidth_bound, got '{kernel_type}'"
        _LOGGER.info(
            "GEMM_AFP8WFP8 PRESHUFFLE [gluon/gfx1250]: x=%s w=%s kernel=%s",
            tuple(x.shape),
            tuple(w_view.shape),
            kernel_type,
        )

        # Shape-derived clamp, not a tuning default: the pipeline computes
        # NUM_BUFFERS - 1 tiles outside the main loop, so the depth cannot
        # exceed the K-tile count of a single split.
        num_k_iter = triton.cdiv(config["SPLITK_BLOCK_SIZE"], config["BLOCK_SIZE_K"])
        num_buffers = max(2, min(config["NUM_BUFFERS"], num_k_iter + 1))

        # warp_bases mirrors the a8w8 blockscale gluon wrapper: warp 0 walks N,
        # the remaining log2(num_warps // 2) warps walk M.
        warp_bases = [(0, 1)]
        for i in range(int(math.log2(config["num_warps"] // 2))):
            warp_bases.append((1 << i, 0))

        _PRESHUFFLE_KERNEL_MAP[kernel_type][grid](
            x,
            w_view,
            y if config["NUM_KSPLIT"] == 1 else y_pp,
            x_scales,
            w_scales,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            w_view.stride(0),
            w_view.stride(1),
            0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
            y.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
            y.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
            stride_asm,
            stride_ask,
            w_scales.stride(0),
            w_scales.stride(1),
            BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
            GROUP_SIZE_M=config["GROUP_SIZE_M"],
            A_SCALE_K_GROUP=x_scale_group_size,
            # Not foldable into strides like the rest of the layout: staging the
            # scales in LDS needs a TDM descriptor whose innermost stride is 1,
            # so the kernel has to know which axis is contiguous.
            A_SCALE_TRANSPOSED=is_x_scale_transposed,
            NUM_KSPLIT=config["NUM_KSPLIT"],
            SPLITK_BLOCK_SIZE=config["SPLITK_BLOCK_SIZE"],
            num_warps=config["num_warps"],
            warp_bases=tuple(warp_bases),
            cache_modifier=config["cache_modifier"],
            NUM_BUFFERS=num_buffers,
            # Every gluon preshuffle config declares this; configs/CLAUDE.md
            # forbids Python-side defaults for tuning values, so a missing key
            # is a config bug and should fail loudly rather than silently tune
            # itself off. The kernel clamps it to the main-loop trip count.
            LOOP_UNROLL_FACTOR=config["LOOP_UNROLL_FACTOR"],
            # Which B-scale fill to use. Tuned per shape: the TDM fill drops the
            # register staging and a barrier, but the global pre-load streams
            # better on some shapes. Both fill the same compact slab and both
            # work at any BLOCK_SIZE_N.
            B_SCALE_TDM=config["B_SCALE_TDM"],
            # Not read by the kernel: triton forwards it to the AMD backend as
            # the amdgpu-waves-per-eu occupancy hint. 0 emits no attribute.
            waves_per_eu=config["waves_per_eu"],
            CTAS_M=ctas_m,
            CTAS_N=ctas_n,
            # Reserved launch option (sets the cluster dim) that the kernel also
            # declares, so it is bound both ways -- like waves_per_eu above.
            num_ctas=num_ctas,
        )
    else:
        _gemm_afp8wfp8_preshuffle_kernel[grid](
            x,
            w_view,
            y if config["NUM_KSPLIT"] == 1 else y_pp,
            x_scales,
            w_scales,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            w_view.stride(0),
            w_view.stride(1),
            0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
            y.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
            y.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
            stride_asm,
            stride_ask,
            w_scales.stride(0),
            w_scales.stride(1),
            A_SCALE_K_GROUP=x_scale_group_size,
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
            None,
            M,
            N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            y.stride(0),
            y.stride(1),
            BLOCK_SIZE_M=REDUCE_BLOCK_SIZE_M,
            BLOCK_SIZE_N=REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT=ACTUAL_KSPLIT,
            MAX_KSPLIT=triton.next_power_of_2(config["NUM_KSPLIT"]),
            ADD_BIAS=False,
            activation=None,
            use_activation=False,
            KERNEL_NAME="_gemm_afp8wfp8_preshuffle_reduce_kernel",
        )

    return y
