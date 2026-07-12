# SPDX-License-Identifier: MIT

"""High-level FlyDSL decode TopK-per-row API."""

from __future__ import annotations
import functools
import math
import os
import torch

from .kernels.tensor_shim import _run_compiled
from .kernels.topk_per_row_decode_tiered import (
    create_topk_per_row_decode_tiered_kernel,
    needs_workspace_zero,
    topk_workspace_slots,
)

from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SMEM_CAPACITY_MAP


##################################

SUPPORTED_TOP_KS = (256, 512, 1024, 2048)
_TIERED_BLOCK_THREADS = 1024
_TIERED_LOAD_VEC = 4
_TIERED_SCAN_STAGES = 2

# Both thresholds are independent of K.
# K only affects the final O(K) index scatter,
# which is negligible compared to the O(L) scan.
_TIERED_SHORT_MAX = 16384
_TIERED_MID_MAX = 65536


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return default

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _default_tiered_bpp(arch: str | None = None) -> int:
    cu_count = _multi_processor_count(arch)
    lds_per_cu = SMEM_CAPACITY_MAP.get(arch, 0)
    return 11 if cu_count >= 128 or lds_per_cu >= 128 * 1024 else 10


def _default_tiered_blocks_per_row(
    *,
    max_model_len: int,
    block_threads: int,
) -> int:
    max_blocks = 64 if block_threads == 512 else 32
    items_per_block = _TIERED_LOAD_VEC * block_threads
    blocks = max(2, math.ceil(max_model_len / items_per_block))
    return min(blocks, max_blocks)


@functools.cache
def _multi_processor_count(arch: str | None = None) -> int:
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return int(props.multi_processor_count)
    except Exception:
        fallback_min_cu = {"gfx942": 228, "gfx950": 256}
        return fallback_min_cu.get(arch or get_rocm_arch(), 64)


@functools.cache
def _environ_kernel_config() -> dict:
    cfg = dict(
        scan_stages=_env_int("FLYDSL_TOPK_SCAN_STAGES"),
        tiered_short_max=_env_int("FLYDSL_TOPK_TIERED_SHORT_MAX"),
        tiered_mid_cap=_env_int("FLYDSL_TOPK_TIERED_MID_CAP"),
        tiered_mid_max=_env_int("FLYDSL_TOPK_TIERED_MID_MAX"),
        tiered_long_cap=_env_int("FLYDSL_TOPK_TIERED_LONG_CAP"),
        bits_per_pass=_env_int("FLYDSL_TOPK_TIERED_BPP"),
    )
    return {k: v for k, v in cfg.items() if v is not None}


def _default_kernel_config(
    num_rows: int,
    max_model_len: int,
) -> dict:
    blocks_per_row = _default_tiered_blocks_per_row(
        max_model_len=max_model_len,
        block_threads=_TIERED_BLOCK_THREADS,
    )
    bits_per_pass = _default_tiered_bpp()

    # Max cooperating workgroups per row for the mid/long tiers (the real wg count
    # is min(blocks_per_row, cap)). Scales down with batch size: a single long row
    # wants the full wg32 set, while multi-row batches already fill the device so
    # fewer workgroups/row cut barrier and histogram-merge cost.
    if num_rows <= 1:
        tiered_mid_cap_default = 32
    else:  # num_rows > 1
        tiered_mid_cap_default = 8

    if num_rows <= 1:
        tiered_long_cap_default = 32
    elif num_rows < 8:
        tiered_long_cap_default = 16
    else:  #  num_rows >= 8
        tiered_long_cap_default = 8

    # Batch-aware short vs multi-block crossover. The multi-block barrier
    # floor grows under CU contention as more rows launch, while the single
    # workgroup path is flat in batch.
    tiered_short_max = min(40960, _TIERED_SHORT_MAX + num_rows * 1536)

    return dict(
        blocks_per_row=blocks_per_row,
        bits_per_pass=bits_per_pass,
        tiered=True,
        scan_stages=_TIERED_SCAN_STAGES,
        tiered_short_max=tiered_short_max,
        tiered_mid_cap=tiered_mid_cap_default,
        tiered_mid_max=_TIERED_MID_MAX,
        tiered_long_cap=tiered_long_cap_default,
    )


def _kernel_config(num_rows: int, max_model_len: int) -> dict:
    default_config = _default_kernel_config(num_rows, max_model_len)
    environ_config = _environ_kernel_config()

    kernel_config = {
        **default_config,
        **environ_config,
    }

    bits_per_pass = kernel_config["bits_per_pass"]
    if bits_per_pass not in (10, 11):
        raise ValueError(f"bits_per_pass must be 10 or 11, got {bits_per_pass}")

    kernel_config = _apply_deadlock_guard(kernel_config, num_rows, max_model_len)
    return kernel_config


def _apply_deadlock_guard(
    kernel_config: dict,
    num_rows: int,
    max_model_len: int,
) -> dict:
    """Clamp the tiered config so the mid/long-tier inter-workgroup barrier cannot
    deadlock.

    The tiered kernels spin on a barrier over a non-cooperative launch, which gives
    no guarantee that a row's participating workgroups are all resident at the same
    time. A workgroup spinning at the barrier holds its slot until every participant
    of its row arrives. A deadlock can potentially happen once the barrier-blocked
    workgroups exceeds the resident capacity. In this case, we cap the active
    workgroups or force the barrier-free short tier (single workgroup/row).

    No-op unless rows can reach the barrier tiers (max_model_len > short_max)
    at a batch large enough to matter, so the common decode shape is untouched.

    Example for MI300X:
    The deadlock guard is active around num_rows >= 87 with max_model_len > 32768
    CUs * Occupancy = 304 * 2 = 608
    8 cooperating workgroups, 87*7 >= 608
    """
    if num_rows <= 0:
        return kernel_config

    short_max = kernel_config["tiered_short_max"]
    if max_model_len <= short_max:
        return kernel_config  # all rows short-tier -> barrier-free

    # Worst-case workgroups any single row can put on the barrier, given the tier
    # its length can reach (mid vs long) and the grid width.
    mid_cap = kernel_config["tiered_mid_cap"]
    long_cap = kernel_config["tiered_long_cap"]
    blocks_per_row = kernel_config["blocks_per_row"]
    if max_model_len <= kernel_config["tiered_mid_max"]:
        max_active_workgroups_per_row = min(blocks_per_row, mid_cap)
    else:
        max_active_workgroups_per_row = min(blocks_per_row, max(mid_cap, long_cap))

    is_single_workgroup = max_active_workgroups_per_row <= 1
    if is_single_workgroup:
        return kernel_config  # single-workgroup tier -> barrier-free

    # Co-resident envelope N = num_CU x occupancy. Occupancy is 2 on all CDNA:
    # the 1024-thread block is wave-limited (32 waves/CU / 16), with VGPR/LDS
    # headroom (measured gfx942: VGPR=40, LDS=8.7KB). Re-check if scan_stages or
    # the histogram grows enough to push VGPR>64 / LDS>32KB (would drop occ to 1).
    max_coresident_workgroups = _multi_processor_count() * 2
    is_deadlock_free = (
        num_rows * (max_active_workgroups_per_row - 1) < max_coresident_workgroups
    )
    if is_deadlock_free:
        return kernel_config

    # Largest cap A satisfying num_rows * (A - 1) < N.
    max_safe_active_workgroups = (max_coresident_workgroups - 1) // num_rows + 1
    if max_safe_active_workgroups >= 2:
        kernel_config["tiered_mid_cap"] = min(mid_cap, max_safe_active_workgroups)
        kernel_config["tiered_long_cap"] = min(long_cap, max_safe_active_workgroups)
    else:
        # Even 2-way cooperation exceeds the envelope: route every row to the
        # barrier-free short tier. Requires bits_per_pass == 11 (the short tier);
        # target archs (gfx942/gfx950) satisfy this.
        kernel_config["tiered_short_max"] = max_model_len
    return kernel_config


def flydsl_top_k_per_row_decode_workspace_size(
    num_rows: int,
    max_model_len: int,
) -> int:
    """
    Number of int32 elements the decode TopK workspace needs for this shape.
    max_model_len = int(logits.shape[1])
    """
    if num_rows <= 0:
        return 0

    kernel_config = _kernel_config(num_rows, max_model_len)
    workspace_slots = topk_workspace_slots(
        num_rows,
        kernel_config["bits_per_pass"],
    )
    return workspace_slots


@functools.lru_cache(maxsize=16384)
def _compile_launcher(
    top_k: int,
    num_rows: int,
    max_model_len: int,
):
    kernel_config = _kernel_config(num_rows, max_model_len)

    workspace_slots = topk_workspace_slots(
        num_rows,
        kernel_config["bits_per_pass"],
    )
    workspace_zero = needs_workspace_zero(
        max_model_len,
        top_k,
        kernel_config["tiered_short_max"],
    )
    launcher = create_topk_per_row_decode_tiered_kernel(
        top_k=top_k,
        **kernel_config,
    )
    return launcher, workspace_slots, workspace_zero


def _check_cuda_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA/ROCm tensor")


def _required_seq_rows(next_n: int, num_rows: int) -> int:
    if num_rows <= 0:
        return 0
    return math.ceil(num_rows / next_n)


def _validate_inputs(
    logits: torch.Tensor,
    next_n: int,
    seqLens: torch.Tensor,
    indices: torch.Tensor,
    numRows: int,
    stride0: int,
    stride1: int,
    k: int = 2048,
    ordered: bool = False,
    workspace: torch.Tensor | None = None,
):
    _check_cuda_tensor("logits", logits)
    _check_cuda_tensor("seqLens", seqLens)
    _check_cuda_tensor("indices", indices)

    if logits.dtype is not torch.float32:
        raise TypeError(f"logits must be torch.float32, got {logits.dtype}")
    if seqLens.dtype is not torch.int32:
        raise TypeError(f"seqLens must be torch.int32, got {seqLens.dtype}")
    if indices.dtype is not torch.int32:
        raise TypeError(f"indices must be torch.int32, got {indices.dtype}")
    if logits.device != seqLens.device or logits.device != indices.device:
        raise ValueError("logits, seqLens, and indices must be on the same device")
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D, got shape={tuple(logits.shape)}")
    if indices.ndim != 2:
        raise ValueError(f"indices must be 2D, got shape={tuple(indices.shape)}")
    if next_n <= 0:
        raise ValueError(f"next_n must be positive, got {next_n}")
    if numRows < 0:
        raise ValueError(f"numRows must be non-negative, got {numRows}")
    if numRows > logits.shape[0]:
        raise ValueError(f"numRows={numRows} exceeds logits rows={logits.shape[0]}")
    if numRows > indices.shape[0]:
        raise ValueError(f"numRows={numRows} exceeds indices rows={indices.shape[0]}")
    if k not in SUPPORTED_TOP_KS:
        raise NotImplementedError(
            f"FlyDSL decode TopK only accepts compile-time k in "
            f"{SUPPORTED_TOP_KS}, got {k}"
        )
    if ordered:
        raise ValueError(
            "FlyDSL decode TopK returns unordered set output only; "
            "ordered=True is not supported"
        )
    if indices.shape[1] < k:
        raise ValueError(f"indices second dimension must be at least k={k}")
    if indices.stride(1) != 1:
        raise ValueError("indices must have contiguous per-row storage")
    if stride1 != 1:
        raise NotImplementedError(
            f"FlyDSL decode TopK currently supports stride1 == 1 only, got {stride1}"
        )
    if stride0 != logits.stride(0) or stride1 != logits.stride(1):
        raise ValueError(
            "stride0/stride1 must match logits.stride(); received "
            f"({stride0}, {stride1}) for logits.stride()={logits.stride()}"
        )

    required_seq_rows = _required_seq_rows(next_n, numRows)
    if required_seq_rows > seqLens.numel():
        raise ValueError(
            f"numRows={numRows} with next_n={next_n} requires at least "
            f"{required_seq_rows} seqLens entries, got {seqLens.numel()}"
        )

    if workspace is not None:
        _check_cuda_tensor("workspace", workspace)
        if workspace.dtype is not torch.int32:
            raise TypeError(f"workspace must be torch.int32, got {workspace.dtype}")
        if workspace.device != logits.device:
            raise ValueError("workspace must be on the same device as logits")


def flydsl_top_k_per_row_decode(
    logits: torch.Tensor,
    next_n: int,
    seqLens: torch.Tensor,
    indices: torch.Tensor,
    numRows: int,
    stride0: int,
    stride1: int,
    k: int,
    stream: torch.cuda.Stream | None = None,
    ordered: bool = False,
    workspace: torch.Tensor | None = None,
) -> None:
    if numRows == 0:
        return

    _validate_inputs(
        logits,
        next_n,
        seqLens,
        indices,
        numRows,
        stride0,
        stride1,
        k,
        ordered,
        workspace,
    )

    launcher, workspace_slots, workspace_zero = _compile_launcher(
        k,
        numRows,
        logits.shape[1],
    )

    if workspace is None:
        workspace = torch.empty(
            workspace_slots,
            dtype=torch.int32,
            device=logits.device,
        )
    elif workspace.numel() < workspace_slots:
        raise ValueError(
            f"workspace too small: need >= {workspace_slots} int32 "
            f"elements, got {workspace.numel()} (use "
            f"flydsl_top_k_per_row_decode_workspace_size)"
        )

    if stream is None:
        stream = torch.cuda.current_stream(logits.device)

    if workspace_zero:
        with torch.cuda.stream(stream):
            workspace.zero_()

    with torch.cuda.device(logits.device.index):
        _run_compiled(
            launcher,
            logits,
            int(next_n),
            seqLens,
            indices,
            workspace,
            int(numRows),
            int(stride0),
            int(stride1),
            stream,
        )


def flydsl_top_k_per_row_decode_unordered(
    logits: torch.Tensor,
    next_n: int,
    seqLens: torch.Tensor,
    indices: torch.Tensor,
    numRows: int,
    stride0: int,
    stride1: int,
    k: int,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Benchmark-friendly wrapper for the unordered set-output path."""

    flydsl_top_k_per_row_decode(
        logits,
        next_n,
        seqLens,
        indices,
        numRows,
        stride0,
        stride1,
        k=k,
        stream=stream,
        ordered=False,
    )
