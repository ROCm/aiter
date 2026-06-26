# SPDX-License-Identifier: MIT

"""High-level FlyDSL decode TopK-per-row API."""

from __future__ import annotations

import math
import os

import torch

from .kernels.tensor_shim import _run_compiled
from .kernels.topk_per_row_decode_tiered import (
    create_topk_per_row_decode_tiered_k512_kernel,
    tiered_topk_workspace_slots,
)
from .kernels.topk_per_row_decode import create_topk_per_row_decode_kernel

SUPPORTED_TOP_KS = (256, 512, 1024, 2048)
_TIERED_BLOCK_THREADS = 1024
_TIERED_LOAD_VEC = 4
_TIERED_SCAN_STAGES = 2
_TIERED_SHORT_MAX = 16384
_TIERED_MID_MAX = 65536

# Cached tiered workspaces are keyed by layout/size only. Tier caps and scan
# staging change codegen/active participants, not workspace shape.
_TIERED_WORKSPACES: dict[tuple[int, int, int, int, int], torch.Tensor] = {}


def _check_cuda_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA/ROCm tensor")


def _required_seq_rows(next_n: int, num_rows: int) -> int:
    if num_rows <= 0:
        return 0
    return math.ceil(num_rows / next_n)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _get_tiered_workspace(
    device: torch.device,
    num_rows: int,
    blocks_per_row: int,
    bits_per_pass: int,
) -> torch.Tensor:
    slots = tiered_topk_workspace_slots(
        num_rows,
        blocks_per_row,
        bits_per_pass=bits_per_pass,
    )
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    key = (
        device_index,
        int(num_rows),
        int(blocks_per_row),
        int(bits_per_pass),
        slots,
    )
    workspace = _TIERED_WORKSPACES.get(key)
    if workspace is None or workspace.device.index != device_index or workspace.numel() != slots:
        workspace = torch.empty(slots, device=device, dtype=torch.int32)
        _TIERED_WORKSPACES[key] = workspace
    return workspace


def _default_tiered_bpp(device: torch.device) -> int:
    forced = os.environ.get("FLYDSL_TOPK_TIERED_BPP")
    if forced is not None:
        try:
            value = int(forced)
        except ValueError as exc:
            raise ValueError(
                "FLYDSL_TOPK_TIERED_BPP must be 10 or 11, "
                f"got {forced!r}"
            ) from exc
        if value not in (10, 11):
            raise ValueError(
                "FLYDSL_TOPK_TIERED_BPP must be 10 or 11, "
                f"got {value}"
            )
        return value

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    cu_count = int(getattr(props, "multi_processor_count", 0))
    lds_per_cu = int(getattr(props, "shared_memory_per_multiprocessor", 0))
    return 11 if cu_count >= 128 or lds_per_cu >= 128 * 1024 else 10


def _default_tiered_blocks_per_row(
    *,
    max_model_len: int,
    block_threads: int = _TIERED_BLOCK_THREADS,
) -> int:
    max_blocks = 64 if block_threads == 512 else 32
    items_per_block = _TIERED_LOAD_VEC * block_threads
    blocks = max(2, math.ceil(max_model_len / items_per_block))
    return min(blocks, max_blocks)


def _use_tiered_k512(
    *,
    k: int,
    max_model_len: int,
) -> bool:
    if k != 512:
        return False

    enabled = os.environ.get("FLYDSL_TOPK_TIERED", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    if not enabled:
        raise NotImplementedError(
            "K=512 FlyDSL decode TopK now uses only the tiered persistent path; "
            "FLYDSL_TOPK_TIERED=0 has no non-tiered fallback"
        )
    min_len = _env_int("FLYDSL_TOPK_TIERED_MIN_ROW_LEN", 0)
    if max_model_len < min_len:
        raise NotImplementedError(
            "K=512 FlyDSL decode TopK now uses only the tiered persistent path; "
            f"max_model_len={max_model_len} is below FLYDSL_TOPK_TIERED_MIN_ROW_LEN={min_len}"
        )
    return True


def flydsl_top_k_per_row_decode(
    logits: torch.Tensor,
    next_n: int,
    seqLens: torch.Tensor,
    indices: torch.Tensor,
    numRows: int,
    stride0: int,
    stride1: int,
    k: int = 2048,
    stream: torch.cuda.Stream | None = None,
    ordered: bool = False,
) -> None:
    """Decode TopK-per-row FlyDSL entry point.

    This matches the existing decode TopK interface while returning the Top-K
    as an unordered set of column indices. K=512 dispatches to the tiered
    persistent kernel; other supported K values use the single-CTA unordered
    radix-select kernel.

    ``ordered=True`` is no longer supported. The intended contract is the
    unordered set-equivalent output used by the HIP/vLLM/model consumers.
    """

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
        raise ValueError(
            f"numRows={numRows} exceeds logits rows={logits.shape[0]}"
        )
    if numRows > indices.shape[0]:
        raise ValueError(
            f"numRows={numRows} exceeds indices rows={indices.shape[0]}"
        )
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

    # Only the cheap host-side consistency check is performed here. Validating
    # that seqLens implies a row length within the logits column count would
    # require a device->host reduction (``seqLens.max().item()``) on every
    # call, which dominates latency for the single-row decode shape. The kernel
    # reads logits through a bounded buffer resource, so an inconsistent
    # seqLens cannot read out of bounds; we therefore avoid the per-call sync.
    required_seq_rows = _required_seq_rows(next_n, numRows)
    if required_seq_rows > seqLens.numel():
        raise ValueError(
            f"numRows={numRows} with next_n={next_n} requires at least "
            f"{required_seq_rows} seqLens entries, got {seqLens.numel()}"
        )
    if numRows == 0:
        return
    if stream is None:
        stream = torch.cuda.current_stream(logits.device)

    use_tiered_persistent = _use_tiered_k512(
        k=k,
        max_model_len=int(logits.shape[1]),
    )
    with torch.cuda.device(logits.device.index):
        if use_tiered_persistent:
            block_threads = _TIERED_BLOCK_THREADS
            partitions_per_row = _default_tiered_blocks_per_row(
                max_model_len=int(logits.shape[1]),
                block_threads=block_threads,
            )
            bits_per_pass = _default_tiered_bpp(logits.device)
            # Active-part caps over the fixed launch grid. The optimum scales
            # down with batch size: a single long decode row benefits from the
            # full g32 active set, while multi-row batches already fill the
            # device so fewer parts per row cut barrier and histogram-merge cost.
            tiered_short_max = _env_int(
                "FLYDSL_TOPK_TIERED_SHORT_MAX",
                _TIERED_SHORT_MAX,
            )
            tiered_mid_cap_default = 16 if int(numRows) <= 1 else 8
            tiered_mid_cap = _env_int(
                "FLYDSL_TOPK_TIERED_MID_CAP",
                tiered_mid_cap_default,
            )
            tiered_mid_max = _env_int("FLYDSL_TOPK_TIERED_MID_MAX", _TIERED_MID_MAX)
            if int(numRows) <= 1:
                tiered_long_cap_default = 32
            elif int(numRows) >= 8:
                tiered_long_cap_default = 8
            else:
                tiered_long_cap_default = 16
            tiered_long_cap = _env_int(
                "FLYDSL_TOPK_TIERED_LONG_CAP",
                tiered_long_cap_default,
            )
            workspace = _get_tiered_workspace(
                logits.device,
                int(numRows),
                partitions_per_row,
                bits_per_pass,
            )
            # The HIP host path memset()s counters and histograms before launch.
            # Keep that contract with a cached workspace by enqueueing an async
            # zero on the same stream before the kernel.
            with torch.cuda.stream(stream):
                workspace.zero_()
            exe = create_topk_per_row_decode_tiered_k512_kernel(
                partitions_per_row,
                bits_per_pass=bits_per_pass,
                scan_stages=_TIERED_SCAN_STAGES,
                tiered=True,
                tiered_short_max=tiered_short_max,
                tiered_mid_cap=tiered_mid_cap,
                tiered_mid_max=tiered_mid_max,
                tiered_long_cap=tiered_long_cap,
            )
            _run_compiled(
                exe,
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
        else:
            exe = create_topk_per_row_decode_kernel(k)
            _run_compiled(
                exe,
                logits,
                int(next_n),
                seqLens,
                indices,
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
    k: int = 2048,
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
