# SPDX-License-Identifier: MIT

"""High-level FlyDSL decode TopK-per-row API."""

from __future__ import annotations

import math
import os

import torch

from .kernels.tensor_shim import _run_compiled
from .kernels.topk_per_row_decode_aiter_persistent import (
    aiter_persistent_workspace_slots,
    create_topk_per_row_decode_aiter_persistent_k512_kernel,
)
from .kernels.topk_per_row_decode import create_topk_per_row_decode_kernel
from .kernels.topk_per_row_decode_coop import (
    cooperative_workspace_slots,
    create_topk_per_row_decode_cooperative_k512_kernel,
    create_topk_per_row_decode_cooperative_local_topk_merge_k512_kernel,
)

SUPPORTED_TOP_KS = (256, 512, 1024, 2048)
IMPLEMENTED_TOP_KS = (512, 2048)
_AITER_PERSISTENT_BLOCK_THREADS = 1024
_AITER_PERSISTENT_LOAD_VEC = 4

_COOP_WORKSPACES: dict[tuple[int, int, int, bool, str, int], torch.Tensor] = {}
_AITER_PERSISTENT_WORKSPACES: dict[tuple[int, int, int, int, int], torch.Tensor] = {}
_COOP_EPOCH = 0


def _check_cuda_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA/ROCm tensor")


def _required_seq_rows(next_n: int, num_rows: int) -> int:
    if num_rows <= 0:
        return 0
    return math.ceil(num_rows / next_n)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _next_coop_epoch() -> int:
    global _COOP_EPOCH
    _COOP_EPOCH = (_COOP_EPOCH % 0x3FFFFFFF) + 1
    return _COOP_EPOCH


def _get_coop_workspace(
    device: torch.device,
    num_rows: int,
    partitions_per_row: int,
    atomic_histogram: bool,
    mode: str,
) -> torch.Tensor:
    slots = cooperative_workspace_slots(
        num_rows,
        partitions_per_row,
        atomic_histogram=atomic_histogram,
        local_topk_merge=mode == "local_topk_merge",
    )
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    key = (
        device_index,
        int(num_rows),
        int(partitions_per_row),
        bool(atomic_histogram),
        mode,
        slots,
    )
    workspace = _COOP_WORKSPACES.get(key)
    if workspace is None or workspace.device.index != device_index or workspace.numel() != slots:
        workspace = torch.zeros(slots, device=device, dtype=torch.int32)
        _COOP_WORKSPACES[key] = workspace
    return workspace


def _get_aiter_persistent_workspace(
    device: torch.device,
    num_rows: int,
    blocks_per_row: int,
    bits_per_pass: int,
) -> torch.Tensor:
    slots = aiter_persistent_workspace_slots(
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
    workspace = _AITER_PERSISTENT_WORKSPACES.get(key)
    if workspace is None or workspace.device.index != device_index or workspace.numel() != slots:
        workspace = torch.empty(slots, device=device, dtype=torch.int32)
        _AITER_PERSISTENT_WORKSPACES[key] = workspace
    return workspace


def _default_aiter_persistent_bpp(device: torch.device) -> int:
    forced = os.environ.get("FLYDSL_TOPK_AITER_PERSISTENT_BPP")
    if forced is not None:
        try:
            value = int(forced)
        except ValueError as exc:
            raise ValueError(
                "FLYDSL_TOPK_AITER_PERSISTENT_BPP must be 10 or 11, "
                f"got {forced!r}"
            ) from exc
        if value not in (10, 11):
            raise ValueError(
                "FLYDSL_TOPK_AITER_PERSISTENT_BPP must be 10 or 11, "
                f"got {value}"
            )
        return value

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    cu_count = int(getattr(props, "multi_processor_count", 0))
    lds_per_cu = int(getattr(props, "shared_memory_per_multiprocessor", 0))
    return 11 if cu_count >= 128 or lds_per_cu >= 128 * 1024 else 10


def _aiter_persistent_forced_blocks_per_row() -> int | None:
    forced = os.environ.get("TOPK_FORCE_GRID")
    if forced is None:
        forced = os.environ.get("FLYDSL_TOPK_AITER_PERSISTENT_BLOCKS_PER_ROW")
    if forced is None:
        return None
    try:
        value = int(forced)
    except ValueError as exc:
        raise ValueError(
            "AITER persistent blocks-per-row override must be an integer, "
            f"got {forced!r}"
        ) from exc
    if not 2 <= value <= 32:
        raise ValueError(
            "AITER persistent blocks-per-row override must be in [2, 32], "
            f"got {value}"
        )
    return value


def _default_aiter_persistent_blocks_per_row(
    device: torch.device,
    *,
    num_rows: int,
    max_model_len: int,
    block_threads: int = _AITER_PERSISTENT_BLOCK_THREADS,
) -> int:
    forced = _aiter_persistent_forced_blocks_per_row()
    if forced is not None:
        return forced

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    cu_count = max(1, int(getattr(props, "multi_processor_count", 1)))
    lds_per_cu = int(getattr(props, "shared_memory_per_multiprocessor", 0))
    active_blocks = max(1, cu_count * 2)
    batch_size = max(1, int(num_rows))
    items_per_block = _AITER_PERSISTENT_LOAD_VEC * block_threads
    max_num_blocks = max(1, math.ceil(max_model_len / items_per_block))

    best_num_blocks = 1
    best_tail_wave_penalty = 1.0
    num_waves = 1
    while True:
        num_blocks = min(max_num_blocks, max(num_waves * active_blocks // batch_size, 1))
        items_per_thread = math.ceil(max_model_len / (num_blocks * block_threads))
        items_per_thread = (
            math.ceil(items_per_thread / _AITER_PERSISTENT_LOAD_VEC)
            * _AITER_PERSISTENT_LOAD_VEC
        )
        num_blocks = math.ceil(
            max_model_len / (items_per_thread * block_threads)
        )
        actual_num_waves = num_blocks * batch_size / active_blocks
        tail_wave_penalty = (
            math.ceil(actual_num_waves) - actual_num_waves
        ) / math.ceil(actual_num_waves)

        if tail_wave_penalty < 0.15:
            best_num_blocks = num_blocks
            break
        if tail_wave_penalty < best_tail_wave_penalty:
            best_num_blocks = num_blocks
            best_tail_wave_penalty = tail_wave_penalty
        if num_blocks == max_num_blocks:
            break
        num_waves += 1

    best_num_blocks = max(best_num_blocks, 2)
    if cu_count >= 128 or lds_per_cu >= 128 * 1024:
        wave_cap = max(2, cu_count // batch_size)
        best_num_blocks = max(min(best_num_blocks, wave_cap), 2)
    return min(best_num_blocks, 64 if block_threads == 512 else 32)


def _default_aiter_persistent_tiered_blocks_per_row(
    *,
    max_model_len: int,
    block_threads: int = _AITER_PERSISTENT_BLOCK_THREADS,
) -> int:
    forced = _aiter_persistent_forced_blocks_per_row()
    if forced is not None:
        return forced

    max_blocks = 64 if block_threads == 512 else 32
    items_per_block = _AITER_PERSISTENT_LOAD_VEC * block_threads
    blocks = max(2, math.ceil(max_model_len / items_per_block))
    return min(blocks, max_blocks)


def _cooperative_config(
    *,
    ordered: bool,
    k: int,
    max_model_len: int,
) -> tuple[bool, int, bool, str]:
    if ordered or k != 512:
        return False, 8, False, "histogram"

    # The cooperative kernel is still experimental; keep the proven one-CTA
    # unordered path as the default unless explicitly requested.
    enabled = _env_flag("FLYDSL_TOPK_COOP", "0")
    min_len = _env_int("FLYDSL_TOPK_COOP_MIN_ROW_LEN", 32768)
    partitions_per_row = _env_int("FLYDSL_TOPK_COOP_PARTITIONS", 8)
    # For long rows, partition-local TopK + merge is the fastest cooperative
    # FlyDSL variant measured so far. The whole cooperative path remains opt-in.
    mode = os.environ.get("FLYDSL_TOPK_COOP_MODE", "local_topk_merge").lower()
    # The partial-histogram experiment avoids global atomic hot bins but needs
    # ordered leader reads for correctness, which is slower than the AITER-like
    # atomic histogram on the measured long-row shapes. Keep it opt-in.
    atomic_histogram = _env_flag("FLYDSL_TOPK_COOP_ATOMIC_HIST", "1")

    if mode not in {
        "histogram",
        "local_topk_merge",
        "aiter_persistent",
        "aiter_persistent_tiered",
    }:
        raise ValueError(
            "FLYDSL_TOPK_COOP_MODE must be 'histogram', 'local_topk_merge', "
            "'aiter_persistent', or 'aiter_persistent_tiered', "
            f"got {mode!r}"
        )
    if mode == "local_topk_merge":
        valid_partitions = (4, 8, 16, 32)
    elif mode in {"aiter_persistent", "aiter_persistent_tiered"}:
        valid_partitions = tuple(range(2, 33))
    else:
        valid_partitions = (4, 8, 16)
    if partitions_per_row not in valid_partitions:
        raise ValueError(
            "FLYDSL_TOPK_COOP_PARTITIONS must be one of "
            f"{valid_partitions}, "
            f"got {partitions_per_row}"
        )

    use_coop = enabled and max_model_len >= min_len
    return use_coop, partitions_per_row, atomic_histogram, mode


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
    ordered: bool = True,
) -> None:
    """Decode TopK-per-row FlyDSL entry point.

    This matches the existing decode TopK interface. The current FlyDSL
    implementation supports the direct-fill case for rows whose effective
    decode length is ``<= k``. ``k=512`` uses a radix-threshold compact path
    with exact ordered refinement over compacted candidates and a correctness
    fallback. Other long-row variants use the correctness-first exact selector
    until their radix paths land.

    ``ordered`` (default ``True``) returns indices in ``torch.topk`` order
    (value descending, lower index wins ties). Pass ``ordered=False`` to select
    the AITER-style radix-**select** path that returns the Top-K as a *set* in
    arbitrary slot order. The unordered path is markedly faster because it drops
    the candidate compaction + bitonic sort, and is valid wherever the consumer
    only needs the set of Top-K column indices (e.g. the sparse-attention
    indexer). The selected set is equivalent to ``torch.topk`` with ties broken
    by value rather than by index.
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
    # The ordered path currently ships specialized builders only for the
    # implemented Ks; the unordered radix-select path is generic over all
    # supported Ks.
    if ordered and k not in IMPLEMENTED_TOP_KS:
        raise NotImplementedError(
            f"FlyDSL ordered decode TopK dispatcher is reserved for "
            f"k={SUPPORTED_TOP_KS}, but this pass only implements "
            f"k={IMPLEMENTED_TOP_KS}; pass ordered=False for other k"
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

    use_coop, partitions_per_row, atomic_histogram, coop_mode = _cooperative_config(
        ordered=ordered,
        k=k,
        max_model_len=int(logits.shape[1]),
    )
    with torch.cuda.device(logits.device.index):
        if use_coop:
            if coop_mode in {"aiter_persistent", "aiter_persistent_tiered"}:
                tiered = coop_mode == "aiter_persistent_tiered" or _env_flag(
                    "FLYDSL_TOPK_AITER_PERSISTENT_TIERED", "0"
                )
                block_threads = _AITER_PERSISTENT_BLOCK_THREADS
                if "FLYDSL_TOPK_COOP_PARTITIONS" not in os.environ:
                    if tiered:
                        partitions_per_row = _default_aiter_persistent_tiered_blocks_per_row(
                            max_model_len=int(logits.shape[1]),
                            block_threads=block_threads,
                        )
                    else:
                        partitions_per_row = _default_aiter_persistent_blocks_per_row(
                            logits.device,
                            num_rows=int(numRows),
                            max_model_len=int(logits.shape[1]),
                            block_threads=block_threads,
                        )
                bits_per_pass = _default_aiter_persistent_bpp(logits.device)
                scan_stages = _env_int("FLYDSL_TOPK_AITER_PERSISTENT_SCAN_STAGES", 2)
                tiered_short_max = _env_int(
                    "FLYDSL_TOPK_AITER_PERSISTENT_TIERED_SHORT_MAX", 16384
                )
                # Per-bucket active-part caps over the fixed launch grid. The
                # optimum scales down with batch size: a single long decode row
                # benefits from the full g32 active set, while multi-row batches
                # already fill the device so fewer parts per row cut barrier and
                # global-histogram merge cost.
                default_mid_cap = 16 if int(numRows) <= 1 else 8
                if int(numRows) <= 1:
                    default_long_cap = 32
                elif int(numRows) >= 8:
                    default_long_cap = 8
                else:
                    default_long_cap = 16
                tiered_mid_cap = _env_int(
                    "FLYDSL_TOPK_AITER_PERSISTENT_TIERED_MID_CAP", default_mid_cap
                )
                tiered_mid_max = _env_int(
                    "FLYDSL_TOPK_AITER_PERSISTENT_TIERED_MID_MAX", 65536
                )
                tiered_long_cap = _env_int(
                    "FLYDSL_TOPK_AITER_PERSISTENT_TIERED_LONG_CAP", default_long_cap
                )
                workspace = _get_aiter_persistent_workspace(
                    logits.device,
                    int(numRows),
                    partitions_per_row,
                    bits_per_pass,
                )
                # The AITER HIP host path memset()s Counter + histograms before
                # launching.  Keep that contract with a cached workspace by
                # enqueueing an async zero on the same stream before the kernel.
                with torch.cuda.stream(stream):
                    workspace.zero_()
                exe = create_topk_per_row_decode_aiter_persistent_k512_kernel(
                    partitions_per_row,
                    bits_per_pass=bits_per_pass,
                    scan_stages=scan_stages,
                    tiered=tiered,
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
                workspace = _get_coop_workspace(
                    logits.device,
                    int(numRows),
                    partitions_per_row,
                    atomic_histogram,
                    coop_mode,
                )
                if coop_mode == "local_topk_merge":
                    exe = create_topk_per_row_decode_cooperative_local_topk_merge_k512_kernel(
                        partitions_per_row
                    )
                else:
                    exe = create_topk_per_row_decode_cooperative_k512_kernel(
                        partitions_per_row,
                        atomic_histogram=atomic_histogram,
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
                    _next_coop_epoch(),
                    stream,
                )
        else:
            exe = create_topk_per_row_decode_kernel(k, ordered)
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
