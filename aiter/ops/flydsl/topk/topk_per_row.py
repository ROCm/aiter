# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL decode TopK interface."""

import ctypes
from functools import lru_cache

import torch

from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.asm.asm_utils import load_hip
from aiter.ops.topk import (
    BACKEND_ADAPTIVE,
    _decode_cu_count,
    decode_adaptive_width,
    decode_backend_for_call,
)
from aiter.utility.graph_alloc import ROUTES_INSIDE_CAPTURE, persistent_alloc

from ..kernels.kernels_common import get_warp_size
from ..kernels.tensor_shim import _run_compiled
from ..kernels.topk import topk_per_row_decode_adaptive as _adaptive
from ..kernels.topk.radix_topk_one_block import (
    _COMPACT_CAPACITY,
    _MAX_ROW_ELEMENTS,
    build_radix_topk_one_block_module,
)
from ..kernels.topk.topk_per_row_decode import (
    clear_topk_per_row_decode_workspace_cache as _clear_split_workspace_cache,
)
from ..kernels.topk.topk_per_row_decode import (
    launch_topk_per_row_decode,
)
from ..kernels.topk.topk_per_row_decode_persistent import (
    build_topk_per_row_decode_one_workgroup_module,
)

# Widths up to this value use the persistent one-workgroup kernel.
_ONE_WORKGROUP_MAX_ROW_WIDTH = 20_000
_SHORT_ROWS_1024_THREAD_MAX_ROWS = 256


# One adaptive control buffer per (device, stream). Every launch leaves it zero, so
# only its first launch needs it zeroed. A buffer a larger one replaced stays alive,
# since a captured graph may still replay over it.
_adaptive_control: dict[tuple[torch.device, int], torch.Tensor] = {}
_retired_adaptive_control: list[torch.Tensor] = []
# Buffers whose zeroing exists only as a node in captured graphs, mapped to the
# capture that recorded it last. Until one of those replays the buffer is garbage,
# so every other capture records its own zeroing and an eager launch zeroes it.
_unzeroed_adaptive_control: dict[tuple[torch.device, int], int] = {}
_hip_stream_get_capture_info = None


def _capture_id(stream_id: int) -> int | None:
    """Id of the graph capture `stream_id` is recording, or None."""
    global _hip_stream_get_capture_info
    if _hip_stream_get_capture_info is None:
        fn = load_hip().hipStreamGetCaptureInfo
        fn.restype = ctypes.c_int
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_ulonglong),
        ]
        _hip_stream_get_capture_info = fn
    status, capture = ctypes.c_int(), ctypes.c_ulonglong()
    err = _hip_stream_get_capture_info(
        ctypes.c_void_p(stream_id), ctypes.byref(status), ctypes.byref(capture)
    )
    if err != 0:
        raise RuntimeError(f"hipStreamGetCaptureInfo failed with error {err}")
    return capture.value if status.value != 0 else None


def _get_adaptive_control(
    device: torch.device, stream_id: int, rows: int
) -> torch.Tensor:
    """The control buffer for `rows` rows, zero when the launch runs.

    Grown in powers of two: per device and stream, live and retired buffers stay
    under four times `decode_control_slots(rows)` for the largest `rows` seen,
    until `clear_topk_per_row_decode_workspace_cache()`.
    """
    need = _adaptive.decode_control_slots(rows)
    key = (device, stream_id)
    buf = _adaptive_control.get(key)
    if (
        buf is not None
        and buf.numel() >= need
        and key not in _unzeroed_adaptive_control
    ):
        return buf
    capture = _capture_id(stream_id)
    if capture is not None and not ROUTES_INSIDE_CAPTURE:
        # This torch serves a capture from the graph's pool, where a cached buffer
        # could share an address with an intermediate; keep it graph-private.
        return torch.zeros(need, device=device, dtype=torch.int32)
    if buf is None or buf.numel() < need:
        if buf is not None:
            _retired_adaptive_control.append(buf)
        with persistent_alloc(device):
            buf = torch.zeros(
                _adaptive.decode_control_slots(1 << (rows - 1).bit_length()),
                device=device,
                dtype=torch.int32,
            )
        _adaptive_control[key] = buf
    elif capture is None or _unzeroed_adaptive_control[key] != capture:
        buf.zero_()
    if capture is None:
        _unzeroed_adaptive_control.pop(key, None)
    else:
        _unzeroed_adaptive_control[key] = capture
    return buf


def clear_topk_per_row_decode_workspace_cache() -> None:
    """Free the cached workspaces; no graph captured over them may replay after."""
    _clear_split_workspace_cache()
    _adaptive_control.clear()
    _retired_adaptive_control.clear()
    _unzeroed_adaptive_control.clear()


@lru_cache(maxsize=128)
def _validate_topk_signature(
    logits_shape: torch.Size,
    logits_stride: tuple[int, ...],
    logits_dtype: torch.dtype,
    logits_device: torch.device,
    seq_lens_shape: torch.Size,
    seq_lens_stride: tuple[int, ...],
    seq_lens_dtype: torch.dtype,
    seq_lens_device: torch.device,
    indices_shape: torch.Size,
    indices_stride: tuple[int, ...],
    indices_dtype: torch.dtype,
    indices_device: torch.device,
    next_n: int,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int,
) -> None:
    if len(logits_shape) != 2 or logits_dtype != torch.float32:
        raise ValueError("logits must be a 2D CUDA float32 tensor")
    if logits_device.type != "cuda":
        raise ValueError("logits must be a 2D CUDA float32 tensor")
    if k <= 0 or k > logits_shape[1]:
        raise ValueError("k must be in the range [1, logits.shape[1]]")
    if logits_stride[1] != 1:
        raise ValueError("logits must have inner stride 1")

    rows = logits_shape[0]
    if num_rows != rows:
        raise ValueError("num_rows must equal logits.shape[0]")
    # There used to be a refusal here at 4 GiB. The kernels built one buffer
    # descriptor over the whole logits tensor, and both `num_records` and the
    # load's voffset in it are 32-bit byte quantities, so a row past 4 GiB was
    # unaddressable and read as zero -- silently wrong rather than faulting, at
    # 4096 x 262144 but not 4095 x 262144. They now slice the row before
    # building the descriptor, which puts the row's base in the descriptor's
    # 48-bit base address and leaves only an offset within the row, so the
    # tensor as a whole is no longer bounded. The refusal went with it.
    if (stride0, stride1) != logits_stride:
        raise ValueError("stride0 and stride1 must match logits strides")

    if len(seq_lens_shape) != 1 or seq_lens_dtype != torch.int32:
        raise ValueError("seq_lens must be a 1D int32 tensor")
    if seq_lens_stride != (1,):
        raise ValueError("seq_lens must be contiguous")
    if seq_lens_device != logits_device:
        raise ValueError("seq_lens must be on the same CUDA device as logits")
    if next_n <= 0:
        raise ValueError("next_n must be positive")
    required_seq_lens = (rows + next_n - 1) // next_n
    if seq_lens_shape[0] < required_seq_lens:
        raise ValueError("seq_lens does not have enough entries for logits rows")

    if indices_shape != (rows, k) or indices_dtype != torch.int32:
        raise ValueError("indices must be an int32 tensor with shape [rows, k]")
    if indices_stride != (k, 1):
        raise ValueError("indices must be contiguous")
    if indices_device != logits_device:
        raise ValueError("indices must be on the same CUDA device as logits")


@lru_cache(maxsize=128)
def _validate_values_signature(
    values_shape: torch.Size,
    values_stride: tuple[int, ...],
    values_dtype: torch.dtype,
    values_device: torch.device,
    rows: int,
    k: int,
    logits_device: torch.device,
) -> None:
    if values_dtype != torch.float32:
        raise ValueError("values must be a float32 tensor")
    if values_shape != (rows, k):
        raise ValueError("values must have shape [rows, k]")
    if values_stride != (k, 1):
        raise ValueError("values must be contiguous")
    if values_device != logits_device:
        raise ValueError("values must be on the same CUDA device as logits")


def _validate_flydsl_topk_call(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int,
    values: torch.Tensor | None,
) -> None:
    _validate_topk_signature(
        logits.shape,
        logits.stride(),
        logits.dtype,
        logits.device,
        seq_lens.shape,
        seq_lens.stride(),
        seq_lens.dtype,
        seq_lens.device,
        indices.shape,
        indices.stride(),
        indices.dtype,
        indices.device,
        next_n,
        num_rows,
        stride0,
        stride1,
        k,
    )
    if values is not None:
        _validate_values_signature(
            values.shape,
            values.stride(),
            values.dtype,
            values.device,
            logits.shape[0],
            k,
            logits.device,
        )


_FLYDSL_TOPK_ONE_BLOCK_ARCHES = ("gfx950", "gfx1250")


def _validate_radix_topk_one_block_call(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor | None,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int,
    *,
    is_decode: bool,
    next_n: int,
    width: int | None = None,
) -> None:
    """Raise if this call cannot run the one-block radix kernel."""
    _validate_flydsl_topk_call(
        logits, next_n, row_ends, indices, num_rows, stride0, stride1, k, values
    )
    if row_starts is not row_ends:
        _validate_flydsl_topk_call(
            logits, next_n, row_starts, indices, num_rows, stride0, stride1, k, values
        )
    if (logits.shape[1] if width is None else width) > _MAX_ROW_ELEMENTS:
        raise ValueError("one logits row exceeds the AMD buffer descriptor span")


_TensorSignature = tuple[torch.Size, tuple[int, ...], torch.dtype, torch.device]


def _tensor_signature(tensor: torch.Tensor) -> _TensorSignature:
    return tensor.shape, tensor.stride(), tensor.dtype, tensor.device


@lru_cache(maxsize=128)
def _is_flydsl_topk_call_supported(
    logits_signature: _TensorSignature,
    seq_lens_signature: _TensorSignature,
    indices_signature: _TensorSignature,
    next_n: int,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int,
    values_signature: _TensorSignature | None,
) -> bool:
    try:
        _validate_topk_signature(
            *logits_signature,
            *seq_lens_signature,
            *indices_signature,
            next_n,
            num_rows,
            stride0,
            stride1,
            k,
        )
        if values_signature is not None:
            _validate_values_signature(
                *values_signature, logits_signature[0][0], k, logits_signature[3]
            )
    except (RuntimeError, TypeError, ValueError):
        return False
    return True


@lru_cache(maxsize=128)
def _is_flydsl_radix_topk_one_block_call_supported(
    logits_signature: _TensorSignature,
    row_starts_signature: _TensorSignature | None,
    row_ends_signature: _TensorSignature,
    indices_signature: _TensorSignature,
    next_n: int,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int,
    values_signature: _TensorSignature | None,
    is_decode: bool,
    arch: str,
) -> bool:
    if arch not in _FLYDSL_TOPK_ONE_BLOCK_ARCHES:
        return False
    if logits_signature[0][1] > _MAX_ROW_ELEMENTS:
        return False
    if not is_decode and row_starts_signature is None:
        return False
    if not _is_flydsl_topk_call_supported(
        logits_signature,
        row_ends_signature,
        indices_signature,
        next_n,
        num_rows,
        stride0,
        stride1,
        k,
        values_signature,
    ):
        return False
    if row_starts_signature is None or row_starts_signature == row_ends_signature:
        return True
    return _is_flydsl_topk_call_supported(
        logits_signature,
        row_starts_signature,
        indices_signature,
        next_n,
        num_rows,
        stride0,
        stride1,
        k,
        values_signature,
    )


def _is_flydsl_radix_topk_one_block_supported(
    logits: torch.Tensor,
    row_starts: torch.Tensor | None,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor | None,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int,
    *,
    is_decode: bool = False,
    next_n: int = 1,
) -> bool:
    """Return whether the one-block radix kernel supports this call."""
    return _is_flydsl_radix_topk_one_block_call_supported(
        _tensor_signature(logits),
        None if row_starts is None else _tensor_signature(row_starts),
        _tensor_signature(row_ends),
        _tensor_signature(indices),
        next_n,
        num_rows,
        stride0,
        stride1,
        k,
        None if values is None else _tensor_signature(values),
        is_decode,
        get_gfx_runtime(),
    )


def is_flydsl_top_k_per_row_decode_supported(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int,
    values: torch.Tensor | None = None,
) -> bool:
    """Return whether a call satisfies every FlyDSL-only precondition."""
    return _is_flydsl_topk_call_supported(
        _tensor_signature(logits),
        _tensor_signature(seq_lens),
        _tensor_signature(indices),
        next_n,
        num_rows,
        stride0,
        stride1,
        k,
        None if values is None else _tensor_signature(values),
    )


def _run_adaptive(
    logits,
    next_n,
    seq_lens,
    indices,
    rows,
    width,
    stride0,
    stride1,
    k,
    stable,
    stream,
    cfg_width=None,
    row_starts=None,
):
    """Launch the adaptive kernel for a shape the gate has already chosen it for.

    `cfg_width` is the longest row this launch can contain and `width` is how
    wide the buffer holding it is; they differ when a caller passed a
    `max_row_len`. The config takes the former, because every quantity it picks
    -- the tier, the blocks per row, the parts and the compact path -- is sized
    for the work, and the padding is not
    work. The kernel still folds each row onto as many of the launch's
    workgroups as that row's own `seq_lens` entry needs.

    `cfg_width` defaults to `width` for a caller that states no bound.
    `row_starts` makes the rows packed, `width` then being the plane's.
    """
    if cfg_width is None:
        cfg_width = width
    # The gate's own count, so the band it admitted and the grid built here
    # describe the same card.
    cfg = _adaptive.decode_adaptive_config(
        rows,
        cfg_width,
        k,
        ordered=stable,
        cu_count=_decode_cu_count(logits.device.index),
    )
    packed = row_starts is not None
    launcher = _adaptive.create_topk_per_row_decode_adaptive_kernel(
        top_k=k, packed_rows=packed, **cfg.kw
    )
    control = _get_adaptive_control(logits.device, stream.cuda_stream, rows)
    scratch = (
        torch.empty(cfg.scratch_slots, device=logits.device, dtype=torch.int32)
        if cfg.scratch_slots
        else control  # unread without compact
    )
    _run_compiled(
        launcher,
        logits,
        next_n,
        seq_lens,
        # unread unless packed; seq_lens keeps the launch's signature
        row_starts if packed else seq_lens,
        indices,
        control,
        scratch,
        rows,
        width,
        stride0,
        stride1,
        cfg.grid,
        cfg.parts,
        cfg.parts_magic,
        cfg.short_max,
        stream,
    )


def flydsl_top_k_per_row_decode(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int = 2048,
    stable: bool = False,
    values: torch.Tensor | None = None,
    max_row_len: int | None = None,
    *,
    row_starts: torch.Tensor | None = None,
    plane_width: int | None = None,
) -> None:
    """Write per-row TopK indices using each request's effective context length.

    `aiter.ops.topk.top_k_per_row_decode` holds the `max_row_len` contract.
    Before CUDAGraph capture, warm up on the capture stream so the split and
    adaptive paths find their stream-local counters already allocated and zeroed.

    `row_starts` [rows] i32: row r's scores start at element row_starts[r]
    of `logits`' first row (a packed buffer viewed with stride0 0), the
    indices still relative to that start. `plane_width` (default the
    logits' width) bounds every row and picks the kernel, as that plane's
    width would.
    """
    _decode_with_backend(
        logits,
        next_n,
        seq_lens,
        indices,
        num_rows,
        stride0,
        stride1,
        k,
        stable,
        values,
        None,
        max_row_len,
        row_starts=row_starts,
        plane_width=plane_width,
    )


def _decode_with_backend(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int,
    stable: bool,
    values: torch.Tensor | None,
    backend: str | None,
    max_row_len: int | None,
    *,
    row_starts: torch.Tensor | None = None,
    plane_width: int | None = None,
) -> None:
    """Run the decode on `backend`, which must be the gate's answer or None to ask it.

    A `default` answer still runs the single-launch kernel here, because this entry
    point has no one-block dispatch below it.
    """
    _validate_flydsl_topk_call(
        logits, next_n, seq_lens, indices, num_rows, stride0, stride1, k, values
    )

    rows, width = logits.shape
    width = width if plane_width is None else plane_width
    if rows == 0:
        return
    # The bare arch name: the selector keys its radix layout and LDS on it.
    arch = get_gfx_runtime()
    wave_size = get_warp_size(arch)
    stream = torch.cuda.current_stream(logits.device)
    packed = row_starts is not None
    starts = row_starts if packed else seq_lens

    if backend is None:
        backend = decode_backend_for_call(
            logits,
            next_n,
            seq_lens,
            indices,
            rows,
            stride0,
            stride1,
            k,
            stable,
            values,
            max_row_len=max_row_len,
            width=plane_width,
        )

    if backend == BACKEND_ADAPTIVE:
        # Only the adaptive path takes a host-side config, and it is reached only
        # when the gate admitted the call, which needs `max_row_len`. So the
        # width is computed here, not before the gate: `decode_adaptive_width`
        # requires a bound and a `None` reaching it is a bug, not a fallback.
        cfg_width = decode_adaptive_width(width, max_row_len)
        _run_adaptive(
            logits,
            next_n,
            seq_lens,
            indices,
            rows,
            width,
            stride0,
            stride1,
            k,
            stable,
            stream,
            cfg_width=cfg_width,
            row_starts=row_starts,
        )
        return

    if width <= _ONE_WORKGROUP_MAX_ROW_WIDTH:
        launcher = build_topk_per_row_decode_one_workgroup_module(
            k,
            wave_size=wave_size,
            write_values=values is not None,
            packed_rows=packed,
        )
        _run_compiled(
            launcher,
            logits,
            starts,
            seq_lens,
            indices,
            values if values is not None else logits,
            width,
            next_n,
            stride0,
            rows,
            stream,
        )
        return

    # Allocated width bounds every row, but a wide buffer may still hold short
    # rows: the kernel reads each row's length and splits only the long ones.
    launch_topk_per_row_decode(
        logits=logits,
        next_n=next_n,
        seq_lens=seq_lens,
        row_starts=starts,
        indices=indices,
        rows=rows,
        width=width,
        k=k,
        stable=stable,
        values=values,
        arch=arch,
        wave_size=wave_size,
        num_cus=get_cu_num(),
        stream=stream,
        packed_rows=packed,
    )


def flydsl_radix_topk_one_block(
    logits: torch.Tensor,
    row_starts: torch.Tensor | None,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor | None,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int = 2048,
    stable: bool = False,
    *,
    is_decode: bool = False,
    next_n: int = 1,
    plane_width: int | None = None,
    packed_rows: bool = False,
) -> None:
    """Launch the one-block radix TopK kernel for one prefill or decode call.

    With ``packed_rows`` in decode mode, row r starts at ``row_starts[r]`` in
    the input's first row, while output indices remain relative to that start.
    ``plane_width`` defaults to the logits width; it bounds every row and
    selects the same kernel variant that an unpacked plane of that width uses.
    """
    if packed_rows and (not is_decode or row_starts is None):
        raise ValueError("packed rows require decode mode and row_starts")
    if row_starts is None:
        if not is_decode:
            raise ValueError("row_starts is required for prefill")
        row_starts = row_ends
    _validate_radix_topk_one_block_call(
        logits,
        row_starts,
        row_ends,
        indices,
        values,
        num_rows,
        stride0,
        stride1,
        k,
        is_decode=is_decode,
        next_n=next_n,
        width=plane_width,
    )
    if num_rows == 0:
        return

    arch = get_gfx_runtime()
    width = logits.shape[1] if plane_width is None else plane_width
    wave_size = get_warp_size(arch)
    stream = torch.cuda.current_stream(logits.device)
    short_rows = width <= _COMPACT_CAPACITY
    block_threads = (
        1024 if not short_rows or num_rows <= _SHORT_ROWS_1024_THREAD_MAX_ROWS else 256
    )
    launcher = build_radix_topk_one_block_module(
        k,
        block_threads=block_threads,
        write_values=values is not None,
        stable=stable,
        short_rows=short_rows,
        is_decode=is_decode,
        wave_size=wave_size,
        arch=arch,
        packed_rows=packed_rows,
    )
    _run_compiled(
        launcher,
        logits,
        row_starts,
        row_ends,
        indices,
        values if values is not None else logits,
        width,
        next_n,
        num_rows,
        stream,
    )
