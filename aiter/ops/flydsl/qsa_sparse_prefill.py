# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""QSA sparse prefill on gfx942 and gfx950: BF16 D256 sparse attention over packed K/V.

Public entry for the ``sparse_prefill_*`` kernels in ``kernels/qsa/``. Each call
recovers the selected four-token blocks (FlyDSL K1-K3), routes every query tile to
the union kernel (queries of a tile share one compact block list with exact
per-query membership) or to the direct kernel (one query per wave over its own
blocks), and writes O in place. Plans and scratch are private. K/V may also be a
token pool read in place through a slot table, FP8 (E4M3FNUZ) with FP32 scales on
gfx942, and on gfx950 their rows may be strided (an interleaved K|V cache read in
place).
"""

import math
import threading
import weakref
from collections import OrderedDict
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

from .kernels.qsa import sparse_prefill_direct as direct
from .kernels.qsa import sparse_prefill_direct_fp8 as direct_fp8
from .kernels.qsa import sparse_prefill_prepare as prepare
from .kernels.qsa import sparse_prefill_union as union
from .kernels.qsa import sparse_prefill_union_fp8 as union_fp8
from .kernels.qsa.sparse_prefill_common import (
    ARENA_GROWTH_PERCENT,
    SCRATCH_ALIGN,
    UNION_SCRATCH_LIMIT,
    WORKSPACE_CACHE,
    platform,
)

__all__ = [
    "SELECTION_WIDTHS",
    "flydsl_qsa_sparse_prefill",
    "flydsl_qsa_sparse_prefill_available",
    "flydsl_qsa_sparse_prefill_supports_selection",
    "qsa_sparse_prefill_problem",
    "qsa_sparse_prefill_warmup",
]

# The FP8 kernels run E4M3FNUZ, gfx942's FP8 (aiter's dtypes.fp8 there).
FP8 = torch.float8_e4m3fnuz
SHAPE = "Q/K/V must be BF16 [tokens, heads, 256] (K/V may be FP8 E4M3FNUZ on gfx942)"

# Kernels use unsigned 32-bit buffer byte offsets; the 1 MiB margin keeps out-of-range
# sentinels plus their lane offsets from wrapping.
_MAX_SPAN = 2**32 - 2**20
# Slot-table pools past this span take raw direct's 64-bit loads (WIDE).
_WIDE_SPAN = _MAX_SPAN
# Kernels form a four-row K/V block's extent and offsets (up to 4 x a row's bytes) as
# 32-bit values: a row stride below 2**28 elements keeps them below 2**31.
_MAX_KV_STRIDE = 2**28

# Selection widths 4T + 3 for the per-row block budgets T the kernels take.
SELECTION_WIDTHS = tuple(4 * topk + 3 for topk in (256, 512, 768, 1024))
# The selection is addressed at signed 32-bit byte offsets.
_MAX_SELECTION_BYTES = 2**31 - 1


def flydsl_qsa_sparse_prefill_available() -> bool:
    """Whether ``flydsl_qsa_sparse_prefill`` runs on the live GPU (gfx942 or gfx950)."""
    return get_gfx_runtime() in ("gfx942", "gfx950")


def flydsl_qsa_sparse_prefill_supports_selection(rows: int, width: int) -> bool:
    """Whether ``flydsl_qsa_sparse_prefill`` takes an int32 [rows, width] selection:
    width is one of ``SELECTION_WIDTHS`` and the selection spans fewer than 2**31
    bytes. Only the selection's shape is answered; the GPU is answered by
    ``flydsl_qsa_sparse_prefill_available`` and Q/K/V, kv_slots and the lengths
    are checked by the call itself.
    """
    return width in SELECTION_WIDTHS and 0 <= rows * width * 4 <= _MAX_SELECTION_BYTES


def _union_min_rows(group, fp8=None):
    config = platform(fp8)
    return max(config.union_min_rows, config.union_min_rows_per_head * group)


def _aligned(nbytes):
    return -(-nbytes // SCRATCH_ALIGN) * SCRATCH_ALIGN


def _kv_stride(tensor):
    """Elements between the rows of K or V [rows, heads, 256], or None unless every row
    holds dense [heads, 256] values, 16-byte aligned, before the next row starts, and
    the next row starts fewer than _MAX_KV_STRIDE elements on. Size-1 dims have no
    stride of their own (torch's contiguity rule): one row is dense."""
    rows, heads, dim = tensor.shape
    stride = tensor.stride(0) if rows > 1 else heads * dim
    if (
        tensor.stride(2) != 1
        or (heads > 1 and tensor.stride(1) != dim)
        or stride < heads * dim
        or stride % 8
        or stride >= _MAX_KV_STRIDE
    ):
        return None
    return stride


def _span_bytes(rows, stride, heads, size=2):
    """Bytes from the first of rows [heads, 256] rows of size-byte elements, stride
    elements apart, to the end of the last."""
    return ((rows - 1) * stride + heads * 256) * size if rows else 0


class _Arena:
    """Scratch shared by one stream's workspaces: every call rewrites what it reads.

    Growth replaces the buffer; captured workspaces keep theirs alive.
    """

    def __init__(self):
        self.buffers = [None]
        self.generation = 0
        self.users = weakref.WeakSet()

    def reserve(self, size, device):
        buffer = self.buffers[0]
        if buffer is None or buffer.numel() < size:
            growth = 0 if buffer is None else buffer.numel() * ARENA_GROWTH_PERCENT
            size = max(size, growth // 100)
            self.buffers[0] = torch.empty(size, dtype=torch.uint8, device=device)
            self.generation += 1
            for user in list(self.users):
                user.release()


# Weak values: an arena lives while a cached or captured workspace uses it.
_arenas = weakref.WeakValueDictionary()


class _Workspace:
    def __init__(
        self,
        q,
        k,
        v,
        indices,
        query_lens,
        prefix_lens,
        kv_stride,
        kv_slots,
        invariant,
        fp8,
    ):
        # fp8: None for BF16 K/V, else "tensor" / "token" for FP8 K/V and its scale layout.
        self.fp8 = fp8
        topk = (indices.shape[1] - 3) // 4
        q_lens = np.asarray(query_lens, dtype=np.int64)
        lengths = q_lens + np.asarray(prefix_lens, dtype=np.int64)
        sequence = np.repeat(np.arange(len(q_lens)), q_lens)
        positions = np.arange(len(sequence)) + (lengths - np.cumsum(q_lens))[sequence]
        slotted = kv_slots is not None
        # Past this pool span raw direct byte offsets no longer reach every row.
        span = _span_bytes(k.shape[0], kv_stride, k.shape[1], k.element_size())
        wide = slotted and span > _WIDE_SPAN
        self.metadata = {
            "query_lens": query_lens,
            "prefix_lens": prefix_lens,
            "topk": topk,
            "query_positions": prepare.upload(positions, q.device),
            "max_seqlen_k": int(lengths.max(initial=0)),
            "wide": wide,
            "kv_stride": kv_stride,
        }
        self.captured = False
        self.block_indices = None
        inputs = SimpleNamespace(
            q=q,
            k=k,
            v=v,
            indices=indices,
            block_indices=None,
            kv_slots=kv_slots,
            **self.metadata,
        )
        self.union = None
        minimum = _union_min_rows(q.shape[1] // k.shape[1], fp8)
        # Batch-invariant layouts plan union tiles only for requests of at least minimum rows.
        if (max(query_lens) if invariant else q.shape[0]) >= minimum:
            plan = prepare.allocate_plan(
                inputs=inputs,
                query_tile=32,
                scratch=False,
                slotted=slotted,
                min_request_rows=minimum if invariant else None,
                fp8=fp8,
            )
            # Union scratch grows with tiles x their block capacity.
            needed = sum(
                _aligned(math.prod(shape) * dtype.itemsize)
                for _, shape, dtype in prepare.scratch_specs(plan)
            )
            if needed <= UNION_SCRATCH_LIMIT:
                self.union = plan
        self.specs = [(self, "block_indices", (q.shape[0], topk), torch.int32)]
        if fp8 is None:
            self.direct = direct.prepare(
                inputs=inputs, union=self.union, scratch=False, invariant=invariant
            )
            direct_specs = direct.scratch_specs(self.direct)
        else:
            self.direct = direct_fp8.prepare(
                inputs=inputs, union=self.union, per_token=fp8 == "token", scratch=False
            )
            direct_specs = direct_fp8.scratch_specs(self.direct)
            # FP8 Q of the pack launch, read by both FP8 kernels.
            self.specs += [
                (self, "q8", tuple(q.shape), torch.uint8),
                (self, "qs", tuple(q.shape[:2]), torch.float32),
            ]
        if self.union is not None:
            self.specs += [
                (self.union, *spec) for spec in prepare.scratch_specs(self.union)
            ]
        self.specs += [(self.direct, *spec) for spec in direct_specs]
        size = sum(
            _aligned(math.prod(shape) * dtype.itemsize)
            for *_, shape, dtype in self.specs
        )
        self.arena = _arenas.setdefault(
            (q.device, torch.cuda.current_stream(q.device).cuda_stream), _Arena()
        )
        self.generation = None
        self.pinned = None
        self.hot = None
        self.arena.reserve(size, q.device)

    def release(self):
        if self.pinned is None:
            for owner, name, *_ in self.specs:
                setattr(owner, name, None)
            self.direct.source_blocks = None
            self.generation = None
            self.hot = None

    def attach(self):
        """Point the plans at this stream's scratch (again after arena growth); pin it for capture."""
        if self.pinned is None and self.generation != self.arena.generation:
            offset = 0
            for owner, name, shape, dtype in self.specs:
                nbytes = math.prod(shape) * dtype.itemsize
                view = self.arena.buffers[0][offset : offset + nbytes]
                setattr(owner, name, view.view(dtype).view(shape))
                offset += _aligned(nbytes)
            self.generation = self.arena.generation
            self.arena.users.add(self)
            self.hot = None
        self.direct.source_blocks = self.block_indices
        if self.pinned is None and torch.cuda.is_current_stream_capturing():
            # Graph pointers must outlive arena growth.
            self.pinned = tuple(self.arena.buffers)

    def bind(self, q, k, v, indices, kv_slots, scale_log2, k_scale=None, v_scale=None):
        self.attach()
        inputs = SimpleNamespace(
            q=q,
            k=k,
            v=v,
            indices=indices,
            block_indices=self.block_indices,
            kv_slots=kv_slots,
            scale_log2=scale_log2,
            **self.metadata,
        )
        if self.fp8 is not None:
            inputs.k_scale, inputs.v_scale = k_scale, v_scale
            inputs.q8, inputs.qs = self.q8, self.qs
        return inputs


class _Hot:
    """Replays one workspace binding: the FlyDSL launchers with prebuilt arguments.
    Building it performs the current call through the normal paths.
    """

    def __init__(
        self,
        workspace,
        q,
        k,
        v,
        indices,
        out,
        kv_slots,
        scale_log2,
        k_scale=None,
        v_scale=None,
    ):
        inputs = workspace.bind(
            q, k, v, indices, kv_slots, scale_log2, k_scale, v_scale
        )
        prepare.run(inputs=inputs, plan=workspace.union)
        self.prepare = prepare.launchers(inputs=inputs, plan=workspace.union)
        self.compute = []
        self.fp8 = workspace.fp8 is not None
        if self.fp8:
            # Q8/QS stay bound to this hot's scratch views.
            self.q8, self.qs = inputs.q8.view(-1), inputs.qs.view(-1)
            # The FP8 pack (and Q quantization in its launch) precedes the union kernel,
            # which reads its Q8.
            direct_fp8.run(
                inputs=inputs, prepared=workspace.direct, out=out, stage="pack"
            )
            self.union_fp8 = None
            if workspace.union is not None:
                union_fp8.run(inputs=inputs, plan=workspace.union, out=out)
                self.union_fp8 = union_fp8.launcher(
                    inputs=inputs, plan=workspace.union, out=out
                )
            direct_fp8.run(
                inputs=inputs, prepared=workspace.direct, out=out, stage="direct"
            )
            launches = direct_fp8.launchers(
                inputs=inputs, prepared=workspace.direct, out=out
            )
            self.pack_fp8, self.direct_fp8 = launches or (None, None)
            return
        if workspace.union is not None:
            union.run(inputs=inputs, plan=workspace.union, out=out)
            self.compute.append(
                union.launcher(inputs=inputs, plan=workspace.union, out=out)
            )
        direct.run(inputs=inputs, prepared=workspace.direct, out=out)
        self.compute.append(
            direct.launcher(inputs=inputs, prepared=workspace.direct, out=out)
        )
        self.compute = [launch for launch in self.compute if launch is not None]

    def prepare_call(self, indices, kv_slots):
        recover, *rest = self.prepare
        recover(indices)
        for launch in rest:
            launch(kv_slots)

    def compute_call(self, q, k, v, out, kv_slots, scale_log2, k_scale, v_scale):
        # K/V stay 3-D: FlyDSL passes int32 dims and K/V may exceed 2**31 elements.
        if self.fp8:
            scales = (k_scale.view(-1), v_scale.view(-1))
            if self.pack_fp8 is not None:
                self.pack_fp8(q.view(-1), k, v, *scales, kv_slots)
            if self.union_fp8 is not None:
                self.union_fp8(
                    self.q8, self.qs, k, v, out.view(-1), *scales, scale_log2
                )
            if self.direct_fp8 is not None:
                self.direct_fp8(
                    self.q8, self.qs, k, v, out.view(-1), *scales, kv_slots, scale_log2
                )
            return
        views = (q.view(-1), k, v, out.view(-1), kv_slots, scale_log2)
        for launch in self.compute:
            launch(*views)


class _Rejected(NamedTuple):
    error: type[Exception]
    message: str


def _is_fp8(tensor):
    return tensor.dtype.is_floating_point and tensor.element_size() == 1


def _fp8_layout(k, v, k_scale, v_scale):
    """None for BF16 K/V; "tensor" or "token" for FP8 K/V (of any FP8 format: the
    architecture and dtype rules reject what the kernels do not run) with valid scales;
    else the scale rule broken, as _Rejected."""
    if not _is_fp8(k) and not _is_fp8(v):
        if k_scale is not None or v_scale is not None:
            return _Rejected(ValueError, "k_scale/v_scale apply only to FP8 K/V")
        return None
    per_tensor = (1,)
    per_token = tuple(k.shape[:2])
    for scale in (k_scale, v_scale):
        if (
            not isinstance(scale, torch.Tensor)
            or scale.dtype != torch.float32
            or scale.device != k.device
            or not scale.is_contiguous()
            or tuple(scale.shape) not in (per_tensor, per_token)
        ):
            return _Rejected(
                ValueError,
                "FP8 K/V require contiguous FP32 k_scale/v_scale on their device, of "
                f"shape [1] or {list(per_token)} (K's [N, HK], or [P, HK] for a pool)",
            )
    if k_scale.shape != v_scale.shape:
        return _Rejected(
            ValueError, "k_scale and v_scale must both be per tensor or per token"
        )
    return "tensor" if tuple(k_scale.shape) == per_tensor else "token"


def _dequantize(k, v, k_scale, v_scale, kv_slots):
    """BF16 K/V of the packed context (gathered through kv_slots) for compute="bf16"."""
    result = []
    for x, scale in ((k, k_scale), (v, v_scale)):
        if kv_slots is not None:
            index = kv_slots.long()
            x = x.view(torch.uint8).index_select(0, index).view(FP8)
            if scale.ndim == 2:
                scale = scale.index_select(0, index)
        # FP8 -> BF16 is exact; the FP32 scale product rounds once to BF16.
        x = x.to(torch.bfloat16)
        result.append(x.mul_(scale[..., None] if scale.ndim == 2 else scale))
    return result


_workspaces = OrderedDict()
_lock = threading.RLock()
_device = None


class _Checked(NamedTuple):
    scale_log2: float
    query_lens: tuple
    prefix_lens: tuple
    stride: int
    topk: int
    fp8: str | None


def _on_rocm_device(q):
    return q.device.type == "cuda" and torch.version.hip is not None


def _check_call(
    q,
    k,
    v,
    indices,
    query_lens,
    prefix_lens,
    softmax_scale,
    out,
    kv_slots,
    batch_invariant,
    k_scale,
    v_scale,
    compute,
):
    """The call's operand rules in order: the first one it breaks as _Rejected (what the
    op raises and qsa_sparse_prefill_problem reports), else its normalized arguments.
    Rules on state (graph capture, one GPU per process) stay in the op."""
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        return _Rejected(ValueError, SHAPE)
    if compute not in ("fp8", "bf16"):
        return _Rejected(ValueError, 'compute must be "fp8" or "bf16"')
    fp8 = _fp8_layout(k, v, k_scale, v_scale)
    if isinstance(fp8, _Rejected):
        return fp8
    if isinstance(softmax_scale, (torch.Tensor, bool)):
        return _Rejected(TypeError, "softmax_scale must be a host scalar or None")
    try:
        scale = 0.0625 if softmax_scale is None else float(softmax_scale)
    except OverflowError:
        scale = math.inf  # too large for a float: the range rule below rejects it
    except (TypeError, ValueError):
        return _Rejected(TypeError, "softmax_scale must be a host scalar or None")
    # The kernels multiply masked -inf scores by it: an FP32 zero would give NaN.
    scaled = scale * math.log2(math.e)
    if not math.isfinite(scaled) or not 2**-149 <= scaled <= (2 - 2**-23) * 2**127:
        return _Rejected(
            ValueError, "softmax_scale * log2(e) must be finite and positive in FP32"
        )
    if not isinstance(batch_invariant, bool):
        return _Rejected(TypeError, "batch_invariant must be a bool")
    if kv_slots is not None and (
        not isinstance(kv_slots, torch.Tensor)
        or kv_slots.dtype != torch.int32
        or kv_slots.ndim != 1
        or kv_slots.device != q.device
        or not kv_slots.is_contiguous()
    ):
        return _Rejected(
            ValueError, "kv_slots must be a contiguous device int32 [N] tensor"
        )
    # The kernels read the slot table at unsigned 32-bit buffer offsets and upload
    # each request's start in it as int32.
    if kv_slots is not None and kv_slots.numel() * kv_slots.element_size() > _MAX_SPAN:
        return _Rejected(ValueError, "kv_slots requires a byte span <= 2**32 - 2**20")
    keys = k.shape[0] if kv_slots is None else kv_slots.shape[0]
    query_lens = (q.shape[0],) if query_lens is None else tuple(query_lens)
    if prefix_lens is None:
        if len(query_lens) == 1:
            prefix_lens = (keys - q.shape[0],)
        else:
            prefix_lens = (0,) * len(query_lens)
    else:
        prefix_lens = tuple(prefix_lens)
    if (
        len(query_lens) != len(prefix_lens)
        or any(
            type(n) is not int or n < 0 or n >= 2**31
            for n in (*query_lens, *prefix_lens)
        )
        or sum(query_lens) != q.shape[0]
        or sum(query_lens) + sum(prefix_lens) != keys
    ):
        return _Rejected(
            ValueError, "Host query/prefix lengths must describe the packed Q/K/V"
        )
    if not _on_rocm_device(q):
        return _Rejected(
            ValueError, "QSA sparse attention requires ROCm gfx942 or gfx950"
        )
    if fp8 is not None and get_gfx_runtime() != "gfx942":
        return _Rejected(ValueError, "FP8 K/V are supported on gfx942 only")
    kv_dtype = torch.bfloat16 if fp8 is None else FP8
    for tensor, dtype in ((q, torch.bfloat16), (k, kv_dtype), (v, kv_dtype)):
        if (
            tensor.dtype != dtype
            or tensor.device != q.device
            or tensor.layout != torch.strided
            or tensor.shape[-1] != 256
        ):
            return _Rejected(ValueError, SHAPE)
        if tensor.data_ptr() % 16:
            return _Rejected(ValueError, "Q/K/V require 16-byte alignment")
    if not q.is_contiguous():
        return _Rejected(ValueError, "Q must be contiguous")
    if (
        k.shape != v.shape
        or q.shape[1] <= 0
        or k.shape[1] <= 0
        or q.shape[1] % k.shape[1]
    ):
        return _Rejected(
            ValueError,
            "K/V must match and Q heads must be a positive multiple of KV heads",
        )
    stride = _kv_stride(k)
    if stride is None or _kv_stride(v) != stride:
        return _Rejected(
            ValueError,
            "K/V rows must hold dense [heads, 256] values one 16-byte aligned stride "
            ">= heads x 256 and < 2**28 elements apart, the same for K and V",
        )
    if q.shape[1] // k.shape[1] > 16:
        return _Rejected(
            ValueError, "At most 16 Q heads per KV head fit the direct fallback wave"
        )
    # K/V are addressed per request, so only each request's K (or V) span is bounded,
    # not the packed total: about 8M context tokens at HK1, 4M at HK2 (dense rows). A
    # slot-table pool is addressed from its base instead, except with compute="bf16":
    # FP8 K/V then run as packed BF16 copies.
    tokens = max(map(sum, zip(query_lens, prefix_lens)), default=0)
    span = q.numel() * q.element_size()
    if kv_slots is None or (fp8 is not None and compute == "bf16"):
        span = max(span, _span_bytes(tokens, stride, k.shape[1]))
    if span > _MAX_SPAN:
        return _Rejected(
            ValueError, "Q and each request's K/V require byte spans <= 2**32 - 2**20"
        )
    if not flydsl_qsa_sparse_prefill_available():
        return _Rejected(ValueError, "QSA sparse attention requires gfx942 or gfx950")
    # Strided rows are validated on gfx950 only.
    if stride != k.shape[1] * 256 and get_gfx_runtime() != "gfx950":
        return _Rejected(ValueError, "Strided K/V rows are supported on gfx950 only")
    if (
        indices.ndim != 2
        or indices.shape[0] != q.shape[0]
        or not flydsl_qsa_sparse_prefill_supports_selection(*indices.shape)
        or indices.dtype != torch.int32
        or indices.device != q.device
        or not indices.is_contiguous()
    ):
        return _Rejected(
            ValueError,
            "indices must be contiguous device int32 [M,4T+3], T = 256/512/768/1024, "
            "with byte span < 2**31",
        )
    topk = (indices.shape[1] - 3) // 4
    if out is not None:
        if (
            out.shape != q.shape
            or out.dtype != q.dtype
            or out.device != q.device
            or not out.is_contiguous()
            or out.data_ptr() % 16
        ):
            return _Rejected(ValueError, "out must be aligned, contiguous and match Q")
        begin, end = out.data_ptr(), out.data_ptr() + out.numel() * out.element_size()
        # Strided K/V rows hold their heads x 256 values every stride elements; the
        # padding between rows is not read, so only the rows themselves are checked.
        size = k.element_size()
        row_bytes, row_step = k.shape[1] * 256 * size, stride * size
        overlaps = False
        for tensor in (k, v):
            # Rows r with begin < start + r * step + row_bytes, start + r * step < end.
            first = max(0, (begin - tensor.data_ptr() - row_bytes) // row_step + 1)
            last = min(k.shape[0] - 1, (end - tensor.data_ptr() - 1) // row_step)
            overlaps = overlaps or first <= last
        sources = (q, indices) + (() if kv_slots is None else (kv_slots,))
        if fp8 is not None:
            sources += (k_scale, v_scale)
        for source in sources:
            pointer = source.data_ptr()
            if (
                begin < pointer + source.numel() * source.element_size()
                and pointer < end
            ):
                overlaps = True
        if overlaps:
            return _Rejected(
                ValueError, "out must not overlap Q/K/V, indices, kv_slots or scales"
            )
    scale_log2 = float(np.float32(scaled))
    return _Checked(scale_log2, query_lens, prefix_lens, stride, topk, fp8)


def qsa_sparse_prefill_problem(
    q,
    k,
    v,
    indices,
    *,
    query_lens=None,
    prefix_lens=None,
    softmax_scale=None,
    out=None,
    kv_slots=None,
    batch_invariant=False,
    k_scale=None,
    v_scale=None,
    compute="fp8",
):
    """The first operand rule this call breaks, as the message flydsl_qsa_sparse_prefill
    would raise it, or None when the operands are accepted. Takes the op's arguments;
    states of the process (graph capture, one GPU per process) are not operand rules,
    so a call accepted here can still raise them."""
    checked = _check_call(
        q,
        k,
        v,
        indices,
        query_lens,
        prefix_lens,
        softmax_scale,
        out,
        kv_slots,
        batch_invariant,
        k_scale,
        v_scale,
        compute,
    )
    return checked.message if isinstance(checked, _Rejected) else None


# A prefix keeps a request off the full-prefill tiles (BQ21/BQ42) in both batch modes.
_WARMUP_PREFIX = 4


def _warmup_layouts(group, hk):
    """(query_lens, prefix_lens) of one call per kernel-variant class of Q heads per KV
    head `group` and `hk` KV heads, in both batch modes and for every K/V dtype the
    architecture takes. A call's variants follow from whether it gets a union plan (from
    `_union_min_rows` rows; with batch_invariant=True per request), that plan's query
    tile, and for FP8 K/V whether the plan splits its tasks (below
    `prepare.whole_task_rows`); the rest of what a variant compiles in (heads, width,
    stride, pool span, K/V dtype and scale layout, batch mode) is the caller's and fixed
    across calls."""
    layouts = []
    for fp8 in (None, "tensor", "token") if get_gfx_runtime() == "gfx942" else (None,):
        union_rows = _union_min_rows(group, fp8)
        # Direct kernels only; then the base tile and each wide tile the policy takes.
        layouts.append(((max(union_rows - 1, 1),), (0,)))
        union = [((union_rows,), (_WARMUP_PREFIX,))]
        for tile in (21, 42):
            # The first row count that has a union plan and takes the wide tile (a device
            # with few CUs takes it from fewer rows than the union gate).
            rows = max(prepare.wide_tile_rows(tile), union_rows)
            if prepare.plan_query_tile(group, hk, (rows,), (0,), 32) == tile:
                union.append(((rows,), (0,)))
        for query_lens, prefix_lens in union:
            layouts.append((query_lens, prefix_lens))
            if fp8 is not None:
                # And the same tile with whole tasks (a split plan keeps whole union
                # tiles; an unsplit one may take core tiles).
                tile = prepare.plan_query_tile(group, hk, query_lens, prefix_lens, 32)
                rows = max(query_lens[0], prepare.whole_task_rows(group, hk, tile))
                layouts.append(((rows,), prefix_lens))
    return tuple(dict.fromkeys(layouts))


def _prefill_selection(query_lens, prefix_lens, topk, device):
    """indices [rows, 4 topk + 3] in the layout the op takes: the row at request position
    p (v = prefix + p + 1 tokens) selects its last min(v // 4, topk) complete four-token
    blocks, then its v % 4 tail tokens; the rest of the row is -1."""
    visible = torch.cat(
        [
            torch.arange(p + 1, p + n + 1, device=device)
            for n, p in zip(query_lens, prefix_lens)
        ]
    )
    first = 4 * (visible // 4 - topk).clamp(min=0)
    indices = first[:, None] + torch.arange(4 * topk + 3, device=device)
    return torch.where(indices < visible[:, None], indices, -1).to(torch.int32)


def qsa_sparse_prefill_warmup(
    run, *, num_q_heads, num_kv_heads, selection_width, device
):
    """Compile the kernels the op's calls can use, so that no served call compiles them.

    ``run(q, indices, query_lens, prefix_lens)`` makes one call to the op the way the
    caller serves it (its K/V of each request's prefix_lens[i] + query_lens[i] tokens,
    scale, slot table, out, batch mode and K/V dtype), and returns whether it did; the
    compile keys are those of that call. It is called once per kernel-variant class of
    either batch mode and every K/V dtype the architecture takes, on synthetic Q (zeros)
    and indices: below the union threshold, over it with a short prefix, at each wide
    tile's first row count where the tile policy takes one, and (gfx942, for FP8 K/V)
    with as many tasks as keep a plan unsplit. K/V values are not read for compiling,
    so any values do. Returns the (query_lens, prefix_lens) for which ``run`` returned
    true; none on an architecture other than gfx942 and gfx950 (which have no tile
    policy), or when `selection_width` is not a width the op takes.
    """
    if get_gfx_runtime() not in ("gfx942", "gfx950"):
        return ()
    topk, rest = divmod(selection_width - 3, 4)
    if rest or topk % 256 or not 256 <= topk <= 1024:
        return ()
    warmed = []
    for layout in _warmup_layouts(num_q_heads // num_kv_heads, num_kv_heads):
        query_lens, prefix_lens = layout
        q = torch.zeros(
            sum(query_lens), num_q_heads, 256, dtype=torch.bfloat16, device=device
        )
        indices = _prefill_selection(query_lens, prefix_lens, topk, device)
        if run(q, indices, query_lens, prefix_lens):
            warmed.append(layout)
    torch.cuda.synchronize(device)
    return tuple(warmed)


def flydsl_qsa_sparse_prefill(
    q,
    k,
    v,
    indices,
    *,
    query_lens=None,
    prefix_lens=None,
    softmax_scale=None,
    out=None,
    kv_slots=None,
    batch_invariant=False,
    k_scale=None,
    v_scale=None,
    compute="fp8",
):
    """Return QSA sparse attention for BF16 Q[M,H,256] and K/V[N,HK,256].

    indices is contiguous int32[M,4T+3] (T = 256, 512, 768 or 1024, set by the width).
    For a row whose causal window holds v tokens (request-local 0 .. v - 1, the last
    being its own), columns [0, 4 * min(v // 4, T)) hold exactly min(v // 4, T)
    distinct complete blocks j < v // 4, each as its tokens 4j .. 4j + 3, in any
    order; the next v % 4 columns hold the tail tokens 4 * (v // 4) .. v - 1 and the
    rest are -1. The kernels derive both counts from v, so fewer blocks or a missing
    tail give wrong results. K/V must be finite: rows of a request share union tiles,
    so a non-finite value can also reach rows that did not select it. The layout is
    not validated: a selection that breaks it gives wrong results, also in later
    calls on the same stream.

    Host query_lens/prefix_lens describe packed requests. For one request they
    default to (M,) and (N-M,). Plans are built and cached per layout. With kv_slots
    (contiguous int32[N] of at most 2**32 - 2**20 bytes, packed token -> pool row),
    K/V are a token pool [P,HK,256] read in place: each request's four-token blocks,
    the final partial one included, must occupy four consecutive pool rows starting
    at a multiple of four (paged pools whose pages are a multiple of four tokens
    satisfy this), and the result is the same as for K/V packed by those slots. A
    pool may exceed 4 GiB (raw direct then uses 64-bit loads); slot values are not
    validated. On gfx950, K/V may be strided views: every row holds dense [HK, 256]
    values and rows sit one 16-byte aligned stride >= HK x 256 and < 2**28 elements
    apart, the same for K and V (e.g. the halves of one interleaved K|V record per
    token, read in place); each stride compiles its own kernels. Hot calls rebuild the
    selection: per-layout plans share one private per-stream scratch arena. out is
    optional and must not overlap Q/K/V, indices, kv_slots or scales. softmax_scale is
    a kernel argument (FP32 softmax_scale * log2(e)): a new scale compiles nothing and
    reuses the layout's workspace, also under graph capture. Warm the same stream and
    layout before graph capture. Captured workspaces stay pinned for graph pointer
    lifetime and share their stream's scratch: do not replay them concurrently with
    each other or with that stream's calls. Each kernel variant compiles on its first
    use, a few seconds with empty FlyDSL caches; none is prebuilt (PREBUILD_KERNELS /
    AOT do not cover QSA). One GPU per process (TP uses separate workers).

    Union tiles and the direct kernel round differently. By default the choice between
    them spans the call: union tiles are planned by the call's row count, BQ21/BQ42
    union tiles serve only a single full H6/H3 prefill, and a cost model over all
    tiles routes them, so the low bits of a request's rows may change with the other
    requests of the call. batch_invariant=True decides per request: only requests of
    at least the union planning rows take union tiles, each request picks its own tile
    rows and no tile is routed by the others, so a request gives the bits it gives
    alone (with batch_invariant=True too; the modes may differ). Batches of short
    requests then run more rows direct, which can be slower. In both modes, a layout
    whose union scratch would exceed 4 GiB (millions of rows) runs direct only, so its
    requests may differ from their runs alone. Whether direct reads packed or raw K/V
    (a call-wide budget) does not change the bits with batch_invariant=True; by default
    on gfx950 it can, packed K/V running 16x16x32 MFMAs. Tuning targets MI308X (gfx942,
    80 CUs) and MI355X (gfx950, 256 CUs) only: other parts of those architectures log a
    warning and reuse those settings; plans and routes follow the CU count, performance
    may not be optimal.

    On gfx942, K/V may also be FP8 (its E4M3FNUZ, torch.float8_e4m3fnuz: aiter's
    dtypes.fp8 there) with FP32 dequantization scales k_scale / v_scale, both per token
    and KV head ([N,HK], with kv_slots [P,HK]) or both per tensor ([1]); K/V values are
    K8 * k_scale and V8 * v_scale. compute="fp8" (default) runs FP8 MFMA kernels: Q is
    quantized per (query, head) row with scale max(amax, 2**-118) / 240 and the
    probabilities in FP8 (relative output error of a few percent on random data), and
    per-token V scales are clamped to [2**-60, 2**60] (they join the probabilities); as
    for BF16 K/V, direct rows read the context repacked per call while it fits a
    budget, else the raw K/V. Its union plans may split a short prefill's longest tasks
    into two key ranges merged at the end, and keep only a core of an H12/H6 union
    tile, the rows' other blocks running direct and merging by log-sum-exp; whether a
    task splits depends on the whole call, so batch_invariant=True does not split.
    compute="bf16" dequantizes the request tokens to BF16 (each value the FP32 product
    rounded once) and runs the BF16 kernels (BF16 accuracy, temporary BF16 K/V of the
    packed context). Workspaces are kept per K/V dtype and scale layout too; the scale
    tensors are read on every call, also under graph replay.
    """
    global _device
    checked = _check_call(
        q,
        k,
        v,
        indices,
        query_lens,
        prefix_lens,
        softmax_scale,
        out,
        kv_slots,
        batch_invariant,
        k_scale,
        v_scale,
        compute,
    )
    if isinstance(checked, _Rejected):
        raise checked.error(checked.message)
    scale_log2, query_lens, prefix_lens, stride, topk, fp8 = checked
    if out is None:
        out = torch.empty_like(q)
    if q.shape[0] == 0:
        return out
    if fp8 is not None and compute == "bf16":
        k, v = _dequantize(k, v, k_scale, v_scale, kv_slots)
        return flydsl_qsa_sparse_prefill(
            q,
            k,
            v,
            indices,
            query_lens=query_lens,
            prefix_lens=prefix_lens,
            softmax_scale=softmax_scale,
            out=out,
            batch_invariant=batch_invariant,
        )
    with _lock, torch.cuda.device(q.device):
        if _device is not None and _device != q.device:
            raise ValueError("QSA sparse attention supports one GPU per process")
        _device = q.device
        capturing = torch.cuda.is_current_stream_capturing()
        key = (
            q.device,
            torch._C._cuda_getCurrentRawStream(q.device.index),
            query_lens,
            prefix_lens,
            q.shape[1],
            k.shape[1],
            topk,
            # Pool rows: they set the raw direct extent and the WIDE path.
            None if kv_slots is None else k.shape[0],
            batch_invariant,
            # FP8 K/V and their scale layout: None, "tensor" or "token".
            fp8,
            # Row stride: every kernel that reads K/V compiles it in.
            stride,
        )
        workspace = _workspaces.get(key)
        if workspace is None:
            if capturing:
                raise RuntimeError(
                    "Warm attention on this stream and layout before graph capture"
                )
            workspace = _Workspace(
                q,
                k,
                v,
                indices,
                query_lens,
                prefix_lens,
                stride,
                kv_slots,
                batch_invariant,
                fp8,
            )
            _workspaces[key] = workspace
            for stale in list(_workspaces):
                if len(_workspaces) <= WORKSPACE_CACHE:
                    break
                if stale != key and not _workspaces[stale].captured:
                    del _workspaces[stale]
        _workspaces.move_to_end(key)
        workspace.captured |= capturing
        workspace.attach()
        hot = workspace.hot
        if hot is not None:
            hot.prepare_call(indices, kv_slots)
            hot.compute_call(q, k, v, out, kv_slots, scale_log2, k_scale, v_scale)
        else:
            workspace.hot = _Hot(
                workspace,
                q,
                k,
                v,
                indices,
                out,
                kv_slots,
                scale_log2,
                k_scale,
                v_scale,
            )
    return out
