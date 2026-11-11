# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""QSA sparse prefill on gfx942 and gfx950: BF16 D256 sparse attention over packed K/V.

Public entry for the ``sparse_prefill_*`` kernels in ``kernels/qsa/``. Each call
recovers the selected four-token blocks (FlyDSL K1-K3), routes every query tile to
the union kernel (queries of a tile share one compact block list with exact
per-query membership) or to the direct kernel (one query per wave over its own
blocks), and writes O in place. Plans and scratch are private. K/V may also be a
token pool read in place through a slot table, and on gfx950 their rows may be
strided (an interleaved K|V cache read in place).
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
from .kernels.qsa import sparse_prefill_prepare as prepare
from .kernels.qsa import sparse_prefill_union as union
from .kernels.qsa.sparse_prefill_common import (
    ARENA_GROWTH_PERCENT,
    SCRATCH_ALIGN,
    UNION_SCRATCH_LIMIT,
    WORKSPACE_CACHE,
    platform,
)

__all__ = ["flydsl_qsa_sparse_prefill", "qsa_sparse_prefill_problem"]

# Kernels use unsigned 32-bit buffer byte offsets; the 1 MiB margin keeps out-of-range
# sentinels plus their lane offsets from wrapping.
_MAX_SPAN = 2**32 - 2**20
# Kernels form a four-row K/V block's extent and offsets (up to 4 x a row's bytes) as
# 32-bit values: a row stride below 2**28 elements keeps them below 2**31.
_MAX_KV_STRIDE = 2**28


def _union_min_rows(group):
    config = platform()
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


def _span_bytes(rows, stride, heads):
    """Bytes from the first of rows BF16 [heads, 256] rows, stride elements apart, to the
    end of the last."""
    return ((rows - 1) * stride + heads * 256) * 2 if rows else 0


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
        self, q, k, v, indices, query_lens, prefix_lens, scale, kv_stride, kv_slots=None
    ):
        topk = (indices.shape[1] - 3) // 4
        q_lens = np.asarray(query_lens, dtype=np.int64)
        lengths = q_lens + np.asarray(prefix_lens, dtype=np.int64)
        sequence = np.repeat(np.arange(len(q_lens)), q_lens)
        positions = np.arange(len(sequence)) + (lengths - np.cumsum(q_lens))[sequence]
        slotted = kv_slots is not None
        # Past this pool span raw direct byte offsets no longer reach every row.
        wide = slotted and _span_bytes(k.shape[0], kv_stride, k.shape[1]) > _MAX_SPAN
        self.metadata = {
            "query_lens": query_lens,
            "prefix_lens": prefix_lens,
            "scale": scale,
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
        if q.shape[0] >= _union_min_rows(q.shape[1] // k.shape[1]):
            plan = prepare.allocate_plan(
                inputs=inputs, query_tile=32, scratch=False, slotted=slotted
            )
            # Union scratch grows with tiles x their block capacity.
            needed = sum(
                _aligned(math.prod(shape) * dtype.itemsize)
                for _, shape, dtype in prepare.scratch_specs(plan)
            )
            if needed <= UNION_SCRATCH_LIMIT:
                self.union = plan
        self.direct = direct.prepare(inputs=inputs, union=self.union, scratch=False)
        self.specs = [(self, "block_indices", (q.shape[0], topk), torch.int32)]
        if self.union is not None:
            self.specs += [
                (self.union, *spec) for spec in prepare.scratch_specs(self.union)
            ]
        self.specs += [
            (self.direct, *spec) for spec in direct.scratch_specs(self.direct)
        ]
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

    def bind(self, q, k, v, indices, kv_slots=None):
        self.attach()
        return SimpleNamespace(
            q=q,
            k=k,
            v=v,
            indices=indices,
            block_indices=self.block_indices,
            kv_slots=kv_slots,
            **self.metadata,
        )


class _Hot:
    """Replays one workspace binding: the FlyDSL launchers with prebuilt arguments.
    Building it performs the current call through the normal paths.
    """

    def __init__(self, workspace, q, k, v, indices, out, kv_slots=None):
        inputs = workspace.bind(q, k, v, indices, kv_slots)
        prepare.run(inputs=inputs, plan=workspace.union)
        self.prepare = prepare.launchers(inputs=inputs, plan=workspace.union)
        self.compute = []
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

    def compute_call(self, q, k, v, out, kv_slots):
        # K/V stay 3-D: FlyDSL passes int32 dims and K/V may exceed 2**31 elements.
        views = (q.view(-1), k, v, out.view(-1), kv_slots)
        for launch in self.compute:
            launch(*views)


_workspaces = OrderedDict()
_lock = threading.RLock()
_device = None


class _Rejected(NamedTuple):
    error: type[Exception]
    message: str


class _Checked(NamedTuple):
    scale: float
    query_lens: tuple
    prefix_lens: tuple
    stride: int
    topk: int


def _on_rocm_device(q):
    return q.device.type == "cuda" and torch.version.hip is not None


def _check_call(
    q, k, v, indices, query_lens, prefix_lens, softmax_scale, out, kv_slots
):
    """The call's operand rules in order: the first one it breaks as _Rejected (what the
    op raises and qsa_sparse_prefill_problem reports), else its normalized arguments.
    Rules on state (graph capture, one GPU per process) stay in the op."""
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        return _Rejected(ValueError, "Q/K/V must be BF16 [tokens, heads, 256]")
    if isinstance(softmax_scale, (torch.Tensor, bool)):
        return _Rejected(TypeError, "softmax_scale must be a host scalar or None")
    scale = 0.0625 if softmax_scale is None else float(softmax_scale)
    # The kernels multiply masked -inf scores by it: an FP32 zero would give NaN.
    scaled = scale * math.log2(math.e)
    if not math.isfinite(scaled) or not 2**-149 <= scaled <= (2 - 2**-23) * 2**127:
        return _Rejected(
            ValueError, "softmax_scale * log2(e) must be finite and positive in FP32"
        )
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
    for tensor in (q, k, v):
        if (
            tensor.dtype != torch.bfloat16
            or tensor.device != q.device
            or tensor.layout != torch.strided
            or tensor.shape[-1] != 256
        ):
            return _Rejected(ValueError, "Q/K/V must be BF16 [tokens, heads, 256]")
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
    # slot-table pool is addressed from its base instead.
    tokens = max(map(sum, zip(query_lens, prefix_lens)), default=0)
    span = q.numel() * q.element_size()
    if kv_slots is None:
        span = max(span, _span_bytes(tokens, stride, k.shape[1]))
    if span > _MAX_SPAN:
        return _Rejected(
            ValueError, "Q and each request's K/V require byte spans <= 2**32 - 2**20"
        )
    gfx = get_gfx_runtime()
    if gfx not in ("gfx942", "gfx950"):
        return _Rejected(ValueError, "QSA sparse attention requires gfx942 or gfx950")
    # Strided rows are validated on gfx950 only.
    if stride != k.shape[1] * 256 and gfx != "gfx950":
        return _Rejected(ValueError, "Strided K/V rows are supported on gfx950 only")
    topk = (indices.shape[-1] - 3) // 4 if indices.ndim else 0
    if (
        indices.shape != (q.shape[0], 4 * topk + 3)
        or topk % 256
        or not 256 <= topk <= 1024
        or indices.dtype != torch.int32
        or indices.device != q.device
        or not indices.is_contiguous()
        or indices.numel() * indices.element_size() >= 2**31
    ):
        return _Rejected(
            ValueError,
            "indices must be contiguous device int32 [M,4T+3], T = 256/512/768/1024, "
            "with byte span < 2**31",
        )
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
        # Strided K/V rows span more bytes than they hold.
        kv_bytes = _span_bytes(k.shape[0], stride, k.shape[1])
        sources = [(k.data_ptr(), kv_bytes), (v.data_ptr(), kv_bytes)] + [
            (source.data_ptr(), source.numel() * source.element_size())
            for source in (q, indices) + (() if kv_slots is None else (kv_slots,))
        ]
        for pointer, size in sources:
            if begin < pointer + size and pointer < end:
                return _Rejected(
                    ValueError, "out must not overlap Q/K/V, indices or kv_slots"
                )
    return _Checked(scale, query_lens, prefix_lens, stride, topk)


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
):
    """The first operand rule this call breaks, as the message flydsl_qsa_sparse_prefill
    would raise it, or None when the operands are accepted. Takes the op's arguments;
    states of the process (graph capture, one GPU per process) are not operand rules,
    so a call accepted here can still raise them."""
    checked = _check_call(
        q, k, v, indices, query_lens, prefix_lens, softmax_scale, out, kv_slots
    )
    return checked.message if isinstance(checked, _Rejected) else None


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
    token, read in place); each stride compiles its own kernels. Hot calls rebuild
    the selection: per-layout plans share one private per-stream scratch arena. out
    is optional and must not overlap Q/K/V, indices or kv_slots. Warm the same
    stream and layout before graph capture. Captured workspaces stay pinned for graph
    pointer lifetime and share their stream's scratch: do not replay them
    concurrently with each other or with that stream's calls. Each kernel variant
    compiles on its first use, a few seconds with empty FlyDSL caches; none is
    prebuilt (PREBUILD_KERNELS / AOT do not cover QSA). One GPU per process (TP uses
    separate workers).
    """
    global _device
    checked = _check_call(
        q, k, v, indices, query_lens, prefix_lens, softmax_scale, out, kv_slots
    )
    if isinstance(checked, _Rejected):
        raise checked.error(checked.message)
    scale, query_lens, prefix_lens, stride, topk = checked
    if out is None:
        out = torch.empty_like(q)
    if q.shape[0] == 0:
        return out
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
            scale,
            topk,
            # Pool rows: they set the raw direct extent and the WIDE path.
            None if kv_slots is None else k.shape[0],
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
                q, k, v, indices, query_lens, prefix_lens, scale, stride, kv_slots
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
            hot.compute_call(q, k, v, out, kv_slots)
        else:
            workspace.hot = _Hot(workspace, q, k, v, indices, out, kv_slots)
    return out
