# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""QSA sparse prefill on gfx942: BF16 D256 sparse attention over packed K/V.

Public entry for the ``sparse_prefill_*`` kernels in ``kernels/qsa/``. Each call
recovers the selected four-token blocks (FlyDSL K1-K3), routes every query tile to
the union kernel (queries of a tile share one compact block list with exact
per-query membership) or to the direct kernel (one query per wave over its own
blocks), and writes O in place. Plans and scratch are private. K/V may also be a
token pool read in place through a slot table, and FP8 (E4M3FNUZ) with FP32 scales.
"""

import math
import threading
import weakref
from collections import OrderedDict
from types import SimpleNamespace

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
]

# The FP8 kernels run E4M3FNUZ, gfx942's FP8 (aiter's dtypes.fp8 there).
FP8 = torch.float8_e4m3fnuz

# Kernels use unsigned 32-bit buffer byte offsets; the 1 MiB margin keeps out-of-range
# sentinels plus their lane offsets from wrapping.
_MAX_SPAN = 2**32 - 2**20
# Slot-table pools past this span take raw direct's 64-bit loads (WIDE).
_WIDE_SPAN = _MAX_SPAN

# Selection widths 4T + 3 for the per-row block budgets T the kernels take.
SELECTION_WIDTHS = tuple(4 * topk + 3 for topk in (256, 512, 768, 1024))
# The selection is addressed at signed 32-bit byte offsets.
_MAX_SELECTION_BYTES = 2**31 - 1


def flydsl_qsa_sparse_prefill_available() -> bool:
    """Whether ``flydsl_qsa_sparse_prefill`` runs on the live GPU (gfx942 only)."""
    return get_gfx_runtime() == "gfx942"


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
        self, q, k, v, indices, query_lens, prefix_lens, kv_slots, invariant, fp8
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
        wide = slotted and k.numel() * k.element_size() > _WIDE_SPAN
        self.metadata = {
            "query_lens": query_lens,
            "prefix_lens": prefix_lens,
            "topk": topk,
            "query_positions": prepare.upload(positions, q.device),
            "max_seqlen_k": int(lengths.max(initial=0)),
            "wide": wide,
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
            self.direct = direct.prepare(inputs=inputs, union=self.union, scratch=False)
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


def _fp8_layout(k, v, k_scale, v_scale):
    """None for BF16 K/V; "tensor" or "token" for FP8 K/V with valid scales."""
    if k.dtype != FP8 and v.dtype != FP8:
        if k_scale is not None or v_scale is not None:
            raise ValueError("k_scale/v_scale apply only to FP8 K/V")
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
            raise ValueError(
                "FP8 K/V require contiguous FP32 k_scale/v_scale on their device, of "
                f"shape [1] or {list(per_token)} (K's [N, HK], or [P, HK] for a pool)"
            )
    if k_scale.shape != v_scale.shape:
        raise ValueError("k_scale and v_scale must both be per tensor or per token")
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
    validated. Hot calls rebuild the selection: per-layout plans share one private
    per-stream scratch arena. out is optional and must not overlap Q/K/V, indices or
    kv_slots. softmax_scale is a kernel argument (FP32 softmax_scale * log2(e)): a new
    scale compiles nothing and reuses the layout's workspace, also under graph capture.
    Warm the same stream and layout before graph capture. Captured
    workspaces stay pinned for graph pointer lifetime and share their stream's
    scratch: do not replay them concurrently with each other or with that stream's
    calls. Each kernel variant compiles on its first use: a few seconds with empty
    FlyDSL caches. One GPU per process (TP uses separate workers).

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
    (a call-wide budget) does not change the bits. Tuning targets MI308X (gfx942,
    80 CUs) only: other gfx942 parts log a warning and reuse the 80-CU MI308X
    settings; plans and routes follow the CU count, performance may not be optimal.

    K/V may also be FP8 (gfx942's E4M3FNUZ, torch.float8_e4m3fnuz: aiter's dtypes.fp8)
    with FP32 dequantization scales k_scale / v_scale, both per token and KV head
    ([N,HK], with kv_slots [P,HK]) or both per tensor ([1]); K/V values are K8 * k_scale
    and V8 * v_scale. compute="fp8" (default) runs FP8 MFMA kernels: Q is quantized per
    (query, head) row with scale max(amax, 2**-118) / 240 and the probabilities in FP8
    (relative output error of a few percent on random data), and per-token V scales are
    clamped to [2**-60, 2**60] (they join the probabilities); as for BF16 K/V, direct
    rows read the context repacked per call while it fits a budget, else the raw K/V.
    Its union plans may split a short prefill's longest tasks into two key ranges merged
    at the end, and keep only a core of an H12/H6 union tile, the rows' other blocks
    running direct and merging by log-sum-exp; whether a task splits depends on the whole
    call, so batch_invariant=True does not split. compute="bf16" dequantizes the
    request tokens to BF16 (each value the FP32 product rounded once) and runs the BF16
    kernels (BF16 accuracy, temporary BF16 K/V of the packed context). Workspaces are
    kept per K/V dtype and scale layout too; the scale tensors are read on every call,
    also under graph replay.
    """
    global _device
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("Q/K/V must be contiguous BF16 [tokens, heads, 256]")
    if compute not in ("fp8", "bf16"):
        raise ValueError('compute must be "fp8" or "bf16"')
    fp8 = _fp8_layout(k, v, k_scale, v_scale)
    if isinstance(softmax_scale, (torch.Tensor, bool)):
        raise TypeError("softmax_scale must be a host scalar or None")
    scale = 0.0625 if softmax_scale is None else float(softmax_scale)
    # The kernels multiply masked -inf scores by it: an FP32 zero would give NaN.
    scaled = scale * math.log2(math.e)
    if not math.isfinite(scaled) or not 2**-149 <= scaled <= (2 - 2**-23) * 2**127:
        raise ValueError("softmax_scale * log2(e) must be finite and positive in FP32")
    scale_log2 = float(np.float32(scaled))
    if not isinstance(batch_invariant, bool):
        raise TypeError("batch_invariant must be a bool")
    if kv_slots is not None and (
        not isinstance(kv_slots, torch.Tensor)
        or kv_slots.dtype != torch.int32
        or kv_slots.ndim != 1
        or kv_slots.device != q.device
        or not kv_slots.is_contiguous()
    ):
        raise ValueError("kv_slots must be a contiguous device int32 [N] tensor")
    # The kernels read the slot table at unsigned 32-bit buffer offsets and upload
    # each request's start in it as int32.
    if kv_slots is not None and kv_slots.numel() * kv_slots.element_size() > _MAX_SPAN:
        raise ValueError("kv_slots requires a byte span <= 2**32 - 2**20")
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
        raise ValueError("Host query/prefix lengths must describe the packed Q/K/V")
    if q.device.type != "cuda" or torch.version.hip is None:
        raise ValueError("QSA sparse attention requires ROCm gfx942")
    kv_dtype = torch.bfloat16 if fp8 is None else FP8
    for tensor, dtype in ((q, torch.bfloat16), (k, kv_dtype), (v, kv_dtype)):
        if (
            tensor.dtype != dtype
            or tensor.device != q.device
            or tensor.layout != torch.strided
            or not tensor.is_contiguous()
            or tensor.shape[-1] != 256
        ):
            raise ValueError(
                "Q/K/V must be contiguous BF16 [tokens, heads, 256] (K/V may be FP8)"
            )
        if tensor.data_ptr() % 16:
            raise ValueError("Q/K/V require 16-byte alignment")
    if (
        k.shape != v.shape
        or q.shape[1] <= 0
        or k.shape[1] <= 0
        or q.shape[1] % k.shape[1]
    ):
        raise ValueError(
            "K/V must match and Q heads must be a positive multiple of KV heads"
        )
    if q.shape[1] // k.shape[1] > 16:
        raise ValueError("At most 16 Q heads per KV head fit the direct fallback wave")
    # K/V are addressed per request, so only each request's K (or V) span is bounded,
    # not the packed total: about 8M context tokens at HK1, 4M at HK2. A slot-table
    # pool is addressed from its base instead.
    tokens = max(map(sum, zip(query_lens, prefix_lens)), default=0)
    span = q.numel() if kv_slots is not None else max(q.numel(), tokens * k.stride(0))
    if span * q.element_size() > _MAX_SPAN:
        raise ValueError("Q and each request's K/V require byte spans <= 2**32 - 2**20")
    if not flydsl_qsa_sparse_prefill_available():
        raise ValueError("QSA sparse attention requires gfx942")
    if (
        indices.ndim != 2
        or indices.shape[0] != q.shape[0]
        or not flydsl_qsa_sparse_prefill_supports_selection(*indices.shape)
        or indices.dtype != torch.int32
        or indices.device != q.device
        or not indices.is_contiguous()
    ):
        raise ValueError(
            "indices must be contiguous device int32 [M,4T+3], T = 256/512/768/1024, "
            "with byte span < 2**31"
        )
    topk = (indices.shape[1] - 3) // 4
    if out is None:
        out = torch.empty_like(q)
    if (
        out.shape != q.shape
        or out.dtype != q.dtype
        or out.device != q.device
        or not out.is_contiguous()
        or out.data_ptr() % 16
    ):
        raise ValueError("out must be aligned, contiguous and match Q")
    begin, end = out.data_ptr(), out.data_ptr() + out.numel() * out.element_size()
    sources = (q, k, v, indices) + (() if kv_slots is None else (kv_slots,))
    if fp8 is not None:
        sources += (k_scale, v_scale)
    for source in sources:
        if (
            begin < source.data_ptr() + source.numel() * source.element_size()
            and source.data_ptr() < end
        ):
            raise ValueError("out must not overlap Q/K/V, indices, kv_slots or scales")
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
