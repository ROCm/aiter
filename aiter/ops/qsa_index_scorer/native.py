# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Native (pure torch) implementation of ``qsa_paged_mqa_logits`` -- contract v1.

Serves the valid calls aiter's existing kernels (Triton, FlyDSL page-16 prefill) cannot
take (e.g. CPU tensors) when no provider serves them. Semantics are those of ATOM's
scorer and of the frozen golden::

    visible[t]   = clamp(min((pos[t] + 1) // ratio, ctx[req(t)] // ratio), min=0)
                   (0 when req(t) is outside [0, requests))
    logits[t, c] = sum_h relu(q[t, h, :] . k[c]) / divisor   if c < visible[t] and the
                   page-table entry of column c is a valid page
                 = -inf                                         otherwise
    k[c]         = cache[page_table[req, c // P], c % P, 0, :]          (bf16)
                 = cache_e4m3[...] * scale[page_table[req, c // P], c % P, 0]  (fp8)
    row_starts   = 0 (if given)

Design (graph-capture safe from a prepared plan):

* :class:`NativePlan` is built per call (:func:`qsa_paged_mqa_logits_native`) and owns a
  workspace sized for one tile of ``tile_rows x tile_cols`` (bounded by
  :data:`TILE_ELEMS`, independent of the problem size). The tile loop is static
  (rows / columns are part of the class key).
* :meth:`NativePlan.run` reads no tensor value on the host, never synchronizes and
  allocates nothing when the caller supplies ``logits`` and ``visible_groups``
  (every intermediate is an ``out=`` / in-place op into the workspace), for
  contiguous caches. Per-row key gathers make it independent of the request layout
  (no ``nonzero``/per-request host loop as in the golden).
* Pure torch (no Triton / HIP): device-agnostic, also runs on CPU.

Numerics: fp32 throughout; per tile the head sum is accumulated in head order
(``relu(h0) + relu(h1) + ...``) and divided once, as the golden. The dot products
are batched GEMMs (``bmm``), so the accumulation order inside a dot product differs
from the golden's GEMM; agreement is well inside the contract tolerance
(``rtol = atol = 1e-4``, measured on CPU in
``op_tests/test_qsa_paged_mqa_logits_route_cpu.py``).
"""

from __future__ import annotations

import math

import torch

#: variant id reported by ``Decision.vid`` for the native path
NATIVE_VID = "native.torch.v1"
#: tile size in (row, column) elements; the workspace is ~0.85 KiB per element for
#: H = 4, D = 128 (about 54 MiB at the default), whatever the problem size
TILE_ELEMS = 1 << 16
#: widest column tile
TILE_COLS_MAX = 8192

_FP8 = getattr(torch, "float8_e4m3fn", None)
_INT_DTYPES = (torch.int32, torch.int64)
_Q_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


class NativeContractError(ValueError):
    """The call violates the op contract (also invalid for ATOM's scorer)."""


def _vector(name: str, t, n: int) -> None:
    if not isinstance(t, torch.Tensor) or t.ndim != 1 or t.dtype not in _INT_DTYPES:
        raise NativeContractError(
            f"{name} must be a one-dimensional int32/int64 tensor"
        )
    if int(t.shape[0]) != n:
        raise NativeContractError(f"{name} must contain {n} entries")
    if not t.is_contiguous():
        raise NativeContractError(f"{name} must be contiguous")


def cache_kind(cache: torch.Tensor, scale) -> str:
    """'bf16' (any unquantized float cache: bf16 / fp16 / fp32, read in q's dtype) or
    'fp8' (E4M3 + fp32 per-row scale); anything else raises."""
    if cache.dtype in _Q_DTYPES and scale is None:
        return "bf16"
    if _FP8 is not None and cache.dtype == _FP8:
        if scale is None:
            raise NativeContractError("an fp8 QSA cache needs its fp32 scale plane")
        if scale.dtype != torch.float32:
            raise NativeContractError(
                f"compressed_k_scale dtype {scale.dtype}: this op serves fp8 E4M3 with an "
                "fp32 per-row scale only (MXFP8 / E8M0 scales are a non-goal)"
            )
        return "fp8"
    if cache.dtype in _Q_DTYPES:
        raise NativeContractError("an unquantized QSA cache takes no scale plane")
    raise NativeContractError(
        f"unsupported QSA cache dtype {cache.dtype} (contract v1: bf16, or fp8 E4M3 + fp32 "
        "per-row scale; MXFP4 / MXFP8 are non-goals)"
    )


def validate(
    q,
    compressed_k_cache,
    page_table,
    token_to_request,
    query_positions,
    context_lens,
    compress_ratio=4,
    score_divisor=None,
    max_columns=None,
    row_starts=None,
    compressed_k_scale=None,
    logits=None,
    visible_groups=None,
) -> dict:
    """Contract check of one call (raises :class:`NativeContractError`); returns the
    class geometry ``{rows, heads, dim, kind, pages, page, requests, width, columns,
    divisor}``. Mirrors the checks of ATOM's scorer."""
    if (
        not isinstance(q, torch.Tensor)
        or q.ndim != 3
        or q.shape[1] <= 0
        or q.shape[2] <= 0
    ):
        raise NativeContractError("q must have shape [tokens, heads, head_dim]")
    if q.dtype not in _Q_DTYPES:
        raise NativeContractError(f"q dtype {q.dtype} (bf16 / fp16 / fp32)")
    cache, scale = compressed_k_cache, compressed_k_scale
    if not isinstance(cache, torch.Tensor):
        raise NativeContractError("compressed_k_cache must be a tensor")
    kind = cache_kind(cache, scale)
    rows, heads, dim = (int(x) for x in q.shape)
    if cache.ndim != 4 or cache.shape[2] != 1 or cache.shape[3] != dim:
        raise NativeContractError(
            "compressed_k_cache must have shape [pages, page_size, 1, head_dim]"
        )
    if kind == "bf16" and q.dtype != cache.dtype:
        raise NativeContractError("q and compressed_k_cache must have the same dtype")
    if kind == "fp8" and (
        scale.ndim != 3 or tuple(scale.shape) != tuple(cache.shape[:3])
    ):
        raise NativeContractError(
            "compressed_k_scale must page like compressed_k_cache "
            "([pages, page_size, 1] fp32)"
        )
    if (
        not isinstance(page_table, torch.Tensor)
        or page_table.ndim != 2
        or page_table.dtype not in _INT_DTYPES
    ):
        raise NativeContractError("page_table must be a two-dimensional integer tensor")
    _vector("token_to_request", token_to_request, rows)
    _vector("query_positions", query_positions, rows)
    _vector("context_lens", context_lens, int(page_table.shape[0]))
    if row_starts is not None:
        _vector("row_starts", row_starts, rows)
    ratio = int(compress_ratio)
    if ratio <= 0:
        raise NativeContractError("compress_ratio must be positive")
    divisor = math.sqrt(dim) if score_divisor is None else float(score_divisor)
    if not divisor > 0:
        raise NativeContractError("score_divisor must be positive")
    page = int(cache.shape[1])
    width = int(page_table.shape[1])
    columns = width * page
    if max_columns is not None:
        columns = max(1, min(columns, int(max_columns)))
    dev = q.device
    for nm, t in (
        ("compressed_k_cache", cache),
        ("compressed_k_scale", scale),
        ("page_table", page_table),
        ("token_to_request", token_to_request),
        ("query_positions", query_positions),
        ("context_lens", context_lens),
        ("row_starts", row_starts),
    ):
        if t is not None and t.device != dev:
            raise NativeContractError(f"{nm} on {t.device}, q on {dev}")
    if logits is not None and (
        tuple(logits.shape) != (rows, columns)
        or logits.dtype != torch.float32
        or logits.device != dev
    ):
        raise NativeContractError(f"logits must be fp32 {(rows, columns)} on {dev}")
    if visible_groups is not None and (
        visible_groups.numel() != rows
        or visible_groups.dtype != torch.int32
        or not visible_groups.is_contiguous()
        or visible_groups.device != dev
    ):
        raise NativeContractError(
            f"visible_groups must be contiguous int32 [{rows}] on {dev}"
        )
    return {
        "rows": rows,
        "heads": heads,
        "dim": dim,
        "kind": kind,
        "pages": int(cache.shape[0]),
        "page": page,
        "requests": int(page_table.shape[0]),
        "width": width,
        "columns": columns,
        "divisor": divisor,
        "ratio": ratio,
        "device": dev,
        "q_dtype": q.dtype,
        "cache_dtype": cache.dtype,
        "pt_dtype": page_table.dtype,
        "cl_dtype": context_lens.dtype,
    }


class NativePlan:
    """Prepared native launch of one call class (workspace + static tile loop)."""

    vid = NATIVE_VID

    def __init__(self, geo: dict, tile_elems: int = TILE_ELEMS):
        self.geo = dict(geo)
        g = self.geo
        self.rows, self.columns = g["rows"], g["columns"]
        self.heads, self.dim, self.page = g["heads"], g["dim"], g["page"]
        self.pages, self.requests, self.width = g["pages"], g["requests"], g["width"]
        self.divisor, self.ratio, self.kind = g["divisor"], g["ratio"], g["kind"]
        self.device = g["device"]
        cols = max(1, self.columns)
        self.tile_cols = min(cols, TILE_COLS_MAX, max(1, int(tile_elems)))
        self.tile_rows = max(
            1, min(max(self.rows, 1), int(tile_elems) // self.tile_cols)
        )
        self._alloc()

    # workspace ---------------------------------------------------------------
    def _alloc(self) -> None:
        dev, R, C, H, D = (
            self.device,
            self.tile_rows,
            self.tile_cols,
            self.heads,
            self.dim,
        )
        i64, f32 = torch.int64, torch.float32
        elem = torch.empty((), dtype=self.geo["cache_dtype"]).element_size()
        col = torch.arange(max(1, self.columns), dtype=i64, device=dev)
        self.col_idx = col  # [columns]
        self.col_page = torch.div(col, self.page, rounding_mode="floor")
        self.col_off = col - self.col_page * self.page
        self.req = torch.empty(R, dtype=i64, device=dev)
        self.req_ok = torch.empty(R, dtype=torch.bool, device=dev)
        self.tmp_b = torch.empty(R, dtype=torch.bool, device=dev)
        self.ctx = torch.empty(R, dtype=i64, device=dev)
        self.pos = torch.empty(R, dtype=i64, device=dev)
        self.vis = torch.empty(R, dtype=i64, device=dev)
        self.ctx_raw = torch.empty(R, dtype=self.geo["cl_dtype"], device=dev)
        self.idx = torch.empty(R * C, dtype=i64, device=dev)
        self.phys_raw = torch.empty(R * C, dtype=self.geo["pt_dtype"], device=dev)
        self.ok = torch.empty(R * C, dtype=torch.bool, device=dev)
        self.ok2 = torch.empty(R * C, dtype=torch.bool, device=dev)
        self.kraw = torch.empty((R * C, D * elem), dtype=torch.uint8, device=dev)
        self.kf = torch.empty((R * C, D), dtype=f32, device=dev)
        self.sc = (
            torch.empty(R * C, dtype=f32, device=dev) if self.kind == "fp8" else None
        )
        self.qf = torch.empty((R, H, D), dtype=f32, device=dev)
        self.hs = torch.empty(R * H * C, dtype=f32, device=dev)
        self.acc = torch.empty(R * C, dtype=f32, device=dev)

    def workspace_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in vars(self).values()
            if isinstance(t, torch.Tensor)
        )

    def matches(self, q, compressed_k_cache, page_table) -> bool:
        """O(1) class-key guard used by ``api.launch`` (no tensor reads)."""
        return (
            int(q.shape[0]) == self.rows
            and q.dtype == self.geo["q_dtype"]
            and tuple(q.shape[1:]) == (self.heads, self.dim)
            and compressed_k_cache.dtype == self.geo["cache_dtype"]
            and tuple(compressed_k_cache.shape[:2]) == (self.pages, self.page)
            and tuple(page_table.shape) == (self.requests, self.width)
            and page_table.dtype == self.geo["pt_dtype"]
            and q.device == self.device
        )

    # launch ------------------------------------------------------------------
    def run(
        self,
        q,
        compressed_k_cache,
        page_table,
        token_to_request,
        query_positions,
        context_lens,
        row_starts=None,
        compressed_k_scale=None,
        logits=None,
        visible_groups=None,
    ):
        rows, columns, dev = self.rows, self.columns, self.device
        if logits is None:
            logits = torch.empty((rows, columns), dtype=torch.float32, device=dev)
        vg = (
            torch.empty(rows, dtype=torch.int32, device=dev)
            if visible_groups is None
            else visible_groups.view(-1)
        )
        if row_starts is not None:
            row_starts.zero_()
        if rows == 0:
            return logits, vg
        if self.requests == 0:  # every row has an invalid request
            vg.zero_()
            logits.fill_(float("-inf"))
            return logits, vg
        R = self.tile_rows
        if self.width == 0:  # no page: nothing scored, visible as usual
            for r0 in range(0, rows, R):
                n = min(rows, r0 + R) - r0
                self._visible(
                    token_to_request[r0 : r0 + n],
                    query_positions[r0 : r0 + n],
                    context_lens,
                    n,
                )
                vg[r0 : r0 + n].copy_(self.vis[:n])
            logits.fill_(float("-inf"))
            return logits, vg
        cache = compressed_k_cache
        if not cache.is_contiguous():  # documented: allocates (not graph-safe)
            cache = cache.contiguous()
        kv = cache.view(torch.uint8).view(self.pages * self.page, -1)
        scale = None
        if self.kind == "fp8":
            scale = compressed_k_scale
            if not scale.is_contiguous():
                scale = scale.contiguous()
            scale = scale.view(-1)
        pt = page_table if page_table.is_contiguous() else page_table.contiguous()
        pt = pt.view(-1)
        C = self.tile_cols
        for r0 in range(0, rows, R):
            r1 = min(rows, r0 + R)
            n = r1 - r0
            self._visible(
                token_to_request[r0:r1], query_positions[r0:r1], context_lens, n
            )
            vg[r0:r1].copy_(self.vis[:n])
            self.qf[:n].copy_(q[r0:r1])
            for c0 in range(0, columns, C):
                c1 = min(columns, c0 + C)
                self._tile(kv, scale, pt, n, c0, c1, logits[r0:r1, c0:c1])
        return logits, vg

    def _visible(self, t2r, pos, ctx_lens, n: int) -> None:
        req, ok, tmp = self.req[:n], self.req_ok[:n], self.tmp_b[:n]
        req.copy_(t2r)
        torch.ge(req, 0, out=ok)
        torch.lt(req, self.requests, out=tmp)
        ok.logical_and_(tmp)
        req.clamp_(0, self.requests - 1)
        ctx, p, vis = self.ctx[:n], self.pos[:n], self.vis[:n]
        torch.index_select(ctx_lens, 0, req, out=self.ctx_raw[:n])
        ctx.copy_(self.ctx_raw[:n])
        ctx.div_(self.ratio, rounding_mode="floor")
        p.copy_(pos)
        p.add_(1).div_(self.ratio, rounding_mode="floor")
        torch.minimum(p, ctx, out=vis)
        vis.clamp_(min=0)
        vis.mul_(ok)  # invalid request -> 0 visible

    def _tile(self, kv, scale, pt, n: int, c0: int, c1: int, out) -> None:
        w = c1 - c0
        m = n * w
        # page-table slot of (row, column): req * width + column // page
        idx = self.idx[:m].view(n, w)
        idx.copy_(self.req[:n, None].expand(n, w))
        idx.mul_(self.width).add_(self.col_page[None, c0:c1])
        phys_raw = self.phys_raw[:m]
        torch.index_select(pt, 0, self.idx[:m], out=phys_raw)
        phys = idx  # reuse: int64 physical page
        phys.copy_(phys_raw.view(n, w))
        ok, ok2 = self.ok[:m].view(n, w), self.ok2[:m].view(n, w)
        torch.ge(phys, 0, out=ok)
        torch.lt(phys, self.pages, out=ok2)
        ok.logical_and_(ok2)
        phys.clamp_(0, self.pages - 1)
        phys.mul_(self.page).add_(self.col_off[None, c0:c1])  # cache row index
        kraw = self.kraw[:m]
        torch.index_select(kv, 0, self.idx[:m], out=kraw)
        kf = self.kf[:m]
        kf.copy_(kraw.view(self.geo["cache_dtype"]))
        if scale is not None:
            sc = self.sc[:m]
            torch.index_select(scale, 0, self.idx[:m], out=sc)
            kf.mul_(sc[:, None])
        hs = self.hs[: m * self.heads].view(n, self.heads, w)
        torch.bmm(self.qf[:n], kf.view(n, w, self.dim).transpose(1, 2), out=hs)
        hs.relu_()
        acc = self.acc[:m].view(n, w)
        acc.copy_(hs[:, 0])
        for h in range(1, self.heads):
            acc.add_(hs[:, h])
        acc.div_(self.divisor)
        # visible mask: column < visible[row], and a valid page
        torch.lt(self.col_idx[None, c0:c1], self.vis[:n, None], out=ok2)
        ok.logical_and_(ok2)
        ok.logical_not_()
        acc.masked_fill_(ok, float("-inf"))
        out.copy_(acc)


def qsa_paged_mqa_logits_native(
    q,
    compressed_k_cache,
    page_table,
    token_to_request,
    query_positions,
    context_lens,
    compress_ratio=4,
    score_divisor=None,
    max_columns=None,
    row_starts=None,
    allow_flydsl=False,
    compressed_k_scale=None,
    *,
    logits=None,
    visible_groups=None,
):
    """One-shot native call (validate + plan + run)."""
    geo = validate(
        q,
        compressed_k_cache,
        page_table,
        token_to_request,
        query_positions,
        context_lens,
        compress_ratio,
        score_divisor,
        max_columns,
        row_starts,
        compressed_k_scale,
        logits,
        visible_groups,
    )
    return NativePlan(geo).run(
        q,
        compressed_k_cache,
        page_table,
        token_to_request,
        query_positions,
        context_lens,
        row_starts,
        compressed_k_scale,
        logits,
        visible_groups,
    )
