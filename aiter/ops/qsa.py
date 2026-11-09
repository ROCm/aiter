# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""QSA top-level dispatch: selects the best backend (triton, FlyDSL, etc.)
for each QSA operator based on the input shape and available backends.

Consumers should import from here rather than from the backend-specific
modules directly.
"""

import functools
from collections.abc import Sequence
from dataclasses import replace as _dc_replace

import torch

from aiter.ops.qsa_index_scorer import dispatch as _D
from aiter.ops.topk import top_k_per_row_prefill
from aiter.ops.triton.attention.qsa import (
    FP4_GROUP,
    FP8_DTYPE,
    FP8_MAX,
    qsa_cache_row_layout,
    qsa_expand_block_indices,
    qsa_sparse_paged_gqa,
    qsa_store_quantized_rows,
)
from aiter.ops.triton.attention.qsa import (
    qsa_paged_mqa_logits as _triton_qsa_paged_mqa_logits,
)

try:
    from aiter.ops.flydsl.kernels.mqa_logits.qsa_flydsl_layout import (
        PAGE_SIZE as _FLYDSL_PAGE_SIZE,
    )
    from aiter.ops.flydsl.kernels.mqa_logits.qsa_flydsl_layout import (
        qsa_paged_mqa_logits_flydsl,
    )

    _HAS_FLYDSL = True
except ImportError:
    _HAS_FLYDSL = False

try:
    from aiter.ops.flydsl.qsa_sparse_prefill import flydsl_qsa_sparse_prefill

    _HAS_FLYDSL_PREFILL = True
except ImportError:
    _HAS_FLYDSL_PREFILL = False

DEFAULT_LOGITS_WORKSPACE_BYTES = 128 * 1024 * 1024
DEFAULT_LOGITS_WORKSPACE_BYTES = 128 * 1024 * 1024

# ---------------------------------------------------------------------------
# qsa_paged_mqa_logits: neutral op (contract v1) + fused-path policy
# ---------------------------------------------------------------------------
# Route-ready: the op is routed through aiter.plugin when that interface exists
# (provider stack #7465/#7484); until then the decorator is a no-op.
try:
    from aiter.plugin import route as _route
except ImportError:  # aiter.plugin not in this tree yet

    def _route(*, op: str, version: int = 1, supersedes=None):
        def wrap(func):
            func.__aiter_route__ = (op, version)
            return func

        return wrap

    ROUTED = False
else:
    ROUTED = True

OP_NAME = _D.OP_NAME
OP_CONTRACT_VERSION = _D.OP_CONTRACT_VERSION
API_VERSION = _D.API_VERSION
Decision = _D.Decision


def _on_gpu(t) -> bool:
    return t.is_cuda


#: Row limit of the stacked Triton decode scorer (the Triton wrapper's own limit).
STACKED_DECODE_MAX_ROWS = 256


def _stacked_decode_auto(q, compressed_k_cache, page_table) -> bool:
    """``stacked_decode=None`` (the default): aiter decides. True for the calls ATOM's
    own scorer stacks heads for (``ATOM_QSA_STACKED_DECODE_SCORER``, on by default):
    at most 256 rows, rows contiguous in the head dim (q, cache, page table), a
    power-of-two head count and a power-of-two ``head_dim >= 16``. The Triton wrapper
    then also needs the arch's ``stacked_decode`` tile (gfx942 / gfx950 / gfx1250 have
    one) and ``BMT * heads >= 16``, as ATOM's scorer does; otherwise it scores the call
    as with ``stacked_decode=False``."""
    heads, head_dim = q.shape[1], q.shape[2]
    return (
        q.shape[0] <= STACKED_DECODE_MAX_ROWS
        and q.stride(2) == 1
        and compressed_k_cache.stride(3) == 1
        and page_table.stride(1) == 1
        and heads & (heads - 1) == 0
        and head_dim & (head_dim - 1) == 0
        and head_dim >= 16
    )


#: Arches the FlyDSL page-16 prefill scorer (``qsa_flydsl_layout``) may be chosen on.
#: It is an MFMA (bf16 16x16x16) kernel with wave64 shuffles, i.e. CDNA (gfx9) only: on
#: gfx1250 LLVM aborts the whole process while compiling it (SIGABRT, "LLVM ERROR:
#: Cannot select: intrinsic llvm.amdgcn.permlane32.swap"; reproduced compile-only with
#: FlyDSL 0.3.4.1 and 0.4.0), so elsewhere the Triton scorer serves those calls.
FLYDSL_PAGE16_ARCHS = ("gfx9",)


@functools.cache
def _cuda_arch(index: int) -> str:
    return torch.cuda.get_device_properties(index).gcnArchName.split(":")[0]


def _device_arch(device) -> str:
    """``gfx...`` name of a GPU device (cached per device index)."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _cuda_arch(index)


def _flydsl_page16_arch_ok(q) -> bool:
    try:
        return _device_arch(q.device).startswith(FLYDSL_PAGE16_ARCHS)
    except Exception:  # noqa: BLE001 -- unknown arch: not the FlyDSL kernel
        return False


def _flydsl_page16(q, compressed_k_cache, page_table, allow_flydsl, compressed_k_scale):
    """main's FlyDSL page-16 prefill predicate, plus the arch gate above."""
    return (
        allow_flydsl
        and _HAS_FLYDSL
        and compressed_k_scale is None
        and q.shape[0] > 256
        and q.stride(2) == 1
        and compressed_k_cache.stride(3) == 1
        and page_table.stride(1) == 1
        and q.shape[2] >= 16
        and compressed_k_cache.shape[1] == _FLYDSL_PAGE_SIZE
        and _flydsl_page16_arch_ok(q)
    )


def _existing_route(
    q, compressed_k_cache, page_table, allow_flydsl, compressed_k_scale
):
    """Which of aiter's existing native kernels takes this call (``None``: neither --
    the pure-torch implementation serves a valid call, e.g. on a CPU device)."""
    if not _on_gpu(q):
        return None
    if _flydsl_page16(
        q, compressed_k_cache, page_table, allow_flydsl, compressed_k_scale
    ):
        return _D.VID_FLYDSL_PAGE16
    dt = compressed_k_cache.dtype
    if dt == torch.uint8 or (FP8_MAX is not None and dt == FP8_DTYPE):
        return _D.VID_TRITON if compressed_k_scale is not None else None
    if dt.is_floating_point and dt.itemsize > 1:
        return _D.VID_TRITON
    return None


def _existing_native(
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
    stacked_decode=None,
    logits=None,
    visible_groups=None,
):
    """aiter's native dispatch as before the fused path: the FlyDSL page-16 prefill
    scorer for large ``allow_flydsl`` batches, otherwise Triton (bf16/fp16/fp32, fp8 +
    fp32 scale, MXFP4). ``stacked_decode=None`` resolves with
    :func:`_stacked_decode_auto`; True / False are passed to Triton as given.
    Caller-supplied ``logits`` / ``visible_groups`` are filled."""
    if _flydsl_page16(
        q, compressed_k_cache, page_table, allow_flydsl, compressed_k_scale
    ):
        divisor = q.shape[2] ** 0.5 if score_divisor is None else score_divisor
        columns = page_table.shape[1] * compressed_k_cache.shape[1]
        if max_columns is not None:
            columns = max(1, min(columns, int(max_columns)))
        if logits is None:
            logits = torch.empty(
                (q.shape[0], columns), dtype=torch.float32, device=q.device
            )
        if visible_groups is None:
            visible_groups = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
        qsa_paged_mqa_logits_flydsl(
            q,
            compressed_k_cache,
            page_table,
            token_to_request,
            query_positions,
            context_lens,
            compress_ratio,
            float(divisor),
            logits,
            visible_groups,
            row_starts,
        )
        return logits, visible_groups
    if stacked_decode is None:
        stacked_decode = _stacked_decode_auto(q, compressed_k_cache, page_table)
    out_logits, out_visible = _triton_qsa_paged_mqa_logits(
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
        compressed_k_scale=compressed_k_scale,
        stacked_decode=stacked_decode,
    )
    if logits is not None:
        logits.copy_(out_logits)
        out_logits = logits
    if visible_groups is not None:
        visible_groups.view(-1).copy_(out_visible)
        out_visible = visible_groups
    return out_logits, out_visible


@_route(op="qsa_paged_mqa_logits", version=1)
def qsa_paged_mqa_logits(
    q: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    compress_ratio: int = 4,
    score_divisor: float | None = None,
    max_columns: int | None = None,
    row_starts: torch.Tensor | None = None,
    allow_flydsl: bool = False,
    compressed_k_scale: torch.Tensor | None = None,
    stacked_decode: bool | None = None,
    *,
    logits: torch.Tensor | None = None,
    visible_groups: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score compressed key groups (QSA indexer); op contract
    ``qsa_paged_mqa_logits`` v1 (``docs/qsa_index_scorer.md``).

    ``q`` ``[rows, heads, head_dim]``; ``compressed_k_cache`` ``[pages, page_size, 1,
    head_dim]``: unquantized (Q's dtype), FP8 E4M3 + ``compressed_k_scale`` fp32
    ``[pages, page_size, 1]``, or MXFP4 + E8M0 scale (Triton). ``page_table``
    ``[requests, width]``; ``token_to_request`` / ``query_positions`` ``[rows]``;
    ``context_lens`` ``[requests]`` (int32/int64). Returns ``(logits fp32 [rows,
    columns], visible_groups int32 [rows])``, ``columns = max(1, min(width *
    page_size, max_columns))``; ``visible = clamp(min((pos + 1) // ratio, ctx //
    ratio), 0)``; ``logits[t, c] = sum_h relu(q[t, h] . k[c]) / divisor`` for ``c <
    visible`` on a valid page, else ``-inf``; ``row_starts`` (if given) zero-filled.
    The 12 leading parameters are the frozen ATOM scorer's; ``logits`` /
    ``visible_groups`` are optional output buffers. ``stacked_decode`` is a Triton
    scheduling hint (results agree within tolerance either way): ``None`` (default)
    lets aiter decide -- decode-sized calls (<= 256 rows, contiguous, power-of-two
    heads / head_dim) take the stacked decode scorer, as ATOM's own scorer does by
    default; ``True`` / ``False`` are honoured as given.

    Implementation (a dispatch decision, per call class, decided once on the host):
    the fused FlyDSL gfx1250 kernels when a profile is active (:func:`configure`,
    ``$AITER_QSA_PROFILE``; kill switch ``$AITER_QSA_FUSED=0``) and covers the class;
    else aiter's existing native kernels (FlyDSL page-16 prefill with ``allow_flydsl``,
    Triton); else a pure-torch implementation (e.g. CPU tensors). Default: no profile,
    i.e. the existing native dispatch. :func:`explain` names the path of a call.

    Raises ``ValueError`` only for calls that violate the contract.
    """
    if _D._ACTIVE is None and isinstance(q, torch.Tensor):
        try:
            route = _existing_route(
                q, compressed_k_cache, page_table, allow_flydsl, compressed_k_scale
            )
        except Exception:  # noqa: BLE001 -- the dispatch core decides
            route = None
        if route is not None:  # default (fused off): exactly the existing dispatch
            return _existing_native(
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
                allow_flydsl,
                compressed_k_scale,
                stacked_decode,
                logits,
                visible_groups,
            )
    return _D.call(
        _existing_native,
        _existing_route,
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
        allow_flydsl,
        compressed_k_scale,
        stacked_decode,
        logits,
        visible_groups,
    )


def configure(*, profile: str | None, source: str) -> None:
    """Enable the fused ``qsa_paged_mqa_logits`` path with a named profile (``None``:
    off, the default). Process-global, idempotent; ``source`` names the caller (e.g.
    ``"atom:<config>"``). ``$AITER_QSA_PROFILE`` (operator) overrides ``profile``
    (``off`` disables); ``$AITER_QSA_FUSED=0`` is the kill switch. Call before model
    load / warm-up / graph capture."""
    _D.configure(profile=profile, source=source)


def prewarm(
    *,
    page_sizes,
    cache_dtypes,
    pos_dtypes=(torch.int64,),
    compressed_k_cache: torch.Tensor | None = None,
    compressed_k_scale: torch.Tensor | None = None,
    device=None,
) -> list[str]:
    """Load (AOT cache hit) or compile the fused variants the active profile can select
    for these index caches: ``page_sizes`` (4 / 8 / 16), ``cache_dtypes`` (``"bf16"`` /
    ``"fp8"`` or torch dtypes), ``pos_dtypes``; a bound ``compressed_k_cache`` (+ scale)
    narrows to what it can select. Returns the warmed variant tags (``[]`` when the
    fused path is off). Call after KV allocation, before graph capture."""
    return _D.prewarm(
        page_sizes=page_sizes,
        cache_dtypes=cache_dtypes,
        pos_dtypes=pos_dtypes,
        compressed_k_cache=compressed_k_cache,
        compressed_k_scale=compressed_k_scale,
        device=device,
    )


def explain(
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
    stacked_decode=None,
    *,
    logits=None,
    visible_groups=None,
) -> Decision:
    """``Decision(path, reason, vid, ...)`` for this call's class without running it
    (``path`` in ``fused`` / ``native-triton`` / ``native-torch``; ``invalid`` for a
    contract violation). Decides and caches the class like the first call would. A
    ``native.triton.v1`` decision's ``detail`` ends with the ``stacked_decode`` hint
    passed to Triton (``stacked_decode=auto:True`` / ``auto:False`` / ``True`` /
    ``False``)."""
    d = None
    if _D._ACTIVE is None and isinstance(q, torch.Tensor):
        try:
            vid = _existing_route(
                q, compressed_k_cache, page_table, allow_flydsl, compressed_k_scale
            )
        except Exception:  # noqa: BLE001
            vid = None
        if vid is not None:
            reason = "kill_switch" if _D._ORIGIN.startswith("kill") else "disabled"
            d = Decision(_D.PATH_NATIVE_TRITON, reason, vid, _D._ORIGIN)
    if d is None:
        d = _explain_decide(
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
            allow_flydsl,
            compressed_k_scale,
            stacked_decode,
            logits,
            visible_groups,
        )
    if d.path == _D.PATH_NATIVE_TRITON and d.vid == _D.VID_TRITON:
        if stacked_decode is None:
            try:
                auto = _stacked_decode_auto(q, compressed_k_cache, page_table)
                hint = f"auto:{auto}"
            except Exception:  # noqa: BLE001
                hint = "auto:?"
        else:
            hint = str(bool(stacked_decode))
        detail = (
            f"{d.detail}; stacked_decode={hint}"
            if d.detail
            else f"stacked_decode={hint}"
        )
        d = _dc_replace(d, detail=detail)
    return d


def _explain_decide(*args) -> Decision:
    return _D.explain(_existing_native, _existing_route, *args)


def status() -> dict:
    """Fused-path state: enabled, profile (+ origin: configure source / env / kill
    switch), kernel and profile sha256, op / API versions, decided classes per path,
    route (``aiter.plugin`` or shim)."""
    out = _D.status()
    out["routed"] = ROUTED
    return out


def qsa_select_paged_tokens(
    q: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    token_topk: int,
    compress_ratio: int = 4,
    out: torch.Tensor | None = None,
    logits_workspace_bytes: int = DEFAULT_LOGITS_WORKSPACE_BYTES,
    max_seq_len: int | None = None,
    allow_flydsl: bool = False,
    compressed_k_scale: torch.Tensor | None = None,
    stacked_decode: bool = False,
) -> torch.Tensor:
    """Score, select and expand in one call."""
    if compress_ratio <= 0 or token_topk <= 0 or token_topk % compress_ratio:
        raise ValueError("token_topk must be positive and divisible by compress_ratio")
    rows = q.shape[0]

    columns = page_table.shape[1] * compressed_k_cache.shape[1]
    if columns < 1:
        raise ValueError("compressed paged-cache capacity must be positive")
    block_topk = min(token_topk // compress_ratio, columns)
    if max_seq_len is not None:
        reachable = (int(max_seq_len) + compress_ratio - 1) // compress_ratio
        bucket = 1 << (max(reachable, 1) - 1).bit_length()
        columns = max(1, min(columns, bucket))
        block_topk = max(1, min(block_topk, bucket))
    output_width = block_topk * compress_ratio + compress_ratio - 1

    if out is None:
        out = torch.empty((rows, output_width), dtype=torch.int32, device=q.device)
    elif out.shape[0] != rows or out.shape[1] < output_width:
        raise ValueError("out has an invalid shape")
    else:
        out = out[:, :output_width]
    if rows == 0:
        return out

    rows_per_chunk = max(1, logits_workspace_bytes // max(columns * 4, 1))
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        row_slice = slice(row_start, row_end)
        row_starts = torch.empty(
            row_end - row_start, dtype=torch.int32, device=q.device
        )
        logits, visible_groups = qsa_paged_mqa_logits(
            q[row_slice],
            compressed_k_cache,
            page_table,
            token_to_request[row_slice],
            query_positions[row_slice],
            context_lens,
            compress_ratio,
            max_columns=columns,
            row_starts=row_starts,
            allow_flydsl=allow_flydsl,
            compressed_k_scale=compressed_k_scale,
            stacked_decode=stacked_decode,
        )
        selected_groups = torch.empty(
            (row_end - row_start, block_topk), dtype=torch.int32, device=q.device
        )
        top_k_per_row_prefill(
            logits,
            row_starts,
            visible_groups,
            selected_groups,
            None,
            logits.shape[0],
            logits.stride(0),
            logits.stride(1),
            block_topk,
            stable=True,
        )
        qsa_expand_block_indices(
            selected_groups,
            query_positions[row_slice],
            context_lens,
            token_to_request[row_slice],
            compress_ratio,
            token_topk,
            out[row_slice],
            output_width=output_width,
        )
    return out


def qsa_sparse_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    *,
    query_lens: Sequence[int] | None = None,
    prefix_lens: Sequence[int] | None = None,
    softmax_scale: float | None = None,
    out: torch.Tensor | None = None,
    kv_slots: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse prefill/extend attention over packed K/V, or a token pool read through
    kv_slots; FlyDSL on gfx942 only.

    See ``aiter.ops.flydsl.qsa_sparse_prefill.flydsl_qsa_sparse_prefill``.
    """
    if not _HAS_FLYDSL_PREFILL:
        raise RuntimeError("QSA sparse prefill requires FlyDSL")
    return flydsl_qsa_sparse_prefill(
        q,
        k,
        v,
        indices,
        query_lens=query_lens,
        prefix_lens=prefix_lens,
        softmax_scale=softmax_scale,
        out=out,
        kv_slots=kv_slots,
    )


__all__ = [
    "DEFAULT_LOGITS_WORKSPACE_BYTES",
    "FP4_GROUP",
    "FP8_DTYPE",
    "FP8_MAX",
    "Decision",
    "configure",
    "explain",
    "prewarm",
    "qsa_cache_row_layout",
    "qsa_expand_block_indices",
    "qsa_paged_mqa_logits",
    "qsa_select_paged_tokens",
    "qsa_sparse_paged_gqa",
    "qsa_sparse_prefill",
    "qsa_store_quantized_rows",
    "status",
]
