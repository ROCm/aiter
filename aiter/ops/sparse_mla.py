# SPDX-License-Identifier: MIT
"""HIP sparse MLA with FP8 latent K/V and BF16 output on gfx950."""

import torch

from aiter.jit.core import compile_ops
from aiter.jit.utils.chip_info import get_gfx_runtime
from csrc.cpp_itfs.torch_utils import direct_register_custom_op


def _split_count(queries: int) -> int:
    if queries <= 6:
        return 17
    if queries <= 12:
        return 9
    if queries <= 21:
        return 6
    if queries <= 23:
        return 4
    if queries <= 42:
        return 6
    if queries <= 64:
        return 4
    if queries <= 128:
        return 2
    return 1


def allocate_sparse_mla_fp8_workspace(
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate opaque scratch storage for ``sparse_mla_fp8_fwd``.

    Reuse this pair for the same Q shape and device. No initialization or GPU
    kernel is required. The layout is private and may vary with query count.
    """
    if q.ndim != 3 or q.shape[1:] != (64, 512) or not 0 <= q.shape[0] <= 16384:
        raise ValueError("expected q=[Q,64,512], 0 <= Q <= 16384")
    queries = q.shape[0]
    splits = _split_count(queries)
    rows = queries * splits * 64 if splits > 1 else 0
    return (
        torch.empty((rows * 512,), dtype=torch.bfloat16, device=q.device),
        torch.empty((rows,), dtype=torch.float32, device=q.device),
    )


@compile_ops(
    "module_sparse_mla_fp8", fc_name="sparse_mla_fp8_fwd_out", ffi_type="ctypes"
)
def _jit_sparse_mla_fp8_fwd_out(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    kv_indices: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor | None,
    partial_o: torch.Tensor,
    partial_lse: torch.Tensor,
    softmax_scale: float,
    q_scale: float,
    kv_scale: float,
) -> None: ...


def _sparse_mla_fp8_fwd_out(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    kv_indices: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor | None,
    partial_o: torch.Tensor,
    partial_lse: torch.Tensor,
    softmax_scale: float,
    q_scale: float,
    kv_scale: float,
) -> None:
    # Runtime discovery stays inside the opaque operation, outside Dynamo.
    if get_gfx_runtime() != "gfx950":
        raise NotImplementedError("sparse_mla_fp8_fwd requires gfx950")
    if q.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError("q must be BF16 or float8_e4m3fn")
    if kv_buffer.dtype != torch.float8_e4m3fn:
        raise ValueError("kv_buffer must be float8_e4m3fn")
    _jit_sparse_mla_fp8_fwd_out(
        q,
        kv_buffer,
        kv_indices,
        out,
        lse,
        partial_o,
        partial_lse,
        softmax_scale,
        q_scale,
        kv_scale,
    )


def _fake_out(
    q,
    kv_buffer,
    kv_indices,
    out,
    lse,
    partial_o,
    partial_lse,
    softmax_scale,
    q_scale,
    kv_scale,
):
    return None


direct_register_custom_op(
    "sparse_mla_fp8_fwd_out",
    _sparse_mla_fp8_fwd_out,
    ["out", "lse", "partial_o", "partial_lse"],
    fake_impl=_fake_out,
)


def sparse_mla_fp8_fwd(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    kv_indices: torch.Tensor,
    softmax_scale: float,
    *,
    q_scale: float = 1.0,
    kv_scale: float = 1.0,
    out: torch.Tensor | None = None,
    return_lse: bool = False,
    lse: torch.Tensor | None = None,
    workspace: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sparse attention with a shared FP8 K/V latent and no appended RoPE.

    Q is BF16 or E4M3FN [Q,64,512], with Q <= 16384. Q rows must be aligned
    to 16 bytes, non-overlapping, and have unit inner stride. KV is contiguous
    E4M3FN [slots,512], [pages,page_size,512], or [slots,1,1,512].

    ``kv_indices`` is contiguous int32 [Q,K] or [Q,1,K], K <= 2176. Its
    entries are physical slot IDs, not page IDs. Negative/out-of-range IDs
    are masked; duplicates count separately. Empty rows return zero output
    and negative-infinity natural-log LSE.

    ``q_scale`` and ``kv_scale`` are finite positive scalar decoding scales.
    BF16 Q is rounded to E4M3FN as Q/q_scale inside attention. FP8 Q is used
    directly. One FP8 QK and FP8 PV are computed, with FP32 accumulation,
    softmax and LSE. Split partial output and final output are BF16.

    Supply ``out``, optional ``lse``, and a pair from
    ``allocate_sparse_mla_fp8_workspace(q)`` to reuse storage. Writable tensors
    must not overlap any other argument. Warm the JIT before HIP Graph capture;
    dispatch uses host shape metadata and replay reads updated input values.
    This opt-in operator supports gfx950 only and has no engine fallback.
    """
    if q.ndim != 3 or q.shape[1:] != (64, 512) or not 0 <= q.shape[0] <= 16384:
        raise ValueError("expected q=[Q,64,512], 0 <= Q <= 16384")
    if kv_buffer.ndim == 4 and kv_buffer.shape[1:3] == (1, 1):
        kv_buffer = kv_buffer.view(kv_buffer.shape[0], kv_buffer.shape[-1])
    elif kv_buffer.ndim == 3:
        kv_buffer = kv_buffer.view(-1, kv_buffer.shape[-1])
    if kv_indices.ndim == 3 and kv_indices.shape[1] == 1:
        kv_indices = kv_indices.view(kv_indices.shape[0], kv_indices.shape[2])
    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    if lse is not None and not return_lse:
        raise ValueError("supplying lse requires return_lse=True")
    if return_lse and lse is None:
        lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    if workspace is None:
        workspace = allocate_sparse_mla_fp8_workspace(q)
    partial_o, partial_lse = workspace
    torch.ops.aiter.sparse_mla_fp8_fwd_out(
        q,
        kv_buffer,
        kv_indices,
        out,
        lse,
        partial_o,
        partial_lse,
        float(softmax_scale),
        float(q_scale),
        float(kv_scale),
    )
    return out, lse
