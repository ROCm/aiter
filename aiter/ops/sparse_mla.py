# SPDX-License-Identifier: MIT
"""HIP sparse MLA for BF16 D512 latent K/V on MI350 (gfx950)."""

import torch

from aiter.jit.core import compile_ops
from aiter.jit.utils.chip_info import get_gfx_runtime
from csrc.cpp_itfs.torch_utils import direct_register_custom_op


@compile_ops(
    "module_sparse_mla_bf16", fc_name="sparse_mla_bf16_fwd_out", ffi_type="ctypes"
)
def _jit_sparse_mla_bf16_fwd_out(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor | None,
    partial_o: torch.Tensor | None,
    partial_lse: torch.Tensor | None,
    softmax_scale: float,
    kv_splits: int,
    version: int,
) -> None: ...


def _sparse_mla_bf16_fwd_out(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor | None,
    partial_o: torch.Tensor | None,
    partial_lse: torch.Tensor | None,
    softmax_scale: float,
    kv_splits: int,
    version: int,
) -> None:
    # Runtime discovery stays inside the opaque op, outside Dynamo tracing.
    if get_gfx_runtime() != "gfx950":
        raise NotImplementedError("sparse_mla_bf16_fwd requires gfx950")
    _jit_sparse_mla_bf16_fwd_out(
        q,
        kv_buffer,
        kv_indptr,
        kv_indices,
        out,
        lse,
        partial_o,
        partial_lse,
        softmax_scale,
        kv_splits,
        version,
    )


def _fake_out(
    q,
    kv_buffer,
    kv_indptr,
    kv_indices,
    out,
    lse,
    partial_o,
    partial_lse,
    softmax_scale,
    kv_splits,
    version,
):
    return None


direct_register_custom_op(
    "sparse_mla_bf16_fwd_out",
    _sparse_mla_bf16_fwd_out,
    ["out", "lse", "partial_o", "partial_lse"],
    fake_impl=_fake_out,
)


def sparse_mla_bf16_fwd(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    softmax_scale: float,
    *,
    kv_splits: int = 1,
    version: int = 2,
    out: torch.Tensor | None = None,
    return_lse: bool = False,
    lse: torch.Tensor | None = None,
    workspace: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sparse attention with a shared BF16 K/V latent and no appended RoPE.

    ``q`` is [Q, H, 512], H in {16, 64}. ``kv_buffer`` is [slots, 512],
    [pages, page_size, 512], or [slots, 1, 1, 512]. Paged pools must be viewable
    as flat rows without copying. Q/KV permit padded outer strides and BF16
    storage offsets; their last dimension must have stride one.

    ``kv_indptr`` is contiguous int32 [Q+1], and ``kv_indices`` is a contiguous
    int32 vector of GLOBAL slot ids. Offsets must start at zero, be monotone,
    and end at indices.numel(). These device values are caller invariants;
    the hot path does not synchronize to validate them. Negative and
    out-of-range slot ids are masked. Duplicates retain their multiplicity.
    Empty rows produce O=0 and natural-log LSE=-inf.

    ``version=1`` selects Full140 for H64 (139,520 B LDS) and the one-wave
    H16 kernel. ``version=2`` selects the hybrid: H16 uses four waves, and
    H64 uses four head tiles when Q*kv_splits <= 128, otherwise four waves.
    Both H64 versions and v2 H16 use the fully unrolled split-K reducer.
    ``kv_splits`` is an explicit power of two from 1 through 32.

    O is contiguous BF16 [Q,H,512]; optional LSE is contiguous FP32 [Q,H].
    For split-K, workspace is contiguous FP32 ([Q,S,H,512], [Q,S,H]).
    Supply out/lse/workspace to reuse all storage without allocation. Writable
    storage must not overlap other arguments. Warm the JIT before graph capture.
    Only gfx950 is supported; unsupported configurations raise an error.
    """
    if q.ndim != 3:
        raise ValueError("q must have shape [Q,H,512]")
    if version not in (1, 2) or kv_splits not in (1, 2, 4, 8, 16, 32):
        raise ValueError("version must be 1/2 and kv_splits must be 1/2/4/8/16/32")
    queries, heads, width = q.shape
    if kv_buffer.ndim == 4 and kv_buffer.shape[1:3] == (1, 1):
        kv_buffer = kv_buffer.view(kv_buffer.shape[0], kv_buffer.shape[-1])
    elif kv_buffer.ndim == 3:
        kv_buffer = kv_buffer.view(-1, kv_buffer.shape[-1])
    if out is None:
        out = torch.empty((queries, heads, width), dtype=q.dtype, device=q.device)
    if lse is not None and not return_lse:
        raise ValueError("supplying lse requires return_lse=True")
    if return_lse and lse is None:
        lse = torch.empty((queries, heads), dtype=torch.float32, device=q.device)
    partial_o = partial_lse = None
    if kv_splits > 1:
        if workspace is None:
            partial_o = torch.empty(
                (queries, kv_splits, heads, 512), dtype=torch.float32, device=q.device
            )
            partial_lse = torch.empty(
                (queries, kv_splits, heads), dtype=torch.float32, device=q.device
            )
        else:
            partial_o, partial_lse = workspace
    torch.ops.aiter.sparse_mla_bf16_fwd_out(
        q,
        kv_buffer,
        kv_indptr,
        kv_indices,
        out,
        lse,
        partial_o,
        partial_lse,
        float(softmax_scale),
        kv_splits,
        version,
    )
    return out, lse
