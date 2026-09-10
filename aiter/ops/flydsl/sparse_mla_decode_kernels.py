# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Public launcher for gfx950 FlyDSL sparse MLA decode."""

from __future__ import annotations

import math

import flydsl.expr as fx
import torch

from .kernels.sparse_mla_decode import BLOCK_I, DIM, DV, H, compile_sparse_mla_partial
from .kernels.tensor_shim import _run_compiled, ptr_arg
from .mla_reduce_kernels import _flydsl_sparse_mla_decode_combine

_WARM_KEYS: dict[str, set] = {}


def _require_warm(key, what: str, compile_fn):
    """Compile a variant outside graph capture, recording only success.

    Compiling runs MLIR, hipModuleLoad and disk IO; doing that between
    capture_begin and capture_end is not legal and does not fail cleanly.
    Every (seq, width) bucket must be run eagerly once first. The key is
    recorded after `compile_fn` returns, so a compile that raises leaves the
    variant cold and the next capture still refuses it.
    """
    seen = _WARM_KEYS.setdefault(what, set())
    if key in seen:
        return compile_fn()
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"sparse MLA decode: {what} variant {key} is not compiled and this "
            "stream is capturing. Run this shape once outside the capture first."
        )
    compiled = compile_fn()
    seen.add(key)
    return compiled


def _pick_inner_iter(seq: int, ng_total: int) -> int:
    """Return the producer grouping factor for this shape.

    Merging adjacent 64-key tiles halves the producer grid and doubles the
    serial depth per CTA, which trades concurrency for scratch traffic and
    combine work. The floors below are measured thresholds, not a coverage
    guarantee: 192 is under a 256-CU part, so the first merge does leave the
    device short, and seq=12 and 16 are correspondingly the weakest points
    against a Triton baseline. Raising the floors is a separate change --
    it would need the reducer policy revisited with it.
    """
    inner_iter = 1
    while inner_iter < 4:
        candidate = inner_iter * 2
        if ng_total % candidate != 0:
            break
        min_producer_ctas = 192 if candidate == 2 else 384
        if seq * (ng_total // candidate) < min_producer_ctas:
            break
        inner_iter = candidate
    return inner_iter


def _partial_groups(ng_total: int, inner_iter: int) -> int:
    if inner_iter < 1 or ng_total % inner_iter != 0:
        raise ValueError(
            f"ng_total={ng_total} must be divisible by inner_iter={inner_iter}"
        )
    return ng_total // inner_iter


def _use_split_major(seq: int, n_groups: int, num_cu: int) -> bool:
    """Use split-major ownership only once the producer grid is saturated."""
    return seq * n_groups >= 2 * num_cu


def sparse_mla_decode_workspace_shape(
    seq: int, width: int
) -> tuple[tuple[int, int, int, int], tuple[int, int, int]]:
    """Return the partial-output and partial-LSE shapes for sparse MLA decode."""
    if not 1 <= seq <= 96:
        raise ValueError(f"supported sparse decode seq values are 1..96; got {seq}")
    if width % BLOCK_I != 0:
        raise ValueError(f"index width must be padded to {BLOCK_I}, got {width}")
    ng = width // BLOCK_I
    if not 1 <= ng <= 33:
        raise ValueError(f"supported split count is 1..33, got {ng}")
    ng_partial = _partial_groups(ng, _pick_inner_iter(seq, ng))
    return (seq, ng_partial, H, DV), (seq, ng_partial, H)


def _require_cuda_tensor(
    name: str, tensor: torch.Tensor, *, dtype: torch.dtype
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)!r}")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a ROCm tensor, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")


def _validate_sparse_decode_inputs(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor | None,
) -> tuple[int, int, int]:
    _require_cuda_tensor("q", q, dtype=torch.float8_e4m3fn)
    _require_cuda_tensor("kv", kv, dtype=torch.float8_e4m3fn)
    _require_cuda_tensor("indices", indices, dtype=torch.int32)
    if out is not None:
        _require_cuda_tensor("out", out, dtype=torch.bfloat16)

    # The MFMA tile is 16 rows wide, so the kernel always runs 16 head
    # slots. A model with fewer -- GLM-5.2 at TP8 has 8 -- is read in place
    # at its own row stride, not staged into a padded copy; the slots past
    # it re-read head 0 and are dropped downstream.
    if q.ndim != 3 or int(q.shape[2]) != DIM or not 1 <= int(q.shape[1]) <= H:
        raise ValueError(
            f"q must have shape [seq,heads,{DIM}] with 1 <= heads <= {H}, "
            f"got {tuple(q.shape)}"
        )
    heads = int(q.shape[1])
    seq = int(q.shape[0])
    # seq is runtime; kernels are compiled per split count `ng`, not per seq.
    if not 1 <= seq <= 96:
        raise ValueError(f"supported sparse decode seq values are 1..96; got {seq}")
    if not (
        (kv.ndim == 2 and int(kv.shape[1]) == DIM)
        or (kv.ndim == 3 and tuple(kv.shape[1:]) == (1, DIM))
    ):
        raise ValueError(
            f"kv must have shape [P,{DIM}] or [P,1,{DIM}], got {tuple(kv.shape)}"
        )
    # An empty pool hands the kernel a null base pointer and faults it. Every
    # index into it is out of range anyway, so there is nothing to attend to.
    if int(kv.shape[0]) == 0:
        raise ValueError("kv must have at least one row, got an empty pool")
    if out is not None and (out.ndim != 3 or tuple(out.shape) != (seq, heads, DV)):
        raise ValueError(
            f"out must have shape [{seq},{heads},{DV}], got {tuple(out.shape)}"
        )
    if (
        q.device != kv.device
        or q.device != indices.device
        or (out is not None and q.device != out.device)
    ):
        raise ValueError("all sparse decode tensors must be on the same device")

    width = int(indices.numel() // seq)
    if tuple(indices.shape) != (seq, width):
        raise ValueError(
            f"indices must have shape [{seq},width], got {tuple(indices.shape)}"
        )
    if width % BLOCK_I != 0:
        raise ValueError(f"index width must be padded to {BLOCK_I}, got {width}")
    ng = width // BLOCK_I
    if not 1 <= ng <= 33:
        raise ValueError(f"supported split count is 1..33, got {ng}")

    # A CUDA build has no `gcnArchName`, so read it defensively: the gate
    # should refuse the device, not raise AttributeError from inside it.
    props = torch.cuda.get_device_properties(q.device)
    arch = str(getattr(props, "gcnArchName", "")).split(":")[0]
    if arch != "gfx950":
        raise ValueError(f"FlyDSL sparse MLA decode is gated to gfx950, got {arch}")
    return seq, ng, heads


def _validate_workspace(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    *,
    seq: int,
    ng_partial: int,
    device: torch.device,
) -> None:
    _require_cuda_tensor("partial_output", partial_output, dtype=torch.bfloat16)
    _require_cuda_tensor("partial_lse", partial_lse, dtype=torch.float32)
    if partial_output.device != device or partial_lse.device != device:
        raise ValueError("sparse decode workspace must share the decode device")
    if tuple(partial_output.shape) != (seq, ng_partial, H, DV):
        raise ValueError(
            f"partial_output must have shape [{seq},{ng_partial},{H},{DV}], got "
            f"{tuple(partial_output.shape)}"
        )
    if tuple(partial_lse.shape) != (seq, ng_partial, H):
        raise ValueError(
            f"partial_lse must have shape [{seq},{ng_partial},{H}], got "
            f"{tuple(partial_lse.shape)}"
        )


def _launch_partial(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    sm_scale: float,
    *,
    ng: int,
    inner_iter: int,
) -> None:
    n_groups = _partial_groups(ng, inner_iter)
    num_cu = int(torch.cuda.get_device_properties(q.device).multi_processor_count)
    split_major = _use_split_major(int(q.shape[0]), n_groups, num_cu)
    launch = _require_warm(
        (ng, inner_iter, split_major),
        "partial",
        lambda: compile_sparse_mla_partial(
            ng, inner_iter=inner_iter, split_major=split_major
        ),
    )
    _run_compiled(
        launch,
        ptr_arg(q, fx.Uint8),
        ptr_arg(kv.reshape(-1, DIM), fx.Uint8),
        ptr_arg(indices, fx.Int32),
        ptr_arg(partial_output, fx.BFloat16),
        ptr_arg(partial_lse, fx.Float32),
        float(sm_scale) * math.log2(math.e),
        int(q.shape[0]),
        int(kv.reshape(-1, DIM).shape[0]),
        int(q.shape[1]),
        fx.Stream(torch.cuda.current_stream(q.device)),
    )


def flydsl_sparse_mla_decode(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
    sm_scale: float,
    *,
    partial_output: torch.Tensor | None = None,
    partial_lse: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run sparse MLA decode via FlyDSL partials and the shared reducer.

    Provide persistent ``partial_output`` and ``partial_lse`` buffers when
    capturing the call in a HIP graph. If they are omitted, temporary scratch is
    allocated eagerly for convenience.
    """
    seq, ng, heads = _validate_sparse_decode_inputs(q, kv, indices, out)
    # The producer reads `q` at the caller's head count, but the partials and
    # the shared reducer are fixed at the 16-row tile, so a narrow call still
    # narrows once at the end.
    narrow_final = None
    if heads != H:
        narrow_final = out
        out = out.new_empty((seq, H, DV))
    inner_iter = _pick_inner_iter(seq, ng)
    ng_partial = _partial_groups(ng, inner_iter)
    if (partial_output is None) != (partial_lse is None):
        raise ValueError(
            "partial_output and partial_lse must be provided together for "
            "graph-safe scratch reuse"
        )
    if partial_output is None:
        partial_output = torch.empty(
            (seq, ng_partial, H, DV), device=q.device, dtype=torch.bfloat16
        )
        partial_lse = torch.empty(
            (seq, ng_partial, H), device=q.device, dtype=torch.float32
        )
    else:
        _validate_workspace(
            partial_output,
            partial_lse,
            seq=seq,
            ng_partial=ng_partial,
            device=q.device,
        )

    _launch_partial(
        q,
        kv,
        indices,
        partial_output,
        partial_lse,
        sm_scale,
        ng=ng,
        inner_iter=inner_iter,
    )
    _flydsl_sparse_mla_decode_combine(
        partial_output.unsqueeze(0),
        partial_lse.unsqueeze(0),
        out.unsqueeze(0),
    )
    if narrow_final is not None:
        narrow_final.copy_(out[:, :heads])
        return narrow_final
    return out


__all__ = ["flydsl_sparse_mla_decode", "sparse_mla_decode_workspace_shape"]
