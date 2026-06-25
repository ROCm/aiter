# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL MLA decode reduce API.

Drop-in fallback for the HIP ``aiter.mla_reduce_v1`` (csrc/kernels/mla/reduce.cu).
Same signature, same input/output contract, so production paths can swap it in.

The FlyDSL kernel is compiled per (H, Dv, out_dtype, tier, output_lse) and cached
(``compile_mla_reduce`` is ``lru_cache``-backed). The host picks the split-tier
from the observed ``num_splits`` (``select_tier``) and launches one workgroup per
``(head, tile)`` work item.

This is an OPT-IN fallback. Production code (``aiter/mla.py``) keeps calling the
HIP ``aiter.mla_reduce_v1`` by default; set ``AITER_MLA_REDUCE_FLYDSL=1`` to route
through this wrapper instead.
"""

from __future__ import annotations

import torch

from .kernels.mla_reduce import compile_mla_reduce, select_tier

__all__ = [
    "flydsl_mla_reduce_v1",
]


def _out_dtype_str(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    raise ValueError(
        f"flydsl_mla_reduce_v1 only supports bf16/fp16 output, got {dtype!r}"
    )


def flydsl_mla_reduce_v1(
    partial_output: torch.Tensor,  # fp32 [rows, H, Dv] contiguous
    partial_lse: torch.Tensor,  # fp32 [rows, H] contiguous
    reduce_indptr: torch.Tensor,  # i32  [num_reduce_tile + 1]
    reduce_final_map: torch.Tensor | None,  # i32 [num_reduce_tile, 2] (optional)
    reduce_partial_map: torch.Tensor,  # i32 [reduce_indptr[-1]]
    max_seqlen_q: int,
    final_output: torch.Tensor,  # bf16/fp16 [bs, H, Dv]
    final_lse: torch.Tensor | None = None,  # fp32 [bs, H] (optional)
    num_kv_splits: int = 0,  # signature parity; grid derived from num_cu + CSR width
    *,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """FlyDSL drop-in replacement for ``aiter.mla_reduce_v1``.

    Matches the HIP kernel's signature and write-in-place contract: results land
    in ``final_output`` (and ``final_lse`` if provided). No return value.

    Args:
        partial_output / partial_lse: fp32 per-split partials.
        reduce_indptr / reduce_partial_map: CSR over splits (+ gather map).
        reduce_final_map: optional [tile, 2] {q_start, q_end}; when ``None`` the
            q-range is derived from the partial map (uniform qo-len), mirroring
            the HIP ``use_reduce_final_map = false`` path.
        max_seqlen_q: number of q-positions per decode token group; drives grid-y
            (NTG), so multi-token decode (decode_qlen > 1) is handled correctly.
        final_output: output tensor (bf16/fp16); strides are read at runtime.
        final_lse: optional merged LSE output (fp32).
        num_kv_splits: kept for signature parity with the HIP kernel (unused; the
            split count is read from the CSR map / num_cu, not this arg).
        stream: launch stream; defaults to the current stream of the device.
    """
    if reduce_indptr.numel() < 2:
        return

    H = partial_output.size(-2)
    Dv = final_output.size(-1)
    out_dtype_str = _out_dtype_str(final_output.dtype)
    num_reduce_tile = reduce_indptr.numel() - 1
    output_lse = final_lse is not None
    use_reduce_final_map = reduce_final_map is not None

    num_cu = torch.cuda.get_device_properties(
        final_output.device.index
    ).multi_processor_count

    # Tier from the observed split count = CSR row-0 width (indptr[1] - indptr[0]),
    # matching the in-kernel `n_splits = indptr[tile+1] - indptr[tile]`. The old
    # `reduce_partial_map.numel() // num_reduce_tile` miscounts once partial rows
    # carry a q-dimension (decode_qlen > 1), so derive it from indptr directly.
    num_splits = int(reduce_indptr[1].item() - reduce_indptr[0].item())
    tier = select_tier(num_splits)

    kernel = compile_mla_reduce(
        H=H,
        Dv=Dv,
        out_dtype=out_dtype_str,
        tier=tier,
        output_lse=output_lse,
        use_reduce_final_map=use_reduce_final_map,
    )

    if final_lse is None:
        final_lse = torch.empty(1, dtype=torch.float32, device=final_output.device)
    if reduce_final_map is None:
        reduce_final_map = torch.empty(1, dtype=torch.int32, device=final_output.device)

    if stream is None:
        stream = torch.cuda.current_stream(final_output.device)

    kernel(
        partial_output,
        partial_lse,
        reduce_indptr,
        reduce_partial_map,
        reduce_final_map,
        final_output,
        final_lse,
        int(final_output.stride(-3)),
        int(final_output.stride(-2)),
        int(num_cu),
        int(num_reduce_tile),
        int(max_seqlen_q),
        stream,
    )
