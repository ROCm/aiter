# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""NVFP4 sparse-prefill attention with K/V reuse across query tokens (gfx1250).

The prefill counterpart of ``pa_decode``'s NVFP4 wave tile for MiniMax-M3.1's
block-sparse attention (16 query heads per KV head, head_dim 128, 128-token
NVFP4 pages, top-16 pages a token): a CTA takes ``tile_rows`` consecutive tokens
and walks the union of their selected pages once. See
``kernels.pa_sparse_prefill_nvfp4``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.utility import dtypes

from .kernels.pa_sparse_prefill_nvfp4 import (
    GROUP,
    HEAD_DIM,
    PAGE,
    TILE_ROWS,
    TOPK,
    compile_pa_sparse_prefill_nvfp4,
)
from .kernels.pa_sparse_prefill_nvfp4_v2 import (
    TILE_ROWS as TILE_ROWS_V2,
    compile_pa_sparse_prefill_nvfp4_v2,
)
from .kernels.pa_sparse_prefill_nvfp4_v3 import (
    TILE_ROWS as TILE_ROWS_V3,
    compile_pa_sparse_prefill_nvfp4_v3,
)
from .kernels.tensor_shim import _run_compiled, ptr_arg

# The arguments MiniMax-M3.1's prefill (ATOM's P path) passes after kv_heads.
M31_PREFILL_CALL = {"impl": "v3", "tile_rows": 8}
# impl "v3" runs v2 (tile_rows 2) on grids of fewer v3 CTAs than this, on one KV
# head and for the MXFP8 output.
V3_MIN_CTAS = 4096


def pa_sparse_prefill_nvfp4_supported(query, key_cache, table) -> bool:
    """Whether the reuse kernel serves these shapes (gfx1250, 16 query heads a
    sequence, head_dim 128, 128-token pages, a top-16 table)."""
    return (
        get_gfx_runtime() == "gfx1250"
        and query.dim() == 3
        and query.shape[1:] == (GROUP, HEAD_DIM)
        and query.dtype == torch.bfloat16
        and key_cache.dim() == 4
        and tuple(key_cache.shape[1:]) == (1, PAGE, HEAD_DIM // 2)
        and table.dim() == 2
        and table.shape[1] == TOPK
    )


def pa_sparse_prefill_nvfp4(
    output: torch.Tensor,  # [seqs, 16, 128] bf16, query's strides
    query: torch.Tensor,  # [seqs, 16, 128] bf16; seq = row * kv_heads + kv_head
    key_cache: torch.Tensor,  # [pages, 1, 128, 64] uint8/fp4x2 e2m1, contiguous
    value_cache: torch.Tensor,  # same as key_cache
    key_scale: torch.Tensor,  # [pages, 1, 128, 8] float8_e4m3fn
    value_scale: torch.Tensor,  # same layout and strides as key_scale
    table: torch.Tensor,  # [seqs, 16] int32 compacted page ids, tail page last
    context: torch.Tensor,  # [seqs] int32 compacted lengths
    softmax_scale: float,
    *,
    kv_heads: int = 1,
    tile_rows: int = 4,
    nvfp4_buffers: int = 3,
    q_regs: bool = False,
    row_group: int = 1,
    skip_rescale: bool = False,
    fma_exp: bool = False,
    quantized_output: torch.Tensor | None = None,
    output_scale: torch.Tensor | None = None,
    quantized_pad_rows: int = 0,
    impl: str = "v1",
    v2: dict | None = None,
    v3: dict | None = None,
) -> None:
    """Sparse prefill attention of every sequence over its compacted page table.

    The table and lengths are those of ``pa_decode``'s NVFP4 sparse path (ATOM's
    ``nvfp4_page_table`` or the selector's emission): row ``s`` lists its pages,
    full pages first and the partially visible tail page last, and attends
    ``context[s]`` tokens of them. Rows ``row * kv_heads + h`` for consecutive
    ``row`` share a CTA (``tile_rows`` of them), so ``kv_heads`` must be the
    number of KV heads interleaved in the flattened rows. A row of length 0
    stores zeros. Equal to ``pa_decode`` to rounding (each row's pages are
    accumulated in the tile's union order). Page ids outside the cache select
    nothing, so unwritten padding rows never read past it.

    ``quantized_output`` (FP8 ``[seqs + quantized_pad_rows, 16, 128]``, contiguous)
    with E8M0 ``output_scale`` (``[seqs + quantized_pad_rows, 64]``): o_proj's
    MXFP8 input instead of ``output`` (left unwritten), bit for bit the separate
    group-32 quant of the BF16 rows; the padded rows get FP8 zeros and unit
    scales. One KV head.

    ``impl``: "v1", "v2" (``v2``: its compile options) or "v3" (the tile's rows
    split over waves, ``tile_rows`` 4 or 8; ``v3``: its compile options and
    ``min_ctas``, default ``V3_MIN_CTAS``). "v3" runs "v2" at ``tile_rows`` 2 on
    one KV head, for the MXFP8 output and on grids under ``min_ctas`` v3 CTAs.
    """
    served = {"v2": TILE_ROWS_V2, "v3": TILE_ROWS_V3}.get(impl, TILE_ROWS)
    if tile_rows not in served:
        raise ValueError(f"tile_rows {tile_rows} is not served by impl {impl}")
    seqs = query.shape[0]
    dev = query.device
    if query.shape[1:] != (GROUP, HEAD_DIM) or query.dtype != torch.bfloat16:
        raise ValueError(f"query must be bf16 [seqs, {GROUP}, {HEAD_DIM}]")
    if (
        output.shape != query.shape
        or output.dtype != query.dtype
        or output.stride() != query.stride()
    ):
        raise ValueError("output must match the query's shape, dtype and strides")
    if query.stride(2) != 1 or query.stride(1) % 8 or query.stride(0) % 8:
        raise ValueError(
            "query rows and heads must be 16 B aligned with unit dim stride"
        )
    payload_shape = (key_cache.shape[0], 1, PAGE, HEAD_DIM // 2)
    for name, t in (("key_cache", key_cache), ("value_cache", value_cache)):
        if (
            tuple(t.shape) != payload_shape
            or t.dtype not in (torch.uint8, dtypes.fp4x2)
            or not t.is_contiguous()
            or t.data_ptr() % 16
            or t.device != dev
        ):
            raise ValueError(
                f"{name} must be a contiguous 16 B aligned {list(payload_shape)} uint8"
            )
    scale_shape = (key_cache.shape[0], 1, PAGE, HEAD_DIM // 16)
    for name, t in (("key_scale", key_scale), ("value_scale", value_scale)):
        if (
            tuple(t.shape) != scale_shape
            or t.dtype != torch.float8_e4m3fn
            or t.stride() != key_scale.stride()
            or t.stride()[2:] != (HEAD_DIM // 16, 1)
            or t.stride(0) % 16
            or t.data_ptr() % 16
            or t.device != dev
        ):
            raise ValueError(
                f"{name} must be float8_e4m3fn {list(scale_shape)} with 16 B aligned pages"
            )
    if seqs % kv_heads:
        raise ValueError("the flattened rows must hold every KV head of each row")
    if (
        table.dtype != torch.int32
        or table.dim() != 2
        or table.shape[0] < seqs
        or table.shape[1] != TOPK
        or table.stride(1) != 1
        or context.dtype != torch.int32
        or context.shape[0] < seqs
        or context.stride(0) != 1
    ):
        raise ValueError(
            "table must be int32 [seqs, 16] with unit stride, context int32 [seqs]"
        )
    quant = quantized_output is not None
    quant_rows = 0
    if quant:
        quant_rows = seqs + quantized_pad_rows
        if (
            kv_heads != 1
            or quantized_output.dtype not in (torch.float8_e4m3fn, torch.uint8)
            or not quantized_output.is_contiguous()
            or quantized_output.numel() != quant_rows * GROUP * HEAD_DIM
            or output_scale is None
            or output_scale.dtype not in (torch.float8_e8m0fnu, torch.uint8)
            or not output_scale.is_contiguous()
            or output_scale.numel() != quant_rows * GROUP * HEAD_DIM // 32
        ):
            raise ValueError(
                "the MXFP8 output needs one KV head, contiguous FP8 "
                "[seqs + pad, 16, 128] values and E8M0 [seqs + pad, 64] scales"
            )
    if seqs == 0 and quant_rows == 0:
        return
    v3 = dict(v3 or {})
    min_ctas = v3.pop("min_ctas", V3_MIN_CTAS)
    if impl == "v3" and (
        kv_heads == 1
        or quant
        or -(-seqs // (kv_heads * tile_rows)) * kv_heads < min_ctas
    ):
        impl, tile_rows, v2 = "v2", 2, None
    if impl == "v3":
        compiled = compile_pa_sparse_prefill_nvfp4_v3(
            tile_rows=tile_rows,
            softmax_scale=softmax_scale,
            nvfp4_buffers=nvfp4_buffers,
            **v3,
        )
    elif impl == "v2":
        compiled = compile_pa_sparse_prefill_nvfp4_v2(
            tile_rows=tile_rows,
            softmax_scale=softmax_scale,
            nvfp4_buffers=nvfp4_buffers,
            quant_mx=quant,
            **(v2 or {}),
        )
    else:
        compiled = compile_pa_sparse_prefill_nvfp4(
            tile_rows=tile_rows,
            softmax_scale=softmax_scale,
            nvfp4_buffers=nvfp4_buffers,
            q_regs=q_regs,
            row_group=row_group,
            skip_rescale=skip_rescale,
            fma_exp=fma_exp,
            quant_mx=quant,
        )
    _run_compiled(
        compiled["launch"],
        ptr_arg(query, fx.BFloat16),
        ptr_arg(key_cache, fx.Uint8),
        ptr_arg(value_cache, fx.Uint8),
        ptr_arg(key_scale, fx.Uint8),
        ptr_arg(value_scale, fx.Uint8),
        ptr_arg(table, fx.Int32),
        ptr_arg(context, fx.Int32),
        ptr_arg(output, fx.BFloat16),
        seqs // kv_heads,
        kv_heads,
        table.stride(0),
        key_scale.stride(0),
        query.stride(0),
        query.stride(1),
        key_cache.shape[0],
        # Null without the MXFP8 output, as pa_decode passes them.
        *(
            (ptr_arg(quantized_output, fx.Uint8), ptr_arg(output_scale, fx.Uint8))
            if quant
            else (flyc.from_c_void_p(fx.Uint8, 0),) * 2
        ),
        quant_rows,
        torch.cuda.current_stream(dev),
        specialization_key=(dev.index,),
    )
