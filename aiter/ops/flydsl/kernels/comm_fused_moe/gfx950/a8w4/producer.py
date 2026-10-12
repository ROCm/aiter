# SPDX-License-Identifier: Apache-2.0
"""Prepare route storage and adapt its ABI to the MXMoE GEMM2 producer."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, ptrtoint
from flydsl.expr.typing import T

from ....mxfp4_gemm_common import global_typed_ptr
from ....mxmoe_dispatcher import compile_gemm2_a4w4_port
from .config import MegakernelConfig, WindowConfig

_ROUTE_STORE_CACHE_MODIFIER = 0x10  # sc1


@functools.cache
def compile_mark_written_routes(tokens: int, topk: int):
    """Mark only real (token, slot) entries in the producer's sorted work."""

    @flyc.kernel(
        name=f"comm_fused_moe_mark_routes_v1_m{tokens}_k{topk}",
        known_block_size=[256, 1, 1],
    )
    def kernel(validity: fx.Pointer, sorted_ids: fx.Pointer, counts: fx.Pointer):
        valid = global_typed_ptr(fx.Int64(ptrtoint(validity)), T.i32)
        ids = global_typed_ptr(fx.Int64(ptrtoint(sorted_ids)), T.i32)
        count = global_typed_ptr(fx.Int64(ptrtoint(counts)), T.i32)[0]
        index = fx.Int32(gpu.block_id("x")) * fx.Int32(256) + fx.Int32(
            gpu.thread_id("x")
        )
        stride = fx.Int32(gpu.grid_dim.x) * fx.Int32(256)
        for item in range(index, count, stride):
            packed = fx.Int32(ids[item])
            token = packed & fx.Int32(0x00FFFFFF)
            slot = packed >> fx.Int32(24)
            if (
                (token < fx.Int32(tokens))
                & (slot >= fx.Int32(0))
                & (slot < fx.Int32(topk))
            ):
                # Sorting emits each retained (token, slot) exactly once.
                valid[token * fx.Int32(topk) + slot] = fx.Int32(1)

    @flyc.jit
    def launch(validity, sorted_ids, counts, stream):
        kernel(validity, sorted_ids, counts).launch(
            grid=(min(256, max(1, (tokens * topk + 255) // 256)), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch


@functools.cache
def compile_clear_missing_routes(rows: int, row_bytes: int, payload_bytes: int):
    """Clear missing payloads without reading them; FP8 scales become 1."""

    if not 0 < payload_bytes <= row_bytes or row_bytes % 4 or payload_bytes % 4:
        raise ValueError("route payload and row sizes must be positive whole dwords")
    words_per_row = row_bytes // 4
    payload_words = payload_bytes // 4

    @flyc.kernel(
        name=(
            f"comm_fused_moe_clear_missing_routes_v1_r{rows}"
            f"_b{row_bytes}_p{payload_bytes}"
        ),
        known_block_size=[256, 1, 1],
    )
    def kernel(route: fx.Pointer, validity: fx.Pointer):
        output = global_typed_ptr(fx.Int64(ptrtoint(route)), T.i32)
        valid = global_typed_ptr(fx.Int64(ptrtoint(validity)), T.i32)
        index = fx.Int32(gpu.block_id("x")) * fx.Int32(256) + fx.Int32(
            gpu.thread_id("x")
        )
        stride = fx.Int32(gpu.grid_dim.x) * fx.Int32(256)
        for item in range(index, fx.Int32(rows * words_per_row), stride):
            row = item // fx.Int32(words_per_row)
            if fx.Int32(valid[row]) == fx.Int32(0):
                column = item % fx.Int32(words_per_row)
                output[item] = (column < fx.Int32(payload_words)).select(
                    fx.Int32(0), fx.Int32(0x7F7F7F7F)
                )

    @flyc.jit
    def launch(route, validity, stream):
        kernel(route, validity).launch(
            grid=(min(256, max(1, (rows * words_per_row + 255) // 256)), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch


@flyc.jit
def resolve_route_input_row(packed, tokens, topk):
    packed = fx.Int32(packed)
    token = packed & fx.Int32(0x00FFFFFF)
    route_slot = packed >> fx.Int32(24)
    valid = (token < tokens) & (route_slot < fx.Int32(topk))
    return valid.select(
        token * fx.Int32(topk) + route_slot,
        fx.Int32(0),
    )


def _input_row_resolver_for(config):
    if config.sorted_input:
        return None
    topk = config.shape.topk
    return lambda packed, tokens: resolve_route_input_row(packed, tokens, topk)


def compile_megakernel_producer(config: MegakernelConfig, composition):
    shape = config.shape

    return compile_gemm2_a4w4_port(
        BM=config.tile_m,
        BN=config.tile_n,
        BK=config.tile_k,
        use_nt=config.b_cache_modifier == 2,
        HIDDEN_MAX=shape.model_dim,
        epilog="reduce",
        INTER_MAX=shape.inter_dim,
        a_dtype="fp8",
        b_dtype="fp4",
        topk=shape.topk,
        SBM=config.sort_block_m,
        persist=False,
        g2_bhoist=True,
        g2_ascale_pf=True,
        g2_spart=0,
        g2_bf16_lds=False,
        out_dtype="bf16",
        enable_bias=False,
        _composition=composition,
        _reduce_store_cache_modifier=_ROUTE_STORE_CACHE_MODIFIER,
        _input_row_resolver=_input_row_resolver_for(config),
    )


def compile_window_producer(config: WindowConfig, window: int, composition):
    shape = config.shape
    n_start = window * config.window
    n_end = n_start + config.window

    return compile_gemm2_a4w4_port(
        BM=config.tile_m,
        BN=config.tile_n,
        BK=config.tile_k,
        HIDDEN_MAX=shape.model_dim,
        epilog="reduce",
        INTER_MAX=shape.inter_dim,
        a_dtype="fp8",
        b_dtype="fp4",
        topk=shape.topk,
        SBM=config.sort_block_m,
        persist=False,
        g2_bhoist=False,
        g2_ascale_pf=True,
        g2_spart=0,
        g2_bf16_lds=True,
        out_dtype="fp8",
        enable_bias=False,
        _composition=composition,
        _input_row_resolver=_input_row_resolver_for(config),
        _output_n_range=(n_start, n_end),
    )
