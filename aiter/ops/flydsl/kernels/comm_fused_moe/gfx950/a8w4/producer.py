# SPDX-License-Identifier: Apache-2.0
"""Adapt the communication-fused route ABI to the MXMoE GEMM2 producer."""

import flydsl.compiler as flyc
import flydsl.expr as fx

from ....mxmoe_dispatcher import compile_gemm2_a4w4_port
from .config import MegakernelConfig, WindowConfig

_ROUTE_STORE_CACHE_MODIFIER = 0x10  # sc1


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
        epilog=("atomic" if config.producer_mode == "atomic_shared" else "reduce"),
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
        g2_kstatic=True,
        out_dtype="bf16",
        enable_bias=False,
        _composition=composition,
        _reduce_store_cache_modifier=(
            _ROUTE_STORE_CACHE_MODIFIER if config.producer_mode == "routes" else None
        ),
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
        g2_kstatic=True,
        out_dtype="fp8",
        enable_bias=False,
        _composition=composition,
        _input_row_resolver=_input_row_resolver_for(config),
        _output_n_range=(n_start, n_end),
    )
