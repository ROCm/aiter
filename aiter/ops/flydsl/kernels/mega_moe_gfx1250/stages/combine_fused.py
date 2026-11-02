# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The gemm2 epilogue's P2P scatter and the FlyDSL reduce as a combine stage."""

import flydsl.expr as fx
import torch

from ...tensor_shim import _run_compiled
from ..combine import (
    CHUNK_ELEMS as _COMBINE_CHUNK_ELEMS,
)
from ..combine import (
    COMBINE_LDS_BUDGET,
    _make_combine_fused_reduce,
    _make_combine_fused_sync,
    combine_reduce_lds_bytes,
)
from ..config import _WAVE_SIZE
from ..recipe import _COMBINE_WIRE_QUANT, MegaMoEConfig
from ..types import Stage2ScatterContext
from .base import (
    BatchLayout,
    Routing,
    _ComputeTarget,
    _DispatchBatch,
    _EpResources,
)

# Preferred lane tile for the TDM combine: T tokens x C chunks per block
# iteration, which the reduce turns into T*C*256/16 lanes -- so the quantized
# path runs a wider block than the bf16 one. A ceiling, not the answer:
# `_combine_tile` walks these down until the tile fits a given topk.
_COMBINE_TOKENS_PER_BLOCK = 16
_COMBINE_CHUNKS_PER_ITER = 1
# The MXFP8 wire wants a larger C: its scale plane puts only C*8 bytes in a TDM
# row, so a small C makes that load pull a whole cache line per row for a
# handful of bytes.
_COMBINE_QUANT_TOKENS_PER_BLOCK = 8
_COMBINE_QUANT_CHUNKS_PER_ITER = 4


def _combine_tile(*, topk, hidden_dim, quant_bits):
    """(tokens_per_block, chunks_per_iter) for the reduce, widest that fits.

    C has to divide the hidden chunk count, since the reduce has no tail path
    along hidden, and 2*topk, so the tile's T*topk rows spread evenly over the
    block's T*C/2 warps.

    T then takes the tile as high as the LDS budget allows, since trips per block
    is what keeps the prefetch pipeline fed. LDS grows as T*topk, so a large topk
    trades height away: topk=8 on the mxfp8 wire needs 164KB at the preferred
    T=8 and lands on T=7.
    """
    n_chunks = hidden_dim // _COMBINE_CHUNK_ELEMS
    max_chunks, max_toks = (
        (_COMBINE_QUANT_CHUNKS_PER_ITER, _COMBINE_QUANT_TOKENS_PER_BLOCK)
        if quant_bits
        else (_COMBINE_CHUNKS_PER_ITER, _COMBINE_TOKENS_PER_BLOCK)
    )
    for chunks in range(max_chunks, 0, -1):
        if n_chunks % chunks or (2 * topk) % chunks:
            continue
        for toks in range(max_toks, 0, -1):
            lanes = toks * chunks * _COMBINE_CHUNK_ELEMS // 16
            if lanes % _WAVE_SIZE:
                continue
            # The tile's copies pass the block's warp count to a TDM atom, which
            # takes only a power of two.
            warps = lanes // _WAVE_SIZE
            if warps & (warps - 1):
                continue
            if (
                combine_reduce_lds_bytes(
                    experts_per_token=topk,
                    quant_bits=quant_bits,
                    tokens_per_block=toks,
                    chunks_per_iter=chunks,
                )
                <= COMBINE_LDS_BUDGET
            ):
                return toks, chunks
    raise ValueError(
        f"no combine reduce tile fits the {COMBINE_LDS_BUDGET // 1024}KB LDS "
        f"budget for topk={topk}, hidden_dim={hidden_dim}, "
        f"combine_quant_bits={quant_bits}"
    )


class _FusedScatterCombine:
    """The gemm2 epilogue P2P-scatters each row into the slot of the peer its
    token came from, packed as the step's wire; after a cross-device barrier a
    FlyDSL reduce sums each token's slots."""

    wires = tuple(_COMBINE_WIRE_QUANT)
    accepted_layouts = frozenset(BatchLayout)
    uses_mori = False
    needs_aggregated_reservation = False

    def __init__(self, config: MegaMoEConfig, caps=None):
        self.config = config
        self.res = None
        self._tiles = {
            bits: _combine_tile(
                topk=config.topk, hidden_dim=config.hidden_dim, quant_bits=bits
            )
            for bits in config.combine_quant_bit_modes
        }

    @classmethod
    def check(cls, config: MegaMoEConfig) -> None:
        # Every wire, not just the quantized ones: the reduce tiles hidden in
        # whole chunks and has no tail path.
        if config.hidden_dim % _COMBINE_CHUNK_ELEMS:
            raise ValueError(
                f"the combine reduce tiles hidden in {_COMBINE_CHUNK_ELEMS}-"
                f"element chunks with no tail path, so it needs hidden_dim % "
                f"{_COMBINE_CHUNK_ELEMS} == 0, got {config.hidden_dim}"
            )

    @classmethod
    def regions(cls, config: MegaMoEConfig, caps=None) -> dict:
        return {
            "cross_device_barrier": config.world_size * 8,
            # Cut to the bf16 pitch whatever wire runs: it is the widest, so this
            # holds a step on any of them and a quantized one just packs into the
            # front of it at its own pitch.
            "comb_inp": (
                config.max_tokens_per_rank
                * config.topk
                * config.combine_slot_stride_bytes
            ),
        }

    def build(self, res: _EpResources) -> None:
        self.res = res
        config = self.config
        self._cross_device_flag = torch.ones(
            1, dtype=torch.int64, device=res.total_recv.device
        )
        # Keep the cross-device barrier in its own 1-block kernel so the reduce
        # grid is unconstrained. Both wires stage through LDS, so the block is
        # sized to the lane tile (T*C*256/16 lanes) -- which differs per wire,
        # hence one build per width rather than one shared spec.
        self._reduces = {}
        for bits, (toks, chunks) in self._tiles.items():
            lanes = toks * chunks * _COMBINE_CHUNK_ELEMS // 16
            self._reduces[bits] = _make_combine_fused_reduce(
                experts_per_token=config.topk,
                hidden_dim=config.hidden_dim,
                block_num=512,
                warp_num_per_block=lanes // _WAVE_SIZE,
                slot_stride_nbytes=config.combine_slot_stride_for(bits),
                quant_bits=bits,
                tokens_per_block=toks,
                chunks_per_iter=chunks,
            )
        self._sync = _make_combine_fused_sync(
            rank=config.rank,
            npes=config.world_size,
            off_xdb_mem=res.arena.offset("cross_device_barrier"),
        )

    def prepare(self, batch: _DispatchBatch, quant_bits: int) -> _ComputeTarget:
        config, arena = self.config, self.res.arena
        grouped = batch.layout is BatchLayout.EXPERT_GROUPED
        scatter = Stage2ScatterContext(
            arena_handle=arena.handle,
            combine_input_offset=arena.offset("comb_inp"),
            slot_stride_bytes=config.combine_slot_stride_for(quant_bits),
            max_tokens_per_rank=config.max_tokens_per_rank,
            world_size=config.world_size,
            source_token_map=batch.routing.source_token_map,
            **(batch.grouping.scatter_fields() if grouped else {}),
            combine_quant_bits=quant_bits,
        )
        return _ComputeTarget(
            scatter=scatter, dispatch_context=scatter if grouped else None
        )

    def finish(
        self, routing: Routing, quant_bits: int, expert_out=None
    ) -> torch.Tensor:
        """The gemm2 epilogue has already scattered the results into the peers'
        slots, packed as the ``quant_bits`` wire; reduce them."""
        res = self.res
        stream = fx.Stream(torch.cuda.current_stream())
        # 1-block cross-device barrier, then the barrier-free reduce on the
        # same stream; the kernel boundary gives the reduce its visibility.
        _run_compiled(
            self._sync,
            res.arena.handle,
            self._cross_device_flag.data_ptr(),
            self.config.rank,
            stream,
        )
        _run_compiled(
            self._reduces[quant_bits],
            res.arena.local_ptr("comb_inp"),
            res.combine_output.data_ptr(),
            routing.token_count,
            stream,
        )
        return res.output(routing.token_count)

    def close(self) -> None:
        """Nothing to release: its kernels and buffers are not held by mori."""
