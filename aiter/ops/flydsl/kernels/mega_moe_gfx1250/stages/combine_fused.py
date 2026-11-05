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


def _combine_tile(*, topk, hidden_dim, quant_bits, ll128=False):
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
                    ll128=ll128,
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
        self._ll128_tile = (
            _combine_tile(
                topk=config.topk,
                hidden_dim=config.hidden_dim,
                quant_bits=0,
                ll128=True,
            )
            if config.combine_ll128
            else None
        )

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
        self._reduce_ll128 = None
        if self._ll128_tile is not None:
            toks, chunks = self._ll128_tile
            lanes = toks * chunks * _COMBINE_CHUNK_ELEMS // 16
            self._reduce_ll128 = _make_combine_fused_reduce(
                experts_per_token=config.topk,
                hidden_dim=config.hidden_dim,
                block_num=512,
                warp_num_per_block=lanes // _WAVE_SIZE,
                slot_stride_nbytes=config.combine_slot_stride_for(0),
                quant_bits=0,
                tokens_per_block=toks,
                chunks_per_iter=chunks,
                ll128=True,
                n_experts_total=config.world_size * config.experts_per_rank,
            )
        self._sync = _make_combine_fused_sync(
            rank=config.rank,
            npes=config.world_size,
            off_xdb_mem=res.arena.offset("cross_device_barrier"),
        )
        # S2F: the fused kernels by gemm2 configuration, their control block,
        # and this step's routing (set by prepare, read by the gemm2 stand-in).
        self._s2f_launches = {}
        self._s2f_routing = None
        self._s2f_ran = False
        # This step's LL128 verdict, from prepare to finish.
        self._ll128_now = False
        self._s2f_ctl = None
        if config.s2f:
            from ..s2f_stage2 import CTL_WORDS

            self._s2f_ctl = torch.zeros(
                CTL_WORDS, dtype=torch.int32, device=res.total_recv.device
            )

    def _s2f_gemm2(self, out, a, w, a_scales, w_scales, m_tile_map, **kw):
        """gemm2 stand-in (``flydsl_grouped_gemm_a8w4_masked``'s signature): runs
        the S2F kernel, gemm2 and this step's combine reduce in one launch, or
        plain gemm2 when its configuration is one S2F does not embed."""
        from .... import grouped_gemm_mxfp4 as gg
        from ...tensor_shim import ptr_arg
        from ..s2f_stage2 import make_s2f_stage2

        config, res = self.config, self.res
        tile_m, tile_n, tile_k = kw["tile_m"], kw["tile_n"], kw["tile_k"]
        K, N = kw["K"], kw["N"]
        n_tiles = (N + tile_n - 1) // tile_n
        cluster_n = gg._select_cluster_n(n_tiles, kw.get("cluster_n", -1))
        cluster_m = gg._select_cluster_m(kw.get("cluster_m", -1), cluster_n, 0)
        scatter = kw["stage2_scatter"]
        if (
            cluster_n != 1
            or cluster_m != 1
            or kw.get("bias") is not None
            or kw.get("a_preshuffled")
            or kw.get("stage1_act")
            or not scatter.combine_ll128
        ):
            self._s2f_ran = False
            return gg.flydsl_grouped_gemm_a8w4_masked(
                out, a, w, a_scales, w_scales, m_tile_map, **kw
            )
        k_tiles = (K + tile_k - 1) // tile_k
        gemm = {
            "K": K,
            "tile_m": tile_m,
            "tile_n": tile_n,
            "tile_k": tile_k,
            "m_warp": kw.get("m_warp", 1),
            "n_warp": kw.get("n_warp", 4),
            "out_is_f16": kw.get("out_is_f16", 0),
            "num_buffers": min(kw.get("num_buffers", 3), max(1, k_tiles)),
            "a_is_fp4": kw.get("a_is_fp4", 0),
            "n_experts": kw["n_experts"],
            "stage1_act": 0,
            "has_bias": 0,
            "stage1_quant_out": 0,
            "quant_wmma_rep": 1,
            "cluster_n": 1,
            "cluster_m": 1,
            "next_stage_prefetch": gg._select_bool_env(
                "AITER_TDM_NEXT_STAGE_PREFETCH", kw.get("next_stage_prefetch", 0)
            ),
            "num_waves_per_tensor_tdm": gg._select_num_waves_per_tensor_tdm(
                kw.get("waves_per_tensor_tdm", -1)
            ),
            "tdm_as_in_prologue": gg._select_bool_env(
                "AITER_GROUPED_GEMM_AS_PROLOGUE", kw.get("tdm_as_in_prologue", 0)
            ),
            "tdm_b_th": gg._select_tdm_b_th(kw.get("tdm_b_th", 0)),
            "enable_ep_scatter": 1,
            "ep_arena_handle": int(scatter.arena_handle),
            "ep_combine_input_offset": int(scatter.combine_input_offset),
            "ep_slot_stride_bytes": int(scatter.slot_stride_bytes),
            "ep_destination_stride": int(kw["ep_destination_stride"]),
            "ep_world_size": int(scatter.world_size),
            "ep_quant_bits": 0,
            "row_major_ascale": int(kw.get("row_major_ascale", 0)),
            "a_row_stride_bytes": int(kw.get("a_row_stride_bytes", 0)),
            "a_scale_row_stride_bytes": int(kw.get("a_scale_row_stride_bytes", 0)),
            "lds_soa_load_interleave": int(kw.get("lds_soa_load_interleave", 0)),
            "rolled_drain": int(kw.get("rolled_drain", 0)),
            "a_preshuffled": 0,
            "m_align": int(kw.get("m_align", 0)),
        }
        key = tuple(sorted(gemm.items()))
        launch = self._s2f_launches.get(key)
        if launch is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "S2F kernel for this gemm2 tile was not compiled before graph "
                    "capture; warm this token bucket eagerly first"
                )
            launch = make_s2f_stage2(
                gemm=gemm,
                topk=config.topk,
                hidden_dim=config.hidden_dim,
                slot_stride=config.combine_slot_stride_for(0),
                n_experts_total=config.world_size * config.experts_per_rank,
            )
            self._s2f_launches[key] = launch
        routing = self._s2f_routing
        row_map = kw["ep_row_map"]
        _run_compiled(
            launch,
            gg._launch_address(out, out.dtype),
            ptr_arg(a),
            ptr_arg(w),
            gg._launch_address(a_scales, torch.int32),
            gg._launch_address(w_scales, torch.int32),
            ptr_arg(m_tile_map),
            ptr_arg(a),
            gg._launch_address(row_map, row_map.dtype),
            res.arena.local_ptr("comb_inp"),
            res.combine_output.data_ptr(),
            routing.topk_ids.data_ptr(),
            m_tile_map.data_ptr(),
            self._s2f_ctl.data_ptr(),
            int(kw["contiguous_m"]),
            routing.token_count,
            fx.Stream(torch.cuda.current_stream()),
        )
        self._s2f_ran = True
        return out

    def ll128_for(self, recv_bound: int, quant_bits: int) -> bool:
        """Whether a step of this recv-token bound runs LL128: the bf16 wire,
        below the tokens-a-rank threshold. The bound is the caller's, the same
        on every rank, so every rank's gemm2 and reduce agree on the layout.

        An LL128 step has no combine barrier, so the next layer's plan has to
        come after this rank's reduce: the caller must not prefetch it."""
        if self._reduce_ll128 is None or quant_bits:
            return False
        tpr = -(-int(recv_bound) // self.config.world_size)
        return tpr < self.config.ll128_tpr_lt

    def _ll128_step(self, batch: _DispatchBatch, quant_bits: int) -> bool:
        """An LL128 step also needs the compact plan's generation word."""
        if batch.layout is not BatchLayout.EXPERT_GROUPED or batch.routing.gen is None:
            return False
        return self.ll128_for(batch.grouping.recv_bound, quant_bits)

    def prepare(self, batch: _DispatchBatch, quant_bits: int) -> _ComputeTarget:
        config, arena = self.config, self.res.arena
        grouped = batch.layout is BatchLayout.EXPERT_GROUPED
        ll128 = self._ll128_step(batch, quant_bits)
        s2f = ll128 and config.s2f
        self._ll128_now = ll128
        self._s2f_routing = batch.routing if s2f else None
        self._s2f_ran = False
        scatter = Stage2ScatterContext(
            arena_handle=arena.handle,
            combine_input_offset=arena.offset("comb_inp"),
            slot_stride_bytes=config.combine_slot_stride_for(quant_bits),
            max_tokens_per_rank=config.max_tokens_per_rank,
            world_size=config.world_size,
            source_token_map=batch.routing.source_token_map,
            **(batch.grouping.scatter_fields() if grouped else {}),
            combine_quant_bits=quant_bits,
            combine_ll128=ll128,
            # Every live gemm2 tile in one round: worth it in the same small-batch
            # regime, and the persistent EP scatter is bf16-only anyway.
            gemm2_persistent_workers=-1 if ll128 else 0,
            stage2_gemm2=self._s2f_gemm2 if s2f else None,
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
        if self._s2f_ran:
            # The S2F kernel has reduced this step already.
            self._s2f_ran = False
            return res.output(routing.token_count)
        if self._ll128_now:
            self._ll128_now = False
            # Every line carries its writer's generation: the reduce waits per
            # tile, so neither the barrier nor its kernel boundary is needed.
            _run_compiled(
                self._reduce_ll128,
                res.arena.local_ptr("comb_inp"),
                res.combine_output.data_ptr(),
                routing.token_count,
                routing.topk_ids.data_ptr(),
                routing.gen.data_ptr(),
                stream,
            )
            return res.output(routing.token_count)
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
            0,
            0,
            stream,
        )
        return res.output(routing.token_count)

    def close(self) -> None:
        """Nothing to release: its kernels and buffers are not held by mori."""
