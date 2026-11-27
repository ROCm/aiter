# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""mori's EpCombinePlan as a combine stage."""

import torch

from ..mori_caps import _MoriCapabilities
from ..recipe import MegaMoEConfig
from ..types import _from_gpu_ptr
from .base import (
    BatchLayout,
    Routing,
    _close_all,
    _ComputeTarget,
    _DispatchBatch,
    _EpResources,
    _pick,
    _schedule_specs,
)

# The fp4 push combine reduces 32 elements a lane over a 32-lane wave.
_MORI_FP4_COMBINE_HIDDEN_ALIGN = 1024


class _MoriCombine:
    """mori's EpCombinePlan over fused_moe's per-recv-row output.

    bf16 gathers: fused_moe writes its rows straight into this rank's comb_inp
    staging rows, the combine meets every peer on cross_device_barrier, then
    pulls each token's rows back from the peers the dispatch's tok_map names and
    sums them. MXFP4 pushes instead: each rank quantizes its rows into the
    landing rows of the peer each came from, and after the barrier dequantizes
    and sums what landed for its own tokens. Both zero total_recv on the way out.
    """

    wires = ("bf16", "fp4")
    accepted_layouts = frozenset({BatchLayout.TOKEN_MAJOR})
    uses_mori = True

    def __init__(self, config: MegaMoEConfig, caps: _MoriCapabilities):
        self.config = config
        self.caps = caps
        self.res = None
        self._schedules = {
            bits: caps.combine_schedule(config, bits)
            for bits in config.combine_quant_bit_modes
        }
        self._specs = {
            bits: _schedule_specs(schedule)
            for bits, schedule in self._schedules.items()
        }
        self._max_blocks = max(
            block for wire in self._specs.values() for block, _ in wire
        )
        slots = caps.xdb_flag_slots
        if self._max_blocks > slots:
            raise ValueError(
                f"mori combine block_num {self._max_blocks} exceeds the "
                f"{slots} per-block epoch slots its entry barrier owns"
            )
        # The fp4 push combine reads its landing offsets off the reservation.
        self.needs_aggregated_reservation = bool(config.combine_quant_bits)
        self._plans = {}

    @classmethod
    def check(cls, config: MegaMoEConfig) -> None:
        if config.combine_quant_bits and (
            config.hidden_dim % _MORI_FP4_COMBINE_HIDDEN_ALIGN
        ):
            raise ValueError(
                "mori's fp4 combine needs hidden_dim % "
                f"{_MORI_FP4_COMBINE_HIDDEN_ALIGN} == 0, got {config.hidden_dim}"
            )

    @staticmethod
    def request(config: MegaMoEConfig, quant_bits: int, spec: tuple) -> dict:
        """The plan request for ``spec`` on the ``quant_bits`` wire."""
        return {
            "world_size": config.world_size,
            "hidden_dim": config.hidden_dim,
            "max_tok_per_rank": config.max_tokens_per_rank,
            "num_expert_per_rank": config.experts_per_rank,
            "num_expert_per_token": config.topk,
            "max_recv": config.max_recv,
            "dtype": torch.bfloat16,
            "use_weights": True,
            "combine_fp4": bool(quant_bits),
            "block_num": spec[0],
            "warp_per_block": spec[1],
        }

    @classmethod
    def regions(cls, config: MegaMoEConfig, caps) -> dict:
        requests = [
            cls.request(config, bits, spec)
            for bits in config.combine_quant_bit_modes
            for spec in _schedule_specs(caps.combine_schedule(config, bits))
        ]
        return caps.region_sizes(caps.describe_combine(requests))

    def build(self, res: _EpResources) -> None:
        self.res = res
        config = self.config
        device = res.total_recv.device
        for bits, wire_specs in self._specs.items():
            for spec in wire_specs:
                plan = self.caps.combine_plan(
                    self.request(config, bits, spec), res.arena
                )
                self._plans[bits, spec] = plan
                plan.bind(rank=config.rank)

        # One epoch per block: each block is the only writer of its own slot.
        # Starts at 1 so the zeroed peer slots cannot alias the first launch.
        self._cross_device_flag = torch.ones(
            self.caps.xdb_flag_slots, dtype=torch.int64, device=device
        )
        self._grid_barrier = torch.zeros(1, dtype=torch.uint32, device=device)
        # The entry barrier's intra-grid fan-out: 16 lines per block.
        self._barrier_fan = torch.zeros(
            self._max_blocks * 16, dtype=torch.int32, device=device
        )
        self._static_args = {
            "out_token_buf": res.combine_output.data_ptr(),
            "out_weights_buf": None,
            "disp_dest_tok_id_map": res.token_destination_map.data_ptr(),
            # The push combine's landing offsets, which the dispatch wrote.
            "dest_pe_token_counter": res.destination_peer_counter.data_ptr(),
            "total_recv_token_num": res.total_recv.data_ptr(),
            "grid_barrier": self._grid_barrier.data_ptr(),
            "xdb_flag": self._cross_device_flag.data_ptr(),
            "combine_barrier_fan": self._barrier_fan.data_ptr(),
        }
        # The bf16 staging rows at the front of comb_inp. fused_moe writes its
        # rows straight into them, so the bf16 combine finds its input already
        # staged and skips its copy, and the fp4 send, which reads ahead as far
        # as row max_recv-1, stays inside the arena.
        self._stage = _from_gpu_ptr(
            res.arena.local_ptr("comb_inp"),
            (config.max_recv, config.hidden_dim),
            torch.bfloat16,
        )

    def prepare(self, batch: _DispatchBatch, quant_bits: int) -> _ComputeTarget:
        """fused_moe writes its rows straight into this rank's comb_inp staging.

        Overwriting the previous step's rows there is safe because of the
        dispatch, not the combine: peers read those rows during their combine,
        and this step's dispatch, which the GEMM waits for, does not finish on
        this rank until every peer has dispatched to it -- which each peer does
        only after its own previous combine, in stream order. A dispatch that
        let a rank run ahead of a peer's combine would break this.
        """
        return _ComputeTarget(output=self._stage[: batch.x.shape[0]])

    def finish(
        self, routing: Routing, quant_bits: int, expert_out: torch.Tensor
    ) -> torch.Tensor:
        """``expert_out`` is fused_moe's per-recv-slot output, already in the
        staging rows; mori moves it from there."""
        count = routing.token_count
        plan = self._plans[quant_bits, _pick(self._schedules[quant_bits], count)]
        plan.launch(
            stream=torch.cuda.current_stream().cuda_stream,
            inp_token_buf=expert_out,
            num_tokens=count,
            **self._static_args,
        )
        return self.res.output(count)

    def close(self) -> None:
        plans, self._plans = self._plans, {}
        _close_all(plans.values())
