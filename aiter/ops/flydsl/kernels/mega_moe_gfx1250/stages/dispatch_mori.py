# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""mori's EpDispatchPlan as a dispatch stage."""

import torch

from ..mori_caps import _MoriCapabilities
from ..recipe import _DISPATCH_WIRES, MegaMoEConfig
from ..types import _from_gpu_ptr
from .base import (
    BatchLayout,
    _check_recv_token_bound,
    _close_all,
    _Dispatch,
    _DispatchBatch,
    _EpResources,
    _pick,
    _schedule_specs,
)


class _MoriDispatch(_Dispatch):
    """mori's EpDispatchPlan: token-major recv rows.

    The routing regions are laid out as on the compact path (out_idx/out_wts at
    slot*topk+k, recv_to_src_token as src_pe*max_tok+src_tok), and mori never
    touches cross_device_barrier. Recv slots are reserved one atomic per block
    and handed out block-local.
    """

    wires = _DISPATCH_WIRES
    output_layout = BatchLayout.TOKEN_MAJOR
    uses_mori = True

    def __init__(self, config: MegaMoEConfig, caps: _MoriCapabilities):
        # mori's geometry is a compile-time Cfg field, so it brings its own
        # tuned buckets.
        super().__init__(config, caps.dispatch_schedule(config))
        self.caps = caps
        # Where mori's slot word lives decides how the plans reserve slots:
        # outside the window, there is no aggregated reservation.
        self.tokoff_ext = caps.tokoff_ext_wanted(config)
        self.provides_aggregated_reservation = not self.tokoff_ext
        infos = self.describe(config, caps, self.specs)
        # One staging buffer serves every plan, as only one launches at a time.
        self._staging_nbytes = max(info["stagingBytes"] for info in infos)
        self.scale_stride = infos[0]["scaleStride"]
        self._staging = None
        self._plans = {}

    @staticmethod
    def request(config: MegaMoEConfig, spec: tuple, tokoff_ext=False) -> dict:
        """The plan request for ``spec``: what describe() reads, and what the
        plan is built from once the arena is added."""
        return {
            "world_size": config.world_size,
            # see dispatch_wire_elem_count; mori's plan_api: "the caller halves
            # hiddenDim"
            "hidden_dim": config.dispatch_wire_elem_count,
            "max_tok_per_rank": config.max_tokens_per_rank,
            "num_expert_per_rank": config.experts_per_rank,
            "num_expert_per_token": config.topk,
            "max_recv": config.max_recv,
            "dtype": config.dispatch_wire_spec.mori_dtype,
            "use_weights": True,
            "scale_bytes": config.dispatch_scale_nbytes,
            "tok_off_ext": int(tokoff_ext),
            "block_num": spec[0],
            "warp_per_block": spec[1],
        }

    @staticmethod
    def describe(config: MegaMoEConfig, caps, specs=None) -> list[dict]:
        """mori's info for every plan this dispatch builds.

        Without tokoff-ext: where the slot word lives sizes nothing, and asking
        would be collective.
        """
        if specs is None:
            specs = _schedule_specs(caps.dispatch_schedule(config))
        return caps.describe_dispatch(
            [_MoriDispatch.request(config, spec) for spec in specs]
        )

    @classmethod
    def check(cls, config: MegaMoEConfig) -> None:
        """Nothing beyond what mori's describe() refuses."""

    @classmethod
    def regions(cls, config: MegaMoEConfig, caps) -> dict:
        return caps.region_sizes(cls.describe(config, caps))

    def build(self, res: _EpResources) -> None:
        super().build(res)
        config, arena = self.config, res.arena
        device = res.total_recv.device
        if self.tokoff_ext:
            res.tokoff_ext = self.caps.tokoff_ext(config, device)
        self._staging = torch.zeros(
            self._staging_nbytes, dtype=torch.uint8, device=device
        )
        for spec in self.specs:
            plan = self.caps.dispatch_plan(
                self.request(config, spec, self.tokoff_ext), arena
            )
            # The kernels dereference the window, so the plans have to outlive
            # them; held from here so a failed bind still closes it.
            self._plans[spec] = plan
            plan.bind(rank=config.rank)
            # The tensor, not its address: mori checks it against the plan's
            # stagingBytes.
            plan.bind(staging_base=self._staging)
        self._tok_off_peers = None if res.tokoff_ext is None else res.tokoff_ext.peers
        # Width in whatever recv_dtype counts: features for bf16/fp8, bytes for
        # fp4 -- see _DispatchWire.recv_dtype.
        width = (
            config.dispatch_token_nbytes
            // config.dispatch_wire_spec.recv_dtype.itemsize
        )
        self.recv_x = _from_gpu_ptr(
            arena.local_ptr("disp_out"),
            (config.max_recv, width),
            config.dispatch_wire_spec.recv_dtype,
        )
        self.recv_scales = (
            _from_gpu_ptr(
                arena.local_ptr("disp_out_scales"),
                (config.max_recv, self.scale_stride),
                torch.uint8,
            )
            if config.is_quant_dispatch_wire
            else None
        )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        recv_token_bound: int | None = None,
        next_topk_ids: torch.Tensor | None = None,
        next_recv_token_bound: int | None = None,
        begin_step: bool = True,
    ) -> _DispatchBatch:
        """``recv_token_bound`` cuts the recv rows the GEMM is shown. There is no
        plan to run ahead, so the next layer's ids go unused."""
        config, res = self.config, self.res
        token_count = hidden_states.shape[0]
        _check_recv_token_bound(config, "recv_token_bound", recv_token_bound)
        _check_recv_token_bound(config, "next_recv_token_bound", next_recv_token_bound)
        plan = self._plans[_pick(self.schedule, token_count)]
        payload, scales_ptr = self._quantize(hidden_states, token_count)
        # No total_recv.zero_(): the dispatch overwrites it (see
        # _MoriCapabilities).
        plan.launch(
            stream=torch.cuda.current_stream().cuda_stream,
            token_indices=topk_ids.data_ptr(),
            inp_token_buf=payload.data_ptr(),
            weights_buf=topk_weights.data_ptr(),
            disp_dest_tok_id_map=res.token_destination_map.data_ptr(),
            dest_pe_token_counter=res.destination_peer_counter.data_ptr(),
            total_recv_token_num=res.total_recv.data_ptr(),
            grid_barrier=res.dispatch_barrier.data_ptr(),
            tok_off_peers=self._tok_off_peers,
            num_tokens=token_count,
            scales_buf=scales_ptr,
        )
        x, weights, ids, scales = (
            self.recv_x,
            self.recv_weights,
            self.recv_ids,
            self.recv_scales,
        )
        if recv_token_bound is not None:
            bound = int(recv_token_bound)
            x, weights, ids = x[:bound], weights[:bound], ids[:bound]
            if scales is not None:
                scales = scales[:bound]
        return _DispatchBatch(
            x=x,
            topk_weights=weights,
            topk_ids=ids,
            scales=scales,
            num_local_tokens=res.total_recv,
            routing=self._routing(token_count),
            layout=BatchLayout.TOKEN_MAJOR,
        )

    def prefetch(self, topk_ids, recv_token_bound=None) -> None:
        """Nothing to run ahead: the plan is the dispatch."""
        _check_recv_token_bound(self.config, "recv_token_bound", recv_token_bound)

    def warmup(self, recv_token_bound=None, token_count=None) -> None:
        """Nothing to compile ahead: every plan was built with the arena."""
        _check_recv_token_bound(self.config, "recv_token_bound", recv_token_bound)

    def close(self) -> None:
        plans, self._plans = self._plans, {}
        _close_all(plans.values())
        # After the plans, which hold its address.
        self._staging = None
