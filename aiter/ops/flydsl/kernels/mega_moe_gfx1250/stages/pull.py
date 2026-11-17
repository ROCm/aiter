# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc.

"""Non-S1F plan/quant and pull dispatch resources for the compact stage."""

import os

import flydsl.expr as fx
import torch

from ...moe_apreshuffled_quant_scatter import validate_apre_k
from ...tensor_shim import _preload_compiled, _run_compiled
from ..compact_plan import compact_plan_blocks, compile_tdm_compact_plan
from ..dispatch_pull import compile_pull_dispatch
from ..s1f_ctl import _QUANT_FUSE_BLOCKS
from .base import BatchLayout, ExpertGrouping, _DispatchBatch


def enabled(config):
    """An explicit opt-in; S1F's actual per-step verdict takes precedence."""
    return os.environ.get("AITER_MEGA_DISPATCH_PULL", "0") == "1"


def check(config):
    """Reject unsupported pull wire/layout combinations before allocation."""
    if not enabled(config):
        return
    if config.dispatch_wire != "fp4":
        raise ValueError("AITER_MEGA_DISPATCH_PULL requires FP4 dispatch")
    validate_apre_k(config.hidden_dim)
    if config.geometry.inter_dim % 128:
        raise ValueError("pull preshuffle requires intermediate % 128 == 0")
    if _QUANT_FUSE_BLOCKS <= 0:
        raise ValueError("pull requires AITER_MEGA_QUANT_FUSE > 0")


def regions(config, geometry):
    """Extra symmetric source storage; destination rowmap is shared with push."""
    if not enabled(config):
        return {}
    return {
        "pull_wire": config.max_tokens_per_rank * geometry.wire_row,
        "pull_hist": geometry.hist_stride * 4,
        "pull_done": 4,
        "pull_ready": 4,
    }


class PullDispatch:
    """Reuse the existing planner and supply its result to receiver-side pull."""

    def __init__(self, dispatch):
        self.dispatch = dispatch
        config, geo = dispatch.config, dispatch.compact
        self.config = config
        self.tdm_gather = os.environ.get("AITER_MEGA_PULL_TDM_GATHER", "0") == "1"
        self.arena = dispatch.res.arena
        device = dispatch.res.total_recv.device
        routes = geo.max_routes
        segments = config.world_size * config.experts_per_rank
        self.tok_map = torch.empty(routes, dtype=torch.int32, device=device)
        self.block_hist = torch.empty(
            16 * segments, dtype=torch.int32, device=device
        )
        self.send_base = torch.empty(segments, dtype=torch.int32, device=device)
        self.masked_m = torch.empty(
            config.experts_per_rank, dtype=torch.int32, device=device
        )
        self.psum = torch.empty_like(self.masked_m)
        self.barrier = torch.zeros(4, dtype=torch.int32, device=device)
        self.payload = torch.empty(
            (geo.cap, config.hidden_dim // 2), dtype=torch.uint8, device=device
        )
        self.scales = torch.empty(
            (geo.cap, config.hidden_dim // 32), dtype=torch.uint8, device=device
        )
        self.wire_ptr = self.arena.local_ptr("pull_wire")
        self._launches = {}

    def _plan_args(self, ids, hidden, weights, tokens, stream):
        return (
            self.arena.handle,
            ids,
            self.tok_map.data_ptr(),
            self.block_hist.data_ptr(),
            self.send_base.data_ptr(),
            self.masked_m.data_ptr(),
            self.psum.data_ptr(),
            self.barrier.data_ptr(),
            self.config.rank,
            tokens,
            hidden,
            self.wire_ptr,
            weights,
            stream,
        )

    def _pull_args(self, rows, stream):
        return (
            self.arena.handle,
            self.psum,
            self.barrier,
            self.payload.view(torch.int32).reshape(-1),
            self.scales.view(torch.int32).reshape(-1),
            rows,
            stream,
        )

    def warmup(self, step, token_count):
        """Compile this alignment before graph capture, without GPU launches."""
        key = (
            step.align_m,
            compact_plan_blocks(token_count * self.config.topk),
        )
        if key in self._launches:
            return self._launches[key]
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "warm the pull dispatch token bucket before capture"
            )
        config, geo, arena = self.config, self.dispatch.compact, self.arena
        plan = compile_tdm_compact_plan(
            rank=config.rank,
            npes=config.world_size,
            experts_per_rank=config.experts_per_rank,
            topk=config.topk,
            tile_m=step.align_m,
            compact_cap=geo.cap,
            off_hist=arena.offset("pull_hist"),
            off_done=arena.offset("pull_done"),
            hist_stride=geo.hist_stride,
            max_routes=geo.max_routes,
            hist_pingpong=False,
            plan_blocks=key[1],
            quant_blocks=_QUANT_FUSE_BLOCKS,
            quant_model_dim=config.hidden_dim,
            quant_row_stride=geo.wire_row,
            quant_payload_bytes=config.dispatch_token_nbytes,
            pull=True,
            off_pull_map=arena.offset("ep_rowmap"),
        )
        pull = compile_pull_dispatch(
            rank=config.rank,
            world=config.world_size,
            experts=config.experts_per_rank,
            max_routes=geo.max_routes,
            model_dim=config.hidden_dim,
            wire_row=geo.wire_row,
            off_wire=arena.offset("pull_wire"),
            off_ready=arena.offset("pull_ready"),
            off_rowmap=arena.offset("ep_rowmap"),
            align_m=step.align_m,
            topk=config.topk,
            tdm_gather=self.tdm_gather,
        )
        stream = fx.Stream(torch.cuda.current_stream())
        _preload_compiled(plan, *self._plan_args(0, 0, 0, token_count, stream))
        _preload_compiled(pull, *self._pull_args(step.rows, stream))
        self._launches[key] = (plan, pull)
        return plan, pull

    def run(self, hidden, weights, ids):
        """Plan/quant on the plan stream, then pull on the caller's stream."""
        dispatch, step = self.dispatch, self.dispatch._step
        tokens = int(hidden.shape[0])
        plan, pull = self.warmup(step, tokens)
        # A previously prefetched ordinary plan cannot supply quantized inputs.
        # Join it before reusing ep_rowmap; future non-S1F prefetch is suppressed.
        dispatch._wait_plan()
        plan_stream = dispatch._plan_stream
        plan_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(plan_stream):
            _run_compiled(
                plan,
                *self._plan_args(
                    ids.data_ptr(),
                    hidden.data_ptr(),
                    weights.data_ptr(),
                    tokens,
                    fx.Stream(plan_stream),
                ),
            )
        dispatch._plan_event.record(plan_stream)
        torch.cuda.current_stream().wait_event(dispatch._plan_event)
        _run_compiled(
            pull,
            *self._pull_args(step.rows, fx.Stream(torch.cuda.current_stream())),
        )
        return _DispatchBatch(
            x=self.payload[: step.rows],
            topk_weights=dispatch._dummy_wts,
            topk_ids=dispatch._dummy_ids,
            scales=self.scales[: step.rows],
            num_local_tokens=None,
            routing=dispatch._routing(tokens),
            layout=BatchLayout.EXPERT_GROUPED,
            grouping=ExpertGrouping(
                masked_m=self.masked_m,
                psum=self.psum,
                ep_rowmap=dispatch._ep_rowmap,
                wire_row_stride=0,
                recv_bound=step.recv_bound,
                align_m=step.align_m,
                rows=step.rows,
                a_preshuffled=True,
            ),
        )
