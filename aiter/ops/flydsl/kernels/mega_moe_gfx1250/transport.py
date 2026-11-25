# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The recipe's dispatch and combine stages over one shared arena."""

import copy
import dataclasses

import torch

from .arena import SymmetricArena, _ArenaLayout
from .mori_caps import _mori_capabilities, _MoriCapabilities
from .recipe import DispatchKind, MegaMoEConfig, MegaMoERecipe, _stage_classes
from .s1f_ctl import read_s1f_fallback
from .stages.base import (
    CombineStage,
    DispatchStage,
    Routing,
    _close_all,
    _ComputeTarget,
    _DispatchBatch,
    _EpResources,
)


def _capabilities_for(recipe: MegaMoERecipe) -> _MoriCapabilities | None:
    """mori's capabilities if a stage ``recipe`` names drives mori, else None."""
    if any(stage.uses_mori for stage in _stage_classes(recipe)):
        return _mori_capabilities()
    return None


def _fallback_config(config: MegaMoEConfig) -> MegaMoEConfig | None:
    """The mori-dispatch config a compact recipe's non-fused steps run on
    (``AITER_MEGA_S1F_FALLBACK=mori``), else None."""
    if config.recipe.dispatch is not DispatchKind.COMPACT:
        return None
    if read_s1f_fallback() != "mori":
        return None
    recipe = dataclasses.replace(config.recipe, dispatch=DispatchKind.MORI)
    return MegaMoEConfig(recipe=recipe, geometry=config.geometry)


def _arena_layout(config: MegaMoEConfig, caps) -> _ArenaLayout:
    """The one arena formula: what the runtime allocates and what
    ``required_vmm_bytes`` reports."""
    dispatch, combine = _stage_classes(config.recipe)
    requests = [dispatch.regions(config, caps), combine.regions(config, caps)]
    fallback = _fallback_config(config)
    if fallback is not None:
        from .stages.dispatch_mori import _MoriDispatch

        mori_caps = _mori_capabilities()
        requests.append(_MoriDispatch.regions(fallback, mori_caps))
        requests.append(combine.regions(fallback, mori_caps))
    return _ArenaLayout.merge(*requests)


class _EpDispatcher:
    """The recipe's dispatch and combine over one shared arena.

    Everything that can fail, and the host collectives, run before the arena --
    itself collective -- is allocated; a failure after it frees what was built.
    """

    def __init__(self, config: MegaMoEConfig, communicator):
        self.config = config
        self.device = torch.device("cuda", torch.cuda.current_device())
        self._closed = False
        recipe = config.recipe
        dispatch_cls, combine_cls = _stage_classes(recipe)
        caps = _capabilities_for(recipe)
        self._dispatch: DispatchStage = dispatch_cls(config, caps)
        self._combine: CombineStage = combine_cls(config, caps)
        # mori's op layer refuses this pair the same way.
        if (
            self._combine.needs_aggregated_reservation
            and not self._dispatch.provides_aggregated_reservation
        ):
            raise ValueError(
                f"combine={recipe.combine.value} on the {recipe.combine_wires[-1]} "
                "wire reads its landing offsets off mori's aggregated dispatch "
                "reservation, which this dispatch does not make "
                "(MORI_EP_TOKOFF_EXT=1 turns it off)"
            )
        # The compact recipe's non-fused steps, when they go through mori.
        self._fallback: DispatchStage | None = None
        fallback = _fallback_config(config)
        if fallback is not None:
            from .stages.dispatch_mori import _MoriDispatch

            self._fallback = _MoriDispatch(fallback, _mori_capabilities())
        self.layout = _arena_layout(config, caps)
        self.res = _EpResources(SymmetricArena(communicator, self.layout))
        try:
            self.res.allocate(config, self.device)
            self._dispatch.build(self.res)
            self._combine.build(self.res)
            if self._fallback is not None:
                self._build_fallback()
        except BaseException:
            self.close()
            raise

    def _build_fallback(self) -> None:
        """mori over the shared arena, with its own slot counters, grid
        barrier and received count: both dispatches leave theirs in states
        only they expect."""
        res = copy.copy(self.res)
        res.allocate_dispatch_state(self._fallback.config, self.device)
        self._fallback.build(res)
        # Closed with the arena (see _EpResources.close).
        self.res.tokoff_ext = res.tokoff_ext

    @property
    def fallback(self) -> DispatchStage | None:
        """The mori dispatch the non-fused steps of a compact recipe take, or
        None."""
        return self._fallback

    def dispatch_fallback(
        self, hidden_states, topk_weights, topk_ids, *, recv_token_bound=None
    ) -> _DispatchBatch:
        return self._fallback.dispatch(
            hidden_states, topk_weights, topk_ids, recv_token_bound=recv_token_bound
        )

    @property
    def arena(self) -> SymmetricArena:
        return self.res.arena

    @property
    def total_recv(self) -> torch.Tensor:
        return self.res.total_recv

    @property
    def recv_scales(self) -> torch.Tensor | None:
        """The forwarded e8m0 rows over the whole arena; None on bf16."""
        return self._dispatch.recv_scales

    def recv_views(self) -> tuple:
        """(recv_x, recv_weights, recv_ids) over the whole arena."""
        return self._dispatch.recv_views()

    def dispatch(self, hidden_states, topk_weights, topk_ids, **step) -> _DispatchBatch:
        """``step`` is recv_token_bound, next_topk_ids, next_recv_token_bound
        and begin_step, as the dispatch stage takes them."""
        batch = self._dispatch.dispatch(hidden_states, topk_weights, topk_ids, **step)
        # The fused reduce must not consume a stale slot for a route the
        # compact plan rejected, including graph padding with expert ID -1.
        batch.routing.topk_ids = topk_ids
        return batch

    def prepare(self, batch: _DispatchBatch, quant_bits: int) -> _ComputeTarget:
        return self._combine.prepare(batch, quant_bits)

    def combine(
        self, routing: Routing, quant_bits: int, expert_out=None, shared_add=None
    ):
        """``shared_add``: bf16 rows added to the combined output. The fused
        combine adds them inside its reduce; any other stage adds them after."""
        if shared_add is None:
            return self._combine.finish(routing, quant_bits, expert_out)
        if getattr(self._combine, "adds_shared", False):
            return self._combine.finish(
                routing, quant_bits, expert_out, shared_add=shared_add
            )
        out = self._combine.finish(routing, quant_bits, expert_out)
        return out + shared_add[: out.shape[0]]

    def prefetch(self, topk_ids, recv_token_bound=None) -> None:
        self._dispatch.prefetch(topk_ids, recv_token_bound)

    def warmup(self, recv_token_bound=None, token_count=None) -> None:
        self._dispatch.warmup(recv_token_bound, token_count)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            # A launch still in flight (a graph replay, say) reads the window:
            # let it finish before anything it reads is freed.
            torch.cuda.synchronize(self.device)
        finally:
            # The plans hold the window and their kernels dereference it, so
            # they go before the arena does.
            stages = [self._combine, self._dispatch]
            if self._fallback is not None:
                stages.append(self._fallback)
            _close_all((*stages, self.res))
