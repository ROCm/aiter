# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 MegaMoE: dispatch, the grouped expert GEMM, and combine, built from a
recipe by ``MegaMoEGfx1250.from_recipe``.

The rest of the package: ``recipe`` (what runs, at what shape), ``transport``
(the dispatch and combine ``stages`` over one ``arena``) and ``mori_caps``.
"""

import os
import warnings

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode

from .arena import (
    _CCO_VMM_GRANULARITY,
    SymmetricArena,
    _ArenaLayout,
    _cco_alloc_bytes,
)
from .mori_caps import _mori_capabilities
from .recipe import (
    _COMBINE_QUANT_BITS,
    _COMBINE_QUANT_MODES,
    _COMBINE_WIRE_QUANT,
    _DISPATCH_WIRES,
    CombineKind,
    DispatchKind,
    ExpertComputeConfig,
    MegaMoEConfig,
    MegaMoEGeometry,
    MegaMoERecipe,
    _stage_classes,
    read_dispatch_wire_env,
)
from .stages.base import Routing, _ComputeTarget, _DispatchBatch
from .stages.dispatch_compact import _MAX_COMPACT_TILE_M, _CompactDispatch
from .transport import _arena_layout, _capabilities_for, _EpDispatcher

__all__ = [
    "CombineKind",
    "DispatchKind",
    "ExpertComputeConfig",
    "MegaMoEGeometry",
    "MegaMoEGfx1250",
    "MegaMoERecipe",
    "Routing",
    "read_dispatch_wire_env",
]


def _require_gfx1250():
    gfx = get_gfx()
    if gfx != "gfx1250":
        raise RuntimeError(f"MegaMoEGfx1250 requires gfx1250, got {gfx}")


def _alias(owner: str, name: str) -> property:
    """A read-only attribute that is ``self.<owner>.<name>``."""
    return property(lambda self: getattr(getattr(self, owner), name))


class MegaMoEGfx1250:
    """A8W4 EP MoE: dispatch, the grouped expert GEMM, combine.

    Built by ``from_recipe`` from a ``MegaMoERecipe`` (what runs) and a
    ``MegaMoEGeometry`` (the shape it runs at); forward() takes the rest. A
    model's MoE layers share their geometry, recipe and communication arena and
    differ only in their weights, so one instance serves every layer (weights
    are forward() arguments) and the model keeps a single cco symmetric arena
    instead of one per layer.

    The keyword constructor, which takes the older stage1_fused / stage2_fused
    flags, is deprecated. ``_config``, ``_arena``, ``_dispatch()``,
    ``_combine()`` and ``_recv_dispatch_scales()`` are kept for ATOM's Triton
    experts, which drive the dispatch and the combine themselves.
    """

    model_dim = _alias("geometry", "model_dim")
    inter_dim = _alias("geometry", "inter_dim")
    experts = _alias("geometry", "experts")
    experts_per_rank = _alias("geometry", "experts_per_rank")
    topk = _alias("geometry", "topk")
    max_tokens_per_rank = _alias("geometry", "max_tokens_per_rank")
    activation = _alias("compute", "activation")
    gate_mode = _alias("compute", "gate_mode")
    quant_type = _alias("compute", "quant_type")
    hidden_pad = _alias("compute", "hidden_pad")
    intermediate_pad = _alias("compute", "intermediate_pad")
    swiglu_limit = _alias("compute", "swiglu_limit")

    def __init__(
        self,
        *,
        communicator,
        rank: int,
        world_size: int,
        model_dim: int,
        inter_dim: int,
        experts: int,
        topk: int,
        max_tokens_per_rank: int,
        activation: ActivationType = ActivationType.Silu,
        gate_mode: int = GateMode.INTERLEAVE.value,
        quant_type: QuantType = QuantType.per_1x32,
        hidden_pad: int = 0,
        intermediate_pad: int = 0,
        swiglu_limit: float = 0.0,
        situ_beta: torch.Tensor | None = None,
        situ_linear_beta: torch.Tensor | None = None,
        dispatch_wire: str | None = None,
        stage1_fused: bool = False,
        stage2_fused: bool = True,
        combine_quant: str | None = None,
        dispatch_backend: str | None = None,
    ):
        """Deprecated: build a ``MegaMoERecipe`` and a ``MegaMoEGeometry`` and use
        ``from_recipe``. This turns the flags into a recipe the same way
        (``MegaMoERecipe.from_legacy``)."""
        warnings.warn(
            "MegaMoEGfx1250(**kwargs) is deprecated; use MegaMoEGfx1250.from_recipe "
            "with a MegaMoERecipe and a MegaMoEGeometry",
            DeprecationWarning,
            stacklevel=2,
        )
        _require_gfx1250()
        geometry = MegaMoEGeometry(
            rank=rank,
            world_size=world_size,
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            max_tokens_per_rank=max_tokens_per_rank,
        )
        recipe = MegaMoERecipe.from_legacy(
            stage1_fused=stage1_fused,
            stage2_fused=stage2_fused,
            dispatch_backend=dispatch_backend,
            dispatch_wire=dispatch_wire,
            combine_quant=combine_quant,
            activation=activation,
            gate_mode=gate_mode,
            quant_type=quant_type,
            hidden_pad=hidden_pad,
            intermediate_pad=intermediate_pad,
            swiglu_limit=swiglu_limit,
        )
        self._build(recipe, geometry, communicator, situ_beta, situ_linear_beta)

    @classmethod
    def from_recipe(
        cls,
        recipe: MegaMoERecipe,
        geometry: MegaMoEGeometry,
        *,
        communicator,
        situ_beta: torch.Tensor | None = None,
        situ_linear_beta: torch.Tensor | None = None,
    ) -> "MegaMoEGfx1250":
        """Build the MegaMoE ``recipe`` describes at ``geometry``.

        ``situ_beta`` / ``situ_linear_beta`` are the Situv2 activation's tensors,
        model-wide and needed only for that activation.
        """
        _require_gfx1250()
        self = cls.__new__(cls)
        self._build(recipe, geometry, communicator, situ_beta, situ_linear_beta)
        return self

    @classmethod
    def required_vmm_bytes(
        cls,
        recipe: MegaMoERecipe | None = None,
        geometry: MegaMoEGeometry | None = None,
        *,
        granularity: int = _CCO_VMM_GRANULARITY,
        **legacy,
    ) -> int:
        """The per-rank cco VMM (``per_rank_vmm``) this MegaMoE takes.

        Its arena -- the one cco allocation it makes -- rounded up the way
        ccoMemAlloc rounds it. The constructor allocates this same layout, so the
        allocation never exceeds it. It needs no communicator, so it can size the
        one about to be created; but it is MegaMoE's share only, and a
        communicator that also carries other cco allocations -- mori's own EP
        ops, or a DevComm's resource window (one granule each) -- needs theirs
        added on top. ``granularity`` is the communicator's VMM granularity, when
        known.

        The keyword form ``required_vmm_bytes(world_size=, hidden_dim=,
        max_tokens_per_rank=, experts=, topk=, stage1_fused=)`` names neither the
        wires nor the combine nor inter_dim, so it returns the largest any recipe
        with that dispatch could take.
        """
        if recipe is None:
            return cls._legacy_vmm_bound(granularity, **legacy)
        config = MegaMoEConfig(recipe=recipe, geometry=geometry)
        layout = _arena_layout(config, _capabilities_for(recipe))
        return _cco_alloc_bytes(layout.total_bytes, granularity)

    @classmethod
    def _legacy_vmm_bound(
        cls,
        granularity,
        *,
        world_size,
        hidden_dim,
        max_tokens_per_rank,
        experts,
        topk,
        stage1_fused=False,
    ):
        """The largest arena any recipe with this dispatch takes. inter_dim is
        not named, so the compact rows are sized for the widest GEMM tile a
        CSV row names."""
        geometry = MegaMoEGeometry(
            rank=0,
            world_size=world_size,
            model_dim=hidden_dim,
            inter_dim=1,
            experts=experts,
            topk=topk,
            max_tokens_per_rank=max_tokens_per_rank,
        )
        if stage1_fused:
            candidates = [
                MegaMoERecipe(dispatch=DispatchKind.COMPACT, dispatch_wire=wire)
                for wire in ("fp8", "fp4")
            ]
        else:
            candidates = [
                MegaMoERecipe(dispatch_wire=wire, combine=combine, combine_wires=wires)
                for wire in _DISPATCH_WIRES
                for combine, wires in (
                    (CombineKind.FUSED_SCATTER, ("bf16",)),
                    (CombineKind.MORI, ("bf16", "fp4")),
                )
            ]
        caps = None if stage1_fused else _mori_capabilities()
        sizes, refused = [], []
        for recipe in candidates:
            try:
                config = MegaMoEConfig(recipe=recipe, geometry=geometry)
            except ValueError as error:
                refused.append(error)  # e.g. mori's fp4 combine at this hidden
                continue
            dispatch, combine = _stage_classes(recipe)
            dispatch_regions = (
                _CompactDispatch.regions_for_tile(config, _MAX_COMPACT_TILE_M)
                if stage1_fused
                else dispatch.regions(config, caps)
            )
            layout = _ArenaLayout.merge(dispatch_regions, combine.regions(config, caps))
            sizes.append(layout.total_bytes)
        if not sizes:
            raise ValueError(
                f"no recipe with stage1_fused={bool(stage1_fused)} runs at "
                f"world_size={world_size}, hidden_dim={hidden_dim}, "
                f"max_tokens_per_rank={max_tokens_per_rank}, experts={experts}, "
                f"topk={topk}: {refused[-1]}"
            )
        return _cco_alloc_bytes(max(sizes), granularity)

    def _build(self, recipe, geometry, communicator, situ_beta, situ_linear_beta):
        config = MegaMoEConfig(recipe=recipe, geometry=geometry)
        self.recipe = recipe
        self.geometry = geometry
        self.situ_beta = situ_beta
        self.situ_linear_beta = situ_linear_beta

        device = torch.device("cuda", torch.cuda.current_device())
        self.expert_mask = torch.zeros(
            geometry.experts, dtype=torch.int32, device=device
        )
        first_expert = geometry.rank * geometry.experts_per_rank
        self.expert_mask[first_expert : first_expert + geometry.experts_per_rank] = 1
        self._config = config
        self._ep = _EpDispatcher(config, communicator)

    @property
    def compute(self) -> ExpertComputeConfig:
        return self.recipe.compute

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        bias1: torch.Tensor | None = None,
        bias2: torch.Tensor | None = None,
        a1_scale: torch.Tensor | None = None,
        a2_scale: torch.Tensor | None = None,
        recv_token_bound: int | None = None,
        next_topk_ids: torch.Tensor | None = None,
        next_recv_token_bound: int | None = None,
        combine_quant: str | None = None,
        combine_wire: str | None = None,
        recv_counter_is_zero: bool = False,
    ) -> torch.Tensor:
        """Run one MoE layer: dispatch, its expert GEMM, then combine.

        ``recv_token_bound`` caps how many recv slots the GEMM is shown. Dispatch
        always fills a world_size*max_tokens_per_rank arena, so a caller that
        knows a tighter static bound (e.g. graph_bs*topk*world_size for a uniform
        decode batch) can shrink the GEMM's grid with it. It must be a python int
        so the shape stays static under graph capture, and must not cut below the
        received count -- the kernels skip the tail past the device-side count on
        their own.

        ``next_topk_ids`` (and optional ``next_recv_token_bound``) start the next
        layer's compact plan on a side stream after this dispatch, so it overlaps
        the expert GEMM. The two plans use independent hist/done slots and local
        tok_map/psum buffers. Omit them to keep the plan sequential.

        ``combine_wire`` picks this step's combine wire out of the recipe's
        ``combine_wires``, so a caller can quantize the steps with enough tokens
        on the wire to pay for it and leave the rest on bf16. Omitting it means
        bf16 -- building a format only makes it available, each step still has to
        ask for it. ``combine_quant`` is the same choice in the older spelling
        (none / mxfp8 / mxfp4).

        ``recv_counter_is_zero`` is accepted and ignored: the dispatch
        overwrites the received count, so there is no reset to skip.
        """
        if combine_wire is not None:
            if combine_quant is not None:
                raise ValueError("pass combine_wire or combine_quant, not both")
            if combine_wire not in _COMBINE_WIRE_QUANT:
                raise ValueError(
                    f"combine_wire must be one of {tuple(_COMBINE_WIRE_QUANT)}, "
                    f"got {combine_wire!r}"
                )
            combine_quant = _COMBINE_WIRE_QUANT[combine_wire]
        quant_bits = self._resolve_combine_quant(combine_quant)
        self._check_inputs(
            hidden_states, topk_weights, topk_ids, w1, w2, w1_scale, w2_scale, a1_scale
        )
        ep = self._ep
        batch = ep.dispatch(
            hidden_states,
            topk_weights,
            topk_ids,
            recv_token_bound=recv_token_bound,
            next_topk_ids=next_topk_ids,
            next_recv_token_bound=next_recv_token_bound,
        )
        target = ep.prepare(batch, quant_bits)
        expert_out = self._compute(
            batch,
            target,
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            bias1=bias1,
            bias2=bias2,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
        )
        return ep.combine(batch.routing, quant_bits, expert_out)

    __call__ = forward

    def _check_inputs(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        w1,
        w2,
        w1_scale,
        w2_scale,
        a1_scale,
    ) -> None:
        if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_contiguous():
            raise ValueError("hidden_states must be contiguous bfloat16")
        if topk_weights.dtype != torch.float32 or not topk_weights.is_contiguous():
            raise ValueError("topk_weights must be contiguous float32")
        if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be contiguous int32")
        if os.environ.get("AITER_DISABLE_GROUPED_A8W4", "0") == "1":
            raise RuntimeError(
                "MegaMoE fused stage2 scatter requires the grouped A8W4 kernel; "
                "AITER_DISABLE_GROUPED_A8W4=1 is not supported"
            )
        if w1_scale is None or w2_scale is None:
            raise ValueError(
                "MegaMoE fused stage2 scatter requires both w1_scale and w2_scale"
            )
        supported_weight_dtypes = (torch.uint8, dtypes.fp4x2)
        if (
            w1.dtype not in supported_weight_dtypes
            or w2.dtype not in supported_weight_dtypes
        ):
            raise ValueError(
                "MegaMoE fused stage2 scatter requires MXFP4 w1/w2 weights "
                f"(uint8 or fp4x2), got {w1.dtype} and {w2.dtype}"
            )
        expected_w1_shape = (
            self.experts_per_rank,
            2 * self.inter_dim,
            self.model_dim // 2,
        )
        expected_w2_shape = (
            self.experts_per_rank,
            self.model_dim,
            self.inter_dim // 2,
        )
        if tuple(w1.shape) != expected_w1_shape:
            raise ValueError(
                "MegaMoE fused stage2 scatter requires interleaved G1U1 w1 with "
                f"shape {expected_w1_shape}, got {tuple(w1.shape)}"
            )
        if tuple(w2.shape) != expected_w2_shape:
            raise ValueError(
                f"MegaMoE fused stage2 scatter requires w2 shape "
                f"{expected_w2_shape}, got {tuple(w2.shape)}"
            )
        token_count = int(hidden_states.shape[0])
        if token_count > self.max_tokens_per_rank:
            raise ValueError(
                f"tokens={token_count} exceeds max_tokens_per_rank="
                f"{self.max_tokens_per_rank}"
            )
        expected_shape = (token_count, self.topk)
        if tuple(topk_weights.shape) != expected_shape:
            raise ValueError(
                f"topk_weights must have shape {expected_shape}, "
                f"got {tuple(topk_weights.shape)}"
            )
        if tuple(topk_ids.shape) != expected_shape:
            raise ValueError(
                f"topk_ids must have shape {expected_shape}, "
                f"got {tuple(topk_ids.shape)}"
            )
        if self._config.is_quant_dispatch_wire and a1_scale is not None:
            raise ValueError(
                "a1_scale is produced by the quantizing dispatch wire itself; a "
                "caller-supplied one would be silently discarded"
            )

    def _compute(
        self,
        batch: _DispatchBatch,
        target: _ComputeTarget,
        *,
        w1,
        w2,
        w1_scale,
        w2_scale,
        bias1,
        bias2,
        a1_scale,
        a2_scale,
    ) -> torch.Tensor:
        """The expert GEMM over one dispatched batch, leaving its rows where
        ``target`` says. Knows nothing of the transport beyond those two."""
        from aiter.ops.flydsl.grouped_moe_gfx1250 import (
            set_flydsl_dispatch_context,
        )

        compute = self.recipe.compute
        extra = {}
        if compute.activation == ActivationType.Situv2:
            extra["beta"] = self.situ_beta
            extra["linear_beta"] = self.situ_linear_beta
        # The grouped GEMM finds an expert-grouped batch's grouping here, not in
        # its arguments.
        set_flydsl_dispatch_context(target.dispatch_context)
        try:
            return fused_moe(
                batch.x,
                w1,
                w2,
                batch.topk_weights,
                batch.topk_ids,
                expert_mask=self.expert_mask,
                activation=compute.activation,
                gate_mode=compute.gate_mode,
                quant_type=compute.quant_type,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale if batch.scales is None else batch.scales,
                a2_scale=a2_scale,
                bias1=bias1,
                bias2=bias2,
                hidden_pad=compute.hidden_pad,
                intermediate_pad=compute.intermediate_pad,
                dtype=dtypes.bf16,
                num_local_tokens=batch.num_local_tokens,
                swiglu_limit=compute.swiglu_limit,
                stage2_scatter=target.scatter,
                output=target.output,
                **extra,
            )
        finally:
            set_flydsl_dispatch_context(None)

    def _resolve_combine_quant(self, combine_quant: str | None) -> int:
        """This step's combine payload width, checked against what was built.

        Unnamed means bf16. The constructor's ``combine_quant`` says which
        formats exist, not which one runs: a quantized wire only pays off once
        there are enough tokens on it to outweigh the per-token quant/dequant
        pair, and that is a property of the step, so a step has to ask.
        """
        if combine_quant is None:
            return 0
        if combine_quant not in _COMBINE_QUANT_MODES:
            raise ValueError(
                f"combine_quant must be one of {_COMBINE_QUANT_MODES}, "
                f"got {combine_quant!r}"
            )
        bits = _COMBINE_QUANT_BITS[combine_quant]
        built = self._config.combine_quant_bit_modes
        if bits not in built:
            raise ValueError(
                f"combine_quant={combine_quant!r} was not built: this MegaMoE "
                f"was constructed with combine_quant="
                f"{self._config.combine_quant!r}, which builds a combine for "
                f"{sorted(built)} bit widths only. Name the format at "
                "construction to make it available per step."
            )
        return bits

    def warmup_compact_plan(
        self,
        recv_token_bound: int | None = None,
        token_count: int | None = None,
    ) -> None:
        """Compile the compact plan this bound needs, before any graph capture.

        The plan is compiled per tile alignment and the alignment follows the
        bound's token bucket, so a captured batch whose bucket never ran eagerly
        would reach the JIT inside the capture -- which the plan refuses. A
        caller that captures a ladder of batch sizes warms every rung through
        here first. A no-op off the compact path.
        """
        self._ep.warmup(recv_token_bound, token_count)

    def prefetch_compact_plan(
        self,
        topk_ids: torch.Tensor,
        recv_token_bound: int | None = None,
    ) -> None:
        """Launch ``tdm_compact_plan`` on a side stream into the current slot.

        The plan only needs ``topk_ids``, so the caller can hide the first
        layer's plan behind RMSNorm. Later layers should pass ``next_topk_ids``
        to ``forward`` so the plan overlaps the previous expert GEMM instead.
        A no-op off the compact path.
        """
        self._ep.prefetch(topk_ids, recv_token_bound)

    # -- for callers that drive the transport themselves (ATOM's Triton experts)
    @property
    def _arena(self) -> SymmetricArena:
        return self._ep.arena

    def _dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        recv_counter_is_zero: bool = False,
    ):
        """Dispatch alone: (recv_x, recv_weights, recv_indices, total_recv,
        routing), each over the arena's full recv rows."""
        ep = self._ep
        batch = ep.dispatch(hidden_states, topk_weights, topk_ids, begin_step=False)
        return (*ep.recv_views(), ep.total_recv, batch.routing)

    def _recv_dispatch_scales(self) -> torch.Tensor | None:
        """The forwarded e8m0 rows, or None on the bf16 wire."""
        return self._ep.recv_scales

    def _combine(
        self,
        routing: Routing,
        combine_quant_bits: int = 0,
        expert_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Combine alone, on the ``combine_quant_bits`` wire (bf16 by default).
        On the fused path the results are already in the peers' slots; on
        mori's, ``expert_out`` is fused_moe's output in its staging rows."""
        return self._ep.combine(routing, combine_quant_bits, expert_out)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._ep.close()
