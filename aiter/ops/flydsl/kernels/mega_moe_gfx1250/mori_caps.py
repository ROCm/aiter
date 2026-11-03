# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""mori, as MegaMoE drives it: every mori name the pipeline uses is reached
through ``_MoriCapabilities``."""

import functools
from dataclasses import dataclass

# The combine wires mori offers, as its quant_type names.
# "fp4_blockwise" is MXFP4 -- e2m1 plus one e8m0 per 32 -- the same format the
# fused reduce calls "mxfp4". It has no MXFP8.
_MORI_COMBINE_QUANT = {0: "none", 4: "fp4_blockwise"}


# mori's C++ EpArgs offset stems -> this package's arena region names, shared by
# the dispatch and the combine plan. A dispatch plan binds all of them but never
# touches `outTok`; the combine stages its input there and pulls peers' rows
# from it.
_MORI_REGION_NAMES = {
    "tokOff": "tok_off",
    "recvNum": "recv_num",
    "recvToSrc": "recv_to_src_token",
    "outIdx": "out_idx",
    "outWts": "out_wts",
    "dispOut": "disp_out",
    "outTok": "comb_inp",
    "xdb": "cross_device_barrier",
    # Only laid out on a quantizing wire; plan_api binds a missing region to 0 and
    # the kernel's `if constexpr` keeps that 0 from being read.
    "outScales": "disp_out_scales",
}


# What mori's describe() resolves a request against: MegaMoE runs only here, and
# naming it keeps the sizing off the GPU.
_MORI_ARCH = "gfx1250"


@dataclass(frozen=True)
class _MoriCapabilities:
    """The installed mori, as MegaMoE drives it.

    MegaMoE drives mori's EpDispatchPlan / EpCombinePlan directly, past its op
    layer, so every mori name it uses is reached through here. It needs plans
    that describe() themselves (ROCm/mori#844): the arena regions, the staging
    buffer and the scale stride all come from that, so none of mori's sizing is
    copied here. Such a mori also has what the pipeline relies on beyond sizing:
    the aggregated dispatch reservation the fp4 push combine reads its landing
    offsets from, and a dispatch that overwrites total_recv, so nothing zeroes
    it ahead of one.
    """

    ep_plans: object
    hip_backend: object
    lookup: object

    def tuning(self, config, quant_type: str = "none") -> dict:
        return self.lookup(
            config.world_size,
            config.hidden_dim,
            config.topk,
            dtype=config.dispatch_wire_spec.mori_tuning_dtype,
            experts_per_rank=config.experts_per_rank,
            quant_type=quant_type,
        )

    def dispatch_schedule(self, config) -> tuple:
        """mori's own tuned buckets, as this package's (bound, block, warp) triples.

        Not `_select_dispatch_config`'s: that table asks for 32 warps above 256
        tokens, and mori's gfx1250 dispatch stages a hidden-dim tile per warp in
        dynamic LDS -- 32 * 7168 * 2 = 458 KB, past the 320 KB budget. EpCfgIsValid
        does not check LDS, so the plan would build and then fail at launch.
        """
        tuned = self.tuning(config)
        schedule = tuned["schedule"]
        if schedule:
            return tuple((bound, block, warp) for bound, block, warp, _, _ in schedule)
        return ((None, tuned["dispatch_block_num"], tuned["warp_num_per_block"]),)

    def combine_schedule(self, config, quant_bits: int = 0) -> tuple:
        """The combine half of mori's tuned buckets for the ``quant_bits`` wire, as
        (bound, block, warp)."""
        tuned = self.tuning(config, _MORI_COMBINE_QUANT[quant_bits])
        schedule = tuned["schedule"]
        if schedule:
            return tuple((bound, block, warp) for bound, _, _, block, warp in schedule)
        return (
            (None, tuned["combine_block_num"], tuned["combine_warp_num_per_block"]),
        )

    def describe_dispatch(self, requests) -> list[dict]:
        """What mori reports for each dispatch plan request -- its arena regions,
        staging bytes and scale stride -- without building or compiling one."""
        plan = self.ep_plans.EpDispatchPlan
        return [plan.describe(arch=_MORI_ARCH, **request) for request in requests]

    def describe_combine(self, requests) -> list[dict]:
        """What mori reports for each combine plan request."""
        plan = self.ep_plans.EpCombinePlan
        return [plan.describe(arch=_MORI_ARCH, **request) for request in requests]

    def region_sizes(self, infos) -> dict:
        """The arena regions ``infos`` ask for, under this package's names."""
        return dict(
            self.ep_plans.arena_regions(*infos, region_names=_MORI_REGION_NAMES)
        )

    def dispatch_plan(self, request: dict, arena):
        return self.ep_plans.EpDispatchPlan(
            **request, arena=arena, region_names=_MORI_REGION_NAMES
        )

    def combine_plan(self, request: dict, arena):
        return self.ep_plans.EpCombinePlan(
            **request, arena=arena, region_names=_MORI_REGION_NAMES
        )

    def tokoff_ext_wanted(self, config) -> bool:
        """Whether mori's slot allocator word moves out of the cco window into
        tokoff-ext memory, as mori's op layer decides it.

        Collective when $MORI_EP_TOKOFF_EXT is on: TokOffExt.wanted checks that
        every rank shares the host.
        """
        return bool(self.hip_backend.TokOffExt.wanted(config.world_size))

    def tokoff_ext(self, config, device):
        """mori's tokoff-ext allocation, reused rather than duplicated: its IPC
        handle protocol must match the kernel byte for byte. Collective."""
        return self.hip_backend.TokOffExt(config.rank, config.world_size, device)

    @property
    def xdb_flag_slots(self) -> int:
        """Per-block epoch slots of the combine's entry barrier."""
        return self.hip_backend._XDB_FLAG_SLOTS


@functools.cache
def _mori_capabilities() -> _MoriCapabilities:
    try:
        from mori.ops.dispatch_combine_v2 import ep_plans
    except ImportError as error:
        raise RuntimeError(
            "MegaMoE's mori dispatch and combine need a mori with "
            "ops/dispatch_combine_v2"
        ) from error
    except OSError as error:
        raise RuntimeError(
            "MegaMoE's mori dispatch and combine need mori's libmori_ops_v2.so; "
            "it is built by mori's CMake and is not shipped by every install"
        ) from error
    if not (
        hasattr(ep_plans.EpDispatchPlan, "describe")
        and hasattr(ep_plans, "arena_regions")
    ):
        raise RuntimeError(
            "MegaMoE sizes mori's arena regions with Plan.describe(), which needs "
            "ROCm/mori#844 or later; the installed mori has none"
        )
    from mori.ops.dispatch_combine_v2 import hip_backend
    from mori.ops.dispatch_combine_v2.hip_tuning_configs import lookup

    return _MoriCapabilities(ep_plans=ep_plans, hip_backend=hip_backend, lookup=lookup)
