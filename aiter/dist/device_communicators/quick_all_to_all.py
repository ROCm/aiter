# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The FlyDSL all-to-all slot of ``CudaCommunicator``.

``QuickAllToAll`` owns one ``FlyQuickAllToAll`` engine for its process group
when ``AITER_FLY_A2A=1`` (see ``aiter.ops.flydsl.alltoall_policy``), and
answers, per call, whether that engine should serve it. Anything it declines
falls through to RCCL in ``CudaCommunicator.all_to_all``.

Same lifecycle as the FlyDSL backend of ``QuickAllReduce``: every rank builds
its engine, the ranks agree that all of them succeeded, and the served binaries
are compiled before any CUDA graph capture can begin. Any failure disables the
slot on every rank, so the group never splits between FlyDSL and RCCL.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from ..parallel_state import in_the_same_node_as
from .flydsl_utils import all_ranks_agree, preload_fly_engines

logger = logging.getLogger(__name__)

try:
    from aiter.ops.flydsl import alltoall_policy as a2a_policy
    from aiter.ops.flydsl.quick_alltoall import FlyQuickAllToAll

    _FLY_IMPORT_OK = True
except Exception:  # noqa: BLE001
    a2a_policy = None
    FlyQuickAllToAll = None
    _FLY_IMPORT_OK = False

_SUPPORTED_ARCHS = ("gfx942", "gfx950")
_SUPPORTED_WORLD_SIZES = (2, 4, 8)


class QuickAllToAll:
    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        *,
        enable: bool | None = None,
        codec: str | None = None,
    ) -> None:
        """*group* must be a non-NCCL group: the engine exchanges HIP IPC
        handles over it. *group* may be any single-node subgroup.

        *enable* ``None`` follows ``AITER_FLY_A2A``; ``True`` builds the engine
        whatever the environment says, for a host framework with its own switch.
        *codec* ``None`` follows ``AITER_FLY_A2A_CODEC``; ``"none"`` pins the
        lossless wire, which a caller moving packed or non-float payloads needs.
        """
        self.disabled = True
        self._engines = {}
        self._policy = None
        if not _FLY_IMPORT_OK:
            return
        if not (a2a_policy.enabled() if enable is None else enable):
            return

        from aiter.jit.utils.chip_info import get_gfx_runtime

        arch = get_gfx_runtime()
        if arch not in _SUPPORTED_ARCHS:
            logger.debug("FlyDSL all-to-all disabled: unsupported arch %s.", arch)
            return
        if dist.get_backend(group) == dist.Backend.NCCL:
            logger.warning(
                "FlyDSL all-to-all disabled: it must be attached to a non-NCCL group."
            )
            return
        if not all(in_the_same_node_as(group, source_rank=0)):
            logger.warning(
                "FlyDSL all-to-all disabled: the process group spans nodes and "
                "HIP IPC handles are node-local."
            )
            return
        self.group = group
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        if self.world_size not in _SUPPORTED_WORLD_SIZES:
            if self.world_size > 1:
                logger.warning(
                    "FlyDSL all-to-all disabled: unsupported world size %d "
                    "(supported: %s).",
                    self.world_size,
                    _SUPPORTED_WORLD_SIZES,
                )
            return
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        policy = a2a_policy.resolve(self.world_size, codec=codec)
        ok = True
        try:
            # One engine per schedule the policy can select, built in a fixed
            # order: each construction exchanges IPC handles, a collective.
            for family in policy.families():
                self._engines[family] = FlyQuickAllToAll(
                    group=group,
                    device=device,
                    rank=self.rank,
                    world_size=self.world_size,
                    algorithm=family,
                    codec=policy.codec,
                    link=policy.link,
                )
        except Exception:
            logger.warning(
                "FlyDSL all-to-all disabled: engine construction failed.",
                exc_info=True,
            )
            ok = False
        if not all_ranks_agree(ok, group):
            if ok:
                logger.warning(
                    "FlyDSL all-to-all disabled: a peer rank failed to build its engine."
                )
            self.close()
            return

        ok = True
        try:
            preload_fly_engines(
                (engine, policy.family_range(family))
                for family, engine in self._engines.items()
            )
        except Exception:
            logger.warning("FlyDSL all-to-all disabled: warmup failed.", exc_info=True)
            ok = False
        if not all_ranks_agree(ok, group):
            if ok:
                logger.warning(
                    "FlyDSL all-to-all disabled: a peer rank failed to warm up."
                )
            self.close()
            return

        self._policy = policy
        self.disabled = False
        logger.info(
            "FlyDSL all-to-all: %s (ring from %s), codec %s, TP%d on %s, "
            "%d B <= bytes <= %s, %.1f MiB of IPC inbox.",
            "+".join(self._engines),
            "never" if policy.ring_from is None else f"{policy.ring_from} B",
            policy.codec,
            self.world_size,
            policy.link,
            policy.min_bytes,
            "no limit" if policy.max_bytes >= a2a_policy.NO_MAX else policy.max_bytes,
            sum(e.inbox_bytes for e in self._engines.values()) / 2**20,
        )

    def _engine_for(self, nbytes: int):
        return self._engines[self._policy.pick(nbytes)]

    def should_all_to_all(
        self, inp: torch.Tensor, out: torch.Tensor | None = None
    ) -> bool:
        """Whether a FlyDSL engine can and should serve *inp* (into *out*, when
        the caller already has it). Never raises."""
        if self.disabled:
            return False
        nbytes = inp.numel() * inp.element_size()
        return self._policy.routes(nbytes) and self._engine_for(nbytes).supports(
            inp, out
        )

    def all_to_all(
        self, inp: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Equal-split all-to-all of *inp* into *out* (allocated if None)."""
        if out is None:
            out = torch.empty_like(inp)
        nbytes = inp.numel() * inp.element_size()
        self._engine_for(nbytes).all_to_all(inp, out)
        return out

    def variant(self, nbytes: int) -> str | None:
        """The binary an *nbytes* payload runs, or None when disabled."""
        return None if self.disabled else self._engine_for(nbytes).variant(nbytes)

    def close(self) -> None:
        # Not gated on self.disabled: an engine built before a peer failed still
        # holds IPC handles opened against every peer.
        for engine in getattr(self, "_engines", {}).values():
            try:
                engine.close()
            except (AttributeError, RuntimeError):
                pass
        self._engines = {}
        self._policy = None
        self.disabled = True

    def __del__(self):
        self.close()
