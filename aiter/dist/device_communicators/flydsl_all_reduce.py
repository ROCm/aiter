# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Production dispatch for the FlyDSL all-reduce family.

Three schedules -- exact one-shot, quantized two-shot mesh, quantized two-shot
ring -- with the fastest one a function of payload size and world size. This
class owns the engines and consults ``qr_ar_policy`` to choose between them.

Opt-in: unset ``AITER_FLY_AR`` leaves it disabled and the dispatch chain
unchanged. Set ``AITER_FLY_AR=1`` to enable.

The one-shot is bit-exact (fp32 accumulate, one bf16 rounding, comparable
with ``cross_device_reduce``). The two-shot schedules quantize to INT4/INT6.
The default policy keeps the exact schedule wherever it is within 10% of the fastest option.
``AITER_FLY_AR_ACCURACY=fast`` enables fast kernels everywhere.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from ..parallel_state import in_the_same_node_as
from .quick_all_reduce import is_weak_contiguous

logger = logging.getLogger(__name__)

try:
    from aiter.ops.flydsl.kernels import qr_ar_policy as policy
    from aiter.ops.flydsl.kernels.qr_1stage import OneShotAllReduce
    from aiter.ops.flydsl.kernels.qr_int4 import QRInt4, has_xgmi_peer_links
    from aiter.ops.flydsl.utils import is_flydsl_available

    _IMPORT_OK = True
except Exception:  # noqa: BLE001
    policy = None
    OneShotAllReduce = QRInt4 = has_xgmi_peer_links = is_flydsl_available = None
    _IMPORT_OK = False

_SUPPORTED_ARCHS = ("gfx942", "gfx950")


class FlyDSLAllReduce:
    """Engines for every reachable FlyDSL schedule, plus the size-keyed choice.

    ``self.disabled`` is set True first and only cleared on the last line of a
    fully successful init, every unsupported condition is an early ``return``
    rather than a raise, and ``should_fly_all_reduce`` is a pure predicate that
    answers False instead of throwing.
    """

    _SUPPORTED_WORLD_SIZES: ClassVar[list[Any]] = [2, 4, 8]
    _SUPPORTED_DTYPES: ClassVar[list[Any]] = [torch.bfloat16]

    def __init__(self, group: ProcessGroup, device: torch.device | int):
        self.disabled = True
        self._engines: dict[str, Any] = {}
        self.policy = None

        if not _IMPORT_OK or policy.enabled() is not True:
            # policy.enabled() is tristate, but only True enables.
            return
        if not is_flydsl_available():
            logger.debug("FlyDSL all-reduce disabled: FlyDSL is unavailable.")
            return

        from aiter.jit.utils.chip_info import get_gfx_runtime

        arch = get_gfx_runtime()
        if arch not in _SUPPORTED_ARCHS:
            logger.debug("FlyDSL all-reduce disabled: unsupported arch %s.", arch)
            return

        # IPC handle exchange rides the gloo group and needs CPU-side
        # broadcast_object_list; an NCCL group cannot carry it.
        if dist.get_backend(group) == dist.Backend.NCCL:
            logger.warning(
                "FlyDSL all-reduce disabled: it must be attached to a "
                "non-NCCL group (got %s).",
                dist.get_backend(group),
            )
            return
        if not all(in_the_same_node_as(group, source_rank=0)):
            logger.warning(
                "FlyDSL all-reduce disabled: HIP IPC handles are node-local "
                "and this process group spans nodes."
            )
            return

        rank = dist.get_rank(group=group)
        world_size = dist.get_world_size(group=group)
        if world_size == 1:
            return
        if world_size not in self._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "FlyDSL all-reduce disabled: unsupported world size %d "
                "(supported: %s).",
                world_size,
                self._SUPPORTED_WORLD_SIZES,
            )
            return

        # Arch does not identify the fabric -- MI350X (xGMI) and MI350P
        # (PCIe-only) both report gfx950 and want different thresholds.
        link = "xgmi" if has_xgmi_peer_links() else "pcie"
        try:
            resolved = policy.resolve(link, world_size)
        except ValueError as e:
            logger.warning("FlyDSL all-reduce disabled: %s", e)
            return
        if link == "xgmi":
            logger.warning(
                "FlyDSL all-reduce: the xGMI policy rows are a conservative placeholder."
            )

        self.group = group
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.link = link
        self.policy = resolved
        self.accuracy = policy.accuracy_mode()

        # Every engine the table can reach has to exist before the first
        # allreduce: construction performs an IPC handle exchange, which is a
        # collective, so it cannot be deferred to the first payload that needs
        # it -- one rank building while the others reduce is a deadlock. Built
        # in the fixed order of ``FAMILIES`` for the same reason.
        try:
            for family in policy.families_reachable(resolved):
                self._engines[family] = self._build(family)
        except Exception:
            # A partial build leaves this rank holding inboxes its peers may not
            # have. Release them and stay disabled rather than dispatching into
            # a half-built set.
            logger.warning(
                "FlyDSL all-reduce disabled: engine construction failed.",
                exc_info=True,
            )
            self.close()
            return

        logger.info(
            "FlyDSL all-reduce enabled: TP%d on %s, accuracy=%s, "
            "one-shot <= %d KiB, mesh <= %d KiB, %s, %.1f MiB of IPC inbox.",
            world_size,
            link,
            self.accuracy,
            resolved.oneshot_max >> 10,
            resolved.mesh_max >> 10,
            "+".join(self._engines),
            self.inbox_bytes / 2**20,
        )
        self.disabled = False

    def _build(self, family: str):
        """One engine, with its tuning left to that schedule's own ladder."""
        common = {
            "group": self.group,
            "device": self.device,
            "rank": self.rank,
            "world_size": self.world_size,
        }
        if family == "oneshot":
            # max_bytes from the resolved policy rather than the class default,
            # so an AITER_FLY_AR_ONESHOT_MAX_BYTES override reaches the engine's
            # own guard and the two cannot disagree.
            return OneShotAllReduce(**common, max_bytes=self.policy.oneshot_max)
        # min_bytes=0: the family boundary above already decided this engine is
        # the right one for the payload, and QRInt4's own floor is a standalone
        # guard rail that would otherwise reject sizes the policy just chose it
        # for.
        return QRInt4(
            **common,
            algorithm="mesh" if family == "mesh" else "ring",
            min_bytes=0,
        )

    @property
    def inbox_bytes(self) -> int:
        return sum(e.inbox_bytes for e in self._engines.values())

    def family_for(self, nbytes: int) -> str:
        """Which schedule *nbytes* dispatches to. Public for tests and reports."""
        return policy.pick_family(int(nbytes), self.policy)

    def variant(self, nbytes: int) -> str:
        """``<family>:<jit symbol>/g<cap>/x<blocks>`` for a payload of *nbytes*.
        """
        family = self.family_for(int(nbytes))
        eng = self._engines.get(family)
        return f"{family}:{eng.variant(int(nbytes))}" if eng is not None else family

    def should_fly_all_reduce(self, inp: torch.Tensor) -> bool:
        """Whether this path can and should handle *inp*. Never raises."""
        if self.disabled:
            return False
        if inp.dtype not in self._SUPPORTED_DTYPES:
            return False
        nbytes = inp.numel() * inp.element_size()
        # Every FlyDSL schedule reads and writes 16 B atoms.
        if nbytes % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        if not self.policy.min_bytes <= nbytes <= self.policy.max_bytes:
            return False
        # A family the table can name but this rank did not build (an env
        # override can widen a window past what ``families_reachable`` saw at
        # construction). Decline rather than KeyError on the critical path.
        return self.family_for(nbytes) in self._engines

    def fly_all_reduce(self, inp: torch.Tensor, out: torch.Tensor | None = None):
        if out is None:
            out = torch.empty_like(inp)
        nbytes = inp.numel() * inp.element_size()
        self._engines[self.family_for(nbytes)].allreduce(inp, out)
        return out

    def close(self):
        for eng in self._engines.values():
            try:
                eng.close()
            except (AttributeError, RuntimeError):
                pass
        self._engines = {}
        self.disabled = True

    def __del__(self):
        # Interpreter shutdown can already have torn down the modules close()
        # reaches, which is why this swallows rather than reports.
        try:
            self.close()
        except (AttributeError, RuntimeError, TypeError):
            pass
