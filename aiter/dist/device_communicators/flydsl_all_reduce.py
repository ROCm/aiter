# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Production dispatch for the FlyDSL all-reduce family.

Three schedules -- exact one-shot, quantized two-shot mesh, quantized two-shot
ring -- with the fastest one a function of payload size and world size. This
class owns the engines and consults ``allreduce_policy`` to choose between
them.

Opt-in: unset ``AITER_FLY_AR`` leaves it disabled and the dispatch chain
unchanged. Set ``AITER_FLY_AR=1`` to enable.

The one-shot is bit-exact (fp32 accumulate, one bf16 rounding, comparable
with ``cross_device_reduce``); the two-shot schedules quantize to INT4/INT6.
By default (``AITER_FLY_AR_ACCURACY=exact``) only the one-shot is ever
reachable -- above its ceiling this path declines the payload rather than
quantize it, so the caller falls through to whatever it would otherwise
dispatch to. ``AITER_FLY_AR_ACCURACY=fast`` unlocks the quantized mesh/ring
schedules for larger payloads, still preferring the exact one-shot wherever that
costs nothing.
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

# Importing these pulls in flydsl, which is an optional dependency and whose
# version is checked at import time. A failure here is the "FlyDSL is not
# usable in this environment" signal -- there is no separate availability
# predicate to consult.
try:
    from aiter.ops.flydsl import allreduce_policy as policy
    from aiter.ops.flydsl.kernels.quick_allreduce_fusions import (
        fused_qr_row_atoms,
    )
    from aiter.ops.flydsl.one_shot_allreduce import (
        OneShotAllReduce,
        OneShotAllReduceRMSNorm,
    )
    from aiter.ops.flydsl.quick_allreduce_int4 import (
        QuickAllReduceInt4,
        QuickAllReduceInt4RMSNorm,
        has_xgmi_peer_links,
    )

    _IMPORT_OK = True
except Exception:  # noqa: BLE001
    policy = None
    OneShotAllReduce = QuickAllReduceInt4 = has_xgmi_peer_links = None
    OneShotAllReduceRMSNorm = QuickAllReduceInt4RMSNorm = None
    fused_qr_row_atoms = None
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

        if not _IMPORT_OK:
            logger.debug("FlyDSL all-reduce disabled: FlyDSL is unavailable.")
            return
        if not policy.enabled():
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
            # have. Release everything and stay disabled rather than dispatching into
            # a half-built set.
            logger.warning(
                "FlyDSL all-reduce disabled: engine construction failed.",
                exc_info=True,
            )
            self.close()
            return

        mesh_kib = "unbounded" if resolved.mesh_max is None else f"{resolved.mesh_max >> 10}"
        logger.info(
            "FlyDSL all-reduce enabled: TP%d on %s, accuracy=%s, "
            "one-shot <= %d KiB, mesh <= %s KiB, %s, %.1f MiB of IPC inbox.",
            world_size,
            link,
            self.accuracy,
            resolved.oneshot_max >> 10,
            mesh_kib,
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
            # own guard and the two cannot disagree. ``link`` is passed rather
            # than left to re-detection so the engine's tuning ladder is keyed
            # on the same fabric this object resolved its policy against.
            return OneShotAllReduce(
                **common, max_bytes=self.policy.oneshot_max, link=self.link
            )
        # min_bytes=0: the family boundary above already decided this engine is
        # the right one for the payload, and QuickAllReduceInt4's own floor is
        # a standalone guard rail that would otherwise reject sizes the policy
        # just chose it for.
        return QuickAllReduceInt4(
            **common,
            algorithm="mesh" if family == "mesh" else "ring",
            min_bytes=0,
        )

    @property
    def inbox_bytes(self) -> int:
        return sum(e.inbox_bytes for e in self._engines.values())

    @property
    def max_bytes(self) -> int | None:
        """Upper bound on the message size the dispatcher accepts, or ``None`` if unbounded.

        Returns 0 when the instance is disabled so callers can gate on
        ``max_bytes > 0`` without a separate ``disabled`` check.
        """
        if self.disabled:
            return 0
        return self.policy.max_bytes

    def family_for(self, nbytes: int) -> str:
        """Which schedule *nbytes* dispatches to. Public for tests and reports."""
        return policy.pick_family(int(nbytes), self.policy)

    def variant(self, nbytes: int) -> str:
        """``<family>:<jit symbol>/g<cap>/x<blocks>`` for a payload of *nbytes*."""
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
        if nbytes < self.policy.min_bytes:
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

class FlyDSLAllReduceRMSNorm:
    """The same three schedules with residual-add + RMSNorm fused on.

    * Engines are per (family, hidden). The fused tile is one token row, so the
      wire layout depends on the width, and a width cannot be built until it is
      known. Building one allocates an IPC inbox and exchanges handles, which a
      HIP graph capture cannot contain -- so a width is made ready before
      capture, three ways, in order of preference:

      1. ``hiddens=`` at construction, or ``AITER_FLY_AR_FUSED_HIDDENS``, which
         builds up front and is the deployment answer.
      2. :meth:`prime`, for a caller that learns its widths later.
      3. The first ordinary call at that width, which every warmup pass makes.

      If none of those happened, :meth:`should_fly_fused_ar_rms` **declines
      while a capture is open** rather than building inside it. The capture then
      records the caller's unfused path -- slower, but correct, and it cannot
      hang. That decision is rank-symmetric: ranks capture in lockstep and
      ``_ready`` only ever changes through collectives.
    * ``min_bytes`` is live. Below it this path declines and the caller runs
      aiter's own fused kernel, which at decode shapes can be faster.
    """

    _SUPPORTED_WORLD_SIZES: ClassVar[list[Any]] = [2, 4, 8]
    _SUPPORTED_DTYPES: ClassVar[list[Any]] = [torch.bfloat16]

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device | int,
        hiddens: tuple[int, ...] = (),
    ):
        self.disabled = True
        self._engines: dict[str, Any] = {}
        # (family, hidden) pairs that are built *and* JIT-compiled, so a
        # call on them is a pure launch and safe inside a capture.
        self._ready: set[tuple[str, int]] = set()
        self._warned_capture = False
        self.policy = None

        if not _IMPORT_OK or not policy.enabled():
            return

        from aiter.jit.utils.chip_info import get_gfx_runtime

        arch = get_gfx_runtime()
        if arch not in _SUPPORTED_ARCHS:
            logger.debug("FlyDSL fused AR+RMSNorm disabled: unsupported arch %s.", arch)
            return
        if dist.get_backend(group) == dist.Backend.NCCL:
            logger.warning(
                "FlyDSL fused AR+RMSNorm disabled: it must be attached to a "
                "non-NCCL group (got %s).",
                dist.get_backend(group),
            )
            return
        if not all(in_the_same_node_as(group, source_rank=0)):
            logger.warning(
                "FlyDSL fused AR+RMSNorm disabled: HIP IPC handles are "
                "node-local and this process group spans nodes."
            )
            return

        rank = dist.get_rank(group=group)
        world_size = dist.get_world_size(group=group)
        if world_size == 1:
            return
        if world_size not in self._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "FlyDSL fused AR+RMSNorm disabled: unsupported world size %d "
                "(supported: %s).",
                world_size,
                self._SUPPORTED_WORLD_SIZES,
            )
            return

        link = "xgmi" if has_xgmi_peer_links() else "pcie"
        try:
            resolved = policy.resolve_fused(link, world_size)
        except ValueError as e:
            logger.warning("FlyDSL fused AR+RMSNorm disabled: %s", e)
            return

        self.group = group
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.link = link
        self.policy = resolved
        self.accuracy = policy.accuracy_mode()

        try:
            for family in policy.families_reachable(resolved):
                self._engines[family] = self._build(family)
        except Exception:
            logger.warning(
                "FlyDSL fused AR+RMSNorm disabled: engine construction failed.",
                exc_info=True,
            )
            self.close()
            return

        mesh_kib = "unbounded" if resolved.mesh_max is None else f"{resolved.mesh_max >> 10}"
        logger.info(
            "FlyDSL fused AR+RMSNorm enabled: TP%d on %s, accuracy=%s, "
            ">= %d B, one-shot <= %d KiB, mesh <= %s KiB, %s.",
            world_size,
            link,
            self.accuracy,
            resolved.min_bytes,
            resolved.oneshot_max >> 10,
            mesh_kib,
            "+".join(self._engines),
        )
        self.disabled = False

        # Widths declared up front are built now, while nothing is
        # capturing. This is the only path that needs no cooperation from
        # the caller's warmup.
        for h in policy.fused_hiddens(hiddens):
            try:
                self.prime(h)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "FlyDSL fused AR+RMSNorm: could not pre-build hidden=%d; "
                    "it will be built on first use instead.",
                    h,
                    exc_info=True,
                )

    def _build(self, family: str):
        """One engine, tuning left to that schedule's own ladder.

        No ``hiddens=``: a width is built on first use. Widths are not known
        here, and building every legal one up front would hold an IPC inbox per
        (family, width) for shapes the model never runs.
        """
        common = {
            "group": self.group,
            "device": self.device,
            "rank": self.rank,
            "world_size": self.world_size,
        }
        if family == "oneshot":
            return OneShotAllReduceRMSNorm(
                **common, max_bytes=self.policy.oneshot_max, link=self.link
            )
        algorithm = "mesh" if family == "mesh" else "ring"
        eng = QuickAllReduceInt4RMSNorm(
            **common,
            algorithm=algorithm,
            atoms_per_row=fused_qr_row_atoms(self.world_size, algorithm, self.link),
        )
        eng.min_bytes = 0
        return eng

    @property
    def inbox_bytes(self) -> int:
        return sum(e.inbox_bytes for e in self._engines.values())

    @property
    def max_bytes(self) -> int | None:
        """Upper bound on the message size the dispatcher accepts, or ``None`` if unbounded.

        Returns 0 when the instance is disabled so callers can gate on
        ``max_bytes > 0`` without a separate ``disabled`` check.
        """
        if self.disabled:
            return 0
        return self.policy.max_bytes

    def family_for(self, nbytes: int) -> str:
        return policy.pick_family(int(nbytes), self.policy)

    def variant(self, hidden: int, nbytes: int) -> str:
        family = self.family_for(int(nbytes))
        eng = self._engines.get(family)
        if eng is None:
            return family
        return f"{family}:{eng.variant(int(hidden), int(nbytes))}"

    #: Rows in the probe :meth:`prime` launches. Small, but >1 so a partial
    #: last tile is exercised the way a real decode step would be.
    _PRIME_TOKENS: ClassVar[int] = 8

    def prime(self, hidden: int) -> bool:
        """Build and JIT every reachable family at *hidden*. Collective.

        Makes a width safe to use inside a HIP graph capture. Every rank must
        call it with the same hidden dim, in the same order -- the build exchanges
        IPC handles. Idempotent; a width already primed costs one launch.

        Returns False when this width fuses in no family, which is a geometry
        fact (``supports_hidden``) and the same answer on every rank.
        """
        if self.disabled:
            return False
        hidden = int(hidden)
        usable = [
            (fam, eng)
            for fam, eng in self._engines.items()
            if eng.supports_hidden(hidden)
        ]
        if not usable:
            return False
        probe = torch.zeros(
            (self._PRIME_TOKENS, hidden), dtype=torch.bfloat16, device=self.device
        )
        weight = torch.zeros(hidden, dtype=torch.bfloat16, device=self.device)
        for fam, eng in usable:
            eng.compile_and_launch(probe, probe.clone(), weight, 1e-6)
            self._ready.add((fam, hidden))
        del probe, weight
        return True

    def _capture_blocked(self, family: str, hidden: int) -> bool:
        """Whether this call would have to build, with a capture already open.

        Building allocates and does a CPU-side handle exchange; neither belongs
        inside a capture. Declining records the caller's unfused path instead,
        which is slow but correct.
        """
        if (family, int(hidden)) in self._ready:
            return False
        if not torch.cuda.is_current_stream_capturing():
            return False
        if not self._warned_capture:
            self._warned_capture = True
            logger.warning(
                "FlyDSL fused AR+RMSNorm: declining hidden=%d inside a graph "
                "capture because %s is not built yet, so this capture will not "
                "fuse. Pre-build it with AITER_FLY_AR_FUSED_HIDDENS=%d, the "
                "hiddens= argument, or a warmup call before capture.",
                hidden,
                family,
                hidden,
            )
        return True

    def should_fly_fused_ar_rms(self, inp, residual, weight) -> bool:
        """Whether this path can and should handle this call. Never raises."""
        if self.disabled:
            return False
        for t in (inp, residual, weight):
            if not isinstance(t, torch.Tensor) or t.dtype not in self._SUPPORTED_DTYPES:
                return False
        if not is_weak_contiguous(inp) or not is_weak_contiguous(residual):
            return False
        hidden = int(inp.shape[-1])
        if weight.numel() != hidden or tuple(residual.shape) != tuple(inp.shape):
            return False
        nbytes = inp.numel() * inp.element_size()
        # Every FlyDSL schedule reads and writes 16 B atoms.
        if nbytes % 16 != 0:
            return False
        if nbytes < self.policy.min_bytes:
            return False
        family = self.family_for(nbytes)
        eng = self._engines.get(family)
        # supports_hidden is the geometry gate: one workgroup covers one row, so
        # most widths have no build at all in a given family.
        if eng is None or not eng.supports_hidden(hidden):
            return False
        return not self._capture_blocked(family, hidden)

    def fly_fused_ar_rms(self, inp, residual, weight, eps, *, out=None, res_out=None):
        """``(out, residual_out)``. Only call when the predicate said yes.

        *out*/*res_out* are for callers that already own the destinations --
        the bench, so its rows are not charged for a copy the shipped path does
        not make. Production passes neither and gets fresh tensors.
        """
        nbytes = inp.numel() * inp.element_size()
        family = self.family_for(nbytes)
        eng = self._engines[family]
        got = eng.allreduce_rmsnorm(
            inp, residual, weight, eps, out=out, residual_out=res_out
        )
        # Built and JIT-compiled by getting here, so a later call on this
        # (family, width) is a pure launch and may sit inside a capture.
        self._ready.add((family, int(inp.shape[-1])))
        return got

    def close(self):
        for eng in self._engines.values():
            try:
                eng.close()
            except (AttributeError, RuntimeError):
                pass
        self._engines = {}
        self.disabled = True

    def __del__(self):
        try:
            self.close()
        except (AttributeError, RuntimeError, TypeError):
            pass
