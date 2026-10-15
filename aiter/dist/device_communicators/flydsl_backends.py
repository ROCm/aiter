# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The FlyDSL fused all-reduce + RMSNorm backend.

The message size windows come from the fused policy, ``resolve_fused_policy``,
and ``AITER_FLY_AR_FUSED_<ALG>_{MIN,MAX}_BYTES`` override them.
"""

import logging
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)

# Importing these pulls in flydsl, an optional dependency whose version is
# checked at import time.
try:
    from aiter.ops.flydsl import allreduce_policy as fly_policy
    from aiter.ops.flydsl.kernels.comm.quick_allreduce_fusions import (
        PAD_MASK_MAX_BYTES,
    )
    from aiter.ops.flydsl.one_shot_allreduce import OneShotAllReduceRMSNorm
    from aiter.ops.flydsl.quick_allreduce import FlyQuickAllReduceRMSNorm

    _FLY_IMPORT_OK = True
except Exception:  # noqa: BLE001
    fly_policy = None
    PAD_MASK_MAX_BYTES = None
    OneShotAllReduceRMSNorm = None
    FlyQuickAllReduceRMSNorm = None
    _FLY_IMPORT_OK = False


def all_ranks_agree(ok: bool, group) -> bool:
    """Whether *every* rank in *group* reports success."""

    flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def preload_fly_engines(engines, *args) -> None:
    """Compile the binaries each engine serves, without launching them.

    *engines* holds ``(engine, (lo, hi))`` pairs, ``lo..hi`` being the payload
    bytes (inclusive) the dispatcher routes to that engine. Binaries its ladder
    assigns only to payloads outside that range are left alone: they never run.
    *args* go to every ``preload`` ahead of the range: the hidden dim, for the
    fused engines, whose binaries are per width.

    FlyDSL compiles a binary on its first launch, and if that happens inside an
    active CUDA graph capture, the capture pays seconds of JIT. The kernels take
    the payload size at runtime, so one preload per binary covers every size.

    The HIP module load is left to the first launch.
    """

    for engine, payload_range in engines:
        engine.preload(*args, payload_range=payload_range)


class FlyDSLAllReduceRMSNorm:
    """FlyDSL all-reduce with residual-add + RMSNorm fused on.

    *quantized* picks the families: the exact one-shot (``False``), or the
     quantized mesh and ring (``True``).

    * Engines are per (family, hidden). The fused tile is one token row, so the
      wire layout depends on the width, and a width cannot be built until it is
      known. The width is made ready before HIP graph capture in three ways:

      1. ``hiddens=`` at construction, or ``AITER_FLY_AR_FUSED_HIDDENS``, which
         builds up front and is the deployment answer.
      2. :meth:`prime`, for a caller that learns its widths later.
      3. The first ordinary call at that width, which every warmup pass makes.

      If none of those happened, :meth:`should_fly_fused_ar_rms` **declines
      while a capture is open** rather than building inside it. The capture then
      records the caller's unfused path, and it cannot hang.
    """

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        *,
        quantized: bool,
        hiddens: tuple[int, ...] = (),
    ):
        self.disabled = True
        self._engines: dict[str, Any] = {}
        # (family, hidden) pairs that are built *and* JIT-compiled, so a
        # call on them is a pure launch and safe inside a capture.
        self._ready: set[tuple[str, int]] = set()
        self._warned_capture = False
        self._quantized = bool(quantized)
        self.policy = None
        self._tag = "FlyDSL fused AR+RMSNorm (%s)" % (
            "quantized" if self._quantized else "exact"
        )

        if not _FLY_IMPORT_OK or not fly_policy.enabled():
            return

        world_size = dist.get_world_size(group=group)
        if world_size not in fly_policy.SUPPORTED_WORLDS:
            return

        from aiter.jit.utils.chip_info import get_gfx_runtime

        arch = get_gfx_runtime()
        if arch not in fly_policy.SUPPORTED_ARCHS:
            logger.debug("%s disabled: unsupported arch %s.", self._tag, arch)
            return

        # The link and the policy are identical on every rank, so all ranks
        # decline together.
        try:
            link = fly_policy.detect_link()
            policy = fly_policy.resolve_fused_policy(
                world_size=world_size, arch=arch, link=link
            )
        except fly_policy.NoPolicyError as e:
            logger.debug("%s disabled: %s", self._tag, e)
            return
        except ValueError as e:
            logger.warning("%s disabled: %s", self._tag, e)
            return

        self.group = group
        self.device = device
        self.rank = dist.get_rank(group=group)
        self.world_size = world_size
        self.link = link
        self.policy = policy
        # The enabled windows, by family, in payload order. The quantized
        # instance leaves the one-shot to the exact one.
        self._windows = {
            alg.value: policy.window(alg, quantized=self._quantized)
            for alg in policy.algorithms(quantized=self._quantized)
            if self._quantized != (alg == fly_policy.AllReduceAlgorithm.OneShot)
        }
        if not self._windows:
            logger.debug("%s disabled by the policy.", self._tag)
            return

        ok = True
        try:
            for family in self._windows:
                self._engines[family] = self._build(family)
        except Exception:
            logger.warning(
                "%s disabled: engine construction failed.", self._tag, exc_info=True
            )
            ok = False

        if not all_ranks_agree(ok, group):
            if ok:
                logger.warning(
                    "%s disabled: a peer rank failed to build its engines.",
                    self._tag,
                )
            self.close()
            return

        logger.info(
            "%s enabled: TP%d on %s, %s.",
            self._tag,
            world_size,
            link,
            ", ".join(f"{fam} {w}" for fam, w in self._windows.items()),
        )
        self.disabled = False

        # Widths declared up front are built now, while nothing is
        # capturing. This is the only path that needs no cooperation from
        # the caller's warmup.
        for h in fly_policy.fused_hiddens(hiddens):
            self._prime_up_front(h)

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
            "link": self.link,
        }
        if family == fly_policy.AllReduceAlgorithm.OneShot.value:
            return OneShotAllReduceRMSNorm(
                **common, max_bytes=self._windows[family].inclusive_range()[1]
            )
        # The policy window is the floor here, not the engine's own.
        return FlyQuickAllReduceRMSNorm(
            **common,
            algorithm=family,
            atoms_per_row=fly_policy.fused_qr_row_atoms(
                world_size=self.world_size, algorithm=family, link=self.link
            ),
            min_bytes=0,
        )

    def _prime_up_front(self, hidden: int) -> None:
        """:meth:`prime` at construction, agreed across ranks."""
        ok = True
        try:
            self.prime(hidden)
        except Exception:
            logger.warning(
                "%s: could not pre-build hidden=%d.", self._tag, hidden, exc_info=True
            )
            ok = False
        if not all_ranks_agree(ok, self.group):
            self._ready = {k for k in self._ready if k[1] != int(hidden)}
            logger.warning(
                "%s: hidden=%d is not pre-built; it will be built on first use.",
                self._tag,
                hidden,
            )

    @property
    def inbox_bytes(self) -> int:
        return sum(e.inbox_bytes for e in self._engines.values())

    @property
    def max_bytes(self) -> int | None:
        """Upper bound on the message size this path accepts, or ``None`` if unbounded.

        Returns 0 when the instance is disabled so callers can gate on
        ``max_bytes > 0`` without a separate ``disabled`` check.
        """
        if self.disabled:
            return 0
        if any(w.max is None for w in self._windows.values()):
            return None
        return max((w.inclusive_range()[1] for w in self._windows.values()), default=0)

    def family_for(self, nbytes: int) -> str | None:
        """The family that runs *nbytes*, or None when this path declines."""
        alg = self.policy.select(int(nbytes), quantized=self._quantized)
        return None if alg is None or alg.value not in self._engines else alg.value

    def variant(self, hidden: int, nbytes: int) -> str:
        family = self.family_for(int(nbytes))
        eng = self._engines.get(family)
        if eng is None:
            return family or "fallback"
        got = f"{family}:{eng.variant(int(hidden), int(nbytes))}"
        h_pad = eng.pads_hidden(int(hidden))
        return got if h_pad == int(hidden) else f"{got}/pad{h_pad}"

    def prime(self, hidden: int) -> bool:
        """Build every family at *hidden* and JIT the binaries its payload
        window can select.

        Makes a width safe to use inside a HIP graph capture. Every rank must
        call it with the same hidden dim, in the same order. The HIP module load is
        left to each binary's first launch.

        Returns False when this width fuses in no family, which is a geometry
        fact (``supports_hidden``).
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
        preload_fly_engines(
            ((eng, self._windows[fam].inclusive_range()) for fam, eng in usable),
            hidden,
        )
        self._ready.update((fam, hidden) for fam, _eng in usable)
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
                "%s: declining hidden=%d inside a graph capture because %s is "
                "not built yet, so this capture will not fuse. Pre-build it "
                "with AITER_FLY_AR_FUSED_HIDDENS=%d, the hiddens= argument, or "
                "a warmup call before capture.",
                self._tag,
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
            if not isinstance(t, torch.Tensor) or t.dtype is not torch.bfloat16:
                return False
            # Strictly contiguous: the engines reject weakly-contiguous views.
            if not t.is_contiguous() or t.data_ptr() % 16:
                return False
            if t.device.index != self.device.index:
                return False
        hidden = int(inp.shape[-1])
        if weight.numel() != hidden or tuple(residual.shape) != tuple(inp.shape):
            return False
        nbytes = inp.numel() * inp.element_size()
        # Every FlyDSL schedule reads and writes 16 B atoms.
        if nbytes % 16 or nbytes > fly_policy.MAX_PAYLOAD_BYTES:
            return False
        family = self.family_for(nbytes)
        eng = self._engines.get(family)
        # supports_hidden is the geometry gate: one workgroup covers one row, so
        # most widths have no build at all in a given family.
        if eng is None or not eng.supports_hidden(hidden):
            return False
        if nbytes > PAD_MASK_MAX_BYTES and eng.pads_hidden(hidden) != hidden:
            return False
        return not self._capture_blocked(family, hidden)

    def fly_fused_ar_rms(self, inp, residual, weight, eps, *, out=None, res_out=None):
        """``(out, residual_out)``. Only call when the predicate said yes.

        *out*/*res_out* are for callers that already own the destinations.
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
