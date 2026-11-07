# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The FlyDSL backends of the communicators: the quantized plain all-reduce
(``FlyDSLQuickAllReduce``) and the fused all-reduce + RMSNorm
(``FlyDSLAllReduceRMSNorm``).

The message size windows come from the policies, ``resolve_family_policy`` and
``resolve_fused_policy``, and ``AITER_FLY_AR_<ALG>_{MIN,MAX}_BYTES`` and
``AITER_FLY_AR_FUSED_<ALG>_{MIN,MAX}_BYTES`` override them.
"""

import hashlib
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
    from aiter.ops.flydsl.one_shot_allreduce import OneShotAllReduceRMSNorm
    from aiter.ops.flydsl.quick_allreduce import (
        FlyQuickAllReduce,
        FlyQuickAllReduceRMSNorm,
    )

    _FLY_IMPORT_OK = True
except Exception:  # noqa: BLE001
    fly_policy = None
    OneShotAllReduceRMSNorm = None
    FlyQuickAllReduce = None
    FlyQuickAllReduceRMSNorm = None
    _FLY_IMPORT_OK = False

# Quantization levels the FlyDSL plain all-reduce serves, by quick-reduce
# regime name, mapped to the wire formats to pin on the ring's two laps.
# ``None`` means "use the schedule's own per-world default".
FLY_QUANTIZATIONS = {"INT4": (None, None)}


def all_ranks_agree(ok: bool, group) -> bool:
    """Whether *every* rank in *group* reports success."""

    flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def all_ranks_same(value: int, group) -> bool:
    """Whether every rank in group passes the same integer value."""

    both = torch.tensor([value, -value], dtype=torch.int64)
    dist.all_reduce(both, op=dist.ReduceOp.MAX, group=group)
    return int(both[0]) == value == -int(both[1])


def all_ranks_agree_on_plan(plan, group, *, owner: str) -> bool:
    """Whether every rank in the group holds the same FlyDSL plan.

    plan is this rank's dict of every input to its engine build, or an str
    naming why this rank declines. It is computed locally, from the environment
    and the host. Hence, ranks can disagree. One ``all_gather_object`` on the
    CPU group settles it before any engine is built.

    Every rank that reaches the FlyDSL init must call this, whatever its own
    verdict, including when FlyDSL is off or not importable.

    True only if every rank holds the same dict.
    """

    plans = [None] * dist.get_world_size(group)
    dist.all_gather_object(plans, plan, group=group)
    if all(isinstance(p, str) for p in plans):
        return False
    if all(p == plans[0] for p in plans):
        return True
    logger.warning(
        "%s disabled: the ranks disagree on the FlyDSL setup: %s",
        owner,
        _plan_differences(plans),
    )
    return False


def _plan_differences(plans) -> str:
    """The ranks that decline and the plan fields that differ, per rank."""

    parts = [
        f"rank {r} declines ({p})" for r, p in enumerate(plans) if isinstance(p, str)
    ]
    built = {r: p for r, p in enumerate(plans) if not isinstance(p, str)}
    for key in sorted({k for p in built.values() for k in p}):
        values = {r: p.get(key) for r, p in built.items()}
        if any(v != values[min(values)] for v in values.values()):
            parts.append(
                f"{key}: " + ", ".join(f"rank {r}={v}" for r, v in values.items())
            )
    return "; ".join(parts)


_POLICY_DECLINED = 0


def vote_on_digest(digest: int, group, *, tag: str) -> bool:
    """Whether every rank of *group* holds the same digest, and it is not the
    declined marker.

    Building an engine is a collective, so a rank that declines alone must not
    leave its peers waiting in one. Every rank that reaches a FlyDSL init calls
    this, whatever its own verdict: FlyDSL off, not importable, an unsupported
    arch or a policy it cannot resolve all vote ``_POLICY_DECLINED``. If any
    rank would build and the digests differ, every rank logs it.
    """
    both = torch.tensor([digest, -digest], dtype=torch.int64)
    dist.all_reduce(both, op=dist.ReduceOp.MAX, group=group)
    highest, lowest = int(both[0]), -int(both[1])
    if highest == lowest:
        return digest != _POLICY_DECLINED
    logger.warning(
        "%s disabled: the ranks disagree on the FlyDSL setup (check that the "
        "AITER_FLY_AR* variables, the arch and the link are the same on every "
        "rank).",
        tag,
    )
    return False


def _flydsl_version():
    try:
        from aiter.ops.flydsl import installed_flydsl_version

        return installed_flydsl_version
    except Exception:  # noqa: BLE001
        return None


def _digest(parts) -> int:
    """A nonzero number that is equal on two ranks exactly when *parts* are."""
    digest = hashlib.sha256(repr(parts).encode()).digest()
    return int.from_bytes(digest[:8], "little") % ((1 << 62) - 1) + 1


def _policy_digest(
    *, quantized: bool, link: str, arch: str, policy, world_size: int, hiddens
) -> int:
    """A number that is equal on two ranks exactly when they would build the
    same engines in the same order.
    """
    parts = (
        _flydsl_version(),
        quantized,
        link,
        arch,
        policy,
        fly_policy.fused_platform_ladders(world_size=world_size, link=link, arch=arch),
        fly_policy.fused_hiddens(hiddens),
    )
    return _digest(parts)


def preload_fly_engines(engines, *args) -> None:
    """Compile the binaries each engine serves and load them onto the device,
    without launching them.

    *engines* holds ``(engine, (lo, hi))`` pairs, ``lo..hi`` being the payload
    bytes (inclusive) the dispatcher routes to that engine. Binaries its ladder
    assigns only to payloads outside that range are left alone: they never run.
    *args* go to every ``preload`` ahead of the range: the hidden dim, for the
    fused engines, whose binaries are per width.

    FlyDSL would otherwise compile a binary, JIT its host wrapper and load its
    HIP module on its first launch, and if that happens inside an active HIP
    graph capture, the capture pays for it. The kernels take the payload size at
    runtime, so one preload per binary covers every size.
    """

    for engine, payload_range in engines:
        engine.preload(*args, payload_range=payload_range)


class FlyDSLQuickAllReduce:
    """Quantized FlyDSL plain all-reduce: the mesh and ring schedules.

    The policy (``resolve_family_policy``, quantized) picks the schedule per
    payload and leaves the one-shot window to the exact one-shot, which the
    custom all-reduce serves; :meth:`should_fly_ar` declines those sizes.

    Construction is collective: every rank of *group* must construct it, and
    all of them build or decline together. Every binary the policy routes to is
    compiled here, so a call is a pure launch and safe inside a HIP graph
    capture. Calls must use one stream and run in the same order on every rank.
    """

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        *,
        quantization: str | None = "INT4",
    ):
        self.disabled = True
        self.policy = None
        self.engines: dict[Any, Any] = {}
        self._tag = f"FlyDSL quick all-reduce ({quantization})"

        world_size = dist.get_world_size(group=group)
        # Every rank votes below, whatever it decided here: ``digest`` stays
        # declined unless this rank would build.
        digest = _POLICY_DECLINED
        link = policy = None
        if not _FLY_IMPORT_OK:
            logger.debug("%s disabled: flydsl is not importable.", self._tag)
        elif not fly_policy.enabled():
            pass
        elif quantization is None:
            logger.debug("%s disabled: no FlyDSL schedule for this regime.", self._tag)
        elif quantization not in FLY_QUANTIZATIONS:
            logger.warning(
                "%s disabled: supported quantizations are %s.",
                self._tag,
                tuple(FLY_QUANTIZATIONS),
            )
        elif world_size not in fly_policy.SUPPORTED_WORLDS:
            logger.debug("%s disabled: no schedule for TP%d.", self._tag, world_size)
        else:
            from aiter.jit.utils.chip_info import get_gfx_runtime

            arch = get_gfx_runtime()
            if arch not in fly_policy.SUPPORTED_ARCHS:
                logger.debug("%s disabled: unsupported arch %s.", self._tag, arch)
            else:
                # The link and the policy come from the environment of each
                # process, and a digest of everything the build depends on is
                # what the ranks compare.
                try:
                    link = fly_policy.detect_link()
                    policy = fly_policy.resolve_family_policy(
                        world_size=world_size, arch=arch, link=link
                    )
                    digest = _digest(
                        (
                            "plain",
                            _flydsl_version(),
                            quantization,
                            link,
                            arch,
                            world_size,
                            policy,
                        )
                    )
                except fly_policy.NoPolicyError as e:
                    logger.debug("%s disabled: %s", self._tag, e)
                except ValueError as e:
                    logger.warning("%s disabled: %s", self._tag, e)
        if not vote_on_digest(digest, group, tag=self._tag):
            return

        self.group = group
        self.device = device
        self.rank = dist.get_rank(group=group)
        self.world_size = world_size
        self.link = link
        algorithms = [
            alg
            for alg in policy.algorithms(quantized=True)
            if alg != fly_policy.AllReduceAlgorithm.OneShot
        ]
        if not algorithms:
            logger.debug("%s disabled by the policy.", self._tag)
            return

        rs_codec, ag_codec = FLY_QUANTIZATIONS[quantization]
        ok = True
        try:
            for alg in algorithms:
                # The policy window is the floor here, not the engine's own.
                self.engines[alg] = FlyQuickAllReduce(
                    group=group,
                    device=device,
                    rank=self.rank,
                    world_size=world_size,
                    algorithm=alg.value,
                    rs_codec=rs_codec,
                    ag_codec=ag_codec,
                    min_bytes=0,
                    link=link,
                )
            preload_fly_engines(
                (eng, policy.window(alg, quantized=True).inclusive_range())
                for alg, eng in self.engines.items()
            )
        except Exception:
            logger.warning(
                "%s disabled: engine construction or warmup failed.",
                self._tag,
                exc_info=True,
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

        self.policy = policy
        self.disabled = False
        logger.info(
            "%s enabled: TP%d on %s, %s, %.1f MiB of IPC inbox.",
            self._tag,
            world_size,
            link,
            ", ".join(
                f"{alg.value} {policy.window(alg, quantized=True)}"
                for alg in self.engines
            ),
            self.inbox_bytes / 2**20,
        )

    @property
    def inbox_bytes(self) -> int:
        return sum(e.inbox_bytes for e in self.engines.values())

    def select(self, nbytes: int):
        """The algorithm that runs *nbytes*, or None when this path declines."""
        if self.policy is None:
            return None
        alg = self.policy.select(int(nbytes), quantized=True)
        return alg if alg in self.engines else None

    def should_fly_ar(self, inp) -> bool:
        """Whether this path can and should all-reduce *inp*. Never raises."""
        if self.disabled:
            return False
        if not isinstance(inp, torch.Tensor) or inp.dtype is not torch.bfloat16:
            return False
        # Strictly contiguous: the engines reject weakly-contiguous views.
        if not inp.is_contiguous() or inp.data_ptr() % 16:
            return False
        nbytes = inp.numel() * inp.element_size()
        # Every FlyDSL schedule reads and writes 16 B atoms.
        if nbytes % 16 or nbytes > fly_policy.MAX_PAYLOAD_BYTES:
            return False
        return self.select(nbytes) is not None

    def fly_ar(self, inp, *, out=None):
        """The all-reduce of *inp*, into *out* when given. Only call when
        :meth:`should_fly_ar` said yes."""
        if out is None:
            out = torch.empty_like(inp)
        alg = self.select(inp.numel() * inp.element_size())
        self.engines[alg].allreduce(inp, out)
        return out

    def close(self):
        for eng in self.engines.values():
            try:
                eng.close()
            except (AttributeError, RuntimeError):
                pass
        self.engines = {}
        self.policy = None
        self.disabled = True

    def __del__(self):
        try:
            self.close()
        except (AttributeError, RuntimeError, TypeError):
            pass


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
         It primes the whole payload window of the family it runs, not only
         the rung its own payload selects.

      If none of those happened, :meth:`should_fly_fused_ar_rms` **declines
      while a capture is open** rather than building inside it. The capture then
      records the caller's unfused path, and it cannot hang.

    * Making a (family, hidden) ready is a collective: building exchanges IPC
      handles, and then the ranks agree that every one of them compiled and loaded
      its binaries.
    * If any rank fails to make a width ready, every rank marks it failed and
      the predicate declines it from then on. The caller's next backend serves the call.
    """

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        *,
        quantized: bool,
        hiddens: tuple[int, ...] = (),
        serve: bool = True,
    ):
        self.disabled = True
        self._engines: dict[str, Any] = {}
        # (family, hidden) pairs that are built and have every rung of their
        # payload window JIT-compiled, so a call on them is a pure launch and
        # safe inside a graph capture.
        self._ready: set[tuple[str, int]] = set()
        # (family, hidden) pairs that failed to become ready on some rank. The
        # same on every rank: only the agreement in _prepare adds to it.
        self._failed: set[tuple[str, int]] = set()
        self._warned_capture = False
        self._quantized = bool(quantized)
        self.policy = None
        self._tag = "FlyDSL fused AR+RMSNorm (%s)" % (
            "quantized" if self._quantized else "exact"
        )

        world_size = dist.get_world_size(group=group)
        # Every rank votes below, whatever it decided here: ``digest`` stays
        # declined unless this rank would build.
        digest = _POLICY_DECLINED
        link = policy = None
        if not _FLY_IMPORT_OK:
            logger.debug("%s disabled: flydsl is not importable.", self._tag)
        elif not fly_policy.enabled():
            pass
        elif not serve:
            logger.debug("%s disabled: not served in this configuration.", self._tag)
        elif world_size not in fly_policy.SUPPORTED_WORLDS:
            logger.debug("%s disabled: no schedule for TP%d.", self._tag, world_size)
        else:
            from aiter.jit.utils.chip_info import get_gfx_runtime

            arch = get_gfx_runtime()
            if arch not in fly_policy.SUPPORTED_ARCHS:
                logger.debug("%s disabled: unsupported arch %s.", self._tag, arch)
            else:
                # The link and the policy come from the environment of each
                # process. Building an engine is a collective, so ranks that
                # resolved different policies would deadlock in the IPC
                # exchange. Every rank therefore votes a digest of what it
                # would build (or the declined marker) first, and all of them
                # proceed only when the digests are the same.
                try:
                    link = fly_policy.detect_link()
                    policy = fly_policy.resolve_fused_policy(
                        world_size=world_size, arch=arch, link=link
                    )
                    digest = _policy_digest(
                        quantized=self._quantized,
                        link=link,
                        arch=arch,
                        policy=policy,
                        world_size=world_size,
                        hiddens=hiddens,
                    )
                except fly_policy.NoPolicyError as e:
                    logger.debug("%s disabled: %s", self._tag, e)
                except ValueError as e:
                    logger.warning("%s disabled: %s", self._tag, e)
        if not vote_on_digest(digest, group, tag=self._tag):
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
        # the caller's warmup. A width that fails here is marked failed, and
        # the predicate declines it.
        for h in fly_policy.fused_hiddens(hiddens):
            for family in self._families_for(h):
                self._prepare(family, h)

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
        # The policy window is the floor here, not the engine's own. The
        # engine takes each width's ``atoms_per_row`` from the policy.
        return FlyQuickAllReduceRMSNorm(**common, algorithm=family, min_bytes=0)

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
        """Build every family at *hidden*, and JIT-compile and load the binaries
        its payload window can select.

        Makes a width safe to use inside a HIP graph capture: a call at it is
        then a pure launch.

        Returns False when this width fuses in no family, which is a geometry
        fact (``supports_hidden``), or when it failed to build in any family on
        some rank. The predicate declines the failed families.
        """
        if self.disabled:
            return False
        hidden = int(hidden)
        families = self._families_for(hidden)
        if not families:
            return False

        ready = [self._prepare(fam, hidden) for fam in families]
        return all(ready)

    def _families_for(self, hidden: int) -> list[str]:
        """The families with a build at *hidden*, in engine order."""
        return [
            fam for fam, eng in self._engines.items() if eng.supports_hidden(hidden)
        ]

    def _prepare(self, family: str, hidden: int) -> bool:
        """Make (family, hidden) ready on every rank, or on none of them."""
        key = (family, int(hidden))
        if key in self._ready:
            return True
        if key in self._failed:
            return False
        ok = True
        try:
            self._prime_family(family, hidden)
        except Exception:
            logger.warning(
                "%s: could not build %s at hidden=%d.",
                self._tag,
                family,
                hidden,
                exc_info=True,
            )
            ok = False
        if all_ranks_agree(ok, self.group):
            return True
        self._ready.discard(key)
        self._failed.add(key)
        logger.warning(
            "%s: %s at hidden=%d failed to build on %s; every rank declines "
            "this width.",
            self._tag,
            family,
            hidden,
            "a peer rank" if ok else "this rank",
        )
        return False

    def _prime_family(self, family: str, hidden: int) -> None:
        """Build family at hidden and JIT every rung binary its payload
        window can select."""

        preload_fly_engines(
            [(self._engines[family], self._windows[family].inclusive_range())],
            hidden,
        )
        self._ready.add((family, int(hidden)))

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
        if nbytes > fly_policy.PAD_MASK_MAX_BYTES and eng.pads_hidden(hidden) != hidden:
            return False
        if (family, hidden) in self._failed or self._capture_blocked(family, hidden):
            return False
        return self._prepare(family, hidden)

    def fly_fused_ar_rms(self, inp, residual, weight, eps, *, out=None, res_out=None):
        """``(out, residual_out)``. Only call when the predicate said yes.

        *out*/*res_out* are for callers that already own the destinations.
        """
        nbytes = inp.numel() * inp.element_size()
        family = self.family_for(nbytes)
        eng = self._engines[family]
        hidden = int(inp.shape[-1])
        # The predicate has prepared the width. A caller that skipped it gets
        # the preparation here, and the same error on every rank if it fails.
        if not self._prepare(family, hidden):
            raise RuntimeError(
                f"{self._tag}: {family} at hidden={hidden} failed to build on "
                "at least one rank. The failing rank logged the cause."
            )
        got = eng.allreduce_rmsnorm(
            inp, residual, weight, eps, out=out, residual_out=res_out
        )
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
