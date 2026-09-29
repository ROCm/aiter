# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import os
from enum import Enum
from typing import Any, ClassVar

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import aiter as ops

from ..parallel_state import in_the_same_node_as
from .flydsl_utils import all_ranks_agree, warm_fly_engines

logger = logging.getLogger(__name__)


class QuickReduceRegime(Enum):
    # Keep integer ids aligned with csrc/include/quick_all_reduce.cuh
    FP = 0
    FP8 = 1
    INT6 = 2
    INT4 = 3
    INT3 = 4
    NONE = 5


try:
    quick_ar = False
    regime_str = os.environ.get("AITER_QUICK_REDUCE_QUANTIZATION", None)
    if regime_str in QuickReduceRegime.__members__:
        ops.qr_max_size()
        quick_ar = True
except Exception:  # noqa: BLE001
    # For CPUs and CUDA
    quick_ar = False


# Importing these pulls in flydsl, an optional dependency whose version is
# checked at import time.
try:
    from aiter.ops.flydsl import allreduce_policy as fly_policy
    from aiter.ops.flydsl.kernels.quick_allreduce_fusions import fused_qr_row_atoms
    from aiter.ops.flydsl.one_shot_allreduce import OneShotAllReduceRMSNorm
    from aiter.ops.flydsl.quick_allreduce import (
        FlyQuickAllReduce as QuickAllReduceInt4,
        FlyQuickAllReduceRMSNorm as QuickAllReduceInt4RMSNorm,
    )

    _FLY_IMPORT_OK = True
except Exception:  # noqa: BLE001
    fly_policy = None
    QuickAllReduceInt4 = None
    QuickAllReduceInt4RMSNorm = None
    OneShotAllReduceRMSNorm = None
    fused_qr_row_atoms = None
    _FLY_IMPORT_OK = False

_FLY_SUPPORTED_ARCHS = ("gfx942", "gfx950")

# Quantization regimes the FlyDSL backend can serve, mapped to the wire formats
# to pin on the ring's two laps. ``None`` means "use the schedule's own per-world
# default".
_FLY_REGIMES = {QuickReduceRegime.INT4: (None, None)}


def qr_rocm_arch_available():
    try:
        props = torch.cuda.get_device_properties(0)
        gcn_arch = getattr(props, "gcnArchName", "")
        supported_archs = ["gfx94", "gfx95"]
        return any(gfx in gcn_arch for gfx in supported_archs)
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to determine ROCm for quick allreduce: %s", e)
        return False


def is_weak_contiguous(inp: torch.Tensor):
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


def qr_exchange_handles(ptr, world_size, group):
    # 64 == sizeof(hipIpcMemHandle_t); must be host memory (qr_get_handle memcpys into data_ptr)
    handle = torch.empty(64, dtype=torch.uint8, device="cpu")
    ops.qr_get_handle(ptr, handle.data_ptr())
    handles = [None] * world_size
    dist.all_gather_object(handles, handle, group=group)
    ops.qr_open_handles(ptr, [h.data_ptr() for h in handles])


MB = 1024 * 1024


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

        if not _FLY_IMPORT_OK or not fly_policy.enabled():
            return

        from aiter.jit.utils.chip_info import get_gfx_runtime

        arch = get_gfx_runtime()
        if arch not in _FLY_SUPPORTED_ARCHS:
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

        link = fly_policy.detect_link()
        try:
            resolved = fly_policy.resolve_fused(link, world_size)
        except ValueError as e:
            logger.warning("FlyDSL fused AR+RMSNorm disabled: %s", e)
            return

        self.group = group
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.link = link
        self.policy = resolved
        self.accuracy = fly_policy.accuracy_mode()

        try:
            for family in fly_policy.fused_families_reachable(resolved):
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
        for h in fly_policy.fused_hiddens(hiddens):
            try:
                self.prime(h)
            except Exception:
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
            "pad": fly_policy.fused_pad_enabled(),
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
        return fly_policy.pick_fused_family(int(nbytes), self.policy)

    def variant(self, hidden: int, nbytes: int) -> str:
        family = self.family_for(int(nbytes))
        eng = self._engines.get(family)
        if eng is None:
            return family
        got = f"{family}:{eng.variant(int(hidden), int(nbytes))}"
        h_pad = eng.pads_hidden(int(hidden))
        return got if h_pad == int(hidden) else f"{got}/pad{h_pad}"

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
        max_b = self.policy.max_bytes
        if max_b is not None and nbytes > max_b:
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


class QuickAllReduce:
    _SUPPORTED_WORLD_SIZES: ClassVar[list[Any]] = [2, 4, 8]
    _SUPPORTED_DTYPES: ClassVar[list[Any]] = [torch.float16, torch.bfloat16]
    # The following data is based on kernel tests.
    # In this order [FP, FP8, INT6, INT4, INT3].
    # INT3 is TP2-only; its entries for world_size 4/8 are unused but kept
    # to keep the per-quant-level list indexable by QuickReduceRegime.value.
    _QR_MIN_SIZE: ClassVar[dict[str, Any]] = {
        (torch.float16, 2): [1 * MB, 2 * MB, 2 * MB, 1 * MB, 1 * MB],
        (torch.float16, 4): [1 * MB, 16 * MB, 4 * MB, 2 * MB, 2 * MB],
        (torch.float16, 8): [16 * MB, 4 * MB, 4 * MB, 8 * MB, 8 * MB],
        (torch.bfloat16, 2): [2 * MB, 8 * MB, 8 * MB, 8 * MB, 8 * MB],
        (torch.bfloat16, 4): [8 * MB, 64 * MB, 64 * MB, 16 * MB, 16 * MB],
        (torch.bfloat16, 8): [16 * MB, 2048 * MB, 2048 * MB, 2048 * MB, 2048 * MB],
    }

    def __init__(self, group: ProcessGroup, device: int | str | torch.device) -> None:
        """
        Quick allreduce leverages quantization for further
        acceleration on ROCm. It currently supports FP8, Q6, Q4, and Q3
        quantization formats and FP(float16, bfloat16). Q3 (INT3) is
        restricted to TP2 (world_size == 2) due to poor performance on
        larger world sizes.
        Quick allreduce is designed as a complement to custom allreduce.
        Its initialization requires even stricter conditions.
        Only the ROCm MI300 series is supported for quick allreduce at
        this time.
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the CustomAllreduce to. If None,
                it will be bind to f"cuda:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self.disabled = True
        self._fly_policy = None
        self._fly_engines: dict[str, Any] = {}
        self._fly_rms = None
        if not qr_rocm_arch_available():
            logger.debug(
                "Custom quick allreduce is only supported on ROCm MI300 series."
            )
            return

        if not quick_ar:
            return

        self.group = group
        assert (
            dist.get_backend(group) != dist.Backend.NCCL
        ), "Custom quick allreduce should be attached to a non-NCCL group."
        if not all(in_the_same_node_as(group, source_rank=0)):
            # No need to initialize custom quick allreduce for
            # multi-node case.
            logger.warning(
                "Custom quick allreduce is disabled because this "
                "process group spans across nodes."
            )
            return
        rank = dist.get_rank(group=self.group)
        world_size = dist.get_world_size(group=self.group)
        self.rank = rank
        self.world_size = world_size
        if world_size == 1:
            # No need to initialize QuickReduce for single GPU case.
            return

        if world_size not in QuickAllReduce._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "Custom quick allreduce is disabled due to an "
                "unsupported world size: %d. Supported world sizes: %s.",
                world_size,
                str(QuickAllReduce._SUPPORTED_WORLD_SIZES),
            )
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.device = device

        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible_devices:
            device_ids = list(map(int, cuda_visible_devices.split(",")))
        else:
            device_ids = list(range(torch.cuda.device_count()))
        physical_device_id = device_ids[device.index]
        tensor = torch.tensor([physical_device_id], dtype=torch.int, device="cpu")
        gather_list = [
            torch.tensor([0], dtype=torch.int, device="cpu")
            for _ in range(self.world_size)
        ]
        dist.all_gather(gather_list, tensor, group=self.group)
        [t.item() for t in gather_list]

        # test nvlink first, this will filter out most of the cases
        # where custom quick allreduce is not supported
        # this checks hardware and driver support for NVLink

        # self.fully_connected = is_full_nvlink(physical_device_ids, self.world_size)
        self.fully_connected = True
        if self.world_size > 2 and not self.fully_connected:
            logger.debug(
                "Custom quick allreduce is disabled because it's not supported "
                "on more than two PCIe-only GPUs. "
            )
            return

        self.init_quick_all_reduce()
        self._init_fly_backend()
        self._init_fly_rms_backend()

    def _init_fly_rms_backend(self):
        """Build the FlyDSL fused all-reduce+RMSNorm engines.

        Same opt-in (``AITER_FLY_AR``) as the plain FlyDSL backend, but its own
        engine set: the fused tile is one token row, so engines are per
        (family, hidden) and widths are built up front (declared via
        ``AITER_FLY_AR_FUSED_HIDDENS``) or on first use. All the lifecycle --
        build, prime, capture-guard, dispatch -- lives in
        :class:`FlyDSLAllReduceRMSNorm`, which this just owns."""
        if not _FLY_IMPORT_OK or not fly_policy.enabled():
            return
        try:
            rms = FlyDSLAllReduceRMSNorm(group=self.group, device=self.device)
        except Exception:
            logger.warning(
                "FlyDSL fused AR+RMSNorm backend disabled: construction failed.",
                exc_info=True,
            )
            return
        if rms.disabled:
            rms.close()
            return
        self._fly_rms = rms

    def _init_fly_backend(self):
        """Build the FlyDSL mesh/ring engines that will serve plain all-reduce,
        and compile the binaries the policy routes to them before any graph
        capture begins."""

        if not _FLY_IMPORT_OK or not fly_policy.enabled():
            return
        if getattr(self, "qr_quant_level", None) not in _FLY_REGIMES:
            return

        from aiter.jit.utils.chip_info import get_gfx_runtime

        arch = get_gfx_runtime()
        if arch not in _FLY_SUPPORTED_ARCHS:
            logger.debug(
                "FlyDSL quick-reduce backend disabled: unsupported arch %s.", arch
            )
            return

        rs_codec, ag_codec = _FLY_REGIMES[self.qr_quant_level]
        link = fly_policy.detect_link()
        ok = True
        try:
            policy = fly_policy.resolve_quant(link, self.world_size)
            for family in fly_policy.quant_families_reachable(policy):
                self._fly_engines[family] = QuickAllReduceInt4(
                    group=self.group,
                    device=self.device,
                    rank=self.rank,
                    world_size=self.world_size,
                    algorithm=family,
                    rs_codec=rs_codec,
                    ag_codec=ag_codec,
                    min_bytes=0,
                    link=link,
                )
        except Exception:
            logger.warning(
                "FlyDSL quick-reduce backend disabled: engine construction failed.",
                exc_info=True,
            )
            ok = False

        if not all_ranks_agree(ok, self.group):
            if ok:
                logger.warning(
                    "FlyDSL quick-reduce backend disabled: a peer rank failed to "
                    "build its engines."
                )
            self._close_fly_engines()
            return

        ok = True
        try:
            warm_fly_engines(
                (
                    (engine, fly_policy.quant_family_range(family, policy))
                    for family, engine in self._fly_engines.items()
                ),
                self.device,
            )
        except Exception:
            logger.warning(
                "FlyDSL quick-reduce backend disabled: warmup failed.", exc_info=True
            )
            ok = False

        if not all_ranks_agree(ok, self.group):
            if ok:
                logger.warning(
                    "FlyDSL quick-reduce backend disabled: a peer rank failed to "
                    "warm up."
                )
            self._close_fly_engines()
            return

        self._fly_policy = policy
        logger.info(
            "Quick-reduce slot: FlyDSL %s serves plain all-reduce (TP%d on %s, "
            "%d KiB < bytes <= %s, %.1f MiB of IPC inbox).",
            "+".join(self._fly_engines),
            self.world_size,
            link,
            policy.floor >> 10,
            (
                "no limit"
                if policy.mesh_max >= fly_policy.NO_MAX
                else f"{policy.mesh_max >> 10} KiB"
            ),
            sum(e.inbox_bytes for e in self._fly_engines.values()) / 2**20,
        )

    def _close_fly_engines(self):
        for eng in getattr(self, "_fly_engines", {}).values():
            try:
                eng.close()
            except (AttributeError, RuntimeError):
                pass
        self._fly_engines = {}
        self._fly_policy = None

    def init_quick_all_reduce(self):
        # On RocM, bfloat16 kernels are slower than fp16
        # due to slower match operations
        # If environment variable is set to 1, we convert input to fp16
        self.use_fp16_kernels = int(
            os.environ.get("AITER_QUICK_REDUCE_CAST_BF16_TO_FP16", "1")
        )
        regime_str = os.environ.get("AITER_QUICK_REDUCE_QUANTIZATION", "NONE")
        if regime_str not in QuickReduceRegime.__members__:
            logger.warning(
                "Custom quick allreduce: "
                f"Invalid quantization level: {regime_str}. "
                "Supported levels: "
                f"{list(QuickReduceRegime.__members__.keys())}",
            )
            return

        if regime_str == "NONE":
            logger.debug(
                "Custom quick allreduce is disabled based "
                "on env variable "
                "AITER_QUICK_REDUCE_QUANTIZATION='NONE'"
            )
            return
        self.qr_quant_level = QuickReduceRegime[regime_str]

        # INT3 is only enabled for TP2 (world_size == 2).
        # Kernel benchmarks show INT3 all-reduce on TP4/TP8 has poor
        # performance (the extra ranks make the 3-bit codec's pack/unpack
        # overhead outweigh the reduced communication volume), so INT3 is
        # restricted to 2-GPU tensor parallelism. For TP4/TP8 use a wider
        # codec (e.g. INT4) or NONE instead.
        if self.qr_quant_level == QuickReduceRegime.INT3 and self.world_size != 2:
            logger.warning(
                "Custom quick allreduce is disabled: INT3 quantization is "
                "only supported for TP2 (world_size == 2), but world_size "
                "is %d. INT3 on TP4/TP8 is disabled due to poor kernel "
                "performance. Use INT4/NONE for this world size.",
                self.world_size,
            )
            return

        # TODO: If the dtype is not bfloat16 or then float16,
        # quickallreduce should not be created.

        # AITER_QUICK_REDUCE_MAX_SIZE_BYTES_MB is specified in MB
        qr_max_size = int(os.environ.get("AITER_QUICK_REDUCE_MAX_SIZE_BYTES_MB", "0"))
        if qr_max_size > 0:
            if qr_max_size < 1:
                logger.info(
                    "You should not set a max_size smaller than 1MB, which can "
                    "lead to error or degradation to custom allreduce or rccl."
                )
            qr_max_size = qr_max_size * MB
        # If qr_max_size is None, then 2GB is used by default.
        self._ptr = ops.init_custom_qr(self.rank, self.world_size, qr_max_size)
        self.qr_max_size = qr_max_size if qr_max_size > 0 else ops.qr_max_size()
        self.create_shared_buffer()
        self.disabled = False

    def create_shared_buffer(self):
        """
        Creates a shared buffer for quickreduce.
        Has to be called after init_custom_qr
        """
        world_size = dist.get_world_size(group=self.group)
        qr_exchange_handles(self._ptr, world_size, self.group)

    def _should_fly(self, inp: torch.Tensor) -> bool:
        """Whether the FlyDSL backend can and should serve *inp*. Never raises."""

        if self._fly_policy is None:
            return False
        if inp.dtype is not torch.bfloat16:
            return False
        if not inp.is_contiguous():
            return False
        if inp.data_ptr() % 16:
            return False
        nbytes = inp.numel() * inp.element_size()
        if nbytes % 16:
            return False
        p = self._fly_policy
        if not p.floor < nbytes <= p.max_bytes:
            return False

        return fly_policy.pick_quant_family(nbytes, p) in self._fly_engines

    def should_quick_allreduce(self, inp: torch.Tensor):
        """
        Check if quickreduce is available
        """
        if self.disabled:
            return False
        return self._should_fly(inp) or self._should_hip(inp)

    def _should_hip(self, inp: torch.Tensor) -> bool:
        """Whether the HIP quick-reduce kernel can serve *inp*."""

        if self.disabled:
            return False
        if inp.dtype not in self._SUPPORTED_DTYPES:
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom quick allreduce requires input byte size to be
        # multiples of 16
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        dtype = inp.dtype
        if self.use_fp16_kernels:
            dtype = torch.float16
        return (
            inp_size <= self.qr_max_size
            and inp_size
            >= self._QR_MIN_SIZE[(dtype, self.world_size)][self.qr_quant_level.value]
        )

    def quick_all_reduce(self, inp: torch.Tensor, *, out: torch.Tensor = None):
        """Performs an out-of-place custom quick all reduce."""
        # quick allreduce doesn't require a separate graph mode,
        # as QR uses static IPC buffer. The same holds for the FlyDSL
        # schedules, whose IPC inbox is likewise allocated once at init
        # and whose served binaries are compiled at init (warm_fly_engines).
        if out is None:
            out = torch.empty_like(inp)
        if self._should_fly(inp):
            nbytes = inp.numel() * inp.element_size()
            family = fly_policy.pick_quant_family(nbytes, self._fly_policy)
            self._fly_engines[family].allreduce(inp, out)
            return out
        ops.qr_all_reduce(
            self._ptr, inp, out, self.qr_quant_level.value, self.use_fp16_kernels
        )
        return out

    def should_fly_allreduce_rmsnorm(
        self,
        inp: torch.Tensor,
        residual_inp: torch.Tensor,
        weight: torch.Tensor,
    ) -> bool:
        """Whether the FlyDSL fused AR+RMSNorm backend can serve this call.

        Never raises; delegates the geometry/capture checks to the backend."""
        if self._fly_rms is None:
            return False
        return self._fly_rms.should_fly_fused_ar_rms(inp, residual_inp, weight)

    def should_quick_allreduce_rmsnorm(
        self,
        inp: torch.Tensor,
        residual_inp: torch.Tensor,
        weight: torch.Tensor,
        hidden_dim: int,
    ):
        if self.disabled:
            return False
        # Prefer the FlyDSL fused schedules where they apply; otherwise fall
        # through to the HIP quick-reduce fused kernel.
        if self.should_fly_allreduce_rmsnorm(inp, residual_inp, weight):
            return True
        if not self._should_hip(inp):
            return False
        if inp.dtype != residual_inp.dtype or inp.dtype != weight.dtype:
            return False
        if not is_weak_contiguous(residual_inp) or not is_weak_contiguous(weight):
            return False
        if weight.numel() != hidden_dim or inp.numel() % hidden_dim != 0:
            return False

        row_size = hidden_dim * inp.element_size()
        tile_size = 32 * 1024
        return row_size > 0 and row_size <= tile_size and tile_size % row_size == 0

    def quick_all_reduce_rmsnorm(
        self,
        inp: torch.Tensor,
        residual_inp: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        hidden_dim: int,
    ):
        """Performs QR allreduce fused with residual add and RMSNorm."""
        # The FlyDSL fused path handles the call when its predicate says yes,
        # mirroring the plain all-reduce dispatch. The predicate is
        # deterministic, so re-checking here matches should_quick_allreduce_rmsnorm.
        if self.should_fly_allreduce_rmsnorm(inp, residual_inp, weight):
            return self._fly_rms.fly_fused_ar_rms(inp, residual_inp, weight, eps)
        out = torch.empty_like(inp)
        residual_out = torch.empty_like(residual_inp)
        ops.qr_all_reduce_rmsnorm(
            self._ptr,
            inp,
            residual_inp,
            residual_out,
            out,
            weight,
            eps,
            hidden_dim,
            self.qr_quant_level.value,
            self.use_fp16_kernels,
        )
        return out, residual_out

    def close(self):
        # Both backends, and not gated on self.disabled: the FlyDSL engines hold
        # IPC inboxes opened against every peer, which have to be released
        # before the process group goes away even if the slot is disabled.
        self._close_fly_engines()
        if getattr(self, "_fly_rms", None) is not None:
            try:
                self._fly_rms.close()
            except (AttributeError, RuntimeError):
                pass
            self._fly_rms = None
        if getattr(self, "_ptr", None):
            if ops is not None:
                ops.qr_destroy(self._ptr)
            self._ptr = 0
        self.disabled = True

    def __del__(self):
        self.close()
