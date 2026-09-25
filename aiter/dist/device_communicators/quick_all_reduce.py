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
    from aiter.ops.flydsl.quick_allreduce_int4 import QuickAllReduceInt4

    _FLY_IMPORT_OK = True
except Exception:  # noqa: BLE001
    fly_policy = None
    QuickAllReduceInt4 = None
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

    def should_quick_allreduce_rmsnorm(
        self,
        inp: torch.Tensor,
        residual_inp: torch.Tensor,
        weight: torch.Tensor,
        hidden_dim: int,
    ):
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
        if getattr(self, "_ptr", None):
            if ops is not None:
                ops.qr_destroy(self._ptr)
            self._ptr = 0
        self.disabled = True

    def __del__(self):
        self.close()
