# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host launch for gfx942/gfx950 TP∈{2,4,8} INT4 two-shot all-reduce."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
from flydsl.expr.typing import Float32, Int32, Int64, Stream

from aiter.jit.utils.chip_info import get_gfx_runtime

from .kernels.quick_allreduce_int4 import (
    ATOMS,
    BLOCK,
    DEFAULT_DIVISOR,
    DEFAULT_GRID_CAP,
    SUPER_TILES,
    SUPPORTED_WORLDS,
    TILE_BYTES,
    WORLD,
    clamp_grid_cap,
    latency_max_tiles,
    make_quick_allreduce_int4_kernel,
    mxfp4_scale_shape,
    rmsnorm_hidden_supported,
)
from .kernels.tensor_shim import _run_compiled
from .quick_allreduce_int4_ipc import UncachedIpcHeap

_SUPPORTED_ARCHS = ("gfx942", "gfx950")
# The sum is restored in fp16 before the bf16 store, by scaling each group's
# dequant scale (at least 2**-10) by the divisor; restored values reach
# 480 * divisor. Both must stay normal fp16: 2**-10 * divisor >= 2**-14 and
# 480 * divisor < 65504.
DIVISOR_MIN = 2.0**-4
DIVISOR_MAX = 128.0


def _check_divisor(divisor: float) -> float:
    """One shared positive divisor. ``1`` disables the scaling."""
    msg = (
        f"QuickAllReduceInt4 divisor must be a finite float in "
        f"[{DIVISOR_MIN:g}, {DIVISOR_MAX:g}], got {divisor!r}"
    )
    try:
        value = float(divisor)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(msg) from exc
    if not math.isfinite(value) or not DIVISOR_MIN <= value <= DIVISOR_MAX:
        raise ValueError(msg)
    return value


def _cuda_index(device) -> int:
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if device.type != "cuda":
            raise ValueError(f"QuickAllReduceInt4 requires a CUDA device, got {device}")
        if device.index is None:
            return int(torch.cuda.current_device())
        return int(device.index)
    return int(device)


def _validate_ipc_process_group(group, *, rank: int) -> None:
    """Reject groups that cannot exchange HIP IPC handles or CPU-side metadata."""
    # Keep parallel_state lazy: this module is imported while aiter's AOT setup
    # is still initializing the top-level package.
    from aiter.dist.parallel_state import in_the_same_node_as

    backend = dist.get_backend(group)
    if backend == dist.Backend.NCCL:
        raise ValueError(
            f"QuickAllReduceInt4 does not support NCCL process groups (got {backend!r} on "
            f"group rank {rank}): IPC handle exchange requires CPU-side "
            "broadcast_object_list."
        )

    same_node = in_the_same_node_as(group, source_rank=0)
    if not all(same_node):
        off_node = [r for r, ok in enumerate(same_node) if not ok]
        raise RuntimeError(
            "QuickAllReduceInt4 does not support multi-node process groups: HIP IPC "
            f"handles are node-local (ranks not on rank 0's node: {off_node})."
        )


@dataclass(frozen=True)
class EngineSpec:
    """One extra ``super_tile=1`` build for the plain all-reduce.

    ``max_bytes``: the largest payload sent to it. ``wgs_per_cu``: the
    pinned residency; ``None`` with the default geometry reuses the
    communicator's own ``super_tile=1`` engine.
    """

    max_bytes: int
    threads: int = BLOCK
    tile_atoms: int = ATOMS
    wgs_per_cu: int | None = None

    @property
    def is_default_st1(self) -> bool:
        return (self.threads, self.tile_atoms, self.wgs_per_cu) == (BLOCK, ATOMS, None)


_MiB = 1 << 20
# Plain all-reduce engine plans, by (arch, world_size). Beyond the last entry
# the default super_tile rule applies.
DEFAULT_ENGINE_PLANS: dict[tuple[str, int], tuple[EngineSpec, ...]] = {
    ("gfx950", 2): (
        EngineSpec(32 * _MiB, threads=512, tile_atoms=4, wgs_per_cu=3),
        EngineSpec(512 * _MiB),
    ),
    ("gfx950", 4): (
        EngineSpec(8 * _MiB, threads=256, tile_atoms=4, wgs_per_cu=2),
        EngineSpec(256 * _MiB, threads=512, tile_atoms=8, wgs_per_cu=2),
    ),
    ("gfx950", 8): (EngineSpec(256 * _MiB),),
}


class _StEngine:
    """One compile-time SUPER inbox + launch."""

    def __init__(self, *, spec, group, rank: int, world_size: int, device):
        self.spec = spec
        self.launch = spec["launch"]
        self._norm_launch = {}
        self.super_tile = spec["super_tile"]
        self.grid = spec["grid"]
        self.buf_bytes = spec["flags_bytes"] + spec["data_bytes"]
        self.lds_bytes = spec["lds_bytes"]
        self.tile_bytes = spec["tile_bytes"]
        self.tile_fp16 = spec["tile_fp16"]
        self.rank_tile_bytes = spec["rank_tile_bytes"]
        self.wire_tile_bytes = spec["wire_tile_bytes"]
        self._peer_bases = [None] * world_size
        self._buf_ptr = None
        self.peer_ptrs = [0] * world_size
        # Only this GPU reads or writes the epochs, so they need not be uncached.
        self.colors = torch.ones(self.grid, dtype=torch.int32, device=device)
        try:
            self._buf_ptr = UncachedIpcHeap.alloc_uncached(self.buf_bytes)
            my_handle = UncachedIpcHeap.get_mem_handle_bytes(self._buf_ptr)
            all_meta = UncachedIpcHeap.gather_object_list_via_broadcast(
                group, (my_handle, 0)
            )

            for r in range(world_size):
                handle, off = all_meta[r]
                if r == rank:
                    self.peer_ptrs[r] = self._buf_ptr + off
                else:
                    base = int(UncachedIpcHeap.open_mem_handle(bytes(handle)))
                    self._peer_bases[r] = base
                    self.peer_ptrs[r] = base + off
        except Exception:
            self.close()
            raise

    def norm_launch(
        self,
        hidden: int,
        eps: float,
        gemma_norm: bool,
        latency: bool = False,
        mxfp4: bool = False,
    ):
        """Fused add+RMSNorm variant; shares this engine's inbox and colors."""
        key = (int(hidden), float(eps), bool(gemma_norm), bool(latency), bool(mxfp4))
        launch = self._norm_launch.get(key)
        if launch is None:
            launch = make_quick_allreduce_int4_kernel(
                world_size=self.spec["world_size"],
                super_tile=self.super_tile,
                grid=self.grid,
                norm_hidden=key[0],
                norm_eps=key[1],
                gemma_norm=key[2],
                norm_latency=key[3],
                norm_mxfp4=key[4],
            )["launch"]
            self._norm_launch[key] = launch
        return launch

    def close(self):
        for b in self._peer_bases:
            if b is not None:
                try:
                    UncachedIpcHeap.close_mem_handle(int(b))
                except RuntimeError:
                    pass
        self._peer_bases = []
        self.colors = None
        if self._buf_ptr:
            try:
                UncachedIpcHeap.free_device_mem(self._buf_ptr)
            except RuntimeError:
                pass
            self._buf_ptr = None


class QuickAllReduceInt4:
    """IPC inbox + flag buffer and launch wrapper for ``quick_allreduce_int4``.

    Requires a non-NCCL, single-node process group for IPC metadata exchange.
    """

    def __init__(
        self,
        *,
        group,
        device,
        rank: int,
        world_size: int = WORLD,
        super_tile: int = 8,
        grid_cap: int | None = None,
        engine_plan: tuple[EngineSpec, ...] | None = None,
    ):
        """``engine_plan``: extra builds for the plain all-reduce, tried in
        order by payload size (``EngineSpec``). ``None`` takes
        ``DEFAULT_ENGINE_PLANS`` for this arch and world size; ``()`` keeps
        only the default ``super_tile`` engines. Every rank must pass the
        same plan. The fused RMSNorm calls always use the default engines.
        """
        if world_size not in SUPPORTED_WORLDS:
            raise ValueError(
                f"world_size must be one of {SUPPORTED_WORLDS}, got {world_size}"
            )
        if super_tile not in SUPER_TILES:
            raise ValueError(
                f"super_tile must be one of {SUPER_TILES}, got {super_tile!r}"
            )
        group_world = dist.get_world_size(group=group)
        group_rank = dist.get_rank(group=group)
        if group_world != int(world_size):
            raise ValueError(
                f"world_size={world_size} does not match group size {group_world}"
            )
        if group_rank != int(rank):
            raise ValueError(f"rank={rank} does not match group rank {group_rank}")
        _validate_ipc_process_group(group, rank=int(rank))
        arch = get_gfx_runtime()
        if arch not in _SUPPORTED_ARCHS:
            raise RuntimeError(
                f"QuickAllReduceInt4 supports {', '.join(_SUPPORTED_ARCHS)}, got {arch}"
            )
        # v_cvt_scalef32_pk_fp4_bf16 is gfx950-only.
        self.supports_mxfp4 = arch == "gfx950"
        cap = DEFAULT_GRID_CAP if grid_cap is None else int(grid_cap)
        if cap < 1:
            raise ValueError(f"grid_cap must be positive, got {cap}")
        self._device_index = _cuda_index(device)
        self.group = group
        self.device = torch.device("cuda", self._device_index)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.super_tile = int(super_tile)
        self._has_launched = False
        cu_count = int(
            torch.cuda.get_device_properties(self._device_index).multi_processor_count
        )
        self._latency_max_tiles = latency_max_tiles(cu_count)

        if engine_plan is None:
            engine_plan = DEFAULT_ENGINE_PLANS.get((arch, self.world_size), ())
        self.engine_plan = tuple(engine_plan)
        for prev, cur in zip(self.engine_plan, self.engine_plan[1:]):
            if cur.max_bytes < prev.max_bytes:
                raise ValueError("engine_plan must list max_bytes in increasing order")

        def _shared_grid(st, resident=None):
            grid = clamp_grid_cap(
                cap,
                arch=arch,
                world_size=self.world_size,
                super_tile=st,
                cu_count=cu_count,
                resident=resident,
            )
            shared_grid = torch.tensor(grid, dtype=torch.int64)
            dist.all_reduce(shared_grid, op=dist.ReduceOp.MIN, group=group)
            return int(shared_grid.item())

        def _engine(st, **geometry):
            spec = make_quick_allreduce_int4_kernel(
                world_size=self.world_size,
                super_tile=st,
                grid=_shared_grid(st, geometry.get("resident_wgs_per_cu")),
                **geometry,
            )
            return _StEngine(
                spec=spec,
                group=self.group,
                rank=self.rank,
                world_size=self.world_size,
                device=self.device,
            )

        sts = [1]
        if self.super_tile != 1:
            sts.append(self.super_tile)
        self._by_st = {}
        self._extra = []
        self._plan = []
        try:
            with torch.cuda.device(self._device_index):
                for st in sts:
                    self._by_st[st] = _engine(st)
                extra = {}
                for entry in self.engine_plan:
                    if entry.is_default_st1:
                        eng = self._by_st[1]
                    else:
                        key = (entry.threads, entry.tile_atoms, entry.wgs_per_cu)
                        if key not in extra:
                            extra[key] = _engine(
                                1,
                                threads=entry.threads,
                                tile_atoms=entry.tile_atoms,
                                resident_wgs_per_cu=entry.wgs_per_cu,
                            )
                            self._extra.append(extra[key])
                        eng = extra[key]
                    self._plan.append((int(entry.max_bytes), eng))
        except Exception:
            self.close()
            raise

        primary = self._by_st[self.super_tile]
        self.buf_bytes = primary.buf_bytes
        self.lds_bytes = primary.lds_bytes
        self.tile_bytes = primary.tile_bytes
        self.tile_fp16 = primary.tile_fp16
        self.rank_tile_bytes = primary.rank_tile_bytes
        self.wire_tile_bytes = primary.wire_tile_bytes

    def _pick_st(self, num_tiles: int) -> int:
        if self.super_tile == 1:
            return 1
        st1 = self._by_st.get(1)
        if st1 is not None and num_tiles <= st1.grid:
            return 1
        return self.super_tile

    def _pick_engine(self, live_bytes: int) -> _StEngine:
        for max_bytes, eng in self._plan:
            if live_bytes <= max_bytes:
                return eng
        num_tiles = max(1, (live_bytes + TILE_BYTES - 1) // TILE_BYTES)
        return self._by_st[self._pick_st(num_tiles)]

    def _engines(self):
        return list(self._by_st.values()) + list(getattr(self, "_extra", []))

    def _check_payload(self, inp, out) -> int:
        if not isinstance(inp, torch.Tensor) or not isinstance(out, torch.Tensor):
            raise TypeError("QuickAllReduceInt4 requires torch.Tensor input/output")
        if inp.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
            raise ValueError("QuickAllReduceInt4 supports bf16 input/output")
        if not inp.is_cuda or not out.is_cuda:
            raise ValueError("QuickAllReduceInt4 requires CUDA tensors")
        if (
            inp.device.index != self._device_index
            or out.device.index != self._device_index
        ):
            raise ValueError(
                f"inp/out must be on cuda:{self._device_index}, "
                f"got {inp.device} / {out.device}"
            )
        if not inp.is_contiguous() or not out.is_contiguous():
            raise ValueError("QuickAllReduceInt4 requires contiguous input/output")
        inp_ptr = int(inp.data_ptr())
        out_ptr = int(out.data_ptr())
        if inp_ptr % 16 != 0 or out_ptr % 16 != 0:
            raise ValueError("QuickAllReduceInt4 requires 16-byte-aligned input/output")
        live_bytes = int(inp.numel()) * int(inp.element_size())
        if live_bytes > 0xFFFFFFFF:
            raise ValueError(
                "QuickAllReduceInt4 payload must not exceed the 4 GiB buffer window"
            )
        if live_bytes % 16 != 0:
            raise ValueError("byte size must be a multiple of 16 (8 bf16)")
        if int(out.numel()) * int(out.element_size()) != live_bytes:
            raise ValueError("inp/out byte size mismatch")
        if max(inp_ptr, out_ptr) < min(inp_ptr + live_bytes, out_ptr + live_bytes):
            raise ValueError("QuickAllReduceInt4 requires non-overlapping input/output")
        return live_bytes

    def _launch_args(
        self,
        eng: _StEngine,
        inp,
        out,
        stream,
        *,
        live_bytes: int,
        num_tiles: int,
        grid_x: int,
        norm_ptrs: tuple[int, ...] = (0,) * 5,
        divisor: float = DEFAULT_DIVISOR,
    ):
        if stream is None:
            stream = Stream(torch.cuda.current_stream(self._device_index))
        elif not isinstance(stream, Stream):
            stream = Stream(stream)
        return (
            Int32(self.rank),
            Int64(live_bytes),
            Int32(num_tiles),
            Int64(int(inp.data_ptr())),
            Int64(int(out.data_ptr())),
            Int64(int(eng.colors.data_ptr())),
            *(Int64(p) for p in norm_ptrs),
            *(Int64(p) for p in eng.peer_ptrs + [0] * (WORLD - self.world_size)),
            Int32(grid_x),
            Float32(divisor),
            stream,
        )

    def _launch_eng(
        self,
        eng: _StEngine,
        inp,
        out,
        stream,
        *,
        live_bytes: int,
        launch=None,
        norm_ptrs: tuple[int, ...] = (0,) * 5,
        divisor: float = DEFAULT_DIVISOR,
    ) -> None:
        num_tiles = max(1, (live_bytes + eng.tile_bytes - 1) // eng.tile_bytes)
        args = self._launch_args(
            eng,
            inp,
            out,
            stream,
            live_bytes=live_bytes,
            num_tiles=num_tiles,
            grid_x=min(num_tiles, eng.grid),
            norm_ptrs=norm_ptrs,
            divisor=divisor,
        )
        # A launch may still be using the raw HIP allocations when Python drops
        # the communicator. Keep cleanup conservative even if launch raises.
        self._has_launched = True
        with torch.cuda.device(self._device_index):
            _run_compiled(eng.launch if launch is None else launch, *args)

    def compile(self, inp, out, stream=None) -> None:
        """Eager-JIT every binary of the plain all-reduce.

        Default ST=8 also builds an ST=1 engine for small payloads, and
        the engine plan adds its own builds. Every rank must call this with
        the same ``inp``/``out`` shape. ``out`` is overwritten. The fused
        RMSNorm builds are compiled by ``compile_rmsnorm``.
        """
        live_bytes = self._check_payload(inp, out)
        for eng in self._engines():
            self._launch_eng(eng, inp, out, stream, live_bytes=live_bytes)

    def compile_rmsnorm(
        self,
        hidden: int,
        eps: float,
        *,
        gemma_norm: bool = False,
        mxfp4: bool = False,
        stream=None,
    ) -> None:
        """Eager-JIT every fused build ``allreduce_rmsnorm`` can pick for ``hidden``.

        Without this the first ``allreduce_rmsnorm`` call of each build pays its
        FlyDSL compile: the low-latency and the occupancy-pinned build of the
        ST=1 engine, and the pinned build of the ST=8 engine, are separate
        binaries. Each is launched once on a zero one-row payload, so every rank
        must call this with the same arguments.

        Args:
            hidden: row width; 2048, 4096, 8192 or 16384.
            eps: RMSNorm epsilon of the calls to warm.
            gemma_norm: warm the ``1 + weight`` builds.
            mxfp4: warm the ``allreduce_rmsnorm_mxfp4`` builds (gfx950).
            stream: ``None`` uses the current PyTorch stream on this device.
        """
        live_bytes = int(hidden) * 2
        self._check_rmsnorm_hidden(hidden, live_bytes)
        if mxfp4 and not self.supports_mxfp4:
            raise RuntimeError("fused MXFP4 quant needs gfx950")
        dev = torch.device("cuda", self._device_index)
        inp, out, residual, residual_out, weight = (
            torch.zeros(hidden, dtype=torch.bfloat16, device=dev) for _ in range(5)
        )
        mxfp4_bufs = ()
        if mxfp4:
            mxfp4_bufs = (
                torch.empty(hidden // 2, dtype=torch.uint8, device=dev),
                torch.empty(
                    mxfp4_scale_shape(1, hidden), dtype=torch.uint8, device=dev
                ),
            )
        norm_ptrs = (
            int(residual.data_ptr()),
            int(residual_out.data_ptr()),
            int(weight.data_ptr()),
            *(int(t.data_ptr()) for t in mxfp4_bufs),
        )
        norm_ptrs += (0,) * (5 - len(norm_ptrs))
        for st, eng in self._by_st.items():
            for latency in (True, False) if st == 1 else (False,):
                self._launch_eng(
                    eng,
                    inp,
                    out,
                    stream,
                    live_bytes=live_bytes,
                    launch=eng.norm_launch(hidden, eps, gemma_norm, latency, mxfp4),
                    norm_ptrs=norm_ptrs,
                )
        # The dummy tensors go out of scope here; the launches may be on a
        # stream other than the one that allocated them.
        torch.cuda.synchronize(self._device_index)

    def close(self):
        engines = getattr(self, "_by_st", None)
        if not engines:
            return
        with torch.cuda.device(self._device_index):
            if getattr(self, "_has_launched", False):
                torch.cuda.synchronize(self._device_index)
                self._has_launched = False
            for eng in self._engines():
                eng.close()
            engines.clear()
            self._extra = []
            self._plan = []

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            # Destructors must not raise, especially during interpreter shutdown.
            return

    def allreduce(self, inp, out, stream=None, divisor: float = DEFAULT_DIVISOR):
        """Two-shot INT4 all-reduce into ``out``.

        ``stream=None`` uses the current PyTorch stream on this device.
        ``divisor`` defaults to 1, which keeps the previous INT4 result.
        Loaded values are divided by it before the first pack, and the
        finished sum is multiplied by it once. Every rank must pass the same
        divisor; this is not checked, and a mismatch silently gives each rank
        a different result.
        """
        divisor = _check_divisor(divisor)
        live_bytes = self._check_payload(inp, out)
        eng = self._pick_engine(live_bytes)
        self._launch_eng(eng, inp, out, stream, live_bytes=live_bytes, divisor=divisor)

    @staticmethod
    def supports_rmsnorm(hidden: int, nbytes: int) -> bool:
        """Whether ``allreduce_rmsnorm`` takes this row width and payload."""
        return rmsnorm_hidden_supported(hidden) and nbytes < 1 << 31

    @staticmethod
    def mxfp4_scale_shape(rows: int, hidden: int) -> tuple[int, int]:
        """Shape of the e8m0 ``mxfp4_scale`` buffer for ``rows x hidden``."""
        return mxfp4_scale_shape(rows, hidden)

    def _check_rmsnorm_hidden(self, hidden: int, live_bytes: int) -> None:
        if not self.supports_rmsnorm(hidden, live_bytes):
            # Atom byte offsets travel in the signed 32-bit soffset.
            raise ValueError(
                "fused RMSNorm supports hidden in (2048, 4096, 8192, 16384) "
                f"and payloads under 2 GiB, got hidden={hidden}, "
                f"{live_bytes} bytes"
            )

    def allreduce_rmsnorm(
        self,
        inp,
        residual,
        weight,
        eps: float,
        out,
        residual_out,
        *,
        gemma_norm: bool = False,
        stream=None,
    ):
        """Fused INT4 all-reduce + residual add + RMSNorm.

        ``residual_out = allreduce(inp) + residual`` and
        ``out = rmsnorm(residual_out, eps) * w`` with ``w = weight`` or
        ``1 + weight`` when ``gemma_norm``.

        Args:
            inp: this rank's bf16 partial sum, ``rows x hidden`` elements.
            residual: bf16 residual, same number of elements as ``inp``.
            weight: bf16 RMSNorm weight of ``hidden`` elements; hidden must
                be 2048, 4096, 8192 or 16384.
            eps: RMSNorm epsilon.
            out: bf16 output for the normed rows, same size as ``inp``.
            residual_out: bf16 output for ``allreduce(inp) + residual``; may
                be ``residual`` (in place), must not overlap ``inp`` or
                ``out``.
            gemma_norm: scale by ``1 + weight`` instead of ``weight``.
            stream: ``None`` uses the current PyTorch stream on this device.
        """
        self._allreduce_rmsnorm(
            inp, residual, weight, eps, out, residual_out, gemma_norm, stream
        )

    def allreduce_rmsnorm_mxfp4(
        self,
        inp,
        residual,
        weight,
        eps: float,
        out,
        residual_out,
        mxfp4_out,
        mxfp4_scale,
        *,
        gemma_norm: bool = False,
        stream=None,
    ):
        """``allreduce_rmsnorm`` that also MXFP4-quantizes ``out`` (gfx950).

        Args:
            inp, residual, weight, eps, out, residual_out, gemma_norm,
                stream: as in ``allreduce_rmsnorm``.
            mxfp4_out: 1-byte tensor of ``rows x hidden // 2`` bytes that
                receives the packed FP4 values.
            mxfp4_scale: 1-byte tensor of ``mxfp4_scale_shape(rows, hidden)``
                that receives the shuffled e8m0 scales.

        ``mxfp4_out`` and ``mxfp4_scale`` hold what
        ``per_1x32_f4_quant_hip(out, shuffle=True)`` returns, bit for bit.
        """
        self._allreduce_rmsnorm(
            inp,
            residual,
            weight,
            eps,
            out,
            residual_out,
            gemma_norm,
            stream,
            mxfp4=(mxfp4_out, mxfp4_scale),
        )

    def _allreduce_rmsnorm(
        self,
        inp,
        residual,
        weight,
        eps,
        out,
        residual_out,
        gemma_norm,
        stream,
        mxfp4=None,
    ):
        live_bytes = self._check_payload(inp, out)
        hidden = int(weight.numel())
        self._check_rmsnorm_hidden(hidden, live_bytes)
        if int(inp.numel()) % hidden != 0:
            raise ValueError("numel must be a multiple of hidden")
        for name, t in (
            ("residual", residual),
            ("residual_out", residual_out),
            ("weight", weight),
        ):
            if not isinstance(t, torch.Tensor) or t.dtype != torch.bfloat16:
                raise ValueError(f"{name} must be a bf16 tensor")
            if not t.is_cuda or t.device.index != self._device_index:
                raise ValueError(f"{name} must be on cuda:{self._device_index}")
            if not t.is_contiguous() or int(t.data_ptr()) % 16 != 0:
                raise ValueError(f"{name} must be contiguous and 16-byte aligned")
        if residual.numel() != inp.numel() or residual_out.numel() != inp.numel():
            raise ValueError("residual/residual_out must match inp numel")

        def _overlap(a, b):
            pa, pb = int(a.data_ptr()), int(b.data_ptr())
            return max(pa, pb) < min(pa, pb) + live_bytes

        if _overlap(residual_out, inp) or _overlap(residual_out, out):
            raise ValueError("residual_out must not overlap inp or out")
        if _overlap(residual, out) or (
            _overlap(residual, residual_out)
            and residual.data_ptr() != residual_out.data_ptr()
        ):
            raise ValueError(
                "residual must be residual_out or not overlap it, and must "
                "not overlap out"
            )
        mxfp4_ptrs = (0, 0)
        if mxfp4 is not None:
            if not self.supports_mxfp4:
                raise RuntimeError("fused MXFP4 quant needs gfx950")
            rows = int(inp.numel()) // hidden
            for name, t, nbytes in (
                ("mxfp4_out", mxfp4[0], live_bytes // 4),
                ("mxfp4_scale", mxfp4[1], math.prod(mxfp4_scale_shape(rows, hidden))),
            ):
                if not isinstance(t, torch.Tensor) or t.element_size() != 1:
                    raise ValueError(f"{name} must be a 1-byte tensor")
                if not t.is_cuda or t.device.index != self._device_index:
                    raise ValueError(f"{name} must be on cuda:{self._device_index}")
                if not t.is_contiguous() or t.numel() < nbytes:
                    raise ValueError(f"{name} must be contiguous with {nbytes} bytes")
            mxfp4_ptrs = (int(mxfp4[0].data_ptr()), int(mxfp4[1].data_ptr()))
        num_tiles = max(1, (live_bytes + TILE_BYTES - 1) // TILE_BYTES)
        st = self._pick_st(num_tiles)
        eng = self._by_st[st]
        # One tile per workgroup at <= 2 workgroups per CU: occupancy is
        # free, so take the low-latency build.
        latency = st == 1 and num_tiles <= min(eng.grid, self._latency_max_tiles)
        self._launch_eng(
            eng,
            inp,
            out,
            stream,
            live_bytes=live_bytes,
            launch=eng.norm_launch(hidden, eps, gemma_norm, latency, mxfp4 is not None),
            norm_ptrs=(
                int(residual.data_ptr()),
                int(residual_out.data_ptr()),
                int(weight.data_ptr()),
                *mxfp4_ptrs,
            ),
        )
