# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host launch for the exact one-shot (1-stage) all-reduce.

Public types ``OneShotAllReduce`` (plain) and ``OneShotAllReduceRMSNorm``
(fused with residual-add + RMSNorm). Decode-only by policy: one round and no
grid-wide barrier, paid for with ``(N-1)*S`` of wire volume against the
mesh's ``2(N-1)/N*S``.
"""

from __future__ import annotations

import dataclasses
import logging

import torch
from flydsl.expr.typing import Float32, Int32, Int64, Stream

from .allreduce_host_shared import (
    _cuda_index,
    _resolve_inbox_flags,
    _StEngine,
    _validate_ipc_process_group,
    check_fused_aliasing,
    check_payload,
    check_world_size,
    kernel_symbol,
    payload_probes,
    preload_launcher,
    require_arch,
)
from .allreduce_policy import (
    PAD_MASK_MAX_BYTES,
    SUPPORTED_ATOMS,
    SUPPORTED_SPLITS,
    AllReduceAlgorithm,
    OneShotRung,
    fused_oneshot_ladder,
    fused_split_enabled,
    oneshot_ladder,
    resolve_family_policy,
    resolve_fused_policy,
    resolve_link,
    select_rung,
)
from .kernels.comm.one_shot_allreduce import make_one_shot_allreduce_kernel
from .kernels.comm.one_shot_allreduce_rmsnorm import (
    fused_block_options,
    fused_padded_block_options,
    fused_split_options,
)
from .kernels.tensor_shim import _run_compiled

logger = logging.getLogger("aiter")


def resolve_oneshot_max_bytes(
    *, world_size: int, link: str | None = None, arch: str | None = None
) -> int:
    """Largest payload this kernel should be asked to move, inclusive.

    The maximum of the exact-mode one-shot window of this rank's policy, with
    the environment overrides applied.
    """
    window = resolve_family_policy(
        world_size=int(world_size), arch=arch, link=link
    ).window(AllReduceAlgorithm.OneShot, quantized=False)
    return 0 if window is None else window.inclusive_range()[1]


def resolve_fused_oneshot_max_bytes(
    *, world_size: int, link: str | None = None, arch: str | None = None
) -> int:
    """Largest payload the fused one-shot should be asked to move, inclusive.

    The maximum of the one-shot window of this rank's fused policy, with the
    environment overrides applied.
    """
    window = resolve_fused_policy(
        world_size=int(world_size), arch=arch, link=link
    ).window(AllReduceAlgorithm.OneShot, quantized=False)
    return 0 if window is None else window.inclusive_range()[1]


#: The AttnRes-only kernel arguments (bank, cw, bank rows per token, score eps)
#: as every build that does not read them passes them.
_NO_ATTN_RES_ARGS = (Int64(0), Int64(0), Int32(0), Float32(0.0), Int64(0), Int64(0))


class OneShotAllReduce:
    """IPC inbox + launch wrapper for ``one_shot_allreduce``.

    The kernel configuration comes from the one-shot tuning ladder of the
    policy (``oneshot_ladder``). ``ladder`` overrides it.

    ``max_bytes`` is the payload above which this ``allreduce`` refuses to run,
    defaulting to ``resolve_oneshot_max_bytes``.

    ``link`` is the interconnect type and it is detected from the KFD topology
    when not explicitly given.

    ``inbox_memory`` type: ``"auto"`` picks ``uncached`` on xGMI
    hosts and ``finegrained`` on PCIe ones. One exception: at TP2 it picks
    ``uncached`` on PCIe too, since a single remote destination cannot collapse.

    A rank never round-trips through its own inbox: its own contribution stays
    in registers. That specialises the kernel to this rank, so the JIT symbol
    carries an ``_r<n>_`` field and the binary is not shared across ranks.
    """

    def __init__(
        self,
        *,
        group,
        device,
        rank: int,
        world_size: int,
        inbox_memory: str = "auto",
        max_bytes: int | None = None,
        link: str | None = None,
        ladder: tuple[OneShotRung, ...] | None = None,
    ):
        check_world_size(world_size)
        link = resolve_link(link)
        self.link = link
        _validate_ipc_process_group(
            group, rank=int(rank), world_size=world_size, owner="OneShotAllReduce"
        )
        arch = require_arch(owner="OneShotAllReduce")
        if ladder is None:
            ladder = oneshot_ladder(world_size=world_size, link=link, arch=arch)
        if not ladder:
            raise ValueError("ladder must have at least one rung")
        self._ladder = tuple(ladder)

        inbox_flags, resolved_inbox = _resolve_inbox_flags(
            inbox_memory, world_size, link
        )
        self._device_index = _cuda_index(device, owner="OneShotAllReduce")
        self.group = group
        self.device = torch.device("cuda", self._device_index)
        self._has_launched = False
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.inbox_memory = resolved_inbox
        self.max_bytes = (
            resolve_oneshot_max_bytes(world_size=world_size, link=link, arch=arch)
            if max_bytes is None
            else int(max_bytes)
        )

        # One engine per distinct rung config, built in ladder order: each does
        # its own IPC handle exchange, which is a collective, so ranks
        # disagreeing on the order would deadlock. The ladder is the same table
        # on every rank, so the order is too.
        #
        # No ``clamp_grid_cap`` here, unlike the two-shot schedules: no rung
        # caps the grid above 128, which is more than an order of magnitude
        # under the residency of any supported device, so the clamp could only
        # ever be a no-op bought with an extra collective per engine.
        self._by_cfg = {}
        try:
            with torch.cuda.device(self._device_index):
                for rung in self._ladder:
                    key = rung.cfg
                    if key in self._by_cfg:
                        continue
                    spec = make_one_shot_allreduce_kernel(
                        world_size=self.world_size,
                        atoms=key[0],
                        grid=key[1],
                        inbox_memory=resolved_inbox,
                        fanout=key[2],
                        block=key[3],
                        rank=self.rank,
                    )
                    self._by_cfg[key] = (
                        _StEngine(
                            spec=spec,
                            group=group,
                            rank=self.rank,
                            world_size=self.world_size,
                            inbox_flags=inbox_flags,
                        ),
                        spec,
                    )
        except Exception:
            self.close()
            raise

        # The first rung's shape, reported as this object's own: it describes
        # the smallest payloads.
        first = self._ladder[0].cfg
        eng, spec = self._by_cfg[first]
        self.atoms = first[0]
        self.grid_cap = first[1]
        self.fanout = first[2]
        self.block = first[3]
        self.tile_bytes = spec["tile_bytes"]
        self.wire_tile_bytes = spec["wire_tile_bytes"]
        self.buf_bytes = eng.buf_bytes

    @property
    def inbox_bytes(self) -> int:
        """IPC inbox bytes this object holds on this rank, across every rung."""
        return sum(eng.buf_bytes for eng, _ in self._by_cfg.values())

    def _pick_cfg(self, live_bytes: int) -> tuple:
        """``(atoms, grid_cap, fanout, block)`` the ladder assigns to
        *live_bytes*."""
        return select_rung(self._ladder, live_bytes).cfg

    @staticmethod
    def _num_tiles(live_bytes: int, tile_bytes: int) -> int:
        return max(1, (live_bytes + tile_bytes - 1) // tile_bytes)

    @staticmethod
    def _grid_x(num_tiles: int, grid_cap: int) -> int:
        return max(1, min(num_tiles, grid_cap))

    def _launch_args(self, eng, spec, inp_ptr, out_ptr, stream, *, live_bytes):
        num_tiles = self._num_tiles(live_bytes, spec["tile_bytes"])
        grid_x = self._grid_x(num_tiles, spec["grid"])
        if stream is None:
            stream = Stream(torch.cuda.current_stream(self._device_index))
        elif not isinstance(stream, Stream):
            stream = Stream(stream)
        return (
            Int32(self.rank),
            Int64(live_bytes),
            Int32(num_tiles),
            Int64(inp_ptr),
            Int64(out_ptr),
            Int64(int(eng._gpu_peer_ptrs)),
            Int64(int(eng._colors)),
            Int32(grid_x),
            # The fused epilogue's operands, then the split build's exchange
            # state. One kernel signature serves both modes (see the factory),
            # and a plain build's ``const_expr(fused)`` arms never read these.
            Int64(0),
            Int64(0),
            Int64(0),
            Float32(0.0),
            Int64(0),
            # The AttnRes build's bank, cw, bank rows per token and score eps.
            *_NO_ATTN_RES_ARGS,
            stream,
        )

    def _launch_eng(self, eng, spec, inp, out, stream, *, live_bytes: int) -> None:
        args = self._launch_args(
            eng,
            spec,
            int(inp.data_ptr()),
            int(out.data_ptr()),
            stream,
            live_bytes=live_bytes,
        )
        # A launch may still be using the raw HIP allocations when Python drops
        # the communicator. Keep cleanup conservative even if launch raises.
        self._has_launched = True
        with torch.cuda.device(self._device_index):
            _run_compiled(eng.launch, *args)

    def _launch(self, inp, out, stream, *, live_bytes: int) -> None:
        eng, spec = self._by_cfg[self._pick_cfg(live_bytes)]
        self._launch_eng(eng, spec, inp, out, stream, live_bytes=live_bytes)

    def cfgs_for(self, lo: int, hi: int) -> list[tuple]:
        """``_by_cfg`` keys a payload of ``lo..hi`` bytes (inclusive) can
        select, in build order."""
        floors = [rung.min_bytes for rung in self._ladder]
        picked = {self._pick_cfg(n) for n in payload_probes(floors, lo, hi)}
        return [key for key in self._by_cfg if key in picked]

    def preload(self, *, payload_range=None) -> None:
        """JIT-compile rung binaries and load them onto the device, without
        launching any of them.

        ``payload_range=(lo, hi)`` takes only the rungs ``allreduce`` would run
        for a payload of ``lo..hi`` bytes (inclusive); ``None`` takes every one.
        The ladder builds engines for the whole size range, while a dispatcher
        routes only its own window here, so the rest never run.

        Local to this rank, not a collective. ``CustomAllreduce`` calls it once
        at init, via ``preload_fly_engines``, to keep each rung's JIT compile and
        module load off the first real all-reduce, which may be inside a HIP
        graph capture.
        """
        keys = self._by_cfg if payload_range is None else self.cfgs_for(*payload_range)
        with torch.cuda.device(self._device_index):
            for key in keys:
                eng, spec = self._by_cfg[key]
                args = self._launch_args(eng, spec, 0, 0, None, live_bytes=0)
                preload_launcher(eng.launch, *args)

    def variant(self, nbytes: int) -> str:
        """Identity of the binary an *nbytes* payload would run."""
        cfg = self._pick_cfg(int(nbytes))
        eng, spec = self._by_cfg[cfg]
        grid_x = self._grid_x(self._num_tiles(int(nbytes), spec["tile_bytes"]), cfg[1])
        return f"{kernel_symbol(eng.launch)}/grid_x{grid_x}"

    def is_beneficial(self, nbytes: int) -> bool:
        return int(nbytes) <= self.max_bytes

    def allreduce(self, inp, out, stream=None):
        live_bytes = check_payload(
            inp, out, device_index=self._device_index, owner="OneShotAllReduce"
        )
        if not self.is_beneficial(live_bytes):
            raise ValueError(
                f"OneShotAllReduce got a {live_bytes} B payload, above the "
                f"{self.max_bytes} B ceiling: this kernel pushes the whole "
                "payload to every peer, so its wire volume is (N-1)x the "
                "message where a mesh schedule moves 2(N-1)/N. Route large messages "
                "to FlyQuickAllReduce or cross_device_reduce, or pass max_bytes to "
                "override."
            )
        self._launch(inp, out, stream, live_bytes=live_bytes)

    def close(self):
        engines = getattr(self, "_by_cfg", None)
        if not engines:
            return
        with torch.cuda.device(self._device_index):
            if getattr(self, "_has_launched", False):
                torch.cuda.synchronize(self._device_index)
                self._has_launched = False
            for eng, _ in engines.values():
                eng.close()
            engines.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            # Destructors must not raise, especially during interpreter shutdown.
            return


class OneShotAllReduceRMSNorm:
    """One-shot all-reduce fused with residual-add and RMSNorm.

    Per token row:

        acc = sum_r input_r            (fp32, rank order)
        acc = float(bf16(acc))         (deliberate: matches the unfused path)
        acc += residual_in
        residual_out = bf16(acc)
        out = bf16(acc * rsqrt(sum(acc^2)/hidden + eps) * weight)

    The wire layout depends on ``hidden``, because the tile is pinned to one
    token row so the norm's reduction fits in one workgroup. A new hidden dim
    therefore needs a new engine, and building one is a collective (IPC handle exchange).
    Engines are built lazily on first use, which is safe because TP ranks enter
    a collective with the same shape in lockstep. Passing ``hiddens=(...)`` builds
    them up front.

    ``split`` spreads each row over that many workgroups instead (``block *
    atoms * 8 * split == hidden``), which joins the row's sum of squares through
    one HBM exchange word; at decode sizes one workgroup per row leaves most of
    the GPU idle. A width with no split geometry for the requested value takes
    the nearest one it has, and a padded width runs unsplit.

    Each width runs its own ladder from the policy (``fused_oneshot_ladder``
    at that hidden), or the policy's default ladder when it has none for the
    width. ``atoms``, ``grid_cap``, ``fanout`` and ``split`` each override that
    field on every rung of every width's ladder, and ``block`` overrides the
    geometry of every rung; the knobs left unset keep the ladder's values.
    """

    def __init__(
        self,
        *,
        group,
        device,
        rank: int,
        world_size: int,
        atoms: int | None = None,
        grid_cap: int | None = None,
        inbox_memory: str = "auto",
        fanout: str | None = None,
        block: int | None = None,
        max_bytes: int | None = None,
        link: str | None = None,
        hiddens: tuple[int, ...] = (),
        split: int | None = None,
    ):
        check_world_size(world_size)
        link = resolve_link(link)
        self.link = link
        if atoms is not None and atoms not in SUPPORTED_ATOMS:
            raise ValueError(f"atoms must be one of {SUPPORTED_ATOMS}, got {atoms!r}")
        if split is not None and int(split) not in SUPPORTED_SPLITS:
            raise ValueError(f"split must be one of {SUPPORTED_SPLITS}, got {split!r}")
        if block is not None and atoms is not None:
            raise ValueError(
                f"pin atoms or block, not both: block={block} and atoms={atoms} "
                "are the same knob here (block * atoms * 8 == hidden)"
            )
        if grid_cap is not None and int(grid_cap) < 1:
            raise ValueError(f"grid_cap must be positive, got {grid_cap}")
        _validate_ipc_process_group(
            group,
            rank=int(rank),
            world_size=world_size,
            owner="OneShotAllReduceRMSNorm",
        )
        arch = require_arch(owner="OneShotAllReduceRMSNorm")

        inbox_flags, resolved_inbox = _resolve_inbox_flags(
            inbox_memory, world_size, link
        )
        self._device_index = _cuda_index(device, owner="OneShotAllReduceRMSNorm")
        self.group = group
        self.device = torch.device("cuda", self._device_index)
        self._has_launched = False
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.inbox_memory = resolved_inbox
        self._inbox_flags = inbox_flags
        self.max_bytes = (
            resolve_fused_oneshot_max_bytes(world_size=world_size, link=link, arch=arch)
            if max_bytes is None
            else int(max_bytes)
        )
        self.block = None if block is None else int(block)

        # A pinned knob overrides its field on every ladder rung. A pinned
        # ``block`` is not a rung field: ``_geom_for`` reads it.
        pins = {
            name: value
            for name, value in (
                ("atoms", None if atoms is None else int(atoms)),
                ("grid_cap", None if grid_cap is None else int(grid_cap)),
                ("fanout", fanout),
                ("split", None if split is None else int(split)),
            )
            if value is not None
        }
        self._pins = pins
        self._arch = arch
        # hidden -> that width's ladder, the policy's for the width or its
        # default, with the pins applied.
        self._ladders: dict[int, tuple] = {}

        # (hidden, atoms, grid_cap, fanout, h_pad, split) -> (engine, spec)
        self._by_cfg: dict[tuple, tuple] = {}
        try:
            for h in sorted({int(x) for x in hiddens}):
                self._build_hidden(h)
        except Exception:
            self.close()
            raise

    # -- engine construction -------------------------------------------------

    @staticmethod
    def _nearest(legal, want: int) -> int:
        """*want* if legal, else the nearest legal value, ties to the smaller."""
        want = int(want)
        return want if want in legal else min(legal, key=lambda v: (abs(v - want), v))

    def _split_geom_for(self, hidden: int, rung_atoms: int, rung_split: int):
        """``(atoms, hidden, split)`` for a split rung, or None to run unsplit.

        Only native widths split. The requested split resolves to the nearest
        one this width has (7168 has {2, 7, 14}); then ``atoms`` resolves as in
        ``_geom_for``, or from a pinned ``block``, which is then the slice's.
        """
        opts = [o for o in fused_split_options(hidden) if o[0] > 1]
        if self.block:
            opts = [o for o in opts if o[1] == self.block]
        if not opts:
            return None
        split = self._nearest(sorted({k for k, _b, _a in opts}), rung_split)
        atoms = self._nearest([a for k, _b, a in opts if k == split], rung_atoms)
        return atoms, hidden, split

    def _geom_for(
        self, hidden: int, rung_atoms: int, rung_split: int = 1
    ) -> tuple[int, int, int]:
        """``(atoms, h_pad, split)`` for a rung at hidden dim.

        A split rung takes a split geometry if the width has one
        (``_split_geom_for``) and otherwise runs unsplit. Unsplit, a width with
        a native geometry resolves exactly and ``h_pad == hidden``. A width that
        native geometry cannot cover falls to the padded set, where the pad
        leads. ``fused_padded_block_options`` is ordered by ascending ``h_pad``,
        so this takes the least wire volume available and only then uses the
        rung's ``atoms`` to break the tie.
        """
        hidden = int(hidden)
        if int(rung_split) > 1:
            geom = self._split_geom_for(hidden, rung_atoms, rung_split)
            if geom is not None:
                return geom
        native = fused_block_options(hidden)
        if self.block:
            native_atoms = next((a for b, a in native if b == self.block), None)
            if native_atoms is not None:
                return native_atoms, hidden, 1
        elif native:
            return self._nearest([a for _b, a in native], rung_atoms), hidden, 1

        opts = fused_padded_block_options(hidden)
        if self.block:
            pinned = [o for o in opts if o[0] == self.block]
            if pinned:
                return pinned[0][1], pinned[0][2], 1
        elif opts:
            least_pad = opts[0][2]
            tied = [o for o in opts if o[2] == least_pad]
            _b, atoms, h_pad = min(
                tied, key=lambda o: (abs(o[1] - int(rung_atoms)), o[1])
            )
            return atoms, h_pad, 1

        # No geometry at all -- or a pinned block padding cannot reach. Returning
        # the rung's own atoms lets the build raise with the message that names
        # the constraint that failed.
        return int(rung_atoms), hidden, 1

    def _cfg_key(self, hidden: int, rung: tuple) -> tuple:
        """A ladder rung's engine key at *hidden*.

        A split build's grid cap is rounded down to a multiple of its split --
        a row group is ``split`` consecutive workgroups and all of them must be
        launched -- so a cap of 64 runs 63 workgroups at split=7.
        """
        atoms, h_pad, split = self._geom_for(hidden, rung.atoms, rung.split)
        cap = max(split, int(rung.grid_cap) // split * split)
        return (int(hidden), atoms, cap, rung.fanout, h_pad, split)

    def _ladder(self, hidden: int) -> tuple:
        """The rungs at hidden: the policy's ladder for this width, or its
        default ladder (``fused_oneshot_ladder``), with the pins applied."""
        hidden = int(hidden)
        ladder = self._ladders.get(hidden)
        if ladder is None:
            ladder = tuple(
                dataclasses.replace(rung, **self._pins)
                for rung in fused_oneshot_ladder(
                    world_size=self.world_size,
                    link=self.link,
                    arch=self._arch,
                    hidden=hidden,
                )
            )
            self._ladders[hidden] = ladder
        return ladder

    def _build_hidden(self, hidden: int) -> None:
        """Build every rung for hidden dim. Collective: all ranks must call it in
        the same order,."""
        for rung in self._ladder(hidden):
            key = self._cfg_key(hidden, rung)
            if key in self._by_cfg:
                continue
            spec = make_one_shot_allreduce_kernel(
                world_size=self.world_size,
                atoms=key[1],
                grid=key[2],
                inbox_memory=self.inbox_memory,
                fanout=key[3],
                rank=self.rank,
                fusion="rmsnorm",
                hidden=key[0],
                h_pad=key[4],
                split=key[5],
            )
            with torch.cuda.device(self._device_index):
                self._by_cfg[key] = (
                    _StEngine(
                        spec=spec,
                        group=self.group,
                        rank=self.rank,
                        world_size=self.world_size,
                        inbox_flags=self._inbox_flags,
                    ),
                    spec,
                )

    def _pick_cfg(self, hidden: int, live_bytes: int) -> tuple:
        key = self._cfg_key(hidden, select_rung(self._ladder(hidden), live_bytes))
        if key not in self._by_cfg:
            self._build_hidden(int(hidden))
        return key

    @property
    def inbox_bytes(self) -> int:
        """IPC inbox bytes this object holds on this rank, across every engine."""
        return sum(eng.buf_bytes for eng, _ in self._by_cfg.values())

    @property
    def hiddens(self) -> tuple[int, ...]:
        return tuple(sorted({k[0] for k in self._by_cfg}))

    def supports_hidden(self, hidden: int) -> bool:
        """Whether any build exists for hidden dim, padded and split ones included.

        A split rung counts when ``_split_geom_for`` resolves it. With a pinned
        ``block`` that is the slice's width, which the unsplit sets below need
        not list: 4096 at ``block=64`` exists only as a split build.
        """
        hidden = int(hidden)
        if any(
            r.split > 1 and self._split_geom_for(hidden, r.atoms, r.split) is not None
            for r in self._ladder(hidden)
        ):
            return True
        if self.block is not None:
            if any(b == self.block for b, _ in fused_block_options(hidden)):
                return True
            return any(
                b == self.block for b, _a, _h in fused_padded_block_options(hidden)
            )
        if fused_block_options(hidden):
            return True
        return bool(fused_padded_block_options(hidden))

    def pads_hidden(self, hidden: int) -> int:
        """``h_pad`` this width would run at, or hidden dim when it needs no padding."""
        first = self._ladder(hidden)[0]
        return self._geom_for(int(hidden), first.atoms, first.split)[1]

    # -- launch --------------------------------------------------------------

    def _check(self, inp, residual_in, weight, out, residual_out):
        tensors = {
            "inp": inp,
            "residual_in": residual_in,
            "weight": weight,
            "out": out,
            "residual_out": residual_out,
        }
        for name, t in tensors.items():
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"OneShotAllReduceRMSNorm requires a Tensor for {name}")
            if t.dtype != torch.bfloat16:
                raise ValueError(
                    f"OneShotAllReduceRMSNorm is bf16-only, {name} is {t.dtype}"
                )
            if not t.is_cuda or t.device.index != self._device_index:
                raise ValueError(
                    f"{name} must be on cuda:{self._device_index}, got {t.device}"
                )
            if not t.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
            if int(t.data_ptr()) % 16 != 0:
                raise ValueError(f"{name} must be 16-byte aligned")

        hidden = int(inp.shape[-1])
        if weight.numel() != hidden:
            raise ValueError(
                f"weight width {weight.numel()} does not match input width {hidden}"
            )
        if tuple(residual_in.shape) != tuple(inp.shape):
            raise ValueError(
                f"residual_in shape {tuple(residual_in.shape)} != inp {tuple(inp.shape)}"
            )
        if tuple(out.shape) != tuple(inp.shape) or tuple(residual_out.shape) != tuple(
            inp.shape
        ):
            raise ValueError("out/residual_out must have the input's shape")
        if not self.supports_hidden(hidden):
            pin = (
                f"block={self.block}"
                if self.block is not None
                else f"atoms={[r.atoms for r in self._ladder(hidden)]}"
            )
            raise ValueError(
                f"no fused build for hidden={hidden} at {pin}: one block "
                "covers one row, so hidden/(8*atoms) must be a multiple of 64 and "
                f"at most 1024 -- the legal (block, atoms) pairs for this width "
                f"are {fused_block_options(hidden)}"
            )
        live_bytes = int(inp.numel()) * 2
        if live_bytes > 0xFFFFFFFF:
            raise ValueError("payload must not exceed the 4 GiB buffer window")
        if live_bytes > PAD_MASK_MAX_BYTES and self.pads_hidden(hidden) != hidden:
            raise ValueError(
                f"a padded fused build (hidden={hidden} runs at "
                f"h_pad={self.pads_hidden(hidden)}) takes payloads of at most "
                f"{PAD_MASK_MAX_BYTES} B; got {live_bytes} B"
            )
        check_fused_aliasing(inp, residual_in, out, residual_out, live_bytes)
        return hidden, live_bytes

    def _launch_args(self, eng, spec, ptrs, eps, stream, *, hidden, live_bytes):
        """Kernel arguments. *ptrs* is ``(inp, out, residual_in, residual_out,
        weight)`` as raw addresses."""
        inp_ptr, out_ptr, res_in_ptr, res_out_ptr, w_ptr = ptrs
        num_tiles, grid_x = self._tiles_and_grid(spec, hidden, live_bytes)
        if stream is None:
            stream = Stream(torch.cuda.current_stream(self._device_index))
        elif not isinstance(stream, Stream):
            stream = Stream(stream)
        return (
            Int32(self.rank),
            Int64(live_bytes),
            Int32(num_tiles),
            Int64(inp_ptr),
            Int64(out_ptr),
            Int64(int(eng._gpu_peer_ptrs)),
            Int64(int(eng._colors)),
            Int32(grid_x),
            Int64(res_in_ptr),
            Int64(res_out_ptr),
            Int64(w_ptr),
            Float32(float(eps)),
            Int64(int(eng._xchg)),
            *_NO_ATTN_RES_ARGS,
            stream,
        )

    @staticmethod
    def _tiles_and_grid(spec, hidden: int, live_bytes: int) -> tuple[int, int]:
        """``(num_tiles, grid_x)`` for a launch.

        A split build views the ``(M, hidden)`` operands as ``(M*split,
        hidden/split)``, so it has ``split`` tiles per token, and launches whole
        row groups: a group whose workgroups were not all launched would wait
        forever on the missing ones' contributions.
        """
        split = int(spec.get("split", 1))
        tokens = live_bytes // (int(hidden) * 2)
        grid_x = max(1, min(tokens, spec["grid"] // split)) * split
        assert grid_x % split == 0 and grid_x <= spec["grid"], (grid_x, split)
        return tokens * split, grid_x

    def _launch(
        self,
        inp,
        residual_in,
        weight,
        eps,
        out,
        residual_out,
        stream,
        *,
        hidden,
        live_bytes,
    ):
        eng, spec = self._by_cfg[self._pick_cfg(hidden, live_bytes)]
        ptrs = tuple(
            int(t.data_ptr()) for t in (inp, out, residual_in, residual_out, weight)
        )
        args = self._launch_args(
            eng, spec, ptrs, eps, stream, hidden=hidden, live_bytes=live_bytes
        )
        self._has_launched = True
        with torch.cuda.device(self._device_index):
            _run_compiled(eng.launch, *args)

    def allreduce_rmsnorm(
        self,
        inp,
        residual_in,
        weight,
        eps,
        *,
        out=None,
        residual_out=None,
        stream=None,
    ):
        """``(out, residual_out)``; both are allocated when not supplied."""
        if out is None:
            out = torch.empty_like(inp)
        if residual_out is None:
            residual_out = torch.empty_like(residual_in)
        hidden, live_bytes = self._check(inp, residual_in, weight, out, residual_out)
        if not self.is_beneficial(live_bytes):
            raise ValueError(
                f"OneShotAllReduceRMSNorm got a {live_bytes} B payload, above the "
                f"{self.max_bytes} B ceiling: this schedule pushes the whole payload "
                "to every peer, so its wire volume is (N-1)x the message. Pass "
                "max_bytes to override."
            )
        self._launch(
            inp,
            residual_in,
            weight,
            eps,
            out,
            residual_out,
            stream,
            hidden=hidden,
            live_bytes=live_bytes,
        )
        return out, residual_out

    def cfgs_for(self, hidden: int, lo: int, hi: int) -> list[tuple]:
        """``_by_cfg`` keys at hidden dim a payload of ``lo..hi`` bytes
        (inclusive) can select, in build order. Builds the width if needed."""
        hidden = int(hidden)
        floors = [rung.min_bytes for rung in self._ladder(hidden)]
        picked = {self._pick_cfg(hidden, n) for n in payload_probes(floors, lo, hi)}
        return [key for key in self._by_cfg if key in picked]

    def preload(self, hidden: int, *, payload_range=None) -> None:
        """Build hidden dim and JIT-compile its rung binaries and load them onto
        the device, without launching any of them.

        ``payload_range=(lo, hi)`` takes only the rungs ``allreduce_rmsnorm``
        would run for a payload of ``lo..hi`` bytes (inclusive); ``None`` takes
        every one. The build still covers the whole ladder: a width's engines
        are built together.

        A collective only when hidden dim is not built yet, since building
        exchanges IPC handles, so every rank must call it with the same widths
        in the same order. The compile itself is local. ``FlyDSLAllReduceRMSNorm``
        calls it via ``preload_fly_engines`` to keep the JIT and the module load
        off the first real call, which may be inside a HIP graph capture.
        """
        hidden = int(hidden)
        self._build_hidden(hidden)
        if payload_range is None:
            keys = [key for key in self._by_cfg if key[0] == hidden]
        else:
            keys = self.cfgs_for(hidden, *payload_range)
        with torch.cuda.device(self._device_index):
            for key in keys:
                eng, spec = self._by_cfg[key]
                args = self._launch_args(
                    eng, spec, (0,) * 5, 0.0, None, hidden=hidden, live_bytes=0
                )
                preload_launcher(eng.launch, *args)

    def variant(self, hidden: int, nbytes: int) -> str:
        """``<jit symbol>/g<grid_cap>/x<grid_x>`` for this (hidden, payload)."""
        key = self._pick_cfg(int(hidden), int(nbytes))
        eng, spec = self._by_cfg[key]
        # The symbol names a split build (``_k<split>``); grid_x counts its
        # workgroups, ``split`` per token.
        _tiles, grid_x = self._tiles_and_grid(spec, int(hidden), int(nbytes))
        return f"{kernel_symbol(eng.launch)}/g{key[2]}/x{grid_x}"

    def is_beneficial(self, nbytes: int) -> bool:
        return int(nbytes) <= self.max_bytes

    def close(self):
        engines = getattr(self, "_by_cfg", None)
        if not engines:
            return
        with torch.cuda.device(self._device_index):
            if getattr(self, "_has_launched", False):
                torch.cuda.synchronize(self._device_index)
                self._has_launched = False
            for eng, _ in engines.values():
                eng.close()
            engines.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            # Destructors must not raise, especially during interpreter shutdown.
            return


class OneShotAllReduceAttnRes(OneShotAllReduceRMSNorm):
    """One-shot all-reduce fused with Kimi-K3's attention-residual aggregation.

    Per token row ``t`` (``ar`` is the all-reduce, ``v_j`` the bank rows
    ``bank[t, 0..nvb-1]`` and the new prefix):

        prefix_out = bf16(prefix_in + bf16(ar))
        score_j    = dot(v_j, cw) / sqrt(mean(v_j^2) + score_eps)
        w          = softmax(score)                 (global max first)
        mix        = sum_j w_j * v_j
        out        = bf16(mix * rsqrt(mean(mix^2) + out_eps) * ow)   (ow given)
                   = bf16(mix)                                        (ow None)
        bank[t, nvb] = prefix_out                   (write_bank)

    which is SGLang's ``attn_res_hip`` with ``addend`` = the all-reduce. It
    takes the RMSNorm class's ladder and geometry with every rung unsplit (one
    workgroup per row). Each ``(nvb, write_bank, out_norm)`` is its own binary
    and engine, built lazily -- a collective, which TP ranks enter in lockstep
    with the same shapes.
    """

    def __init__(self, **kwargs):
        # SILOTIGER-1277: a split>1 build is legal only for the acc_cache path
        # (the epilogue raises otherwise). Pin it here and every allreduce_attn_res
        # call with bank_acc builds the split kernel; without a split pin the
        # engine is unsplit as before.
        split = kwargs.get("split")
        if split is not None and int(split) not in SUPPORTED_SPLITS:
            raise ValueError(f"split must be one of {SUPPORTED_SPLITS}, got {split!r}")
        # SILOTIGER-1277 iter004: when split is NOT explicitly pinned, the
        # acc_cache path auto-selects split per (tokens, nvb) from the measured
        # decode ladder (``_attn_ladder_split``) under AITER_FLY_AR_FUSED_SPLIT=1;
        # an explicit pin (used by the correctness tests) still forces that split
        # at every T.
        self._split_pinned = split is not None
        kwargs.setdefault("split", 1)
        hiddens = kwargs.pop("hiddens", ())
        if hiddens:
            raise ValueError(
                "OneShotAllReduceAttnRes builds per (hidden, nvb) on first use; "
                "pass no hiddens"
            )
        if kwargs.get("atoms") is None and kwargs.get("block") is None:
            # Measured TP2 on MI355X at hidden 7168 (one token row per
            # workgroup): atoms=2 (448 threads) is 1-12% faster than the two
            # separate kernels at every T in 1..64 and nvb in {4, 8}; atoms=1
            # (896 threads) loses at T <= 8 with nvb=8 -- twice the waves to
            # reduce the 2*(nvb+1) row sums over.
            kwargs["atoms"] = 2
        super().__init__(**kwargs)

    @staticmethod
    def _attn_ladder_split(tokens: int, nvb: int) -> int:
        """Decode split for the acc_cache attn_res epilogue at (tokens, nvb),
        TP8/gfx950/H=7168 (SILOTIGER-1277 iter003/iter004 measurements).

        The split epilogue parallelises the ~8us per-token fused epilogue over
        ``split`` workgroups; it wins while the GPU is occupancy-starved and
        loses once the wire volume (``split*tokens`` workgroups) saturates:
          nvb>=8: split=14 wins at T<=4; for T in [5,18] the single-barrier merge
                  epilogue (iter010 triple-join, shipped on for every split>1)
                  makes the narrower split=7 FASTER -- its cross-WG join has half
                  the writers (split*n_waves), so the atomic-join cost that grows
                  with token count at split=14 (epilogue 3.1->3.46us T4->T8) stays
                  flat at split=7 (~3.1us), winning -2..-3.6% of the fused op at
                  T6-T12 (-0.38us/-3.6% @T8), tie at T16 (SILOTIGER-1277 iter050,
                  runs/iter050_crossover_tp8.json). Hand to cdr_2stage past T18.
          nvb==4: cdr+agg is cheaper (agg is cheap at nvb=4) so the band is
                  narrower -- split wins to T<=8, cdr at T>=16. The acc_cache
                  merge epilogue is nvb-INDEPENDENT (bank depth is in the
                  producer scalar), so the same split=14->split=7 crossover at
                  T5 holds: split=14 for T<=4, split=7 for T5-8 (-2.6..-3.5%,
                  iter050 runs/iter050_nvb4_crossover_tp8.json), else unsplit.
        Legal splits>1 at H=7168 are 2, 7, 14 (a split=16 slice would be a
        sub-wave tile). nvb<4 is uncharacterised -> unsplit.
        """
        if nvb >= 8:
            if tokens <= 4:
                return 14
            return 7 if tokens <= 18 else 1
        if nvb >= 4:
            if tokens <= 4:
                return 14
            return 7 if tokens <= 8 else 1
        return 1

    @staticmethod
    def _attn_ladder_merge(tokens: int, split: int, out_norm: bool) -> bool:
        """Single-barrier out-norm (SILOTIGER-1277 iter010) on/off at (tokens,
        split), TP8/gfx950/H=7168. The out-norm's sum(mix^2) is formed from the
        prefix dot/sq plus one extra joined word cross=sum(pv*acc), with the
        sum(acc^2) term supplied as a producer full-row scalar -- so it joins the
        SAME three words as the two-barrier path but in ONE barrier, saving an
        xGMI round-trip at NO extra atomic cost. Measured net WIN at every decode
        T (-0.5..-0.64us, -3.6% @T16 .. -7.4% @T1, split=14 nvb=8), so enable it
        across the whole split band (``tokens`` is already capped by the split
        ladder; this just requires split>1 and the output norm)."""
        return bool(out_norm) and split > 1

    def _attn_key(self, hidden: int, live_bytes: int, nvb: int, write_bank, out_norm,
                  acc_cache=False):
        rung = select_rung(self._ladder(hidden), live_bytes)
        # SILOTIGER-1277 iter004: auto-select the decode split by live bytes when
        # the caller did not pin one. Only the acc_cache build may split (the
        # epilogue raises otherwise) and only under the AITER_FLY_AR_FUSED_SPLIT
        # opt-in; everything else stays unsplit, as before.
        outnorm_merge = False
        if (
            not self._split_pinned
            and acc_cache
            and fused_split_enabled()
        ):
            tokens = int(live_bytes) // (int(hidden) * 2)
            split = self._attn_ladder_split(tokens, int(nvb))
            if split == 7:
                # iter050: split=7's fast decode geometry is atoms=1 (block=128).
                # The rung's default atoms=2 would make _split_geom_for resolve
                # split=7 to atoms=2 (block=64), which is SLOWER than split=14 --
                # so the mid-band hand-off needs atoms=1 pinned to realize the win.
                rung = dataclasses.replace(rung, split=split, atoms=1)
            else:
                rung = dataclasses.replace(rung, split=split)
            # iter010: fold join point 2 into join point 1 at low T (same opt-in).
            outnorm_merge = self._attn_ladder_merge(tokens, split, bool(out_norm))
        base = self._cfg_key(hidden, rung)
        return base + (int(nvb), bool(write_bank), bool(out_norm), bool(acc_cache),
                       bool(outnorm_merge))

    def _build_attn(self, key) -> None:
        (hidden, atoms, grid, fanout, h_pad, split, nvb, write_bank, out_norm,
         acc_cache, outnorm_merge) = key
        spec = make_one_shot_allreduce_kernel(
            world_size=self.world_size,
            atoms=atoms,
            grid=grid,
            inbox_memory=self.inbox_memory,
            fanout=fanout,
            rank=self.rank,
            fusion="attn_res",
            hidden=hidden,
            h_pad=h_pad,
            split=split,
            nvb=nvb,
            write_bank=write_bank,
            out_norm=out_norm,
            acc_cache=acc_cache,
            outnorm_merge=outnorm_merge,
        )
        with torch.cuda.device(self._device_index):
            self._by_cfg[key] = (
                _StEngine(
                    spec=spec,
                    group=self.group,
                    rank=self.rank,
                    world_size=self.world_size,
                    inbox_flags=self._inbox_flags,
                ),
                spec,
            )

    def _check_attn(
        self, inp, prefix_in, bank, nvb, cw, ow, out, prefix_out, write_bank
    ):
        hidden = int(inp.shape[-1])
        bf16 = {"inp": inp, "prefix_in": prefix_in, "bank": bank, "out": out}
        bf16["prefix_out"] = prefix_out
        if ow is not None:
            bf16["ow"] = ow
        for name, t in list(bf16.items()) + [("cw", cw)]:
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"OneShotAllReduceAttnRes requires a Tensor for {name}")
            want = torch.float32 if name == "cw" else torch.bfloat16
            if t.dtype != want:
                raise ValueError(f"{name} must be {want}, got {t.dtype}")
            if not t.is_cuda or t.device.index != self._device_index:
                raise ValueError(f"{name} must be on cuda:{self._device_index}")
            if not t.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
            if int(t.data_ptr()) % 16 != 0:
                raise ValueError(f"{name} must be 16-byte aligned")
        tokens = int(inp.numel()) // hidden
        if inp.dim() != 2 or bank.dim() != 3:
            raise ValueError("inp must be (T, hidden) and bank (T, NB, hidden)")
        if tuple(bank.shape[::2]) != (tokens, hidden):
            raise ValueError(
                f"bank {tuple(bank.shape)} does not match inp {tuple(inp.shape)}"
            )
        n_rows = int(bank.shape[1])
        if not 1 <= int(nvb) <= n_rows - (1 if write_bank else 0):
            raise ValueError(
                f"nvb={nvb} does not fit a bank of {n_rows} rows"
                + (" with the snapshot row" if write_bank else "")
            )
        if cw.numel() != hidden or (ow is not None and ow.numel() != hidden):
            raise ValueError("cw and ow must have hidden elements")
        for name, t in (
            ("prefix_in", prefix_in),
            ("out", out),
            ("prefix_out", prefix_out),
        ):
            if tuple(t.shape) != tuple(inp.shape):
                raise ValueError(f"{name} must have the input's shape")
        live_bytes = int(inp.numel()) * 2
        if not self.is_beneficial(live_bytes):
            raise ValueError(
                f"OneShotAllReduceAttnRes got a {live_bytes} B payload, above the "
                f"{self.max_bytes} B ceiling; pass max_bytes to override"
            )
        check_fused_aliasing(inp, prefix_in, out, prefix_out, live_bytes)
        return hidden, live_bytes, n_rows

    def precompute_bank_scores(self, bank, cw, nvb, score_eps):
        """Public: (T, nvb) fp32 bank scores to pass to
        :meth:`allreduce_attn_res` as ``bank_scores`` (SILOTIGER-1277). Compute
        once per (bank, cw); reuse across all-reduces that share the bank."""
        return self._precompute_bank_scores(
            bank, cw, int(nvb), int(bank.shape[-1]), float(score_eps)
        )

    def precompute_bank_acc(self, bank, cw, nvb, score_eps):
        """Public: Kevin Mi's acc-cache (sglang#44988 bank_partial_hip.cuh)
        ported to aiter (SILOTIGER-1277). Returns ``(acc, ms)``:
          ms[t]  = (m, sum, accsq): m = max_r s_r over the bank rows, sum = sum_r
                   exp(s_r - m), accsq = sum_h acc[t,h]^2; (T, 3) fp32. accsq
                   (iter010) feeds the single-barrier out-norm so sum(acc^2) is
                   never joined across workgroups; m/sum are columns 0/1 as before.
          acc[t] = sum_r exp(s_r - m) * bank[t, r]; (T, hidden) fp32.
        where s_r = dot(bank[t,r], cw) / sqrt(mean(bank[t,r]^2) + score_eps).
        The fused read then folds ONLY the new row against these, never reading
        the bank VALUES. Normalised to the BANK's max m (NOT the global max); the
        read renormalises by exp(m - m') when the stream raises the max -- the
        exact order of Kevin's kernel, so the softmax is the same math.
        Compute once per (bank, cw); SGLang does it one layer ahead on the route
        sort's spare workgroups (overlapped, ~free)."""
        hidden = int(bank.shape[-1])
        v = bank[:, :nvb, :].to(torch.float32)              # (T, nvb, H)
        cwf = cw.to(torch.float32).view(1, 1, hidden)
        dot = (v * cwf).sum(dim=-1)                          # (T, nvb)
        meansq = (v * v).sum(dim=-1) * (1.0 / hidden)
        scores = dot * torch.rsqrt(meansq + float(score_eps))  # (T, nvb)
        m = scores.max(dim=-1, keepdim=True).values          # (T, 1) bank max
        w = torch.exp(scores - m)                            # (T, nvb)
        s = w.sum(dim=-1, keepdim=True)                      # (T, 1)
        acc = torch.einsum("tr,trh->th", w, v)               # (T, H) fp32
        # SILOTIGER-1277 iter010: accsq = sum_h acc^2 per token, a full-row
        # scalar the producer is already positioned to form (acc is in hand and
        # AR-independent). The single-barrier out-norm epilogue reads it directly
        # so it never has to join sum(acc^2) across workgroups -- the out-norm's
        # sum of squares then needs only ONE extra joined word (cross=sum(pv*acc))
        # on top of the prefix dot/sq, i.e. the SAME word count as the two-barrier
        # path but one fewer barrier -> a net win at every decode T (not just the
        # low-T tiles the 4-word variant was limited to).
        accsq = (acc * acc).sum(dim=-1, keepdim=True)        # (T, 1) fp32
        ms = torch.cat([m, s, accsq], dim=-1).contiguous()   # (T, 3)
        return acc.contiguous(), ms

    def _precompute_bank_scores(self, bank, cw, nvb, hidden, score_eps):
        """fp32 (T, nvb) softmax scores for the bank rows (SILOTIGER-1277).

        The bank rows are pre-existing context -- not products of this
        all-reduce -- so their score ``dot(v_j, cw) / sqrt(mean(v_j^2) +
        score_eps)`` can be computed once here instead of block-reduced in the
        fused epilogue for every launch. Returned contiguous (T, nvb) fp32.

        Math matches the kernel: bank values are cast bf16->fp32, cw is fp32,
        ``mean = sum(v^2) / hidden``. The reduction order differs from the
        kernel's reassoc tree, so this is validated bit-for-bit by the op_test
        gate (<= 1 ulp vs Triton), not assumed.
        """
        tokens = int(bank.shape[0])
        v = bank[:, :nvb, :].to(torch.float32)  # (T, nvb, hidden)
        cwf = cw.to(torch.float32).view(1, 1, hidden)
        dot = (v * cwf).sum(dim=-1)  # (T, nvb)
        meansq = (v * v).sum(dim=-1) * (1.0 / hidden)
        scores = dot * torch.rsqrt(meansq + score_eps)
        return scores.contiguous()

    def allreduce_attn_res(
        self,
        inp,
        prefix_in,
        bank,
        nvb: int,
        cw,
        ow,
        score_eps: float,
        out_eps: float,
        *,
        out=None,
        prefix_out=None,
        write_bank: bool = False,
        bank_scores=None,
        bank_acc=None,
        stream=None,
    ):
        """``(out, prefix_out)``; both are allocated when not supplied.

        ``bank_scores`` (SILOTIGER-1277): optional caller-precomputed (T, nvb)
        fp32 bank softmax scores. When the bank is reused across calls (it is
        context, not a product of this all-reduce) the caller computes these
        once with :meth:`precompute_bank_scores` and passes them in, keeping the
        reduction out of the fused op's hot path. When ``None`` they are
        computed here (convenient, but on the critical path).

        ``bank_acc`` (SILOTIGER-1277, Kevin Mi acc-cache): optional ``(acc, ms)``
        from :meth:`precompute_bank_acc`. When given, the fused read NEVER touches
        bank values -- it folds only the new row against the cached weighted bank
        sum. Selects the ``acc_cache`` build (its own binary). Takes precedence
        over ``bank_scores``."""
        if out is None:
            out = torch.empty_like(inp)
        if prefix_out is None:
            prefix_out = torch.empty_like(prefix_in)
        hidden, live_bytes, n_rows = self._check_attn(
            inp, prefix_in, bank, nvb, cw, ow, out, prefix_out, write_bank
        )
        acc_cache = bank_acc is not None
        key = self._attn_key(hidden, live_bytes, nvb, write_bank, ow is not None,
                             acc_cache=acc_cache)
        if key not in self._by_cfg:
            self._build_attn(key)
        eng, spec = self._by_cfg[key]
        ptrs = tuple(
            int(t.data_ptr()) if t is not None else 0
            for t in (inp, out, prefix_in, prefix_out, ow)
        )
        args = self._launch_args(
            eng, spec, ptrs, out_eps, stream, hidden=hidden, live_bytes=live_bytes
        )
        # Bank-norm cache (SILOTIGER-1277): the nvb bank rows are pre-existing
        # context, not products of this all-reduce, so their softmax score
        # (dot(v_j, cw) / sqrt(mean(v_j^2) + score_eps)) does not depend on the
        # reduced row. Precompute the nvb bank scores here once, in fp32, and
        # pass them in; the fused epilogue then only block-reduces the new
        # prefix row (2 words) instead of every bank row (2*(nvb+1) words),
        # dropping the cross-wave join from 2*(nvb+1)+1 to 3. The math is
        # unchanged -- the softmax still takes the global max before exp.
        #
        # acc-cache (Kevin Mi port): when bank_acc=(acc, ms) is given, the read
        # never touches bank values; score_ptr carries acc (T,H fp32), the extra
        # ms_ptr carries (m, sum, accsq) (T,3 fp32; iter010). Otherwise score_ptr carries the
        # (T, nvb) scores and ms_ptr is 0.
        if acc_cache:
            acc, ms = bank_acc
            score_data, ms_data = int(acc.data_ptr()), int(ms.data_ptr())
        else:
            if bank_scores is None:
                bank_scores = self._precompute_bank_scores(
                    bank, cw, nvb, hidden, float(score_eps)
                )
            score_data, ms_data = int(bank_scores.data_ptr()), 0
        # The RMSNorm launch passes zeros for the AttnRes arguments; swap in
        # this call's (they sit just before the stream).
        tail = len(_NO_ATTN_RES_ARGS) + 1
        args = args[:-tail] + (
            Int64(int(bank.data_ptr())),
            Int64(int(cw.data_ptr())),
            Int32(n_rows),
            Float32(float(score_eps)),
            Int64(score_data),
            Int64(ms_data),
            args[-1],
        )
        self._has_launched = True
        with torch.cuda.device(self._device_index):
            _run_compiled(eng.launch, *args)
        return out, prefix_out
