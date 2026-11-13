# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host side of the gfx1250 stage-1 kernel (on by default, ``AITER_MEGA_S1F=0``
turns it off).

The compact dispatch owns one ``_Stage1Fusion``. When the switch is on and the
shape fits, the compact plan emits the run table the kernel gathers from, and
GEMM1 is replaced by that kernel. Anything else keeps dispatch and GEMM1 as
separate launches.
"""

from __future__ import annotations

import logging
import math
import os

import flydsl.expr as fx
import torch

from aiter import ActivationType

from ..tensor_shim import _run_compiled, ptr_arg
from .arena import _align_up

_log = logging.getLogger("aiter")

#: Workgroups the plan kernel adds for the dispatch quant, which otherwise
#: costs its own launch. The plan holds 1-7 of 256 CUs for its whole
#: cross-rank wait, so the quant rides along on idle ones.
#:
#: The kernel ends when its last block does, so the plan's measured time
#: becomes max(plan, quant) and the count has to be large enough to keep the
#: quant under it. A step at tpr 1536 streams ~28 MB, which is HBM traffic,
#: not the L2-resident kind a GEMM sees: 32 blocks pushed the plan from 22 to
#: 66-86 us, and it takes 128 to bring it back to 41-54.
#:
#: 0 keeps the standalone HIP quant, which a step also falls back to when its
#: plan was prefetched (that runs before this step's hidden states exist).
_QUANT_FUSE_BLOCKS = int(os.environ.get("AITER_MEGA_QUANT_FUSE", "128"))

# Smallest m-tile the flag region is sized for (one flag per tile/source).
_S1F_MIN_TILE_M = 16
#: The GEMM1 tile n / k every stage-1 variant embeds (with 1x4 waves).
_S1F_TILE_N, _S1F_TILE_K = 256, 256
#: Stage-1 variants by GEMM1 tile_m: (pipeline buffers, CTAs per CU).
#:
#: * 16 rows (decode): its own flag tile. Even 3 buffers take 112,384 B, so
#:   three CTAs do not fit a CU; 4 buffers edge out 3 at tpr 8 / 64.
#: * 32 rows (EP2 tpr 256 / 512): its own flag tile too, with the 16-row
#:   layout; the tuned GEMM1 picks tile 32 there.
#: * 64 rows: ~230 VGPRs, 133,632 B of arena, two CTAs per CU.
#: * 128 rows: twice the accumulators (398 VGPRs, no spills) in 162,816 B.
#:   Two still fit: 2 x 163,840 B is exactly the 320 KB of LDS and two waves
#:   of 398 VGPRs a SIMD.
#: * 256 rows: ~800 VGPRs leave one wave per SIMD, so the CTA has the CU to
#:   itself; 3 buffers (221,184 B) beat 4 at tpr 1536.
#:
#: Every worker CTA has to be resident at once -- the producers hand tiles to
#: consumers inside one launch -- so a variant whose CTAs per CU exceeds what
#: VGPRs and LDS allow would hang, not just run slow.
_S1F_VARIANTS = {16: (4, 2), 32: (4, 2), 64: (3, 2), 128: (3, 2), 256: (3, 1)}
#: GEMM1 (m_warp, n_warp) per tile_m (default 1x4). 2x2 at 256 rows takes the
#: GEMM body's pinned-VGPR / wmma_2x4 prefill schedule (waves_per_eu=1, which
#: the one-CTA-per-CU 256 variant already implies).
_S1F_WARPS = {256: (2, 2)}
#: The grouped-MoE prefill GEMM1 options, by tile_m. Packed A (the storers
#: write disp_out as M16 x K32 slabs) is what lets the others pay off: on
#: DSV4.1-Flash EP4 tpr 16384 the stage-1 kernel takes 606 us with all three
#: against 686 without packed A and 676 with none.
_S1F_GEMM_OPTS = {256: {"a_preshuffled": 1, "lds_soa": 1, "next_stage_prefetch": 1}}
#: The smallest recv bound whose step widens GEMM1 to the 256-row prefill tile,
#: by (world_size, topk); the 256 alignment costs GEMM2 more than it saves below
#: it. DSV4.1 Flash EP4 per layer, 128 tile vs 256 prefill tile: tpr 8192
#: 894 vs 907 us, 16384 1704 vs 1646, 32768 3264 vs 3168.
_S1F_PREFILL_MIN_RECV_BY_SHAPE: dict = {(4, 6): 65536}
#: m-tiles per group in the GEMM1 work order.
_S1F_GROUP_M = 8
# The S1F settings below (with S1F_BLOCK_RUNS in s1f_stage1.py and the compact
# route schedule in config.py) come from DSV4.1-Flash DEP4 on MI455X, MLPerf
# Endpoints agentic c512: +2.1% gen tok/s/GPU over S1F off.
#: A step whose receive bound exceeds this runs the whole S1F=0 path (see
#: _Stage1Fusion.step_off), so the widest prefill steps keep dispatch + GEMM1.
_S1F_MAX_RECV = 4096
#: Receive caps that differ from _S1F_MAX_RECV, by (world_size, topk).
#: DSV4.1 Flash (top-6) runs stage-1 fused at every width on EP2 and EP4. At
#: 8192 tokens/rank: 886 vs 959 us/layer on EP2, 900 vs 1063 on EP4. Top-7 is
#: the same model with its shared expert fused: 1200 vs 1405 on EP2. Past
#: 4096 tokens this compiles a few more plan widths (7 instead of 4 at 8192).
_S1F_MAX_RECV_BY_SHAPE: dict = {
    (2, 6): 1 << 30,
    (4, 6): 1 << 30,
    (2, 7): 1 << 30,
    (4, 7): 1 << 30,
}


def s1f_max_recv(world_size: int, topk: int) -> int:
    """Widest receive bound a step can have and still run the stage-1 kernel.

    ``AITER_S1F_MAX_RECV`` overrides it for every shape; 0 removes the cap.
    """
    env = os.environ.get("AITER_S1F_MAX_RECV", "").strip()
    if env:
        cap = int(env)
        return cap if cap > 0 else 1 << 30
    return _S1F_MAX_RECV_BY_SHAPE.get((int(world_size), int(topk)), _S1F_MAX_RECV)


def s1f_prefill_tile(world_size: int, topk: int, recv_bound: int) -> int:
    """GEMM1 tile_m the stage-1 kernel runs for this recv bound instead of the
    CSV's, or 0. ``AITER_MEGA_S1F_PREFILL_RECV`` overrides the bound (0: off)."""
    if not read_s1f_mode():
        return 0
    env = os.environ.get("AITER_MEGA_S1F_PREFILL_RECV", "").strip()
    lo = (
        int(env)
        if env
        else _S1F_PREFILL_MIN_RECV_BY_SHAPE.get((int(world_size), int(topk)), 0)
    )
    return 256 if lo > 0 and int(recv_bound) >= lo else 0


def _s1f_warps(tile_m: int) -> tuple:
    """GEMM1 (m_warp, n_warp); ``AITER_MEGA_S1F_WARPS=MxN`` forces one grid."""
    env = os.environ.get("AITER_MEGA_S1F_WARPS", "").strip().lower()
    if env:
        m, n = (int(v) for v in env.split("x"))
        if m * n != 4:
            raise ValueError(f"AITER_MEGA_S1F_WARPS={env!r} must be a 4-wave grid")
        return m, n
    return _S1F_WARPS.get(int(tile_m), (1, 4))


def read_s1f_mode() -> bool:
    """Dispatch and GEMM1 as one persistent stage-1 kernel; on by default.

    Compact flydsl path on the fp4 wire only: a config or step that does not
    fit keeps dispatch and GEMM1 as separate launches. ``AITER_MEGA_S1F=0``
    turns it off.
    """
    mode = os.environ.get("AITER_MEGA_S1F", "1").strip() or "1"
    if mode not in ("0", "1"):
        raise ValueError(f"AITER_MEGA_S1F must be 0 or 1, got {mode!r}")
    return mode == "1"


def _s1f_cu_num() -> int:
    from aiter.jit.utils.chip_info import get_cu_num

    return int(get_cu_num())


def _fallback_warn(owner, reasons) -> None:
    """Warn once per instance that this configuration keeps the default path."""
    if not reasons or getattr(owner, "_warned", False):
        return
    owner._warned = True
    # S1F is on by default, so a config that does not fit is not a warning.
    _log.info("S1F falls back to dispatch + GEMM1: %s", "; ".join(reasons))


def tile_flag_nbytes(*, compact_rows: int, world_size: int) -> int:
    from .s1f_stage1 import FLAG_STRIDE

    tiles = (int(compact_rows) + _S1F_MIN_TILE_M - 1) // _S1F_MIN_TILE_M
    assert int(world_size) * 4 <= FLAG_STRIDE
    return tiles * FLAG_STRIDE


def _wire_row(config) -> int:
    return _align_up(config.dispatch_token_nbytes + config.dispatch_scale_nbytes, 128)


def blocking_reasons(config) -> list:
    """Why this compact config cannot build the stage-1 kernel.

    Empty means it can. The env switch is the caller's to check: this is the
    shape gate only, shared by the arena (so ``required_vmm_bytes`` matches
    what the instance allocates) and by ``_Stage1Fusion.arm``.
    """
    reasons = []
    if config.dispatch_wire != "fp4":
        reasons.append(f"needs the fp4 dispatch wire, got {config.dispatch_wire}")
    if config.world_size > 8:
        reasons.append(f"world_size {config.world_size} > 8 (8-bit source mask)")
    if config.experts_per_rank > 255:
        reasons.append(f"{config.experts_per_rank} experts per rank > 255")
    if config.max_tokens_per_rank > 0xFFFF:
        reasons.append(f"max_tokens_per_rank {config.max_tokens_per_rank} > 65535")
    wire = _wire_row(config)
    if wire > 0xFFFF:
        reasons.append(f"wire row {wire} B > 65535")
    from .s1f_stage1 import _ITEM_N_BITS

    max_n_tiles = 1 << _ITEM_N_BITS
    inter = int(config.geometry.inter_dim)
    if 2 * inter > max_n_tiles * _S1F_TILE_N:
        reasons.append(f"inter_dim {inter} needs more than {max_n_tiles} n-tiles")
    from .s1f_stage1 import _SCALE_PIECE, N_XCC, S1F_PRODUCERS

    if config.dispatch_scale_nbytes % _SCALE_PIECE:
        reasons.append(
            f"e8m0 row {config.dispatch_scale_nbytes} B is not a "
            f"{_SCALE_PIECE} B multiple"
        )
    try:
        cu = _s1f_cu_num()
    except Exception as exc:  # noqa: BLE001 -- sizing may run with no GPU
        reasons.append(f"CU count unavailable ({exc})")
        return reasons
    model_dim = int(config.hidden_dim)
    tiles = {
        t
        for t, (nb, cpc) in _S1F_VARIANTS.items()
        if model_dim // _S1F_TILE_K >= nb
        and (cpc * cu) % N_XCC == 0
        and max(S1F_PRODUCERS) < cpc * cu
    }
    if not tiles:
        reasons.append(f"no stage-1 variant for model_dim {model_dim} on {cu} CUs")
    return reasons


def allocates_flags(config) -> bool:
    """Whether the arena for ``config`` carries the stage-1 flag region."""
    if not config.stage1_fused or not read_s1f_mode():
        return False
    return not blocking_reasons(config)


class _Stage1Fusion:
    """The stage-1 kernel's buffers and the GEMM1 stand-in, on one dispatch."""

    def __init__(self, dispatch):
        self.d = dispatch
        self.enabled = False
        self._warned = False
        self.other_ok = False
        self.cu = 0
        self.tiles: set[int] = set()
        self.tile_cap = 0
        # (align_m, quant_wmma_rep) of every step bound that runs the kernel.
        self.launch_keys: set = set()
        self.step = None
        self.wire = None
        self.send_lists = ()
        self.run_bufs = ()
        self.tile_srcs = ()
        self.ctl = None
        self.ticket_span = 0
        self.launches: dict = {}
        self.precompiled: set = set()
        self._packed_sc = None

    def arm(self) -> None:
        """Allocate the stage-1 buffers, or record why this instance cannot.

        The arena was sized by ``allocates_flags``, so a config that arms asks
        for a flag region that is already there.
        """
        self.enabled = False
        self._packed_sc = None
        if not read_s1f_mode():
            return
        config = self.d.config
        reasons = []
        if not config.stage1_fused:
            reasons.append("needs stage1_fused (the compact plan)")
        reasons += blocking_reasons(config)
        if reasons:
            _fallback_warn(self, reasons)
            return
        from .compact_plan import compact_run_buffer_dwords
        from .s1f_stage1 import N_XCC, S1F_PRODUCERS, s1f_ctl_words

        device = self.d.res.total_recv.device
        self.cu = _s1f_cu_num()
        model_dim = int(config.hidden_dim)
        self.tiles = {
            t
            for t, (nb, cpc) in _S1F_VARIANTS.items()
            if model_dim // _S1F_TILE_K >= nb
            and (cpc * self.cu) % N_XCC == 0
            and max(S1F_PRODUCERS) < cpc * self.cu
        }
        rows = int(self.d.compact.cap)
        self.tile_cap = (rows + _S1F_MIN_TILE_M - 1) // _S1F_MIN_TILE_M
        slots = self.d._PLAN_SLOTS
        max_routes = int(self.d.compact.max_routes)
        self.send_lists = tuple(
            torch.zeros(max_routes, dtype=torch.int32, device=device)
            for _ in range(slots)
        )
        run_words = compact_run_buffer_dwords(
            npes=config.world_size,
            experts_per_rank=config.experts_per_rank,
            max_routes=max_routes,
            s1f=True,
        )
        self.run_bufs = tuple(
            torch.zeros(run_words, dtype=torch.int32, device=device)
            for _ in range(slots)
        )
        self.tile_srcs = tuple(
            torch.zeros(self.tile_cap, dtype=torch.int32, device=device)
            for _ in range(slots)
        )
        self.ctl = torch.zeros(
            s1f_ctl_words(npes=config.world_size, tile_cap=self.tile_cap),
            dtype=torch.int32,
            device=device,
        )
        self.ticket_span = self.cu * math.lcm(*(c for _, c in _S1F_VARIANTS.values()))
        # The producer gathers whole wire rows, so the sender quant writes
        # payload | e8m0 | pad at the compact wire pitch.
        self.wire = torch.empty(
            (config.max_tokens_per_rank, self.d.compact.wire_row),
            dtype=torch.uint8,
            device=device,
        )
        self.enabled = True

    def extra_bufs(self, slot: int) -> dict:
        if not self.enabled:
            return {}
        return {
            "send_list": self.send_lists[slot],
            "runs": self.run_bufs[slot],
            "tile_src": self.tile_srcs[slot],
        }

    def tile_reasons(self, align_m: int) -> list:
        """Why a step aligned to ``align_m`` cannot run the stage-1 kernel."""
        from .s1f_stage1 import s1f_flag_m

        reasons = []
        if int(align_m) not in _S1F_VARIANTS:
            reasons.append(f"step tile_m {align_m} not in {sorted(_S1F_VARIANTS)}")
        elif int(align_m) not in self.tiles:
            reasons.append(
                f"stage-1 tile {align_m} does not fit model_dim "
                f"{self.d.config.hidden_dim} on {self.cu} CUs"
            )
        elif int(align_m) > s1f_flag_m(align_m) and self.d.config.world_size > 4:
            reasons.append(
                f"step tile_m {align_m} with {self.d.config.world_size} ranks > 4"
            )
        if int(self.d.config.experts_per_rank) > 255:
            reasons.append(f"{self.d.config.experts_per_rank} experts per rank > 255")
        return reasons

    def plan_mode(self, align_m: int) -> bool:
        """Whether the plan for ``align_m`` emits the stage-1 kernel's inputs."""
        return self.enabled and not self.tile_reasons(align_m)

    def step_reasons(self, step) -> list:
        """Why ``step`` cannot run the stage-1 kernel, bias and activation aside.

        GEMM1's tile_m is resolved with the step (``_compact_gemm_tiles``), so
        the verdict is known before the step runs, captured or not.
        """
        reasons = self.tile_reasons(step.align_m)
        if int(step.recv_bound) > self.max_recv():
            reasons.append(f"recv_bound {int(step.recv_bound)} > {self.max_recv()}")
        if int(step.g1_tile_m) != int(step.align_m):
            reasons.append(f"GEMM1 tile_m {step.g1_tile_m} != {step.align_m}")
        return reasons

    def _recv_too_big(self) -> bool:
        # The receive bound is identical on every rank.
        step = getattr(self.d, "_step", None)
        return step is not None and int(step.recv_bound) > self.max_recv()

    def max_recv(self) -> int:
        config = self.d.config
        return s1f_max_recv(config.world_size, config.topk)

    def step_off(self) -> bool:
        """A step over ``max_recv()`` runs exactly the S1F=0 path.

        Not only GEMM1: its compact plan keeps the S1F=0 key and width (one
        kernel per alignment and slot, instead of one per launch width, block
        count, tok_map and fused-quant choice). Before,
        every new prefill width JIT-compiled a stage-1 plan inside the serving
        loop (first ~100 s of a c512 run: 250-470 ms prefill steps), and the
        wide stage-1 plan left the remaining prefill steps ~6% slower.
        """
        return self.enabled and self._recv_too_big()

    def eligible(self, bias1, bias2) -> bool:
        """Whether this step runs the stage-1 kernel (else: today's path)."""
        if not self.enabled:
            return False
        reasons = []
        for bias in (bias1, bias2):
            if bias is not None and bias.numel() > 0:
                reasons.append("bias is not supported")
        act = self.d.config.recipe.compute.activation
        if act not in (ActivationType.Silu, ActivationType.Swiglu):
            reasons.append(f"activation {act} is not supported")
        if os.environ.get("AITER_FLYDSL_DISABLE_GEMM1_REQUANT", "0") not in (
            "",
            "0",
            "false",
            "False",
        ):
            reasons.append("needs the fused GEMM1 requant epilogue")
        self.other_ok = not reasons
        reasons += self.step_reasons(self.d._step)
        if reasons:
            _fallback_warn(self, reasons)
            return False
        return True

    def step_will_run(self) -> bool:
        """Whether the step whose geometry is current will run the stage-1 kernel.

        Used to drop the tok_map rewrite from a plan launched ahead of the
        step. Anything unknown answers False, so the plan keeps tok_map and
        the default dispatch stays possible.
        """
        if not (self.enabled and self.other_ok):
            return False
        return not self.step_reasons(self.d._step)

    def wants_fused_quant(self, run: bool) -> bool:
        """The plan can quantize only when it is launched with the hidden states."""
        d = self.d
        return bool(
            _QUANT_FUSE_BLOCKS
            and run
            and self.wire is not None
            and not d._plan_pending
            and self.plan_mode(d._step.align_m)
            and not self.step_off()
        )

    def quantize(self, hidden, token_count: int, run: bool):
        """``(payload, scales_ptr, quant_src)`` for this step.

        ``quant_src`` is the hidden states when the plan's own quant partition
        writes the wire rows; None means the quant already ran here.
        """
        d = self.d
        if d._quant_payload is None:
            return hidden, 0, None
        from aiter.ops.quant import dynamic_per_group_scaled_quant

        config = d.config
        payload = d._quant_payload[:token_count]
        scale_rows = d._quant_scales[:token_count]
        wire = self.wire if run else None
        if self.wants_fused_quant(run):
            return (
                payload,
                wire[:token_count].data_ptr() + config.dispatch_token_nbytes,
                hidden,
            )
        if wire is not None:
            pb = config.dispatch_token_nbytes
            sb = config.dispatch_scale_nbytes
            rows = wire[:token_count]
            dynamic_per_group_scaled_quant(
                rows[:, :pb].view(payload.dtype),
                hidden,
                rows[:, pb : pb + sb].view(scale_rows.dtype),
                32,
                shuffle_scale=False,
            )
            return payload, rows.data_ptr() + pb, None
        out, scales_ptr = d._quantize(hidden, token_count)
        return out, scales_ptr, None

    def stash(self, *, slot, token_count, payload, scales_ptr, wts_ptr) -> None:
        wire = self.wire
        self.step = {
            "slot": int(slot),
            "token_count": int(token_count),
            "payload_ptr": payload.data_ptr() if wire is None else wire.data_ptr(),
            "scale_ptr": int(scales_ptr),
            "wts_ptr": int(wts_ptr),
        }

    def finish(self, ran: bool) -> None:
        if ran and self.step is not None:
            self.step = None
            raise RuntimeError(
                "AITER_MEGA_S1F: the GEMM1 launch never reached the stage-1 "
                "hook, so this layer's rows were never dispatched"
            )

    def _variant(self, tokens: int):
        from .s1f_stage1 import s1f_block_runs, s1f_producers

        return s1f_producers(tokens), s1f_block_runs(tokens)

    def _launcher(
        self, *, tile_m, producers, group_m, stage1_act, quant_wmma_rep, block_runs
    ):
        from .s1f_stage1 import make_s1f_stage1

        key = (tile_m, producers, group_m, stage1_act, quant_wmma_rep, block_runs)
        launch = self.launches.get(key)
        if launch is None:
            d, config = self.d, self.d.config
            num_buffers, ctas_per_cu = _S1F_VARIANTS[tile_m]
            m_warp, n_warp = _s1f_warps(tile_m)
            launch = make_s1f_stage1(
                rank=config.rank,
                npes=config.world_size,
                experts_per_rank=config.experts_per_rank,
                topk=config.topk,
                payload_bytes=config.dispatch_token_nbytes,
                scale_bytes=config.dispatch_scale_nbytes,
                row_stride=d.compact.wire_row,
                compact_cap=d.compact.cap,
                max_tok_per_rank=config.max_tokens_per_rank,
                max_routes=d.compact.max_routes,
                off_out_tok=d.res.arena.offset("disp_out"),
                off_ep_rowmap=d.res.arena.offset("ep_rowmap"),
                off_tile_flag=d.res.arena.offset("tile_flag"),
                model_dim=config.hidden_dim,
                inter_dim=config.geometry.inter_dim,
                stage1_act=stage1_act,
                quant_wmma_rep=quant_wmma_rep,
                workers=ctas_per_cu * self.cu,
                producers=producers,
                group_m=group_m,
                tile_m=tile_m,
                num_buffers=num_buffers,
                block_runs=block_runs,
                ticket_span=self.ticket_span,
                ctas_per_cu=ctas_per_cu,
                m_warp=m_warp,
                n_warp=n_warp,
                **_S1F_GEMM_OPTS.get(tile_m, {}),
            )
            self.launches[key] = launch
        return launch

    def _packed_scales(self) -> torch.Tensor:
        """The e8m0 rows behind the packed A in disp_out (see make_s1f_stage1)."""
        if self._packed_sc is None:
            from .types import _from_gpu_ptr

            config, geo = self.d.config, self.d.compact
            sb = int(config.dispatch_scale_nbytes)
            base = self.d.res.arena.local_ptr("disp_out")
            self._packed_sc = _from_gpu_ptr(
                base + int(geo.cap) * int(config.dispatch_token_nbytes),
                (int(geo.cap), sb),
                torch.uint8,
            ).view(torch.int32)
        return self._packed_sc

    @staticmethod
    def _run(launch, args) -> None:
        # Calling the @flyc.jit launcher rebuilds its cache key on every launch
        # (~110 us/layer of host time in eager prefill). Its tensor arguments
        # are layout-dynamic, so only their dtype and rank pick the artifact.
        spec = tuple((a.dtype, a.dim()) for a in args if isinstance(a, torch.Tensor))
        _run_compiled(launch, *args, specialization_key=spec)

    def _precompile(self, launch_args, *, tile_m, group_m, stage1_act, quant_wmma_rep):
        from .compact_plan import compact_run_buffer_dwords

        config, d = self.d.config, self.d
        dev = self.ctl.device
        runs = torch.zeros(
            compact_run_buffer_dwords(
                npes=config.world_size,
                experts_per_rank=config.experts_per_rank,
                max_routes=d.compact.max_routes,
                s1f=True,
            ),
            dtype=torch.int32,
            device=dev,
        )
        send = torch.zeros(d.compact.max_routes, dtype=torch.int32, device=dev)
        tsrc = torch.zeros(max(1, self.tile_cap), dtype=torch.int32, device=dev)
        psum = torch.zeros(
            max(1, config.experts_per_rank), dtype=torch.int32, device=dev
        )
        seen = set()
        for tok in (1, 129, 1025, 2049):
            producers, block_runs = self._variant(tok)
            launch = self._launcher(
                tile_m=tile_m,
                producers=producers,
                group_m=group_m,
                stage1_act=stage1_act,
                quant_wmma_rep=quant_wmma_rep,
                block_runs=block_runs,
            )
            if id(launch) not in seen:
                seen.add(id(launch))
                self._run(launch, launch_args(psum, send, runs, tsrc, 0, tile_m))
        torch.cuda.synchronize()
        d.res.arena._communicator.barrier()

    def gemm1(
        self,
        out,
        a,
        w,
        a_scales,
        w_scales,
        m_tile_map,
        *,
        n_experts,
        contiguous_m,
        N,
        K,
        tile_m=64,
        tile_n=256,
        tile_k=256,
        m_warp=1,
        n_warp=4,
        num_buffers=3,
        out_is_f16=0,
        a_is_fp4=0,
        stage1_act=0,
        bias=None,
        swiglu_limit=7.0,
        stream=None,
        stage1_quant_out=0,
        quant_scale=None,
        quant_wmma_rep=1,
        cluster_m=-1,
        cluster_n=-1,
        waves_per_tensor_tdm=-1,
        next_stage_prefetch=0,
        tdm_as_in_prologue=0,
        tdm_b_th=0,
        lds_soa_load_interleave=0,
        row_major_ascale=0,
        a_row_stride_bytes=0,
        a_scale_row_stride_bytes=0,
        situ_beta=1.0,
        situ_linear_beta=1.0,
        persistent_workers=0,
        rolled_drain=0,
        **_unused,
    ):
        """GEMM1 launch stand-in: runs this layer's dispatch and GEMM1."""
        step = self.step
        if step is None:
            raise RuntimeError("AITER_MEGA_S1F: stage-1 hook called without a step")
        self.step = None
        align_m = int(self.d._step.align_m)
        wire_row = int(self.d.compact.wire_row)
        got = {
            "tile": (tile_m, tile_n, tile_k),
            "out_is_f16": int(out_is_f16),
            "a_is_fp4": int(a_is_fp4),
            "bias": bias is not None,
            "stage1_quant_out": int(stage1_quant_out),
            "row_major_ascale": int(row_major_ascale),
            "a_strides": (int(a_row_stride_bytes), int(a_scale_row_stride_bytes)),
            "variant": align_m,
        }
        want = {
            "tile": (align_m, _S1F_TILE_N, _S1F_TILE_K),
            "out_is_f16": 0,
            "a_is_fp4": 1,
            "bias": False,
            "stage1_quant_out": 1,
            "row_major_ascale": 1,
            "a_strides": (wire_row, wire_row),
            "variant": align_m if align_m in _S1F_VARIANTS else sorted(_S1F_VARIANTS),
        }
        # The stage-1 kernel runs its own 256x256 n/k tiles.
        got["tile"], want["tile"] = got["tile"][:1], want["tile"][:1]
        config = self.d.config
        prefill = s1f_prefill_tile(
            config.world_size, config.topk, self.d._step.recv_bound
        )
        if prefill == align_m and int(tile_m) <= align_m:
            # fused_moe keeps the CSV GEMM1 tile; this step widened it.
            got["tile"] = (align_m,)
        bad = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
        if bad or stage1_act not in (1, 2) or quant_scale is None:
            raise RuntimeError(
                "AITER_MEGA_S1F: the stage-1 kernel embeds only its variants' "
                f"GEMM1; mismatches (got, want): {bad}, stage1_act={stage1_act}. "
                "A tile mismatch means _compact_gemm_tiles no longer resolves "
                "the tiles fused_moe does"
            )
        tokens = int(step["token_count"])
        producers, block_runs = self._variant(tokens)
        group_m = _S1F_GROUP_M
        bufs = {**self.d._slot_bufs(step["slot"])}
        if stream is None:
            stream = torch.cuda.current_stream()

        def launch_args(psum_t, send_t, runs_t, tsrc_t, n_tok, tile=align_m):
            packed = _S1F_GEMM_OPTS.get(int(tile), {}).get("a_preshuffled", 0)
            return (
                out,
                ptr_arg(a),
                ptr_arg(w),
                self._packed_scales() if packed else a_scales.view(torch.int32),
                w_scales.view(torch.int32),
                quant_scale.view(torch.uint8),
                ptr_arg(psum_t),
                self.d.res.arena.handle,
                step["payload_ptr"],
                step["scale_ptr"],
                step["wts_ptr"],
                send_t.data_ptr(),
                runs_t.data_ptr(),
                tsrc_t.data_ptr(),
                self.ctl.data_ptr(),
                int(contiguous_m),
                n_tok,
                float(swiglu_limit),
                fx.Stream(stream),
            )

        # The first eager launch builds the launcher of every step bound that
        # can run the kernel, so a bucket first seen in a graph capture does
        # not JIT-compile there. The precompile launches are collective.
        keys = {(t, int(stage1_act), q) for t, q in self.launch_keys}
        keys.add((align_m, int(stage1_act), int(quant_wmma_rep)))
        if not keys <= self.precompiled and (
            not torch.cuda.is_current_stream_capturing()
        ):
            for t, act, q in sorted(keys - self.precompiled):
                self.precompiled.add((t, act, q))
                self._precompile(
                    launch_args,
                    tile_m=t,
                    group_m=group_m,
                    stage1_act=act,
                    quant_wmma_rep=q,
                )
        launch = self._launcher(
            tile_m=align_m,
            producers=producers,
            group_m=group_m,
            stage1_act=int(stage1_act),
            quant_wmma_rep=int(quant_wmma_rep),
            block_runs=block_runs,
        )
        self._run(
            launch,
            launch_args(
                m_tile_map, bufs["send_list"], bufs["runs"], bufs["tile_src"], tokens
            ),
        )
