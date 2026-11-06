# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host side of the single-kernel EP MegaMoE: launch configs, launches, and the
pure-torch reference oracle used to validate the kernel."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F

from aiter.utility import fp4_utils

from ..mega_moe.quant import per_1x32_mx_quant
from ..mega_moe_tp.mega_moe_tp import SymmetricArena
from ..tensor_shim import _preload_compiled, _run_compiled
from .mega_moe_ep_config import (
    EPOCH_SLOT,
    FLAG_INTS,
    MAX_NPES,
    N_XCD,
    NCTA_MAX,
    SD_BASE,
    SD_DUM,
    SD_INTS,
    SD_QP,
    SD_QS,
    SDMA_ENV,
    ctrl_ints,
    ep_shape_supported,
    inter_rows,
    mega_moe_ep_consts,
    sbx_on,
)
from .mega_moe_ep_kernel import compile_mega_moe_ep

__all__ = ["LaunchCfg", "MegaMoeEPEngine", "ep_reference"]

LDS_LIMIT = 160 * 1024
# a4w4 row tile: mt=8 (RG=128: each W tile serves twice the rows) from MT8_MIN
# tokens, else mt=4.  mt=8 needs the per-echunk inter rebase (kernel_config forces it):
# without it the K-loop spills.  a8w4 mt=8 needs > 160 KB LDS (only a 2-bank A ring
# fits), so a8 stays at mt=4.
MT8_MIN = 768
# Small-batch kernel variant (LaunchCfg.small) for engines of max_tok_per_rank <= SB_MAX:
# picked from mtpr, so every rank runs the same variant.  a8w4 row tile mt=2 up to
# SB_MT2_MAX tokens; mt only shapes the rank-local echunks.
SB_MAX = 256
SB_MT2_MAX = 159
# a4w4 likewise up to SB_MT2_MAX_A4.
SB_MT2_MAX_A4 = 159
# SB kernel streams its weights (kernel_config WNT) from this many tokens.
SB_WNT_MIN = 2


def _dequant_expert(weight, scale, rows, cols):
    """Dequantize one MXFP4 expert weight (packed FP4 values * per-1x32 E8M0 scale)."""
    values = fp4_utils.mxfp4_to_f32(weight).view(rows, cols)
    scales = fp4_utils.e8m0_to_f32(scale).view(rows, cols // 32)
    return values * scales.repeat_interleave(32, dim=-1)


def _all_gather_variable(tensor, token_counts=None):
    """All-gather rows across ranks with per-rank variable counts (pad, gather, trim)."""
    world = dist.get_world_size()
    if token_counts is None:
        local = torch.tensor(tensor.shape[0], dtype=torch.int64, device=tensor.device)
        counts = [torch.empty_like(local) for _ in range(world)]
        dist.all_gather(counts, local)
        token_counts = [int(c.item()) for c in counts]
    max_count = max(token_counts)
    if tensor.shape[0] < max_count:
        padding = torch.empty(
            (max_count - tensor.shape[0], *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        tensor = torch.cat((tensor, padding))
    gathered = [torch.empty_like(tensor) for _ in range(world)]
    dist.all_gather(gathered, tensor)
    return torch.cat([v[:c] for v, c in zip(gathered, token_counts)]), token_counts


@torch.no_grad()
def ep_reference(
    x,
    route_weights,
    ids,
    ref_weights,
    *,
    rank,
    world_size,
    model_dim,
    inter_dim,
    experts,
    swiglu_limit,
    act="silu",
    situ_beta=1.0,
    situ_linear_beta=1.0,
):
    """Pure-torch EP MoE oracle matching MegaMoEV2 numerics: dispatch (all-gather +
    route to the expert owner) -> per-local-expert GEMM1 + act + GEMM2 scaled by the
    top-k weight -> combine (all-reduce the partials, keep this rank's rows).

    ``ref_weights`` are the *unshuffled* quantized weights
    ``(w1_q, w1_scale, w2_q, w2_scale)`` as handed to MegaMoEV2 (see the op-test's
    ``_quantize_weights``). Collective when torch.distributed is initialized and
    ``world_size > 1``; otherwise a single-process local MoE (all experts local).
    """
    distributed = world_size > 1 and dist.is_available() and dist.is_initialized()
    if distributed:
        x_all, token_counts = _all_gather_variable(x)
        weights_all, _ = _all_gather_variable(route_weights, token_counts)
        ids_all, _ = _all_gather_variable(ids, token_counts)
    else:
        x_all, weights_all, ids_all = x, route_weights, ids
        token_counts = [x.shape[0]]
    partial = torch.zeros(
        (x_all.shape[0], model_dim), dtype=torch.float32, device=x.device
    )
    w1_q, w1_scale, w2_q, w2_scale = ref_weights
    local_experts = experts // world_size
    expert_start = rank * local_experts
    active = torch.unique(ids_all)
    active = active[(active >= expert_start) & (active < expert_start + local_experts)]
    for expert in active.tolist():
        positions = torch.nonzero(ids_all == expert, as_tuple=False)
        rows, slots = positions[:, 0], positions[:, 1]
        lid = expert - expert_start
        w1 = _dequant_expert(w1_q[lid], w1_scale[lid], 2 * inter_dim, model_dim)
        w2 = _dequant_expert(w2_q[lid], w2_scale[lid], model_dim, inter_dim)
        inp = x_all[rows].float()
        gate = inp @ w1[:inter_dim].T
        up = inp @ w1[inter_dim:].T
        if act == "situv2":
            gate = situ_beta * torch.tanh(gate / situ_beta) * torch.sigmoid(gate)
            up = situ_linear_beta * torch.tanh(up / situ_linear_beta)
            hidden = gate * up
        else:
            if swiglu_limit > 0:
                gate = gate.clamp(max=swiglu_limit)
                up = up.clamp(-swiglu_limit, swiglu_limit)
            hidden = F.silu(gate) * up
        out = (hidden @ w2.T) * weights_all[rows, slots, None]
        partial.index_add_(0, rows, out)
    if distributed:
        dist.all_reduce(partial)
        start = sum(token_counts[:rank])
        return partial[start : start + x.shape[0]]
    return partial


@dataclass(frozen=True)
class LaunchCfg:
    """One EP kernel variant: row tile 16*mt; ``a8`` MXFP8 activations.  The payload is
    always received source-indexed (row = rank*mtpr + token, bounded by npes*mtpr)."""

    mt: int
    a8: bool = False
    small: bool = False  # small-batch kernel variant (mtpr <= SB_MAX)
    wnt: bool = False  # SB: non-temporal weight loads (m >= SB_WNT_MIN)


class MegaMoeEPEngine:
    """Host side of the fused EP MoE (see the module docstring): owns the symmetric
    arena and the plan tables, binds the weights, and launches one grid=#CU, block=512
    kernel per ``forward``.  ``reference_forward`` / ``ep_reference`` are the pure-torch
    oracle."""

    def __init__(
        self,
        *,
        rank,
        world_size,
        model_dim,
        inter_dim,
        experts,
        topk,
        max_tok_per_rank,
        w1,
        w1_scale,
        w2,
        w2_scale,
        quant="a4w4",
        act="silu",
        situ_beta=1.0,
        situ_linear_beta=1.0,
        swiglu_limit=0.0,
        group=None,
        device=None,
        weights_preshuffled=False,
    ):
        if quant not in ("a4w4", "a8w4"):
            raise ValueError("MegaMoeEP quant must be 'a4w4' or 'a8w4'")
        if not ep_shape_supported(model_dim, inter_dim, experts, world_size):
            raise ValueError(
                f"fused EP MegaMoE does not tile h{model_dim} i{inter_dim} "
                f"e{experts} np{world_size}"
            )
        mtpr = int(max_tok_per_rank)
        if mtpr <= 0 or mtpr & (mtpr - 1):
            raise ValueError(f"max_tok_per_rank={mtpr} must be a power of two")
        if not 0 < topk <= 16:
            raise ValueError(f"MegaMoeEP topk must be in [1, 16], got {topk}")
        if act not in ("silu", "situv2"):
            raise ValueError(f"MegaMoeEP act must be 'silu' or 'situv2', got {act!r}")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.H = int(model_dim)
        self.I = int(inter_dim)
        self.E = int(experts)
        self.epr = int(experts // world_size)
        self.K = int(topk)
        self.mtpr = mtpr
        self.a8 = quant == "a8w4"
        self.act = act
        self.situ = (float(situ_beta), float(situ_linear_beta))
        self.swiglu_limit = float(swiglu_limit)
        self.group = group
        if device is None:
            device = torch.device("cuda", self.rank)
        elif not isinstance(device, torch.device):
            device = torch.device(device)
        if device.index is None:
            device = torch.device(device.type, self.rank)
        self.device = device
        self._bind_weights(w1, w1_scale, w2, w2_scale, weights_preshuffled)
        props = torch.cuda.get_device_properties(self.device)
        self.n_cta = int(props.multi_processor_count)
        if self.n_cta > NCTA_MAX:
            raise ValueError(f"{self.n_cta} CUs > NCTA_MAX={NCTA_MAX} (flag sizing)")
        self.gfx = getattr(props, "gcnArchName", "").split(":")[0]
        if self.gfx != "gfx950":
            raise ValueError(f"fused EP MegaMoE needs gfx950, not {self.gfx}")
        self._cfgs: dict = {}
        self._launchers: dict = {}
        self._armed = False
        # hipIpc symmetric arena: dispatch deposits each routed token's payload here on
        # the expert-owner rank (indexed layout, world_size*mtpr rows); combine deposits
        # the weighted GEMM2 partials back on the source rank; the epoch flags live in it
        # too. Reserved in the same order on every rank (collective commit maps peers).
        tot = self.world_size * self.mtpr
        a_row = self.H if self.a8 else self.H // 2
        arena = SymmetricArena(group=group, device=self.device)
        self._payload = arena.reserve("payload", (tot, a_row), torch.uint8)
        self._pscale = arena.reserve("pscale", (tot, self.H // 32), torch.uint8)
        self._recv = arena.reserve("recv_le", (tot * self.K,), torch.int32)
        # combine partials sent fp8 per-8-blockwise (half the a2a traffic): each slot row is
        # H fp8-e4m3 bytes then H/8 per-8 E8M0 scale bytes (co-located, one buffer).
        # SBX: two banks by epoch parity (no barrier orders a launch's part stores
        # after the previous launch's combine on the source).
        nbank = 2 if sbx_on(self.mtpr <= SB_MAX) else 1
        self._part = arena.reserve(
            "part", (nbank * self.mtpr * self.K, self.H + self.H // 8), torch.uint8
        )
        self._flag = arena.reserve("flag", (FLAG_INTS,), torch.int32)
        self._out = arena.reserve("out", (self.mtpr, self.H), torch.bfloat16)
        # AITER_MEGAMOE_EP_SDMA (large kernel): the payload rows / scales packed per owner
        # (stage / sstage), the owner's compacted row of each received (source, token)
        # (rpos), the target of a round without rows (sdum)
        self.sdma = SDMA_ENV if self.mtpr > SB_MAX else 0
        sd = bool(self.sdma)
        self._stage = arena.reserve("stage", (tot if sd else 1, a_row), torch.uint8)
        self._sstage = arena.reserve(
            "sstage", (tot if sd else 1, self.H // 32), torch.uint8
        )
        self._rpos = arena.reserve("rpos", (tot if sd else 1,), torch.int32)
        self._sdum = arena.reserve("sdum", (MAX_NPES * SD_DUM,), torch.uint8)
        self._sdq = None
        if sd:
            from .sdma import SdmaQueues

            self._sdq = SdmaQueues(group=group, device=self.device)
        arena.commit()
        self.arena = arena
        self._sdst = torch.zeros(SD_INTS, dtype=torch.int32, device=self.device)
        self._spos = torch.empty(
            self.mtpr * MAX_NPES if sd else 1, dtype=torch.int32, device=self.device
        )
        if sd:  # this launch's signal base per (source, payload / scale queue)
            sig = self._sdq.signals().cpu()
            for s_ in range(self.world_size):
                self._sdst[SD_BASE + 2 * s_] = int(sig[s_, SD_QP])
                self._sdst[SD_BASE + 2 * s_ + 1] = int(sig[s_, SD_QS])
        self._recv.local.fill_(-1)  # SBX: scrubbed words (SBX_INV), never a launch's
        self.ctrl = torch.zeros(
            ctrl_ints(tot, self.K, self.epr), dtype=torch.int32, device=self.device
        )
        # Epoch starts at 1: the arena ready-flags are zero-initialized and the waits are
        # serial-number ``flag >= epoch`` (wrap-safe ``before``), so epoch 0 would make
        # every dispatch / combine wait pass instantly (0 >= 0) and all cross-wave /
        # cross-rank ordering would collapse into a race.  It lives in ctrl (the kernel
        # reads and bumps it) so CUDA-graph replays each get a new one.
        self.ctrl[EPOCH_SLOT] = 1
        # Per-CTA GEMM scratch (local, NOT peer-visible): GEMM1 writes the E8M0-quantized
        # intermediate (inter / inters), GEMM2 reads it.
        a_pack = 1 if self.a8 else 2
        # inter / inters / rowmap / rowidx are indexed by ECHUNK id (0..nch) * RG + row, not
        # bid: sized for the worst-case routing (inter_rows), so any skew fits.
        rows_cap = inter_rows(tot, self.K, self.epr)
        self._inter = torch.empty(
            rows_cap * (self.I // a_pack), dtype=torch.uint8, device=self.device
        )
        self._inters = torch.empty(
            rows_cap * (self.I // 32), dtype=torch.uint8, device=self.device
        )
        # Per-echunk GEMM1 A-gather source keys written to global by the gather pass:
        # rowmap = payload row (recv_slot // topk) for the V2 ATileLoader indexed gather;
        # rowidx = the full recv_slot (the combine pass recovers route = idx % topk and
        # the source rank/token from it -- read in a later pass so it cannot stay in the
        # per-CTA LDS L_RIX). Indexed echunk*RG + r.
        self._rowmap = torch.empty(rows_cap, dtype=torch.int32, device=self.device)
        self._rowidx = torch.empty(rows_cap, dtype=torch.int32, device=self.device)
        # Balanced-gather tables: the planner's pre-pass assigns each received
        # (token,route) pair a GLOBAL per-local-expert position (0..R_e-1 rank via _ecnt
        # atomics) and buckets its recv slot at _rowlist[le*cap + pos] (cap = world*mtpr:
        # a token routes to an expert at most once). The planner chunks an expert's rows
        # by position//RG, so every chunk is a full RG-row tile (no M-padding waste) and
        # the gather reads it by position. _ecnt is zeroed each forward; the small-batch
        # kernel uses one EPR block of it per XCD.
        self._rowlist = torch.empty(
            self.epr * self.world_size * self.mtpr,
            dtype=torch.int32,
            device=self.device,
        )
        self._ecnt = torch.zeros(
            self.epr * N_XCD, dtype=torch.int32, device=self.device
        )

    def _bind_weights(self, w1, w1_scale, w2, w2_scale, preshuffled=False):
        # GEMM1 is the V2 pipelined path: its BWeightLoader / BScaleLoader read W1 and
        # its E8M0 scale in the shuffle_weight_a16w4 / shuffle_scale_a16w4 (gate|up
        # interleave) B-layout, so shuffle them host-side (natural weights are fed by
        # the tests / oracle). GEMM2 stays the naive natural-layout read -> W2 natural.
        if preshuffled:
            # Already in those layouts (MegaMoEV2's): bound by pointer, not copied; the
            # caller keeps them alive (and may update them in place, e.g. EPLB).
            want = (
                (w1, self.epr * 2 * self.I * self.H // 2),
                (w1_scale, self.epr * 2 * self.I * self.H // 32),
                (w2, self.epr * self.H * self.I // 2),
                (w2_scale, self.epr * self.H * self.I // 32),
            )
            for t, n in want:
                if not t.is_contiguous() or t.numel() * t.element_size() != n:
                    raise ValueError(
                        f"MegaMoeEP: preshuffled weight {tuple(t.shape)} {t.dtype} is not "
                        f"a contiguous {n}-byte tensor"
                    )
            self.w1, self.w1s, self.w2, self.w2s = w1, w1_scale, w2, w2_scale
            return
        from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

        w1 = w1 if w1.is_contiguous() else w1.contiguous()
        self.w1 = shuffle_weight_a16w4(w1, 16, True).contiguous()
        self.w1s = shuffle_scale_a16w4(
            w1_scale.reshape(-1, self.H // 32).contiguous(), self.epr, True
        ).contiguous()
        # GEMM2 is now the V2 pipelined path too: shuffle W2 + its scale into the
        # down-proj (gate_up=False) B-layout the BWeightLoader / BScaleLoader read.
        w2 = w2 if w2.is_contiguous() else w2.contiguous()
        self.w2 = shuffle_weight_a16w4(w2, 16, False).contiguous()
        self.w2s = shuffle_scale_a16w4(
            w2_scale.reshape(-1, self.I // 32).contiguous(), self.epr, False
        ).contiguous()

    def set_weights(self, w1, w1_scale, w2, w2_scale, preshuffled=False):
        self._bind_weights(w1, w1_scale, w2, w2_scale, preshuffled)

    def _fit_mt(self, m, small):
        # The V2 pipelined GEMM1 packs rows in pairs (MfmaScaleGU / AScaleLoader use
        # m_repeat//2 scale groups), so the row tile mt (=m_repeat) must be EVEN.
        # The inline-A GEMM stages A with all NT=512 threads (16 B/thread); a smaller A
        # tile (a4 mt=2: 32 x 128 B) has the surplus threads repeat a chunk.
        if small and m <= (SB_MT2_MAX if self.a8 else SB_MT2_MAX_A4):
            mt = 2
        else:
            mt = 8 if not self.a8 and m >= MT8_MIN else 4
        if mt % 2:
            raise ValueError(f"mt must be even, got mt={mt}")
        while mt > 4:
            if (
                mega_moe_ep_consts(
                    self.H,
                    self.I,
                    mt,
                    self.epr,
                    self.mtpr * self.world_size,
                    self.a8,
                    small,
                )["LDS_BYTES"]
                <= LDS_LIMIT
            ):
                break
            mt -= 4
        return mt

    def config(self, m):
        cfg = self._cfgs.get(m)
        if cfg is None:
            small = self.mtpr <= SB_MAX
            cfg = LaunchCfg(
                mt=self._fit_mt(m, small),
                a8=self.a8,
                small=small,
                wnt=small and m >= SB_WNT_MIN,
            )
            self._cfgs[m] = cfg
        return cfg

    def _variants(self):
        """Every LaunchCfg a token count in [0, mtpr] selects (mt / wnt only change at the
        SB_WNT_MIN / SB_MT2 / MT8_MIN bounds)."""
        th = SB_MT2_MAX if self.a8 else SB_MT2_MAX_A4
        ms = {0, 1, SB_WNT_MIN, th, th + 1, MT8_MIN - 1, MT8_MIN, self.mtpr}
        return sorted(
            {self.config(m) for m in ms if 0 <= m <= self.mtpr}, key=lambda c: c.mt
        )

    def prepare(self):
        """Collective: compile every launch variant on every rank, then barrier.  The
        ranks of one launch may pick different variants (per-rank m), and a peer that
        JITs inside the launch stalls the others past the watchdog DEADLINE; forward()
        runs this on its first call (must not be under CUDA graph capture)."""
        if self._armed:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "MegaMoeEP: first forward inside CUDA graph capture; call prepare() "
                "(or run forward eagerly) first"
            )
        stream = torch.cuda.current_stream()
        for cfg in self._variants():
            _preload_compiled(
                self._launcher(cfg),
                *self._launch_args(0, None, None, None, None, None),
                stream,
            )
        torch.cuda.synchronize(self.device)
        self.arena.barrier()
        self._armed = True

    def _launcher(self, cfg):
        fn = self._launchers.get(cfg)
        if fn is None:
            fn = compile_mega_moe_ep(
                H=self.H,
                I=self.I,
                TOPK=self.K,
                MT=cfg.mt,
                TMAX=self.mtpr * self.world_size,
                E=self.E,
                EPR=self.epr,
                NPES=self.world_size,
                act=self.act,
                situ_beta=self.situ[0],
                situ_linear_beta=self.situ[1],
                swiglu_limit=self.swiglu_limit,
                a8=cfg.a8,
                SB=cfg.small,
                WNT=cfg.wnt,
            )
            self._launchers[cfg] = fn
        return fn

    def reference_forward(self, x_bf16, wts, topk_ids, ref_weights):
        """Run the pure-torch oracle for these inputs. ``ref_weights`` are the
        unshuffled quantized weights ``(w1_q, w1_scale, w2_q, w2_scale)``."""
        return ep_reference(
            x_bf16,
            wts,
            topk_ids,
            ref_weights,
            rank=self.rank,
            world_size=self.world_size,
            model_dim=self.H,
            inter_dim=self.I,
            experts=self.E,
            swiglu_limit=self.swiglu_limit,
            act=self.act,
            situ_beta=self.situ[0],
            situ_linear_beta=self.situ[1],
        )

    def _launch_args(self, m, x, xs, ids, wts, out):
        """The kernel's positional args (minus stream): weights, local input, output,
        global ctrl, arena peer bases, arena region offsets, scalars, grid (#CU)."""
        peers = [int(b) for b in self.arena.base_ptrs]
        peers += [0] * (MAX_NPES - self.world_size)

        def ptr(t):
            return 0 if t is None else t.data_ptr()

        return (
            self.w1.data_ptr(),
            self.w1s.data_ptr(),
            self.w2.data_ptr(),
            self.w2s.data_ptr(),
            ptr(x),
            ptr(xs),
            ptr(ids),
            ptr(wts),
            ptr(out),
            self.ctrl.data_ptr(),
            *peers,
            self._payload.offset,
            self._pscale.offset,
            self._recv.offset,
            self._part.offset,
            self._flag.offset,
            self._out.offset,
            self._inter.data_ptr(),
            self._inters.data_ptr(),
            self._rowmap.data_ptr(),
            self._rowidx.data_ptr(),
            self._rowlist.data_ptr(),
            self._ecnt.data_ptr(),
            self.rank,
            self.world_size,
            self.epr,
            m,
            self.mtpr,
            self._sdq.ptr if self._sdq is not None else 0,
            self._sdst.data_ptr(),
            self._spos.data_ptr(),
            self._stage.offset,
            self._sstage.offset,
            self._rpos.offset,
            self._sdum.offset,
            self.n_cta,
        )

    def forward(
        self,
        x_bf16,
        wts,
        topk_ids,
        *,
        stream=None,
        slice_output=True,
        mask_invalid_slots=None,
    ):
        run_tokens = int(x_bf16.shape[0])
        if run_tokens > self.mtpr:
            raise ValueError(f"run_tokens={run_tokens} > max_tok_per_rank={self.mtpr}")
        del mask_invalid_slots  # -1 top-k slots are always skipped in-kernel
        self.prepare()
        cfg = self.config(run_tokens)
        fn = self._launcher(cfg)
        # Dynamic MX quant of the activations (same per-1x32 MX path as MegaMoEV2); the
        # scales are natural [m, H/32], which dispatch copies verbatim into pscale.  SBX:
        # sbx_send quantizes the bf16 rows itself (one launch less).
        if sbx_on(cfg.small):
            x_q, xs = x_bf16.contiguous(), None
        elif run_tokens:
            x_q, xs = per_1x32_mx_quant(
                x_bf16.contiguous(), quant_mode="fp8" if self.a8 else "fp4"
            )
        else:  # HIP rejects the quant's zero grid; the rank still launches (peers wait on it)
            x_q = torch.empty(
                (0, self.H if self.a8 else self.H // 2),
                dtype=torch.uint8,
                device=self.device,
            )
            xs = torch.empty((0, self.H // 32), dtype=torch.uint8, device=self.device)
        if stream is None:
            stream = torch.cuda.current_stream()
        # combine_final writes every row of [0, run_tokens); only the unsliced tail needs zeros
        with torch.cuda.stream(stream):
            out = torch.empty(
                (self.mtpr, self.H), dtype=torch.bfloat16, device=self.device
            )
            if not slice_output:
                out[run_tokens:].zero_()
        # recv_le init (SENTINEL = epr, masked by the planner) on the kernel stream, so it
        # is ordered before the launch; the kernel's barrier_init orders it cross-rank.
        # No host sync / dist.barrier -> forward() is a pure stream pipeline.
        with torch.cuda.stream(stream):
            # SBX: senders write whole routing blocks; sb_bucket re-zeroes its counters
            if not sbx_on(cfg.small):
                self._recv.local.fill_(self.epr)
                self._ecnt.zero_()  # per-expert position counters for the balanced-gather pre-pass
        _run_compiled(
            fn, *self._launch_args(run_tokens, x_q, xs, topk_ids, wts, out), stream
        )
        return out[:run_tokens] if slice_output else out

    __call__ = forward

    # --- watchdog / lifecycle (parity with MegaMoeTP; fully wired with the arena) ----
    def poll_errors(self):
        return int(self.ctrl[0].item())

    def error_flag(self):
        return self.ctrl[0:1]

    def clear_errors(self):
        self.ctrl[0] = 0

    def reset(self):
        ep = self.ctrl[EPOCH_SLOT].clone()  # the arena flags hold past epochs
        self.ctrl.zero_()
        self.ctrl[EPOCH_SLOT] = ep
        self._ecnt.zero_()
